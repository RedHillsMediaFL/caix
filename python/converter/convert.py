#!/usr/bin/env python3
"""Registry-driven Core AI converter — wraps Apple's `coreai.llm.export`.

Resolves per-model compression / compute-precision / context / output-name from
models/registry.json (or accepts a raw --hf-id), then runs the real exporter in the
vendored Apple env. Architecture support for new model_types (gemma4, qwen3_5,
diffusion_gemma) is provided by authored modeling under the vendored repo; this wrapper
only orchestrates the export call.

Usage:
  convert.py gemma-4-31B-it-assistant            # registry key
  convert.py --hf-id Qwen/Qwen3-0.6B --name qwen3-0.6b-coreai
  convert.py gemma-4-31B-it --num-layers 2 --dry-run
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PIPELINE_ROOT / "models" / "registry.json"


def caix_env(name: str, legacy_suffix: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or os.environ.get("C" + "AIX_" + legacy_suffix, default)


# Apple's coreai-models python checkout (provides coreai.llm.export). Required for CONVERSION only.
_coreai_models = caix_env("caix_coreai_models", "COREAI_MODELS")
_apple_env_candidates = [
    Path(_coreai_models).expanduser() if _coreai_models else None,
    PIPELINE_ROOT.parent / "coreai-models" / "python",
    Path("/Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python"),
]
_normalized_apple_env_candidates = [
    p / "python" if p and (p / "python" / "pyproject.toml").exists() else p
    for p in _apple_env_candidates
]
APPLE_ENV = next((p for p in _normalized_apple_env_candidates if p and p.exists()), _normalized_apple_env_candidates[1])
EXPORTS = Path(caix_env("caix_exports", "EXPORTS", str(PIPELINE_ROOT / "exports"))).expanduser()
HF_HOME = os.environ.get("HF_HOME", "/Volumes/SSD/hf-cache")
TMPDIR_EXPORT = caix_env("caix_tmpdir", "TMPDIR", "/Volumes/SSD/coreai-tmp")
CHECK_SUPPORT = Path(__file__).resolve().parent / "check_support.py"
GGUF_DEQUANT = Path(__file__).resolve().parent / "gguf_dequant.py"
CLUSTER_SCHEMA = "caix.cluster.stage_manifest.v0"


def dequant_gguf(repo: str, gguf_file: str | None, out_dir: str) -> dict:
    """Dequantize a GGUF repo/file to an HF dir (in the Apple env). Returns the parsed JSON dict."""
    env = {**os.environ, "HF_HOME": HF_HOME}
    # transformers needs `gguf>=0.10.0` to dequantize a GGUF checkpoint; add it for this run.
    cmd = ["uv", "run", "--directory", str(APPLE_ENV), "--with", "gguf>=0.10.0",
           "python", str(GGUF_DEQUANT), "--repo", repo, "--out", out_dir]
    if gguf_file:
        cmd += ["--gguf-file", gguf_file]
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        line = [l for l in out.stdout.splitlines() if l.strip().startswith("{")]
        return json.loads(line[-1]) if line else {"ok": False, "reason": f"no JSON (stderr: {out.stderr[-300:]})"}
    except Exception as e:
        return {"ok": False, "reason": f"dequant failed: {type(e).__name__}: {e}"}


def check_support(hf_id: str) -> dict:
    """Run the architecture support check in the Apple env; returns the parsed JSON dict."""
    env = {**os.environ, "HF_HOME": HF_HOME}
    try:
        direct = subprocess.run(
            [sys.executable, str(CHECK_SUPPORT), "--hf-id", hf_id],
            env=env, capture_output=True, text=True, timeout=45)
        direct_line = [l for l in direct.stdout.splitlines() if l.strip().startswith("{")]
        if direct_line:
            return json.loads(direct_line[-1])
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["uv", "run", "--directory", str(APPLE_ENV), "python", str(CHECK_SUPPORT),
             "--hf-id", hf_id],
            env=env, capture_output=True, text=True, timeout=180)
        line = [l for l in out.stdout.splitlines() if l.strip().startswith("{")]
        return json.loads(line[-1]) if line else {
            "ok": False, "supported": False, "hf_id": hf_id,
            "reason": f"support check produced no JSON (stderr: {out.stderr[-300:]})"}
    except Exception as e:
        return {"ok": False, "supported": False, "hf_id": hf_id,
                "reason": f"support check failed: {type(e).__name__}: {e}"}

# Gemma-family decoders require bfloat16 for parity; qwen3_5 (hybrid SSM) also validated in
# bfloat16 (the proven Qwythos path) — fp16's narrow range risks overflow in its activations.
BF16_FAMILIES = ("gemma4", "gemma4_assistant", "diffusion_gemma", "qwen3_5", "qwen3_5_moe", "glm4")


NO_THINK_KWARG_BRANCH = """{%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}"""

NO_THINK_GENERATION_PROMPT = "{{- '<think>\\n\\n</think>\\n\\n' }}"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"models": {}}


def resolve(args, support: dict | None = None) -> dict:
    reg = load_registry()["models"]
    if args.model and args.model in reg:
        m = reg[args.model]
        hf_id = m["hf_repo"]
        model_type = m.get("model_type", "")
        compression = args.compression or m.get("compression", "4bit")
        compression_config = args.compression_config or m.get("compression_config")
        if compression == "optional-4bit":
            compression = args.compression or "4bit"
        context = args.context or min(int(m.get("context", 4096)), args.max_context_cap)
        name = args.name or f"{args.model}-coreai"
        precision = args.compute_precision or ("bfloat16" if model_type in BF16_FAMILIES else "float16")
    else:
        hf_id = args.hf_id or args.model
        if not hf_id:
            sys.exit("error: provide a registry key, or --hf-id")
        model_type = (support or {}).get("model_type", "")
        compression = args.compression or "4bit"
        compression_config = args.compression_config
        context = args.context or args.max_context_cap
        name = args.name or hf_id.split("/")[-1].lower() + "-coreai"
        precision = args.compute_precision or ("bfloat16" if model_type in BF16_FAMILIES else "float16")
    if compression_config and not args.compression:
        config_path = Path(compression_config).expanduser()
        if not config_path.is_absolute():
            config_path = PIPELINE_ROOT / config_path
        compression = config_path.stem
        compression_config = str(config_path)
    return {"hf_id": hf_id, "compression": compression, "compression_config": compression_config,
            "context": context, "name": name, "precision": precision}


def build_cmd(r: dict, args) -> list[str]:
    cmd = ["uv", "run", "--directory", str(APPLE_ENV), "coreai.llm.export", r["hf_id"],
           "--platform", "macOS"]
    if r.get("compression_config"):
        cmd += ["--compression-config", r["compression_config"]]
    else:
        cmd += ["--compression", r["compression"]]
    cmd += ["--compute-precision", r["precision"], "--max-context-length", str(r["context"]),
           "--output-dir", str(EXPORTS), "--output-name", r["name"], "--experimental", "--overwrite"]
    if args.num_layers:
        cmd += ["--num-layers", str(args.num_layers)]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def _log_convert_event(target: str, kind: str, reason: str, model_type) -> None:
    """Persist gated/failed conversion diagnostics without affecting conversion flow."""
    try:
        import time as _time
        log_dir = os.path.join(os.path.expanduser("~"), ".caix")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "convert-failures.log"), "a") as f:
            f.write(f"{_time.strftime('%Y-%m-%dT%H:%M:%S')}\t{kind}\t{target}\t"
                    f"model_type={model_type}\t{reason}\n")
    except Exception:
        pass


def patch_chat_templates(bundle: Path, hf_id: str) -> list[str]:
    """Apply caix runtime compatibility patches to exported tokenizer templates.

    Apple's CoreAILanguageModels tokenizer API currently lets us pass messages/tools, but not
    arbitrary chat-template kwargs. Several Qwen3.5 hybrid/MoE templates expose the right
    no-thinking path behind `enable_thinking=false`; without that kwarg their generation prompt
    dangles an open `<think>`, which makes OpenAI clients receive reasoning_content with empty
    visible content.
    """
    patched: list[str] = []
    tok_dir = bundle / "tokenizer"
    candidates = [tok_dir / "chat_template.jinja"]
    cfg = tok_dir / "tokenizer_config.json"
    if cfg.exists():
        candidates.append(cfg)

    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text()
        if NO_THINK_KWARG_BRANCH not in text:
            continue
        path.write_text(text.replace(NO_THINK_KWARG_BRANCH, NO_THINK_GENERATION_PROMPT))
        patched.append(str(path.relative_to(bundle)))
    return patched


def infer_min_kv_capacity(hf_id: str, registry_entry: dict | None, support: dict | None) -> int | None:
    """Return the fixed KV-cache floor required by hybrid qwen3_5 recurrent-state packing."""
    if registry_entry:
        value = registry_entry.get("min_kv_capacity")
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    model_type = (support or {}).get("model_type") or (registry_entry or {}).get("model_type") or ""
    name = hf_id.lower()
    if "ornith" in name or model_type == "qwen3_5_moe":
        return 1024
    if "qwen3.6-27b" in name:
        return 768
    if model_type == "qwen3_5" or "qwythos" in name or "qwen3_5" in name or "qwen3.5" in name:
        return 512
    return None


def patch_min_kv_capacity(
    bundle: Path,
    hf_id: str,
    registry_entry: dict | None,
    support: dict | None,
) -> int | None:
    """Bake a hybrid model's required KV-cache floor into root metadata.json."""
    min_kv = infer_min_kv_capacity(hf_id, registry_entry, support)
    if not min_kv:
        return None
    meta_path = bundle / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    language = meta.setdefault("language", {})
    if not isinstance(language, dict):
        return None
    if language.get("min_kv_capacity") == min_kv:
        return min_kv
    language["min_kv_capacity"] = min_kv
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return min_kv


def _cluster_error(message: str) -> SystemExit:
    return SystemExit(f"error: staged cluster manifest: {message}")


def _json_object(path: Path, label: str) -> dict:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise _cluster_error(f"{label} not found: {path}") from None
    except json.JSONDecodeError as e:
        raise _cluster_error(f"{label} is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise _cluster_error(f"{label} must be a JSON object")
    return data


def _first_non_empty(*values) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool):
        raise _cluster_error(f"{field} must be a positive integer")
    if isinstance(value, int) and value > 0:
        return value
    raise _cluster_error(f"{field} must be a positive integer")


def _positive_number(value, field: str) -> float:
    if isinstance(value, bool):
        raise _cluster_error(f"{field} must be positive")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    raise _cluster_error(f"{field} must be positive")


def _stage_asset_name(stage: dict, stage_id: str) -> str:
    name = _first_non_empty(
        stage.get("bundle"),
        stage.get("path"),
        stage.get("bundle_path"),
        stage.get("aimodel"),
    )
    if not name:
        raise _cluster_error(f"stage {stage_id} is missing bundle")
    return name


def _bundle_asset_path(bundle: Path, asset_name: str, field: str) -> Path:
    if "://" in asset_name:
        raise _cluster_error(f"{field} must be a local bundle-relative .aimodel path")
    path = Path(asset_name).expanduser()
    if path.is_absolute():
        raise _cluster_error(f"{field} must be relative to the bundle")
    if any(part in ("..", "") for part in path.parts):
        raise _cluster_error(f"{field} must stay inside the bundle")
    if path.suffix != ".aimodel":
        raise _cluster_error(f"{field} must point to a .aimodel directory")
    resolved = bundle / path
    if not resolved.is_dir():
        raise _cluster_error(f"{field} path is missing: {asset_name}")
    if not (resolved / "main.mlirb").is_file():
        raise _cluster_error(f"{field} is missing main.mlirb: {asset_name}")
    return resolved


def _validate_function_map(stage: dict, stage_id: str) -> None:
    function_map = stage.get("function_map")
    if not isinstance(function_map, dict):
        raise _cluster_error(f"stage {stage_id} is missing function_map")
    main = function_map.get("main")
    if not isinstance(main, list) or not any(isinstance(name, str) and name.strip() for name in main):
        raise _cluster_error(f"stage {stage_id} function_map.main must be non-empty")
    if "decode" in function_map:
        decode = function_map["decode"]
        if not isinstance(decode, list) or not any(
            isinstance(name, str) and name.strip() for name in decode
        ):
            raise _cluster_error(f"stage {stage_id} function_map.decode must be non-empty")


def _validate_rope(stage: dict, stage_id: str, role: str | None) -> None:
    rope = stage.get("rope")
    if rope is None:
        return
    if role != "transformer_layers":
        raise _cluster_error(f"stage {stage_id} rope inputs are only valid for transformer_layers")
    if not isinstance(rope, dict):
        raise _cluster_error(f"stage {stage_id} rope must be an object")
    cos_input = _first_non_empty(rope.get("cos_input"))
    sin_input = _first_non_empty(rope.get("sin_input"))
    if not cos_input:
        raise _cluster_error(f"stage {stage_id} rope cos_input must be non-empty")
    if not sin_input:
        raise _cluster_error(f"stage {stage_id} rope sin_input must be non-empty")
    if cos_input == sin_input:
        raise _cluster_error(f"stage {stage_id} rope cos_input and sin_input must differ")
    head_dim = rope.get("head_dim")
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0 or head_dim % 2:
        raise _cluster_error(f"stage {stage_id} rope head_dim must be a positive even integer")
    theta = rope.get("theta")
    if (
        isinstance(theta, bool)
        or not isinstance(theta, (int, float))
        or not math.isfinite(float(theta))
        or theta <= 0
    ):
        raise _cluster_error(f"stage {stage_id} rope theta must be positive")


def _normalize_boundary(cluster: dict) -> None:
    boundary = cluster.get("boundary")
    if boundary is None and isinstance(cluster.get("boundary_tensor"), dict):
        boundary = {"hidden_state": cluster["boundary_tensor"]}
        cluster["boundary"] = boundary
        cluster.pop("boundary_tensor", None)
    if not isinstance(boundary, dict) or not isinstance(boundary.get("hidden_state"), dict):
        raise _cluster_error("boundary.hidden_state is required")
    hidden = boundary["hidden_state"]
    if hidden.get("name") != "hidden_states":
        raise _cluster_error("boundary.hidden_state.name must be hidden_states")
    shape = hidden.get("shape")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and shape[0] == 1
        and (shape[1] == -1 or (isinstance(shape[1], int) and shape[1] > 0))
        and isinstance(shape[2], int)
        and shape[2] > 0
    ):
        raise _cluster_error("boundary.hidden_state.shape must be [1, -1, H]")
    if hidden.get("scalar_type") not in ("float16", "float32"):
        raise _cluster_error("boundary.hidden_state.scalar_type must be float16 or float32")


def _cluster_block(raw_manifest: dict, bundle: Path, model_name: str) -> dict:
    if isinstance(raw_manifest.get("cluster"), dict):
        cluster = dict(raw_manifest["cluster"])
        cluster.setdefault("schema", raw_manifest.get("schema") or CLUSTER_SCHEMA)
        cluster.setdefault(
            "model",
            _first_non_empty(
                raw_manifest.get("model"),
                raw_manifest.get("model_name"),
                raw_manifest.get("name"),
                model_name,
            ),
        )
    else:
        cluster = {
            "schema": raw_manifest.get("schema") or CLUSTER_SCHEMA,
            "model": _first_non_empty(
                raw_manifest.get("model"),
                raw_manifest.get("model_name"),
                raw_manifest.get("name"),
                model_name,
            ),
            "total_layer_count": raw_manifest.get("total_layer_count")
            or raw_manifest.get("total_layers"),
            "position_mode": raw_manifest.get("position_mode"),
            "boundary": raw_manifest.get("boundary"),
            "stages": raw_manifest.get("stages"),
        }
        if cluster["boundary"] is None and isinstance(raw_manifest.get("boundary_tensor"), dict):
            cluster["boundary"] = {"hidden_state": raw_manifest["boundary_tensor"]}

    if cluster.get("schema") != CLUSTER_SCHEMA:
        raise _cluster_error(f"schema must be {CLUSTER_SCHEMA}")
    if not _first_non_empty(cluster.get("model")):
        raise _cluster_error("model is required")
    total_layers = _positive_int(
        cluster.get("total_layer_count") or cluster.get("total_layers"),
        "total_layer_count",
    )
    cluster["total_layer_count"] = total_layers
    cluster.pop("total_layers", None)
    if cluster.get("position_mode") not in ("full_prefix", "current"):
        raise _cluster_error("position_mode must be full_prefix or current")
    _normalize_boundary(cluster)

    stages = cluster.get("stages")
    if not isinstance(stages, list) or len(stages) < 3:
        raise _cluster_error("stages must include embeddings, transformer_layers, and final_norm_head")
    roles = [stage.get("role") if isinstance(stage, dict) else None for stage in stages]
    if roles[0] != "embeddings" or roles[-1] != "final_norm_head":
        raise _cluster_error("stages must start with embeddings and end with final_norm_head")
    if any(role != "transformer_layers" for role in roles[1:-1]):
        raise _cluster_error("only transformer_layers stages may sit between embeddings and final_norm_head")

    expected_layer = 0
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise _cluster_error(f"stage {index} must be an object")
        stage_id = _first_non_empty(stage.get("id"), stage.get("name")) or f"stage-{index + 1}"
        stage.setdefault("id", stage_id)
        role = stage.get("role")
        _positive_number(stage.get("memory_gb"), f"stage {stage_id} memory_gb")
        _validate_function_map(stage, stage_id)
        _validate_rope(stage, stage_id, role)
        _bundle_asset_path(bundle, _stage_asset_name(stage, stage_id), f"stage {stage_id} bundle")
        decode_asset = _first_non_empty(
            stage.get("decode_asset"),
            stage.get("decode_asset_name"),
            stage.get("decode_bundle"),
        )
        if decode_asset:
            _bundle_asset_path(bundle, decode_asset, f"stage {stage_id} decode_asset")

        layers = stage.get("layers", stage.get("layer_range"))
        if role == "transformer_layers":
            if not (
                isinstance(layers, list)
                and len(layers) == 2
                and all(isinstance(value, int) for value in layers)
                and layers[0] == expected_layer
                and layers[1] > layers[0]
            ):
                raise _cluster_error(
                    f"stage {stage_id} layers must be contiguous [lower, upper]"
                )
            expected_layer = layers[1]
        else:
            if not isinstance(layers, str) or not layers.strip():
                raise _cluster_error(f"stage {stage_id} layers must be a label")
            if role == "final_norm_head":
                _positive_int(stage.get("vocab_size"), f"stage {stage_id} vocab_size")

    if expected_layer != total_layers:
        raise _cluster_error(
            f"transformer layer ranges end at {expected_layer}; expected {total_layers}"
        )
    return cluster


def attach_cluster_manifest(bundle: Path, manifest_path: Path) -> int:
    bundle = bundle.expanduser()
    if not bundle.is_dir():
        raise _cluster_error(f"bundle not found: {bundle}")
    meta_path = bundle / "metadata.json"
    metadata = _json_object(meta_path, "metadata.json")
    if metadata.get("kind") != "llm":
        raise _cluster_error("metadata.json kind must be llm")
    manifest = _json_object(manifest_path.expanduser(), "stage manifest")
    cluster = _cluster_block(manifest, bundle, str(metadata.get("name") or bundle.name))
    metadata["cluster"] = cluster
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return len(cluster["stages"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a model to Core AI .aimodel")
    ap.add_argument("model", nargs="?", help="registry key (see models/registry.json)")
    ap.add_argument("--hf-id", help="raw HuggingFace id (if not a registry key)")
    ap.add_argument("--name", help="output bundle name")
    ap.add_argument("--compression", help="override (e.g. 4bit | none)")
    ap.add_argument("--compression-config", help="override with a coreai-opt YAML compression config")
    ap.add_argument("--compute-precision", help="override (float16 | bfloat16 | float32)")
    ap.add_argument("--context", type=int, help="override max context length")
    ap.add_argument("--max-context-cap", type=int, default=8192,
                    help="cap registry context for first exports (default 8192)")
    ap.add_argument("--num-layers", type=int, help="export only N layers (fast structural test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="only check Core AI support for the HF repo; print JSON and exit")
    ap.add_argument("--force", action="store_true",
                    help="convert even if the support check has not passed")
    ap.add_argument("--gguf", help="GGUF repo id or local .gguf path — dequantize to HF, then convert")
    ap.add_argument("--gguf-file", help="specific .gguf filename in the repo (else best quant chosen)")
    ap.add_argument("--bundle", help="existing exported bundle for --attach-cluster-manifest")
    ap.add_argument("--attach-cluster-manifest",
                    help="validate existing staged .aimodel assets and write metadata.json cluster block; does not create stages")
    args = ap.parse_args()

    if args.bundle:
        if not args.attach_cluster_manifest:
            sys.exit("error: --bundle requires --attach-cluster-manifest")
        count = attach_cluster_manifest(Path(args.bundle), Path(args.attach_cluster_manifest))
        print(f"  attached cluster metadata for {count} stages")
        return 0
    if args.attach_cluster_manifest and args.dry_run:
        sys.exit("error: --attach-cluster-manifest cannot be combined with --dry-run")

    # GGUF path: dequantize to a temp HF dir, then convert that as a local model.
    if args.gguf:
        import tempfile
        os.makedirs(TMPDIR_EXPORT, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="caix-gguf-hf-",
                               dir=(TMPDIR_EXPORT))
        print(f"[gguf] dequantizing {args.gguf} -> {tmp}")
        deq = dequant_gguf(args.gguf, args.gguf_file, tmp)
        if not deq.get("ok"):
            print(json.dumps(deq)); return 3
        print(f"[gguf] dequantized model_type={deq.get('model_type')} from {deq.get('gguf_file')}")
        # convert the local dequantized dir from here on
        args.hf_id = tmp
        if not args.name:
            base = args.gguf.rstrip("/").split("/")[-1].replace(".gguf", "")
            args.name = base.lower() + "-coreai"
        mt = deq.get("model_type", "")
        if not args.compute_precision:
            args.compute_precision = "bfloat16" if mt in ("gemma3", "gemma2", "gemma4") else "float16"

    # Architecture support gate: for raw HF ids (not registry keys), verify the model_type is
    # authored for Core AI before downloading/converting. Not-yet-authored => flag, don't convert.
    registry_models = load_registry()["models"]
    registry_entry = registry_models.get(args.model) if args.model else None
    hf_for_check = args.hf_id or (
        registry_entry["hf_repo"] if args.check and registry_entry
        else args.model if args.model and args.model not in registry_models
        else None
    )
    support_info = None
    if args.check or hf_for_check:
        target = hf_for_check or args.hf_id or args.model
        sup = check_support(target)
        support_info = sup
        if args.check:
            print(json.dumps(sup))
            return 0 if sup.get("supported") else 3
        if not sup.get("supported") and not args.force:
            print(json.dumps(sup))
            print(f"  FLAGGED: {sup.get('next_step') or sup.get('reason', 'Core AI authoring required')}")
            _log_convert_event(target, "authoring_required", sup.get("reason", ""), sup.get("model_type"))
            return 3

    r = resolve(args, support=support_info)
    cmd = build_cmd(r, args)
    # Export quantization checkpoints (mmap_dir) default to the system TMPDIR on the small
    # boot disk; 31B+ models overflow it. Redirect temp to the 2 TB SSD.
    os.makedirs(TMPDIR_EXPORT, exist_ok=True)
    env = {**os.environ, "HF_HOME": HF_HOME, "TMPDIR": TMPDIR_EXPORT}
    print(f"convert: {r['hf_id']}  ->  {EXPORTS / r['name']}")
    print(f"  compression={r['compression']} precision={r['precision']} context={r['context']}")
    print("  $", " ".join(cmd))
    rc = subprocess.run(cmd, env=env).returncode
    bundle = EXPORTS / r["name"]
    aimodel = next(iter(bundle.glob("*.aimodel")), None) if bundle.exists() else None
    ok = bool(aimodel and (aimodel / "main.mlirb").exists())
    # Ship the model's OWN published output dictionary (generation_config.json) inside the bundle's
    # tokenizer dir, so the runtime honors the model's real stop tokens (e.g. gemma-4 eos_token_id
    # [1,106,50]) instead of guessing. Sourced from the HF cache populated during export.
    if ok and not args.dry_run:
        try:
            import glob as _glob, shutil as _shutil
            org_name = r["hf_id"].replace("/", "--")
            hits = _glob.glob(f"{HF_HOME}/hub/models--{org_name}/snapshots/*/generation_config.json")
            tok_dir = bundle / "tokenizer"
            if hits and tok_dir.exists():
                _shutil.copy(hits[0], tok_dir / "generation_config.json")
                print(f"  bundled generation_config.json (published stop tokens)")
        except Exception as e:
            print(f"  note: could not bundle generation_config.json: {e}")
        try:
            patched = patch_chat_templates(bundle, r["hf_id"])
            if patched:
                print(f"  patched chat template(s) for OpenAI-visible output: {', '.join(patched)}")
        except Exception as e:
            print(f"  note: could not patch chat template: {e}")
        try:
            min_kv = patch_min_kv_capacity(bundle, r["hf_id"], registry_entry, support_info)
            if min_kv:
                print(f"  baked language.min_kv_capacity={min_kv}")
        except Exception as e:
            print(f"  note: could not patch min_kv_capacity metadata: {e}")
        if args.attach_cluster_manifest:
            count = attach_cluster_manifest(bundle, Path(args.attach_cluster_manifest))
            print(f"  attached cluster metadata for {count} stages")
    if not args.dry_run:
        print(f"  result: {'OK ' + str(bundle) if ok else 'FAILED (no .aimodel/main.mlirb)'}  exit={rc}")
        if not ok:
            _log_convert_event(r["hf_id"], "failed",
                               f"export exit={rc}; no .aimodel/main.mlirb "
                               f"(compression={r['compression']} precision={r['precision']} context={r['context']})",
                               support_info.get("model_type") if support_info else None)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
