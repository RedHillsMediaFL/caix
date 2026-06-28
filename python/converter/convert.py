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
import argparse, json, os, subprocess, sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PIPELINE_ROOT / "models" / "registry.json"
# Apple's coreai-models python checkout (provides coreai.llm.export). Required for CONVERSION only.
_apple_env_candidates = [
    Path(os.environ["CAIX_COREAI_MODELS"]).expanduser() if os.environ.get("CAIX_COREAI_MODELS") else None,
    PIPELINE_ROOT.parent / "coreai-models" / "python",
    Path("/Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python"),
]
_normalized_apple_env_candidates = [
    p / "python" if p and (p / "python" / "pyproject.toml").exists() else p
    for p in _apple_env_candidates
]
APPLE_ENV = next((p for p in _normalized_apple_env_candidates if p and p.exists()), _normalized_apple_env_candidates[1])
EXPORTS = Path(os.environ.get("CAIX_EXPORTS", str(PIPELINE_ROOT / "exports"))).expanduser()
HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
TMPDIR_EXPORT = os.environ.get("CAIX_TMPDIR", str(PIPELINE_ROOT.parent / "coreai-tmp"))
CHECK_SUPPORT = Path(__file__).resolve().parent / "check_support.py"
GGUF_DEQUANT = Path(__file__).resolve().parent / "gguf_dequant.py"


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


QWYTHOS_THINKING_BRANCH = """{%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}"""

QWYTHOS_NO_THINK_BRANCH = "{{- '<think>\\n\\n</think>\\n\\n' }}"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"models": {}}


def resolve(args, support: dict | None = None) -> dict:
    reg = load_registry()["models"]
    if args.model and args.model in reg:
        m = reg[args.model]
        hf_id = m["hf_repo"]
        model_type = m.get("model_type", "")
        compression = args.compression or m.get("compression", "4bit")
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
        context = args.context or args.max_context_cap
        name = args.name or hf_id.split("/")[-1].lower() + "-coreai"
        precision = args.compute_precision or ("bfloat16" if model_type in BF16_FAMILIES else "float16")
    return {"hf_id": hf_id, "compression": compression, "context": context,
            "name": name, "precision": precision}


def build_cmd(r: dict, args) -> list[str]:
    cmd = ["uv", "run", "--directory", str(APPLE_ENV), "coreai.llm.export", r["hf_id"],
           "--platform", "macOS", "--compression", r["compression"],
           "--compute-precision", r["precision"], "--max-context-length", str(r["context"]),
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
    """Apply CAIX runtime compatibility patches to exported tokenizer templates.

    Apple's CoreAILanguageModels tokenizer API currently lets us pass messages/tools, but not
    arbitrary chat-template kwargs. Qwythos exposes the right no-thinking path behind
    `enable_thinking=false`; without that kwarg its generation prompt dangles an open `<think>`,
    which makes OpenAI clients receive reasoning_content with empty visible content.
    """
    patched: list[str] = []
    if "Qwythos-9B-Claude-Mythos-5-1M" not in hf_id:
        return patched

    tok_dir = bundle / "tokenizer"
    candidates = [tok_dir / "chat_template.jinja"]
    cfg = tok_dir / "tokenizer_config.json"
    if cfg.exists():
        candidates.append(cfg)

    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text()
        if QWYTHOS_THINKING_BRANCH not in text:
            continue
        path.write_text(text.replace(QWYTHOS_THINKING_BRANCH, QWYTHOS_NO_THINK_BRANCH))
        patched.append(str(path.relative_to(bundle)))
    return patched


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a model to Core AI .aimodel")
    ap.add_argument("model", nargs="?", help="registry key (see models/registry.json)")
    ap.add_argument("--hf-id", help="raw HuggingFace id (if not a registry key)")
    ap.add_argument("--name", help="output bundle name")
    ap.add_argument("--compression", help="override (e.g. 4bit | none)")
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
    args = ap.parse_args()

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
    hf_for_check = args.hf_id or (args.model if args.model and args.model not in load_registry()["models"] else None)
    support_info = None
    if args.check or hf_for_check:
        target = args.hf_id or args.model
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
