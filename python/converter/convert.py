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
# Path to Apple's coreai-models python checkout (provides `coreai.llm.export`). Needed only for
# CONVERSION (advanced) — not for serving a pre-converted model. Override with CAIX_COREAI_MODELS.
APPLE_ENV = Path(os.environ.get("CAIX_COREAI_MODELS",
                                str(PIPELINE_ROOT.parent / "coreai-models" / "python"))).expanduser()
# Where exported .aimodel bundles are written (point your server's --exports here).
EXPORTS = Path(os.environ.get("CAIX_EXPORTS", str(PIPELINE_ROOT / "exports"))).expanduser()
# HuggingFace cache (downloads land here).
HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
TMPDIR_EXPORT = os.environ.get("CAIX_TMPDIR", str(Path.home() / ".cache" / "caix-export"))
CHECK_SUPPORT = Path(__file__).resolve().parent / "check_support.py"


def check_support(hf_id: str) -> dict:
    """Run the architecture support check in the Apple env; returns the parsed JSON dict."""
    env = {**os.environ, "HF_HOME": HF_HOME}
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
BF16_FAMILIES = ("gemma4", "gemma4_assistant", "diffusion_gemma", "qwen3_5")


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"models": {}}


def resolve(args) -> dict:
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
        compression = args.compression or "4bit"
        context = args.context or args.max_context_cap
        name = args.name or hf_id.split("/")[-1].lower() + "-coreai"
        precision = args.compute_precision or "float16"
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
                    help="convert even if the support check says the type is unsupported")
    args = ap.parse_args()

    # Architecture support gate: for raw HF ids (not registry keys), verify the model_type is
    # authored for Core AI before downloading/converting. Unsupported => flag, don't convert.
    hf_for_check = args.hf_id or (args.model if args.model and args.model not in load_registry()["models"] else None)
    if args.check or hf_for_check:
        target = args.hf_id or args.model
        sup = check_support(target)
        if args.check:
            print(json.dumps(sup))
            return 0 if sup.get("supported") else 3
        if not sup.get("supported") and not args.force:
            print(json.dumps(sup))
            print(f"  FLAGGED: {sup.get('reason', 'unsupported architecture')}")
            return 3

    r = resolve(args)
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
    if not args.dry_run:
        print(f"  result: {'OK ' + str(bundle) if ok else 'FAILED (no .aimodel/main.mlirb)'}  exit={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
