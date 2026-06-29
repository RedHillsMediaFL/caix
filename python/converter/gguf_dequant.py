#!/usr/bin/env python3
"""Dequantize a GGUF model back to an HF safetensors directory (so caix can convert it).

Uses transformers' GGUF support (`from_pretrained(gguf_file=...)`) to dequantize, then
`save_pretrained` to a normal HF dir (config.json + safetensors + tokenizer). The caix converter
then runs on that dir. Run in the Apple env (has transformers).

  uv run --directory <coreai-models/python> python gguf_dequant.py --repo <hf-or-.gguf> --out <dir>

Notes:
- Quality: a GGUF is already quantized; dequant→requant to Core AI 4-bit stacks error. We pick the
  HIGHEST-quality quant available to minimize it. Best for Q4_K_M and up; IQ2/Q2 will be poor.
- Memory: the dequantized model is held at fp16 in RAM — fine for small/medium models, heavy for 30B+.
"""
from __future__ import annotations
import argparse, json, os, sys

# higher quality first — less error baked in before we re-quantize for Core AI
QUANT_PREF = ["F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S",
              "Q4_0", "IQ4_NL", "IQ4_XS", "Q3_K_L", "Q3_K_M", "IQ3_M", "Q3_K_S", "Q2_K", "IQ2_M"]


def pick_gguf(files: list[str]) -> str | None:
    ggufs = [f for f in files if f.lower().endswith(".gguf") and "of-0" not in f and "00002-of" not in f]
    if not ggufs:
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
    def rank(f: str) -> int:
        for i, q in enumerate(QUANT_PREF):
            if q.lower() in f.lower():
                return i
        return len(QUANT_PREF)
    ggufs.sort(key=rank)
    return ggufs[0] if ggufs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, or a local .gguf path")
    ap.add_argument("--gguf-file", help="specific .gguf filename (else best quant is chosen)")
    ap.add_argument("--out", required=True, help="output HF directory")
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print(json.dumps({"ok": False, "reason": f"transformers import failed: {e}"})); return 1

    if args.repo.lower().endswith(".gguf") and os.path.exists(args.repo):
        repo, gfile = os.path.dirname(args.repo) or ".", os.path.basename(args.repo)
    else:
        try:
            from huggingface_hub import HfApi
            files = list(HfApi().list_repo_files(args.repo))
        except Exception as e:
            print(json.dumps({"ok": False, "reason": f"could not list repo: {e}"})); return 1
        gfile = args.gguf_file or pick_gguf(files)
        if not gfile:
            print(json.dumps({"ok": False, "reason": "no .gguf file found in repo"})); return 1
        repo = args.repo

    try:
        print(f"[gguf] dequantizing {repo} :: {gfile} (this loads the full model into RAM)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(repo, gguf_file=gfile)
        tok = AutoTokenizer.from_pretrained(repo, gguf_file=gfile)
    except Exception as e:
        print(json.dumps({"ok": False, "reason": f"dequant failed ({type(e).__name__}: {e}). "
                          "The architecture may not be GGUF-loadable by transformers."})); return 1

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(json.dumps({"ok": True, "out": args.out, "gguf_file": gfile,
                      "model_type": getattr(model.config, "model_type", "")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
