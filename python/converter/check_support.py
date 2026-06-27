#!/usr/bin/env python3
"""Check whether a HuggingFace repo is a Core AI-supported architecture.

Fetches the repo's config.json, reads its model_type, and checks it against the authored
Core AI modeling registry (coreai_models.models.registry). Emits a single JSON line:

  {"ok":true,"supported":true,"hf_id":"...","model_type":"qwen2","coreai_type":"qwen2",
   "params_b":0.5,"suggested_compression":"4bit","suggested_precision":"float16",
   "supported_types":[...],"reason":""}

Run inside the vendored Apple env so the registry import is authoritative:
  uv run --directory /Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python \
      python /Volumes/SSD/ai-dev/coreai-pipeline/python/converter/check_support.py --hf-id Qwen/Qwen2-0.5B
"""
from __future__ import annotations
import argparse, json, os, sys

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

BF16_TYPES = {"gemma4", "gemma4_assistant", "diffusion_gemma", "qwen3_5"}


def emit(d: dict) -> int:
    print(json.dumps(d))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    args = ap.parse_args()
    hf_id = args.hf_id.strip()

    # 1) fetch config.json (public or token-gated)
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        cfg_path = hf_hub_download(hf_id, "config.json", token=token)
        cfg = json.load(open(cfg_path))
    except Exception as e:
        return emit({"ok": False, "supported": False, "hf_id": hf_id,
                     "reason": f"could not fetch config.json ({type(e).__name__}: {e}). "
                               "Check the repo id and access (gated repos need HF_TOKEN)."})

    # 2) model_type (top-level, or nested text_config for multimodal)
    mt = cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type") or ""
    archs = cfg.get("architectures") or []

    # 3) authoritative supported set from the Core AI model registry
    try:
        from coreai_models.models.registry import list_models, MODEL_TYPE_REMAPPING
        supported_types = list_models()
        remapped = MODEL_TYPE_REMAPPING.get(mt, mt)
    except Exception as e:
        return emit({"ok": False, "supported": False, "hf_id": hf_id, "model_type": mt,
                     "reason": f"registry import failed: {e}"})

    supported = remapped in supported_types

    # 4) rough param count for sizing/UX
    tc = cfg.get("text_config") or cfg
    params_b = None
    try:
        h = tc.get("hidden_size"); L = tc.get("num_hidden_layers"); V = tc.get("vocab_size")
        if h and L and V:
            params_b = round((12 * L * h * h + 2 * V * h) / 1e9, 2)
    except Exception:
        pass

    return emit({
        "ok": True, "supported": supported, "hf_id": hf_id,
        "model_type": mt, "coreai_type": remapped if supported else None,
        "architectures": archs, "supported_types": supported_types,
        "params_b": params_b,
        "suggested_compression": "4bit",
        "suggested_precision": "bfloat16" if remapped in BF16_TYPES else "float16",
        "reason": "" if supported else (
            f"model_type '{mt or '?'}' is not authored for Core AI. "
            f"Supported types: {', '.join(supported_types)}. Flagged for review."),
    })


if __name__ == "__main__":
    raise SystemExit(main())
