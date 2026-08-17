#!/usr/bin/env python3
"""Check whether a HuggingFace repo has a Core AI-authored architecture.

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
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", "/Volumes/SSD/hf-cache")

BF16_TYPES = {"gemma4", "gemma4_assistant", "diffusion_gemma", "qwen3_5", "qwen3_5_moe", "glm4"}
STATIC_SUPPORTED_TYPES = [
    "gemma3_text", "gemma4", "gemma4_assistant", "glm4", "gpt_oss", "mistral",
    "mixtral", "qwen2", "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_moe",
]
STATIC_MODEL_TYPE_REMAPPING = {
    "gemma3": "gemma3_text",
    "qwen2_5": "qwen2",
}

AUTHORING_GAPS = {
    "qwen3_5_moe": {
        "status": "needs_coreai_authoring",
        "summary": (
            "Qwen3.5 MoE combines the qwen3_5 hybrid recurrent/full-attention decoder "
            "with qwen3_moe-style SwitchGLU experts."
        ),
        "requirements": [
            "register qwen3_5_moe and qwen3_5_moe_text AutoConfig shims before AutoConfig.from_pretrained",
            "unwrap top-level multimodal checkpoints through text_config and model.language_model weights",
            "reuse qwen3_5 recurrent-state packing for linear_attention layers",
            "replace dense qwen3_5 MLP blocks with router + shared expert + top-k SwitchGLU experts",
            "remap per-expert safetensors into SwitchGLU layout and preserve top-k/router normalization semantics",
            "verify parity on a tiny random-weight qwen3_5_moe config, then run a structural export before full conversion",
        ],
        "next_step": "author coreai_models.models.macos.qwen3_5_moe and register it in coreai_models.models.registry",
    },
}


def emit(d: dict) -> int:
    print(json.dumps(d))
    return 0


def _download_config_stdlib(hf_id: str) -> dict:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    encoded = quote(hf_id.strip("/"), safe="/")
    url = f"{endpoint}/{encoded}/resolve/main/config.json"
    headers = {"user-agent": "caix-support-check"}
    with urlopen(Request(url, headers=headers), timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _list_repo_files_stdlib(hf_id: str) -> list[str]:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    encoded = quote(hf_id.strip("/"), safe="/")
    url = f"{endpoint}/api/models/{encoded}/tree/main?recursive=1"
    headers = {"user-agent": "caix-support-check"}
    with urlopen(Request(url, headers=headers), timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("siblings"), list):
        return [item.get("rfilename", "") for item in payload["siblings"] if item.get("rfilename")]
    if isinstance(payload, list):
        return [item.get("path", "") for item in payload if isinstance(item, dict) and item.get("path")]
    raise ValueError("unexpected Hugging Face tree response")


def _load_config(hf_id: str) -> tuple[dict | None, str | None]:
    if os.path.isdir(hf_id) and os.path.exists(os.path.join(hf_id, "config.json")):
        with open(os.path.join(hf_id, "config.json")) as f:
            return json.load(f), None
    try:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(hf_id, "config.json")
        with open(cfg_path) as f:
            return json.load(f), None
    except Exception as hub_error:
        try:
            return _download_config_stdlib(hf_id), None
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as std_error:
            return None, (
                f"could not fetch config.json ({type(hub_error).__name__}: {hub_error}; "
                f"stdlib fallback {type(std_error).__name__}: {std_error}). "
                "Check the repo id and local Hugging Face auth."
            )


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def _rough_params_b(cfg: dict) -> float | None:
    """Rough sizing for UI planning; MoE estimate is total parameters, not active params."""
    tc = _text_config(cfg)
    try:
        h = tc.get("hidden_size")
        layers = tc.get("num_hidden_layers")
        vocab = tc.get("vocab_size")
        if not (h and layers and vocab):
            return None

        embed = 2 * vocab * h
        num_experts = tc.get("num_experts") or 0
        moe_hidden = tc.get("moe_intermediate_size") or 0
        shared_hidden = tc.get("shared_expert_intermediate_size") or 0
        if num_experts and moe_hidden:
            # gate/up/down per expert, plus a small router. Attention/SSM terms are intentionally
            # coarse; the goal is sizing guidance, not a parameter-count audit.
            expert_ffn = layers * (3 * h * moe_hidden * num_experts + h * num_experts)
            shared_ffn = layers * (3 * h * shared_hidden) if shared_hidden else 0
            attentionish = layers * 6 * h * h
            return round((embed + expert_ffn + shared_ffn + attentionish) / 1e9, 2)

        return round((12 * layers * h * h + embed) / 1e9, 2)
    except Exception:
        return None


def _authoring_gap(model_type: str, cfg: dict) -> dict:
    gap = AUTHORING_GAPS.get(model_type)
    if gap:
        return dict(gap)
    return {
        "status": "needs_coreai_authoring",
        "summary": f"Core AI does not yet have an authored macOS model for model_type '{model_type or '?'}'.",
        "requirements": [
            "identify the HF config and state-dict layout",
            "add or remap the model_type in coreai_models.models.registry",
            "author the macOS model class using existing Core AI primitives where possible",
            "add parity checks against the HF reference or a minimal reference implementation",
            "run structural export, full export, and load/generate coherence before publishing",
        ],
        "next_step": "author the Core AI macOS model path and register the model_type",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    args = ap.parse_args()
    hf_id = args.hf_id.strip()

    # 1) fetch config.json — local dir (e.g. a dequantized GGUF) or HF repo (public/gated).
    cfg, config_error = _load_config(hf_id)
    if config_error:
        # No config.json is most often a GGUF-only repo. The converter can try a
        # dequantized text path, but architecture support is only known after that
        # dequant step and quality is lower than converting original safetensors.
        try:
            try:
                from huggingface_hub import HfApi
                files = list(HfApi().list_repo_files(hf_id))
            except Exception:
                files = _list_repo_files_stdlib(hf_id)
            ggufs = [f for f in files if f.lower().endswith(".gguf")]
            mmprojs = [f for f in ggufs if Path(f).name.lower().startswith("mmproj")]
            model_ggufs = [f for f in ggufs if f not in mmprojs]
            has_st = any(f.lower().endswith(".safetensors") for f in files)
            if ggufs and not has_st:
                model_label = "file" if len(model_ggufs) == 1 else "files"
                counts = f"{len(model_ggufs)} model .gguf {model_label}"
                if mmprojs:
                    mmproj_label = "file" if len(mmprojs) == 1 else "files"
                    counts += f", {len(mmprojs)} mmproj sidecar {mmproj_label}"
                reason = (
                    f"GGUF-only repo ({counts}). caix can dequantize + convert the text model "
                    "(quality is reduced vs the original safetensors); the architecture is "
                    "verified after dequant."
                )
                next_step = "Click Convert to try a text-only GGUF dequant smoke."
                if mmprojs:
                    reason += (
                        " mmproj sidecar files are present, but the current GGUF path does not "
                        "consume them; use the matching safetensors/base checkpoint or add a "
                        "dedicated GGUF+mmproj importer for image-text conversion."
                    )
                    next_step = (
                        "Run text-only GGUF dequant first; image-text conversion needs a "
                        "safetensors source or verified GGUF+mmproj importer."
                    )
                return emit({
                    "ok": False,
                    "supported": False,
                    "hf_id": hf_id,
                    "gguf_only": True,
                    "gguf_model_file_count": len(model_ggufs),
                    "gguf_mmproj": bool(mmprojs),
                    "gguf_mmproj_file_count": len(mmprojs),
                    "reason": reason,
                    "next_step": next_step,
                })
        except Exception:
            pass
        return emit({"ok": False, "supported": False, "hf_id": hf_id, "reason": config_error})

    # 2) model_type (top-level, or nested text_config for multimodal)
    mt = cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type") or ""
    archs = cfg.get("architectures") or []

    # 3) authoritative supported set from the Core AI model registry
    try:
        from coreai_models.models.registry import list_models, MODEL_TYPE_REMAPPING
        supported_types = list_models()
        remapped = MODEL_TYPE_REMAPPING.get(mt, mt)
        registry_source = "coreai_models"
    except Exception as e:
        supported_types = STATIC_SUPPORTED_TYPES
        remapped = STATIC_MODEL_TYPE_REMAPPING.get(mt, mt)
        registry_source = f"static ({type(e).__name__}: {e})"

    supported = remapped in supported_types

    # 4) rough param count for sizing/UX
    params_b = _rough_params_b(cfg)
    gap = {} if supported else _authoring_gap(mt, cfg)

    return emit({
        "ok": True, "supported": supported, "hf_id": hf_id,
        "model_type": mt, "coreai_type": remapped if supported else None,
        "architectures": archs, "supported_types": supported_types,
        "registry_source": registry_source,
        "params_b": params_b,
        "suggested_compression": "4bit",
        "suggested_precision": "bfloat16" if remapped in BF16_TYPES else "float16",
        "support_status": "supported" if supported else gap.get("status", "needs_coreai_authoring"),
        "authoring_required": not supported,
        "requirements": gap.get("requirements", []),
        "next_step": gap.get("next_step", ""),
        "reason": "" if supported else gap.get("summary", ""),
    })


if __name__ == "__main__":
    raise SystemExit(main())
