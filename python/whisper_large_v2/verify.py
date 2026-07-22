"""Execute the complete saved Whisper large-v2 CoreAI state machine."""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from whisper_large_v2.abi import NativeWhisperABI
from whisper_large_v2.export import WhisperExportError
from whisper_large_v2.manifest import validate_caix_asset


def _peak_resident_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _check_cap(max_resident_bytes: int, *, phase: str) -> None:
    peak = _peak_resident_bytes()
    if peak > max_resident_bytes:
        raise WhisperExportError(
            f"full Whisper verification exceeded resident cap during {phase}: "
            f"{peak} > {max_resident_bytes} bytes"
        )


async def _verify(asset: Path, max_resident_bytes: int) -> dict[str, object]:
    asset_validation = validate_caix_asset(asset)
    from coreai.runtime import AIModel, NDArray

    if not asset.is_dir() or asset.suffix != ".aimodel":
        raise WhisperExportError(f"Whisper asset is missing or not .aimodel: {asset}")
    abi = NativeWhisperABI.large_v2()
    float_dtype = torch.float16
    state = {
        "cross_key_cache": NDArray(
            data=torch.zeros(abi.tensors["cross_key_cache"].shape, dtype=float_dtype)
        ),
        "cross_value_cache": NDArray(
            data=torch.zeros(abi.tensors["cross_value_cache"].shape, dtype=float_dtype)
        ),
        "self_key_cache": NDArray(
            data=torch.zeros(abi.tensors["self_key_cache"].shape, dtype=float_dtype)
        ),
        "self_value_cache": NDArray(
            data=torch.zeros(abi.tensors["self_value_cache"].shape, dtype=float_dtype)
        ),
        "position": NDArray(data=torch.zeros((1,), dtype=torch.int32)),
        "cross_ready": NDArray(data=torch.zeros((1,), dtype=torch.int32)),
    }
    features = NDArray(data=torch.zeros(abi.tensors["input_features"].shape, dtype=float_dtype))
    token_id = NDArray(data=torch.tensor([[50_258]], dtype=torch.int32))

    started = time.perf_counter()
    model = await AIModel.load(asset)
    load_seconds = time.perf_counter() - started
    _check_cap(max_resident_bytes, phase="asset load")
    entrypoints = sorted(model.function_names)
    if entrypoints != ["decode_step", "encode", "load_cross_kv"]:
        raise WhisperExportError(f"Whisper asset entrypoints differ: {entrypoints!r}")

    encode = model.load_function("encode")
    load_cross_kv = model.load_function("load_cross_kv")
    decode_step = model.load_function("decode_step")

    started = time.perf_counter()
    encoded = await encode({"input_features": features})
    encode_seconds = time.perf_counter() - started
    _check_cap(max_resident_bytes, phase="encode")
    await load_cross_kv(
        {
            "cross_key_payload": encoded["cross_key_payload"],
            "cross_value_payload": encoded["cross_value_payload"],
        },
        state={
            "cross_key_cache": state["cross_key_cache"],
            "cross_value_cache": state["cross_value_cache"],
            "cross_ready": state["cross_ready"],
        },
    )
    _check_cap(max_resident_bytes, phase="cross cache load")

    started = time.perf_counter()
    decoded = await decode_step({"token_id": token_id}, state=state)
    decode_seconds = time.perf_counter() - started
    _check_cap(max_resident_bytes, phase="decode step")

    cross_keys = encoded["cross_key_payload"].numpy()
    cross_values = encoded["cross_value_payload"].numpy()
    logits = decoded["logits"].numpy()
    self_keys = state["self_key_cache"].numpy()
    self_values = state["self_value_cache"].numpy()
    return {
        "cross_key_shape": list(cross_keys.shape),
        "cross_ready": state["cross_ready"].numpy().reshape(-1).tolist(),
        "cross_value_shape": list(cross_values.shape),
        "decode_seconds": decode_seconds,
        "encode_seconds": encode_seconds,
        "entrypoints": entrypoints,
        "load_seconds": load_seconds,
        "logits_all_finite": bool(np.isfinite(logits).all()),
        "logits_shape": list(logits.shape),
        "main_sha256": asset_validation.main_sha256,
        "peak_resident_bytes": _peak_resident_bytes(),
        "position": state["position"].numpy().reshape(-1).tolist(),
        "self_key_tail_nonzero": int(np.count_nonzero(self_keys[..., 1:, :])),
        "self_value_tail_nonzero": int(np.count_nonzero(self_values[..., 1:, :])),
    }


def verify_full_asset(asset: Path, *, max_resident_bytes: int) -> dict[str, object]:
    return asyncio.run(_verify(asset, max_resident_bytes))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Verify full native Whisper large-v2 asset")
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--max-resident-gib", type=float, default=42.0)
    args = parser.parse_args()
    result = verify_full_asset(
        args.asset,
        max_resident_bytes=int(args.max_resident_gib * 1024**3),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
