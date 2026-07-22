"""Execute the complete saved Whisper large-v2 CoreAI state machine."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from whisper_large_v2.abi import NativeWhisperABI
from whisper_large_v2.export import WhisperExportError
from whisper_large_v2.manifest import validate_caix_asset


def _peak_resident_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


_STATE_NAMES = (
    "cross_key_cache",
    "cross_value_cache",
    "self_key_cache",
    "self_value_cache",
    "position",
    "cross_ready",
)


def _check_recorded_peak(max_resident_bytes: int, *, phase: str) -> int:
    peak = _peak_resident_bytes()
    if peak > max_resident_bytes:
        raise WhisperExportError(
            f"full Whisper verification recorded peak RSS exceeded the threshold "
            f"during {phase}: "
            f"{peak} > {max_resident_bytes} bytes"
        )
    return peak


def _require_array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    array = value.numpy()
    if tuple(array.shape) != shape or array.dtype.name != dtype:
        raise WhisperExportError(
            f"{label} tensor contract differs: "
            f"expected={dtype}{shape!r}, actual={array.dtype.name}{array.shape!r}"
        )
    return array


def _require_status(value: Any, *, label: str, expected: int) -> np.ndarray:
    array = _require_array(value, label=label, shape=(1,), dtype="int32")
    if array.tolist() != [expected]:
        raise WhisperExportError(
            f"{label} differs: expected={[expected]!r}, actual={array.tolist()!r}"
        )
    return array


def _snapshot_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: np.array(state[name].numpy(), copy=True)
        for name in _STATE_NAMES
    }


def _state_matches(
    state: dict[str, Any],
    expected: dict[str, np.ndarray],
) -> bool:
    return all(
        np.array_equal(state[name].numpy(), expected[name])
        for name in _STATE_NAMES
    )


def _exact_max_error(actual: np.ndarray, expected: np.ndarray, *, label: str) -> float:
    """Compare exactly while bounding temporary float32 differences to one layer."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise WhisperExportError(f"{label} tensor metadata differs")
    actual_slices = actual if actual.ndim else actual.reshape((1,))
    expected_slices = expected if expected.ndim else expected.reshape((1,))
    maximum = 0.0
    for actual_slice, expected_slice in zip(actual_slices, expected_slices, strict=True):
        difference = np.abs(
            actual_slice.astype(np.float32) - expected_slice.astype(np.float32)
        )
        if difference.size:
            maximum = max(maximum, float(np.max(difference)))
        if not np.array_equal(actual_slice, expected_slice):
            raise WhisperExportError(
                f"{label} differs from the authenticated encoder payload; "
                f"max_error={maximum}"
            )
    return maximum


def _new_state(
    ndarray_type: Any,
    abi: Any,
    *,
    readiness: int = 0,
    position: int = 0,
    sentinel: bool = False,
) -> dict[str, Any]:
    fills = {
        "cross_key_cache": 1.25 if sentinel else 0.0,
        "cross_value_cache": -2.5 if sentinel else 0.0,
        "self_key_cache": 3.75 if sentinel else 0.0,
        "self_value_cache": -4.5 if sentinel else 0.0,
    }
    state = {
        name: ndarray_type(
            data=torch.full(
                abi.tensors[name].shape,
                fill_value,
                dtype=torch.float16,
            )
        )
        for name, fill_value in fills.items()
    }
    state["position"] = ndarray_type(
        data=torch.tensor([position], dtype=torch.int32)
    )
    state["cross_ready"] = ndarray_type(
        data=torch.tensor([readiness], dtype=torch.int32)
    )
    return state


async def _verify(asset: Path, max_resident_bytes: int) -> dict[str, object]:
    asset_validation = validate_caix_asset(asset)
    from coreai.runtime import AIModel, NDArray

    if not asset.is_dir() or asset.suffix != ".aimodel":
        raise WhisperExportError(f"Whisper asset is missing or not .aimodel: {asset}")
    abi = NativeWhisperABI.large_v2()
    state = _new_state(NDArray, abi)
    features = NDArray(
        data=torch.zeros(abi.tensors["input_features"].shape, dtype=torch.float16)
    )
    token_id = NDArray(data=torch.tensor([[50_258]], dtype=torch.int32))

    started = time.perf_counter()
    model = await AIModel.load(asset)
    asset_load_seconds = time.perf_counter() - started
    _check_recorded_peak(max_resident_bytes, phase="asset load")
    entrypoints = sorted(model.function_names)
    if entrypoints != ["decode_step", "encode", "load_cross_kv"]:
        raise WhisperExportError(f"Whisper asset entrypoints differ: {entrypoints!r}")

    encode = model.load_function("encode")
    load_cross_kv = model.load_function("load_cross_kv")
    decode_step = model.load_function("decode_step")

    invalid_before_snapshot = _snapshot_state(state)
    invalid_before_load = await decode_step({"token_id": token_id}, state=state)
    invalid_before_status = _require_status(
        invalid_before_load["decode_status"],
        label="decode-before-load status",
        expected=abi.invalid_state_status,
    )
    invalid_before_logits = _require_array(
        invalid_before_load["logits"],
        label="decode-before-load logits",
        shape=abi.tensors["logits"].shape,
        dtype=abi.tensors["logits"].dtype,
    )
    invalid_before_unchanged = _state_matches(state, invalid_before_snapshot)
    if not invalid_before_unchanged or np.count_nonzero(invalid_before_logits):
        raise WhisperExportError("decode-before-load mutated state or returned usable logits")
    del invalid_before_snapshot
    _check_recorded_peak(max_resident_bytes, phase="decode-before-load proof")

    started = time.perf_counter()
    encoded = await encode({"input_features": features})
    encode_seconds = time.perf_counter() - started
    _check_recorded_peak(max_resident_bytes, phase="encode")
    cross_keys = _require_array(
        encoded["cross_key_payload"],
        label="cross key payload",
        shape=abi.tensors["cross_key_cache"].shape,
        dtype=abi.tensors["cross_key_cache"].dtype,
    )
    cross_values = _require_array(
        encoded["cross_value_payload"],
        label="cross value payload",
        shape=abi.tensors["cross_value_cache"].shape,
        dtype=abi.tensors["cross_value_cache"].dtype,
    )

    started = time.perf_counter()
    loaded = await load_cross_kv(
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
    cross_kv_load_seconds = time.perf_counter() - started
    load_status = _require_status(
        loaded["load_status"],
        label="cross cache load status",
        expected=abi.success_status,
    )
    loaded_cross_keys = _require_array(
        state["cross_key_cache"],
        label="loaded cross key cache",
        shape=abi.tensors["cross_key_cache"].shape,
        dtype=abi.tensors["cross_key_cache"].dtype,
    )
    loaded_cross_values = _require_array(
        state["cross_value_cache"],
        label="loaded cross value cache",
        shape=abi.tensors["cross_value_cache"].shape,
        dtype=abi.tensors["cross_value_cache"].dtype,
    )
    cross_key_error = _exact_max_error(
        loaded_cross_keys,
        cross_keys,
        label="loaded cross key cache",
    )
    cross_value_error = _exact_max_error(
        loaded_cross_values,
        cross_values,
        label="loaded cross value cache",
    )
    _check_recorded_peak(max_resident_bytes, phase="cross cache load")

    before_second_load = _snapshot_state(state)
    second_load = await load_cross_kv(
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
    second_load_status = _require_status(
        second_load["load_status"],
        label="second cross cache load status",
        expected=abi.invalid_state_status,
    )
    second_load_unchanged = _state_matches(state, before_second_load)
    if not second_load_unchanged:
        raise WhisperExportError("second cross cache load mutated decoder state")
    del before_second_load
    _check_recorded_peak(max_resident_bytes, phase="second-load proof")

    started = time.perf_counter()
    decoded = await decode_step({"token_id": token_id}, state=state)
    decode_seconds = time.perf_counter() - started
    decode_status = _require_status(
        decoded["decode_status"],
        label="decode status",
        expected=abi.success_status,
    )
    logits = _require_array(
        decoded["logits"],
        label="decode logits",
        shape=abi.tensors["logits"].shape,
        dtype=abi.tensors["logits"].dtype,
    )
    position = _require_array(
        state["position"],
        label="decoder position",
        shape=(1,),
        dtype="int32",
    )
    cross_ready = _require_array(
        state["cross_ready"],
        label="cross readiness",
        shape=(1,),
        dtype="int32",
    )
    if position.tolist() != [1] or cross_ready.tolist() != [1]:
        raise WhisperExportError("successful decode state transition differs")
    if not np.isfinite(logits).all():
        raise WhisperExportError("successful decode returned non-finite logits")
    self_keys = _require_array(
        state["self_key_cache"],
        label="self key cache",
        shape=abi.tensors["self_key_cache"].shape,
        dtype=abi.tensors["self_key_cache"].dtype,
    )
    self_values = _require_array(
        state["self_value_cache"],
        label="self value cache",
        shape=abi.tensors["self_value_cache"].shape,
        dtype=abi.tensors["self_value_cache"].dtype,
    )
    self_key_current_nonzero = int(np.count_nonzero(self_keys[..., 0, :]))
    self_value_current_nonzero = int(np.count_nonzero(self_values[..., 0, :]))
    self_key_tail_nonzero = int(np.count_nonzero(self_keys[..., 1:, :]))
    self_value_tail_nonzero = int(np.count_nonzero(self_values[..., 1:, :]))
    if (
        not self_key_current_nonzero
        or not self_value_current_nonzero
        or self_key_tail_nonzero
        or self_value_tail_nonzero
    ):
        raise WhisperExportError("successful decode self-cache slot mutation differs")
    _check_recorded_peak(max_resident_bytes, phase="decode step")

    async def prove_invalid_decode(
        *, readiness: int, invalid_position: int
    ) -> tuple[list[int], bool, bool]:
        invalid_state = _new_state(
            NDArray,
            abi,
            readiness=readiness,
            position=invalid_position,
            sentinel=True,
        )
        snapshot = _snapshot_state(invalid_state)
        output = await decode_step({"token_id": token_id}, state=invalid_state)
        status = _require_status(
            output["decode_status"],
            label="invalid decode status",
            expected=abi.invalid_state_status,
        )
        invalid_logits = _require_array(
            output["logits"],
            label="invalid decode logits",
            shape=abi.tensors["logits"].shape,
            dtype=abi.tensors["logits"].dtype,
        )
        unchanged = _state_matches(invalid_state, snapshot)
        zero_logits = bool(np.count_nonzero(invalid_logits) == 0)
        del invalid_state, snapshot
        gc.collect()
        if not unchanged or not zero_logits:
            raise WhisperExportError("invalid decode mutated state or returned usable logits")
        return status.tolist(), unchanged, zero_logits

    capacity = abi.tensors["self_key_cache"].shape[-2]
    invalid_position_results = [
        await prove_invalid_decode(readiness=1, invalid_position=value)
        for value in (-1, capacity)
    ]
    invalid_readiness_results = [
        await prove_invalid_decode(readiness=value, invalid_position=0)
        for value in (-1, 2)
    ]
    _check_recorded_peak(max_resident_bytes, phase="invalid-state proofs")
    recorded_peak = _check_recorded_peak(max_resident_bytes, phase="verification result")

    return {
        "asset_bytes": asset_validation.asset_bytes,
        "asset_load_seconds": asset_load_seconds,
        "cross_key_shape": list(cross_keys.shape),
        "cross_key_cache_max_error": cross_key_error,
        "cross_key_dtype": cross_keys.dtype.name,
        "cross_kv_load_seconds": cross_kv_load_seconds,
        "cross_ready": cross_ready.tolist(),
        "cross_ready_dtype": cross_ready.dtype.name,
        "cross_value_cache_max_error": cross_value_error,
        "cross_value_dtype": cross_values.dtype.name,
        "cross_value_shape": list(cross_values.shape),
        "decode_status": decode_status.tolist(),
        "decode_status_dtype": decode_status.dtype.name,
        "decode_seconds": decode_seconds,
        "encode_seconds": encode_seconds,
        "entrypoints": entrypoints,
        "invalid_decode_before_load_state_unchanged": invalid_before_unchanged,
        "invalid_decode_before_load_status": invalid_before_status.tolist(),
        "invalid_decode_before_load_zero_logits": True,
        "invalid_position_state_unchanged": [
            result[1] for result in invalid_position_results
        ],
        "invalid_position_statuses": [
            result[0] for result in invalid_position_results
        ],
        "invalid_position_zero_logits": [
            result[2] for result in invalid_position_results
        ],
        "invalid_readiness_state_unchanged": [
            result[1] for result in invalid_readiness_results
        ],
        "invalid_readiness_statuses": [
            result[0] for result in invalid_readiness_results
        ],
        "invalid_readiness_zero_logits": [
            result[2] for result in invalid_readiness_results
        ],
        "invalid_second_load_state_unchanged": second_load_unchanged,
        "invalid_second_load_status": second_load_status.tolist(),
        "load_status": load_status.tolist(),
        "load_status_dtype": load_status.dtype.name,
        "logits_all_finite": bool(np.isfinite(logits).all()),
        "logits_dtype": logits.dtype.name,
        "logits_shape": list(logits.shape),
        "main_sha256": asset_validation.main_sha256,
        "position": position.tolist(),
        "position_dtype": position.dtype.name,
        "recorded_peak_rss_bytes": recorded_peak,
        "self_key_current_slot_nonzero": self_key_current_nonzero,
        "self_key_tail_nonzero": self_key_tail_nonzero,
        "self_value_current_slot_nonzero": self_value_current_nonzero,
        "self_value_tail_nonzero": self_value_tail_nonzero,
    }


def verify_full_asset(asset: Path, *, max_resident_bytes: int) -> dict[str, object]:
    return asyncio.run(_verify(asset, max_resident_bytes))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Verify full native Whisper large-v2 asset")
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--max-resident-gib", type=float, default=12.0)
    args = parser.parse_args()
    result = verify_full_asset(
        args.asset,
        max_resident_bytes=int(args.max_resident_gib * 1024**3),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
