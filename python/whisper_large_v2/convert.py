"""Pinned, bounded-memory conversion entrypoint for Whisper large-v2."""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.metadata
import json
import math
import os
import resource
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from whisper_large_v2.authoring_source import (
    authenticated_authoring_source,
    load_authoring_source_contract,
)
from whisper_large_v2.checkpoint import (
    WhisperSourceContract,
    _verified_regular_file,
    expected_large_v2_inventory,
    load_verified_json_asset,
    load_source_contract,
    validate_pinned_snapshot,
)
from whisper_large_v2.export import (
    WhisperDecodeStep,
    WhisperEncode,
    WhisperExportError,
    WhisperLoadCrossKV,
    WhisperSplitModules,
    create_coreai_program,
)
from whisper_large_v2.manifest import (
    EXACT_AUTHORING_STACK,
    AssetValidation,
    validate_caix_asset,
    write_caix_manifest,
)

_FP32_BYTES = 4
_FP16_BYTES = 2
_DEFAULT_MAX_RESIDENT_BYTES = 12 * 1024**3
_MINIMUM_AVAILABLE_BYTES = 8 * 1024**3
_EXPECTED_ASSET_BYTES = 3_100_000_000
_DISK_HEADROOM_BYTES = 2 * 1024**3
_AT_FDCWD = -2
_RENAME_SWAP = 0x00000002
_RENAME_EXCL = 0x00000004


class AssetPromotionRecoveryError(RuntimeError):
    """Promotion changed the final path but retained both assets for recovery."""


def _is_encoder_cross_key(key: str) -> bool:
    return key.startswith("model.encoder.") or (
        ".encoder_attn." in key
        and key.endswith(("k_proj.weight", "v_proj.weight", "v_proj.bias"))
    )


@dataclass(frozen=True)
class WhisperWeightPlan:
    encoder_keys: tuple[str, ...]
    decoder_keys: tuple[str, ...]
    fp16_encoder_bytes: int
    fp16_decoder_bytes: int
    largest_source_tensor_bytes: int

    @classmethod
    def large_v2(cls) -> WhisperWeightPlan:
        inventory = expected_large_v2_inventory()
        encoder_keys = tuple(sorted(key for key in inventory if _is_encoder_cross_key(key)))
        decoder_keys = tuple(sorted(set(inventory) - set(encoder_keys)))

        def tensor_bytes(keys: tuple[str, ...], width: int) -> int:
            return sum(math.prod(inventory[key].shape) * width for key in keys)

        largest_source = max(
            math.prod(entry.shape) * _FP32_BYTES for entry in inventory.values()
        )
        return cls(
            encoder_keys=encoder_keys,
            decoder_keys=decoder_keys,
            fp16_encoder_bytes=tensor_bytes(encoder_keys, _FP16_BYTES),
            fp16_decoder_bytes=tensor_bytes(decoder_keys, _FP16_BYTES),
            largest_source_tensor_bytes=largest_source,
        )

    @property
    def tensor_count(self) -> int:
        return len(self.encoder_keys) + len(self.decoder_keys)

    @property
    def fp16_parameter_bytes(self) -> int:
        return self.fp16_encoder_bytes + self.fp16_decoder_bytes

    @property
    def bounded_weight_working_set_bytes(self) -> int:
        return self.fp16_parameter_bytes + self.largest_source_tensor_bytes


@dataclass(frozen=True)
class LoadedWhisperLargeV2:
    split: WhisperSplitModules
    loaded_tensor_count: int
    source_sha256: str


def require_exact_authoring_stack() -> dict[str, str]:
    actual = {
        distribution: importlib.metadata.version(distribution)
        for distribution in EXACT_AUTHORING_STACK
    }
    if actual != EXACT_AUTHORING_STACK:
        raise WhisperExportError(
            f"Whisper authoring stack differs: expected={EXACT_AUTHORING_STACK!r}, "
            f"actual={actual!r}"
        )
    return actual


def full_export_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-shape inputs for the pinned large-v2 ABI."""
    return (
        torch.zeros((1, 80, 3000), dtype=torch.float16),
        torch.tensor([[50_258]], dtype=torch.int32),
    )


def _resident_bytes() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def _peak_resident_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _enforce_peak_resident_budget(
    max_resident_bytes: int,
    *,
    phase: str,
    peak_reader: Callable[[], int] = _peak_resident_bytes,
) -> int:
    peak = peak_reader()
    if peak > max_resident_bytes:
        raise WhisperExportError(
            f"Whisper conversion resident peak exceeded cap during {phase}: "
            f"{peak} > {max_resident_bytes} bytes"
        )
    return peak


def _enforce_available_memory() -> None:
    import psutil

    available = psutil.virtual_memory().available
    if available < _MINIMUM_AVAILABLE_BYTES:
        raise WhisperExportError(
            f"Whisper conversion requires 8 GiB available memory; found {available} bytes"
        )


def require_available_disk(
    output: Path,
    *,
    expected_asset_bytes: int = _EXPECTED_ASSET_BYTES,
    headroom_bytes: int = _DISK_HEADROOM_BYTES,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> int:
    """Require space for a complete sibling candidate plus conservative headroom."""
    probe = output.absolute().parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = int(disk_usage(probe).free)
    required = expected_asset_bytes + headroom_bytes
    if available < required:
        raise WhisperExportError(
            f"Whisper conversion has insufficient available disk: "
            f"{available} < {required} bytes"
        )
    return available


def _empty_model(config: object) -> torch.nn.Module:
    from transformers import WhisperForConditionalGeneration

    with torch.device("meta"):
        model = WhisperForConditionalGeneration(config)
    return model.eval()


def _load_pinned_config(
    snapshot: Path,
    contract: WhisperSourceContract,
) -> object:
    from transformers import WhisperConfig

    return WhisperConfig.from_dict(
        load_verified_json_asset(snapshot, contract, "config.json")
    )


def _load_partition(
    handle: object,
    *,
    config: object,
    keys: tuple[str, ...],
    dtype: torch.dtype,
) -> torch.nn.Module:
    model = _empty_model(config)
    values: dict[str, torch.Tensor] = {}
    for key in keys:
        source = handle.get_tensor(key)
        if source.dtype != torch.float32:
            raise WhisperExportError(f"source tensor is not FP32: {key}")
        values[key] = source.to(dtype=dtype).contiguous()
        del source

    incompatible = model.load_state_dict(values, strict=False, assign=True)
    if incompatible.unexpected_keys:
        raise WhisperExportError(
            f"unexpected Whisper parameters while loading: {incompatible.unexpected_keys!r}"
        )
    values.clear()
    return model


def _require_materialized(module: torch.nn.Module, *, label: str) -> None:
    meta = [name for name, parameter in module.named_parameters() if parameter.is_meta]
    if meta:
        raise WhisperExportError(f"{label} retained meta parameters: {meta!r}")
    dtypes = {parameter.dtype for parameter in module.parameters()}
    if dtypes != {torch.float16}:
        raise WhisperExportError(f"{label} parameter dtypes differ: {dtypes!r}")


def load_pinned_large_v2(
    snapshot: Path,
    source_contract_path: Path,
    *,
    max_resident_bytes: int = _DEFAULT_MAX_RESIDENT_BYTES,
) -> LoadedWhisperLargeV2:
    """Load exact large-v2 FP32 tensors into a partitioned FP16 meta model."""
    contract = load_source_contract(source_contract_path)
    if snapshot.name != contract.revision:
        raise WhisperExportError("snapshot directory does not match the pinned revision")
    validate_pinned_snapshot(snapshot, contract)
    plan = WhisperWeightPlan.large_v2()
    if _resident_bytes() + plan.bounded_weight_working_set_bytes > max_resident_bytes:
        raise WhisperExportError(
            "resident cap cannot hold the bounded Whisper FP16 weight working set"
        )

    config = _load_pinned_config(snapshot, contract)
    weight_path = snapshot / contract.weights.path
    with _verified_regular_file(
        weight_path,
        expected_sha256=contract.weights.sha256,
        expected_size=contract.weights.size_bytes,
        label="pinned checkpoint weight",
    ) as verified_weight, safe_open(
        str(verified_weight.descriptor_path),
        framework="pt",
        device="cpu",
    ) as handle:
        encoder_model = _load_partition(
            handle,
            config=config,
            keys=plan.encoder_keys,
            dtype=torch.float16,
        )
        encode = WhisperEncode(encoder_model).eval()
        _require_materialized(encode, label="Whisper encoder/cross partition")
        del encoder_model
        gc.collect()
        _enforce_peak_resident_budget(max_resident_bytes, phase="encoder load")

        decoder_model = _load_partition(
            handle,
            config=config,
            keys=plan.decoder_keys,
            dtype=torch.float16,
        )
        decode_step = WhisperDecodeStep(decoder_model).eval()
        _require_materialized(decode_step, label="Whisper decoder partition")
        del decoder_model
        gc.collect()
        _enforce_peak_resident_budget(max_resident_bytes, phase="decoder load")

    split = WhisperSplitModules(
        encode=encode,
        load_cross_kv=WhisperLoadCrossKV().eval(),
        decode_step=decode_step,
        decoder_layers=config.decoder_layers,
        decoder_heads=config.decoder_attention_heads,
        head_dim=config.d_model // config.decoder_attention_heads,
        source_positions=config.max_source_positions,
    )
    return LoadedWhisperLargeV2(
        split=split,
        loaded_tensor_count=plan.tensor_count,
        source_sha256=contract.weights.sha256,
    )


def _renameatx(source: Path, destination: Path, flags: int) -> None:
    if sys.platform != "darwin":
        if flags == _RENAME_EXCL and not os.path.lexists(destination):
            os.rename(source, destination)
            return
        raise WhisperExportError("recoverable asset promotion requires Darwin renameatx_np")
    libc = ctypes.CDLL(None, use_errno=True)
    renameatx_np = libc.renameatx_np
    renameatx_np.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameatx_np.restype = ctypes.c_int
    result = renameatx_np(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def promote_candidate(candidate: Path, output: Path) -> None:
    """Atomically promote a verified sibling while retaining the prior asset."""
    if os.path.lexists(output):
        validate_caix_asset(output)
        _renameatx(candidate, output, _RENAME_SWAP)
        try:
            validate_caix_asset(output)
            _fsync_parent(output)
        except Exception as promotion_error:
            try:
                _renameatx(candidate, output, _RENAME_SWAP)
                _fsync_parent(output)
            except Exception as recovery_error:
                raise AssetPromotionRecoveryError(
                    "asset promotion recovery failed; both verified assets were retained"
                ) from recovery_error
            raise promotion_error
        try:
            shutil.rmtree(candidate)
            _fsync_parent(output)
        except Exception as cleanup_error:
            raise AssetPromotionRecoveryError(
                "new asset is published and the prior verified asset was retained"
            ) from cleanup_error
        return
    _renameatx(candidate, output, _RENAME_EXCL)
    _fsync_parent(output)


def save_program_atomically(
    program: object,
    output: Path,
    *,
    max_resident_bytes: int,
    manifest_writer: Callable[[Path], object] = write_caix_manifest,
    asset_validator: Callable[[Path], AssetValidation] = validate_caix_asset,
    candidate_promoter: Callable[[Path, Path], None] | None = None,
    peak_reader: Callable[[], int] = _peak_resident_bytes,
) -> AssetValidation:
    """Save, authenticate, and recoverably promote one sibling candidate asset."""
    output = Path(os.path.abspath(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    _enforce_peak_resident_budget(
        max_resident_bytes,
        phase="before CoreAI asset save",
        peak_reader=peak_reader,
    )
    candidate = output.parent / (
        f".{output.name}.candidate-{uuid.uuid4().hex}.aimodel"
    )
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    staged_asset = staging_root / output.name
    preserve_candidate = False
    try:
        program.save_asset(staged_asset)
        manifest_writer(staged_asset)
        _renameatx(staged_asset, candidate, _RENAME_EXCL)
        _fsync_parent(candidate)
        validation = asset_validator(candidate)
        _enforce_peak_resident_budget(
            max_resident_bytes,
            phase="pre-publication callback",
            peak_reader=peak_reader,
        )
        promoter = candidate_promoter or promote_candidate
        try:
            promoter(candidate, output)
        except AssetPromotionRecoveryError:
            preserve_candidate = True
            raise
        except Exception:
            if os.path.lexists(candidate):
                shutil.rmtree(candidate, ignore_errors=True)
            raise
        return validation
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if not preserve_candidate and os.path.lexists(candidate):
            shutil.rmtree(candidate, ignore_errors=True)


def export_pinned_large_v2(
    snapshot: Path,
    source_contract_path: Path,
    output: Path,
    *,
    authoring_source_path: Path,
    coreai_models_repository: Path,
    authoring_temp_root: Path,
    max_resident_bytes: int,
) -> dict[str, object]:
    """Author and atomically save the complete FP16 three-entrypoint AIProgram."""
    authoring_contract = load_authoring_source_contract(authoring_source_path)
    require_available_disk(output)
    _enforce_available_memory()
    with authenticated_authoring_source(
        coreai_models_repository,
        authoring_contract,
        temp_root=authoring_temp_root,
    ):
        stack = require_exact_authoring_stack()
        loaded = load_pinned_large_v2(
            snapshot,
            source_contract_path,
            max_resident_bytes=max_resident_bytes,
        )
        _enforce_peak_resident_budget(
            max_resident_bytes,
            phase="full checkpoint load",
        )
        input_features, token_id = full_export_inputs()
        program = create_coreai_program(
            loaded.split,
            input_features=input_features,
            token_id=token_id,
        )
        _enforce_peak_resident_budget(
            max_resident_bytes,
            phase="CoreAI graph authoring",
        )
        del loaded
        gc.collect()
        validation = save_program_atomically(
            program,
            output,
            max_resident_bytes=max_resident_bytes,
        )
        return {
            "asset_bytes": validation.asset_bytes,
            "authoring_stack": stack,
            "coreai_models_revision": authoring_contract.revision,
            "coreai_models_tree": authoring_contract.package_tree,
            "dtype": "float16",
            "entrypoints": ["encode", "load_cross_kv", "decode_step"],
            "main_sha256": validation.main_sha256,
            "output": str(output.resolve()),
            "peak_resident_bytes": _peak_resident_bytes(),
            "schema": "caix.whisper-split.v2",
        }


def _dry_run_payload(
    contract: WhisperSourceContract,
    *,
    output: Path,
) -> dict[str, object]:
    plan = WhisperWeightPlan.large_v2()
    return {
        "bounded_weight_working_set_bytes": plan.bounded_weight_working_set_bytes,
        "dtype": "float16",
        "output": str(output),
        "repository": contract.repository,
        "revision": contract.revision,
        "schema": "caix.whisper-split.v2",
        "tensor_count": plan.tensor_count,
    }


def _load_proof_payload(loaded: LoadedWhisperLargeV2) -> dict[str, object]:
    encode_parameters = tuple(loaded.split.encode.parameters())
    decode_parameters = tuple(loaded.split.decode_step.parameters())
    parameters = encode_parameters + decode_parameters
    return {
        "decoder_parameter_count": sum(value.numel() for value in decode_parameters),
        "encoder_parameter_count": sum(value.numel() for value in encode_parameters),
        "loaded_tensor_count": loaded.loaded_tensor_count,
        "meta_parameter_count": sum(int(value.is_meta) for value in parameters),
        "parameter_dtype": str(parameters[0].dtype).removeprefix("torch."),
        "peak_resident_bytes": _peak_resident_bytes(),
        "source_sha256": loaded.source_sha256,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Convert pinned Whisper large-v2 to CoreAI")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--authoring-source", type=Path)
    parser.add_argument("--coreai-models-repository", type=Path)
    parser.add_argument("--authoring-temp-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--load-proof", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--max-resident-gib", type=float, default=12.0)
    args = parser.parse_args()

    if sum((args.dry_run, args.load_proof, args.export)) != 1:
        parser.error("select exactly one of --dry-run, --load-proof, or --export")
    contract = load_source_contract(args.source_contract)
    if args.snapshot.name != contract.revision:
        parser.error("--snapshot must name the pinned revision")

    if args.dry_run:
        if args.output is None:
            parser.error("--output is required with --dry-run")
        payload = _dry_run_payload(contract, output=args.output)
    elif args.load_proof:
        loaded = load_pinned_large_v2(
            args.snapshot,
            args.source_contract,
            max_resident_bytes=int(args.max_resident_gib * 1024**3),
        )
        payload = _load_proof_payload(loaded)
    else:
        if args.output is None:
            parser.error("--output is required with --export")
        if args.authoring_source is None:
            parser.error("--authoring-source is required with --export")
        if args.coreai_models_repository is None:
            parser.error("--coreai-models-repository is required with --export")
        if args.authoring_temp_root is None:
            parser.error("--authoring-temp-root is required with --export")
        payload = export_pinned_large_v2(
            args.snapshot,
            args.source_contract,
            args.output,
            authoring_source_path=args.authoring_source,
            coreai_models_repository=args.coreai_models_repository,
            authoring_temp_root=args.authoring_temp_root,
            max_resident_bytes=int(args.max_resident_gib * 1024**3),
        )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
