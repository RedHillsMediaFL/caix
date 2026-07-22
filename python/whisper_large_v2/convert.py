"""Pinned, bounded-memory conversion entrypoint for Whisper large-v2."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import resource
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from whisper_large_v2.checkpoint import (
    WhisperSourceContract,
    _verified_regular_file,
    expected_large_v2_inventory,
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

_FP32_BYTES = 4
_FP16_BYTES = 2
_DEFAULT_MAX_RESIDENT_BYTES = 12 * 1024**3
_MINIMUM_AVAILABLE_BYTES = 8 * 1024**3
_EXACT_AUTHORING_STACK = {
    "coreai-core": "1.0.0b2",
    "coreai-opt": "0.2.0",
    "coreai-torch": "0.4.1",
    "torch": "2.9.0",
    "transformers": "4.57.6",
}


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
        for distribution in _EXACT_AUTHORING_STACK
    }
    if actual != _EXACT_AUTHORING_STACK:
        raise WhisperExportError(
            f"Whisper authoring stack differs: expected={_EXACT_AUTHORING_STACK!r}, "
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


def _enforce_resident_budget(max_resident_bytes: int, *, phase: str) -> None:
    resident = _resident_bytes()
    if resident > max_resident_bytes:
        raise WhisperExportError(
            f"Whisper conversion exceeded resident cap during {phase}: "
            f"{resident} > {max_resident_bytes} bytes"
        )


def _enforce_available_memory() -> None:
    import psutil

    available = psutil.virtual_memory().available
    if available < _MINIMUM_AVAILABLE_BYTES:
        raise WhisperExportError(
            f"Whisper conversion requires 8 GiB available memory; found {available} bytes"
        )


def _empty_model(config: object) -> torch.nn.Module:
    from transformers import WhisperForConditionalGeneration

    with torch.device("meta"):
        model = WhisperForConditionalGeneration(config)
    return model.eval()


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
    from transformers import WhisperConfig

    contract = load_source_contract(source_contract_path)
    if snapshot.name != contract.revision:
        raise WhisperExportError("snapshot directory does not match the pinned revision")
    validate_pinned_snapshot(snapshot, contract)
    plan = WhisperWeightPlan.large_v2()
    if _resident_bytes() + plan.bounded_weight_working_set_bytes > max_resident_bytes:
        raise WhisperExportError(
            "resident cap cannot hold the bounded Whisper FP16 weight working set"
        )

    config = WhisperConfig.from_pretrained(snapshot, local_files_only=True)
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
        _enforce_resident_budget(max_resident_bytes, phase="encoder load")

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
        _enforce_resident_budget(max_resident_bytes, phase="decoder load")

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


def save_program_atomically(program: object, output: Path) -> None:
    """Save one asset through a sibling staging directory without partial output."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"CoreAI output already exists: {output}")
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    staged_asset = staging_root / output.name
    try:
        program.save_asset(staged_asset)
        if output.exists():
            raise FileExistsError(f"CoreAI output appeared while saving: {output}")
        staged_asset.replace(output)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def export_pinned_large_v2(
    snapshot: Path,
    source_contract_path: Path,
    output: Path,
    *,
    max_resident_bytes: int,
) -> dict[str, object]:
    """Author and atomically save the complete FP16 three-entrypoint AIProgram."""
    stack = require_exact_authoring_stack()
    _enforce_available_memory()
    loaded = load_pinned_large_v2(
        snapshot,
        source_contract_path,
        max_resident_bytes=max_resident_bytes,
    )
    _enforce_resident_budget(max_resident_bytes, phase="full checkpoint load")
    input_features, token_id = full_export_inputs()
    program = create_coreai_program(
        loaded.split,
        input_features=input_features,
        token_id=token_id,
    )
    _enforce_resident_budget(max_resident_bytes, phase="CoreAI graph authoring")
    del loaded
    gc.collect()
    save_program_atomically(program, output)
    _enforce_resident_budget(max_resident_bytes, phase="CoreAI asset save")
    asset_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    return {
        "asset_bytes": asset_bytes,
        "authoring_stack": stack,
        "dtype": "float16",
        "entrypoints": ["encode", "load_cross_kv", "decode_step"],
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
        payload = export_pinned_large_v2(
            args.snapshot,
            args.source_contract,
            args.output,
            max_resident_bytes=int(args.max_resident_gib * 1024**3),
        )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
