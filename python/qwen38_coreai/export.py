"""Safe Qwen3.8-27B Core AI export entrypoint.

The real 27B export is intentionally opt-in and starts only after this module verifies the source
geometry and at least 160 GiB of free scratch space. It never removes a model cache, an existing
bundle, or a scratch directory to make room.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .contract import GIB, Qwen38Architecture, Qwen38ContractError, build_metadata
from .weights import quantization_spec_for_module


def _coreai_weight_spec(*, bits: int, block_size: int) -> dict[str, Any]:
    return {
        "dtype": f"int{bits}",
        "qscheme": "asymmetric",
        "granularity": {
            "type": "per_block",
            "block_size": block_size,
            "axis": 1,
        },
    }


def _source_module_to_author_module(module_name: str) -> str:
    if module_name.startswith("language_model.model."):
        return "model." + module_name[len("language_model.model.") :]
    if module_name.startswith("language_model.lm_head"):
        return "lm_head" + module_name[len("language_model.lm_head") :]
    raise Qwen38ContractError(
        f"unsupported Qwen3.8 quantization override module: {module_name}"
    )


def build_coreai_quantization_config(source_config: dict[str, Any]) -> dict[str, Any]:
    """Translate the derivative's affine Q4/Q8 policy to CoreAI weight quantization."""
    raw = source_config.get("quantization_config", source_config.get("quantization"))
    if not isinstance(raw, dict):
        raise Qwen38ContractError("source config has no quantization policy")
    default = quantization_spec_for_module(raw, "__checkpoint_default__")
    if (default.bits, default.group_size, default.mode) != (4, 32, "affine"):
        raise Qwen38ContractError(
            "Qwen3.8 source default quantization must be affine int4 group32"
        )

    int8_module_config = {
        "module_state_spec": {"weight": _coreai_weight_spec(bits=8, block_size=64)},
        "op_input_spec": None,
        "op_output_spec": None,
    }
    module_name_configs: dict[str, Any] = {}
    for module_name, override in raw.items():
        if not isinstance(override, dict):
            continue
        spec = quantization_spec_for_module(raw, module_name)
        if spec.bits == 8 and spec.group_size == 64:
            module_name_configs[_source_module_to_author_module(module_name)] = int8_module_config
        elif (spec.bits, spec.group_size) != (4, 32):
            raise Qwen38ContractError(
                f"unsupported Qwen3.8 override for {module_name}: "
                f"int{spec.bits} group{spec.group_size}"
            )

    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": _coreai_weight_spec(bits=4, block_size=32)},
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
        },
        "module_name_configs": module_name_configs,
    }


@dataclass(frozen=True)
class Qwen38ExportPlan:
    """Validated, allocation-free recipe for a native Qwen3.8 bundle export."""

    source_model: str
    output_bundle: str
    scratch_root: str
    architecture: Qwen38Architecture

    minimum_scratch_bytes = 160 * GIB

    @property
    def metadata(self) -> dict[str, Any]:
        return build_metadata(
            name=Path(self.output_bundle).name,
            architecture=self.architecture,
        )

    def require_scratch_capacity(self, available_bytes: int) -> None:
        if available_bytes < self.minimum_scratch_bytes:
            available_gib = available_bytes / GIB
            raise Qwen38ContractError(
                "Qwen3.8-27B Core AI export requires at least 160 GiB free scratch; "
                f"found {available_gib:.1f} GiB. caix will not delete existing data or "
                "silently use a smaller workspace."
            )

    def validate_local_paths(self) -> None:
        source = Path(self.source_model)
        scratch = Path(self.scratch_root)
        if not source.is_dir():
            raise Qwen38ContractError(f"source model directory does not exist: {source}")
        if not (source / "config.json").is_file():
            raise Qwen38ContractError(f"source model has no config.json: {source}")
        if not scratch.is_dir():
            raise Qwen38ContractError(f"scratch root must already exist: {scratch}")
        if Path(self.output_bundle).exists():
            raise Qwen38ContractError(
                f"refusing to overwrite existing export bundle: {self.output_bundle}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "caix.qwen3_8.export-plan.v1",
            "source_model": self.source_model,
            "output_bundle": self.output_bundle,
            "scratch_root": self.scratch_root,
            "minimum_scratch_bytes": self.minimum_scratch_bytes,
            "key_value_cache_bytes_at_262k": self.architecture.key_value_cache_bytes,
            "metadata": self.metadata,
            "quantization": {
                "body": "affine_int4_group32",
                "embedding_lm_head_linear_out_and_tail_mlp": "affine_int8_group64",
                "mtp_sidecar": "float16",
            },
        }


def plan_from_source(*, source_model: Path, output_bundle: Path, scratch_root: Path) -> Qwen38ExportPlan:
    try:
        config = json.loads((source_model / "config.json").read_text())
    except FileNotFoundError as error:
        raise Qwen38ContractError(f"source model has no config.json: {source_model}") from error
    except json.JSONDecodeError as error:
        raise Qwen38ContractError(f"source config.json is invalid JSON: {error}") from error
    return Qwen38ExportPlan(
        source_model=str(source_model),
        output_bundle=str(output_bundle),
        scratch_root=str(scratch_root),
        architecture=Qwen38Architecture.from_config(config),
    )


def export_target(plan: Qwen38ExportPlan, *, mmap_directory: Path | None = None) -> Path:
    """Author the validated 27B target with its source Q4/Q8 policy.

    Heavy dependencies stay lazy so structural validation and tests do not require
    the CoreAI Python toolchain. Existing mmap shards are reusable, while every
    final/quantized output path is fail-closed to prevent accidental overwrite.
    """
    import torch
    from coreai_models.export.compression import quantize_for_export
    from coreai_models.export.macos import export_macos_model
    from coreai_models.export.metadata import build_aimodel_metadata
    from transformers import AutoTokenizer

    from .model import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    source = Path(plan.source_model)
    output = Path(plan.output_bundle)
    scratch = Path(plan.scratch_root)
    partial = output.with_name(output.name + ".partial")
    mmap_root = mmap_directory or scratch / (output.name + ".fp16-mmap")
    quantized_root = scratch / (output.name + ".quantized-mmap")
    if partial.exists() or quantized_root.exists():
        raise Qwen38ContractError(
            f"refusing stale partial/quantized export at {partial} or {quantized_root}"
        )

    raw_config = json.loads((source / "config.json").read_text())
    text_config = Qwen3_5TextConfig(**raw_config["text_config"])
    if mmap_root.exists():
        model = Qwen3_5ForCausalLM.from_mmap_checkpoint(
            source, mmap_directory=mmap_root, target_dtype=torch.float16
        )
    else:
        model = Qwen3_5ForCausalLM.from_mlx_checkpoint(
            source, mmap_directory=mmap_root, target_dtype=torch.float16
        )
    model.eval()
    quantized_root.mkdir(parents=True, exist_ok=False)
    quantized = quantize_for_export(
        model,
        text_config,
        torch.float16,
        build_coreai_quantization_config(raw_config),
        mmap_dir=str(quantized_root),
    )
    program = export_macos_model(
        quantized,
        text_config,
        SimpleNamespace(max_context_length=plan.architecture.max_context_length,
                        include_debug_info=False),
    )

    partial.mkdir(parents=True, exist_ok=False)
    program.save_asset(
        partial / "model.aimodel",
        build_aimodel_metadata("Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"),
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.save_pretrained(partial / "tokenizer")
    (partial / "metadata.json").write_text(json.dumps(plan.metadata, indent=2) + "\n")
    (partial / "export-plan.json").write_text(
        json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n"
    )
    partial.rename(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument(
        "--mmap-directory",
        type=Path,
        help="reuse validated FP16 mmap shards from a prior checkpoint reconstruction",
    )
    parser.add_argument(
        "--write-structural-plan",
        action="store_true",
        help="write only metadata.json and export-plan.json; does not convert model weights",
    )
    args = parser.parse_args(argv)

    plan = plan_from_source(
        source_model=args.source_model,
        output_bundle=args.output_bundle,
        scratch_root=args.scratch_root,
    )
    plan.validate_local_paths()
    plan.require_scratch_capacity(shutil.disk_usage(args.scratch_root).free)
    rendered = json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n"
    if args.write_structural_plan:
        output = Path(plan.output_bundle)
        output.mkdir(parents=True, exist_ok=False)
        (output / "metadata.json").write_text(json.dumps(plan.metadata, indent=2) + "\n")
        (output / "export-plan.json").write_text(rendered)
        return 0

    exported = export_target(plan, mmap_directory=args.mmap_directory)
    print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
