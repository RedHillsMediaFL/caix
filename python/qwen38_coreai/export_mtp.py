"""Author the native Qwen3.8 MTP CoreAI sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .contract import Qwen38ContractError


def export_mtp_sidecar(
    *,
    source_model: Path,
    mmap_directory: Path,
    quantized_directory: Path,
    output_asset: Path,
) -> Path:
    """Reconstruct, Q8-compress, author, and atomically save the one-layer sidecar."""
    if not (source_model / "config.json").is_file():
        raise Qwen38ContractError(f"Qwen3.8 source has no config.json: {source_model}")
    if not (mmap_directory / "mtp.safetensors").is_file():
        raise Qwen38ContractError(f"Qwen3.8 MTP mmap is missing: {mmap_directory}")
    if quantized_directory.exists() or output_asset.exists():
        raise Qwen38ContractError(
            f"refusing existing MTP output: {quantized_directory} or {output_asset}")

    import torch
    from coreai_models.export.compression import quantize_for_export
    from coreai_models.export.macos import export_macos_model
    from coreai_models.export.metadata import build_aimodel_metadata

    from .export import build_mtp_quantization_config
    from .model import Qwen3_5TextConfig
    from .mtp_model import Qwen3_5MTPForCausalLM

    raw = json.loads((source_model / "config.json").read_text())
    config = Qwen3_5TextConfig(**raw["text_config"])
    model = Qwen3_5MTPForCausalLM.from_mmap_checkpoint(
        source_model,
        mmap_directory=mmap_directory,
        target_dtype=torch.float16,
    ).eval()
    quantized_directory.mkdir(parents=True, exist_ok=False)
    quantized = quantize_for_export(
        model,
        config,
        torch.float16,
        build_mtp_quantization_config(),
        mmap_dir=str(quantized_directory),
    )
    program = export_macos_model(
        quantized,
        config,
        SimpleNamespace(max_context_length=262_144, include_debug_info=False),
    )
    partial = output_asset.with_name(output_asset.name + ".partial")
    if partial.exists():
        raise Qwen38ContractError(f"refusing stale MTP partial asset: {partial}")
    program.save_asset(
        partial,
        build_aimodel_metadata(
            "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"
        ),
    )
    partial.rename(output_asset)
    return output_asset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--mmap-directory", type=Path, required=True)
    parser.add_argument("--quantized-directory", type=Path, required=True)
    parser.add_argument("--output-asset", type=Path, required=True)
    args = parser.parse_args(argv)
    print(export_mtp_sidecar(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
