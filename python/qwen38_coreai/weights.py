"""Streaming-safe decoding of MLX affine-packed Qwen3.8 checkpoint weights."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open


PLUS_ONE_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".model.norm.weight",
    ".mtp.norm.weight",
)


def convert_mlx_norm_scale(key: str, value: torch.Tensor) -> torch.Tensor:
    """Convert an MLX RMSNorm scale to CoreAI's ``1 + weight`` parameter.

    MLX-VLM stores Qwen3.5's ordinary RMSNorm tensors as the effective scale,
    after applying the Hugging Face ``+1`` convention during sanitization.
    CoreAI's RMSNormPlusOne applies that offset in the operator, so its stored
    parameter must be the delta.  The gated delta-net norm is deliberately not
    in this allowlist because that operator uses a plain scale.
    """

    if value.is_floating_point() and key.endswith(PLUS_ONE_NORM_SUFFIXES):
        return value - 1.0
    return value


@dataclass(frozen=True)
class QuantizationSpec:
    """The only source packing modes accepted by the Qwen3.8 exporter."""

    bits: int
    group_size: int
    mode: str = "affine"

    def validate(self) -> None:
        if self.mode != "affine":
            raise ValueError(f"Qwen3.8 source quantization must be affine, got {self.mode!r}")
        if self.bits not in (4, 8):
            raise ValueError(
                f"Qwen3.8 source weights must use 4-bit or 8-bit packing, got {self.bits}"
            )
        if self.group_size <= 0 or self.group_size % (32 // self.bits) != 0:
            raise ValueError(
                f"group_size must be positive and divisible by {32 // self.bits}, "
                f"got {self.group_size}"
            )


def quantization_spec_for_module(
    quantization_config: Mapping[str, Any], module_name: str
) -> QuantizationSpec:
    """Resolve an exact MLX module override, falling back to checkpoint defaults."""

    override = quantization_config.get(module_name, {})
    if override is None:
        override = {}
    if not isinstance(override, Mapping):
        raise ValueError(f"quantization override for {module_name!r} must be an object")
    spec = QuantizationSpec(
        bits=int(override.get("bits", quantization_config.get("bits", 0))),
        group_size=int(override.get("group_size", quantization_config.get("group_size", 0))),
        mode=str(override.get("mode", quantization_config.get("mode", "affine"))),
    )
    spec.validate()
    return spec


def unpack_uint32(packed: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Unpack MLX words along the last axis, least-significant field first."""

    if packed.dtype != torch.uint32:
        raise TypeError(f"packed weight must be torch.uint32, got {packed.dtype}")
    if bits not in (4, 8):
        raise ValueError(f"packed weight must be 4-bit or 8-bit, got {bits}")

    values_per_word = 32 // bits
    shifts = torch.arange(values_per_word, dtype=torch.int64, device=packed.device) * bits
    fields = torch.bitwise_right_shift(packed.to(torch.int64).unsqueeze(-1), shifts)
    fields = torch.bitwise_and(fields, (1 << bits) - 1)
    return fields.reshape(*packed.shape[:-1], packed.shape[-1] * values_per_word)


def dequantize_affine_packed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct an MLX affine tensor as ``code * scale + bias``.

    The operation materializes exactly one destination tensor plus temporary unpacked codes,
    which lets the checkpoint loader process one module (and one layer) at a time.
    """

    spec = QuantizationSpec(bits=bits, group_size=group_size)
    spec.validate()
    if scales.shape != biases.shape:
        raise ValueError(
            f"scale and bias shapes must match, got {tuple(scales.shape)} and {tuple(biases.shape)}"
        )
    if packed.ndim != scales.ndim or packed.shape[:-1] != scales.shape[:-1]:
        raise ValueError(
            "packed weights, scales, and biases must have equal rank and leading dimensions"
        )

    unpacked = unpack_uint32(packed, bits=bits)
    expected_values = scales.shape[-1] * group_size
    if unpacked.shape[-1] != expected_values:
        raise ValueError(
            "quantization group count does not cover the unpacked weight axis: "
            f"{scales.shape[-1]} groups * {group_size} != {unpacked.shape[-1]} values"
        )

    expanded_scales = scales.to(torch.float32).repeat_interleave(group_size, dim=-1)
    expanded_biases = biases.to(torch.float32).repeat_interleave(group_size, dim=-1)
    return (unpacked.to(torch.float32) * expanded_scales + expanded_biases).to(dtype)


class Qwen38Checkpoint:
    """Strict, shard-aware reader for the local MLX Qwen3.8 checkpoint.

    Each call opens only the shard files containing the requested layer/shared tensors. Quantized
    triplets collapse to one floating-point ``.weight`` entry, so MLX-only ``.scales`` and
    ``.biases`` never leak into the PyTorch model state dict.
    """

    index_filename = "model.safetensors.index.json"

    def __init__(self, model_directory: str | Path) -> None:
        self.model_directory = Path(model_directory)
        config_path = self.model_directory / "config.json"
        index_path = self.model_directory / self.index_filename
        try:
            config = json.loads(config_path.read_text())
            index = json.loads(index_path.read_text())
        except FileNotFoundError as error:
            raise ValueError(f"Qwen3.8 checkpoint file is missing: {error.filename}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Qwen3.8 checkpoint metadata is invalid JSON: {error}") from error

        quantization = config.get("quantization_config", config.get("quantization"))
        if not isinstance(quantization, Mapping):
            raise ValueError("Qwen3.8 config must contain quantization or quantization_config")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("Qwen3.8 safetensors index must contain a non-empty weight_map")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
            raise ValueError("Qwen3.8 weight_map keys and shard names must be strings")

        self.quantization = quantization
        self.weight_map: dict[str, str] = dict(weight_map)
        for shard_name in set(self.weight_map.values()):
            if not (self.model_directory / shard_name).is_file():
                raise ValueError(f"Qwen3.8 checkpoint shard is missing: {shard_name}")

    def _load_raw(self, keys: set[str]) -> dict[str, torch.Tensor]:
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            by_shard[self.weight_map[key]].append(key)

        tensors: dict[str, torch.Tensor] = {}
        for shard_name, shard_keys in by_shard.items():
            with safe_open(
                self.model_directory / shard_name,
                framework="pt",
                device="cpu",
            ) as shard:
                for key in shard_keys:
                    tensors[key] = shard.get_tensor(key)
        return tensors

    def _load_state_dict(self, selected_keys: set[str], dtype: torch.dtype) -> dict[str, torch.Tensor]:
        keys_to_read = set(selected_keys)
        logical_keys: list[str] = []
        for key in sorted(selected_keys):
            if key.endswith((".scales", ".biases")):
                continue
            logical_keys.append(key)
            if key.endswith(".weight"):
                module_name = key.removesuffix(".weight")
                scales_key = module_name + ".scales"
                biases_key = module_name + ".biases"
                companions = (scales_key, biases_key)
                present = tuple(companion in self.weight_map for companion in companions)
                if any(present) and not all(present):
                    missing = [name for name in companions if name not in self.weight_map]
                    raise ValueError(
                        f"quantized weight {key!r} is missing companion tensor(s): {missing}"
                    )
                if all(present):
                    keys_to_read.update(companions)

        raw = self._load_raw(keys_to_read)
        decoded: dict[str, torch.Tensor] = {}
        for key in logical_keys:
            value = raw[key]
            if value.dtype == torch.uint32:
                module_name = key.removesuffix(".weight")
                scales_key = module_name + ".scales"
                biases_key = module_name + ".biases"
                if scales_key not in raw or biases_key not in raw:
                    raise ValueError(
                        f"quantized weight {key!r} is missing companion scale or bias tensor"
                    )
                spec = quantization_spec_for_module(self.quantization, module_name)
                value = dequantize_affine_packed(
                    value,
                    raw[scales_key],
                    raw[biases_key],
                    bits=spec.bits,
                    group_size=spec.group_size,
                    dtype=dtype,
                )
            elif value.is_floating_point():
                value = value.to(dtype)
            value = convert_mlx_norm_scale(key, value)
            decoded[key] = value
        return decoded

    def load_layer_state_dict(
        self, layer_index: int, *, dtype: torch.dtype = torch.float16
    ) -> dict[str, torch.Tensor]:
        if layer_index < 0:
            raise ValueError(f"layer_index must be non-negative, got {layer_index}")
        prefix = f"language_model.model.layers.{layer_index}."
        keys = {key for key in self.weight_map if key.startswith(prefix)}
        if not keys:
            raise ValueError(f"Qwen3.8 checkpoint contains no tensors for layer {layer_index}")
        return self._load_state_dict(keys, dtype)

    def load_shared_state_dict(
        self, *, dtype: torch.dtype = torch.float16
    ) -> dict[str, torch.Tensor]:
        layer_prefix = "language_model.model.layers."
        mtp_prefix = "language_model.mtp."
        keys = {
            key
            for key in self.weight_map
            if not key.startswith(layer_prefix) and not key.startswith(mtp_prefix)
        }
        return self._load_state_dict(keys, dtype)

    def load_mtp_state_dict(
        self, *, dtype: torch.dtype = torch.float16
    ) -> dict[str, torch.Tensor]:
        keys = {key for key in self.weight_map if key.startswith("language_model.mtp.")}
        if not keys:
            raise ValueError("Qwen3.8 checkpoint contains no embedded MTP tensors")
        return self._load_state_dict(keys, dtype)
