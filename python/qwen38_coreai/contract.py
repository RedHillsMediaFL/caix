"""Exact structural contract for caix's native Qwen3.8-27B Core AI bundle.

This module intentionally has no Core AI or torch import. It validates the checkpoint geometry
and produces the runtime metadata before an expensive model load/export is allowed to begin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


GIB = 1024**3
SOURCE_MODEL_ID = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"
NATIVE_STATE_NAMES = ("keyCache", "valueCache", "convState", "recurrentState")


class Qwen38ContractError(ValueError):
    """The source is not the supported Qwen3.8-27B hybrid architecture."""


def _require_equal(config: Mapping[str, Any], key: str, expected: Any) -> Any:
    actual = config.get(key)
    if actual != expected:
        raise Qwen38ContractError(f"{key} must be {expected!r}, got {actual!r}")
    return actual


def _require_rope_theta(config: Mapping[str, Any]) -> int:
    """Accept Qwen's canonical nested RoPE metadata without weakening the value contract."""
    direct = config.get("rope_theta")
    parameters = config.get("rope_parameters")
    nested = parameters.get("rope_theta") if isinstance(parameters, Mapping) else None
    if direct is not None and nested is not None and direct != nested:
        raise Qwen38ContractError(
            f"rope_theta disagrees with rope_parameters.rope_theta: {direct!r} vs {nested!r}"
        )
    actual = direct if direct is not None else nested
    if actual != 10000000:
        raise Qwen38ContractError(f"rope_theta must be 10000000, got {actual!r}")
    return actual


@dataclass(frozen=True)
class Qwen38Architecture:
    """Verified Qwen3.8 text-decoder geometry and persistent-state layout."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_context_length: int
    partial_rotary_factor: float
    rope_theta: int
    conv_kernel: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    layer_types: tuple[str, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Qwen38Architecture":
        _require_equal(config, "model_type", "qwen3_5")
        text = config.get("text_config")
        if not isinstance(text, Mapping):
            raise Qwen38ContractError("text_config must be an object")
        _require_equal(text, "model_type", "qwen3_5_text")

        values = {
            "vocab_size": _require_equal(text, "vocab_size", 248320),
            "hidden_size": _require_equal(text, "hidden_size", 5120),
            "intermediate_size": _require_equal(text, "intermediate_size", 17408),
            "num_hidden_layers": _require_equal(text, "num_hidden_layers", 64),
            "num_attention_heads": _require_equal(text, "num_attention_heads", 24),
            "num_key_value_heads": _require_equal(text, "num_key_value_heads", 4),
            "head_dim": _require_equal(text, "head_dim", 256),
            "max_context_length": _require_equal(text, "max_position_embeddings", 262144),
            "partial_rotary_factor": _require_equal(text, "partial_rotary_factor", 0.25),
            "rope_theta": _require_rope_theta(text),
            "conv_kernel": _require_equal(text, "linear_conv_kernel_dim", 4),
            "linear_num_key_heads": _require_equal(text, "linear_num_key_heads", 16),
            "linear_num_value_heads": _require_equal(text, "linear_num_value_heads", 48),
            "linear_key_head_dim": _require_equal(text, "linear_key_head_dim", 128),
            "linear_value_head_dim": _require_equal(text, "linear_value_head_dim", 128),
        }
        raw_layer_types = text.get("layer_types")
        if not isinstance(raw_layer_types, list):
            raise Qwen38ContractError("layer_types must be a 64-element list")
        layer_types = tuple(raw_layer_types)
        expected_layers = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 16
        if layer_types != expected_layers:
            raise Qwen38ContractError(
                "layer_types must be sixteen repetitions of 3 linear_attention + 1 full_attention"
            )
        return cls(layer_types=layer_types, **values)

    @property
    def full_attention_layers(self) -> int:
        return self.layer_types.count("full_attention")

    @property
    def linear_attention_layers(self) -> int:
        return self.layer_types.count("linear_attention")

    @property
    def conv_dimension(self) -> int:
        return 2 * self.linear_num_key_heads * self.linear_key_head_dim + self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def kv_cache_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.full_attention_layers,
            1,
            self.num_key_value_heads,
            -1,
            self.head_dim,
        )

    @property
    def conv_state_shape(self) -> tuple[int, int, int, int]:
        return (
            self.linear_attention_layers,
            1,
            self.conv_dimension,
            self.conv_kernel - 1,
        )

    @property
    def recurrent_state_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.linear_attention_layers,
            1,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
        )

    @property
    def key_value_cache_bytes(self) -> int:
        return (
            self.full_attention_layers
            * 2
            * self.num_key_value_heads
            * self.max_context_length
            * self.head_dim
            * 2
        )


def build_metadata(*, name: str, architecture: Qwen38Architecture) -> dict[str, Any]:
    """Return the versioned bundle metadata consumed by the native caix route.

    The `states` block is also understood by current CoreAILanguageModels. It prevents heuristic
    classification from treating Qwen's fixed conv/recurrent buffers as truncatable cache state.
    """

    return {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": name,
        "assets": {
            "main": "model.aimodel",
        },
        "language": {
            "tokenizer": SOURCE_MODEL_ID,
            "vocab_size": architecture.vocab_size,
            "max_context_length": architecture.max_context_length,
            "embedded_tokenizer": True,
            "function_map": {
                "main": ["main"],
            },
        },
        "states": {
            "keyCache": "kv_cache",
            "valueCache": "kv_cache",
            "convState": "fixed",
            "recurrentState": "fixed",
        },
        "qwen3_8": {
            "state_layout": {
                "names": list(NATIVE_STATE_NAMES),
                "full_attention_layers": architecture.full_attention_layers,
                "kv_heads": architecture.num_key_value_heads,
                "head_dimension": architecture.head_dim,
                "conv_dtype": "float16",
                "recurrent_dtype": "float32",
            },
            "thinking_default": True,
        },
        "source": {"hf_model_id": SOURCE_MODEL_ID},
    }
