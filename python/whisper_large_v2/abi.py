"""Stable tensor and entrypoint ABI for native Whisper large-v2 serving."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class TensorABI:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class NativeWhisperABI:
    schema: str
    strategy: str
    call_order: tuple[str, ...]
    entrypoints: Mapping[str, str]
    tensors: Mapping[str, TensorABI]
    load_cross_kv_calls_per_utterance: int
    cross_projection_calls_per_utterance: int
    cross_projection_calls_per_decode_step: int
    persists_request_content: bool
    allows_temporary_audio_files: bool

    @classmethod
    def large_v2(cls) -> NativeWhisperABI:
        """Return the immutable ABI selected from the converter/runtime evidence."""
        return cls(
            schema="caix.whisper-split.v1",
            strategy="explicit_cross_kv_bridge",
            call_order=("encode", "load_cross_kv", "decode_step*"),
            entrypoints=MappingProxyType(
                {
                    "encoder": "encode",
                    "decoder_load": "load_cross_kv",
                    "decoder_step": "decode_step",
                }
            ),
            tensors=MappingProxyType(
                {
                    "input_features": TensorABI("float16", (1, 80, 3000)),
                    "cross_key_cache": TensorABI("float16", (32, 1, 20, 1500, 64)),
                    "cross_value_cache": TensorABI("float16", (32, 1, 20, 1500, 64)),
                    "self_key_cache": TensorABI("float16", (32, 1, 20, 448, 64)),
                    "self_value_cache": TensorABI("float16", (32, 1, 20, 448, 64)),
                    "token_id": TensorABI("int32", (1, 1)),
                    "position": TensorABI("int32", (1,)),
                    "cross_ready": TensorABI("int32", (1,)),
                    "logits": TensorABI("float16", (1, 1, 51_865)),
                }
            ),
            load_cross_kv_calls_per_utterance=1,
            cross_projection_calls_per_utterance=1,
            cross_projection_calls_per_decode_step=0,
            persists_request_content=False,
            allows_temporary_audio_files=False,
        )
