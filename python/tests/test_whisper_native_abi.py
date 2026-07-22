from __future__ import annotations

import importlib


def test_large_v2_abi_uses_the_explicit_load_once_cross_kv_bridge() -> None:
    abi_module = importlib.import_module("whisper_large_v2.abi")

    abi = abi_module.NativeWhisperABI.large_v2()

    assert abi.schema == "caix.whisper-split.v1"
    assert abi.strategy == "explicit_cross_kv_bridge"
    assert abi.call_order == ("encode", "load_cross_kv", "decode_step*")
    assert abi.entrypoints == {
        "encoder": "encode",
        "decoder_load": "load_cross_kv",
        "decoder_step": "decode_step",
    }
    assert abi.tensors == {
        "input_features": abi_module.TensorABI("float16", (1, 80, 3000)),
        "cross_key_cache": abi_module.TensorABI("float16", (32, 1, 20, 1500, 64)),
        "cross_value_cache": abi_module.TensorABI("float16", (32, 1, 20, 1500, 64)),
        "self_key_cache": abi_module.TensorABI("float16", (32, 1, 20, 448, 64)),
        "self_value_cache": abi_module.TensorABI("float16", (32, 1, 20, 448, 64)),
        "token_id": abi_module.TensorABI("int32", (1, 1)),
        "position": abi_module.TensorABI("int32", (1,)),
        "cross_ready": abi_module.TensorABI("int32", (1,)),
        "logits": abi_module.TensorABI("float16", (1, 1, 51865)),
    }
    assert abi.load_cross_kv_calls_per_utterance == 1
    assert abi.cross_projection_calls_per_utterance == 1
    assert abi.cross_projection_calls_per_decode_step == 0
    assert abi.persists_request_content is False
    assert abi.allows_temporary_audio_files is False
