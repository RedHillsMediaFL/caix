from __future__ import annotations

import importlib

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration


def _tiny_model() -> WhisperForConditionalGeneration:
    torch.manual_seed(19)
    config = WhisperConfig(
        vocab_size=31,
        num_mel_bins=4,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=6,
        max_target_positions=448,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_function="gelu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        use_cache=False,
    )
    return WhisperForConditionalGeneration(config).eval()


def _snapshot_state(state: object) -> dict[str, torch.Tensor]:
    return {
        name: getattr(state, name).clone()
        for name in (
            "cross_key_cache",
            "cross_value_cache",
            "self_key_cache",
            "self_value_cache",
            "position",
            "cross_ready",
        )
    }


def _assert_state_exact(state: object, expected: dict[str, torch.Tensor]) -> None:
    for name, value in expected.items():
        assert torch.equal(getattr(state, name), value), name


def test_split_one_token_decoder_matches_transformers_and_owns_exact_weights() -> None:
    export = importlib.import_module("whisper_large_v2.export")
    model = _tiny_model()
    split = export.WhisperSplitModules.from_hf(model)
    features = torch.randn((1, 4, 12), dtype=torch.float32)
    token_ids = torch.tensor([[1, 7, 4, 9]], dtype=torch.int32)

    with torch.no_grad():
        expected = model(
            input_features=features,
            decoder_input_ids=token_ids,
            use_cache=False,
        ).logits
        cross_key_payload, cross_value_payload = split.encode(features)
        state = split.new_state(dtype=torch.float32)
        load_status = split.load_cross_kv(
            cross_key_payload,
            cross_value_payload,
            state.cross_key_cache,
            state.cross_value_cache,
            state.cross_ready,
        )
        steps = []
        for index in range(token_ids.shape[1]):
            logits, decode_status = split.decode_step(
                token_ids[:, index : index + 1],
                state.cross_key_cache,
                state.cross_value_cache,
                state.self_key_cache,
                state.self_value_cache,
                state.position,
                state.cross_ready,
            )
            steps.append(logits)
            assert decode_status.tolist() == [1]

    actual = torch.cat(steps, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert state.cross_key_cache.shape == (1, 1, 4, 6, 4)
    assert state.cross_value_cache.shape == (1, 1, 4, 6, 4)
    assert state.self_key_cache.shape == (1, 1, 4, 448, 4)
    assert state.self_value_cache.shape == (1, 1, 4, 448, 4)
    assert state.position.dtype == torch.int32
    assert state.position.tolist() == [4]
    assert state.cross_ready.dtype == torch.int32
    assert state.cross_ready.tolist() == [1]
    assert load_status.tolist() == [1]

    encode_names = set(dict(split.encode.named_parameters()))
    decode_names = set(dict(split.decode_step.named_parameters()))
    assert any("cross_key_projections" in name for name in encode_names)
    assert any("cross_value_projections" in name for name in encode_names)
    assert not any("cross_key_projections" in name for name in decode_names)
    assert not any("cross_value_projections" in name for name in decode_names)
    assert not any("encoder_attn.k_proj" in name for name in decode_names)
    assert not any("encoder_attn.v_proj" in name for name in decode_names)


def test_second_cross_load_returns_failure_and_preserves_all_state_exactly() -> None:
    export = importlib.import_module("whisper_large_v2.export")
    split = export.WhisperSplitModules.from_hf(_tiny_model())
    features = torch.randn((1, 4, 12), dtype=torch.float32)

    with torch.no_grad():
        keys, values = split.encode(features)
        state = split.new_state(dtype=torch.float32)
        assert split.load_cross_kv(
            keys,
            values,
            state.cross_key_cache,
            state.cross_value_cache,
            state.cross_ready,
        ).tolist() == [1]
        before = _snapshot_state(state)
        status = split.load_cross_kv(
            keys,
            values,
            state.cross_key_cache,
            state.cross_value_cache,
            state.cross_ready,
        )

    assert status.tolist() == [0]
    _assert_state_exact(state, before)


def test_decode_before_load_returns_failure_and_preserves_all_state_exactly() -> None:
    export = importlib.import_module("whisper_large_v2.export")
    split = export.WhisperSplitModules.from_hf(_tiny_model())
    state = split.new_state(dtype=torch.float32)
    token = torch.tensor([[1]], dtype=torch.int32)
    state.cross_key_cache.uniform_(-1, 1)
    state.cross_value_cache.uniform_(-1, 1)
    state.self_key_cache.uniform_(-1, 1)
    state.self_value_cache.uniform_(-1, 1)
    before = _snapshot_state(state)

    with torch.no_grad():
        logits, status = split.decode_step(
            token,
            state.cross_key_cache,
            state.cross_value_cache,
            state.self_key_cache,
            state.self_value_cache,
            state.position,
            state.cross_ready,
        )

    assert status.tolist() == [0]
    assert torch.count_nonzero(logits) == 0
    _assert_state_exact(state, before)


def test_decode_rejects_invalid_readiness_and_positions_without_mutation() -> None:
    export = importlib.import_module("whisper_large_v2.export")
    split = export.WhisperSplitModules.from_hf(_tiny_model())
    features = torch.randn((1, 4, 12), dtype=torch.float32)
    token = torch.tensor([[1]], dtype=torch.int32)

    with torch.no_grad():
        keys, values = split.encode(features)
        for readiness, position in ((-1, 0), (2, 0), (1, -1), (1, 448)):
            state = split.new_state(dtype=torch.float32)
            assert split.load_cross_kv(
                keys,
                values,
                state.cross_key_cache,
                state.cross_value_cache,
                state.cross_ready,
            ).tolist() == [1]
            state.self_key_cache.uniform_(-1, 1)
            state.self_value_cache.uniform_(-1, 1)
            state.cross_ready.fill_(readiness)
            state.position.fill_(position)
            before = _snapshot_state(state)

            logits, status = split.decode_step(
                token,
                state.cross_key_cache,
                state.cross_value_cache,
                state.self_key_cache,
                state.self_value_cache,
                state.position,
                state.cross_ready,
            )

            assert status.tolist() == [0]
            assert torch.count_nonzero(logits) == 0
            _assert_state_exact(state, before)


def test_decoder_rejects_an_incompatible_whisper_shape_contract() -> None:
    export = importlib.import_module("whisper_large_v2.export")
    model = _tiny_model()
    model.config.max_target_positions = 32

    try:
        export.WhisperSplitModules.from_hf(model)
    except export.WhisperExportError as error:
        assert "448" in str(error)
    else:
        raise AssertionError("accepted a non-448 Whisper decoder cache")
