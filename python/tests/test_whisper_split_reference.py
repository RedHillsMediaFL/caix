from __future__ import annotations

import importlib

import pytest
import torch


def _fixture() -> tuple[object, torch.Tensor, torch.Tensor]:
    reference = importlib.import_module("whisper_large_v2.reference")
    torch.manual_seed(7)
    model = reference.TinyWhisperReference(
        reference.TinyWhisperConfig(
            d_model=16,
            heads=4,
            encoder_layers=2,
            decoder_layers=2,
            feed_forward=32,
            vocabulary_size=29,
            max_source_positions=6,
            max_decoder_positions=448,
        )
    ).eval()
    features = torch.randn(1, 6, 16)
    token_ids = torch.tensor([[2, 7, 3, 11, 5]], dtype=torch.long)
    return model, features, token_ids


def test_stateful_one_token_decode_matches_monolithic_at_every_position() -> None:
    model, features, token_ids = _fixture()

    with torch.no_grad():
        expected = model(features, token_ids)
        payload = model.encode_cross_kv(features)
        state = model.new_decoder_state(features)
        assert state.position.dtype == torch.int32
        assert state.position.tolist() == [0]
        assert state.cross_ready.dtype == torch.int32
        assert state.cross_ready.tolist() == [0]
        assert state.self_keys.shape == (2, 1, 4, 448, 4)
        assert state.self_values.shape == (2, 1, 4, 448, 4)
        model.load_cross_kv(payload, state)
        assert state.cross_ready.tolist() == [1]
        steps = []
        for position in range(token_ids.shape[1]):
            step = model.decode_step(token_ids[:, position : position + 1], state)
            steps.append(step)
            assert state.position.tolist() == [position + 1]
            assert state.self_keys.shape == (2, 1, 4, 448, 4)
            assert state.self_values.shape == (2, 1, 4, 448, 4)
            assert torch.count_nonzero(state.self_keys[..., : position + 1, :]) > 0
            assert torch.count_nonzero(state.self_values[..., : position + 1, :]) > 0
            assert torch.count_nonzero(state.self_keys[..., position + 1 :, :]) == 0
            assert torch.count_nonzero(state.self_values[..., position + 1 :, :]) == 0

    actual = torch.cat(steps, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_cross_key_value_projections_run_once_per_utterance_not_per_token() -> None:
    model, features, token_ids = _fixture()
    model.reset_cross_projection_counts()

    with torch.no_grad():
        payload = model.encode_cross_kv(features)
        after_precompute = model.cross_projection_counts
        state = model.new_decoder_state(features)
        model.load_cross_kv(payload, state)
        for position in range(token_ids.shape[1]):
            model.decode_step(token_ids[:, position : position + 1], state)

    assert after_precompute == ((1, 1), (1, 1))
    assert model.cross_projection_counts == after_precompute


def test_cross_kv_must_load_exactly_once_before_decode() -> None:
    reference = importlib.import_module("whisper_large_v2.reference")
    model, features, token_ids = _fixture()
    payload = model.encode_cross_kv(features)
    state = model.new_decoder_state(features)

    with pytest.raises(reference.DecoderStateError, match="load_cross_kv"):
        model.decode_step(token_ids[:, :1], state)

    model.load_cross_kv(payload, state)

    with pytest.raises(reference.DecoderStateError, match="exactly once"):
        model.load_cross_kv(payload, state)


def test_reset_zeroes_every_cache_and_allows_a_new_utterance() -> None:
    model, features, token_ids = _fixture()
    payload = model.encode_cross_kv(features)
    state = model.new_decoder_state(features)
    model.load_cross_kv(payload, state)

    with torch.no_grad():
        first = model.decode_step(token_ids[:, :1], state)
        model.decode_step(token_ids[:, 1:2], state)
        model.reset_decoder_state(state)

    assert state.position.tolist() == [0]
    assert state.cross_ready.tolist() == [0]
    assert torch.count_nonzero(state.cross_keys) == 0
    assert torch.count_nonzero(state.cross_values) == 0
    assert torch.count_nonzero(state.self_keys) == 0
    assert torch.count_nonzero(state.self_values) == 0

    with torch.no_grad():
        model.load_cross_kv(payload, state)
        after_reset = model.decode_step(token_ids[:, :1], state)

    torch.testing.assert_close(after_reset, first)


def test_last_cache_slot_is_indexed_then_overflow_is_rejected_without_mutation() -> None:
    reference = importlib.import_module("whisper_large_v2.reference")
    model, features, token_ids = _fixture()
    payload = model.encode_cross_kv(features)
    state = model.new_decoder_state(features)
    model.load_cross_kv(payload, state)
    state.position.fill_(447)

    with torch.no_grad():
        model.decode_step(token_ids[:, :1], state)

    assert state.position.tolist() == [448]
    assert torch.count_nonzero(state.self_keys[..., 447:448, :]) > 0
    assert torch.count_nonzero(state.self_values[..., 447:448, :]) > 0
    keys_at_capacity = state.self_keys.clone()
    values_at_capacity = state.self_values.clone()

    with pytest.raises(reference.DecoderStateError, match="448-token capacity"):
        model.decode_step(token_ids[:, 1:2], state)

    assert state.position.tolist() == [448]
    torch.testing.assert_close(state.self_keys, keys_at_capacity)
    torch.testing.assert_close(state.self_values, values_at_capacity)
