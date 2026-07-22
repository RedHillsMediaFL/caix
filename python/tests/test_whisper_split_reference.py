from __future__ import annotations

import importlib

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
            max_decoder_positions=8,
        )
    ).eval()
    features = torch.randn(1, 6, 16)
    token_ids = torch.tensor([[2, 7, 3, 11, 5]], dtype=torch.long)
    return model, features, token_ids


def test_stateful_one_token_decode_matches_monolithic_at_every_position() -> None:
    model, features, token_ids = _fixture()

    with torch.no_grad():
        expected = model(features, token_ids)
        state = model.begin_split(features)
        steps = []
        for position in range(token_ids.shape[1]):
            step = model.decode_step(token_ids[:, position : position + 1], state)
            steps.append(step)
            assert state.position == position + 1
            assert all(cache.shape[2] == position + 1 for cache in state.self_keys)
            assert all(cache.shape[2] == position + 1 for cache in state.self_values)

    actual = torch.cat(steps, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_cross_key_value_projections_run_once_per_utterance_not_per_token() -> None:
    model, features, token_ids = _fixture()
    model.reset_cross_projection_counts()

    with torch.no_grad():
        state = model.begin_split(features)
        after_precompute = model.cross_projection_counts
        for position in range(token_ids.shape[1]):
            model.decode_step(token_ids[:, position : position + 1], state)

    assert after_precompute == ((1, 1), (1, 1))
    assert model.cross_projection_counts == after_precompute
