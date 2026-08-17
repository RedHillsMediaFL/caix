from __future__ import annotations

import unittest

import torch

from qwen38_coreai.model import Qwen3_5TextConfig


def tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        full_attention_interval=1,
        layer_types=["full_attention"],
        partial_rotary_factor=1.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000},
        max_position_embeddings=128,
    )


class Qwen38MTPAuthorTests(unittest.TestCase):
    def test_mtp_exports_hidden_conditioning_and_its_own_kv_state(self) -> None:
        from qwen38_coreai.mtp_model import Qwen3_5MTPForCausalLM

        model = Qwen3_5MTPForCausalLM(tiny_config())

        self.assertEqual(
            model.export_input_names()["main"],
            ("input_ids", "hidden_states", "position_ids"),
        )
        self.assertEqual(model.export_state_names()["main"], ("keyCache", "valueCache"))
        self.assertEqual(
            model.export_output_names()["main"], ("logits", "mtp_hidden_states")
        )

    def test_mtp_forward_matches_the_qwen_shift_conditioning_shapes(self) -> None:
        from coreai_models.models.base import TraceSpec
        from qwen38_coreai.mtp_model import Qwen3_5MTPForCausalLM

        config = tiny_config()
        model = Qwen3_5MTPForCausalLM(config).to(dtype=torch.float16).eval()
        refs = model.build_reference_inputs(
            config, torch.float16, TraceSpec(max_context_length=128, cache_seq_len=32)
        )["main"]

        self.assertEqual(tuple(refs["hidden_states"].shape), (1, 16, 16))
        self.assertEqual(tuple(refs["k_cache"].shape), (1, 1, 1, 32, 8))
        logits, hidden = model(**refs)
        self.assertEqual(tuple(logits.shape), (1, 1, 64))
        self.assertEqual(tuple(hidden.shape), (1, 1, 16))

    def test_mtp_remaps_sidecar_and_shared_target_weights(self) -> None:
        from qwen38_coreai.mtp_model import Qwen3_5MTPForCausalLM

        state = {
            "language_model.mtp.fc.weight": torch.ones((2, 4)),
            "language_model.mtp.layers.0.self_attn.q_proj.weight": torch.ones((2, 2)),
            "language_model.model.embed_tokens.weight": torch.ones((2, 2)),
            "language_model.lm_head.weight": torch.ones((2, 2)),
        }
        Qwen3_5MTPForCausalLM._remap_checkpoint_state(state)

        self.assertEqual(
            set(state),
            {"fc.weight", "layer.self_attn.q_proj.weight", "embed_tokens.weight", "lm_head.weight"},
        )


if __name__ == "__main__":
    unittest.main()
