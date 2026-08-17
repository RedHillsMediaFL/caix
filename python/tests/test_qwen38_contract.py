from __future__ import annotations

import unittest
from pathlib import Path

from qwen38_coreai.contract import (
    Qwen38Architecture,
    Qwen38ContractError,
    build_metadata,
)
from qwen38_coreai.export import (
    Qwen38ExportPlan,
    build_coreai_quantization_config,
    build_mtp_quantization_config,
)


def source_config() -> dict:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
        },
    }


class Qwen38ContractTests(unittest.TestCase):
    def test_validates_exact_hybrid_geometry_and_four_states(self) -> None:
        architecture = Qwen38Architecture.from_config(source_config())

        self.assertEqual(architecture.full_attention_layers, 16)
        self.assertEqual(architecture.linear_attention_layers, 48)
        self.assertEqual(architecture.kv_cache_shape, (16, 1, 4, -1, 256))
        self.assertEqual(architecture.conv_state_shape, (48, 1, 10240, 3))
        self.assertEqual(architecture.recurrent_state_shape, (48, 1, 48, 128, 128))
        self.assertEqual(architecture.key_value_cache_bytes, 16 * 1024**3)

    def test_rejects_non_qwen38_geometry_before_export(self) -> None:
        config = source_config()
        config["text_config"]["num_hidden_layers"] = 63

        with self.assertRaisesRegex(Qwen38ContractError, "num_hidden_layers"):
            Qwen38Architecture.from_config(config)

    def test_accepts_canonical_nested_rope_theta_from_the_local_oracle(self) -> None:
        config = source_config()
        text = config["text_config"]
        text.pop("rope_theta")
        text["rope_parameters"] = {"type": "default", "rope_theta": 10000000}

        architecture = Qwen38Architecture.from_config(config)

        self.assertEqual(architecture.rope_theta, 10000000)

    def test_metadata_declares_native_functions_states_and_thinking_default(self) -> None:
        metadata = build_metadata(
            name="qwen3.8-27b-caix-r1",
            architecture=Qwen38Architecture.from_config(source_config()),
        )

        self.assertEqual(metadata["language"]["max_context_length"], 262144)
        self.assertEqual(
            metadata["language"]["function_map"],
            {"main": ["main"]},
        )
        self.assertEqual(metadata["assets"], {"main": "model.aimodel"})
        self.assertEqual(metadata["states"]["keyCache"], "kv_cache")
        self.assertEqual(metadata["states"]["convState"], "fixed")
        self.assertEqual(metadata["qwen3_8"]["thinking_default"], True)
        self.assertNotIn("mtp", metadata["qwen3_8"])

    def test_export_plan_requires_160_gib_scratch_without_deleting_anything(self) -> None:
        plan = Qwen38ExportPlan(
            source_model="/models/qwen3.8-27b",
            output_bundle="/exports/qwen3.8-27b-caix-r1",
            scratch_root="/scratch/qwen38",
            architecture=Qwen38Architecture.from_config(source_config()),
        )

        self.assertEqual(plan.minimum_scratch_bytes, 160 * 1024**3)
        with self.assertRaisesRegex(Qwen38ContractError, "160 GiB"):
            plan.require_scratch_capacity(159 * 1024**3)
        plan.require_scratch_capacity(160 * 1024**3)
        self.assertEqual(plan.metadata["name"], "qwen3.8-27b-caix-r1")

    def test_first_party_author_exposes_explicit_four_state_inputs_not_packed_kv(self) -> None:
        author = Path(__file__).parents[1] / "qwen38_coreai" / "model.py"
        source = author.read_text()

        self.assertIn('"convState"', source)
        self.assertIn('"recurrentState"', source)
        self.assertIn("conv_state", source)
        self.assertIn("recurrent_state", source)
        self.assertNotIn("pack the recurrent state into", source.lower())

    def test_target_exports_post_norm_hidden_for_native_mtp_seeding(self) -> None:
        from qwen38_coreai.model import Qwen3_5ForCausalLM

        self.assertEqual(
            Qwen3_5ForCausalLM.export_output_names()["main"],
            ("logits", "hidden_states"),
        )

    def test_coreai_quantization_preserves_source_int4_and_critical_int8_policy(self) -> None:
        config = source_config()
        config["quantization"] = {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            "language_model.model.embed_tokens": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            },
            "language_model.model.layers.56.mlp.down_proj": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            },
        }

        quantization = build_coreai_quantization_config(config)

        global_weight = quantization["global_config"]["op_state_spec"]["weight"]
        self.assertEqual(global_weight["dtype"], "int4")
        self.assertEqual(global_weight["qscheme"], "asymmetric")
        self.assertEqual(global_weight["granularity"]["block_size"], 32)
        module_configs = quantization["module_name_configs"]
        self.assertIn("model.embed_tokens", module_configs)
        self.assertIn("model.layers.56.mlp.down_proj", module_configs)
        int8_weight = module_configs["model.embed_tokens"]["module_state_spec"]["weight"]
        self.assertEqual(int8_weight["dtype"], "int8")
        self.assertEqual(int8_weight["granularity"]["block_size"], 64)

    def test_mtp_quantization_only_compresses_shared_embedding_and_head(self) -> None:
        quantization = build_mtp_quantization_config()

        self.assertIsNone(quantization["global_config"])
        self.assertEqual(
            set(quantization["module_name_configs"]),
            {"embed_tokens", "lm_head"},
        )
        for module in quantization["module_name_configs"].values():
            weight = module["module_state_spec"]["weight"]
            self.assertEqual(weight["dtype"], "int8")
            self.assertEqual(weight["granularity"]["block_size"], 64)


if __name__ == "__main__":
    unittest.main()
