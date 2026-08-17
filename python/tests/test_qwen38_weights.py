from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from qwen38_coreai.weights import (
    Qwen38Checkpoint,
    QuantizationSpec,
    convert_mlx_norm_scale,
    dequantize_affine_packed,
    quantization_spec_for_module,
    unpack_uint32,
)


class Qwen38PackedWeightTests(unittest.TestCase):
    def test_converts_only_effective_mlx_norm_scales_to_plus_one_parameters(self) -> None:
        effective_scale = torch.tensor([0.75, 1.0, 2.5], dtype=torch.float16)

        converted = convert_mlx_norm_scale(
            "language_model.model.layers.0.self_attn.q_norm.weight", effective_scale
        )
        gated = convert_mlx_norm_scale(
            "language_model.model.layers.0.linear_attn.norm.weight", effective_scale
        )

        torch.testing.assert_close(
            converted, torch.tensor([-0.25, 0.0, 1.5], dtype=torch.float16)
        )
        self.assertIs(gated, effective_scale)

    def test_unpacks_mlx_int4_words_low_nibble_first(self) -> None:
        packed = torch.tensor([[0xCCDD_EEFF, 0x8899_AABB]], dtype=torch.uint32)

        unpacked = unpack_uint32(packed, bits=4)

        self.assertEqual(
            unpacked.tolist(),
            [[15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8]],
        )

    def test_unpacks_mlx_int8_words_low_byte_first(self) -> None:
        packed = torch.tensor([[0x0302_0100, 0xFF80_7F04]], dtype=torch.uint32)

        unpacked = unpack_uint32(packed, bits=8)

        self.assertEqual(unpacked.tolist(), [[0, 1, 2, 3, 4, 127, 128, 255]])

    def test_dequantizes_each_group_with_its_own_scale_and_bias(self) -> None:
        packed = torch.tensor([[0x7654_3210, 0xFEDC_BA98]], dtype=torch.uint32)
        scales = torch.tensor([[2.0, -1.0]], dtype=torch.float16)
        biases = torch.tensor([[10.0, 100.0]], dtype=torch.float16)

        actual = dequantize_affine_packed(
            packed,
            scales,
            biases,
            bits=4,
            group_size=8,
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            actual,
            torch.tensor(
                [[10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
                  92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0, 85.0]]
            ),
        )

    def test_rejects_scale_shape_that_does_not_cover_the_unpacked_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "group count"):
            dequantize_affine_packed(
                torch.tensor([[0]], dtype=torch.uint32),
                torch.ones((1, 2), dtype=torch.float16),
                torch.zeros((1, 2), dtype=torch.float16),
                bits=4,
                group_size=32,
            )

    def test_uses_exact_module_override_then_checkpoint_default(self) -> None:
        config = {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            "language_model.model.embed_tokens": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            },
        }

        self.assertEqual(
            quantization_spec_for_module(config, "language_model.model.embed_tokens"),
            QuantizationSpec(bits=8, group_size=64, mode="affine"),
        )
        self.assertEqual(
            quantization_spec_for_module(config, "language_model.model.layers.0.mlp.up_proj"),
            QuantizationSpec(bits=4, group_size=32, mode="affine"),
        )

    def test_rejects_non_affine_or_unsupported_bit_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "affine"):
            quantization_spec_for_module(
                {"bits": 4, "group_size": 32, "mode": "mxfp4"}, "model.x"
            )
        with self.assertRaisesRegex(ValueError, "4-bit or 8-bit"):
            quantization_spec_for_module(
                {"bits": 3, "group_size": 32, "mode": "affine"}, "model.x"
            )

    def test_streams_and_dequantizes_one_layer_without_loading_other_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layer_module = "language_model.model.layers.0.mlp.up_proj"
            shared_key = "language_model.model.norm.weight"
            mtp_key = "language_model.mtp.norm.weight"
            layer_tensors = {
                layer_module + ".weight": torch.tensor(
                    [[0x7654_3210, 0xFEDC_BA98, 0x7654_3210, 0xFEDC_BA98]],
                    dtype=torch.uint32,
                ),
                layer_module + ".scales": torch.tensor([[2.0]], dtype=torch.float16),
                layer_module + ".biases": torch.tensor([[1.0]], dtype=torch.float16),
            }
            save_file(layer_tensors, root / "layer.safetensors")
            save_file(
                {
                    shared_key: torch.tensor([3.0], dtype=torch.float32),
                    mtp_key: torch.tensor([4.0], dtype=torch.float32),
                },
                root / "shared.safetensors",
            )
            weight_map = {
                key: "layer.safetensors" for key in layer_tensors
            } | {shared_key: "shared.safetensors", mtp_key: "shared.safetensors"}
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map})
            )
            (root / "config.json").write_text(
                json.dumps({"quantization": {"bits": 4, "group_size": 32, "mode": "affine"}})
            )

            checkpoint = Qwen38Checkpoint(root)
            layer = checkpoint.load_layer_state_dict(0, dtype=torch.float16)
            shared = checkpoint.load_shared_state_dict(dtype=torch.float16)
            mtp = checkpoint.load_mtp_state_dict(dtype=torch.float16)

            self.assertEqual(set(layer), {layer_module + ".weight"})
            self.assertEqual(layer[layer_module + ".weight"].shape, (1, 32))
            self.assertEqual(layer[layer_module + ".weight"].dtype, torch.float16)
            self.assertEqual(shared[shared_key].dtype, torch.float16)
            self.assertEqual(shared[shared_key].item(), 2.0)
            self.assertNotIn(mtp_key, shared)
            self.assertEqual(set(mtp), {mtp_key})
            self.assertEqual(mtp[mtp_key].item(), 3.0)

    def test_rejects_quantized_weight_missing_its_scale_or_bias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            key = "language_model.model.layers.0.mlp.up_proj.weight"
            save_file({key: torch.zeros((1, 4), dtype=torch.uint32)}, root / "layer.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: "layer.safetensors"}})
            )
            (root / "config.json").write_text(
                json.dumps({"quantization": {"bits": 4, "group_size": 32, "mode": "affine"}})
            )

            with self.assertRaisesRegex(ValueError, "missing companion"):
                Qwen38Checkpoint(root).load_layer_state_dict(0)

    def test_author_remaps_the_local_checkpoint_language_model_prefix(self) -> None:
        from qwen38_coreai.model import Qwen3_5ForCausalLM

        author = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        state_dict = {
            "language_model.model.layers.0.linear_attn.conv1d.weight": torch.ones((2, 4, 1)),
            "language_model.lm_head.weight": torch.ones((2, 2)),
        }

        author._mutate_state_dict(state_dict)

        self.assertEqual(
            set(state_dict),
            {
                "model.layers.0.linear_attn.conv1d.weight",
                "lm_head.weight",
            },
        )
        self.assertEqual(
            state_dict["model.layers.0.linear_attn.conv1d.weight"].shape,
            (2, 1, 4),
        )

    def test_rehydrates_parameters_from_an_existing_mmap_shard(self) -> None:
        from qwen38_coreai.model import Qwen3_5ForCausalLM

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "linear.safetensors"
            save_file(
                {
                    "weight": torch.arange(6, dtype=torch.float16).reshape(2, 3),
                    "bias": torch.tensor([7.0, 8.0], dtype=torch.float16),
                },
                path,
            )
            linear = nn.Linear(3, 2, device="meta", dtype=torch.float16)

            Qwen3_5ForCausalLM._assign_mmap_safetensors(linear, path)

            self.assertFalse(linear.weight.is_meta)
            torch.testing.assert_close(
                linear.weight, torch.arange(6, dtype=torch.float16).reshape(2, 3)
            )
            torch.testing.assert_close(
                linear.bias, torch.tensor([7.0, 8.0], dtype=torch.float16)
            )


if __name__ == "__main__":
    unittest.main()
