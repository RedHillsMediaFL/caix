"""Native split Whisper modules and CoreAI export support.

The hot decoder owns fixed caller-supplied state. Cross-attention K/V projections
run in ``encode`` exactly once per audio window; ``decode_step`` only projects a
single token and updates one slot in the 448-token self-attention cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from whisper_large_v2.reference import WHISPER_DECODER_CACHE_CAPACITY


def _mutable_slice_update(**arguments: torch.Tensor) -> torch.Tensor:
    from coreai_models.primitives._ops import mutable_slice_update

    return mutable_slice_update(**arguments)


class WhisperExportError(RuntimeError):
    """The source model cannot satisfy the frozen native Whisper ABI."""


@dataclass
class WhisperRuntimeState:
    cross_key_cache: torch.Tensor
    cross_value_cache: torch.Tensor
    self_key_cache: torch.Tensor
    self_value_cache: torch.Tensor
    position: torch.Tensor
    cross_ready: torch.Tensor


class WhisperEncode(nn.Module):
    """Run the acoustic encoder and precompute every decoder cross K/V."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.encoder = model.model.encoder
        decoder_layers = model.model.decoder.layers
        self.cross_key_projections = nn.ModuleList(
            layer.encoder_attn.k_proj for layer in decoder_layers
        )
        self.cross_value_projections = nn.ModuleList(
            layer.encoder_attn.v_proj for layer in decoder_layers
        )
        self.heads = model.config.decoder_attention_heads
        self.head_dim = model.config.d_model // self.heads

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = values.shape
        return values.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_hidden = self.encoder(input_features, return_dict=False)[0]
        cross_keys = torch.stack(
            tuple(
                self._split_heads(projection(encoder_hidden))
                for projection in self.cross_key_projections
            )
        )
        cross_values = torch.stack(
            tuple(
                self._split_heads(projection(encoder_hidden))
                for projection in self.cross_value_projections
            )
        )
        return cross_keys, cross_values


class WhisperLoadCrossKV(nn.Module):
    """Load encoder payloads into reset decoder state exactly once."""

    def forward(
        self,
        cross_key_payload: torch.Tensor,
        cross_value_payload: torch.Tensor,
        cross_key_cache: torch.Tensor,
        cross_value_cache: torch.Tensor,
        cross_ready: torch.Tensor,
    ) -> torch.Tensor:
        valid = cross_ready == 0
        status = valid.to(dtype=torch.int32)
        cache_mask = valid.reshape((1, 1, 1, 1, 1))

        # CoreAI b2 reliably lowers additive state initialization. Masking the
        # payload makes a repeated load an exact no-op for ordinary finite
        # cache values while keeping the state transition inside the graph.
        cross_key_cache.add_(
            torch.where(cache_mask, cross_key_payload, torch.zeros_like(cross_key_payload))
        )
        cross_value_cache.add_(
            torch.where(cache_mask, cross_value_payload, torch.zeros_like(cross_value_payload))
        )
        cross_ready.add_(status)
        return status.clone()


class _WhisperDecodeLayer(nn.Module):
    """The decoder-owned weights for one incremental Whisper layer."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.self_attn_layer_norm = source.self_attn_layer_norm
        self.self_q_proj = source.self_attn.q_proj
        self.self_k_proj = source.self_attn.k_proj
        self.self_v_proj = source.self_attn.v_proj
        self.self_out_proj = source.self_attn.out_proj
        self.self_scale = source.self_attn.scaling

        self.cross_attn_layer_norm = source.encoder_attn_layer_norm
        self.cross_q_proj = source.encoder_attn.q_proj
        self.cross_out_proj = source.encoder_attn.out_proj
        self.cross_scale = source.encoder_attn.scaling

        self.fc1 = source.fc1
        self.fc2 = source.fc2
        self.final_layer_norm = source.final_layer_norm
        self.activation = source.activation_fn


class WhisperDecodeStep(nn.Module):
    """Decode one token with fixed self K/V and preloaded cross K/V state."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        decoder = model.model.decoder
        self.embed_tokens = decoder.embed_tokens
        self.embed_positions = decoder.embed_positions
        self.layers = nn.ModuleList(_WhisperDecodeLayer(layer) for layer in decoder.layers)
        self.layer_norm = decoder.layer_norm
        self.heads = model.config.decoder_attention_heads
        self.head_dim = model.config.d_model // self.heads
        self.register_buffer(
            "cache_positions",
            torch.arange(WHISPER_DECODER_CACHE_CAPACITY, dtype=torch.int32),
            persistent=False,
        )

    def _project_heads(self, projection: nn.Module, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = values.shape
        return projection(values).view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _attend(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_projection: nn.Module,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = torch.matmul(query, key.transpose(-1, -2))
        if mask is not None:
            weights = weights.masked_fill(~mask, torch.finfo(weights.dtype).min)
        probabilities = torch.softmax(weights, dim=-1)
        attended = torch.matmul(probabilities, value)
        batch, _, sequence, _ = attended.shape
        merged = attended.transpose(1, 2).reshape(batch, sequence, -1)
        return output_projection(merged)

    @staticmethod
    def _update_cache(
        cache: torch.Tensor,
        update: torch.Tensor,
        *,
        layer_index: int,
        position: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        existing = torch.index_select(
            cache[layer_index],
            dim=-2,
            index=position.to(dtype=torch.int64),
        )
        selected_update = torch.where(
            valid.reshape((1, 1, 1, 1)),
            update,
            existing,
        )
        device = update.device
        begin = torch.cat(
            (
                torch.tensor((layer_index,), dtype=torch.int32, device=device),
                torch.zeros((2,), dtype=torch.int32, device=device),
                position,
                torch.zeros((1,), dtype=torch.int32, device=device),
            )
        )
        end = torch.cat(
            (
                torch.tensor((layer_index + 1,), dtype=torch.int32, device=device),
                torch.tensor((1, update.shape[1]), dtype=torch.int32, device=device),
                position + 1,
                torch.tensor((update.shape[-1],), dtype=torch.int32, device=device),
            )
        )
        _mutable_slice_update(
            x=cache,
            update=selected_update.unsqueeze(0),
            begin=begin,
            end=end,
        )

    def forward(
        self,
        token_id: torch.Tensor,
        cross_key_cache: torch.Tensor,
        cross_value_cache: torch.Tensor,
        self_key_cache: torch.Tensor,
        self_value_cache: torch.Tensor,
        position: torch.Tensor,
        cross_ready: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (
            (cross_ready == 1)
            & (position >= 0)
            & (position < WHISPER_DECODER_CACHE_CAPACITY)
        )
        status = valid.to(dtype=torch.int32)
        safe_position = position.clamp(0, WHISPER_DECODER_CACHE_CAPACITY - 1)

        # Identity mutations retain read-only decoder session values as CoreAI
        # state alongside the genuinely mutable self cache and position.
        cross_key_cache.add_(0.0)
        cross_value_cache.add_(0.0)
        cross_ready.add_(0)

        token_hidden = self.embed_tokens(token_id)
        position_hidden = F.embedding(safe_position, self.embed_positions.weight).unsqueeze(0)
        hidden = token_hidden + position_hidden
        self_mask = (self.cache_positions <= safe_position).reshape(1, 1, 1, -1)

        for index, layer in enumerate(self.layers):
            residual = hidden
            normalized = layer.self_attn_layer_norm(hidden)
            query = self._project_heads(layer.self_q_proj, normalized) * layer.self_scale
            new_key = self._project_heads(layer.self_k_proj, normalized)
            new_value = self._project_heads(layer.self_v_proj, normalized)
            self._update_cache(
                self_key_cache,
                new_key,
                layer_index=index,
                position=safe_position,
                valid=valid,
            )
            self._update_cache(
                self_value_cache,
                new_value,
                layer_index=index,
                position=safe_position,
                valid=valid,
            )
            self_output = self._attend(
                query,
                self_key_cache[index],
                self_value_cache[index],
                layer.self_out_proj,
                self_mask,
            )
            hidden = residual + self_output

            residual = hidden
            normalized = layer.cross_attn_layer_norm(hidden)
            cross_query = self._project_heads(layer.cross_q_proj, normalized) * layer.cross_scale
            cross_output = self._attend(
                cross_query,
                cross_key_cache[index],
                cross_value_cache[index],
                layer.cross_out_proj,
                valid.reshape(1, 1, 1, 1),
            )
            hidden = residual + cross_output

            residual = hidden
            hidden = layer.fc2(layer.activation(layer.fc1(layer.final_layer_norm(hidden))))
            hidden = residual + hidden

        position.add_(status)
        hidden = self.layer_norm(hidden)
        logits = F.linear(hidden, self.embed_tokens.weight)
        logits = torch.where(valid.reshape((1, 1, 1)), logits, torch.zeros_like(logits))
        return logits, status.clone()


@dataclass(frozen=True)
class WhisperSplitModules:
    encode: WhisperEncode
    load_cross_kv: WhisperLoadCrossKV
    decode_step: WhisperDecodeStep
    decoder_layers: int
    decoder_heads: int
    head_dim: int
    source_positions: int

    @classmethod
    def from_hf(cls, model: nn.Module) -> WhisperSplitModules:
        config = model.config
        if config.max_target_positions != WHISPER_DECODER_CACHE_CAPACITY:
            raise WhisperExportError(
                "native Whisper requires exactly 448 decoder cache positions"
            )
        if config.d_model % config.decoder_attention_heads:
            raise WhisperExportError("Whisper d_model must divide evenly across decoder heads")
        return cls(
            encode=WhisperEncode(model).eval(),
            load_cross_kv=WhisperLoadCrossKV().eval(),
            decode_step=WhisperDecodeStep(model).eval(),
            decoder_layers=config.decoder_layers,
            decoder_heads=config.decoder_attention_heads,
            head_dim=config.d_model // config.decoder_attention_heads,
            source_positions=config.max_source_positions,
        )

    def new_state(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> WhisperRuntimeState:
        cross_shape = (
            self.decoder_layers,
            1,
            self.decoder_heads,
            self.source_positions,
            self.head_dim,
        )
        self_shape = (
            self.decoder_layers,
            1,
            self.decoder_heads,
            WHISPER_DECODER_CACHE_CAPACITY,
            self.head_dim,
        )
        return WhisperRuntimeState(
            cross_key_cache=torch.zeros(cross_shape, dtype=dtype, device=device),
            cross_value_cache=torch.zeros(cross_shape, dtype=dtype, device=device),
            self_key_cache=torch.zeros(self_shape, dtype=dtype, device=device),
            self_value_cache=torch.zeros(self_shape, dtype=dtype, device=device),
            position=torch.zeros((1,), dtype=torch.int32, device=device),
            cross_ready=torch.zeros((1,), dtype=torch.int32, device=device),
        )


def _export_program(
    module: nn.Module,
    inputs: dict[str, torch.Tensor],
) -> Any:
    import coreai_torch

    from coreai_models.export.mlir_ops import remove_functionalization

    exported = torch.export.export(module.eval(), args=(), kwargs=inputs)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    remove_functionalization(exported)
    return exported


def create_coreai_program(
    split: WhisperSplitModules,
    *,
    input_features: torch.Tensor,
    token_id: torch.Tensor,
    optimize: bool = True,
) -> Any:
    """Create one AIProgram containing the frozen three-entrypoint ABI."""
    from coreai_torch import TorchConverter

    from coreai_models.export.mlir_ops import register_custom_torch_lowering

    state = split.new_state(dtype=input_features.dtype, device=input_features.device)
    cross_key_payload = torch.zeros_like(state.cross_key_cache)
    cross_value_payload = torch.zeros_like(state.cross_value_cache)

    converter = TorchConverter()
    converter.add_exported_program(
        _export_program(split.encode, {"input_features": input_features}),
        input_names=("input_features",),
        output_names=("cross_key_payload", "cross_value_payload"),
        state_names=(),
        entrypoint_name="encode",
    )
    converter.add_exported_program(
        _export_program(
            split.load_cross_kv,
            {
                "cross_key_payload": cross_key_payload,
                "cross_value_payload": cross_value_payload,
                "cross_key_cache": state.cross_key_cache,
                "cross_value_cache": state.cross_value_cache,
                "cross_ready": state.cross_ready,
            },
        ),
        input_names=("cross_key_payload", "cross_value_payload"),
        output_names=("load_status",),
        state_names=("cross_key_cache", "cross_value_cache", "cross_ready"),
        entrypoint_name="load_cross_kv",
    )
    converter.add_exported_program(
        _export_program(
            split.decode_step,
            {
                "token_id": token_id,
                "cross_key_cache": state.cross_key_cache,
                "cross_value_cache": state.cross_value_cache,
                "self_key_cache": state.self_key_cache,
                "self_value_cache": state.self_value_cache,
                "position": state.position,
                "cross_ready": state.cross_ready,
            },
        ),
        input_names=("token_id",),
        output_names=("logits", "decode_status"),
        state_names=(
            "cross_key_cache",
            "cross_value_cache",
            "self_key_cache",
            "self_value_cache",
            "position",
            "cross_ready",
        ),
        entrypoint_name="decode_step",
    )
    register_custom_torch_lowering(converter)
    program = converter.to_coreai()
    if optimize:
        program.optimize()
    return program


def _minimal_model() -> nn.Module:
    from transformers import WhisperConfig, WhisperForConditionalGeneration

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


async def _run_minimal_coreai_probe(temp_root: Path) -> dict[str, Any]:
    from coreai.runtime import NDArray

    if not temp_root.is_dir():
        raise WhisperExportError(f"CoreAI probe temp root is not a directory: {temp_root}")

    model = _minimal_model()
    split = WhisperSplitModules.from_hf(model)
    torch.manual_seed(23)
    features = torch.randn((1, 4, 12), dtype=torch.float32)
    tokens = (
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([[7]], dtype=torch.int32),
    )

    with torch.no_grad():
        expected_keys, expected_values = split.encode(features)
        expected_state = split.new_state(dtype=torch.float32)
        split.load_cross_kv(
            expected_keys,
            expected_values,
            expected_state.cross_key_cache,
            expected_state.cross_value_cache,
            expected_state.cross_ready,
        )
        expected_steps = [
            split.decode_step(
                token,
                expected_state.cross_key_cache,
                expected_state.cross_value_cache,
                expected_state.self_key_cache,
                expected_state.self_value_cache,
                expected_state.position,
                expected_state.cross_ready,
            )
            for token in tokens
        ]
        expected_logits = [logits for logits, _ in expected_steps]

    program = create_coreai_program(
        split,
        input_features=features,
        token_id=tokens[0],
    )
    state = split.new_state(dtype=torch.float32)
    with tempfile.TemporaryDirectory(
        prefix="whisper-full-minimal-",
        suffix=".aimodel",
        dir=temp_root,
    ) as temp_dir:
        asset = program.save_asset(Path(temp_dir))
        async with asset.executable() as executable:
            encode = executable.load_function("encode")
            load_cross_kv = executable.load_function("load_cross_kv")
            decode_step = executable.load_function("decode_step")
            def runtime_state_from(source: WhisperRuntimeState) -> dict[str, Any]:
                return {
                    "cross_key_cache": NDArray(data=source.cross_key_cache),
                    "cross_value_cache": NDArray(data=source.cross_value_cache),
                    "self_key_cache": NDArray(data=source.self_key_cache),
                    "self_value_cache": NDArray(data=source.self_value_cache),
                    "position": NDArray(data=source.position),
                    "cross_ready": NDArray(data=source.cross_ready),
                }

            def snapshot(runtime_state: dict[str, Any]) -> dict[str, np.ndarray]:
                return {
                    name: np.array(value.numpy(), copy=True)
                    for name, value in runtime_state.items()
                }

            def state_matches(
                runtime_state: dict[str, Any],
                expected: dict[str, np.ndarray],
            ) -> bool:
                return all(
                    np.array_equal(runtime_state[name].numpy(), value)
                    for name, value in expected.items()
                )

            invalid_before_load_source = split.new_state(dtype=torch.float32)
            invalid_before_load_source.cross_key_cache.fill_(1.25)
            invalid_before_load_source.cross_value_cache.fill_(-2.5)
            invalid_before_load_source.self_key_cache.fill_(3.75)
            invalid_before_load_source.self_value_cache.fill_(-4.5)
            invalid_before_load_state = runtime_state_from(invalid_before_load_source)
            invalid_before_load_snapshot = snapshot(invalid_before_load_state)
            invalid_before_load = await decode_step(
                {"token_id": NDArray(data=tokens[0])},
                state=invalid_before_load_state,
            )

            runtime_state = runtime_state_from(state)
            encoded = await encode({"input_features": NDArray(data=features)})
            loaded = await load_cross_kv(
                {
                    "cross_key_payload": encoded["cross_key_payload"],
                    "cross_value_payload": encoded["cross_value_payload"],
                },
                state={
                    "cross_key_cache": runtime_state["cross_key_cache"],
                    "cross_value_cache": runtime_state["cross_value_cache"],
                    "cross_ready": runtime_state["cross_ready"],
                },
            )
            before_second_load = snapshot(runtime_state)
            second_load = await load_cross_kv(
                {
                    "cross_key_payload": encoded["cross_key_payload"],
                    "cross_value_payload": encoded["cross_value_payload"],
                },
                state={
                    "cross_key_cache": runtime_state["cross_key_cache"],
                    "cross_value_cache": runtime_state["cross_value_cache"],
                    "cross_ready": runtime_state["cross_ready"],
                },
            )
            second_load_unchanged = state_matches(runtime_state, before_second_load)
            actual_logits = []
            decode_statuses = []
            for token in tokens:
                output = await decode_step(
                    {"token_id": NDArray(data=token)},
                    state=runtime_state,
                )
                actual_logits.append(output["logits"].numpy())
                decode_statuses.append(
                    output["decode_status"].numpy().reshape(-1).tolist()
                )

            invalid_position_statuses: list[list[int]] = []
            invalid_position_state_unchanged: list[bool] = []
            for readiness, position in ((-1, 0), (2, 0), (1, -1), (1, 448)):
                invalid_source = split.new_state(dtype=torch.float32)
                invalid_source.cross_key_cache.fill_(1.0)
                invalid_source.cross_value_cache.fill_(2.0)
                invalid_source.self_key_cache.fill_(3.0)
                invalid_source.self_value_cache.fill_(4.0)
                invalid_source.cross_ready.fill_(readiness)
                invalid_source.position.fill_(position)
                invalid_state = runtime_state_from(invalid_source)
                invalid_snapshot = snapshot(invalid_state)
                invalid_output = await decode_step(
                    {"token_id": NDArray(data=tokens[0])},
                    state=invalid_state,
                )
                invalid_position_statuses.append(
                    invalid_output["decode_status"].numpy().reshape(-1).tolist()
                )
                invalid_position_state_unchanged.append(
                    state_matches(invalid_state, invalid_snapshot)
                )

            actual_keys = encoded["cross_key_payload"].numpy()
            actual_values = encoded["cross_value_payload"].numpy()
            expected_logits_array = np.concatenate(
                [value.detach().numpy() for value in expected_logits],
                axis=1,
            )
            actual_logits_array = np.concatenate(actual_logits, axis=1)
            self_keys = runtime_state["self_key_cache"].numpy()
            self_values = runtime_state["self_value_cache"].numpy()
            return {
                "call_order": ["encode", "load_cross_kv", "decode_step", "decode_step"],
                "entrypoints": ["decode_step", "encode", "load_cross_kv"],
                "load_status": loaded["load_status"].numpy().reshape(-1).tolist(),
                "decode_statuses": decode_statuses,
                "invalid_decode_before_load_status": invalid_before_load[
                    "decode_status"
                ]
                .numpy()
                .reshape(-1)
                .tolist(),
                "invalid_decode_before_load_zero_logits": bool(
                    np.count_nonzero(invalid_before_load["logits"].numpy()) == 0
                ),
                "invalid_decode_before_load_state_unchanged": state_matches(
                    invalid_before_load_state,
                    invalid_before_load_snapshot,
                ),
                "invalid_second_load_status": second_load["load_status"]
                .numpy()
                .reshape(-1)
                .tolist(),
                "invalid_second_load_state_unchanged": second_load_unchanged,
                "invalid_position_statuses": invalid_position_statuses,
                "invalid_position_state_unchanged": invalid_position_state_unchanged,
                "position": runtime_state["position"].numpy().reshape(-1).tolist(),
                "cross_ready": runtime_state["cross_ready"].numpy().reshape(-1).tolist(),
                "self_key_tail_nonzero": int(np.count_nonzero(self_keys[..., 2:, :])),
                "self_value_tail_nonzero": int(np.count_nonzero(self_values[..., 2:, :])),
                "max_encode_key_error": float(
                    np.max(np.abs(actual_keys - expected_keys.detach().numpy()))
                ),
                "max_encode_value_error": float(
                    np.max(np.abs(actual_values - expected_values.detach().numpy()))
                ),
                "max_decode_error": float(
                    np.max(np.abs(actual_logits_array - expected_logits_array))
                ),
            }


def run_minimal_coreai_probe(temp_root: Path) -> dict[str, Any]:
    """Compile and execute the minimal real Whisper split architecture."""
    return asyncio.run(_run_minimal_coreai_probe(temp_root))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export native split Whisper large-v2")
    parser.add_argument("--minimal-coreai-proof", action="store_true")
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args()
    if not args.minimal_coreai_proof:
        parser.error("no export mode selected")
    if args.temp_root is None:
        parser.error("--temp-root is required for --minimal-coreai-proof")
    result = run_minimal_coreai_probe(args.temp_root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
