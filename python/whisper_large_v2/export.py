"""Native split Whisper modules and CoreAI export support.

The hot decoder owns fixed caller-supplied state. Cross-attention K/V projections
run in ``encode`` exactly once per audio window; ``decode_step`` only projects a
single token and updates one slot in the 448-token self-attention cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from coreai_models.primitives._ops import mutable_slice_update
from whisper_large_v2.reference import WHISPER_DECODER_CACHE_CAPACITY


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
        # CoreAI b2 reliably lowers additive state initialization. The host
        # enforces zeroed state and a single load per audio window.
        cross_key_cache.add_(cross_key_payload)
        cross_value_cache.add_(cross_value_payload)
        cross_ready.add_(1)
        return cross_ready.clone()


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
    ) -> None:
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
        mutable_slice_update(
            x=cache,
            update=update.unsqueeze(0),
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
    ) -> torch.Tensor:
        # Identity mutations retain read-only decoder session values as CoreAI
        # state alongside the genuinely mutable self cache and position.
        cross_key_cache.add_(0.0)
        cross_value_cache.add_(0.0)
        cross_ready.add_(0)

        token_hidden = self.embed_tokens(token_id)
        position_hidden = F.embedding(position, self.embed_positions.weight).unsqueeze(0)
        hidden = token_hidden + position_hidden
        self_mask = (self.cache_positions <= position).reshape(1, 1, 1, -1)

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
                position=position,
            )
            self._update_cache(
                self_value_cache,
                new_value,
                layer_index=index,
                position=position,
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
            )
            hidden = residual + cross_output

            residual = hidden
            hidden = layer.fc2(layer.activation(layer.fc1(layer.final_layer_norm(hidden))))
            hidden = residual + hidden

        position.add_(1)
        hidden = self.layer_norm(hidden)
        return F.linear(hidden, self.embed_tokens.weight)


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
