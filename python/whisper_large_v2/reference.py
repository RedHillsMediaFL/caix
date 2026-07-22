"""Deterministic tiny Whisper split used to prove the native streaming ABI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

WHISPER_DECODER_CACHE_CAPACITY = 448


@dataclass(frozen=True)
class TinyWhisperConfig:
    d_model: int = 16
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feed_forward: int = 32
    vocabulary_size: int = 29
    max_source_positions: int = 6
    max_decoder_positions: int = WHISPER_DECODER_CACHE_CAPACITY

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.max_decoder_positions != WHISPER_DECODER_CACHE_CAPACITY:
            raise ValueError("native decoder cache capacity must be exactly 448 tokens")


@dataclass(frozen=True)
class TinyCrossKVPayload:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class TinyDecoderState:
    cross_keys: torch.Tensor
    cross_values: torch.Tensor
    self_keys: torch.Tensor
    self_values: torch.Tensor
    position: torch.Tensor
    cross_ready: torch.Tensor


class DecoderStateError(RuntimeError):
    """The fixed native decoder state was used out of sequence or overflowed."""


class _CountedLinear(nn.Linear):
    calls: int

    def __init__(self, size: int) -> None:
        super().__init__(size, size)
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(inputs)


class _Attention(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = _CountedLinear(d_model)
        self.v_proj = _CountedLinear(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def query(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._split_heads(self.q_proj(inputs))

    def key(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._split_heads(self.k_proj(inputs))

    def value(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._split_heads(self.v_proj(inputs))

    def projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if causal:
            query_length = query.shape[-2]
            key_length = key.shape[-2]
            mask = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=scores.device,
            ).triu(diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        return self.out_proj(self._merge_heads(torch.matmul(probabilities, value)))

    def forward(
        self,
        query_inputs: torch.Tensor,
        key_value_inputs: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        return self.projected(
            self.query(query_inputs),
            self.key(key_value_inputs),
            self.value(key_value_inputs),
            causal=causal,
        )

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = inputs.shape
        return inputs.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, sequence, _ = inputs.shape
        return inputs.transpose(1, 2).reshape(batch, sequence, self.d_model)


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, hidden_size)
        self.down = nn.Linear(hidden_size, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(inputs)))


class _EncoderLayer(nn.Module):
    def __init__(self, config: TinyWhisperConfig) -> None:
        super().__init__()
        self.self_attention = _Attention(config.d_model, config.heads)
        self.self_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = _FeedForward(config.d_model, config.feed_forward)
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(inputs)
        inputs = inputs + self.self_attention(normalized, normalized, causal=False)
        return inputs + self.feed_forward(self.final_norm(inputs))


class _DecoderLayer(nn.Module):
    def __init__(self, config: TinyWhisperConfig) -> None:
        super().__init__()
        self.self_attention = _Attention(config.d_model, config.heads)
        self.self_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = _Attention(config.d_model, config.heads)
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = _FeedForward(config.d_model, config.feed_forward)
        self.final_norm = nn.LayerNorm(config.d_model)


class TinyWhisperReference(nn.Module):
    """Two-path reference: monolithic sequence decode and cached one-token decode."""

    def __init__(self, config: TinyWhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.encoder_layers = nn.ModuleList(
            _EncoderLayer(config) for _ in range(config.encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.decoder_positions = nn.Embedding(config.max_decoder_positions, config.d_model)
        self.decoder_layers = nn.ModuleList(
            _DecoderLayer(config) for _ in range(config.decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(config.d_model)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        sequence = features.shape[1]
        if sequence > self.config.max_source_positions:
            raise ValueError("features exceed max_source_positions")
        positions = torch.arange(sequence, device=features.device)
        hidden = features + self.encoder_positions(positions).unsqueeze(0)
        for layer in self.encoder_layers:
            hidden = layer(hidden)
        return self.encoder_norm(hidden)

    def forward(self, features: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        encoder_hidden = self.encode(features)
        sequence = token_ids.shape[1]
        if sequence > self.config.max_decoder_positions:
            raise ValueError("tokens exceed max_decoder_positions")
        positions = torch.arange(sequence, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.decoder_positions(positions).unsqueeze(0)
        for layer in self.decoder_layers:
            normalized = layer.self_norm(hidden)
            hidden = hidden + layer.self_attention(normalized, normalized, causal=True)
            normalized = layer.cross_norm(hidden)
            hidden = hidden + layer.cross_attention(normalized, encoder_hidden, causal=False)
            hidden = hidden + layer.feed_forward(layer.final_norm(hidden))
        hidden = self.decoder_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)

    def encode_cross_kv(self, features: torch.Tensor) -> TinyCrossKVPayload:
        encoder_hidden = self.encode(features)
        if encoder_hidden.shape[1] != self.config.max_source_positions:
            raise ValueError("split encoder input must fill max_source_positions")
        cross_keys = torch.stack(
            tuple(layer.cross_attention.key(encoder_hidden) for layer in self.decoder_layers)
        )
        cross_values = torch.stack(
            tuple(layer.cross_attention.value(encoder_hidden) for layer in self.decoder_layers)
        )
        return TinyCrossKVPayload(keys=cross_keys, values=cross_values)

    def new_decoder_state(self, features: torch.Tensor) -> TinyDecoderState:
        batch = features.shape[0]
        head_dim = self.config.d_model // self.config.heads
        cross_shape = (
            self.config.decoder_layers,
            batch,
            self.config.heads,
            self.config.max_source_positions,
            head_dim,
        )
        self_shape = (
            self.config.decoder_layers,
            batch,
            self.config.heads,
            WHISPER_DECODER_CACHE_CAPACITY,
            head_dim,
        )
        return TinyDecoderState(
            cross_keys=features.new_zeros(cross_shape),
            cross_values=features.new_zeros(cross_shape),
            self_keys=features.new_zeros(self_shape),
            self_values=features.new_zeros(self_shape),
            position=torch.zeros((1,), dtype=torch.int32, device=features.device),
            cross_ready=torch.zeros((1,), dtype=torch.int32, device=features.device),
        )

    def load_cross_kv(
        self,
        payload: TinyCrossKVPayload,
        state: TinyDecoderState,
    ) -> None:
        if int(state.cross_ready.item()) != 0:
            raise DecoderStateError("load_cross_kv must run exactly once per utterance")
        if int(state.position.item()) != 0:
            raise DecoderStateError("decoder state must be reset before load_cross_kv")
        if payload.keys.shape != state.cross_keys.shape:
            raise DecoderStateError("cross key payload shape differs from decoder state")
        if payload.values.shape != state.cross_values.shape:
            raise DecoderStateError("cross value payload shape differs from decoder state")
        state.cross_keys.copy_(payload.keys)
        state.cross_values.copy_(payload.values)
        state.cross_ready.fill_(1)

    def begin_split(self, features: torch.Tensor) -> TinyDecoderState:
        payload = self.encode_cross_kv(features)
        state = self.new_decoder_state(features)
        self.load_cross_kv(payload, state)
        return state

    def reset_decoder_state(self, state: TinyDecoderState) -> None:
        state.cross_keys.zero_()
        state.cross_values.zero_()
        state.self_keys.zero_()
        state.self_values.zero_()
        state.position.zero_()
        state.cross_ready.zero_()

    def decode_step(
        self,
        token_id: torch.Tensor,
        state: TinyDecoderState,
    ) -> torch.Tensor:
        if token_id.shape[1] != 1:
            raise ValueError("decode_step accepts exactly one token")
        if int(state.cross_ready.item()) != 1:
            raise DecoderStateError("load_cross_kv must run before decode_step")
        position_index = int(state.position.item())
        if position_index >= WHISPER_DECODER_CACHE_CAPACITY:
            raise DecoderStateError("decoder state exceeds 448-token capacity")

        position = state.position.to(device=token_id.device, dtype=torch.long)
        hidden = self.token_embedding(token_id) + self.decoder_positions(position).unsqueeze(0)
        for index, layer in enumerate(self.decoder_layers):
            normalized = layer.self_norm(hidden)
            query = layer.self_attention.query(normalized)
            new_key = layer.self_attention.key(normalized)
            new_value = layer.self_attention.value(normalized)
            state.self_keys[index, :, :, position_index : position_index + 1, :].copy_(
                new_key
            )
            state.self_values[index, :, :, position_index : position_index + 1, :].copy_(
                new_value
            )
            hidden = hidden + layer.self_attention.projected(
                query,
                state.self_keys[index, :, :, : position_index + 1, :],
                state.self_values[index, :, :, : position_index + 1, :],
                causal=False,
            )
            cross_query = layer.cross_attention.query(layer.cross_norm(hidden))
            hidden = hidden + layer.cross_attention.projected(
                cross_query,
                state.cross_keys[index],
                state.cross_values[index],
                causal=False,
            )
            hidden = hidden + layer.feed_forward(layer.final_norm(hidden))

        state.position.add_(1)
        hidden = self.decoder_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)

    @property
    def cross_projection_counts(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (layer.cross_attention.k_proj.calls, layer.cross_attention.v_proj.calls)
            for layer in self.decoder_layers
        )

    def reset_cross_projection_counts(self) -> None:
        for layer in self.decoder_layers:
            layer.cross_attention.k_proj.calls = 0
            layer.cross_attention.v_proj.calls = 0
