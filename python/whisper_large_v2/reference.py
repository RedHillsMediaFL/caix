"""Deterministic tiny Whisper split used to prove the native streaming ABI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TinyWhisperConfig:
    d_model: int = 16
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feed_forward: int = 32
    vocabulary_size: int = 29
    max_source_positions: int = 6
    max_decoder_positions: int = 8

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")


@dataclass
class TinyDecoderState:
    cross_keys: tuple[torch.Tensor, ...]
    cross_values: tuple[torch.Tensor, ...]
    self_keys: list[torch.Tensor]
    self_values: list[torch.Tensor]
    position: int = 0


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

    def begin_split(self, features: torch.Tensor) -> TinyDecoderState:
        encoder_hidden = self.encode(features)
        cross_keys = tuple(
            layer.cross_attention.key(encoder_hidden) for layer in self.decoder_layers
        )
        cross_values = tuple(
            layer.cross_attention.value(encoder_hidden) for layer in self.decoder_layers
        )
        batch = features.shape[0]
        empty_shape = (batch, self.config.heads, 0, self.config.d_model // self.config.heads)
        self_keys = [features.new_empty(empty_shape) for _ in self.decoder_layers]
        self_values = [features.new_empty(empty_shape) for _ in self.decoder_layers]
        return TinyDecoderState(
            cross_keys=cross_keys,
            cross_values=cross_values,
            self_keys=self_keys,
            self_values=self_values,
        )

    def decode_step(
        self,
        token_id: torch.Tensor,
        state: TinyDecoderState,
    ) -> torch.Tensor:
        if token_id.shape[1] != 1:
            raise ValueError("decode_step accepts exactly one token")
        if state.position >= self.config.max_decoder_positions:
            raise ValueError("decoder state exceeds max_decoder_positions")

        position = torch.tensor([state.position], device=token_id.device)
        hidden = self.token_embedding(token_id) + self.decoder_positions(position).unsqueeze(0)
        for index, layer in enumerate(self.decoder_layers):
            normalized = layer.self_norm(hidden)
            query = layer.self_attention.query(normalized)
            new_key = layer.self_attention.key(normalized)
            new_value = layer.self_attention.value(normalized)
            state.self_keys[index] = torch.cat((state.self_keys[index], new_key), dim=2)
            state.self_values[index] = torch.cat((state.self_values[index], new_value), dim=2)
            hidden = hidden + layer.self_attention.projected(
                query,
                state.self_keys[index],
                state.self_values[index],
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

        state.position += 1
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
