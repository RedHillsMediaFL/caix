# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3.8-27B macOS authoring for caix.

Implements the Qwen3.5 *text* decoder (``qwen3_5``) for the Core AI macOS export
path. Qwen3.5 is a hybrid linear/full-attention model (the same family as
``qwen3_next``): most layers are gated-delta linear-attention (Mamba-style SSM +
short causal conv1d), with a full softmax-attention layer every
``full_attention_interval`` layers. Qwen3.5 is not yet in ``transformers``, so
this module also registers lightweight HuggingFace ``PretrainedConfig`` classes
with ``AutoConfig`` (the export pipeline calls ``AutoConfig.from_pretrained``
before our model registry is consulted).

Architecture handled here (verified against ``empero-ai/Qwythos-9B-...`` and
``Qwen/Qwen3.6-27B``):

* model_type ``qwen3_5`` (multimodal ``Qwen3_5ForConditionalGeneration``; text
  decoder lives under the ``model.language_model.`` safetensors prefix, vision
  tower under ``model.visual.`` -> dropped). Untied embeddings, separate
  ``lm_head``.
* ``layer_types`` alternate 3x ``linear_attention`` : 1x ``full_attention``
  (``full_attention_interval`` 4).
* Linear layers: gated-delta net. Four separate input projections
  (``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_a`` / ``in_proj_b``), depthwise
  ``conv1d`` (kernel ``linear_conv_kernel_dim`` 4) over the q/k/v stream, the
  gated-delta recurrence (``coreai_torch`` ``gated_delta_update`` composite op),
  a gated RMSNorm (``norm`` * silu(z)) and ``out_proj``.
* Full layers: GQA softmax attention with an *output gate* (``q_proj`` emits
  query and gate interleaved, ``attn_output_gate``), per-head q/k RMSNorm,
  ``head_dim`` 256.
* mRoPE with ``partial_rotary_factor`` 0.25 (rotary dim 64), YaRN scaling,
  ``rope_theta`` 1e7. For a text-only decoder mRoPE collapses to standard RoPE
  (all three position sections share the token position), so a single 1-D
  ``position_ids`` suffices.
* Gemma/Qwen3-Next ``(1 + weight)`` RMSNorm everywhere except the gated
  delta-net ``norm`` (plain weight).

THE FOUR-STATE CONTRACT
-----------------------
The hybrid decoder has exactly four mutable Core AI states:

* ``keyCache`` / ``valueCache``: dynamic FP16 K/V for the sixteen full-attention
  layers only, shape ``[16, 1, 4, S, 256]``;
* ``convState``: fixed FP16 convolution history for the 48 linear-attention
  layers, shape ``[48, 1, 10240, 3]``;
* ``recurrentState``: fixed FP32 gated-delta recurrence, shape
  ``[48, 1, 48, 128, 128]``.

This is an explicit Core AI state ABI. It prevents an all-64-layer K/V
allocation (64 GiB at 262K), has no cache-prefix workaround, and lets the
modern CoreAI pipelined engine retain the fixed hybrid states on GPU.
"""

import gc
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch.composite_ops import GatedDeltaUpdate
from safetensors import safe_open
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self, override

from coreai_models._constants import KEY_CACHE_NAME, MAIN_GRAPH_NAME, VALUE_CACHE_NAME
from coreai_models.models.base import BaseForCausalLM, TraceSpec, _save_and_mmap_safetensors
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNormGated
from coreai_models.primitives.macos.rms_norm import RMSNormPlusOne as Qwen3_5RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# HuggingFace config classes (Qwen3.5 is not in transformers yet)
# --------------------------------------------------------------------------- #


class Qwen3_5TextConfig(PretrainedConfig):
    """Text-decoder config for ``qwen3_5``."""

    model_type = "qwen3_5_text"

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        full_attention_interval: int = 4,
        layer_types: list[str] | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        attn_output_gate: bool = True,
        rms_norm_eps: float = 1e-6,
        partial_rotary_factor: float = 0.25,
        rope_parameters: dict | None = None,
        rope_theta: float = 1e7,
        max_position_embeddings: int = 1048576,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.full_attention_interval = full_attention_interval
        self.layer_types = layer_types
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.attn_output_gate = attn_output_gate
        self.rms_norm_eps = rms_norm_eps
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3_5Config(PretrainedConfig):
    """Top-level config for the multimodal ``qwen3_5`` checkpoint."""

    model_type = "qwen3_5"
    sub_configs = {"text_config": Qwen3_5TextConfig}

    def __init__(self, text_config=None, tie_word_embeddings: bool = False, **kwargs) -> None:
        if isinstance(text_config, dict):
            text_config = Qwen3_5TextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3_5TextConfig()
        self.text_config = text_config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def _register_hf() -> None:
    """Register Qwen3.5 configs with AutoConfig and shim the tokenizer loader.

    Idempotent: safe to call from multiple import paths.
    """
    for model_type, cfg_cls in (
        ("qwen3_5", Qwen3_5Config),
        ("qwen3_5_text", Qwen3_5TextConfig),
    ):
        try:
            AutoConfig.register(model_type, cfg_cls)
        except ValueError:
            # Already registered (e.g. re-import) -- fine.
            pass

    # The Qwen3.5 tokenizer_config.json declares ``tokenizer_class``
    # "TokenizersBackend" (a transformers 5.x backend class absent in 4.57), so
    # ``AutoTokenizer.from_pretrained`` (used by the bundle writer) raises. The
    # checkpoint ships a fast ``tokenizer.json``, so resolve the unknown class
    # to ``PreTrainedTokenizerFast``. Narrow: only touches this one class name.
    import transformers.models.auto.tokenization_auto as _ta
    from transformers import PreTrainedTokenizerFast

    _orig = _ta.tokenizer_class_from_name
    if not getattr(_orig, "_qwen3_5_patched", False):

        def _patched(class_name):  # type: ignore[no-untyped-def]
            if class_name == "TokenizersBackend":
                return PreTrainedTokenizerFast
            return _orig(class_name)

        _patched._qwen3_5_patched = True  # type: ignore[attr-defined]
        _ta.tokenizer_class_from_name = _patched


_register_hf()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_full_layer(config, layer_idx: int) -> bool:
    """Whether ``layer_idx`` is a full softmax-attention layer."""
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and layer_idx < len(layer_types):
        return layer_types[layer_idx] == "full_attention"
    interval = getattr(config, "full_attention_interval", 4)
    return (layer_idx + 1) % interval == 0


def _rope_params(config) -> dict:
    params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
    return params if isinstance(params, dict) else {}


def _compute_yarn_freqs(
    dims: int,
    base: float,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """YaRN inverse-frequencies and the attention (m)scale.

    Mirrors :class:`coreai_models.primitives.macos.rope.YarnRoPE` but exposes the
    raw ``freqs`` (length ``dims // 2``) and ``mscale`` so we can drive the
    composite RoPE op with *partial* rotary dims (the primitive only supports
    full-head YaRN when ``mscale != 1``). Returns ``(freqs_fp32, mscale)``.
    """

    def find_correction_dim(num_rotations: float) -> float:
        return (dims * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = math.floor(find_correction_dim(beta_fast))
    high = math.ceil(find_correction_dim(beta_slow))
    low = max(low, 0)
    high = min(high, dims - 1)

    with torch.device("cpu"):
        freq_extra = base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims)
        freq_inter = factor * freq_extra
        denom = high - low if high != low else 1e-3
        ramp = torch.clip((torch.arange(dims // 2, dtype=torch.float32) - low) / denom, 0, 1)
        freq_mask = 1.0 - ramp
        freqs = (freq_inter * freq_mask + freq_extra * (1 - freq_mask)) / (freq_inter * freq_extra)

    mscale = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return freqs, float(mscale)


class Qwen38HybridState:
    """Mutable state views with compact K/V and explicit fixed hybrid tensors."""

    def __init__(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        self.kv = KVCache(key_cache, value_cache)
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state

    @staticmethod
    def _read(tensor: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return tensor.narrow(0, layer_idx, 1).squeeze(0)

    @staticmethod
    def _write(tensor: torch.Tensor, layer_idx: int, update: torch.Tensor) -> None:
        """Mutate a complete fixed-state layer through Core AI's traceable update op."""
        device = tensor.device
        begin = [torch.tensor((layer_idx,), dtype=torch.int32, device=device)]
        end = [torch.tensor((layer_idx + 1,), dtype=torch.int32, device=device)]
        for dimension in range(1, tensor.dim()):
            begin.append(torch.tensor((0,), dtype=torch.int32, device=device))
            end.append(torch.tensor((tensor.size(dimension),), dtype=torch.int32, device=device))
        mutable_slice_update(
            x=tensor,
            update=update.unsqueeze(0),
            begin=torch.cat(begin),
            end=torch.cat(end),
        )

    def read_conv(self, linear_layer_idx: int) -> torch.Tensor:
        return self._read(self.conv_state, linear_layer_idx)

    def write_conv(self, linear_layer_idx: int, update: torch.Tensor) -> None:
        self._write(self.conv_state, linear_layer_idx, update)

    def read_recurrent(self, linear_layer_idx: int) -> torch.Tensor:
        return self._read(self.recurrent_state, linear_layer_idx)

    def write_recurrent(self, linear_layer_idx: int, update: torch.Tensor) -> None:
        self._write(self.recurrent_state, linear_layer_idx, update)


# --------------------------------------------------------------------------- #
# Partial YaRN RoPE
# --------------------------------------------------------------------------- #


class Qwen3_5RoPE(nn.Module):
    """Partial-rotary YaRN RoPE for the full-attention layers.

    Computed with primitive rotate-half ops (``rotate_half(x) = cat(-x2, x1)``)
    rather than the externalized ``rope`` composite. The composite's *native*
    lowering requires an explicit ``freqs`` tensor of width ``head_dim/2`` (the
    "half_embed"), but partial rotary only has ``rotary_dim/2`` real
    frequencies; padding them out to ``head_dim/2`` breaks the composite's own
    torch tracing (the wide cos/sin no longer broadcasts against the
    ``rotary_dim/2`` query slice). gemma4 sidesteps this by never passing
    explicit freqs (it derives them from ``base``), but YaRN needs custom
    blended frequencies, so we apply the rotation directly. This matches HF
    Qwen3-Next ``apply_rotary_pos_emb`` exactly: rotate the first ``rotary_dim``
    dims, pass the tail through, with the YaRN attention scale folded into
    cos/sin.
    """

    def __init__(self, head_dim: int, rotary_dim: int, base: float, rope_params: dict) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.half = rotary_dim // 2

        rope_type = rope_params.get("rope_type") or rope_params.get("type", "default")
        if rope_type == "yarn":
            factor = float(rope_params.get("factor", 1.0))
            freqs, mscale = _compute_yarn_freqs(
                dims=rotary_dim,
                base=base,
                factor=factor,
                original_max_position_embeddings=int(
                    rope_params.get("original_max_position_embeddings", 4096)
                ),
                beta_fast=float(rope_params.get("beta_fast", 32.0)),
                beta_slow=float(rope_params.get("beta_slow", 1.0)),
            )
        else:
            # Plain RoPE: standard inverse frequencies, no attention scaling.
            with torch.device("cpu"):
                exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
                freqs = 1.0 / torch.pow(base, exponent)
            mscale = 1.0

        # Plain attribute (NOT a buffer) pinned to CPU: the model is built on the
        # meta device so this constant must be real, and ``model.to(bf16)`` must
        # leave it fp32 for an accurate angle computation.
        with torch.device("cpu"):
            self._freqs = freqs.to(torch.float32)  # [rotary_dim // 2]
        self.mscale = float(mscale)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, H, S, head_dim]; position_ids: [B, S]
        pos = position_ids.float().unsqueeze(-1)  # [B, S, 1]
        angle = pos * self._freqs  # [B, S, half]
        emb = torch.cat([angle, angle], dim=-1)  # [B, S, rotary_dim]
        cos = (emb.cos() * self.mscale).unsqueeze(1).to(x.dtype)  # [B, 1, S, rotary_dim]
        sin = (emb.sin() * self.mscale).unsqueeze(1).to(x.dtype)

        rot = x[..., : self.rotary_dim]
        passthrough = x[..., self.rotary_dim :]
        x1 = rot[..., : self.half]
        x2 = rot[..., self.half :]
        rotate_half = torch.cat([-x2, x1], dim=-1)
        rot = rot * cos + rotate_half * sin
        return torch.cat([rot, passthrough], dim=-1)


# --------------------------------------------------------------------------- #
# Linear attention (gated delta net)
# --------------------------------------------------------------------------- #


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated delta-net (Mamba-style linear attention) token mixer.

    `convState` and `recurrentState` hold this layer's explicit persistent state.
    """

    def __init__(self, config, layer_idx: int, linear_state_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.linear_state_idx = linear_state_idx
        hidden = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.n_groups = self.num_v_heads // self.num_k_heads
        self.conv_kernel = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Projections (four separate matrices, matching the qwen3_5 checkpoint).
        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_v_heads, bias=False)

        # Depthwise causal conv over the q/k/v stream (weight [conv_dim, 1, k]).
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            bias=False,
            padding=0,
        )

        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        # Gated delta recurrence (externalized composite). use_qk_l2_norm matches
        # HF's ``use_qk_l2norm_in_kernel=True``.
        self.gated_delta = GatedDeltaUpdate(use_qk_l2_norm=True)

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, state: Qwen38HybridState | None = None) -> torch.Tensor:
        b, s, _ = x.shape

        qkv = self.in_proj_qkv(x)  # [B, S, conv_dim]
        z = self.in_proj_z(x)  # [B, S, value_dim]
        a = self.in_proj_a(x)  # [B, S, num_v_heads]
        bd = self.in_proj_b(x)  # [B, S, num_v_heads]

        # --- causal conv1d with carried state ---
        mixed = qkv.transpose(1, 2)  # [B, conv_dim, S]
        if state is not None:
            conv_state = state.read_conv(self.linear_state_idx)
        else:
            conv_state = torch.zeros(
                b, self.conv_dim, self.conv_kernel - 1, dtype=mixed.dtype, device=mixed.device
            )
        x_full = torch.cat([conv_state, mixed], dim=-1)  # [B, conv_dim, (k-1)+S]
        conv_out = F.conv1d(x_full, self.conv1d.weight, groups=self.conv_dim, padding=0)
        conv_out = F.silu(conv_out)  # [B, conv_dim, S]
        if state is not None:
            new_conv_state = x_full.narrow(-1, x_full.shape[-1] - (self.conv_kernel - 1), self.conv_kernel - 1)
            state.write_conv(self.linear_state_idx, new_conv_state)
        mixed = conv_out.transpose(1, 2)  # [B, S, conv_dim]

        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, s, self.num_k_heads, self.head_k_dim)
        k = k.reshape(b, s, self.num_k_heads, self.head_k_dim)
        v = v.reshape(b, s, self.num_v_heads, self.head_v_dim)
        if self.n_groups > 1:
            q = q.repeat_interleave(self.n_groups, dim=2)
            k = k.repeat_interleave(self.n_groups, dim=2)

        beta = torch.sigmoid(bd.float())  # [B, S, Hv]
        g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())

        # Lay out for the gated_delta_update op: [B, Hv, S, D] and [B, Hv, S].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2)

        # --- FP32 recurrent SSM state ---
        if state is not None:
            init_state = state.read_recurrent(self.linear_state_idx)
        else:
            init_state = torch.zeros(
                b, self.num_v_heads, self.head_k_dim, self.head_v_dim, dtype=v.dtype, device=v.device
            )

        core_out, new_state = self.gated_delta(
            q.float(), k.float(), v.float(), g, beta, init_state
        )  # [B,S,Hv,Dv],[B,Hv,Dk,Dv]

        if state is not None:
            state.write_recurrent(self.linear_state_idx, new_state)

        z = z.reshape(b, s, self.num_v_heads, self.head_v_dim)
        core_out = self.norm(core_out.to(z.dtype), z)  # gated RMSNorm over head_v_dim
        core_out = core_out.reshape(b, s, self.value_dim)
        return self.out_proj(core_out)


# --------------------------------------------------------------------------- #
# Full (gated) softmax attention
# --------------------------------------------------------------------------- #


class Qwen3_5Attention(nn.Module):
    """GQA softmax attention with an output gate (``attn_output_gate``)."""

    def __init__(self, config, layer_idx: int, full_attention_state_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.full_attention_state_idx = full_attention_state_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.gated = bool(getattr(config, "attn_output_gate", True))

        q_out = self.n_heads * self.head_dim * (2 if self.gated else 1)
        self.q_proj = nn.Linear(dim, q_out, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        rotary_dim -= rotary_dim % 2
        base = float(_rope_params(config).get("rope_theta", getattr(config, "rope_theta", 1e4)))
        self.rope = Qwen3_5RoPE(self.head_dim, rotary_dim, base, _rope_params(config))
        self.sdpa = SDPA(scale=self.head_dim**-0.5, is_causal=True)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        state: Qwen38HybridState | None = None,
    ) -> torch.Tensor:
        b, query_len, _ = x.shape

        qg = self.q_proj(x)
        if self.gated:
            qg = qg.reshape(b, query_len, self.n_heads, self.head_dim * 2)
            q, gate = qg.split(self.head_dim, dim=-1)
            gate = gate.reshape(b, query_len, self.n_heads * self.head_dim)
        else:
            q = qg.reshape(b, query_len, self.n_heads, self.head_dim)
            gate = None
        q = self.q_norm(q).permute(0, 2, 1, 3)  # [B, Hq, S, Dh]
        k = self.k_norm(
            self.k_proj(x).reshape(b, query_len, self.n_kv_heads, self.head_dim)
        ).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, query_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        q = self.rope(q, rope_positions)
        k = self.rope(k, rope_positions)

        if state is not None:
            k, v = state.kv.update_and_fetch(
                self.full_attention_state_idx, offset, k, v, seq_len=seq_len, query_len=query_len
            )

        out = (
            self.sdpa(query=q, key=k, value=v)
            .permute(0, 2, 1, 3)
            .reshape(b, query_len, self.n_heads * self.head_dim)
        )
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Decoder block / model
# --------------------------------------------------------------------------- #


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int, full_attention_state_idx: int, linear_state_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.is_full = _is_full_layer(config, layer_idx)
        if self.is_full:
            self.self_attn = Qwen3_5Attention(config, layer_idx, full_attention_state_idx)
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx, linear_state_idx)
        self.mlp = MLP(hidden, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        state: Qwen38HybridState | None = None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        if self.is_full:
            h = self.self_attn(h, position_ids, state)
        else:
            h = self.linear_attn(h, state)
        x = x + h
        r = self.mlp(self.post_attention_layernorm(x))
        return x + r


class Qwen3_5Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)
        full_attention_state_idx = 0
        linear_state_idx = 0
        layers: list[TransformerBlock] = []
        for i in range(config.num_hidden_layers):
            layers.append(
                TransformerBlock(config, i, full_attention_state_idx, linear_state_idx)
            )
            if _is_full_layer(config, i):
                full_attention_state_idx += 1
            else:
                linear_state_idx += 1
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        state: Qwen38HybridState | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, state)
        return self.norm(h)


# --------------------------------------------------------------------------- #
# ForCausalLM
# --------------------------------------------------------------------------- #


class Qwen3_5ForCausalLM(BaseForCausalLM):
    """Qwen3.5 text decoder (``qwen3_5``).

    Loads from the multimodal ``Qwen3_5ForConditionalGeneration`` checkpoint:
    text weights live under ``model.language_model.`` and ``lm_head.`` (vision
    under ``model.visual.`` is dropped). The registry uses ``hf_state_dict_prefix
    = ""`` (the top-level ``lm_head.weight`` lies outside ``model.``), so the
    whole checkpoint arrives in :meth:`load_state_dict`, which strips vision keys
    and maps ``model.language_model.<...>`` onto this model's ``model.<...>``.

    The checkpoint's MTP sidecar is exported by the companion native MTP author;
    this class exports the four-state target decoder only.
    """

    _HF_MODEL_CLASS = None  # transformers has no Qwen3.5 class; load via safetensors.

    @staticmethod
    def _assign_mmap_safetensors(module: nn.Module, path: str | Path) -> None:
        """Assign an existing safetensors shard without copying its storage into RAM."""
        shard_path = Path(path)
        if not shard_path.is_file():
            raise ValueError(f"Qwen3.8 mmap shard is missing: {shard_path}")
        expected = module.state_dict()
        parameter_names = {name for name, _ in module.named_parameters()}
        assigned: dict[str, torch.Tensor] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in shard.keys():  # noqa: SIM118
                if key not in expected:
                    raise ValueError(f"unexpected tensor {key!r} in Qwen3.8 mmap shard {shard_path}")
                tensor = shard.get_slice(key)[...]
                if tensor.shape != expected[key].shape:
                    raise ValueError(
                        f"Qwen3.8 mmap tensor {key!r} has shape {tuple(tensor.shape)}, "
                        f"expected {tuple(expected[key].shape)}"
                    )
                assigned[key] = (
                    nn.Parameter(tensor, requires_grad=False)
                    if key in parameter_names
                    else tensor
                )
        module.load_state_dict(assigned, assign=True, strict=False)

    @staticmethod
    def _validate_tensor_assignment(
        actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], scope: str
    ) -> None:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        shape_mismatches = sorted(
            (key, tuple(actual[key].shape), tuple(expected[key].shape))
            for key in actual.keys() & expected.keys()
            if actual[key].shape != expected[key].shape
        )
        if missing or unexpected or shape_mismatches:
            raise ValueError(
                f"Qwen3.8 {scope} checkpoint tensors do not match the CoreAI author: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatches={shape_mismatches}"
            )

    @classmethod
    def from_mlx_checkpoint(
        cls,
        model_directory: str | Path,
        *,
        mmap_directory: str | Path,
        target_dtype: torch.dtype = torch.float16,
        num_layers: int | None = None,
    ) -> Self:
        """Stream the packed local checkpoint into validated disk-backed FP weights."""
        from .weights import Qwen38Checkpoint

        source = Path(model_directory)
        mmap_root = Path(mmap_directory)
        if mmap_root.exists():
            raise ValueError(f"refusing to overwrite existing mmap directory: {mmap_root}")

        try:
            raw_config = json.loads((source / "config.json").read_text())
            text_config = raw_config["text_config"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid Qwen3.8 config at {source / 'config.json'}: {error}") from error

        config = Qwen3_5TextConfig(**text_config)
        if num_layers is not None:
            if not 1 <= num_layers <= config.num_hidden_layers:
                raise ValueError(
                    f"num_layers must be between 1 and {config.num_hidden_layers}, got {num_layers}"
                )
            config.num_hidden_layers = num_layers
            if config.layer_types is not None:
                config.layer_types = config.layer_types[:num_layers]
        mmap_root.mkdir(parents=True, exist_ok=False)
        model = cls(config, model_device="meta")
        model.to(dtype=target_dtype)
        checkpoint = Qwen38Checkpoint(source)

        shared = checkpoint.load_shared_state_dict(dtype=target_dtype)
        model._mutate_state_dict(shared)
        expected_shared = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("model.layers.")
        }
        cls._validate_tensor_assignment(shared, expected_shared, "shared")
        _save_and_mmap_safetensors(model, shared, str(mmap_root / "shared.safetensors"))
        logger.info("Loaded Qwen3.8 shared weights into %s", mmap_root)
        del shared, expected_shared
        gc.collect()

        for layer_index in range(config.num_hidden_layers):
            logger.info("Loading Qwen3.8 layer %d/%d", layer_index + 1, config.num_hidden_layers)
            layer_state = checkpoint.load_layer_state_dict(layer_index, dtype=target_dtype)
            model._mutate_state_dict(layer_state)
            prefix = f"model.layers.{layer_index}."
            relative_state = {
                key.removeprefix(prefix): value
                for key, value in layer_state.items()
                if key.startswith(prefix)
            }
            expected_layer = model.model.layers[layer_index].state_dict()
            cls._validate_tensor_assignment(
                relative_state, expected_layer, f"layer {layer_index}"
            )
            _save_and_mmap_safetensors(
                model.model.layers[layer_index],
                relative_state,
                str(mmap_root / f"layer_{layer_index:02d}.safetensors"),
            )
            del layer_state, relative_state, expected_layer
            gc.collect()

        meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if meta_parameters:
            raise RuntimeError(f"Qwen3.8 parameters were not loaded: {meta_parameters}")
        return model

    @classmethod
    def from_mmap_checkpoint(
        cls,
        model_directory: str | Path,
        *,
        mmap_directory: str | Path,
        target_dtype: torch.dtype = torch.float16,
        num_layers: int | None = None,
    ) -> Self:
        """Reopen a validated checkpoint produced by :meth:`from_mlx_checkpoint`."""
        source = Path(model_directory)
        mmap_root = Path(mmap_directory)
        try:
            raw_config = json.loads((source / "config.json").read_text())
            text_config = raw_config["text_config"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid Qwen3.8 config at {source / 'config.json'}: {error}") from error

        config = Qwen3_5TextConfig(**text_config)
        if num_layers is not None:
            if not 1 <= num_layers <= config.num_hidden_layers:
                raise ValueError(
                    f"num_layers must be between 1 and {config.num_hidden_layers}, got {num_layers}"
                )
            config.num_hidden_layers = num_layers
            if config.layer_types is not None:
                config.layer_types = config.layer_types[:num_layers]

        model = cls(config, model_device="meta")
        model.to(dtype=target_dtype)
        cls._assign_mmap_safetensors(model, mmap_root / "shared.safetensors")
        for layer_index in range(config.num_hidden_layers):
            cls._assign_mmap_safetensors(
                model.model.layers[layer_index], mmap_root / f"layer_{layer_index:02d}.safetensors"
            )
        meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if meta_parameters:
            raise RuntimeError(f"Qwen3.8 mmap parameters were not loaded: {meta_parameters}")
        return model

    @override
    def _init_model(self, config) -> None:
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = Qwen38HybridState(k_cache, v_cache, conv_state, recurrent_state)
        out = self.model(input_ids, position_ids, state)
        return self.lm_head(out), out

    @classmethod
    @override
    def export_output_names(cls) -> dict[str, tuple[str, ...]]:
        return {MAIN_GRAPH_NAME: ("logits", "hidden_states")}

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            MAIN_GRAPH_NAME: (
                KEY_CACHE_NAME,
                VALUE_CACHE_NAME,
                "convState",
                "recurrentState",
            )
        }

    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Trace with sixteen dynamic K/V layers and fixed conv/recurrent states."""
        input_ids = torch.randint(1, config.vocab_size, (1, spec.query_len), dtype=torch.int32)
        position_ids = (
            torch.arange(spec.query_len + spec.offset, dtype=torch.int32)
            .unsqueeze(0)
            .expand(1, spec.query_len + spec.offset)
        )
        full_attention_layers = sum(_is_full_layer(config, i) for i in range(config.num_hidden_layers))
        linear_attention_layers = config.num_hidden_layers - full_attention_layers
        k_cache = torch.zeros(
            full_attention_layers,
            1,
            config.num_key_value_heads,
            spec.cache_seq_len,
            config.head_dim,
            dtype=target_dtype,
        )
        v_cache = torch.zeros_like(k_cache)
        conv_dim = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        conv_state = torch.zeros(
            linear_attention_layers,
            1,
            conv_dim,
            config.linear_conv_kernel_dim - 1,
            dtype=target_dtype,
        )
        recurrent_state = torch.zeros(
            linear_attention_layers,
            1,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            dtype=torch.float32,
        )
        return {
            MAIN_GRAPH_NAME: {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "conv_state": conv_state,
                "recurrent_state": recurrent_state,
            }
        }

    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, dict[str, object]]:
        max_context_length = spec.max_context_length
        cache_shape: object
        if spec.caches_are_static:
            cache_shape = None
        else:
            cache_shape = {3: torch.export.Dim("kv_seq_len", min=spec.cache_seq_len, max=max_context_length)}
        return {
            MAIN_GRAPH_NAME: {
                "input_ids": {1: torch.export.Dim("seq_ids", max=max_context_length - 2)},
                "position_ids": {1: torch.export.Dim("seq_pos", min=spec.query_len, max=max_context_length - 1)},
                "k_cache": cache_shape,
                "v_cache": cache_shape,
                "conv_state": None,
                "recurrent_state": None,
            }
        }

    @staticmethod
    def _is_text_key(key: str) -> bool:
        return not (
            key.startswith(("model.visual.", "visual.", "model.mtp", "mtp."))
            or ".visual." in key
        )

    @staticmethod
    def _remap_text_key(key: str) -> str:
        # The local MLX derivative uses ``language_model.*``. Some upstream
        # multimodal checkpoints use ``model.language_model.*`` instead.
        if key.startswith("language_model.model."):
            return "model." + key[len("language_model.model.") :]
        if key.startswith("language_model.lm_head."):
            return "lm_head." + key[len("language_model.lm_head.") :]
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        return key

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Drop vision / MTP keys and map multimodal text keys onto this module's
        # ``model.<...>`` namespace before the memory-efficient loader writes
        # layer-local mmap shards.
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not self._is_text_key(key):
                continue
            key = self._remap_text_key(key)
            if key.endswith(".linear_attn.conv1d.weight"):
                if value.ndim == 2:
                    value = value.unsqueeze(1)
                elif value.ndim == 3 and value.shape[-1] == 1:
                    # MLX Conv1d: [out_channels, kernel, in_channels_per_group].
                    # PyTorch depthwise Conv1d: [out_channels, 1, kernel].
                    value = value.transpose(1, 2).contiguous()
            remapped[key] = value
        state_dict.clear()
        state_dict.update(remapped)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # The full checkpoint arrives here (prefix ""). Keep text keys, strip the
        # ``model.language_model.`` prefix onto ``model.``, and keep top-level
        # ``lm_head.weight`` as-is.
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not self._is_text_key(key):
                continue
            key = self._remap_text_key(key)
            remapped[key] = value
        super().load_state_dict(remapped, strict=strict, assign=assign)
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
