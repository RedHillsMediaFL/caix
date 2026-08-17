"""Native Qwen3.8 multi-token-prediction sidecar author for CoreAI."""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
from coreai_models._constants import KEY_CACHE_NAME, MAIN_GRAPH_NAME, VALUE_CACHE_NAME
from coreai_models.models.base import BaseForCausalLM, TraceSpec, _save_and_mmap_safetensors
from coreai_models.primitives.macos.cache import KVCache
from safetensors import safe_open
from typing_extensions import Self, override

from .model import Qwen3_5RMSNorm, Qwen3_5TextConfig, TransformerBlock
from .weights import Qwen38Checkpoint


class _MTPState:
    """The MTP layer is full attention, so its persistent state is K/V only."""

    def __init__(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        self.kv = KVCache(key_cache, value_cache)


class Qwen3_5MTPForCausalLM(BaseForCausalLM):
    """One-layer Qwen3.8 MTP decoder conditioned on target hidden states.

    The model mirrors MLX-VLM's Qwen3_5MTPDraftModel: normalize token embeddings
    and target hidden states, concatenate, project to the target width, run one
    forced full-attention decoder layer, normalize, and project with the target
    LM head. Its own one-layer K/V cache is independent from the target's hybrid
    four-state store.
    """

    _HF_MODEL_CLASS = None

    @override
    def _init_model(self, config) -> None:
        layer_config = copy.deepcopy(config)
        layer_config.num_hidden_layers = 1
        layer_config.full_attention_interval = 1
        layer_config.layer_types = ["full_attention"]
        hidden = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.layer = TransformerBlock(layer_config, 0, 0, 0)
        self.norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(hidden, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_hidden = self.pre_fc_norm_embedding(self.embed_tokens(input_ids))
        target_hidden = self.pre_fc_norm_hidden(hidden_states)
        h = self.fc(torch.cat((token_hidden, target_hidden), dim=-1))
        h = self.layer(h, position_ids, _MTPState(k_cache, v_cache))
        h = self.norm(h)
        # The full sequence has already updated K/V. Only its final row can seed the next draft;
        # projecting every prompt row through the 248K vocabulary wastes most MTP prefill time.
        last = h[:, -1:, :]
        return self.lm_head(last), last

    @classmethod
    @override
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        return {MAIN_GRAPH_NAME: ("input_ids", "hidden_states", "position_ids")}

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        return {MAIN_GRAPH_NAME: (KEY_CACHE_NAME, VALUE_CACHE_NAME)}

    @classmethod
    @override
    def export_output_names(cls) -> dict[str, tuple[str, ...]]:
        return {MAIN_GRAPH_NAME: ("logits", "mtp_hidden_states")}

    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict[str, torch.Tensor]]:
        input_ids = torch.randint(1, config.vocab_size, (1, spec.query_len), dtype=torch.int32)
        hidden_states = torch.zeros(
            1, spec.query_len, config.hidden_size, dtype=target_dtype
        )
        position_ids = torch.arange(
            spec.query_len + spec.offset, dtype=torch.int32
        ).unsqueeze(0)
        cache_shape = (1, 1, config.num_key_value_heads, spec.cache_seq_len, config.head_dim)
        k_cache = torch.zeros(cache_shape, dtype=target_dtype)
        v_cache = torch.zeros_like(k_cache)
        return {
            MAIN_GRAPH_NAME: {
                "input_ids": input_ids,
                "hidden_states": hidden_states,
                "position_ids": position_ids,
                "k_cache": k_cache,
                "v_cache": v_cache,
            }
        }

    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict[str, dict[str, object]]:
        seq = torch.export.Dim("mtp_seq", max=spec.max_context_length - 2)
        positions = torch.export.Dim(
            "mtp_positions", min=spec.query_len, max=spec.max_context_length - 1
        )
        cache: object = None
        if not spec.caches_are_static:
            cache = {
                3: torch.export.Dim(
                    "mtp_cache", min=spec.cache_seq_len, max=spec.max_context_length
                )
            }
        return {
            MAIN_GRAPH_NAME: {
                "input_ids": {1: seq},
                "hidden_states": {1: seq},
                "position_ids": {1: positions},
                "k_cache": cache,
                "v_cache": cache,
            }
        }

    @staticmethod
    def _remap_checkpoint_state(state_dict: dict[str, torch.Tensor]) -> None:
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("language_model.mtp.layers.0."):
                key = "layer." + key[len("language_model.mtp.layers.0.") :]
            elif key.startswith("language_model.mtp."):
                key = key[len("language_model.mtp.") :]
            elif key == "language_model.model.embed_tokens.weight":
                key = "embed_tokens.weight"
            elif key == "language_model.lm_head.weight":
                key = "lm_head.weight"
            else:
                continue
            remapped[key] = value
        state_dict.clear()
        state_dict.update(remapped)

    @override
    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._remap_checkpoint_state(state_dict)

    @staticmethod
    def _assign_mmap(module: nn.Module, path: Path) -> None:
        expected = module.state_dict()
        parameters = {name for name, _ in module.named_parameters()}
        assigned: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as shard:
            for key in shard.keys():  # noqa: SIM118
                if key not in expected or shard.get_slice(key).get_shape() != list(expected[key].shape):
                    raise ValueError(f"invalid Qwen3.8 MTP mmap tensor {key!r} in {path}")
                tensor = shard.get_slice(key)[...]
                assigned[key] = (
                    nn.Parameter(tensor, requires_grad=False) if key in parameters else tensor
                )
        module.load_state_dict(assigned, assign=True, strict=True)

    @classmethod
    def from_mlx_checkpoint(
        cls,
        model_directory: str | Path,
        *,
        mmap_directory: str | Path,
        target_dtype: torch.dtype = torch.float16,
    ) -> Self:
        source = Path(model_directory)
        mmap_root = Path(mmap_directory)
        if mmap_root.exists():
            raise ValueError(f"refusing to overwrite existing MTP mmap directory: {mmap_root}")
        raw = json.loads((source / "config.json").read_text())
        config = Qwen3_5TextConfig(**raw["text_config"])
        model = cls(config, model_device="meta").to(dtype=target_dtype)
        checkpoint = Qwen38Checkpoint(source)

        state = checkpoint.load_mtp_state_dict(dtype=target_dtype)
        shared = checkpoint.load_shared_state_dict(dtype=target_dtype)
        for key in (
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
        ):
            if key not in shared:
                raise ValueError(f"Qwen3.8 MTP source is missing shared tensor {key!r}")
            state[key] = shared[key]
        del shared
        cls._remap_checkpoint_state(state)
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError(
                "Qwen3.8 MTP tensors do not match author: "
                f"missing={sorted(set(expected) - set(state))}, "
                f"unexpected={sorted(set(state) - set(expected))}"
            )
        mismatches = [
            key for key in state if state[key].shape != expected[key].shape
        ]
        if mismatches:
            raise ValueError(f"Qwen3.8 MTP tensor shape mismatches: {mismatches}")
        mmap_root.mkdir(parents=True, exist_ok=False)
        _save_and_mmap_safetensors(model, state, str(mmap_root / "mtp.safetensors"))
        del state, expected
        gc.collect()
        return model

    @classmethod
    def from_mmap_checkpoint(
        cls,
        model_directory: str | Path,
        *,
        mmap_directory: str | Path,
        target_dtype: torch.dtype = torch.float16,
    ) -> Self:
        source = Path(model_directory)
        raw = json.loads((source / "config.json").read_text())
        config = Qwen3_5TextConfig(**raw["text_config"])
        model = cls(config, model_device="meta").to(dtype=target_dtype)
        cls._assign_mmap(model, Path(mmap_directory) / "mtp.safetensors")
        return model
