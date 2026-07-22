"""Executable proof for the explicit three-entrypoint native Whisper state ABI."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from coreai_models.primitives._ops import mutable_slice_update
from whisper_large_v2.reference import WHISPER_DECODER_CACHE_CAPACITY

CROSS_CACHE_SHAPE = (1, 1, 1, 2, 1)
SELF_CACHE_SHAPE = (1, 1, 1, WHISPER_DECODER_CACHE_CAPACITY, 1)


class ExplicitEncode(nn.Module):
    """Emit cross K/V payloads as ordinary outputs, without decoder state."""

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = CROSS_CACHE_SHAPE
        return (features * 2.0).reshape(shape), (features * 3.0).reshape(shape)


class ExplicitLoadCrossKV(nn.Module):
    """Load encoder outputs into zeroed decoder-owned cross state and mark it ready."""

    def forward(
        self,
        cross_key_payload: torch.Tensor,
        cross_value_payload: torch.Tensor,
        cross_key_cache: torch.Tensor,
        cross_value_cache: torch.Tensor,
        cross_ready: torch.Tensor,
    ) -> torch.Tensor:
        cross_key_cache.add_(cross_key_payload)
        cross_value_cache.add_(cross_value_payload)
        cross_ready.add_(1)
        return (
            cross_key_payload[..., 0, :].reshape(1, 1)
            + cross_value_payload[..., 0, :].reshape(1, 1)
        )


class ExplicitDecodeStep(nn.Module):
    """Write one token into fixed self K/V state and advance tensor position."""

    def forward(
        self,
        token: torch.Tensor,
        cross_key_cache: torch.Tensor,
        cross_value_cache: torch.Tensor,
        self_key_cache: torch.Tensor,
        self_value_cache: torch.Tensor,
        position: torch.Tensor,
        cross_ready: torch.Tensor,
    ) -> torch.Tensor:
        cross_key_cache.add_(0.0)
        cross_value_cache.add_(0.0)
        new_key = (token + 1.0).reshape((1, 1, 1, 1, 1))
        new_value = (token + 2.0).reshape((1, 1, 1, 1, 1))
        zero = torch.zeros((1,), dtype=torch.int32, device=token.device)
        one = torch.ones((1,), dtype=torch.int32, device=token.device)
        begin = torch.cat((zero, zero, zero, position, zero))
        end = torch.cat((one, one, one, position + one, one))
        mutable_slice_update(
            x=self_key_cache,
            update=new_key,
            begin=begin,
            end=end,
        )
        mutable_slice_update(
            x=self_value_cache,
            update=new_value,
            begin=begin,
            end=end,
        )
        ready = cross_ready.to(dtype=token.dtype).reshape((1, 1))
        output = ready * (
            token
            + cross_key_cache.sum().reshape((1, 1))
            + cross_value_cache.sum().reshape((1, 1))
            + self_key_cache.sum().reshape((1, 1))
            + self_value_cache.sum().reshape((1, 1))
        )
        position.add_(1)
        cross_ready.add_(0)
        return output


class StateContractError(RuntimeError):
    """A caller violated the explicit decoder-session state machine."""


@dataclass
class ProbeDecoderState:
    cross_key_cache: torch.Tensor
    cross_value_cache: torch.Tensor
    self_key_cache: torch.Tensor
    self_value_cache: torch.Tensor
    position: torch.Tensor
    cross_ready: torch.Tensor

    @classmethod
    def zeros(cls) -> ProbeDecoderState:
        return cls(
            cross_key_cache=torch.zeros(CROSS_CACHE_SHAPE, dtype=torch.float32),
            cross_value_cache=torch.zeros(CROSS_CACHE_SHAPE, dtype=torch.float32),
            self_key_cache=torch.zeros(SELF_CACHE_SHAPE, dtype=torch.float32),
            self_value_cache=torch.zeros(SELF_CACHE_SHAPE, dtype=torch.float32),
            position=torch.zeros((1,), dtype=torch.int32),
            cross_ready=torch.zeros((1,), dtype=torch.int32),
        )

    def reset(self) -> None:
        self.cross_key_cache.zero_()
        self.cross_value_cache.zero_()
        self.self_key_cache.zero_()
        self.self_value_cache.zero_()
        self.position.zero_()
        self.cross_ready.zero_()


class ExplicitFallbackSession:
    """Host state machine for encode -> load_cross_kv once -> decode_step."""

    def __init__(self) -> None:
        self._encoder = ExplicitEncode()
        self._loader = ExplicitLoadCrossKV()
        self._decoder = ExplicitDecodeStep()
        self.state = ProbeDecoderState.zeros()
        self.load_cross_kv_calls = 0

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encoder(features)

    def load_cross_kv(
        self,
        cross_key_payload: torch.Tensor,
        cross_value_payload: torch.Tensor,
    ) -> torch.Tensor:
        if self.load_cross_kv_calls:
            raise StateContractError("load_cross_kv must run exactly once per utterance")
        if int(self.state.position.item()) != 0:
            raise StateContractError("decoder state must be reset before load_cross_kv")
        marker = self._loader(
            cross_key_payload,
            cross_value_payload,
            self.state.cross_key_cache,
            self.state.cross_value_cache,
            self.state.cross_ready,
        )
        self.load_cross_kv_calls += 1
        return marker

    def decode_step(self, token: torch.Tensor) -> torch.Tensor:
        if self.load_cross_kv_calls != 1 or self.state.cross_ready.tolist() != [1]:
            raise StateContractError("load_cross_kv must run before decode_step")
        if int(self.state.position.item()) >= WHISPER_DECODER_CACHE_CAPACITY:
            raise StateContractError("decoder state exceeds 448-token capacity")
        return self._decoder(
            token,
            self.state.cross_key_cache,
            self.state.cross_value_cache,
            self.state.self_key_cache,
            self.state.self_value_cache,
            self.state.position,
            self.state.cross_ready,
        )

    def reset(self) -> None:
        self.state.reset()
        self.load_cross_kv_calls = 0


def mutated_user_inputs(
    module: nn.Module,
    inputs: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Expose torch.export's ordered state classification for contract tests."""
    import coreai_torch

    exported = torch.export.export(module.eval(), args=(), kwargs=dict(inputs))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    return tuple(exported.graph_signature.user_inputs_to_mutate.values())


@dataclass(frozen=True)
class CoreAIStateProbeResult:
    explicit_bridge_supported: bool
    call_order: tuple[str, ...]
    load_cross_kv_calls: int
    encode_cross_keys: list[float]
    encode_cross_values: list[float]
    load_marker: list[list[float]]
    decode_outputs: list[list[list[float]]]
    cross_key_state: list[float]
    cross_value_state: list[float]
    self_key_prefix: list[float]
    self_value_prefix: list[float]
    self_tail_nonzero: int
    position: list[int]
    cross_ready: list[int]


def run_coreai_state_probe(temp_root: Path) -> CoreAIStateProbeResult:
    """Compile and execute all three ABI entrypoints under a parent-owned root."""
    if not temp_root.is_dir():
        raise ValueError(f"CoreAI probe temp root is not a directory: {temp_root}")
    return asyncio.run(_run_coreai_state_probe(temp_root))


async def _run_coreai_state_probe(temp_root: Path) -> CoreAIStateProbeResult:
    import coreai_torch
    from coreai.runtime import NDArray
    from coreai_torch import TorchConverter

    from coreai_models.export.mlir_ops import (
        register_custom_torch_lowering,
        remove_functionalization,
    )

    features = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    token = torch.tensor([[0.5]], dtype=torch.float32)
    payloads = ExplicitEncode()(features)
    state = ProbeDecoderState.zeros()

    def export(module: nn.Module, inputs: Mapping[str, torch.Tensor]) -> Any:
        exported = torch.export.export(module.eval(), args=(), kwargs=dict(inputs))
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        remove_functionalization(exported)
        return exported

    converter = TorchConverter()
    converter.add_exported_program(
        export(ExplicitEncode(), {"features": features}),
        input_names=("features",),
        output_names=("cross_key_payload", "cross_value_payload"),
        state_names=(),
        entrypoint_name="encode",
    )
    converter.add_exported_program(
        export(
            ExplicitLoadCrossKV(),
            {
                "cross_key_payload": payloads[0],
                "cross_value_payload": payloads[1],
                "cross_key_cache": state.cross_key_cache,
                "cross_value_cache": state.cross_value_cache,
                "cross_ready": state.cross_ready,
            },
        ),
        input_names=("cross_key_payload", "cross_value_payload"),
        output_names=("load_marker",),
        state_names=("cross_key_cache", "cross_value_cache", "cross_ready"),
        entrypoint_name="load_cross_kv",
    )
    converter.add_exported_program(
        export(
            ExplicitDecodeStep(),
            {
                "token": token,
                "cross_key_cache": state.cross_key_cache,
                "cross_value_cache": state.cross_value_cache,
                "self_key_cache": state.self_key_cache,
                "self_value_cache": state.self_value_cache,
                "position": state.position,
                "cross_ready": state.cross_ready,
            },
        ),
        input_names=("token",),
        output_names=("logits",),
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
    program.optimize()

    with tempfile.TemporaryDirectory(
        prefix="explicit-bridge-",
        suffix=".aimodel",
        dir=temp_root,
    ) as temp_dir:
        asset = program.save_asset(Path(temp_dir))
        async with asset.executable() as executable:
            encode = executable.load_function("encode")
            load_cross_kv = executable.load_function("load_cross_kv")
            decode_step = executable.load_function("decode_step")
            runtime_state = {
                "cross_key_cache": NDArray(data=state.cross_key_cache),
                "cross_value_cache": NDArray(data=state.cross_value_cache),
                "self_key_cache": NDArray(data=state.self_key_cache),
                "self_value_cache": NDArray(data=state.self_value_cache),
                "position": NDArray(data=state.position),
                "cross_ready": NDArray(data=state.cross_ready),
            }
            encoded = await encode({"features": NDArray(data=features)})
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
            first = await decode_step(
                {"token": NDArray(data=token)},
                state=runtime_state,
            )
            second = await decode_step(
                {"token": NDArray(data=token)},
                state=runtime_state,
            )
            self_keys = runtime_state["self_key_cache"].numpy()
            self_values = runtime_state["self_value_cache"].numpy()
            return CoreAIStateProbeResult(
                explicit_bridge_supported=True,
                call_order=("encode", "load_cross_kv", "decode_step", "decode_step"),
                load_cross_kv_calls=1,
                encode_cross_keys=encoded["cross_key_payload"].numpy().reshape(-1).tolist(),
                encode_cross_values=encoded["cross_value_payload"].numpy().reshape(-1).tolist(),
                load_marker=loaded["load_marker"].numpy().tolist(),
                decode_outputs=[
                    first["logits"].numpy().tolist(),
                    second["logits"].numpy().tolist(),
                ],
                cross_key_state=runtime_state["cross_key_cache"]
                .numpy()
                .reshape(-1)
                .tolist(),
                cross_value_state=runtime_state["cross_value_cache"]
                .numpy()
                .reshape(-1)
                .tolist(),
                self_key_prefix=self_keys[..., :2, :].reshape(-1).tolist(),
                self_value_prefix=self_values[..., :2, :].reshape(-1).tolist(),
                self_tail_nonzero=int((self_keys[..., 2:, :] != 0).sum()),
                position=runtime_state["position"].numpy().reshape(-1).tolist(),
                cross_ready=runtime_state["cross_ready"].numpy().reshape(-1).tolist(),
            )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_coreai_state_probe(args.temp_root)
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
