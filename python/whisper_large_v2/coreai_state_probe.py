"""Executable proof that CoreAI entrypoints can share caller-owned mutable state."""

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


class SharedStateEncode(nn.Module):
    """Stand-in encoder that writes deterministic cross-attention state."""

    def forward(
        self,
        features: torch.Tensor,
        cross_state: torch.Tensor,
    ) -> torch.Tensor:
        cross_state.copy_(features * 2.0)
        return features[:, :1] + features[:, 1:2]


class SharedStateDecodeStep(nn.Module):
    """Stand-in decoder that reads cross state and grows self state."""

    def forward(
        self,
        token: torch.Tensor,
        cross_state: torch.Tensor,
        self_state: torch.Tensor,
    ) -> torch.Tensor:
        output = token + cross_state[:, :1] + self_state[:, :1]
        # coreai-torch exposes only mutated user inputs through state_names. The
        # identity write makes immutable cross K/V addressable by both entrypoints.
        cross_state.add_(0.0)
        self_state.add_(token.expand_as(self_state))
        return output


class FallbackEncode(nn.Module):
    """Encoder half of the explicit bridge; cross K/V remain normal outputs."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * 2.0


class FallbackLoadCrossKV(nn.Module):
    """One-time decoder entrypoint that copies encoder output into decoder state."""

    def forward(
        self,
        cross_payload: torch.Tensor,
        cross_state: torch.Tensor,
    ) -> torch.Tensor:
        cross_state.copy_(cross_payload)
        return cross_payload[:, :1]


class FallbackDecodeStep(SharedStateDecodeStep):
    """Stateful decoder used after the one-time explicit cross-KV load."""


class StateContractError(RuntimeError):
    """A caller violated the load-once decoder session sequence."""


class ExplicitFallbackSession:
    """Tiny host-side proof of encode -> load_cross_kv once -> decode_step."""

    def __init__(self) -> None:
        self._encoder = FallbackEncode()
        self._loader = FallbackLoadCrossKV()
        self._decoder = FallbackDecodeStep()
        self._cross_state: torch.Tensor | None = None
        self._self_state: torch.Tensor | None = None
        self.load_cross_kv_calls = 0

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        payload = self._encoder(features)
        self._cross_state = torch.zeros_like(payload)
        self._self_state = torch.zeros_like(payload)
        return payload

    def load_cross_kv(self, payload: torch.Tensor) -> torch.Tensor:
        if self.load_cross_kv_calls:
            raise StateContractError("load_cross_kv must run exactly once per utterance")
        if self._cross_state is None:
            raise StateContractError("encode must run before load_cross_kv")
        marker = self._loader(payload, self._cross_state)
        self.load_cross_kv_calls += 1
        return marker

    def decode_step(self, token: torch.Tensor) -> torch.Tensor:
        if self.load_cross_kv_calls != 1:
            raise StateContractError("load_cross_kv must run before decode_step")
        assert self._cross_state is not None
        assert self._self_state is not None
        return self._decoder(token, self._cross_state, self._self_state)


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
    shared_state_supported: bool
    encode_output: list[list[float]]
    decode_outputs: list[list[list[float]]]
    cross_after_decode: list[list[float]]
    self_after_decode: list[list[float]]


def run_coreai_state_probe(temp_root: Path) -> CoreAIStateProbeResult:
    """Compile and execute the proof, always deleting the generated AIModel."""
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
    cross_state = torch.zeros((1, 2), dtype=torch.float32)
    self_state = torch.zeros((1, 2), dtype=torch.float32)

    def export(module: nn.Module, inputs: Mapping[str, torch.Tensor]) -> Any:
        exported = torch.export.export(module.eval(), args=(), kwargs=dict(inputs))
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        remove_functionalization(exported)
        return exported

    converter = TorchConverter()
    converter.add_exported_program(
        export(
            SharedStateEncode(),
            {"features": features, "cross_state": cross_state},
        ),
        input_names=("features",),
        output_names=("encode_marker",),
        state_names=("cross_state",),
        entrypoint_name="encode",
    )
    converter.add_exported_program(
        export(
            SharedStateDecodeStep(),
            {
                "token": token,
                "cross_state": cross_state,
                "self_state": self_state,
            },
        ),
        input_names=("token",),
        output_names=("logits",),
        state_names=("cross_state", "self_state"),
        entrypoint_name="decode_step",
    )
    register_custom_torch_lowering(converter)
    program = converter.to_coreai()
    program.optimize()

    with tempfile.TemporaryDirectory(
        prefix="whisper-coreai-state-",
        suffix=".aimodel",
        dir=temp_root,
    ) as temp_dir:
        asset = program.save_asset(Path(temp_dir))
        async with asset.executable() as executable:
            encode = executable.load_function("encode")
            decode = executable.load_function("decode_step")
            state = {
                "cross_state": NDArray(data=torch.zeros((1, 2), dtype=torch.float32)),
                "self_state": NDArray(data=torch.zeros((1, 2), dtype=torch.float32)),
            }
            encoded = await encode(
                {"features": NDArray(data=features)},
                state={"cross_state": state["cross_state"]},
            )
            first = await decode({"token": NDArray(data=token)}, state=state)
            second = await decode({"token": NDArray(data=token)}, state=state)
            return CoreAIStateProbeResult(
                shared_state_supported=True,
                encode_output=encoded["encode_marker"].numpy().tolist(),
                decode_outputs=[
                    first["logits"].numpy().tolist(),
                    second["logits"].numpy().tolist(),
                ],
                cross_after_decode=state["cross_state"].numpy().tolist(),
                self_after_decode=state["self_state"].numpy().tolist(),
            )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_coreai_state_probe(args.temp_root)
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
