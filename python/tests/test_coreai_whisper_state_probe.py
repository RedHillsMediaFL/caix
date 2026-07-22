from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch


def _state_tensors(probe: object) -> dict[str, torch.Tensor]:
    return {
        "cross_key_cache": torch.zeros(probe.CROSS_CACHE_SHAPE, dtype=torch.float32),
        "cross_value_cache": torch.zeros(probe.CROSS_CACHE_SHAPE, dtype=torch.float32),
        "self_key_cache": torch.zeros(probe.SELF_CACHE_SHAPE, dtype=torch.float32),
        "self_value_cache": torch.zeros(probe.SELF_CACHE_SHAPE, dtype=torch.float32),
        "position": torch.zeros((1,), dtype=torch.int32),
        "cross_ready": torch.zeros((1,), dtype=torch.int32),
    }


def _snapshot(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in state.items()}


def _assert_state_exact(
    state: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    for name, value in expected.items():
        assert torch.equal(state[name], value), name


def test_three_entrypoint_graph_has_the_declared_mutation_contract() -> None:
    probe = importlib.import_module("whisper_large_v2.coreai_state_probe")
    features = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    token = torch.tensor([[0.5]], dtype=torch.float32)
    cross_key_payload, cross_value_payload = probe.ExplicitEncode()(features)
    state = _state_tensors(probe)

    assert probe.mutated_user_inputs(
        probe.ExplicitEncode(),
        {"features": features},
    ) == ()
    assert probe.mutated_user_inputs(
        probe.ExplicitLoadCrossKV(),
        {
            "cross_key_payload": cross_key_payload,
            "cross_value_payload": cross_value_payload,
            "cross_key_cache": state["cross_key_cache"],
            "cross_value_cache": state["cross_value_cache"],
            "cross_ready": state["cross_ready"],
        },
    ) == ("cross_key_cache", "cross_value_cache", "cross_ready")
    assert probe.mutated_user_inputs(
        probe.ExplicitDecodeStep(),
        {"token": token, **state},
    ) == (
        "cross_key_cache",
        "cross_value_cache",
        "self_key_cache",
        "self_value_cache",
        "position",
        "cross_ready",
    )


def test_explicit_torch_session_loads_once_and_indexes_fixed_448_slot_state() -> None:
    probe = importlib.import_module("whisper_large_v2.coreai_state_probe")
    session = probe.ExplicitFallbackSession()
    features = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    token = torch.tensor([[0.5]], dtype=torch.float32)

    cross_key_payload, cross_value_payload = session.encode(features)
    with pytest.raises(probe.StateContractError, match="load_cross_kv"):
        session.decode_step(token)

    load_status = session.load_cross_kv(cross_key_payload, cross_value_payload)
    first, first_status = session.decode_step(token)
    second, second_status = session.decode_step(token)

    torch.testing.assert_close(cross_key_payload.flatten(), torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(cross_value_payload.flatten(), torch.tensor([3.0, 9.0]))
    assert load_status.tolist() == [1]
    assert first_status.tolist() == [1]
    assert second_status.tolist() == [1]
    torch.testing.assert_close(first, torch.tensor([[24.5]]))
    torch.testing.assert_close(second, torch.tensor([[28.5]]))
    assert session.state.self_key_cache.shape == (1, 1, 1, 448, 1)
    assert session.state.self_value_cache.shape == (1, 1, 1, 448, 1)
    torch.testing.assert_close(
        session.state.self_key_cache[..., :2, :].flatten(),
        torch.tensor([1.5, 1.5]),
    )
    torch.testing.assert_close(
        session.state.self_value_cache[..., :2, :].flatten(),
        torch.tensor([2.5, 2.5]),
    )
    assert torch.count_nonzero(session.state.self_key_cache[..., 2:, :]) == 0
    assert torch.count_nonzero(session.state.self_value_cache[..., 2:, :]) == 0
    assert session.state.position.tolist() == [2]
    assert session.state.cross_ready.tolist() == [1]
    assert session.load_cross_kv_calls == 1

    with pytest.raises(probe.StateContractError, match="exactly once"):
        session.load_cross_kv(cross_key_payload, cross_value_payload)


def test_explicit_graph_rejects_invalid_calls_without_mutating_any_state() -> None:
    probe = importlib.import_module("whisper_large_v2.coreai_state_probe")
    features = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    token = torch.tensor([[0.5]], dtype=torch.float32)
    payloads = probe.ExplicitEncode()(features)

    load_state = _state_tensors(probe)
    loader = probe.ExplicitLoadCrossKV()
    load_inputs = {
        "cross_key_cache": load_state["cross_key_cache"],
        "cross_value_cache": load_state["cross_value_cache"],
        "cross_ready": load_state["cross_ready"],
    }
    assert loader(*payloads, **load_inputs).tolist() == [1]
    load_before = _snapshot(load_state)
    second_status = loader(*payloads, **load_inputs)
    assert second_status.tolist() == [0]
    _assert_state_exact(load_state, load_before)

    for readiness, position in ((-1, 0), (0, 0), (2, 0), (1, -1), (1, 448)):
        state = _state_tensors(probe)
        state["cross_key_cache"].uniform_(-2, 2)
        state["cross_value_cache"].uniform_(-2, 2)
        state["self_key_cache"].uniform_(-2, 2)
        state["self_value_cache"].uniform_(-2, 2)
        state["cross_ready"].fill_(readiness)
        state["position"].fill_(position)
        before = _snapshot(state)

        logits, status = probe.ExplicitDecodeStep()(token, **state)

        assert status.tolist() == [0]
        assert torch.count_nonzero(logits) == 0
        _assert_state_exact(state, before)


def test_explicit_probe_session_resets_and_rejects_448_token_overflow() -> None:
    probe = importlib.import_module("whisper_large_v2.coreai_state_probe")
    session = probe.ExplicitFallbackSession()
    features = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    token = torch.tensor([[0.5]], dtype=torch.float32)
    payloads = session.encode(features)
    session.load_cross_kv(*payloads)
    session.state.position.fill_(447)

    _, status = session.decode_step(token)

    assert status.tolist() == [1]
    assert session.state.position.tolist() == [448]
    assert torch.count_nonzero(session.state.self_key_cache[..., 447:448, :]) == 1
    with pytest.raises(probe.StateContractError, match="448-token capacity"):
        session.decode_step(token)

    session.reset()

    assert session.load_cross_kv_calls == 0
    assert session.state.position.tolist() == [0]
    assert session.state.cross_ready.tolist() == [0]
    assert torch.count_nonzero(session.state.cross_key_cache) == 0
    assert torch.count_nonzero(session.state.cross_value_cache) == 0
    assert torch.count_nonzero(session.state.self_key_cache) == 0
    assert torch.count_nonzero(session.state.self_value_cache) == 0


@pytest.mark.skipif(
    "CAIX_RUN_COREAI_STATE_PROOF" not in os.environ,
    reason="set CAIX_RUN_COREAI_STATE_PROOF=1 to compile and execute the tiny AIProgram",
)
def test_real_coreai_executes_explicit_three_entrypoint_fixed_cache_abi() -> None:
    temp_root = Path(os.environ.get("CAIX_COREAI_PROBE_TMP_ROOT", "/Volumes/SSD/caix/.tmp"))
    parent_owned = Path(
        tempfile.mkdtemp(prefix="whisper-coreai-parent-", dir=temp_root)
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "whisper_large_v2.coreai_state_probe",
                "--temp-root",
                str(parent_owned),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=os.environ.copy(),
        )
    finally:
        shutil.rmtree(parent_owned, ignore_errors=True)
    assert not parent_owned.exists()

    compiler_output = completed.stdout + completed.stderr
    if completed.returncode != 0 and all(
        marker in compiler_output
        for marker in ("expected AICode versioned location", "Failed to convert to versioned IR")
    ):
        pytest.xfail(
            "installed CoreAI compiler aborts during versioned-IR conversion; "
            "Apple's own single-entrypoint stateful control test aborts on the same stack"
        )

    assert completed.returncode == 0, compiler_output
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "call_order": ["encode", "load_cross_kv", "decode_step", "decode_step"],
        "cross_key_state": [2.0, 6.0],
        "cross_ready": [1],
        "cross_value_state": [3.0, 9.0],
        "decode_outputs": [[[24.5]], [[28.5]]],
        "decode_statuses": [[1], [1]],
        "encode_cross_keys": [2.0, 6.0],
        "encode_cross_values": [3.0, 9.0],
        "explicit_bridge_supported": True,
        "load_cross_kv_calls": 1,
        "invalid_decode_before_load_state_unchanged": True,
        "invalid_decode_before_load_status": [0],
        "invalid_decode_before_load_zero_logits": True,
        "invalid_position_state_unchanged": [True, True, True, True],
        "invalid_position_statuses": [[0], [0], [0], [0]],
        "invalid_second_load_state_unchanged": True,
        "invalid_second_load_status": [0],
        "load_status": [1],
        "position": [2],
        "self_key_prefix": [1.5, 1.5],
        "self_tail_nonzero": 0,
        "self_value_prefix": [2.5, 2.5],
    }
