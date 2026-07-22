from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_shared_state_graph_marks_cross_state_in_both_entrypoints() -> None:
    probe = importlib.import_module("whisper_large_v2.coreai_state_probe")
    features = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    token = torch.tensor([[0.5]], dtype=torch.float32)
    cross_state = torch.zeros((1, 2), dtype=torch.float32)
    self_state = torch.zeros((1, 2), dtype=torch.float32)

    assert probe.mutated_user_inputs(
        probe.SharedStateEncode(),
        {"features": features, "cross_state": cross_state},
    ) == ("cross_state",)
    assert probe.mutated_user_inputs(
        probe.SharedStateDecodeStep(),
        {"token": token, "cross_state": cross_state, "self_state": self_state},
    ) == ("cross_state", "self_state")


def test_torch_shared_state_semantics_are_deterministic() -> None:
    probe = importlib.import_module("whisper_large_v2.coreai_state_probe")
    encode = probe.SharedStateEncode()
    decode = probe.SharedStateDecodeStep()
    features = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    token = torch.tensor([[0.5]], dtype=torch.float32)
    cross_state = torch.zeros((1, 2), dtype=torch.float32)
    self_state = torch.zeros((1, 2), dtype=torch.float32)

    marker = encode(features, cross_state)
    first = decode(token, cross_state, self_state)
    second = decode(token, cross_state, self_state)

    torch.testing.assert_close(marker, torch.tensor([[4.0]]))
    torch.testing.assert_close(cross_state, torch.tensor([[2.0, 6.0]]))
    torch.testing.assert_close(first, torch.tensor([[2.5]]))
    torch.testing.assert_close(second, torch.tensor([[3.0]]))
    torch.testing.assert_close(self_state, torch.tensor([[1.0, 1.0]]))


def test_explicit_fallback_loads_encoder_output_into_decoder_state_once() -> None:
    probe = importlib.import_module("whisper_large_v2.coreai_state_probe")
    session = probe.ExplicitFallbackSession()
    features = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    token = torch.tensor([[0.5]], dtype=torch.float32)

    payload = session.encode(features)
    with pytest.raises(probe.StateContractError, match="load_cross_kv"):
        session.decode_step(token)

    session.load_cross_kv(payload)
    first = session.decode_step(token)
    second = session.decode_step(token)

    torch.testing.assert_close(payload, torch.tensor([[2.0, 6.0]]))
    torch.testing.assert_close(first, torch.tensor([[2.5]]))
    torch.testing.assert_close(second, torch.tensor([[3.0]]))
    assert session.load_cross_kv_calls == 1
    with pytest.raises(probe.StateContractError, match="exactly once"):
        session.load_cross_kv(payload)


@pytest.mark.skipif(
    "CAIX_RUN_COREAI_STATE_PROOF" not in os.environ,
    reason="set CAIX_RUN_COREAI_STATE_PROOF=1 to compile and execute the tiny AIProgram",
)
def test_real_coreai_entrypoints_share_cross_and_self_state() -> None:
    temp_root = Path(os.environ.get("CAIX_COREAI_PROBE_TMP_ROOT", "/Volumes/SSD/caix/.tmp"))
    before = set(temp_root.glob("whisper-coreai-state-*"))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "whisper_large_v2.coreai_state_probe",
            "--temp-root",
            str(temp_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    for created in set(temp_root.glob("whisper-coreai-state-*")) - before:
        shutil.rmtree(created, ignore_errors=True)
    assert set(temp_root.glob("whisper-coreai-state-*")) == before

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
        "cross_after_decode": [[2.0, 6.0]],
        "decode_outputs": [[[2.5]], [[3.0]]],
        "encode_output": [[4.0]],
        "self_after_decode": [[1.0, 1.0]],
        "shared_state_supported": True,
    }
