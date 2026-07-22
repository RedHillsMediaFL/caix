from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    "CAIX_WHISPER_FULL_ASSET" not in os.environ,
    reason="set CAIX_WHISPER_FULL_ASSET to execute the complete native artifact",
)
def test_full_asset_loads_and_runs_the_three_entrypoint_state_machine() -> None:
    asset = Path(os.environ["CAIX_WHISPER_FULL_ASSET"])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "whisper_large_v2.verify",
            "--asset",
            str(asset),
            "--max-resident-gib",
            "42",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
        env=os.environ.copy(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["entrypoints"] == ["decode_step", "encode", "load_cross_kv"]
    assert result["cross_key_shape"] == [32, 1, 20, 1500, 64]
    assert result["cross_value_shape"] == [32, 1, 20, 1500, 64]
    assert result["logits_shape"] == [1, 1, 51_865]
    assert result["position"] == [1]
    assert result["cross_ready"] == [1]
    assert result["self_key_tail_nonzero"] == 0
    assert result["self_value_tail_nonzero"] == 0
    assert result["logits_all_finite"] is True
    assert result["peak_resident_bytes"] < 42 * 1024**3
