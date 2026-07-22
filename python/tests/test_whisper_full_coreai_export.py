from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    "CAIX_RUN_WHISPER_FULL_EXPORT_PROOF" not in os.environ,
    reason="set CAIX_RUN_WHISPER_FULL_EXPORT_PROOF=1 to compile the split Whisper graph",
)
def test_minimal_split_whisper_compiles_and_executes_all_three_entrypoints() -> None:
    temp_root = Path(os.environ.get("CAIX_COREAI_PROBE_TMP_ROOT", "/Volumes/SSD/caix/.tmp"))
    parent_owned = Path(tempfile.mkdtemp(prefix="whisper-full-parent-", dir=temp_root))
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "whisper_large_v2.export",
                "--minimal-coreai-proof",
                "--temp-root",
                str(parent_owned),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
            env=os.environ.copy(),
        )
    finally:
        shutil.rmtree(parent_owned, ignore_errors=True)

    assert not parent_owned.exists()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["call_order"] == ["encode", "load_cross_kv", "decode_step", "decode_step"]
    assert result["entrypoints"] == ["decode_step", "encode", "load_cross_kv"]
    assert result["position"] == [2]
    assert result["cross_ready"] == [1]
    assert result["self_key_tail_nonzero"] == 0
    assert result["self_value_tail_nonzero"] == 0
    assert result["max_encode_key_error"] < 1e-4
    assert result["max_encode_value_error"] < 1e-4
    assert result["max_decode_error"] < 1e-4
