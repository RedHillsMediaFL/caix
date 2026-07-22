from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONTRACT = REPOSITORY_ROOT / "models" / "whisper-large-v2-source.json"
PINNED_SNAPSHOT = Path(
    "/Volumes/SSD/hf-cache/models--openai--whisper-large-v2/snapshots/"
    "ae4642769ce2ad8fc292556ccea8e901f1530655"
)


def test_large_v2_weight_plan_is_exact_and_bounds_conversion_memory() -> None:
    convert = importlib.import_module("whisper_large_v2.convert")

    plan = convert.WhisperWeightPlan.large_v2()

    assert len(plan.encoder_keys) == 583
    assert len(plan.decoder_keys) == 676
    assert plan.tensor_count == 1_259
    assert set(plan.encoder_keys).isdisjoint(plan.decoder_keys)
    assert plan.fp16_encoder_bytes == 1_483_366_400
    assert plan.fp16_decoder_bytes == 1_603_243_520
    assert plan.fp16_parameter_bytes == 3_086_609_920
    assert plan.largest_source_tensor_bytes == 265_548_800
    assert plan.bounded_weight_working_set_bytes == 3_352_158_720


def test_full_export_inputs_and_authoring_stack_are_frozen() -> None:
    convert = importlib.import_module("whisper_large_v2.convert")

    features, token_id = convert.full_export_inputs()

    assert features.shape == (1, 80, 3000)
    assert features.dtype == torch.float16
    assert torch.count_nonzero(features) == 0
    assert token_id.tolist() == [[50_258]]
    assert token_id.dtype == torch.int32
    assert convert.require_exact_authoring_stack() == {
        "coreai-core": "1.0.0b2",
        "coreai-opt": "0.2.0",
        "coreai-torch": "0.4.1",
        "torch": "2.9.0",
        "transformers": "4.57.6",
    }


def test_pinned_cli_dry_run_reports_identity_and_does_not_create_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "whisper-large-v2-fp16.aimodel"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "whisper_large_v2.convert",
            "--snapshot",
            str(PINNED_SNAPSHOT),
            "--source-contract",
            str(SOURCE_CONTRACT),
            "--output",
            str(output),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "bounded_weight_working_set_bytes": 3_352_158_720,
        "dtype": "float16",
        "output": str(output),
        "repository": "openai/whisper-large-v2",
        "revision": "ae4642769ce2ad8fc292556ccea8e901f1530655",
        "schema": "caix.whisper-split.v2",
        "tensor_count": 1_259,
    }
    assert not output.exists()


def test_atomic_asset_save_refuses_overwrite_and_cleans_failed_staging(tmp_path: Path) -> None:
    convert = importlib.import_module("whisper_large_v2.convert")
    output = tmp_path / "whisper-large-v2-fp16.aimodel"

    class Program:
        def save_asset(self, path: Path) -> Path:
            path.mkdir()
            (path / "model.bin").write_bytes(b"native")
            return path

    convert.save_program_atomically(Program(), output)
    assert (output / "model.bin").read_bytes() == b"native"

    with pytest.raises(FileExistsError):
        convert.save_program_atomically(Program(), output)
    assert not tuple(tmp_path.glob(".whisper-large-v2-fp16.aimodel.staging-*"))

    failed_output = tmp_path / "failed.aimodel"

    class FailingProgram:
        def save_asset(self, path: Path) -> None:
            path.mkdir()
            (path / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("compiler save failed")

    with pytest.raises(RuntimeError, match="compiler save failed"):
        convert.save_program_atomically(FailingProgram(), failed_output)
    assert not failed_output.exists()
    assert not tuple(tmp_path.glob(".failed.aimodel.staging-*"))


@pytest.mark.skipif(
    "CAIX_RUN_WHISPER_FULL_CHECKPOINT_LOAD" not in os.environ,
    reason="set CAIX_RUN_WHISPER_FULL_CHECKPOINT_LOAD=1 to materialize the pinned FP16 weights",
)
def test_real_checkpoint_loads_every_split_parameter_without_fp32_model() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "whisper_large_v2.convert",
            "--snapshot",
            str(PINNED_SNAPSHOT),
            "--source-contract",
            str(SOURCE_CONTRACT),
            "--load-proof",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ.copy(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["encoder_parameter_count"] == 741_683_200
    assert result["decoder_parameter_count"] == 801_621_760
    assert result["parameter_dtype"] == "float16"
    assert result["meta_parameter_count"] == 0
    assert result["loaded_tensor_count"] == 1_259
    assert 3_086_609_920 <= result["peak_resident_bytes"] < 12 * 1024**3
    assert result["source_sha256"] == (
        "57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b"
    )
