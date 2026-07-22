from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONTRACT = REPOSITORY_ROOT / "models" / "whisper-large-v2-source.json"
AUTHORING_SOURCE = REPOSITORY_ROOT / "models" / "coreai-models-authoring-source.json"
PINNED_SNAPSHOT = Path(
    "/Volumes/SSD/hf-cache/models--openai--whisper-large-v2/snapshots/"
    "ae4642769ce2ad8fc292556ccea8e901f1530655"
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _test_authoring_repository(tmp_path: Path) -> tuple[Path, object]:
    authoring = importlib.import_module("whisper_large_v2.authoring_source")
    repository = tmp_path / "coreai-models"
    package = repository / "python" / "src" / "coreai_models"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'SOURCE_IDENTITY = "committed"\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    _git(repository, "config", "user.name", "CAIX Test")
    _git(repository, "config", "user.email", "caix@example.invalid")
    _git(repository, "add", "python/src/coreai_models/__init__.py")
    _git(repository, "commit", "-q", "-m", "fixture")
    remote = "https://example.invalid/coreai-models.git"
    _git(repository, "remote", "add", "origin", remote)
    revision = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", f"{revision}:python/src/coreai_models")
    contract = authoring.AuthoringSourceContract(
        schema="caix.coreai-models-authoring-source.v1",
        repository=remote,
        revision=revision,
        python_root="python/src",
        package_subtree="python/src/coreai_models",
        package_tree=tree,
    )
    return repository, contract


def _hide_loaded_coreai_models(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name == "coreai_models" or name.startswith("coreai_models."):
            monkeypatch.delitem(sys.modules, name)


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


def test_authoring_source_contract_pins_exact_repository_revision_and_tree() -> None:
    authoring = importlib.import_module("whisper_large_v2.authoring_source")

    contract = authoring.load_authoring_source_contract(AUTHORING_SOURCE)

    assert contract == authoring.AuthoringSourceContract(
        schema="caix.coreai-models-authoring-source.v1",
        repository="https://github.com/kylejfrost/coreai-models.git",
        revision="e666cdc9848fd17f41e43504bc574c8964812c9e",
        python_root="python/src",
        package_subtree="python/src/coreai_models",
        package_tree="b2803957eee13084d06924cfc567a770379234ae",
    )


def test_authoring_source_contract_rejects_revision_and_tree_drift(
    tmp_path: Path,
) -> None:
    authoring = importlib.import_module("whisper_large_v2.authoring_source")
    payload = json.loads(AUTHORING_SOURCE.read_text(encoding="utf-8"))
    drifted = tmp_path / "authoring-source.json"

    for field in ("revision", "package_tree"):
        changed = dict(payload)
        changed[field] = "0" * 40
        drifted.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(authoring.AuthoringSourceError, match=field):
            authoring.load_authoring_source_contract(drifted)


def test_authoring_archive_ignores_dirty_worktree_and_rejects_wrong_git_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authoring = importlib.import_module("whisper_large_v2.authoring_source")
    repository, contract = _test_authoring_repository(tmp_path)
    _hide_loaded_coreai_models(monkeypatch)
    (repository / "python/src/coreai_models/__init__.py").write_text(
        'SOURCE_IDENTITY = "dirty replacement"\n',
        encoding="utf-8",
    )
    temp_root = tmp_path / "authenticated"
    temp_root.mkdir()

    with authoring.authenticated_authoring_source(
        repository,
        contract,
        temp_root=temp_root,
    ) as authenticated:
        coreai_models = importlib.import_module("coreai_models")
        assert coreai_models.SOURCE_IDENTITY == "committed"
        assert Path(coreai_models.__file__).is_relative_to(authenticated.python_root)
        assert authoring.verify_coreai_models_imports(authenticated.python_root) == 1
        extracted_root = authenticated.extraction_root

    assert not extracted_root.exists()

    with pytest.raises(authoring.AuthoringSourceError, match="revision"):
        with authoring.authenticated_authoring_source(
            repository,
            replace(contract, revision="0" * 40),
            temp_root=temp_root,
        ):
            pass

    with pytest.raises(authoring.AuthoringSourceError, match="package_tree"):
        with authoring.authenticated_authoring_source(
            repository,
            replace(contract, package_tree="0" * 40),
            temp_root=temp_root,
        ):
            pass


def test_authoring_bootstrap_rejects_coreai_models_imported_outside_authenticated_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authoring = importlib.import_module("whisper_large_v2.authoring_source")
    repository, contract = _test_authoring_repository(tmp_path)
    _hide_loaded_coreai_models(monkeypatch)
    imported = ModuleType("coreai_models")
    imported.__file__ = str(repository / "python/src/coreai_models/__init__.py")
    monkeypatch.setitem(sys.modules, "coreai_models", imported)
    temp_root = tmp_path / "authenticated"
    temp_root.mkdir()

    with pytest.raises(authoring.AuthoringSourceError, match="outside authenticated root"):
        with authoring.authenticated_authoring_source(
            repository,
            contract,
            temp_root=temp_root,
        ):
            pass


def test_pinned_config_uses_authenticated_json_not_from_pretrained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    convert = importlib.import_module("whisper_large_v2.convert")
    from transformers import WhisperConfig

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config_bytes = b'{"model_type":"whisper","d_model":16}'
    (snapshot / "config.json").write_bytes(config_bytes)
    contract = checkpoint.WhisperSourceContract(
        repository="test/repository",
        revision="test-revision",
        weights=checkpoint.WeightIdentity("unused", 0, "0" * 64),
        tied_embeddings=True,
        assets={"config.json": hashlib.sha256(config_bytes).hexdigest()},
    )

    def reject_from_pretrained(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("snapshot path was reopened through from_pretrained")

    monkeypatch.setattr(WhisperConfig, "from_pretrained", reject_from_pretrained)

    config = convert._load_pinned_config(snapshot, contract)

    assert config.model_type == "whisper"
    assert config.d_model == 16


def test_converter_import_and_export_script_do_not_trust_mutable_python_source() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "python")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import whisper_large_v2.convert; "
                "assert not any(name == 'coreai_models' or "
                "name.startswith('coreai_models.') for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    script = (REPOSITORY_ROOT / "scripts/export-whisper-large-v2-coreai.sh").read_text(
        encoding="utf-8"
    )
    assert 'export PYTHONPATH="$caix_repo_root/python"' in script
    assert "coreai-models/python/src" not in script
    assert "--authoring-source" in script
    assert "--coreai-models-repository" in script


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
