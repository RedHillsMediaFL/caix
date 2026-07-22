from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


def _unmanifested_asset(root: Path, main: bytes = b"tiny mlir payload") -> Path:
    asset = root / "whisper.aimodel"
    asset.mkdir(parents=True)
    (asset / "metadata.json").write_bytes(b'{"producer":"test"}')
    (asset / "main.mlirb").write_bytes(main)
    (asset / "main.hash").write_bytes(hashlib.sha256(main).digest())
    return asset


def _valid_asset(root: Path, main: bytes = b"tiny mlir payload") -> Path:
    manifest = importlib.import_module("whisper_large_v2.manifest")
    asset = _unmanifested_asset(root, main)
    manifest.write_caix_manifest(asset)
    return asset


def test_caix_manifest_is_canonical_and_authenticates_the_exact_asset(tmp_path: Path) -> None:
    manifest = importlib.import_module("whisper_large_v2.manifest")
    asset = _valid_asset(tmp_path)

    validation = manifest.validate_caix_asset(asset)
    payload = json.loads((asset / "caix-manifest.json").read_bytes())

    assert validation.main_sha256 == hashlib.sha256(b"tiny mlir payload").hexdigest()
    assert validation.main_size_bytes == len(b"tiny mlir payload")
    assert validation.asset_bytes == sum(
        path.stat().st_size for path in asset.iterdir()
    )
    assert validation.manifest == payload
    assert (asset / "caix-manifest.json").read_bytes() == manifest.canonical_json_bytes(
        payload
    )
    assert (asset / "main.hash").read_bytes() == bytes.fromhex(
        validation.main_sha256
    )
    assert payload["schema"] == "caix.whisper-asset.v1"
    assert payload["abi"]["schema"] == "caix.whisper-split.v2"
    assert payload["abi"]["call_order"] == [
        "encode",
        "load_cross_kv",
        "decode_step*",
    ]
    assert payload["source"] == {
        "repository": "openai/whisper-large-v2",
        "revision": "ae4642769ce2ad8fc292556ccea8e901f1530655",
        "weights": {
            "path": "model.safetensors",
            "sha256": (
                "57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b"
            ),
            "size_bytes": 6_173_370_152,
        },
    }
    assert payload["authoring"]["coreai_models"]["revision"] == (
        "e666cdc9848fd17f41e43504bc574c8964812c9e"
    )
    assert payload["authoring"]["coreai_models"]["package_tree"] == (
        "b2803957eee13084d06924cfc567a770379234ae"
    )


@pytest.mark.parametrize(
    "removed",
    ("metadata.json", "main.mlirb", "main.hash", "caix-manifest.json"),
)
def test_caix_manifest_rejects_absent_or_extra_asset_entries(
    tmp_path: Path,
    removed: str,
) -> None:
    manifest = importlib.import_module("whisper_large_v2.manifest")
    asset = _valid_asset(tmp_path)
    (asset / removed).unlink()

    with pytest.raises(manifest.AssetManifestError, match="entries"):
        manifest.validate_caix_asset(asset)

    asset = _valid_asset(tmp_path / "extra")
    (asset / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(manifest.AssetManifestError, match="entries"):
        manifest.validate_caix_asset(asset)


@pytest.mark.parametrize(
    "filename",
    ("metadata.json", "main.mlirb", "main.hash", "caix-manifest.json"),
)
def test_caix_manifest_rejects_symlinked_or_nonregular_files(
    tmp_path: Path,
    filename: str,
) -> None:
    manifest = importlib.import_module("whisper_large_v2.manifest")
    asset = _valid_asset(tmp_path / "symlink")
    target = tmp_path / f"{filename}.target"
    target.write_bytes((asset / filename).read_bytes())
    (asset / filename).unlink()
    (asset / filename).symlink_to(target)

    with pytest.raises(manifest.AssetManifestError, match="regular file"):
        manifest.validate_caix_asset(asset)

    asset = _valid_asset(tmp_path / "fifo")
    (asset / filename).unlink()
    os.mkfifo(asset / filename)
    with pytest.raises(manifest.AssetManifestError, match="regular file"):
        manifest.validate_caix_asset(asset)


def test_caix_manifest_rejects_malformed_noncanonical_and_mismatched_hashes(
    tmp_path: Path,
) -> None:
    manifest = importlib.import_module("whisper_large_v2.manifest")
    asset = _valid_asset(tmp_path / "malformed")
    (asset / "caix-manifest.json").write_bytes(b"{")
    with pytest.raises(manifest.AssetManifestError, match="manifest JSON"):
        manifest.validate_caix_asset(asset)

    asset = _valid_asset(tmp_path / "noncanonical")
    payload = json.loads((asset / "caix-manifest.json").read_bytes())
    (asset / "caix-manifest.json").write_text(json.dumps(payload, indent=2))
    with pytest.raises(manifest.AssetManifestError, match="canonical"):
        manifest.validate_caix_asset(asset)

    asset = _valid_asset(tmp_path / "raw-hash")
    (asset / "main.hash").write_bytes(b"\x00" * 32)
    with pytest.raises(manifest.AssetManifestError, match="main.hash"):
        manifest.validate_caix_asset(asset)

    asset = _valid_asset(tmp_path / "manifest-hash")
    payload = json.loads((asset / "caix-manifest.json").read_bytes())
    payload["main"]["sha256"] = "0" * 64
    (asset / "caix-manifest.json").write_bytes(manifest.canonical_json_bytes(payload))
    with pytest.raises(manifest.AssetManifestError, match="manifest fields"):
        manifest.validate_caix_asset(asset)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("abi", "schema"), "caix.whisper-split.v1"),
        (("source", "revision"), "0" * 40),
        (("authoring", "stack", "torch"), "0.0.0"),
        (("authoring", "coreai_models", "package_tree"), "0" * 40),
    ),
)
def test_caix_manifest_rejects_altered_pinned_fields_and_extra_fields(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: str,
) -> None:
    manifest = importlib.import_module("whisper_large_v2.manifest")
    asset = _valid_asset(tmp_path)
    payload = json.loads((asset / "caix-manifest.json").read_bytes())
    changed = deepcopy(payload)
    destination = changed
    for component in path[:-1]:
        destination = destination[component]
    destination[path[-1]] = replacement
    (asset / "caix-manifest.json").write_bytes(manifest.canonical_json_bytes(changed))

    with pytest.raises(manifest.AssetManifestError, match="manifest fields"):
        manifest.validate_caix_asset(asset)

    changed = deepcopy(payload)
    changed["unexpected"] = True
    (asset / "caix-manifest.json").write_bytes(manifest.canonical_json_bytes(changed))
    with pytest.raises(manifest.AssetManifestError, match="manifest fields"):
        manifest.validate_caix_asset(asset)


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
