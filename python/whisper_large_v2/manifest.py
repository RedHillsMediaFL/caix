"""Strict embedded provenance for a native Whisper CoreAI asset."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from whisper_large_v2.abi import NativeWhisperABI, TensorABI

EXACT_AUTHORING_STACK = {
    "coreai-core": "1.0.0b2",
    "coreai-opt": "0.2.0",
    "coreai-torch": "0.4.1",
    "torch": "2.9.0",
    "transformers": "4.57.6",
}

_ASSET_ENTRIES = frozenset(
    {"metadata.json", "main.mlirb", "main.hash", "caix-manifest.json"}
)
_CORE_ASSET_ENTRIES = _ASSET_ENTRIES - {"caix-manifest.json"}
_MAX_MANIFEST_BYTES = 128 * 1024
_MAX_METADATA_BYTES = 1024 * 1024


@dataclass(frozen=True)
class FileIdentity:
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class AssetValidation:
    asset_bytes: int
    main_size_bytes: int
    main_sha256: str
    manifest: dict[str, Any]


class AssetManifestError(RuntimeError):
    """A saved CoreAI asset does not match the exact CAIX contract."""


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a sidecar deterministically, with no insignificant whitespace."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _tensor_payload(tensor: TensorABI) -> dict[str, Any]:
    return {"dtype": tensor.dtype, "shape": list(tensor.shape)}


def expected_caix_manifest(main: FileIdentity) -> dict[str, Any]:
    """Build the only accepted manifest for the pinned Whisper artifact."""
    abi = NativeWhisperABI.large_v2()
    tensors = {name: _tensor_payload(tensor) for name, tensor in abi.tensors.items()}
    tensors["cross_key_payload"] = _tensor_payload(abi.tensors["cross_key_cache"])
    tensors["cross_value_payload"] = _tensor_payload(abi.tensors["cross_value_cache"])
    return {
        "schema": "caix.whisper-asset.v1",
        "abi": {
            "schema": abi.schema,
            "strategy": abi.strategy,
            "call_order": list(abi.call_order),
            "entrypoints": dict(abi.entrypoints),
            "functions": {
                "encode": {
                    "inputs": ["input_features"],
                    "outputs": ["cross_key_payload", "cross_value_payload"],
                    "states": [],
                },
                "load_cross_kv": {
                    "inputs": ["cross_key_payload", "cross_value_payload"],
                    "outputs": ["load_status"],
                    "states": [
                        "cross_key_cache",
                        "cross_value_cache",
                        "cross_ready",
                    ],
                },
                "decode_step": {
                    "inputs": ["token_id"],
                    "outputs": ["logits", "decode_status"],
                    "states": [
                        "cross_key_cache",
                        "cross_value_cache",
                        "self_key_cache",
                        "self_value_cache",
                        "position",
                        "cross_ready",
                    ],
                },
            },
            "statuses": {
                "success": abi.success_status,
                "invalid_state": abi.invalid_state_status,
            },
            "tensors": tensors,
        },
        "source": {
            "repository": "openai/whisper-large-v2",
            "revision": "ae4642769ce2ad8fc292556ccea8e901f1530655",
            "weights": {
                "path": "model.safetensors",
                "size_bytes": 6_173_370_152,
                "sha256": (
                    "57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b"
                ),
            },
        },
        "authoring": {
            "stack": dict(EXACT_AUTHORING_STACK),
            "coreai_models": {
                "repository": "https://github.com/kylejfrost/coreai-models.git",
                "revision": "e666cdc9848fd17f41e43504bc574c8964812c9e",
                "python_root": "python/src",
                "package_subtree": "python/src/coreai_models",
                "package_tree": "b2803957eee13084d06924cfc567a770379234ae",
            },
        },
        "main": {
            "path": "main.mlirb",
            "size_bytes": main.size_bytes,
            "sha256": main.sha256,
        },
    }


def _require_asset_directory(asset: Path) -> None:
    try:
        details = os.lstat(asset)
    except OSError as error:
        raise AssetManifestError("CAIX asset directory is missing") from error
    if not stat.S_ISDIR(details.st_mode) or asset.suffix != ".aimodel":
        raise AssetManifestError("CAIX asset is not a regular .aimodel directory")


def _require_entries(asset: Path, expected: frozenset[str]) -> None:
    _require_asset_directory(asset)
    try:
        with os.scandir(asset) as iterator:
            entries = {entry.name: entry for entry in iterator}
    except OSError as error:
        raise AssetManifestError("CAIX asset entries are unreadable") from error
    if set(entries) != expected:
        raise AssetManifestError(
            f"CAIX asset entries differ: expected={sorted(expected)!r}, "
            f"actual={sorted(entries)!r}"
        )
    for name, entry in entries.items():
        try:
            details = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise AssetManifestError(f"CAIX asset {name} is unreadable") from error
        if not stat.S_ISREG(details.st_mode):
            raise AssetManifestError(f"CAIX asset {name} is not a regular file")


@contextmanager
def _open_regular_file(path: Path, *, label: str) -> Iterator[tuple[int, os.stat_result]]:
    try:
        path_details = os.lstat(path)
    except OSError as error:
        raise AssetManifestError(f"{label} is missing or unreadable") from error
    if not stat.S_ISREG(path_details.st_mode):
        raise AssetManifestError(f"{label} is not a regular file")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AssetManifestError(f"{label} is not a regular file") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (path_details.st_dev, path_details.st_ino):
            raise AssetManifestError(f"{label} changed while opening")
        yield descriptor, opened
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(after, field) != getattr(opened, field) for field in fields):
            raise AssetManifestError(f"{label} changed while reading")
    finally:
        os.close(descriptor)


def sha256_regular_file(path: Path, *, label: str = "file") -> FileIdentity:
    """Hash one no-follow regular-file descriptor with a fixed 1 MiB buffer."""
    with _open_regular_file(path, label=label) as (descriptor, details):
        digest = hashlib.sha256()
        offset = 0
        while chunk := os.pread(descriptor, 1024 * 1024, offset):
            digest.update(chunk)
            offset += len(chunk)
        if offset != details.st_size:
            raise AssetManifestError(f"{label} size changed while hashing")
        return FileIdentity(size_bytes=details.st_size, sha256=digest.hexdigest())


def _read_regular_file(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    with _open_regular_file(path, label=label) as (descriptor, details):
        if details.st_size > maximum_bytes:
            raise AssetManifestError(f"{label} is too large")
        contents = os.pread(descriptor, maximum_bytes + 1, 0)
        if len(contents) != details.st_size:
            raise AssetManifestError(f"{label} size changed while reading")
        return contents


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_caix_manifest(asset: Path) -> FileIdentity:
    """Write and fsync the exact sidecar after checking CoreAI's raw main hash."""
    _require_entries(asset, _CORE_ASSET_ENTRIES)
    main = sha256_regular_file(asset / "main.mlirb", label="main.mlirb")
    raw_hash = _read_regular_file(
        asset / "main.hash",
        label="main.hash",
        maximum_bytes=32,
    )
    if raw_hash != bytes.fromhex(main.sha256):
        raise AssetManifestError("main.hash differs from main.mlirb SHA-256")
    contents = canonical_json_bytes(expected_caix_manifest(main))
    manifest_path = asset / "caix-manifest.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(manifest_path, flags, 0o644)
        try:
            view = memoryview(contents)
            while view:
                written = os.write(descriptor, view)
                if written == 0:
                    raise OSError("caix-manifest.json write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(asset)
    except OSError as error:
        raise AssetManifestError("caix-manifest.json write failed") from error
    return main


def validate_caix_asset(asset: Path) -> AssetValidation:
    """Authenticate every exact asset entry before any CoreAI load occurs."""
    _require_entries(asset, _ASSET_ENTRIES)
    metadata_bytes = _read_regular_file(
        asset / "metadata.json",
        label="metadata.json",
        maximum_bytes=_MAX_METADATA_BYTES,
    )
    main = sha256_regular_file(asset / "main.mlirb", label="main.mlirb")
    raw_hash = _read_regular_file(
        asset / "main.hash",
        label="main.hash",
        maximum_bytes=32,
    )
    if len(raw_hash) != 32 or raw_hash != bytes.fromhex(main.sha256):
        raise AssetManifestError("main.hash differs from main.mlirb SHA-256")
    manifest_bytes = _read_regular_file(
        asset / "caix-manifest.json",
        label="caix-manifest.json",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    try:
        payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssetManifestError("caix-manifest.json manifest JSON is malformed") from error
    if not isinstance(payload, dict):
        raise AssetManifestError("caix-manifest.json manifest JSON is not an object")
    if canonical_json_bytes(payload) != manifest_bytes:
        raise AssetManifestError("caix-manifest.json is not canonical")
    expected = expected_caix_manifest(main)
    if payload != expected:
        raise AssetManifestError("caix-manifest.json manifest fields differ")
    return AssetValidation(
        asset_bytes=(
            len(metadata_bytes) + main.size_bytes + len(raw_hash) + len(manifest_bytes)
        ),
        main_size_bytes=main.size_bytes,
        main_sha256=main.sha256,
        manifest=payload,
    )
