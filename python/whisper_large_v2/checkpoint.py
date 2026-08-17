"""Pinned source contract and memory-safe checkpoint inventory for Whisper large-v2."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

_ENCODER_SHARED_KEYS = (
    "model.encoder.conv1.bias",
    "model.encoder.conv1.weight",
    "model.encoder.conv2.bias",
    "model.encoder.conv2.weight",
    "model.encoder.embed_positions.weight",
    "model.encoder.layer_norm.bias",
    "model.encoder.layer_norm.weight",
)

_ENCODER_LAYER_SUFFIXES = (
    "fc1.bias",
    "fc1.weight",
    "fc2.bias",
    "fc2.weight",
    "final_layer_norm.bias",
    "final_layer_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.out_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn_layer_norm.bias",
    "self_attn_layer_norm.weight",
)

_DECODER_SHARED_KEYS = (
    "model.decoder.embed_positions.weight",
    "model.decoder.embed_tokens.weight",
    "model.decoder.layer_norm.bias",
    "model.decoder.layer_norm.weight",
)

_DECODER_LAYER_SUFFIXES = (
    "encoder_attn.k_proj.weight",
    "encoder_attn.out_proj.bias",
    "encoder_attn.out_proj.weight",
    "encoder_attn.q_proj.bias",
    "encoder_attn.q_proj.weight",
    "encoder_attn.v_proj.bias",
    "encoder_attn.v_proj.weight",
    "encoder_attn_layer_norm.bias",
    "encoder_attn_layer_norm.weight",
    "fc1.bias",
    "fc1.weight",
    "fc2.bias",
    "fc2.weight",
    "final_layer_norm.bias",
    "final_layer_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.out_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn_layer_norm.bias",
    "self_attn_layer_norm.weight",
)

_APPROVED_SCHEMA = "caix.whisper-source.v1"
_APPROVED_REPOSITORY = "openai/whisper-large-v2"
_APPROVED_REVISION = "ae4642769ce2ad8fc292556ccea8e901f1530655"
_APPROVED_ASSETS = {
    "added_tokens.json": "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1",
    "config.json": "5f1573015838f8d679678b09354b537061561c55fd22eecd129ef4cf8588a470",
    "generation_config.json": (
        "031721643aab5be7250eb668c6b9b5c67d2549420522ac1291bfd346bfff6297"
    ),
    "merges.txt": "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6",
    "normalizer.json": "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
    "preprocessor_config.json": (
        "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666"
    ),
    "special_tokens_map.json": (
        "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691"
    ),
    "tokenizer.json": "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
    "tokenizer_config.json": (
        "2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce"
    ),
    "vocab.json": "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9",
}
_MAX_JSON_ASSET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class WeightIdentity:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class WhisperSourceContract:
    repository: str
    revision: str
    weights: WeightIdentity
    tied_embeddings: bool
    assets: Mapping[str, str]


@dataclass(frozen=True)
class WhisperCheckpointPartitions:
    encoder_cross: Mapping[str, Any]
    decoder: Mapping[str, Any]
    token_embedding: Any
    output_projection: Any


@dataclass(frozen=True)
class TensorInventory:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class SnapshotValidation:
    tensor_count: int
    weight_size_bytes: int
    verified_asset_count: int


@dataclass(frozen=True)
class _VerifiedRegularFile:
    descriptor_path: Path
    size_bytes: int


class CheckpointContractError(ValueError):
    """The source checkpoint does not exactly match the pinned key contract."""


def load_source_contract(path: Path) -> WhisperSourceContract:
    """Load the small tracked contract without touching checkpoint tensor payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload["weights"]
    contract = WhisperSourceContract(
        repository=payload["repository"],
        revision=payload["revision"],
        weights=WeightIdentity(
            path=weights["path"],
            size_bytes=weights["size_bytes"],
            sha256=weights["sha256"],
        ),
        tied_embeddings=payload["tied_embeddings"],
        assets=payload["assets"],
    )
    expected_weight = WeightIdentity(
        path="model.safetensors",
        size_bytes=6_173_370_152,
        sha256="57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b",
    )
    exact_fields = {
        "schema": (payload.get("schema"), _APPROVED_SCHEMA),
        "repository": (contract.repository, _APPROVED_REPOSITORY),
        "revision": (contract.revision, _APPROVED_REVISION),
        "weights": (contract.weights, expected_weight),
        "tied_embeddings": (contract.tied_embeddings, True),
    }
    for field, (actual, expected) in exact_fields.items():
        if actual != expected:
            raise CheckpointContractError(f"source contract drift at {field}")
    if set(contract.assets) != set(_APPROVED_ASSETS):
        raise CheckpointContractError("source contract drift at assets")
    for filename, expected_digest in _APPROVED_ASSETS.items():
        if contract.assets[filename] != expected_digest:
            raise CheckpointContractError(f"source contract drift at assets.{filename}")
    return contract


class SafetensorsCheckpointReader:
    """Read checkpoint headers and selected tensors without eager full-file loading."""

    def __init__(
        self,
        path: Path,
        *,
        opener: Callable[..., Any] = safe_open,
    ) -> None:
        self.path = path
        self._opener = opener

    def inventory(self) -> Mapping[str, TensorInventory]:
        entries: dict[str, TensorInventory] = {}
        with self._opener(str(self.path), framework="pt", device="cpu") as handle:
            for key in tuple(handle.keys()):
                tensor_slice = handle.get_slice(key)
                entries[key] = TensorInventory(
                    dtype=tensor_slice.get_dtype(),
                    shape=tuple(tensor_slice.get_shape()),
                )
        return entries


def expected_large_v2_inventory() -> Mapping[str, TensorInventory]:
    """Return the exact 1,259-tensor FP32 inventory at the approved revision."""
    d_model = 1_280
    feed_forward = 5_120
    inventory: dict[str, TensorInventory] = {}

    shared_shapes = {
        "model.encoder.conv1.bias": (d_model,),
        "model.encoder.conv1.weight": (d_model, 80, 3),
        "model.encoder.conv2.bias": (d_model,),
        "model.encoder.conv2.weight": (d_model, d_model, 3),
        "model.encoder.embed_positions.weight": (1_500, d_model),
        "model.encoder.layer_norm.bias": (d_model,),
        "model.encoder.layer_norm.weight": (d_model,),
        "model.decoder.embed_positions.weight": (448, d_model),
        "model.decoder.embed_tokens.weight": (51_865, d_model),
        "model.decoder.layer_norm.bias": (d_model,),
        "model.decoder.layer_norm.weight": (d_model,),
    }
    encoder_layer_shapes = {
        "fc1.bias": (feed_forward,),
        "fc1.weight": (feed_forward, d_model),
        "fc2.bias": (d_model,),
        "fc2.weight": (d_model, feed_forward),
        "final_layer_norm.bias": (d_model,),
        "final_layer_norm.weight": (d_model,),
        "self_attn.k_proj.weight": (d_model, d_model),
        "self_attn.out_proj.bias": (d_model,),
        "self_attn.out_proj.weight": (d_model, d_model),
        "self_attn.q_proj.bias": (d_model,),
        "self_attn.q_proj.weight": (d_model, d_model),
        "self_attn.v_proj.bias": (d_model,),
        "self_attn.v_proj.weight": (d_model, d_model),
        "self_attn_layer_norm.bias": (d_model,),
        "self_attn_layer_norm.weight": (d_model,),
    }
    decoder_layer_shapes = {
        **encoder_layer_shapes,
        "encoder_attn.k_proj.weight": (d_model, d_model),
        "encoder_attn.out_proj.bias": (d_model,),
        "encoder_attn.out_proj.weight": (d_model, d_model),
        "encoder_attn.q_proj.bias": (d_model,),
        "encoder_attn.q_proj.weight": (d_model, d_model),
        "encoder_attn.v_proj.bias": (d_model,),
        "encoder_attn.v_proj.weight": (d_model, d_model),
        "encoder_attn_layer_norm.bias": (d_model,),
        "encoder_attn_layer_norm.weight": (d_model,),
    }

    for key, shape in shared_shapes.items():
        inventory[key] = TensorInventory(dtype="F32", shape=shape)
    for layer in range(32):
        prefix = f"model.encoder.layers.{layer}."
        for suffix, shape in encoder_layer_shapes.items():
            inventory[prefix + suffix] = TensorInventory(dtype="F32", shape=shape)
    for layer in range(32):
        prefix = f"model.decoder.layers.{layer}."
        for suffix, shape in decoder_layer_shapes.items():
            inventory[prefix + suffix] = TensorInventory(dtype="F32", shape=shape)
    return inventory


def validate_pinned_snapshot(
    snapshot: Path,
    contract: WhisperSourceContract,
) -> SnapshotValidation:
    """Validate source identity and tensor headers with bounded resident memory."""
    weight_path = snapshot / contract.weights.path
    for filename, expected_digest in contract.assets.items():
        asset = snapshot / filename
        with _verified_regular_file(
            asset,
            expected_sha256=expected_digest,
            expected_size=None,
            label=f"pinned source asset {filename}",
        ):
            pass

    with _verified_regular_file(
        weight_path,
        expected_sha256=contract.weights.sha256,
        expected_size=contract.weights.size_bytes,
        label="pinned checkpoint weight",
    ) as verified_weight:
        actual_inventory = SafetensorsCheckpointReader(
            verified_weight.descriptor_path
        ).inventory()
        expected_inventory = expected_large_v2_inventory()
        if actual_inventory != expected_inventory:
            actual_keys = set(actual_inventory)
            expected_keys = set(expected_inventory)
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            mismatched = sorted(
                key
                for key in actual_keys & expected_keys
                if actual_inventory[key] != expected_inventory[key]
            )
            raise CheckpointContractError(
                "pinned checkpoint inventory differs: "
                f"missing={missing!r}, unexpected={unexpected!r}, mismatched={mismatched!r}"
            )
    return SnapshotValidation(
        tensor_count=len(actual_inventory),
        weight_size_bytes=verified_weight.size_bytes,
        verified_asset_count=len(contract.assets),
    )


def load_verified_json_asset(
    snapshot: Path,
    contract: WhisperSourceContract,
    filename: str,
) -> dict[str, Any]:
    """Authenticate and parse one bounded JSON asset through its open descriptor."""
    if filename != Path(filename).name or filename not in contract.assets:
        raise CheckpointContractError(f"unapproved pinned JSON asset: {filename}")
    with _verified_regular_file(
        snapshot / filename,
        expected_sha256=contract.assets[filename],
        expected_size=None,
        label=f"pinned source asset {filename}",
    ) as verified:
        if verified.size_bytes > _MAX_JSON_ASSET_BYTES:
            raise CheckpointContractError(f"pinned source asset {filename} is too large")
        try:
            with verified.descriptor_path.open("rb", buffering=0) as source:
                contents = source.read(_MAX_JSON_ASSET_BYTES + 1)
        except OSError as error:
            raise CheckpointContractError(
                f"pinned source asset {filename} descriptor read failed"
            ) from error
        if len(contents) != verified.size_bytes:
            raise CheckpointContractError(f"pinned source asset {filename} size changed")
        try:
            payload = json.loads(contents)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CheckpointContractError(
                f"pinned source asset {filename} is malformed JSON"
            ) from error
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) for key in payload
        ):
            raise CheckpointContractError(
                f"pinned source asset {filename} must contain a JSON object"
            )
        return payload


@contextmanager
def _verified_regular_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int | None,
    label: str,
) -> Iterator[_VerifiedRegularFile]:
    """Hash one stable regular-file descriptor and keep it open for its consumer."""
    descriptor = _open_regular_file_no_follow(path, label=label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointContractError(f"{label} is not a regular file")
        if expected_size is not None and before.st_size != expected_size:
            raise CheckpointContractError(f"{label} size differs")
        if _sha256_descriptor(descriptor) != expected_sha256:
            raise CheckpointContractError(f"{label} sha256 differs")
        _require_unchanged_descriptor(descriptor, before, label=label)
        yield _VerifiedRegularFile(
            descriptor_path=Path("/dev/fd") / str(descriptor),
            size_bytes=before.st_size,
        )
        _require_unchanged_descriptor(descriptor, before, label=label)
    finally:
        os.close(descriptor)


def _open_regular_file_no_follow(path: Path, *, label: str) -> int:
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError as error:
        if error.errno != errno.ELOOP:
            raise CheckpointContractError(f"{label} is missing or unreadable") from error

    try:
        target_text = os.readlink(path)
    except OSError as error:
        raise CheckpointContractError(f"{label} symlink changed while opening") from error
    target = Path(target_text)
    if not target.is_absolute():
        target = path.parent / target
    try:
        return os.open(target, flags)
    except OSError as error:
        raise CheckpointContractError(
            f"{label} symlink target is missing, unreadable, or not a regular file"
        ) from error


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _require_unchanged_descriptor(
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    actual = os.fstat(descriptor)
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(actual, field) != getattr(expected, field) for field in fields):
        raise CheckpointContractError(f"{label} changed while validating")


def expected_tensor_keys(*, encoder_layers: int, decoder_layers: int) -> frozenset[str]:
    """Return every tensor key consumed by the pinned Transformers Whisper layout."""
    keys = set(_ENCODER_SHARED_KEYS)
    keys.update(_DECODER_SHARED_KEYS)
    for layer in range(encoder_layers):
        prefix = f"model.encoder.layers.{layer}."
        keys.update(prefix + suffix for suffix in _ENCODER_LAYER_SUFFIXES)
    for layer in range(decoder_layers):
        prefix = f"model.decoder.layers.{layer}."
        keys.update(prefix + suffix for suffix in _DECODER_LAYER_SUFFIXES)
    return frozenset(keys)


def consume_state_dict(
    state_dict: MutableMapping[str, Any],
    *,
    encoder_layers: int,
    decoder_layers: int,
) -> WhisperCheckpointPartitions:
    """Consume and partition every exact source tensor, rejecting drift atomically."""
    expected = expected_tensor_keys(
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )
    actual = frozenset(state_dict)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise CheckpointContractError(
            f"checkpoint keys differ: missing={missing!r}, unexpected={unexpected!r}"
        )

    encoder_cross: dict[str, Any] = {}
    decoder: dict[str, Any] = {}
    for key in sorted(expected):
        value = state_dict.pop(key)
        is_cross_projection = ".encoder_attn." in key and key.endswith(
            ("k_proj.weight", "v_proj.weight", "v_proj.bias")
        )
        if key.startswith("model.encoder.") or is_cross_projection:
            encoder_cross[key] = value
        else:
            decoder[key] = value

    tied_weight = decoder["model.decoder.embed_tokens.weight"]
    return WhisperCheckpointPartitions(
        encoder_cross=encoder_cross,
        decoder=decoder,
        token_embedding=tied_weight,
        output_projection=tied_weight,
    )
