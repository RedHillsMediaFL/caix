from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPOSITORY_ROOT / "models" / "whisper-large-v2-source.json"


def test_contract_pins_exact_source_and_every_tokenizer_asset() -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")

    contract = checkpoint.load_source_contract(SOURCE_MANIFEST)

    assert contract.repository == "openai/whisper-large-v2"
    assert contract.revision == "ae4642769ce2ad8fc292556ccea8e901f1530655"
    assert contract.weights.path == "model.safetensors"
    assert contract.weights.size_bytes == 6_173_370_152
    assert contract.weights.sha256 == (
        "57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b"
    )
    assert contract.tied_embeddings is True
    assert contract.assets == {
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


def test_contract_loader_rejects_source_or_asset_drift(tmp_path: Path) -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    payload["revision"] = "0" * 40
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(checkpoint.CheckpointContractError, match="revision"):
        checkpoint.load_source_contract(drifted)

    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    payload["assets"]["normalizer.json"] = "f" * 64
    drifted.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(checkpoint.CheckpointContractError, match="normalizer.json"):
        checkpoint.load_source_contract(drifted)


def test_expected_tensor_keys_match_the_exact_hugging_face_layout() -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")

    keys = checkpoint.expected_tensor_keys(encoder_layers=32, decoder_layers=32)

    assert len(keys) == 1_259
    assert "model.encoder.conv1.weight" in keys
    assert "model.encoder.layers.31.self_attn.k_proj.weight" in keys
    assert "model.decoder.layers.31.encoder_attn.v_proj.bias" in keys
    assert "model.decoder.embed_tokens.weight" in keys
    assert "proj_out.weight" not in keys
    assert "model.decoder.layers.0.self_attn.k_proj.bias" not in keys


def test_partition_consumes_every_key_and_keeps_the_output_embedding_tied() -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    keys = checkpoint.expected_tensor_keys(encoder_layers=2, decoder_layers=2)
    state_dict = {key: object() for key in keys}
    tied_weight = state_dict["model.decoder.embed_tokens.weight"]

    partition = checkpoint.consume_state_dict(
        state_dict,
        encoder_layers=2,
        decoder_layers=2,
    )

    assert state_dict == {}
    assert len(partition.encoder_cross) == 43
    assert len(partition.decoder) == 46
    assert "model.decoder.layers.0.encoder_attn.k_proj.weight" in partition.encoder_cross
    assert "model.decoder.layers.0.encoder_attn.v_proj.weight" in partition.encoder_cross
    assert "model.decoder.layers.0.encoder_attn.q_proj.weight" in partition.decoder
    assert "model.decoder.layers.0.encoder_attn.out_proj.weight" in partition.decoder
    assert partition.token_embedding is tied_weight
    assert partition.output_projection is tied_weight


def test_partition_rejects_missing_or_unexpected_keys_without_consuming_input() -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    keys = checkpoint.expected_tensor_keys(encoder_layers=1, decoder_layers=1)
    missing = {key: object() for key in keys}
    missing.pop("model.encoder.conv1.weight")
    missing_before = dict(missing)

    with pytest.raises(checkpoint.CheckpointContractError, match="model.encoder.conv1.weight"):
        checkpoint.consume_state_dict(missing, encoder_layers=1, decoder_layers=1)
    assert missing == missing_before

    unexpected = {key: object() for key in keys}
    unexpected["model.proj_out.weight"] = object()
    unexpected_before = dict(unexpected)

    with pytest.raises(checkpoint.CheckpointContractError, match="model.proj_out.weight"):
        checkpoint.consume_state_dict(unexpected, encoder_layers=1, decoder_layers=1)
    assert unexpected == unexpected_before


def test_inventory_reads_only_safetensors_slice_metadata(tmp_path: Path) -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    calls: list[str] = []

    class Slice:
        def get_dtype(self) -> str:
            calls.append("dtype")
            return "F32"

        def get_shape(self) -> list[int]:
            calls.append("shape")
            return [2, 3]

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def keys(self) -> list[str]:
            calls.append("keys")
            return ["tensor"]

        def get_slice(self, key: str) -> Slice:
            calls.append(f"slice:{key}")
            return Slice()

        def get_tensor(self, key: str) -> object:
            raise AssertionError(f"inventory materialized tensor payload: {key}")

    def opener(path: str, *, framework: str, device: str) -> Handle:
        assert path == str(tmp_path / "model.safetensors")
        assert framework == "pt"
        assert device == "cpu"
        return Handle()

    reader = checkpoint.SafetensorsCheckpointReader(
        tmp_path / "model.safetensors",
        opener=opener,
    )

    assert reader.inventory() == {
        "tensor": checkpoint.TensorInventory(dtype="F32", shape=(2, 3))
    }
    assert calls == ["keys", "slice:tensor", "dtype", "shape"]


def test_expected_inventory_pins_every_large_v2_tensor_shape_and_dtype() -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")

    inventory = checkpoint.expected_large_v2_inventory()

    assert len(inventory) == 1_259
    assert inventory["model.encoder.conv1.weight"] == checkpoint.TensorInventory(
        dtype="F32", shape=(1280, 80, 3)
    )
    assert inventory["model.encoder.embed_positions.weight"].shape == (1500, 1280)
    assert inventory["model.decoder.embed_positions.weight"].shape == (448, 1280)
    assert inventory["model.decoder.embed_tokens.weight"].shape == (51865, 1280)
    assert inventory["model.decoder.layers.31.fc1.weight"].shape == (5120, 1280)
    assert inventory["model.decoder.layers.31.encoder_attn.v_proj.bias"].shape == (1280,)
    assert {entry.dtype for entry in inventory.values()} == {"F32"}


def test_snapshot_symlink_target_name_never_substitutes_for_weight_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    expected_digest = hashlib.sha256(b"approved").hexdigest()
    malicious_blob = blobs / expected_digest
    malicious_blob.write_bytes(b"tampered")
    (snapshot / "model.safetensors").symlink_to(malicious_blob)
    contract = checkpoint.WhisperSourceContract(
        repository="test/repository",
        revision="test-revision",
        weights=checkpoint.WeightIdentity(
            path="model.safetensors",
            size_bytes=len(b"tampered"),
            sha256=expected_digest,
        ),
        tied_embeddings=True,
        assets={},
    )
    monkeypatch.setattr(checkpoint, "expected_large_v2_inventory", lambda: {})
    monkeypatch.setattr(
        checkpoint.SafetensorsCheckpointReader,
        "inventory",
        lambda _self: {},
    )

    with pytest.raises(checkpoint.CheckpointContractError, match="weight sha256"):
        checkpoint.validate_pinned_snapshot(snapshot, contract)


def test_snapshot_symlink_support_does_not_require_digest_shaped_target_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    contents = b"verified checkpoint bytes"
    target = blobs / "arbitrary-hugging-face-blob-name"
    target.write_bytes(contents)
    (snapshot / "model.safetensors").symlink_to(target)
    contract = checkpoint.WhisperSourceContract(
        repository="test/repository",
        revision="test-revision",
        weights=checkpoint.WeightIdentity(
            path="model.safetensors",
            size_bytes=len(contents),
            sha256=hashlib.sha256(contents).hexdigest(),
        ),
        tied_embeddings=True,
        assets={},
    )
    monkeypatch.setattr(checkpoint, "expected_large_v2_inventory", lambda: {})
    monkeypatch.setattr(
        checkpoint.SafetensorsCheckpointReader,
        "inventory",
        lambda _self: {},
    )

    validation = checkpoint.validate_pinned_snapshot(snapshot, contract)

    assert validation.weight_size_bytes == len(contents)


def test_inventory_stays_descriptor_bound_and_detects_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    original = b"verified checkpoint bytes"
    replacement = b"tampered checkpoint bytes"
    assert len(replacement) == len(original)
    weight_path = snapshot / "model.safetensors"
    weight_path.write_bytes(original)
    replacement_path = tmp_path / "replacement.safetensors"
    replacement_path.write_bytes(replacement)
    contract = checkpoint.WhisperSourceContract(
        repository="test/repository",
        revision="test-revision",
        weights=checkpoint.WeightIdentity(
            path="model.safetensors",
            size_bytes=len(original),
            sha256=hashlib.sha256(original).hexdigest(),
        ),
        tied_embeddings=True,
        assets={},
    )
    monkeypatch.setattr(checkpoint, "expected_large_v2_inventory", lambda: {})

    class DescriptorReader:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)

        def inventory(self) -> dict[str, object]:
            assert self.path.parent == Path("/dev/fd")
            replacement_path.replace(weight_path)
            assert self.path.read_bytes() == original
            return {}

    monkeypatch.setattr(checkpoint, "SafetensorsCheckpointReader", DescriptorReader)

    with pytest.raises(checkpoint.CheckpointContractError, match="changed while validating"):
        checkpoint.validate_pinned_snapshot(snapshot, contract)

    assert weight_path.read_bytes() == replacement


@pytest.mark.skipif(
    "CAIX_WHISPER_LARGE_V2_SNAPSHOT" not in os.environ,
    reason="set CAIX_WHISPER_LARGE_V2_SNAPSHOT to inspect the mmap-backed real source",
)
def test_real_source_matches_pinned_contract_without_loading_tensor_payloads() -> None:
    checkpoint = importlib.import_module("whisper_large_v2.checkpoint")
    contract = checkpoint.load_source_contract(SOURCE_MANIFEST)
    snapshot = Path(os.environ["CAIX_WHISPER_LARGE_V2_SNAPSHOT"])

    validation = checkpoint.validate_pinned_snapshot(snapshot, contract)

    assert validation.tensor_count == 1_259
    assert validation.weight_size_bytes == 6_173_370_152
    assert validation.verified_asset_count == 10
