import CryptoKit
import Foundation
import XCTest

@testable import PipelineRuntime

final class WhisperAssetAuthenticatorTests: XCTestCase {
    func testAuthenticatesExactTask4AssetContract() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let main = Data("tiny mlir payload".utf8)
        let manifest = try canonicalJSON(task4Manifest(main: main))
        XCTAssertEqual(manifest.count, 2_257)
        XCTAssertEqual(
            sha256(manifest),
            "1a6dabb062d2448218e6cc9e3d898f41714828d38440fa24b1c5b27d0a39fd72")
        let asset = try writeValidAsset(in: directory, main: main)

        let identity = try WhisperAssetAuthenticator.authenticateAsset(at: asset)

        XCTAssertEqual(identity.mainSizeBytes, UInt64(main.count))
        XCTAssertEqual(identity.mainSHA256, sha256(main))
        XCTAssertGreaterThan(identity.assetSizeBytes, identity.mainSizeBytes)
    }

    func testRejectsAbsentExtraAndSymlinkedAssetEntries() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let absent = try writeValidAsset(in: directory.appendingPathComponent("absent"))
        try FileManager.default.removeItem(at: absent.appendingPathComponent("metadata.json"))
        XCTAssertThrowsError(try WhisperAssetAuthenticator.authenticateAsset(at: absent))

        let extra = try writeValidAsset(in: directory.appendingPathComponent("extra"))
        try Data("extra".utf8).write(to: extra.appendingPathComponent("unexpected.bin"))
        XCTAssertThrowsError(try WhisperAssetAuthenticator.authenticateAsset(at: extra))

        for filename in ["metadata.json", "main.mlirb", "main.hash", "caix-manifest.json"] {
            let root = directory.appendingPathComponent("symlink-\(filename)")
            let symlinked = try writeValidAsset(in: root)
            let fileURL = symlinked.appendingPathComponent(filename)
            let target = root.appendingPathComponent("target")
            try FileManager.default.moveItem(at: fileURL, to: target)
            try FileManager.default.createSymbolicLink(at: fileURL, withDestinationURL: target)
            XCTAssertThrowsError(try WhisperAssetAuthenticator.authenticateAsset(at: symlinked))
        }

        let realAsset = try writeValidAsset(in: directory.appendingPathComponent("real"))
        let linkedAsset = directory.appendingPathComponent("linked.aimodel")
        try FileManager.default.createSymbolicLink(
            at: linkedAsset,
            withDestinationURL: realAsset)
        XCTAssertThrowsError(try WhisperAssetAuthenticator.authenticateAsset(at: linkedAsset))
    }

    func testRejectsMainHashMismatchAndNoncanonicalManifest() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let badHash = try writeValidAsset(in: directory.appendingPathComponent("hash"))
        try Data(repeating: 0, count: 32).write(
            to: badHash.appendingPathComponent("main.hash"))
        XCTAssertThrowsError(try WhisperAssetAuthenticator.authenticateAsset(at: badHash)) {
            XCTAssertEqual(
                $0 as? WhisperResidentEngine.LoadError,
                .mainHashMismatch)
        }

        let noncanonical = try writeValidAsset(in: directory.appendingPathComponent("canonical"))
        let manifestURL = noncanonical.appendingPathComponent("caix-manifest.json")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL))
        let pretty = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted])
        try pretty.write(to: manifestURL)
        XCTAssertThrowsError(try WhisperAssetAuthenticator.authenticateAsset(at: noncanonical)) {
            XCTAssertEqual(
                $0 as? WhisperResidentEngine.LoadError,
                .manifestMismatch)
        }
    }

    func testRejectsCanonicalDriftInABIProvenanceAndAuthoringPins() throws {
        let mutations: [([String], Any)] = [
            (["abi", "schema"], "caix.whisper-split.v1"),
            (["source", "revision"], String(repeating: "0", count: 40)),
            (["authoring", "stack", "torch"], "0.0.0"),
            (["authoring", "coreai_models", "package_tree"], String(repeating: "0", count: 40)),
            (["unexpected"], true),
        ]

        for (index, mutation) in mutations.enumerated() {
            let directory = temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let asset = try writeValidAsset(
                in: directory.appendingPathComponent("mutation-\(index)"))
            let manifestURL = asset.appendingPathComponent("caix-manifest.json")
            var manifest = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL))
                    as? [String: Any])
            set(mutation.1, at: mutation.0, in: &manifest)
            try canonicalJSON(manifest).write(to: manifestURL)

            XCTAssertThrowsError(try WhisperAssetAuthenticator.authenticateAsset(at: asset)) {
                XCTAssertEqual(
                    $0 as? WhisperResidentEngine.LoadError,
                    .manifestMismatch)
            }
        }
    }

    func testResidentLockProjectsEveryPinnedWhisperMetadataDigest() throws {
        let lock = try ResidentModelLock.load(from: canonicalLockURL)

        XCTAssertEqual(
            WhisperAssetAuthenticator.metadataDigests(for: lock),
            [
                "added_tokens.json": lock.speech.metadata.addedTokensSHA256,
                "config.json": lock.speech.metadata.configSHA256,
                "generation_config.json": lock.speech.metadata.generationConfigSHA256,
                "merges.txt": lock.speech.metadata.mergesSHA256,
                "normalizer.json": lock.speech.metadata.normalizerSHA256,
                "preprocessor_config.json": lock.speech.metadata.preprocessorConfigSHA256,
                "tokenizer.json": lock.speech.metadata.tokenizerJSONSHA256,
                "tokenizer_config.json": lock.speech.metadata.tokenizerConfigSHA256,
                "vocab.json": lock.speech.metadata.vocabularySHA256,
            ])
    }

    func testMetadataAuthenticationRetainsOnlyRuntimeInputsAndRejectsEveryDrift() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadata = [
            "added_tokens.json": Data(#"{"added":true}"#.utf8),
            "config.json": Data(#"{"model_type":"whisper"}"#.utf8),
            "generation_config.json": Data(#"{"eos_token_id":50257}"#.utf8),
            "merges.txt": Data("a b\n".utf8),
            "normalizer.json": Data(#"{"normalizer":true}"#.utf8),
            "preprocessor_config.json": Data(#"{"feature_size":80}"#.utf8),
            "tokenizer.json": Data(#"{"model":{}}"#.utf8),
            "tokenizer_config.json": Data(#"{"tokenizer_class":"WhisperTokenizer"}"#.utf8),
            "vocab.json": Data(#"{"a":0}"#.utf8),
        ]
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (filename, data) in metadata {
            try data.write(to: directory.appendingPathComponent(filename))
        }
        let digests = metadata.mapValues(sha256)

        let authenticated = try WhisperAssetAuthenticator.authenticateMetadata(
            at: directory,
            expectedDigests: digests)

        XCTAssertEqual(authenticated.generationConfiguration, metadata["generation_config.json"])
        XCTAssertEqual(authenticated.tokenizer, metadata["tokenizer.json"])
        XCTAssertEqual(authenticated.tokenizerConfiguration, metadata["tokenizer_config.json"])

        for filename in metadata.keys.sorted() {
            let url = directory.appendingPathComponent(filename)
            let original = try Data(contentsOf: url)
            try Data("drift".utf8).write(to: url)
            XCTAssertThrowsError(
                try WhisperAssetAuthenticator.authenticateMetadata(
                    at: directory,
                    expectedDigests: digests)
            ) {
                XCTAssertEqual(
                    $0 as? WhisperResidentEngine.LoadError,
                    .metadataDigestMismatch(filename: filename))
            }
            try original.write(to: url)
        }
    }

    #if COREAI_RUNTIME
    func testPublicLoaderRejectsMetadataBeforeAnyCoreAISpecialization() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let asset = try writeValidAsset(in: directory.appendingPathComponent("asset"))
        let emptyMetadata = directory.appendingPathComponent("metadata", isDirectory: true)
        try FileManager.default.createDirectory(
            at: emptyMetadata,
            withIntermediateDirectories: true)

        do {
            _ = try await WhisperResidentEngine.load(
                assetURL: asset,
                tokenizerDirectory: emptyMetadata,
                modelLockURL: canonicalLockURL)
            XCTFail("loader unexpectedly reached CoreAI specialization")
        } catch let error as WhisperResidentEngine.LoadError {
            XCTAssertEqual(error, .metadataFileInvalid(filename: "added_tokens.json"))
        }
    }
    #endif

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var canonicalLockURL: URL {
        repositoryRoot
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(ResidentModelLock.filename)
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-whisper-auth-\(UUID().uuidString)", isDirectory: true)
    }

    private func writeValidAsset(
        in root: URL,
        main: Data = Data("tiny mlir payload".utf8)
    ) throws -> URL {
        let asset = root.appendingPathComponent("whisper.aimodel", isDirectory: true)
        try FileManager.default.createDirectory(at: asset, withIntermediateDirectories: true)
        try Data(#"{"producer":"test"}"#.utf8).write(
            to: asset.appendingPathComponent("metadata.json"))
        try main.write(to: asset.appendingPathComponent("main.mlirb"))
        let digest = SHA256.hash(data: main)
        try Data(digest).write(to: asset.appendingPathComponent("main.hash"))
        try canonicalJSON(task4Manifest(main: main)).write(
            to: asset.appendingPathComponent("caix-manifest.json"))
        return asset
    }

    private func task4Manifest(main: Data) -> [String: Any] {
        let cross: [String: Any] = [
            "dtype": "float16", "shape": [32, 1, 20, 1_500, 64],
        ]
        let selfCache: [String: Any] = [
            "dtype": "float16", "shape": [32, 1, 20, 448, 64],
        ]
        let intScalar: [String: Any] = ["dtype": "int32", "shape": [1]]
        return [
            "schema": "caix.whisper-asset.v1",
            "abi": [
                "schema": "caix.whisper-split.v2",
                "strategy": "explicit_cross_kv_bridge",
                "call_order": ["encode", "load_cross_kv", "decode_step*"],
                "entrypoints": [
                    "encoder": "encode",
                    "decoder_load": "load_cross_kv",
                    "decoder_step": "decode_step",
                ],
                "functions": [
                    "encode": [
                        "inputs": ["input_features"],
                        "outputs": ["cross_key_payload", "cross_value_payload"],
                        "states": [],
                    ],
                    "load_cross_kv": [
                        "inputs": ["cross_key_payload", "cross_value_payload"],
                        "outputs": ["load_status"],
                        "states": ["cross_key_cache", "cross_value_cache", "cross_ready"],
                    ],
                    "decode_step": [
                        "inputs": ["token_id"],
                        "outputs": ["logits", "decode_status"],
                        "states": [
                            "cross_key_cache", "cross_value_cache", "self_key_cache",
                            "self_value_cache", "position", "cross_ready",
                        ],
                    ],
                ],
                "statuses": ["success": 1, "invalid_state": 0],
                "tensors": [
                    "input_features": ["dtype": "float16", "shape": [1, 80, 3_000]],
                    "cross_key_cache": cross,
                    "cross_value_cache": cross,
                    "self_key_cache": selfCache,
                    "self_value_cache": selfCache,
                    "token_id": ["dtype": "int32", "shape": [1, 1]],
                    "position": intScalar,
                    "cross_ready": intScalar,
                    "load_status": intScalar,
                    "logits": ["dtype": "float16", "shape": [1, 1, 51_865]],
                    "decode_status": intScalar,
                    "cross_key_payload": cross,
                    "cross_value_payload": cross,
                ],
            ],
            "source": [
                "repository": "openai/whisper-large-v2",
                "revision": "ae4642769ce2ad8fc292556ccea8e901f1530655",
                "weights": [
                    "path": "model.safetensors",
                    "size_bytes": 6_173_370_152,
                    "sha256": "57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b",
                ],
            ],
            "authoring": [
                "stack": [
                    "coreai-core": "1.0.0b2",
                    "coreai-opt": "0.2.0",
                    "coreai-torch": "0.4.1",
                    "torch": "2.9.0",
                    "transformers": "4.57.6",
                ],
                "coreai_models": [
                    "repository": "https://github.com/kylejfrost/coreai-models.git",
                    "revision": "e666cdc9848fd17f41e43504bc574c8964812c9e",
                    "python_root": "python/src",
                    "package_subtree": "python/src/coreai_models",
                    "package_tree": "b2803957eee13084d06924cfc567a770379234ae",
                ],
            ],
            "main": [
                "path": "main.mlirb",
                "size_bytes": main.count,
                "sha256": sha256(main),
            ],
        ]
    }

    private func canonicalJSON(_ object: Any) throws -> Data {
        try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys, .withoutEscapingSlashes])
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func set(_ value: Any, at path: [String], in object: inout [String: Any]) {
        precondition(!path.isEmpty)
        if path.count == 1 {
            object[path[0]] = value
            return
        }
        var child = object[path[0]] as! [String: Any]
        set(value, at: Array(path.dropFirst()), in: &child)
        object[path[0]] = child
    }
}
