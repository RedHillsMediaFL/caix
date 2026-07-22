import XCTest

@testable import CoreAIServer

final class ModelManagerTests: XCTestCase {
    func testGemmaInstructionTunedMetadataRepairsServedName() async throws {
        XCTAssertEqual(
            ModelNameRepair.preferredServedName(
                directoryName: "gemma-4-26b-a4b-coreai",
                metadataName: "gemma-4-26b-a4b-coreai",
                sourceModelID: "google/gemma-4-26B-A4B-it",
                tokenizer: "google/gemma-4-26B-A4B-it"),
            "gemma-4-26b-a4b-it-coreai")
        XCTAssertNil(ModelSuitability.chatWarning(for: "gemma-4-26b-a4b-it-coreai"))
        XCTAssertNotNil(ModelSuitability.chatWarning(for: "gemma-4-26b-a4b-coreai"))
    }

    func testExistingGemmaInstallDirectoryIsAliasedToRepairedServedName() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("models/exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)

        let oldDirectory = exports.appendingPathComponent("gemma-4-26b-a4b-coreai", isDirectory: true)
        try writeBundle(
            at: oldDirectory,
            name: "gemma-4-26b-a4b-coreai",
            sourceModelID: "google/gemma-4-26B-A4B-it",
            tokenizer: "google/gemma-4-26B-A4B-it")

        let manager = try ModelManager(exportsDir: exports, registryPath: registry)
        let rows = await manager.listModels()

        XCTAssertTrue(rows.contains { $0.name == "gemma-4-26b-a4b-it-coreai" })
        XCTAssertFalse(rows.contains { $0.name == "gemma-4-26b-a4b-coreai" })
        let oldNameResolved = await manager.resolveServedModelName("gemma-4-26b-a4b-coreai")
        let repairedNameResolved = await manager.resolveServedModelName("gemma-4-26b-a4b-it-coreai")
        XCTAssertEqual(oldNameResolved, "gemma-4-26b-a4b-it-coreai")
        XCTAssertEqual(repairedNameResolved, "gemma-4-26b-a4b-it-coreai")

        let error = await manager.deleteBundle("gemma-4-26b-a4b-it-coreai")
        XCTAssertNil(error)
        XCTAssertFalse(FileManager.default.fileExists(atPath: oldDirectory.path))
    }

    func testExplicitPrimaryStagedBundleAdvertisesCanonicalIDAndResolvesIntentionalAliases() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)

        let bundle = root.appendingPathComponent("gemma-4-31b-qat-staged", isDirectory: true)
        try writeBundle(
            at: exports.appendingPathComponent("qwen3-4b-instruct-coreai", isDirectory: true),
            name: "qwen3-4b-instruct-coreai")
        try writeStagedBundle(
            at: bundle,
            name: "gemma-4-31b-it-qat-q4_0-unquantized",
            sourceModelID: "google/gemma-4-31B-it-qat-q4_0-unquantized")
        let primary = try XCTUnwrap(PrimaryStagedBundleConfiguration.resolve(
            bundlePath: bundle.path,
            modelID: "google/gemma-4-31B-it"))

        let manager = try ModelManager(
            exportsDir: exports,
            registryPath: registry,
            primaryStagedBundle: primary)

        let rows = await manager.listModels()
        XCTAssertEqual(rows.map(\.name), ["google/gemma-4-31B-it", "qwen3-4b-instruct-coreai"])
        let preferred = await manager.servedModelsPreferredForChat()
        XCTAssertEqual(preferred.first?.name, "google/gemma-4-31B-it")
        for alias in [
            "google/gemma-4-31B-it",
            "gemma-4-31b-qat-staged",
            "gemma-4-31b-it-qat-q4_0-unquantized",
            "google/gemma-4-31B-it-qat-q4_0-unquantized",
        ] {
            let resolved = await manager.resolveServedModelName(alias)
            XCTAssertEqual(
                resolved,
                "google/gemma-4-31B-it",
                "expected alias \(alias) to resolve")
        }
    }

    func testExplicitPrimaryStagedBundleRejectsMissingNonStagedAndAliasCollisions() throws {
        let root = try makeTempDir()
        let missing = root.appendingPathComponent("missing", isDirectory: true)
        XCTAssertThrowsError(try PrimaryStagedBundleConfiguration.resolve(
            bundlePath: missing.path,
            modelID: "google/gemma-4-31B-it"))

        let ordinary = root.appendingPathComponent("ordinary", isDirectory: true)
        try writeBundle(at: ordinary, name: "ordinary")
        XCTAssertThrowsError(try PrimaryStagedBundleConfiguration.resolve(
            bundlePath: ordinary.path,
            modelID: "google/gemma-4-31B-it"))

        let exports = root.appendingPathComponent("exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)
        try writeBundle(
            at: exports.appendingPathComponent("google-gemma-4-31b-it", isDirectory: true),
            name: "google/gemma-4-31B-it")
        let staged = root.appendingPathComponent("staged", isDirectory: true)
        try writeStagedBundle(at: staged, name: "staged")
        let primary = try XCTUnwrap(PrimaryStagedBundleConfiguration.resolve(
            bundlePath: staged.path,
            modelID: "google/gemma-4-31B-it"))

        XCTAssertThrowsError(try ModelManager(
            exportsDir: exports,
            registryPath: registry,
            primaryStagedBundle: primary))
    }

    func testLargeGemmaPrewarmIsSkippedButSmallerModelsAreAllowed() {
        let largeGemma = ModelEntry(
            name: "gemma-4-26b-a4b-it-coreai",
            params: "26B",
            status: "available",
            bundle: true,
            memoryBytes: nil,
            mode: "standard")
        let smallerGemma = ModelEntry(
            name: "gemma-4-12b-it-coreai",
            params: "12B",
            status: "available",
            bundle: true,
            memoryBytes: nil,
            mode: "standard")
        let qwen = ModelEntry(
            name: "qwen3-4b-instruct-coreai",
            params: "4B",
            status: "available",
            bundle: true,
            memoryBytes: nil,
            mode: "standard")

        XCTAssertNotNil(ServerRuntime.prewarmSkipReason(for: largeGemma))
        XCTAssertNil(ServerRuntime.prewarmSkipReason(for: smallerGemma))
        XCTAssertNil(ServerRuntime.prewarmSkipReason(for: qwen))
    }

    func testLargeStagedGemmaPrewarmIsAllowedButMonolithicSafeguardRemains() {
        let staged = ModelEntry(
            name: "gemma-4-31b-it-caix",
            params: "31B",
            status: "available",
            bundle: true,
            memoryBytes: nil,
            mode: "staged")
        let monolithic = ModelEntry(
            name: "gemma-4-31b-it-coreai",
            params: "31B",
            status: "available",
            bundle: true,
            memoryBytes: nil,
            mode: "standard")

        XCTAssertNil(ServerRuntime.prewarmSkipReason(for: staged))
        XCTAssertNotNil(ServerRuntime.prewarmSkipReason(for: monolithic))
    }

    func testNativeStagedSnapshotProviderTracksSwapGrowthFromEachLoadBaseline() throws {
        let snapshots = LockedNativeSnapshots([
            .init(
                totalPhysicalMemoryBytes: 64 * 1_073_741_824,
                workerResidentBytes: 2 * 1_073_741_824,
                availableBytes: 40 * 1_073_741_824,
                pressure: .green,
                swapUsedBytes: 1_000),
            .init(
                totalPhysicalMemoryBytes: 64 * 1_073_741_824,
                workerResidentBytes: 3 * 1_073_741_824,
                availableBytes: 39 * 1_073_741_824,
                pressure: .green,
                swapUsedBytes: 1_512),
        ])
        let provider = ModelManager.makeStagedMemorySnapshotProvider {
            snapshots.next()
        }

        let actual = try provider()

        XCTAssertEqual(actual.totalPhysicalMemoryBytes, 64 * 1_073_741_824)
        XCTAssertEqual(actual.workerResidentBytes, 3 * 1_073_741_824)
        XCTAssertEqual(actual.availableBytes, 39 * 1_073_741_824)
        XCTAssertEqual(actual.pressure, .green)
        XCTAssertEqual(actual.swapGrowthBytes, 512)
    }

    func testNestedDraftBundleIsListedAsSpeculative() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)

        try writeBundle(
            at: exports.appendingPathComponent("rhm-qwen3-4b-mtp-caix", isDirectory: true),
            name: "qwen3-4b-coreai")
        try writeBundle(
            at: exports
                .appendingPathComponent("rhm-qwen3-4b-mtp-caix", isDirectory: true)
                .appendingPathComponent("draft", isDirectory: true),
            name: "qwen3-0.6b-coreai")

        let manager = try ModelManager(exportsDir: exports, registryPath: registry)
        let rows = await manager.listModels()

        let row = try XCTUnwrap(rows.first { $0.name == "rhm-qwen3-4b-mtp-caix" })
        XCTAssertEqual(row.mode, "speculative")
        XCTAssertTrue(row.bundle)
    }

    func testReasoningSupportIsReportedFromBundleTokenizer() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)

        try writeBundle(
            at: exports.appendingPathComponent("qwen3-4b-instruct-coreai", isDirectory: true),
            name: "qwen3-4b-instruct-coreai",
            chatTemplate: "<|im_start|>assistant\n<think>\n</think>")
        try writeBundle(
            at: exports.appendingPathComponent("plain-coreai", isDirectory: true),
            name: "plain-coreai")

        let manager = try ModelManager(exportsDir: exports, registryPath: registry)
        let rows = await manager.listModels()

        let qwen = try XCTUnwrap(rows.first { $0.name == "qwen3-4b-instruct-coreai" })
        let plain = try XCTUnwrap(rows.first { $0.name == "plain-coreai" })
        XCTAssertEqual(qwen.reasoningSupported, true)
        XCTAssertEqual(plain.reasoningSupported, false)
    }

    func testMultimodalStagedBundleReportsImageCapabilities() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)

        try writeMultimodalStagedBundle(
            at: exports.appendingPathComponent("gemma4-e2b-it-mm-staged", isDirectory: true),
            name: "gemma4-e2b-it-mm-staged")

        let manager = try ModelManager(exportsDir: exports, registryPath: registry)
        let rows = await manager.listModels()

        let row = try XCTUnwrap(rows.first { $0.name == "gemma4-e2b-it-mm-staged" })
        XCTAssertEqual(row.mode, "multimodal_staged")
        XCTAssertEqual(row.multimodalSupported, true)
        XCTAssertEqual(row.multimodalCapabilities?.family, "gemma4")
        XCTAssertEqual(row.multimodalCapabilities?.backend, "staged")
        XCTAssertEqual(row.multimodalCapabilities?.routeAvailable, true)
        XCTAssertEqual(row.multimodalCapabilities?.supportedModalities, ["image", "text"])
        XCTAssertEqual(row.multimodalCapabilities?.maxImages, 1)
        XCTAssertEqual(row.multimodalCapabilities?.imageSourceTypes, ["base64", "data_url"])
        XCTAssertEqual(row.multimodalCapabilities?.maxSoftTokensPerImage, 280)
        XCTAssertEqual(row.multimodalCapabilities?.supportedDecoding, ["greedy"])
        XCTAssertTrue(row.multimodalCapabilities?.unsupportedFeatures.contains("audio") == true)
        XCTAssertTrue(row.multimodalCapabilities?.unsupportedFeatures.contains("video") == true)
        XCTAssertTrue(row.multimodalCapabilities?.unsupportedFeatures.contains("remote_image_urls") == true)
    }

    func testMonolithicGemmaMultimodalBundleIsListedSeparatelyButBlockedForImageRoute() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)

        try writeMonolithicMultimodalGemmaBundle(
            at: exports.appendingPathComponent("gemma4-e2b-it-mm-monolithic", isDirectory: true),
            name: "gemma4-e2b-it-mm-monolithic")

        let manager = try ModelManager(exportsDir: exports, registryPath: registry)
        let rows = await manager.listModels()

        let row = try XCTUnwrap(rows.first { $0.name == "gemma4-e2b-it-mm-monolithic" })
        XCTAssertEqual(row.mode, "multimodal_monolithic")
        XCTAssertEqual(row.multimodalSupported, false)
        XCTAssertEqual(row.multimodalCapabilities?.family, "gemma4")
        XCTAssertEqual(row.multimodalCapabilities?.backend, "monolithic")
        XCTAssertEqual(row.multimodalCapabilities?.routeAvailable, false)
        XCTAssertEqual(row.multimodalCapabilities?.maxSoftTokensPerImage, 280)
        XCTAssertTrue(
            row.multimodalCapabilities?.unsupportedFeatures.contains("monolithic_prefill_runtime")
                == true)
    }

    func testEagleTargetDraftPackageIsListedAsEagle() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)

        let bundle = exports.appendingPathComponent("rhm-gemma-4-31b-it-mtp-caix", isDirectory: true)
        for child in ["eagle_target.aimodel", "eagle_draft.aimodel", "tokenizer"] {
            try FileManager.default.createDirectory(
                at: bundle.appendingPathComponent(child, isDirectory: true),
                withIntermediateDirectories: true)
        }

        let manager = try ModelManager(exportsDir: exports, registryPath: registry)
        let rows = await manager.listModels()

        let row = try XCTUnwrap(rows.first { $0.name == "rhm-gemma-4-31b-it-mtp-caix" })
        XCTAssertEqual(row.mode, "eagle")
        XCTAssertTrue(row.bundle)
    }

    func testDeleteBundleRefusesWhileHeavyTaskLockExists() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("models/exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)
        try writeBundle(
            at: exports.appendingPathComponent("gemma-4-31b-it-mtp-coreai", isDirectory: true),
            name: "gemma-4-31b-it-mtp-coreai")
        let lock = root.appendingPathComponent(".agent-heavy-task.lock")
        try "pid=123\n".write(to: lock, atomically: true, encoding: .utf8)

        let manager = try ModelManager(exportsDir: exports, registryPath: registry)
        let error = await manager.deleteBundle("gemma-4-31b-it-mtp-coreai")

        XCTAssertTrue(error?.contains("heavy-task lock exists") == true)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: exports.appendingPathComponent("gemma-4-31b-it-mtp-coreai").path))
    }

    func testDeleteBundleRemovesBundleWhenUnlocked() async throws {
        let root = try makeTempDir()
        let exports = root.appendingPathComponent("models/exports", isDirectory: true)
        let registry = root.appendingPathComponent("models/registry.json")
        try FileManager.default.createDirectory(
            at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try #"{"models":{}}"#.write(to: registry, atomically: true, encoding: .utf8)
        let bundle = exports.appendingPathComponent("qwen3-4b-coreai", isDirectory: true)
        try writeBundle(at: bundle, name: "qwen3-4b-coreai")

        let manager = try ModelManager(
            exportsDir: exports,
            registryPath: registry)
        let error = await manager.deleteBundle("qwen3-4b-coreai")
        XCTAssertNil(error)
        XCTAssertFalse(FileManager.default.fileExists(atPath: bundle.path))
    }

    private func writeBundle(
        at root: URL,
        name: String,
        sourceModelID: String? = nil,
        tokenizer: String? = nil,
        chatTemplate: String? = nil
    ) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let tokenizerName = tokenizer ?? name
        let sourceJSON = sourceModelID.map {
            """
              "source": {
                "hf_model_id": "\($0)"
              },
            """
        } ?? ""
        try """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "\(name)",
            \(sourceJSON)
              "assets": {"main": "\(name).aimodel"},
              "language": {
                "tokenizer": "\(tokenizerName)",
                "vocab_size": 151936,
                "max_context_length": 4096,
                "embedded_tokenizer": true
              }
            }
            """.write(to: root.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)
        if let chatTemplate {
            let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
            try FileManager.default.createDirectory(at: tokenizerDir, withIntermediateDirectories: true)
            try chatTemplate.write(
                to: tokenizerDir.appendingPathComponent("chat_template.jinja"),
                atomically: true,
                encoding: .utf8)
        }
    }

    private func writeMultimodalStagedBundle(at root: URL, name: String) throws {
        try writeBundle(at: root, name: name)
        try """
            {
              "schema": "caix.cluster.stage_manifest.v0",
              "model": "\(name)",
              "position_mode": "full_prefix",
              "stages": [],
              "multimodal": {
                "kind": "gemma4",
                "soft_tokens_per_image": 280,
                "embedder_asset": "gemma4-mm-embedder_float32.aimodel",
                "block_ids_required": false
              }
            }
            """.write(
                to: root.appendingPathComponent("stage-manifest.json"),
                atomically: true,
                encoding: .utf8)
    }

    private func writeStagedBundle(
        at root: URL,
        name: String,
        sourceModelID: String? = nil
    ) throws {
        try writeBundle(at: root, name: name, sourceModelID: sourceModelID)
        try """
            {
              "schema": "caix.cluster.stage_manifest.v0",
              "model": "\(name)",
              "position_mode": "full_prefix",
              "stages": []
            }
            """.write(
                to: root.appendingPathComponent("stage-manifest.json"),
                atomically: true,
                encoding: .utf8)
    }

    private func writeMonolithicMultimodalGemmaBundle(at root: URL, name: String) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "\(name)",
              "assets": {"main": "\(name).aimodel"},
              "language": {
                "tokenizer": "google/gemma-4-e2b-it",
                "vocab_size": 258944,
                "max_context_length": 131072,
                "embedded_tokenizer": true,
                "function_map": {
                  "main": ["main"],
                  "multimodal_prefill": ["prefill_multimodal"]
                }
              },
              "multimodal": {
                "kind": "gemma4_monolithic",
                "modalities": ["text", "image"],
                "max_images": 1,
                "soft_tokens_per_image": 280,
                "prefill_function": "prefill_multimodal",
                "vision_function": "embed_vision",
                "block_ids_required": true
              }
            }
            """.write(to: root.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-manager-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}

private final class LockedNativeSnapshots: @unchecked Sendable {
    private let lock = NSLock()
    private var snapshots: [ModelManager.NativeStagedMemorySnapshot]

    init(_ snapshots: [ModelManager.NativeStagedMemorySnapshot]) {
        self.snapshots = snapshots
    }

    func next() -> ModelManager.NativeStagedMemorySnapshot {
        lock.lock()
        defer { lock.unlock() }
        return snapshots.removeFirst()
    }
}
