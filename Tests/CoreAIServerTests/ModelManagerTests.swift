import XCTest

@testable import CoreAIServer

final class ModelManagerTests: XCTestCase {
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

        let manager = ModelManager(exportsDir: exports, registryPath: registry)
        let rows = await manager.listModels()

        let row = try XCTUnwrap(rows.first { $0.name == "rhm-qwen3-4b-mtp-caix" })
        XCTAssertEqual(row.mode, "speculative")
        XCTAssertTrue(row.bundle)
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

        let manager = ModelManager(exportsDir: exports, registryPath: registry)
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

        let manager = ModelManager(exportsDir: exports, registryPath: registry)
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

        let manager = ModelManager(
            exportsDir: exports,
            registryPath: registry)
        let error = await manager.deleteBundle("qwen3-4b-coreai")
        XCTAssertNil(error)
        XCTAssertFalse(FileManager.default.fileExists(atPath: bundle.path))
    }

    private func writeBundle(at root: URL, name: String) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "\(name)",
              "assets": {"main": "\(name).aimodel"},
              "language": {
                "tokenizer": "\(name)",
                "vocab_size": 151936,
                "max_context_length": 4096,
                "embedded_tokenizer": true
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
