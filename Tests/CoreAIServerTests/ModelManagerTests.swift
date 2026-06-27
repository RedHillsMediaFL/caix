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
