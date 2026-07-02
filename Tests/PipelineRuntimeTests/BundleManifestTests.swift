import XCTest

@testable import PipelineRuntime

final class BundleManifestTests: XCTestCase {
    func testLegacyCoreAIAssetMetadataDefaultsMissingBundleFields() throws {
        let data = #"{"assetVersion":"1.0"}"#.data(using: .utf8)!
        let manifest = try JSONDecoder().decode(BundleManifest.self, from: data)

        XCTAssertEqual(manifest.metadataVersion, "legacy-coreai-asset")
        XCTAssertEqual(manifest.kind, "coreai_asset")
        XCTAssertEqual(manifest.name, "legacy-coreai-asset")
        XCTAssertEqual(manifest.assets.primary, ".")
        XCTAssertNil(manifest.language)
    }

    func testExplicitMinKVCapacityWins() throws {
        let root = try makeBundle(
            name: "ornith-1.0-9b-coreai",
            hfModelId: "deepreinforce-ai/Ornith-1.0-9B",
            tokenizer: "deepreinforce-ai/Ornith-1.0-9B",
            minKVCapacity: 640)

        let bundle = try ResolvedBundle.load(at: root.path)
        XCTAssertEqual(bundle.minKVCapacity, 640)
    }

    func testRegistryMinKVCapacityWins() throws {
        let base = try makeTempDir()
        let root = base.appendingPathComponent("exports/qwythos-9b-coreai", isDirectory: true)
        try makeBundle(
            at: root,
            name: "qwythos-9b-coreai",
            hfModelId: "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
            tokenizer: "empero-ai/Qwythos-9B-Claude-Mythos-5-1M")
        let models = base.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        try """
            {"models":{"Qwythos-9B":{"hf_repo":"empero-ai/Qwythos-9B-Claude-Mythos-5-1M","model_type":"qwen3_5","min_kv_capacity":768}}}
            """.write(to: models.appendingPathComponent("registry.json"), atomically: true, encoding: .utf8)

        let bundle = try ResolvedBundle.load(at: root.path)
        XCTAssertEqual(bundle.minKVCapacity, 768)
    }

    func testOrnithInfersQwen35FloorWithoutRegistry() throws {
        let root = try makeBundle(
            name: "ornith-1.0-9b-coreai",
            hfModelId: "deepreinforce-ai/Ornith-1.0-9B",
            tokenizer: "deepreinforce-ai/Ornith-1.0-9B")

        let bundle = try ResolvedBundle.load(at: root.path)
        XCTAssertEqual(bundle.minKVCapacity, 1024)
    }

    func testLanguageFunctionMapDecodesRoles() throws {
        let root = try makeBundle(
            name: "ornith-1.0-35b-coreai",
            hfModelId: "deepreinforce-ai/Ornith-1.0-35B",
            tokenizer: "deepreinforce-ai/Ornith-1.0-35B",
            functionMap: #"{"main":["main"],"decode":["decode"]}"#,
            decodeAsset: "decode.aimodel")

        let bundle = try ResolvedBundle.load(at: root.path)
        XCTAssertEqual(bundle.manifest.language?.functionMap?.name(for: "main"), "main")
        XCTAssertEqual(bundle.manifest.language?.functionMap?.name(for: "decode"), "decode")
        XCTAssertEqual(bundle.decodeAimodelURL?.lastPathComponent, "decode.aimodel")
    }

    func testStandardQwenKeepsZeroFloorWithoutRegistry() throws {
        let root = try makeBundle(
            name: "qwen3-4b-coreai",
            hfModelId: "Qwen/Qwen3-4B",
            tokenizer: "Qwen/Qwen3-4B")

        let bundle = try ResolvedBundle.load(at: root.path)
        XCTAssertEqual(bundle.minKVCapacity, 0)
    }

    @discardableResult
    private func makeBundle(
        name: String,
        hfModelId: String,
        tokenizer: String,
        minKVCapacity: Int? = nil,
        functionMap: String? = nil,
        decodeAsset: String? = nil
    ) throws -> URL {
        let root = try makeTempDir().appendingPathComponent(name, isDirectory: true)
        try makeBundle(
            at: root, name: name, hfModelId: hfModelId, tokenizer: tokenizer,
            minKVCapacity: minKVCapacity, functionMap: functionMap, decodeAsset: decodeAsset)
        return root
    }

    private func makeBundle(
        at root: URL,
        name: String,
        hfModelId: String,
        tokenizer: String,
        minKVCapacity: Int? = nil,
        functionMap: String? = nil,
        decodeAsset: String? = nil
    ) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        try fm.createDirectory(at: root.appendingPathComponent("model.aimodel"), withIntermediateDirectories: true)
        if let decodeAsset {
            try fm.createDirectory(at: root.appendingPathComponent(decodeAsset), withIntermediateDirectories: true)
        }
        let tokDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        try fm.createDirectory(at: tokDir, withIntermediateDirectories: true)
        try "{}".write(to: tokDir.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)

        let minField = minKVCapacity.map { #","min_kv_capacity":\#($0)"# } ?? ""
        let functionMapField = functionMap.map { #","function_map":\#($0)"# } ?? ""
        let decodeAssetField = decodeAsset.map { #","decode":"\#($0)""# } ?? ""
        let json = """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "\(name)",
              "assets": {"main": "model.aimodel"\(decodeAssetField)},
              "language": {
                "tokenizer": "\(tokenizer)",
                "vocab_size": 248320,
                "max_context_length": 8192,
                "embedded_tokenizer": true\(minField)\(functionMapField)
              },
              "source": {
                "model_definition": "torch",
                "hf_model_id": "\(hfModelId)"
              }
            }
            """
        try json.write(to: root.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-bundle-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
