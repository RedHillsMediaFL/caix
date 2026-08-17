import Foundation
import XCTest

@testable import PipelineRuntime

final class TextStagedRealAssetTests: XCTestCase {
    func testQwythosShortSequenceMatchesHFOracle() async throws {
        #if COREAI_RUNTIME
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["COREAI_QWYTHOS_STAGED_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("set COREAI_QWYTHOS_STAGED_BUNDLE to run the Qwythos real-asset gate")
        }
        guard let oraclePath = env["COREAI_QWYTHOS_HF_ORACLE"], !oraclePath.isEmpty else {
            throw XCTSkip("set COREAI_QWYTHOS_HF_ORACLE to run the Qwythos real-asset gate")
        }

        let oracleData = try Data(contentsOf: URL(fileURLWithPath: oraclePath))
        guard
            let oracle = try JSONSerialization.jsonObject(with: oracleData) as? [String: Any],
            let promptIDs = oracle["prompt_ids"] as? [Int],
            let expectedTokens = oracle["generated_token_ids"] as? [Int]
        else {
            XCTFail("malformed HF oracle JSON at \(oraclePath)")
            return
        }
        XCTAssertFalse(promptIDs.isEmpty)
        XCTAssertFalse(expectedTokens.isEmpty)

        let manifestURL = URL(fileURLWithPath: bundlePath, isDirectory: true)
            .appendingPathComponent("stage-manifest.json")
        let model = try await TextStagedModel.load(manifestURL: manifestURL)
        let result = try await model.generate(
            promptTokenIDs: promptIDs.map(Int32.init),
            options: CoreAIPipeline.Options(
                maxTokens: expectedTokens.count,
                temperature: 0,
                applyChatTemplate: false,
                kvCapacity: 512))

        XCTAssertEqual(result.generatedTokenIDs.map(Int.init), expectedTokens)
        XCTAssertEqual(result.promptTokenCount, promptIDs.count)
        XCTAssertEqual(result.generatedTokenCount, expectedTokens.count)
        #else
        throw XCTSkip("Core AI runtime not linked")
        #endif
    }

    func testQwythosLongContextSmoke() async throws {
        #if COREAI_RUNTIME
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["COREAI_QWYTHOS_STAGED_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("set COREAI_QWYTHOS_STAGED_BUNDLE to run the Qwythos real-asset gate")
        }
        guard
            let rawCount = env["COREAI_QWYTHOS_LONG_PROMPT_TOKENS"],
            let promptCount = Int(rawCount),
            promptCount > 0
        else {
            throw XCTSkip("set COREAI_QWYTHOS_LONG_PROMPT_TOKENS to run the Qwythos long-context gate")
        }
        let tokenID = Int32(env["COREAI_QWYTHOS_LONG_TOKEN_ID"].flatMap(Int.init) ?? 198)
        let kvCapacity = Int(env["COREAI_QWYTHOS_LONG_KV_CAPACITY"] ?? "") ?? 1_048_576
        XCTAssertLessThanOrEqual(promptCount, kvCapacity)

        let manifestURL = URL(fileURLWithPath: bundlePath, isDirectory: true)
            .appendingPathComponent("stage-manifest.json")
        let model = try await TextStagedModel.load(manifestURL: manifestURL)
        let prompt = Array(repeating: tokenID, count: promptCount)
        let started = Date()
        let result = try await model.generate(
            promptTokenIDs: prompt,
            options: CoreAIPipeline.Options(
                maxTokens: 1,
                temperature: 0,
                applyChatTemplate: false,
                kvCapacity: kvCapacity))
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertEqual(result.promptTokenCount, promptCount)
        XCTAssertEqual(result.generatedTokenCount, 1)
        XCTAssertLessThanOrEqual(result.promptTokenCount + result.generatedTokenCount, kvCapacity)
        print(
            "QWYTHOS_LONG_CONTEXT prompt_tokens=\(promptCount) generated=\(result.generatedTokenIDs.map(Int.init)) "
                + String(format: "prefill=%.3fs decode=%.3fs total=%.3fs", result.prefillSeconds, result.decodeSeconds, elapsed))
        #else
        throw XCTSkip("Core AI runtime not linked")
        #endif
    }

}
