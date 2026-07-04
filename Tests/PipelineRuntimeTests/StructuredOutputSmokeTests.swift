import Foundation
import XCTest

@testable import PipelineRuntime

final class StructuredOutputSmokeTests: XCTestCase {
    func testRealCoreAILMStructuredOutputSmoke() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["CAIX_STRUCTURED_OUTPUT_MODEL"],
              !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_STRUCTURED_OUTPUT_MODEL to a CoreAILM language bundle")
        }

        #if COREAI_RUNTIME
        let schema = """
        {"type":"object","additionalProperties":false,"properties":{"answer":{"type":"string"}},"required":["answer"]}
        """
        let prompt = ProcessInfo.processInfo.environment["CAIX_STRUCTURED_OUTPUT_PROMPT"]
            ?? "Return a JSON object with one field named answer containing the word ok."
        let maxTokens = max(
            8,
            Int(ProcessInfo.processInfo.environment["CAIX_STRUCTURED_OUTPUT_MAX_TOKENS"] ?? "")
                ?? 64)

        let model = try await PersistentModel.load(bundlePath: modelPath, verbose: true)
        XCTAssertTrue(model.supportsConstrainedDecoding)
        let result = try await model.generate(
            messages: [["role": "user", "content": prompt]],
            options: CoreAIPipeline.Options(
                maxTokens: maxTokens,
                temperature: 0,
                applyChatTemplate: true,
                constrainedJSONSchema: schema,
                verbose: true),
            tools: nil,
            onToken: nil)

        XCTAssertGreaterThan(result.generatedTokenCount, 0)
        let trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let data = try XCTUnwrap(trimmed.data(using: .utf8))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any],
            "constrained output was not a JSON object: \(trimmed)")
        XCTAssertEqual(Set(object.keys), ["answer"])
        XCTAssertNotNil(object["answer"] as? String)

        if let outputPath = ProcessInfo.processInfo.environment["CAIX_STRUCTURED_OUTPUT_SMOKE_OUTPUT"],
           !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            let payload: [String: Any] = [
                "backend": "PersistentModel",
                "model": modelPath,
                "prompt": prompt,
                "schema": schema,
                "text": trimmed,
                "prompt_token_count": result.promptTokenCount,
                "generated_token_count": result.generatedTokenCount,
                "stop_reason": result.stopReason.rawValue,
                "decode_tokens_per_second": result.decodeTokensPerSecond,
            ]
            let outputData = try JSONSerialization.data(
                withJSONObject: payload,
                options: [.prettyPrinted, .sortedKeys])
            try outputData.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }
}
