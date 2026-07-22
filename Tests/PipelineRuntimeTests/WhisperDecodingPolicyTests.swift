import CryptoKit
import XCTest

@testable import PipelineRuntime

final class WhisperDecodingPolicyTests: XCTestCase {
    func testLoadsPinnedShapeAndBuildsForcedTranscriptionPrefix() throws {
        let policy = try authenticatedPolicy()

        XCTAssertEqual(policy.languageTokenID(for: "en"), 50_259)
        XCTAssertEqual(policy.languageTokenID(for: "<|es|>"), 50_262)
        XCTAssertEqual(
            try policy.forcedPrefix(languageTokenID: 50_259, includeTimestamps: false),
            [50_258, 50_259, 50_359, 50_363])
        XCTAssertEqual(policy.maximumSequenceLength, 448)
        XCTAssertEqual(policy.maximumTextTokenCount(includeTimestamps: false), 444)
    }

    func testLanguageDetectionConsidersOnlyLanguageTokens() throws {
        let policy = try authenticatedPolicy()
        var logits = [Float](repeating: -100, count: 51_865)
        logits[42] = 1_000
        logits[50_259] = 4
        logits[50_262] = 7

        XCTAssertEqual(try policy.detectLanguageToken(in: logits), 50_262)
    }

    func testGreedyTextSelectionAppliesPermanentBeginAndTimestampSuppression() throws {
        let policy = try authenticatedPolicy()
        var logits = [Float](repeating: -100, count: 51_865)
        logits[1] = 50                 // permanent suppression
        logits[220] = 40               // beginning-only suppression
        logits[50_364] = 30            // timestamp suppression
        logits[100] = 20               // valid text
        logits[50_257] = 10            // EOS, beginning-only suppression

        XCTAssertEqual(
            try policy.greedyTextToken(
                logits: logits, textPosition: 0, includeTimestamps: false),
            100)
        XCTAssertEqual(
            try policy.greedyTextToken(
                logits: logits, textPosition: 1, includeTimestamps: false),
            220)
        XCTAssertThrowsError(
            try policy.greedyTextToken(
                logits: logits, textPosition: 1, includeTimestamps: true)) { error in
            XCTAssertEqual(
                error as? WhisperDecodingPolicy.PolicyError,
                .timestampsUnsupported)
        }
    }

    func testRejectsWrongABIAndInvalidLogits() throws {
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixture()) as? [String: Any])
        object["max_length"] = 449
        XCTAssertThrowsError(
            try WhisperDecodingPolicy(
                authenticatedData: JSONSerialization.data(withJSONObject: object),
                expectedSHA256: digest(
                    JSONSerialization.data(withJSONObject: object))))

        let policy = try authenticatedPolicy()
        XCTAssertThrowsError(try policy.detectLanguageToken(in: [0]))
        XCTAssertThrowsError(
            try policy.greedyTextToken(
                logits: [Float](repeating: .nan, count: 51_865),
                textPosition: 0,
                includeTimestamps: false))
    }

    func testRejectsAlteredGenerationRulesBeforeDecodingJSON() throws {
        let original = fixture()
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: original) as? [String: Any])
        object["suppress_tokens"] = []
        let altered = try JSONSerialization.data(withJSONObject: object)

        XCTAssertThrowsError(
            try WhisperDecodingPolicy(
                authenticatedData: altered,
                expectedSHA256: digest(original))) { error in
            XCTAssertEqual(
                error as? WhisperDecodingPolicy.PolicyError,
                .generationConfigurationDigestMismatch)
        }
    }

    func testProductionInitializerUsesResidentModelLockDigest() {
        XCTAssertEqual(
            WhisperDecodingPolicy.canonicalGenerationConfigSHA256,
            ResidentModelLock.approvedWhisperGenerationConfigSHA256)
        XCTAssertEqual(
            WhisperDecodingPolicy.canonicalGenerationConfigSHA256,
            "031721643aab5be7250eb668c6b9b5c67d2549420522ac1291bfd346bfff6297")
    }

    private func authenticatedPolicy() throws -> WhisperDecodingPolicy {
        let data = fixture()
        return try WhisperDecodingPolicy(
            authenticatedData: data,
            expectedSHA256: digest(data))
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func fixture() -> Data {
        Data(
            #"""
            {
              "decoder_start_token_id": 50258,
              "eos_token_id": 50257,
              "pad_token_id": 50257,
              "forced_decoder_ids": [[1, null], [2, 50359]],
              "suppress_tokens": [1, 2, 7, 50258, 50358, 50359, 50360, 50361, 50362],
              "begin_suppress_tokens": [220, 50257],
              "no_timestamps_token_id": 50363,
              "max_length": 448,
              "lang_to_id": {"<|en|>": 50259, "<|es|>": 50262},
              "task_to_id": {"transcribe": 50359, "translate": 50358}
            }
            """#.utf8)
    }
}
