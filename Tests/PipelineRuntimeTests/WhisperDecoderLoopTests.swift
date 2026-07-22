import XCTest

@testable import PipelineRuntime

final class WhisperDecoderLoopTests: XCTestCase {
    func testAutoLanguageForcedPrefixAndGreedyTextLoop() async throws {
        let policy = try WhisperDecodingPolicy(data: fixture())
        var consumed: [Int32] = []
        var streamed: [Int32] = []

        let result = try await WhisperDecoderLoop.run(
            policy: policy,
            requestedLanguage: nil,
            includeTimestamps: false,
            step: { token in
                consumed.append(token)
                switch token {
                case 50_258: return Self.logits([(50_259, 9), (50_262, 7)])
                case 50_363: return Self.logits([(100, 9)])
                case 100: return Self.logits([(101, 9)])
                case 101: return Self.logits([(50_257, 9)])
                default: return Self.logits([(42, 9)])
                }
            },
            onTextToken: { streamed.append($0) })

        XCTAssertEqual(consumed, [50_258, 50_259, 50_359, 50_363, 100, 101])
        XCTAssertEqual(streamed, [100, 101])
        XCTAssertEqual(result.textTokenIDs, [100, 101])
        XCTAssertEqual(result.language, "en")
        XCTAssertEqual(result.languageTokenID, 50_259)
        XCTAssertTrue(result.reachedEndToken)
        XCTAssertFalse(result.wasTruncated)
    }

    func testExplicitLanguageOverridesDetectionAndTimestampPrefix() async throws {
        let policy = try WhisperDecodingPolicy(data: fixture())
        var consumed: [Int32] = []
        let result = try await WhisperDecoderLoop.run(
            policy: policy,
            requestedLanguage: "es",
            includeTimestamps: true,
            step: { token in
                consumed.append(token)
                if token == 50_359 { return Self.logits([(100, 10)]) }
                if token == 100 { return Self.logits([(50_257, 10)]) }
                return Self.logits([(50_259, 100)])
            })

        XCTAssertEqual(consumed, [50_258, 50_262, 50_359, 100])
        XCTAssertEqual(result.language, "es")
        XCTAssertEqual(result.textTokenIDs, [100])
        XCTAssertTrue(result.reachedEndToken)
    }

    func testCancellationIsCheckedBetweenDecoderSteps() async throws {
        let policy = try WhisperDecodingPolicy(data: fixture())
        let task = Task.detached { @Sendable [policy] in
            try await WhisperDecoderLoop.run(
                policy: policy,
                requestedLanguage: "en",
                includeTimestamps: false,
                step: { _ in
                    try await Task.sleep(for: .seconds(5))
                    return Self.logits([(100, 1)])
                })
        }
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch is CancellationError {
            // expected
        }
    }

    private static func logits(_ values: [(Int, Float)]) -> [Float] {
        var result = [Float](repeating: -100, count: WhisperDecodingPolicy.vocabularySize)
        for (token, value) in values { result[token] = value }
        return result
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
