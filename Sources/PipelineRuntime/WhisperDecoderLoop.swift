import Foundation

/// Pure host-side orchestration for Whisper's incremental decoder. Core AI owns the encoder and
/// fixed K/V state; this loop owns the required language/task prefix and deterministic token
/// selection so the behavior can be verified without compiling a multi-gigabyte model.
public enum WhisperDecoderLoop {
    public struct Result: Sendable, Equatable {
        public var textTokenIDs: [Int32]
        public var language: String
        public var languageTokenID: Int32
        public var reachedEndToken: Bool
        public var wasTruncated: Bool
    }

    public static func run(
        policy: WhisperDecodingPolicy,
        requestedLanguage: String?,
        includeTimestamps: Bool,
        step: (Int32) async throws -> [Float],
        onTextToken: ((Int32) -> Void)? = nil
    ) async throws -> Result {
        guard !includeTimestamps else {
            throw WhisperDecodingPolicy.PolicyError.timestampsUnsupported
        }
        try Task.checkCancellation()
        let languageLogits = try await step(policy.decoderStartTokenID)
        try Task.checkCancellation()

        let languageToken: Int32
        if let requestedLanguage {
            languageToken = try policy.requireLanguageTokenID(for: requestedLanguage)
        } else {
            languageToken = try policy.detectLanguageToken(in: languageLogits)
        }

        _ = try await step(languageToken)
        try Task.checkCancellation()
        var logits = try await step(policy.transcribeTokenID)
        try Task.checkCancellation()
        if !includeTimestamps {
            logits = try await step(policy.noTimestampsTokenID)
            try Task.checkCancellation()
        }

        var textTokens: [Int32] = []
        let limit = policy.maximumTextTokenCount(includeTimestamps: includeTimestamps)
        var reachedEnd = false
        for position in 0..<limit {
            let token = try policy.greedyTextToken(
                logits: logits,
                textPosition: position,
                includeTimestamps: includeTimestamps)
            if token == policy.endTokenID {
                reachedEnd = true
                break
            }
            textTokens.append(token)
            onTextToken?(token)
            try Task.checkCancellation()
            if position + 1 < limit {
                logits = try await step(token)
                try Task.checkCancellation()
            }
        }

        try Task.checkCancellation()

        return Result(
            textTokenIDs: textTokens,
            language: try policy.languageCode(for: languageToken),
            languageTokenID: languageToken,
            reachedEndToken: reachedEnd,
            wasTruncated: !reachedEnd)
    }
}
