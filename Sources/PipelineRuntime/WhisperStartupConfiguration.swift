import Foundation

/// Validated, all-or-none startup inputs for one resident Whisper engine.
///
/// Keeping validation in `PipelineRuntime` gives the CLI a typed boundary and makes a partially
/// configured transcription endpoint impossible. Paths remain explicit: CAIX never guesses a
/// model, tokenizer, or provenance-lock location.
public struct WhisperStartupConfiguration: Sendable, Equatable {
    public enum ConfigurationError: Error, Sendable, Equatable, CustomStringConvertible {
        case incomplete(missingFlags: [String])
        case emptyValue(flag: String)
        case invalidMaximumQueuedRequests(Int)
        case runtimeUnavailable

        public var description: String {
            switch self {
            case .incomplete(let missingFlags):
                return "resident Whisper requires all of --whisper-asset, --whisper-tokenizer, and --resident-model-lock; missing \(missingFlags.joined(separator: ", "))"
            case .emptyValue(let flag):
                return "\(flag) requires a non-empty path"
            case .invalidMaximumQueuedRequests(let value):
                return "--whisper-max-queued must be a non-negative integer; got \(value)"
            case .runtimeUnavailable:
                return "resident Whisper requires a Core AI runtime-linked caix build; rebuild with COREAI_DIRECT_RUNTIME=1"
            }
        }
    }

    public let assetURL: URL
    public let tokenizerDirectory: URL
    public let modelLockURL: URL
    public let maximumQueuedRequests: Int

    private init(
        assetURL: URL,
        tokenizerDirectory: URL,
        modelLockURL: URL,
        maximumQueuedRequests: Int
    ) {
        self.assetURL = assetURL
        self.tokenizerDirectory = tokenizerDirectory
        self.modelLockURL = modelLockURL
        self.maximumQueuedRequests = maximumQueuedRequests
    }

    /// Return `nil` when resident transcription is not requested, or one fully validated
    /// configuration when it is. Supplying the queue bound alone counts as requesting Whisper and
    /// therefore requires all three authenticated artifact paths.
    public static func resolve(
        assetPath: String?,
        tokenizerPath: String?,
        modelLockPath: String?,
        maximumQueuedRequests: Int?
    ) throws -> WhisperStartupConfiguration? {
        let optionsWereSupplied =
            assetPath != nil || tokenizerPath != nil || modelLockPath != nil
                || maximumQueuedRequests != nil
        guard optionsWereSupplied else { return nil }

        var missingFlags: [String] = []
        if assetPath == nil { missingFlags.append("--whisper-asset") }
        if tokenizerPath == nil { missingFlags.append("--whisper-tokenizer") }
        if modelLockPath == nil { missingFlags.append("--resident-model-lock") }
        guard missingFlags.isEmpty else {
            throw ConfigurationError.incomplete(missingFlags: missingFlags)
        }
        guard let assetPath, let tokenizerPath, let modelLockPath else {
            // The missing-flag guard above makes this unreachable, but keep construction free of
            // force unwraps if that validation changes later.
            throw ConfigurationError.incomplete(missingFlags: missingFlags)
        }

        let paths = [
            ("--whisper-asset", assetPath),
            ("--whisper-tokenizer", tokenizerPath),
            ("--resident-model-lock", modelLockPath),
        ]
        if let empty = paths.first(where: { $0.1.isEmpty }) {
            throw ConfigurationError.emptyValue(flag: empty.0)
        }

        let queueBound = maximumQueuedRequests
            ?? WhisperResidentEngine.defaultMaximumQueuedRequests
        guard queueBound >= 0 else {
            throw ConfigurationError.invalidMaximumQueuedRequests(queueBound)
        }

        return WhisperStartupConfiguration(
            assetURL: URL(fileURLWithPath: assetPath, isDirectory: true).standardizedFileURL,
            tokenizerDirectory: URL(fileURLWithPath: tokenizerPath, isDirectory: true)
                .standardizedFileURL,
            modelLockURL: URL(fileURLWithPath: modelLockPath, isDirectory: false)
                .standardizedFileURL,
            maximumQueuedRequests: queueBound)
    }

    /// Authenticate every model and tokenizer byte and specialize exactly one native engine.
    public func loadResidentEngine() async throws -> WhisperResidentEngine {
        #if COREAI_RUNTIME
        return try await WhisperResidentEngine.load(
            assetURL: assetURL,
            tokenizerDirectory: tokenizerDirectory,
            modelLockURL: modelLockURL,
            maximumQueuedRequests: maximumQueuedRequests)
        #else
        throw ConfigurationError.runtimeUnavailable
        #endif
    }
}
