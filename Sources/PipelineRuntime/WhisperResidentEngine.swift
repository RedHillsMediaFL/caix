import Foundation

/// Native session boundary used by the resident Whisper host.
///
/// A session owns the large mutable decoder state for exactly one audio window. Implementations
/// must keep encoded features, cross-KV payloads, token IDs, and decoder state in memory only.
protocol WhisperNativeSession: AnyObject, Sendable {
    func encode(inputFeatures: [Float16]) async throws
    func loadCrossKV() async throws
    func step(tokenID: Int32) async throws -> [Float]
    func finish() async
}

/// Factory retained beside one specialized Core AI model. Every call creates fresh request state,
/// but never specializes or loads a second model.
protocol WhisperNativeSessionFactory: AnyObject, Sendable {
    func makeSession() async throws -> any WhisperNativeSession
}

/// Small tokenizer seam so the resident orchestration is testable without loading the real
/// tokenizer or the multi-gigabyte Core AI asset.
protocol WhisperTextDecoding: Sendable {
    func decode(textTokenIDs: [Int32]) -> String
}

/// Resident, cancellation-aware Whisper orchestration.
///
/// Swift actors are reentrant across `await`, so actor isolation alone does not stop a second
/// request from allocating another 300+ MiB decoder state. The explicit FIFO admission gate below
/// keeps exactly one native session alive while still allowing queued callers to cancel promptly.
public actor WhisperResidentEngine {
    public static let featureCount = 1 * 80 * 3_000

    public enum EngineError: Error, Sendable, Equatable, CustomStringConvertible {
        case invalidFeatureCount(expected: Int, actual: Int)

        public var description: String {
            switch self {
            case .invalidFeatureCount(let expected, let actual):
                return "Whisper input requires \(expected) Float16 features; got \(actual)"
            }
        }
    }

    public struct Timings: Sendable, Equatable {
        public var queueSeconds: Double
        public var encodeSeconds: Double
        public var crossKVLoadSeconds: Double
        public var decodeSeconds: Double
        public var totalSeconds: Double

        public init(
            queueSeconds: Double,
            encodeSeconds: Double,
            crossKVLoadSeconds: Double,
            decodeSeconds: Double,
            totalSeconds: Double
        ) {
            self.queueSeconds = queueSeconds
            self.encodeSeconds = encodeSeconds
            self.crossKVLoadSeconds = crossKVLoadSeconds
            self.decodeSeconds = decodeSeconds
            self.totalSeconds = totalSeconds
        }
    }

    public struct Result: Sendable, Equatable {
        public var text: String
        public var textTokenIDs: [Int32]
        public var language: String
        public var languageTokenID: Int32
        public var reachedEndToken: Bool
        public var wasTruncated: Bool
        public var timings: Timings

        public init(
            text: String,
            textTokenIDs: [Int32],
            language: String,
            languageTokenID: Int32,
            reachedEndToken: Bool,
            wasTruncated: Bool,
            timings: Timings
        ) {
            self.text = text
            self.textTokenIDs = textTokenIDs
            self.language = language
            self.languageTokenID = languageTokenID
            self.reachedEndToken = reachedEndToken
            self.wasTruncated = wasTruncated
            self.timings = timings
        }
    }

    private let factory: any WhisperNativeSessionFactory
    private let policy: WhisperDecodingPolicy
    private let textDecoder: any WhisperTextDecoding
    private let expectedFeatureCount: Int
    private let admission = WhisperInferenceAdmission()

    init(
        factory: any WhisperNativeSessionFactory,
        policy: WhisperDecodingPolicy,
        textDecoder: any WhisperTextDecoding,
        expectedFeatureCount: Int = WhisperResidentEngine.featureCount
    ) {
        precondition(expectedFeatureCount > 0)
        self.factory = factory
        self.policy = policy
        self.textDecoder = textDecoder
        self.expectedFeatureCount = expectedFeatureCount
    }

    public func transcribe(
        inputFeatures: [Float16],
        requestedLanguage: String?,
        includeTimestamps: Bool = false,
        onTextToken: (@Sendable (Int32) -> Void)? = nil
    ) async throws -> Result {
        guard inputFeatures.count == expectedFeatureCount else {
            throw EngineError.invalidFeatureCount(
                expected: expectedFeatureCount,
                actual: inputFeatures.count)
        }

        let totalStart = ContinuousClock.now
        let queueStart = ContinuousClock.now
        try await admission.acquire()
        let queueSeconds = Self.seconds(since: queueStart)

        var session: (any WhisperNativeSession)?
        do {
            try Task.checkCancellation()
            let created = try await factory.makeSession()
            session = created

            let encodeStart = ContinuousClock.now
            try await created.encode(inputFeatures: inputFeatures)
            let encodeSeconds = Self.seconds(since: encodeStart)
            try Task.checkCancellation()

            let loadStart = ContinuousClock.now
            try await created.loadCrossKV()
            let crossKVLoadSeconds = Self.seconds(since: loadStart)
            try Task.checkCancellation()

            let decodeStart = ContinuousClock.now
            let decoded = try await WhisperDecoderLoop.run(
                policy: policy,
                requestedLanguage: requestedLanguage,
                includeTimestamps: includeTimestamps,
                step: { tokenID in
                    try await created.step(tokenID: tokenID)
                },
                onTextToken: onTextToken)
            let decodeSeconds = Self.seconds(since: decodeStart)
            try Task.checkCancellation()

            let text = textDecoder.decode(textTokenIDs: decoded.textTokenIDs)
            await created.finish()
            session = nil
            await admission.release()
            return Result(
                text: text,
                textTokenIDs: decoded.textTokenIDs,
                language: decoded.language,
                languageTokenID: decoded.languageTokenID,
                reachedEndToken: decoded.reachedEndToken,
                wasTruncated: decoded.wasTruncated,
                timings: Timings(
                    queueSeconds: queueSeconds,
                    encodeSeconds: encodeSeconds,
                    crossKVLoadSeconds: crossKVLoadSeconds,
                    decodeSeconds: decodeSeconds,
                    totalSeconds: Self.seconds(since: totalStart)))
        } catch {
            if let session { await session.finish() }
            await admission.release()
            throw error
        }
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }
}

private actor WhisperInferenceAdmission {
    private struct Waiter {
        var id: UUID
        var continuation: CheckedContinuation<Void, Error>
    }

    private var occupied = false
    private var waiters: [Waiter] = []

    func acquire() async throws {
        try Task.checkCancellation()
        guard occupied else {
            occupied = true
            return
        }

        let id = UUID()
        try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                } else {
                    waiters.append(Waiter(id: id, continuation: continuation))
                }
            }
        }, onCancel: {
            Task { await self.cancel(id: id) }
        })
    }

    func release() {
        precondition(occupied, "released an unoccupied Whisper admission gate")
        if waiters.isEmpty {
            occupied = false
        } else {
            let waiter = waiters.removeFirst()
            waiter.continuation.resume()
        }
    }

    private func cancel(id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }
}
