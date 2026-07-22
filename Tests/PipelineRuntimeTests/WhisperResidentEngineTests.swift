import CryptoKit
import Foundation
import XCTest

@testable import PipelineRuntime

final class WhisperResidentEngineTests: XCTestCase {
    func testTranscriptionRunsEncodeLoadAndDecodeInOrderWithFreshState() async throws {
        let probe = NativeProbe(delayNanoseconds: 0)
        let engine = WhisperResidentEngine(
            factory: probe,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 4)

        let first = try await engine.transcribe(
            inputFeatures: [1, 2, 3, 4],
            requestedLanguage: "en")
        let second = try await engine.transcribe(
            inputFeatures: [5, 6, 7, 8],
            requestedLanguage: "en")

        XCTAssertEqual(first.text, "hello")
        XCTAssertEqual(first.textTokenIDs, [100])
        XCTAssertEqual(first.language, "en")
        XCTAssertTrue(first.reachedEndToken)
        XCTAssertFalse(first.wasTruncated)
        XCTAssertEqual(second.text, "hello")
        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.sessionsCreated, 2)
        XCTAssertEqual(snapshot.sessionsFinished, 2)
        XCTAssertEqual(snapshot.maximumConcurrentSessions, 1)
        XCTAssertEqual(snapshot.events, [
            "session:1", "encode:1", "load:1",
            "step:1:50258", "step:1:50259", "step:1:50359",
            "step:1:50363", "step:1:100", "finish:1",
            "session:2", "encode:2", "load:2",
            "step:2:50258", "step:2:50259", "step:2:50359",
            "step:2:50363", "step:2:100", "finish:2",
        ])
    }

    func testConcurrentRequestsNeverOwnTwoNativeSessions() async throws {
        let probe = NativeProbe(delayNanoseconds: 20_000_000)
        let engine = WhisperResidentEngine(
            factory: probe,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 1)

        async let first = engine.transcribe(
            inputFeatures: [0], requestedLanguage: "en")
        async let second = engine.transcribe(
            inputFeatures: [0], requestedLanguage: "en")
        _ = try await (first, second)

        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.sessionsCreated, 2)
        XCTAssertEqual(snapshot.sessionsFinished, 2)
        XCTAssertEqual(snapshot.maximumConcurrentSessions, 1)
    }

    func testCancelledQueuedRequestDoesNotCreateStateOrBlockNextRequest() async throws {
        let probe = NativeProbe(delayNanoseconds: 40_000_000)
        let engine = WhisperResidentEngine(
            factory: probe,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 1)

        let first = Task {
            try await engine.transcribe(inputFeatures: [0], requestedLanguage: "en")
        }
        await probe.waitUntilSessionCount(1)
        let cancelled = Task {
            try await engine.transcribe(inputFeatures: [0], requestedLanguage: "en")
        }
        await engine.waitUntilQueuedRequestCountForTesting(1)
        cancelled.cancel()
        let third = Task {
            try await engine.transcribe(inputFeatures: [0], requestedLanguage: "en")
        }

        _ = try await first.value
        do {
            _ = try await cancelled.value
            XCTFail("queued cancellation unexpectedly entered native inference")
        } catch is CancellationError {
            // Expected.
        }
        _ = try await third.value

        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.sessionsCreated, 2)
        XCTAssertEqual(snapshot.sessionsFinished, 2)
        XCTAssertEqual(snapshot.maximumConcurrentSessions, 1)
    }

    func testTimestampAndUnsupportedLanguageFailBeforeNativeAdmission() async throws {
        let probe = NativeProbe(delayNanoseconds: 0)
        let engine = WhisperResidentEngine(
            factory: probe,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 1)

        do {
            _ = try await engine.transcribe(
                inputFeatures: [0],
                requestedLanguage: "en",
                includeTimestamps: true)
            XCTFail("timestamp request unexpectedly entered native inference")
        } catch let error as WhisperDecodingPolicy.PolicyError {
            XCTAssertEqual(error, .timestampsUnsupported)
        }
        do {
            _ = try await engine.transcribe(
                inputFeatures: [0],
                requestedLanguage: "not-a-language")
            XCTFail("unsupported language unexpectedly entered native inference")
        } catch let error as WhisperDecodingPolicy.PolicyError {
            XCTAssertEqual(error, .unsupportedLanguage("not-a-language"))
        }
        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.sessionsCreated, 0)
    }

    func testBoundedQueueRejectsOverloadWithoutAllocatingAnotherSession() async throws {
        let probe = NativeProbe(delayNanoseconds: 60_000_000)
        let engine = WhisperResidentEngine(
            factory: probe,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 1,
            maximumQueuedRequests: 1)
        let first = Task {
            try await engine.transcribe(inputFeatures: [0], requestedLanguage: "en")
        }
        await probe.waitUntilSessionCount(1)
        let queued = Task {
            try await engine.transcribe(inputFeatures: [0], requestedLanguage: "en")
        }
        await engine.waitUntilQueuedRequestCountForTesting(1)

        do {
            _ = try await engine.transcribe(inputFeatures: [0], requestedLanguage: "en")
            XCTFail("over-capacity request unexpectedly entered the queue")
        } catch let error as WhisperResidentEngine.EngineError {
            XCTAssertEqual(error, .queueFull(maximumQueuedRequests: 1))
        }

        _ = try await first.value
        _ = try await queued.value
        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.sessionsCreated, 2)
        XCTAssertEqual(snapshot.maximumConcurrentSessions, 1)
    }

    func testCancellationAfterNoncooperativeSessionCreationSkipsEncodeAndCleansState() async throws {
        let probe = NativeProbe(
            delayNanoseconds: 0,
            sessionCreationDelayNanoseconds: 40_000_000)
        let engine = WhisperResidentEngine(
            factory: probe,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 1)
        let request = Task {
            try await engine.transcribe(inputFeatures: [0], requestedLanguage: "en")
        }
        await probe.waitUntilSessionCount(1)
        request.cancel()

        do {
            _ = try await request.value
            XCTFail("cancelled request unexpectedly encoded audio")
        } catch is CancellationError {
            // Expected.
        }
        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.sessionsCreated, 1)
        XCTAssertEqual(snapshot.sessionsFinished, 1)
        XCTAssertEqual(snapshot.events, ["session:1", "finish:1"])
    }

    func testInvalidNativeStatusAndNonFiniteLogitsFailClosedAndReleaseSession() async throws {
        let badStatus = NativeProbe(delayNanoseconds: 0, loadStatus: 0)
        let statusEngine = WhisperResidentEngine(
            factory: badStatus,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 1)
        do {
            _ = try await statusEngine.transcribe(
                inputFeatures: [0], requestedLanguage: "en")
            XCTFail("invalid load status unexpectedly reached decoding")
        } catch let error as WhisperResidentEngine.EngineError {
            XCTAssertEqual(
                error,
                .invalidNativeStatus(operation: "load_cross_kv", actual: 0))
        }
        let badStatusSnapshot = await badStatus.snapshot()
        XCTAssertEqual(badStatusSnapshot.sessionsFinished, 1)

        let badLogits = NativeProbe(delayNanoseconds: 0, returnsNonFiniteLogits: true)
        let logitsEngine = WhisperResidentEngine(
            factory: badLogits,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 1)
        do {
            _ = try await logitsEngine.transcribe(
                inputFeatures: [0], requestedLanguage: "en")
            XCTFail("non-finite logits unexpectedly reached policy decoding")
        } catch let error as WhisperResidentEngine.EngineError {
            XCTAssertEqual(error, .invalidNativeLogits)
        }
        let badLogitsSnapshot = await badLogits.snapshot()
        XCTAssertEqual(badLogitsSnapshot.sessionsFinished, 1)
    }

    func testInvalidFeatureShapeFailsBeforeCreatingNativeState() async throws {
        let probe = NativeProbe(delayNanoseconds: 0)
        let engine = WhisperResidentEngine(
            factory: probe,
            policy: try policy(),
            textDecoder: StubTextDecoder(),
            expectedFeatureCount: 4)

        do {
            _ = try await engine.transcribe(
                inputFeatures: [0, 1], requestedLanguage: "en")
            XCTFail("invalid feature shape unexpectedly reached native inference")
        } catch let error as WhisperResidentEngine.EngineError {
            XCTAssertEqual(error, .invalidFeatureCount(expected: 4, actual: 2))
        }

        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.sessionsCreated, 0)
    }

    private func policy() throws -> WhisperDecodingPolicy {
        let json = """
        {
          "decoder_start_token_id": 50258,
          "eos_token_id": 50257,
          "pad_token_id": 50257,
          "forced_decoder_ids": [[1, 50259], [2, 50359], [3, 50363]],
          "suppress_tokens": [220],
          "begin_suppress_tokens": [220],
          "no_timestamps_token_id": 50363,
          "max_length": 448,
          "lang_to_id": {"<|en|>": 50259},
          "task_to_id": {"transcribe": 50359}
        }
        """
        let data = Data(json.utf8)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return try WhisperDecodingPolicy(
            authenticatedData: data,
            expectedSHA256: digest)
    }
}

private struct StubTextDecoder: WhisperTextDecoding {
    func decode(textTokenIDs: [Int32]) -> String {
        textTokenIDs == [100] ? "hello" : ""
    }
}

private actor NativeProbe: WhisperNativeSessionFactory {
    struct Snapshot: Sendable {
        var sessionsCreated: Int
        var sessionsFinished: Int
        var maximumConcurrentSessions: Int
        var events: [String]
    }

    private let delayNanoseconds: UInt64
    private let sessionCreationDelayNanoseconds: UInt64
    private let loadStatus: Int32
    private let returnsNonFiniteLogits: Bool
    private var sessionsCreated = 0
    private var sessionsFinished = 0
    private var activeSessions = 0
    private var maximumConcurrentSessions = 0
    private var events: [String] = []
    private var sessionWaiters: [CheckedContinuation<Void, Never>] = []

    init(
        delayNanoseconds: UInt64,
        sessionCreationDelayNanoseconds: UInt64 = 0,
        loadStatus: Int32 = 1,
        returnsNonFiniteLogits: Bool = false
    ) {
        self.delayNanoseconds = delayNanoseconds
        self.sessionCreationDelayNanoseconds = sessionCreationDelayNanoseconds
        self.loadStatus = loadStatus
        self.returnsNonFiniteLogits = returnsNonFiniteLogits
    }

    func makeSession() async throws -> any WhisperNativeSession {
        sessionsCreated += 1
        activeSessions += 1
        maximumConcurrentSessions = max(maximumConcurrentSessions, activeSessions)
        let id = sessionsCreated
        events.append("session:\(id)")
        let waiters = sessionWaiters
        sessionWaiters.removeAll()
        waiters.forEach { $0.resume() }
        if sessionCreationDelayNanoseconds > 0 {
            try? await Task.sleep(nanoseconds: sessionCreationDelayNanoseconds)
        }
        return ProbeSession(
            id: id,
            owner: self,
            delayNanoseconds: delayNanoseconds,
            loadStatus: loadStatus,
            returnsNonFiniteLogits: returnsNonFiniteLogits)
    }

    func record(_ event: String) {
        events.append(event)
    }

    func finished(id: Int) {
        events.append("finish:\(id)")
        sessionsFinished += 1
        activeSessions -= 1
    }

    func waitUntilSessionCount(_ count: Int) async {
        while sessionsCreated < count {
            await withCheckedContinuation { continuation in
                sessionWaiters.append(continuation)
            }
        }
    }

    func snapshot() -> Snapshot {
        Snapshot(
            sessionsCreated: sessionsCreated,
            sessionsFinished: sessionsFinished,
            maximumConcurrentSessions: maximumConcurrentSessions,
            events: events)
    }
}

private actor ProbeSession: WhisperNativeSession {
    private let id: Int
    private let owner: NativeProbe
    private let delayNanoseconds: UInt64
    private let loadStatus: Int32
    private let returnsNonFiniteLogits: Bool
    private var encoded = false
    private var loaded = false
    private var stepCount = 0
    private var isFinished = false

    init(
        id: Int,
        owner: NativeProbe,
        delayNanoseconds: UInt64,
        loadStatus: Int32,
        returnsNonFiniteLogits: Bool
    ) {
        self.id = id
        self.owner = owner
        self.delayNanoseconds = delayNanoseconds
        self.loadStatus = loadStatus
        self.returnsNonFiniteLogits = returnsNonFiniteLogits
    }

    func encode(inputFeatures: [Float16]) async throws {
        precondition(!encoded && !loaded)
        encoded = true
        await owner.record("encode:\(id)")
        if delayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: delayNanoseconds)
        }
    }

    func loadCrossKV() async throws -> Int32 {
        precondition(encoded && !loaded)
        loaded = true
        await owner.record("load:\(id)")
        return loadStatus
    }

    func step(tokenID: Int32) async throws -> WhisperNativeStepOutput {
        precondition(loaded && !isFinished)
        stepCount += 1
        await owner.record("step:\(id):\(tokenID)")
        var logits = [Float](repeating: -100, count: WhisperDecodingPolicy.vocabularySize)
        logits[stepCount < 5 ? 100 : 50_257] = 100
        if returnsNonFiniteLogits { logits[0] = .nan }
        return WhisperNativeStepOutput(status: 1, logits: logits)
    }

    func finish() async {
        guard !isFinished else { return }
        isFinished = true
        await owner.finished(id: id)
    }
}
