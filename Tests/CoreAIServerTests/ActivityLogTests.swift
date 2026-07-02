import XCTest
@testable import CoreAIServer

final class ActivityLogTests: XCTestCase {
    func testActivityLogRedactsSensitiveSummariesAndCapsSnapshots() async {
        let log = ActivityLog(capacity: 3)
        for index in 0..<5 {
            await log.record(
                method: "POST",
                path: "/v1/chat/completions",
                status: 200,
                startedAt: Date(),
                model: "qwen",
                summary: "done TOKEN=secret-\(index) Authorization: Bearer abcdef",
                inputTokens: index,
                outputTokens: index + 1)
        }

        let events = await log.snapshot(limit: 10)
        XCTAssertEqual(events.count, 3)
        XCTAssertEqual(events.first?.inputTokens, 4)
        XCTAssertEqual(events.first?.outputTokens, 5)
        XCTAssertFalse(events.map(\.summary).joined(separator: "\n").contains("secret"))
        XCTAssertFalse(events.map(\.summary).joined(separator: "\n").contains("abcdef"))
        XCTAssertTrue(events.allSatisfy { $0.summary.contains("[redacted]") })
    }

    func testActivityLogRecordsGenerationTimings() async {
        let log = ActivityLog()
        await log.record(
            method: "POST",
            path: "/v1/chat/completions",
            status: 200,
            startedAt: Date(),
            model: "qwen",
            summary: "completed",
            inputTokens: 12,
            outputTokens: 8,
            firstTokenSeconds: 0.42,
            loadSeconds: 1.25,
            prefillSeconds: 0.2,
            decodeSeconds: 2.0)

        let event = await log.snapshot(limit: 1)[0]
        XCTAssertEqual(event.firstTokenMs, 420)
        XCTAssertEqual(event.loadMs, 1250)
        XCTAssertEqual(event.prefillMs, 200)
        XCTAssertEqual(event.decodeMs, 2000)
        XCTAssertEqual(event.decodeTokensPerSecond, 4.0)
    }
}
