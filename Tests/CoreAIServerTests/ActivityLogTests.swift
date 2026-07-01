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
}
