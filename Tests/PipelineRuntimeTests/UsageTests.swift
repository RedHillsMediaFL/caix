import XCTest

@testable import PipelineRuntime

final class UsageTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Usage.resetForTesting()
    }

    override func tearDown() {
        Usage.resetForTesting()
        super.tearDown()
    }

    func testSnapshotIncludesRollingAndLastGenerationFields() {
        Usage.record(model: "alpha", inputTokens: 10, outputTokens: 20, decodeSeconds: 2, at: 1_000)
        Usage.record(model: "beta", inputTokens: 5, outputTokens: 30, decodeSeconds: 3, at: 1_050)

        let snap = Usage.snapshot(now: 1_060)

        XCTAssertEqual(snap.totalRequests, 2)
        XCTAssertEqual(snap.totalInputTokens, 15)
        XCTAssertEqual(snap.totalOutputTokens, 50)
        XCTAssertEqual(snap.avgTokensPerSecond, 10.0, accuracy: 1e-9)
        XCTAssertEqual(snap.rollingRequests, 2)
        XCTAssertEqual(snap.rollingOutputTokens, 50)
        XCTAssertEqual(snap.rollingTokensPerSecond, 10.0, accuracy: 1e-9)
        XCTAssertEqual(snap.lastModel, "beta")
        XCTAssertEqual(snap.lastOutputTokens, 30)
        XCTAssertEqual(snap.lastDecodeSeconds, 3)
        XCTAssertEqual(snap.lastTokensPerSecond, 10.0, accuracy: 1e-9)
        XCTAssertEqual(snap.lastAt, 1_050)
        XCTAssertEqual(snap.byModel.count, 2)
    }

    func testRollingWindowExcludesOldGenerations() {
        Usage.record(model: "old", inputTokens: 1, outputTokens: 100, decodeSeconds: 10, at: 100)
        Usage.record(model: "new", inputTokens: 1, outputTokens: 20, decodeSeconds: 2, at: 1_000)

        let snap = Usage.snapshot(now: 1_000)

        XCTAssertEqual(snap.totalRequests, 2)
        XCTAssertEqual(snap.totalOutputTokens, 120)
        XCTAssertEqual(snap.rollingRequests, 1)
        XCTAssertEqual(snap.rollingOutputTokens, 20)
        XCTAssertEqual(snap.rollingTokensPerSecond, 10.0, accuracy: 1e-9)
        XCTAssertEqual(snap.lastModel, "new")
    }
}
