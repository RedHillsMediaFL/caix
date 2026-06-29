import XCTest

@testable import CoreAIServer

final class JobTrackerTests: XCTestCase {
    func testDiskSpaceGuardRejectsWhenPayloadWouldCrossReserve() {
        let result = DiskSpaceGuard.evaluate(
            availableBytes: 1_000,
            incomingBytes: 300,
            reserveBytes: 800)

        guard case .reject(let message) = result else {
            XCTFail("expected disk preflight rejection")
            return
        }
        XCTAssertTrue(message.contains("insufficient disk"))
    }

    func testDiskSpaceGuardAllowsWhenPayloadFitsAboveReserve() {
        XCTAssertEqual(
            DiskSpaceGuard.evaluate(availableBytes: 1_000, incomingBytes: 200, reserveBytes: 800),
            .allow)
    }

    func testStartDownloadRHMRejectsBeforeLaunchWhenReserveIsTooHigh() async throws {
        let root = try makeTempDir()
        let tracker = JobTracker()

        let err = await tracker.startDownloadRHM(
            name: "example-coreai",
            hfRepo: "redhillsmediafl/example-caix",
            exportsDir: root,
            estimatedBytes: 1,
            reserveBytes: Int64.max / 4)

        XCTAssertTrue(err?.contains("insufficient disk") == true)
        let jobs = await tracker.snapshot()
        XCTAssertTrue(jobs.isEmpty)
    }

    func testRunCheckSupportReturnsFastJSON() async throws {
        let root = try makeTempDir()
        let script = root.appendingPathComponent("quick_check.sh")
        try """
            printf '%s\\n' '{"ok":true,"supported":false,"model_type":"qwen3_5_moe","support_status":"needs_coreai_authoring","requirements":["author model"],"reason":"authoring required"}'
            """.write(to: script, atomically: true, encoding: .utf8)

        let tracker = JobTracker()
        let json = await tracker.runCheckSupport(
            hfRepo: "example/qwen3_5_moe",
            script: script.path,
            workingDir: root,
            pythonExecutable: "/bin/sh",
            timeoutSeconds: 5)

        let data = try XCTUnwrap(json.data(using: .utf8))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["supported"] as? Bool, false)
        XCTAssertEqual(object["model_type"] as? String, "qwen3_5_moe")
        XCTAssertEqual(object["support_status"] as? String, "needs_coreai_authoring")
        XCTAssertNotNil(object["support_log"] as? String)
    }

    func testRunCheckSupportTimesOutStuckProcess() async throws {
        let root = try makeTempDir()
        let script = root.appendingPathComponent("sleep_check.sh")
        try """
            sleep 5
            """.write(to: script, atomically: true, encoding: .utf8)

        let tracker = JobTracker()
        let started = Date()
        let json = await tracker.runCheckSupport(
            hfRepo: "example/glm-timeout",
            script: script.path,
            workingDir: root,
            pythonExecutable: "/bin/sh",
            timeoutSeconds: 0.2)
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertLessThan(elapsed, 3)
        let data = try XCTUnwrap(json.data(using: .utf8))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["supported"] as? Bool, false)
        XCTAssertTrue((object["reason"] as? String)?.contains("timed out") == true)
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-job-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
