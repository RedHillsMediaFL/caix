import XCTest

@testable import CoreAIServer

final class JobTrackerTests: XCTestCase {
    func testRunCheckSupportReturnsFastJSON() async throws {
        let root = try makeTempDir()
        let script = root.appendingPathComponent("quick_check.py")
        try """
            import json
            print(json.dumps({
                "ok": True,
                "supported": False,
                "model_type": "qwen3_5_moe",
                "support_status": "needs_coreai_authoring",
                "requirements": ["author model"],
                "reason": "authoring required"
            }))
            """.write(to: script, atomically: true, encoding: .utf8)

        let tracker = JobTracker()
        let json = await tracker.runCheckSupport(
            hfRepo: "example/qwen3_5_moe",
            script: script.path,
            workingDir: root,
            pythonExecutable: "/usr/bin/python3",
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
        let script = root.appendingPathComponent("sleep_check.py")
        try """
            import time
            time.sleep(5)
            """.write(to: script, atomically: true, encoding: .utf8)

        let tracker = JobTracker()
        let started = Date()
        let json = await tracker.runCheckSupport(
            hfRepo: "example/glm-timeout",
            script: script.path,
            workingDir: root,
            pythonExecutable: "/usr/bin/python3",
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
