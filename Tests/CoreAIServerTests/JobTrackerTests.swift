import XCTest
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

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

    func testStartConvertHFRejectsBeforeLaunchWhenReserveIsTooHigh() async throws {
        let root = try makeTempDir()
        let script = try makeNoopScript(in: root)
        let tracker = JobTracker()

        let err = await tracker.startConvertHF(
            name: "example-coreai",
            hfRepo: "example/model",
            compression: "4bit",
            precision: "float16",
            context: nil,
            script: script.path,
            workingDir: root,
            pythonExecutable: "/bin/sh",
            reserveBytes: Int64.max / 4)

        XCTAssertTrue(err?.contains("insufficient disk") == true)
        XCTAssertTrue(err?.contains("model conversion") == true)
        let jobs = await tracker.snapshot()
        XCTAssertTrue(jobs.isEmpty)
    }

    func testStartConvertGGUFRejectsBeforeLaunchWhenReserveIsTooHigh() async throws {
        let root = try makeTempDir()
        let script = try makeNoopScript(in: root)
        let tracker = JobTracker()

        let err = await tracker.startConvertGGUF(
            name: "example-coreai",
            ggufRepo: "example/model-gguf",
            ggufFile: nil,
            compression: "4bit",
            precision: "float16",
            context: nil,
            script: script.path,
            workingDir: root,
            pythonExecutable: "/bin/sh",
            reserveBytes: Int64.max / 4)

        XCTAssertTrue(err?.contains("insufficient disk") == true)
        XCTAssertTrue(err?.contains("model conversion") == true)
        let jobs = await tracker.snapshot()
        XCTAssertTrue(jobs.isEmpty)
    }

    func testStartDownloadRHMPassesExactRevisionToHFAndDoesNotForceHFHome() async throws {
        let root = try makeTempDir()
        let bin = root.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let argvLog = root.appendingPathComponent("hf-argv.txt")
        let envLog = root.appendingPathComponent("hf-env.txt")
        let hf = bin.appendingPathComponent("hf")
        try """
            #!/bin/sh
            printf '%s\\n' "$@" > "\(argvLog.path)"
            printf '%s\\n' "$HF_HOME" > "\(envLog.path)"
            exit 0
            """.write(to: hf, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: hf.path)

        let oldPath = getenv("PATH").map { String(cString: $0) }
        let oldHFHome = getenv("HF_HOME").map { String(cString: $0) }
        setenv("PATH", "\(bin.path):\(oldPath ?? "")", 1)
        unsetenv("HF_HOME")
        addTeardownBlock {
            if let oldPath {
                setenv("PATH", oldPath, 1)
            } else {
                unsetenv("PATH")
            }
            if let oldHFHome {
                setenv("HF_HOME", oldHFHome, 1)
            } else {
                unsetenv("HF_HOME")
            }
        }

        let tracker = JobTracker()
        let err = await tracker.startDownloadRHM(
            name: "example-coreai",
            hfRepo: "redhillsmediafl/example-caix",
            revision: "0123456789abcdef0123456789abcdef01234567",
            exportsDir: root,
            estimatedBytes: 0,
            reserveBytes: 0)

        XCTAssertNil(err)
        let argvLogAppeared = await waitForFile(argvLog)
        XCTAssertTrue(argvLogAppeared)
        let argv = try String(contentsOf: argvLog, encoding: .utf8)
        XCTAssertTrue(argv.contains("download\nredhillsmediafl/example-caix\n--revision\n0123456789abcdef0123456789abcdef01234567\n--local-dir\n"))
        XCTAssertTrue(argv.contains("example-coreai"))
        let hfHome = try String(contentsOf: envLog, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertEqual(hfHome, "")
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

    private func makeNoopScript(in root: URL) throws -> URL {
        let script = root.appendingPathComponent("noop.sh")
        try """
            exit 0
            """.write(to: script, atomically: true, encoding: .utf8)
        return script
    }

    private func waitForFile(_ url: URL, timeout: TimeInterval = 3) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if FileManager.default.fileExists(atPath: url.path) {
                return true
            }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return FileManager.default.fileExists(atPath: url.path)
    }
}
