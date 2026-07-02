import Foundation

struct ConversionGuard: Sendable {
    struct Decision: Equatable, Sendable {
        var reason: String
        var retryAfterSeconds: Int
    }

    let enabled: Bool
    let lockPaths: [URL]
    let retryAfterSeconds: Int

    init(enabled: Bool, lockPaths: [URL], retryAfterSeconds: Int = 60) {
        self.enabled = enabled
        self.lockPaths = lockPaths
        self.retryAfterSeconds = retryAfterSeconds
    }

    func decision() -> Decision? {
        let lockDecision = Self.evaluate(
            enabled: enabled,
            lockPaths: lockPaths,
            retryAfterSeconds: retryAfterSeconds,
            fileExists: { FileManager.default.fileExists(atPath: $0.path) },
            processLines: [])
        if let lockDecision { return lockDecision }
        guard enabled, Self.detectExporterProcess() else { return nil }
        return Decision(
            reason: "Core AI model conversion is active",
            retryAfterSeconds: retryAfterSeconds)
    }

    static func evaluate(
        enabled: Bool,
        lockPaths: [URL],
        retryAfterSeconds: Int = 60,
        fileExists: (URL) -> Bool,
        processLines: [String]
    ) -> Decision? {
        guard enabled else { return nil }
        if let lock = lockPaths.first(where: fileExists) {
            return Decision(
                reason: "heavy-task lock active: \(lock.path)",
                retryAfterSeconds: retryAfterSeconds)
        }
        if processLines.contains(where: isExporterProcessLine) {
            return Decision(
                reason: "Core AI model conversion is active",
                retryAfterSeconds: retryAfterSeconds)
        }
        return nil
    }

    static func defaultLockPaths(
        exportsDir: URL,
        currentDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true),
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [URL] {
        var paths: [URL] = []
        if let raw = caixEnv(environment, "caix_heavy_task_lock", legacy: "HEAVY_TASK_LOCK"),
           !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            paths.append(URL(fileURLWithPath: raw))
        }

        for root in [currentDirectory, exportsDir, exportsDir.deletingLastPathComponent(), exportsDir.deletingLastPathComponent().deletingLastPathComponent()] {
            paths.append(root.appendingPathComponent(".agent-heavy-task.lock"))
        }
        paths.append(URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".caix", isDirectory: true)
            .appendingPathComponent(".agent-heavy-task.lock"))

        var seen = Set<String>()
        return paths
            .map { $0.standardizedFileURL }
            .filter { seen.insert($0.path).inserted }
    }

    static func isExporterProcessLine(_ line: String) -> Bool {
        let lower = line.lowercased()
        if lower.contains("caix serve") {
            return false
        }
        let signals = [
            "coreai_models.export",
            "coreai-models export",
            "coreai_torch",
            "coreai-torch",
            "export.pipeline",
            "run-gemma",
            "run-qwen",
            "staged-4bit",
        ]
        return signals.contains { lower.contains($0) }
    }

    private static func detectExporterProcess() -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        process.arguments = [
            "-f",
            "coreai_models\\.export|coreai_torch|coreai-torch|export\\.pipeline|run-gemma|run-qwen|staged-4bit",
        ]
        process.standardOutput = Pipe()
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return false
        }
        return process.terminationStatus == 0
    }

    private static func caixEnv(_ env: [String: String], _ name: String, legacy suffix: String) -> String? {
        env[name] ?? env["C" + "AIX_" + suffix]
    }
}
