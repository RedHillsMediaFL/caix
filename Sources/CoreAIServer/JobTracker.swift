import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Tracks background `convert.py` runs launched by `POST /api/convert` and surfaces their
/// progress to `GET /api/jobs`.
///
/// The converter wraps Apple's exporter (model download + compile) and doesn't emit a clean
/// percentage, so progress is reported as a monotonic, time-based estimate that ramps toward
/// 95% while the subprocess runs and snaps to 100% on completion. Honest about being an
/// estimate — see the limitations note in the server docs.
actor JobTracker {
    private enum State: Sendable { case running, done, failed }
    private struct Job: Sendable {
        let kind: String
        let model: String
        let start: Date
        let logPath: String?
        var state: State
        var finishedAt: Date?
    }

    private var jobs: [String: Job] = [:]
    private static let logRoot = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent(".caix", isDirectory: true)
    static let convertLogDir = logRoot.appendingPathComponent("logs", isDirectory: true)
    static let supportLogDir = logRoot.appendingPathComponent("support-logs", isDirectory: true)

    private static func subprocessEnvironment(
        pythonUnbuffered: Bool = false,
        disableHFProgressBars: Bool = false,
        disableHFXet: Bool = false
    ) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        if pythonUnbuffered {
            env["PYTHONUNBUFFERED"] = "1"
        }
        if disableHFProgressBars {
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        }
        if disableHFProgressBars, env["HF_HUB_VERBOSITY", default: ""].isEmpty {
            env["HF_HUB_VERBOSITY"] = "error"
        }
        if disableHFXet, env["HF_HUB_DISABLE_XET", default: ""].isEmpty {
            env["HF_HUB_DISABLE_XET"] = "1"
        }
        return env
    }

    /// Launch `python3 <script> <model>` from `workingDir`. Returns `nil` on success or an
    /// error message if it couldn't be started (or is already running).
    func startConvert(model: String, script: String, workingDir: URL, pythonExecutable: String)
        -> String?
    {
        if let existing = jobs[model], existing.state == .running {
            return "conversion already running for \(model)"
        }
        guard FileManager.default.fileExists(atPath: script) else {
            return "converter not found at \(script)"
        }

        let (logURL, logHandle) = Self.openLog(kind: "convert", name: model, argv: [script, model])
        let process = Process()
        process.currentDirectoryURL = workingDir
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [script, model]
        process.environment = Self.subprocessEnvironment(pythonUnbuffered: true)
        process.standardOutput = logHandle ?? FileHandle.nullDevice
        process.standardError = logHandle ?? FileHandle.nullDevice
        process.terminationHandler = { proc in
            let ok = proc.terminationStatus == 0
            Task { await self.finish(model: model, success: ok) }
        }
        do {
            try process.run()
            try? logHandle?.close()
        } catch {
            try? logHandle?.close()
            return "failed to launch converter: \(error.localizedDescription)"
        }
        jobs[model] = Job(
            kind: "convert", model: model, start: Date(), logPath: logURL?.path,
            state: .running, finishedAt: nil)
        return nil
    }

    /// Run the architecture support check (`convert.py --check --hf-id <repo>`) synchronously and
    /// return the raw JSON line it prints. Blocks briefly (fetches the HF config). Used by
    /// `POST /api/check-support`.
    func runCheckSupport(
        hfRepo: String,
        script: String,
        workingDir: URL,
        pythonExecutable: String,
        timeoutSeconds: TimeInterval = 30
    )
        async -> String
    {
        guard FileManager.default.fileExists(atPath: script) else {
            return #"{"ok":false,"supported":false,"reason":"converter not found"}"#
        }
        let process = Process()
        process.currentDirectoryURL = workingDir
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [script, "--check", "--hf-id", hfRepo]
        process.environment = Self.subprocessEnvironment(pythonUnbuffered: true)
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let finished = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in finished.signal() }
        do {
            try process.run()
        } catch {
            return Self.annotateSupportCheck(
                hfRepo: hfRepo,
                rawJSON: Self.jsonString([
                    "ok": false,
                    "supported": false,
                    "reason": "failed to launch check: \(error.localizedDescription)",
                ]))
        }
        let exited = await Self.waitForSignal(finished, timeoutSeconds: timeoutSeconds)
        if !exited {
            Self.terminate(process)
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let text = String(data: data, encoding: .utf8) ?? ""
            return Self.annotateSupportCheck(
                hfRepo: hfRepo,
                rawJSON: Self.jsonString([
                    "ok": false,
                    "supported": false,
                    "reason": "support check timed out after \(Self.formatSeconds(timeoutSeconds))",
                    "output": String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(12000)),
                ]))
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let text = String(data: data, encoding: .utf8) ?? ""
        // last JSON line
        if let line = text.split(separator: "\n").last(where: { $0.trimmingCharacters(in: .whitespaces).hasPrefix("{") }) {
            return Self.annotateSupportCheck(hfRepo: hfRepo, rawJSON: String(line))
        }
        return Self.annotateSupportCheck(
            hfRepo: hfRepo,
            rawJSON: Self.jsonString([
                "ok": false,
                "supported": false,
                "reason": "no JSON from check",
                "exit_status": Int(process.terminationStatus),
                "output": String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(12000)),
            ]))
    }

    /// Launch a settings-aware conversion of a raw HF repo:
    /// `convert.py --hf-id <repo> --name <name> --compression <c> --compute-precision <p> [--context N]`.
    /// convert.py re-gates on support and refuses model types that still need Core AI authoring.
    /// Tracked under `name`.
    func startConvertHF(name: String, hfRepo: String, compression: String, precision: String,
                        context: Int?, script: String, workingDir: URL, pythonExecutable: String,
                        reserveBytes: Int64 = DiskSpaceGuard.reserveBytesFromEnvironment())
        -> String?
    {
        if let existing = jobs[name], existing.state == .running {
            return "conversion already running for \(name)"
        }
        guard FileManager.default.fileExists(atPath: script) else {
            return "converter not found at \(script)"
        }
        if let err = DiskSpaceGuard.preflightInstall(
            destinationRoot: workingDir,
            incomingBytes: nil,
            reserveBytes: reserveBytes,
            label: "model conversion")
        {
            return err
        }
        var argv = [script, "--hf-id", hfRepo, "--name", name,
                    "--compression", compression, "--compute-precision", precision]
        if let c = context { argv += ["--context", String(c)] }
        let (logURL, logHandle) = Self.openLog(kind: "convert-hf", name: name, argv: argv)
        let process = Process()
        process.currentDirectoryURL = workingDir
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = argv
        process.environment = Self.subprocessEnvironment(pythonUnbuffered: true)
        process.standardOutput = logHandle ?? FileHandle.nullDevice
        process.standardError = logHandle ?? FileHandle.nullDevice
        process.terminationHandler = { proc in
            let ok = proc.terminationStatus == 0
            Task { await self.finish(model: name, success: ok) }
        }
        do {
            try process.run()
            try? logHandle?.close()
        } catch {
            try? logHandle?.close()
            return "failed to launch converter: \(error.localizedDescription)"
        }
        jobs[name] = Job(
            kind: "convert", model: name, start: Date(), logPath: logURL?.path,
            state: .running, finishedAt: nil)
        return nil
    }

    /// Launch a GGUF-only repo conversion:
    /// `convert.py --gguf <repo> [--gguf-file f] --name <name> --compression ...`.
    /// Tracked under `name`; support is checked after dequantization inside the converter.
    func startConvertGGUF(name: String, ggufRepo: String, ggufFile: String?, compression: String,
                          precision: String, context: Int?, script: String, workingDir: URL,
                          pythonExecutable: String,
                          reserveBytes: Int64 = DiskSpaceGuard.reserveBytesFromEnvironment()) -> String? {
        if let existing = jobs[name], existing.state == .running {
            return "conversion already running for \(name)"
        }
        guard FileManager.default.fileExists(atPath: script) else {
            return "converter not found at \(script)"
        }
        if let err = DiskSpaceGuard.preflightInstall(
            destinationRoot: workingDir,
            incomingBytes: nil,
            reserveBytes: reserveBytes,
            label: "model conversion")
        {
            return err
        }
        var argv = [script, "--gguf", ggufRepo, "--name", name,
                    "--compression", compression, "--compute-precision", precision]
        if let ggufFile { argv += ["--gguf-file", ggufFile] }
        if let c = context { argv += ["--context", String(c)] }
        let (logURL, logHandle) = Self.openLog(kind: "convert-gguf", name: name, argv: argv)
        let process = Process()
        process.currentDirectoryURL = workingDir
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = argv
        process.environment = Self.subprocessEnvironment(pythonUnbuffered: true)
        process.standardOutput = logHandle ?? FileHandle.nullDevice
        process.standardError = logHandle ?? FileHandle.nullDevice
        process.terminationHandler = { proc in
            let ok = proc.terminationStatus == 0
            Task { await self.finish(model: name, success: ok) }
        }
        do {
            try process.run()
            try? logHandle?.close()
        } catch {
            try? logHandle?.close()
            return "failed to launch converter: \(error.localizedDescription)"
        }
        jobs[name] = Job(
            kind: "convert", model: name, start: Date(), logPath: logURL?.path,
            state: .running, finishedAt: nil)
        return nil
    }

    /// Download an already-converted caix bundle from Hugging Face into `exportsDir/name`.
    /// The repo must already contain root `metadata.json`; conversion is not run here.
    func startDownloadRHM(
        name: String,
        hfRepo: String,
        revision: String? = nil,
        exportsDir: URL,
        estimatedBytes: Int64? = nil,
        reserveBytes: Int64 = DiskSpaceGuard.reserveBytesFromEnvironment()
    ) -> String? {
        if let existing = jobs[name], existing.state == .running {
            return "download already running for \(name)"
        }
        guard Self.isSafeName(name) else {
            return "unsafe bundle name '\(name)'"
        }
        do {
            try FileManager.default.createDirectory(at: exportsDir, withIntermediateDirectories: true)
        } catch {
            return "could not create exports directory: \(error.localizedDescription)"
        }
        if let err = DiskSpaceGuard.preflightInstall(
            destinationRoot: exportsDir,
            incomingBytes: estimatedBytes,
            reserveBytes: reserveBytes)
        {
            return err
        }

        let destination = exportsDir.appendingPathComponent(name, isDirectory: true)
        var argv = ["hf", "download", hfRepo]
        if let revision = revision?.trimmingCharacters(in: .whitespacesAndNewlines), !revision.isEmpty {
            argv += ["--revision", revision]
        }
        argv += ["--max-workers", "1", "--local-dir", destination.path]
        let (logURL, logHandle) = Self.openLog(kind: "download-rhm", name: name, argv: argv)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.environment = Self.subprocessEnvironment(disableHFProgressBars: true, disableHFXet: true)
        process.standardOutput = logHandle ?? FileHandle.nullDevice
        process.standardError = logHandle ?? FileHandle.nullDevice
        process.terminationHandler = { proc in
            let ok = proc.terminationStatus == 0
            Task { await self.finish(model: name, success: ok) }
        }
        do {
            try process.run()
            try? logHandle?.close()
        } catch {
            try? logHandle?.close()
            return "failed to launch hf download: \(error.localizedDescription)"
        }
        jobs[name] = Job(
            kind: "download", model: name, start: Date(), logPath: logURL?.path,
            state: .running, finishedAt: nil)
        return nil
    }

    private func finish(model: String, success: Bool) {
        guard var job = jobs[model] else { return }
        job.state = success ? .done : .failed
        job.finishedAt = Date()
        jobs[model] = job
    }

    /// Active jobs plus those finished within the last 20s, as `[{label, pct}]`.
    func snapshot() -> [JobEntry] {
        let now = Date()
        var out: [JobEntry] = []
        for job in jobs.values.sorted(by: { $0.start < $1.start }) {
            let elapsed = Int(now.timeIntervalSince(job.start).rounded())
            switch job.state {
            case .running:
                let pct = min(95, 5 + elapsed)
                out.append(JobEntry(label: "\(job.kind) \(job.model)", pct: pct, status: "running",
                                    seconds: elapsed, logPath: job.logPath))
            case .done:
                if let f = job.finishedAt, now.timeIntervalSince(f) < 60 {
                    out.append(JobEntry(label: "\(job.kind) \(job.model) ✓", pct: 100, status: "done",
                                        seconds: elapsed, logPath: job.logPath))
                }
            case .failed:
                if let f = job.finishedAt, now.timeIntervalSince(f) < 3600 {
                    out.append(JobEntry(label: "\(job.kind) \(job.model) ✗ failed", pct: 100, status: "failed",
                                        seconds: elapsed, logPath: job.logPath))
                }
            }
        }
        // Prune long-finished jobs.
        jobs = jobs.filter { _, job in
            job.state == .running
                || (job.finishedAt.map { now.timeIntervalSince($0) < (job.state == .failed ? 3600 : 60) } ?? true)
        }
        return out
    }

    private static func openLog(kind: String, name: String, argv: [String]) -> (URL?, FileHandle?) {
        do {
            try FileManager.default.createDirectory(at: convertLogDir, withIntermediateDirectories: true)
            let stamp = Self.timestamp()
            let file = "\(stamp)-\(kind)-\(Self.slug(name)).log"
            let url = convertLogDir.appendingPathComponent(file)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            let handle = try FileHandle(forWritingTo: url)
            let header = """
                caix \(kind)
                started: \(ISO8601DateFormatter().string(from: Date()))
                name: \(name)
                command: \(argv.joined(separator: " "))

                """
            handle.write(Data(header.utf8))
            return (url, handle)
        } catch {
            return (nil, nil)
        }
    }

    static func annotateSupportCheck(hfRepo: String, rawJSON: String) -> String {
        let data = Data(rawJSON.utf8)
        guard var object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return rawJSON
        }
        let supported = (object["supported"] as? Bool) ?? false
        let ggufOnly = (object["gguf_only"] as? Bool) ?? false
        guard !supported && !ggufOnly else { return rawJSON }
        if let path = writeSupportLog(hfRepo: hfRepo, result: object) {
            object["support_log"] = path
        }
        guard JSONSerialization.isValidJSONObject(object),
              let out = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let text = String(data: out, encoding: .utf8)
        else { return rawJSON }
        return text
    }

    private static func writeSupportLog(hfRepo: String, result: [String: Any]) -> String? {
        do {
            try FileManager.default.createDirectory(at: supportLogDir, withIntermediateDirectories: true)
            let file = "\(timestamp())-support-\(slug(hfRepo)).json"
            let url = supportLogDir.appendingPathComponent(file)
            let payload: [String: Any] = [
                "checked_at": ISO8601DateFormatter().string(from: Date()),
                "hf_repo": hfRepo,
                "result": result,
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: url, options: .atomic)
            return url.path
        } catch {
            return nil
        }
    }

    private static func jsonString(_ object: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8)
        else {
            return #"{"ok":false,"supported":false,"reason":"internal JSON encoding failure"}"#
        }
        return text
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    private static func slug(_ s: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let value = s.unicodeScalars.map { allowed.contains($0) ? String($0) : "-" }.joined()
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return String(value.prefix(80))
    }

    private static func isSafeName(_ s: String) -> Bool {
        guard !s.isEmpty, s.count <= 160 else { return false }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        return s.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    private static func waitForExit(_ process: Process, timeoutSeconds: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        return !process.isRunning
    }

    private static func waitForSignal(_ semaphore: DispatchSemaphore, timeoutSeconds: TimeInterval) async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                let ok = semaphore.wait(timeout: .now() + timeoutSeconds) == .success
                continuation.resume(returning: ok)
            }
        }
    }

    private static func terminate(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        let deadline = Date().addingTimeInterval(2)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
        }
    }

    private static func formatSeconds(_ seconds: TimeInterval) -> String {
        if seconds >= 1 {
            return "\(Int(seconds.rounded()))s"
        }
        return String(format: "%.1fs", seconds)
    }
}

/// One row of `GET /api/jobs`.
public struct JobEntry: Codable, Sendable {
    public var label: String
    public var pct: Int
    public var status: String?
    public var seconds: Int?
    public var logPath: String?
    public init(label: String, pct: Int, status: String? = nil, seconds: Int? = nil, logPath: String? = nil) {
        self.label = label
        self.pct = pct
        self.status = status
        self.seconds = seconds
        self.logPath = logPath
    }
}
