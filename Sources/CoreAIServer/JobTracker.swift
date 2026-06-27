import Foundation

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
        let model: String
        let start: Date
        var state: State
        var finishedAt: Date?
    }

    private var jobs: [String: Job] = [:]

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

        let process = Process()
        process.currentDirectoryURL = workingDir
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [script, model]
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        // Discard child output (the exporter is noisy); we only track lifecycle.
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        process.terminationHandler = { proc in
            let ok = proc.terminationStatus == 0
            Task { await self.finish(model: model, success: ok) }
        }
        do {
            try process.run()
        } catch {
            return "failed to launch converter: \(error.localizedDescription)"
        }
        jobs[model] = Job(model: model, start: Date(), state: .running, finishedAt: nil)
        return nil
    }

    /// Run the architecture support check (`convert.py --check --hf-id <repo>`) synchronously and
    /// return the raw JSON line it prints. Blocks ~5-180s (fetches the HF config). Used by
    /// `POST /api/check-support`.
    func runCheckSupport(hfRepo: String, script: String, workingDir: URL, pythonExecutable: String)
        async -> String
    {
        guard FileManager.default.fileExists(atPath: script) else {
            return #"{"ok":false,"supported":false,"reason":"converter not found"}"#
        }
        let process = Process()
        process.currentDirectoryURL = workingDir
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [script, "--check", "--hf-id", hfRepo]
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return #"{"ok":false,"supported":false,"reason":"failed to launch check: \#(error.localizedDescription)"}"#
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let text = String(data: data, encoding: .utf8) ?? ""
        // last JSON line
        if let line = text.split(separator: "\n").last(where: { $0.trimmingCharacters(in: .whitespaces).hasPrefix("{") }) {
            return String(line)
        }
        return #"{"ok":false,"supported":false,"reason":"no JSON from check"}"#
    }

    /// Launch a settings-aware conversion of a raw HF repo:
    /// `convert.py --hf-id <repo> --name <name> --compression <c> --compute-precision <p> [--context N]`.
    /// convert.py re-gates on support and refuses unsupported types. Tracked under `name`.
    func startConvertHF(name: String, hfRepo: String, compression: String, precision: String,
                        context: Int?, script: String, workingDir: URL, pythonExecutable: String)
        -> String?
    {
        if let existing = jobs[name], existing.state == .running {
            return "conversion already running for \(name)"
        }
        guard FileManager.default.fileExists(atPath: script) else {
            return "converter not found at \(script)"
        }
        var argv = [script, "--hf-id", hfRepo, "--name", name,
                    "--compression", compression, "--compute-precision", precision]
        if let c = context { argv += ["--context", String(c)] }
        let process = Process()
        process.currentDirectoryURL = workingDir
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = argv
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        process.terminationHandler = { proc in
            let ok = proc.terminationStatus == 0
            Task { await self.finish(model: name, success: ok) }
        }
        do {
            try process.run()
        } catch {
            return "failed to launch converter: \(error.localizedDescription)"
        }
        jobs[name] = Job(model: name, start: Date(), state: .running, finishedAt: nil)
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
            switch job.state {
            case .running:
                let elapsed = now.timeIntervalSince(job.start)
                let pct = min(95, 5 + Int(elapsed))
                out.append(JobEntry(label: "convert \(job.model)", pct: pct))
            case .done:
                if let f = job.finishedAt, now.timeIntervalSince(f) < 20 {
                    out.append(JobEntry(label: "convert \(job.model) ✓", pct: 100))
                }
            case .failed:
                if let f = job.finishedAt, now.timeIntervalSince(f) < 20 {
                    out.append(JobEntry(label: "convert \(job.model) ✗ failed", pct: 100))
                }
            }
        }
        // Prune long-finished jobs.
        jobs = jobs.filter { _, job in
            job.state == .running || (job.finishedAt.map { now.timeIntervalSince($0) < 20 } ?? true)
        }
        return out
    }
}

/// One row of `GET /api/jobs`.
public struct JobEntry: Codable, Sendable {
    public var label: String
    public var pct: Int
    public init(label: String, pct: Int) {
        self.label = label
        self.pct = pct
    }
}
