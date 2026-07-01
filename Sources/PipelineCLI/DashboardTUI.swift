import Foundation

func dashboardCommand(_ argv: [String]) {
    var options = DashboardTUI.Options()

    func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
        exit(2)
    }

    var i = 0
    func value(_ flag: String) -> String {
        i += 1
        guard i < argv.count else { fail("\(flag) needs a value") }
        return argv[i]
    }
    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--endpoint":
            options.endpoint = value(arg)
        case "--interval":
            guard let parsed = Double(value(arg)), parsed > 0 else {
                fail("--interval must be positive")
            }
            options.interval = parsed
        case "--once":
            options.once = true
        case "--no-clear":
            options.clearScreen = false
        case "-h", "--help":
            print(
                """
                USAGE:
                  caix dashboard [--endpoint URL] [--interval SECONDS] [--once]

                OPTIONS:
                  --endpoint <url>       caix server endpoint (default: http://127.0.0.1:1237)
                  --interval <seconds>   Refresh interval (default: 2)
                  --once                 Print one snapshot and exit
                  --no-clear             Do not clear the terminal between refreshes
                """
            )
            exit(0)
        default:
            fail("unknown dashboard option: \(arg)")
        }
        i += 1
    }

    do {
        try DashboardTUI(options: options).run()
    } catch {
        FileHandle.standardError.write(Data("dashboard error: \(error)\n".utf8))
        exit(1)
    }
}

struct DashboardTUI {
    struct Options {
        var endpoint = "http://127.0.0.1:1237"
        var interval = 2.0
        var once = false
        var clearScreen = true
    }

    private let endpoint: URL
    private let options: Options

    init(options: Options) throws {
        self.options = options
        let rawEndpoint = options.endpoint.hasPrefix("http://") || options.endpoint.hasPrefix("https://")
            ? options.endpoint
            : "http://\(options.endpoint)"
        guard let url = URL(string: rawEndpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))) else {
            throw DashboardError("invalid endpoint \(options.endpoint)")
        }
        self.endpoint = url
    }

    func run() throws {
        while true {
            if options.clearScreen {
                print("\u{001B}[2J\u{001B}[H", terminator: "")
            }
            print(render())
            fflush(stdout)
            if options.once { return }
            Thread.sleep(forTimeInterval: options.interval)
        }
    }

    private func render() -> String {
        var lines: [String] = []
        let now = ISO8601DateFormatter().string(from: Date())
        lines.append("caix dashboard")
        lines.append("endpoint: \(endpoint.absoluteString)   refreshed: \(now)")
        lines.append("press control-c to exit")
        lines.append("")

        let server = fetchObject("api/server")
        let stats = fetchObject("api/stats")
        let usage = fetchObject("api/usage")
        let jobs = fetchArray("api/jobs")
        let genstats = fetchObject("api/genstats")
        let models = fetchArray("api/models")

        lines.append(section("server", [
            field("status", server.ok ? "online" : (server.error ?? "offline")),
            field("version", server.string("caixVersion", "caix_version") ?? "-"),
            field("machine", server.string("machineName", "machine_name") ?? "-"),
            field("runtime", server.bool("runtimeLinked", "runtime_linked").map { $0 ? "core ai linked" : "runtime unavailable" } ?? "-"),
            field("exports", server.string("exportsDir", "exports_dir") ?? "-"),
        ]))

        lines.append(section("machine", [
            field("ram", "\(bytes(stats.number("usedRAMBytes", "used_ram_bytes"))) / \(bytes(stats.number("totalRAMBytes", "total_ram_bytes")))"),
            field("memory", pct(stats.number("memoryUsedFraction", "memory_used_fraction").map { $0 * 100 })),
            field("gpu", pct(stats.number("gpuUtilizationPercent", "gpu_utilization_percent"))),
            field("gpu ram", bytes(stats.number("gpuInUseMemoryBytes", "gpu_in_use_memory_bytes"))),
        ]))

        let modelRows = models.value
        let installed = modelRows.filter { $0.boolValue("bundle") == true }.count
        let loaded = modelRows.filter { ($0["status"] as? String) == "loaded" }.count
        let modelNames = modelRows
            .compactMap { $0["name"] as? String }
            .sorted()
            .prefix(8)
            .joined(separator: ", ")
        lines.append(section("models", [
            field("installed", "\(installed)"),
            field("loaded", "\(loaded)"),
            field("shown", modelNames.isEmpty ? "-" : modelNames),
        ]))

        lines.append(section("usage", [
            field("requests", intString(usage.number("totalRequests", "total_requests"))),
            field("tokens", "\(intString(usage.number("totalInputTokens", "total_input_tokens"))) in / \(intString(usage.number("totalOutputTokens", "total_output_tokens"))) out"),
            field("rolling", tokPerSecond(usage.number("rollingTokensPerSecond", "rolling_tokens_per_second"))),
            field("last", tokPerSecond(usage.number("lastTokensPerSecond", "last_tokens_per_second"))),
            field("last model", usage.string("lastModel", "last_model") ?? "-"),
        ]))

        lines.append(section("generation", [
            field("model", genstats.string("model") ?? "-"),
            field("accepted", intString(genstats.number("acceptedTokens", "accepted_tokens"))),
            field("draft", intString(genstats.number("draftTokens", "draft_tokens"))),
            field("ratio", ratio(genstats.number("acceptanceRate", "acceptance_rate"))),
        ]))

        let jobLines = summarizeJobs(jobs.value)
        lines.append(section("jobs", jobLines.isEmpty ? [field("active", "none")] : jobLines))

        return lines.joined(separator: "\n")
    }

    private func fetchObject(_ path: String) -> DashboardObject {
        do {
            let any = try fetchJSON(path)
            return DashboardObject(value: any as? [String: Any] ?? [:], error: nil)
        } catch {
            return DashboardObject(value: [:], error: compactError(error))
        }
    }

    private func fetchArray(_ path: String) -> DashboardArray {
        do {
            let any = try fetchJSON(path)
            return DashboardArray(value: any as? [[String: Any]] ?? [], error: nil)
        } catch {
            return DashboardArray(value: [], error: compactError(error))
        }
    }

    private func fetchJSON(_ path: String) throws -> Any {
        var request = URLRequest(url: url(for: path), timeoutInterval: 5)
        request.setValue("caix/\(CaixBuildInfo.version)", forHTTPHeaderField: "User-Agent")
        let (data, response) = try synchronousData(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw DashboardError("invalid response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw DashboardError("HTTP \(http.statusCode)")
        }
        if data.isEmpty { return [:] }
        return try JSONSerialization.jsonObject(with: data)
    }

    private func url(for path: String) -> URL {
        path.split(separator: "/").reduce(endpoint) { url, component in
            url.appendingPathComponent(String(component))
        }
    }

    private func synchronousData(for request: URLRequest) throws -> (Data, URLResponse) {
        let semaphore = DispatchSemaphore(value: 0)
        final class Box: @unchecked Sendable {
            var result: Result<(Data, URLResponse), Error>?
        }
        let box = Box()
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                box.result = .failure(error)
            } else if let data, let response {
                box.result = .success((data, response))
            } else {
                box.result = .failure(DashboardError("empty response"))
            }
            semaphore.signal()
        }
        task.resume()
        while semaphore.wait(timeout: .now()) == .timedOut {
            RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
        }
        return try box.result?.get() ?? { throw DashboardError("empty response") }()
    }
}

private struct DashboardObject {
    var value: [String: Any]
    var error: String?
    var ok: Bool { error == nil }

    func string(_ keys: String...) -> String? {
        for key in keys {
            if let value = value[key] as? String, !value.isEmpty { return value }
        }
        return nil
    }

    func bool(_ keys: String...) -> Bool? {
        for key in keys {
            if let value = value[key] as? Bool { return value }
        }
        return nil
    }

    func number(_ keys: String...) -> Double? {
        for key in keys {
            if let value = value[key] as? Double { return value }
            if let value = value[key] as? Int { return Double(value) }
            if let value = value[key] as? NSNumber { return value.doubleValue }
        }
        return nil
    }
}

private struct DashboardArray {
    var value: [[String: Any]]
    var error: String?
}

private struct DashboardError: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

private func section(_ title: String, _ rows: [String]) -> String {
    ([title] + rows.map { "  \($0)" }).joined(separator: "\n")
}

private func field(_ key: String, _ value: String) -> String {
    let padded = key.padding(toLength: 12, withPad: " ", startingAt: 0)
    return "\(padded) \(value)"
}

private func bytes(_ n: Double?) -> String {
    guard let n, n > 0 else { return "-" }
    let units = ["B", "KB", "MB", "GB", "TB"]
    var value = n
    var index = 0
    while value >= 1024, index < units.count - 1 {
        value /= 1024
        index += 1
    }
    if index == 0 { return "\(Int(value)) B" }
    return String(format: "%.1f %@", value, units[index])
}

private func pct(_ n: Double?) -> String {
    guard let n else { return "-" }
    return String(format: "%.0f%%", max(0, min(100, n)))
}

private func intString(_ n: Double?) -> String {
    guard let n else { return "-" }
    return String(Int(n.rounded()))
}

private func tokPerSecond(_ n: Double?) -> String {
    guard let n else { return "-" }
    return String(format: "%.1f tok/s", n)
}

private func ratio(_ n: Double?) -> String {
    guard let n else { return "-" }
    return String(format: "%.2f", n)
}

private func compactError(_ error: Error) -> String {
    let ns = error as NSError
    if ns.domain == NSURLErrorDomain {
        switch ns.code {
        case NSURLErrorCannotConnectToHost:
            return "offline: could not connect"
        case NSURLErrorTimedOut:
            return "offline: timed out"
        case NSURLErrorCannotFindHost, NSURLErrorDNSLookupFailed:
            return "offline: host not found"
        default:
            return "offline: \(ns.localizedDescription)"
        }
    }
    return String(describing: error)
}

private func summarizeJobs(_ rows: [[String: Any]]) -> [String] {
    rows.prefix(6).map { row in
        let label = (row["label"] as? String)
            ?? (row["name"] as? String)
            ?? (row["id"] as? String)
            ?? "job"
        let status = (row["status"] as? String) ?? "-"
        let pct = (row["pct"] as? Int).map { " \($0)%" } ?? ""
        return field(label, status + pct)
    }
}

private extension Dictionary where Key == String, Value == Any {
    func boolValue(_ key: String) -> Bool? {
        if let value = self[key] as? Bool { return value }
        if let value = self[key] as? NSNumber { return value.boolValue }
        return nil
    }
}
