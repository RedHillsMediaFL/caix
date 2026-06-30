import Dispatch
import Foundation

private struct DeployVerifyConfig: Sendable {
    var endpoints: [String]
    var minMachines: Int
    var timeoutSeconds: Double
    var path: String
    var emitJSON: Bool
}

private struct DeployVerifyEndpoint: Sendable {
    var raw: String
    var url: URL
    var host: String
    var port: Int
}

private struct DeployVerifyEndpointResult: Codable, Sendable {
    var endpoint: String
    var url: String
    var host: String
    var port: Int
    var reachable: Bool
    var statusCode: Int?
    var elapsedMS: Int
    var serverName: String?
    var runtimeLinked: Bool?
    var computeUnit: String?
    var error: String?

    enum CodingKeys: String, CodingKey {
        case endpoint
        case url
        case host
        case port
        case reachable
        case statusCode = "status_code"
        case elapsedMS = "elapsed_ms"
        case serverName = "server_name"
        case runtimeLinked = "runtime_linked"
        case computeUnit = "compute_unit"
        case error
    }
}

private struct DeployVerifyOutput: Codable, Sendable {
    var ok: Bool
    var requiredMachines: Int
    var reachableMachines: Int
    var reachableEndpoints: Int
    var endpoints: [DeployVerifyEndpointResult]

    enum CodingKeys: String, CodingKey {
        case ok
        case requiredMachines = "required_machines"
        case reachableMachines = "reachable_machines"
        case reachableEndpoints = "reachable_endpoints"
        case endpoints
    }
}

func deployCommand(_ argv: [String]) {
    guard let subcommand = argv.first else {
        deployUsage()
        exit(2)
    }

    switch subcommand {
    case "verify":
        deployVerifyCommand(Array(argv.dropFirst()))
    case "-h", "--help", "help":
        deployUsage()
        exit(0)
    default:
        FileHandle.standardError.write(Data("unknown deploy command: \(subcommand)\n".utf8))
        deployUsage()
        exit(2)
    }
}

private func deployUsage() {
    print(
        """
        USAGE:
          caix deploy verify --endpoint <host[:port]|url> --endpoint <host[:port]|url> [options]

        verify OPTIONS:
          --endpoint, -e <target>  caix server endpoint; repeatable
          --endpoints <list>       Comma-separated endpoints
          --min-machines <N>       Distinct reachable endpoint hosts required (default: 2)
          --timeout <seconds>      Per-endpoint HTTP timeout (default: 2)
          --path <path>            Probe path when endpoint has no path (default: /api/server)
          --json                   Emit machine-readable JSON

        Verifies caix HTTP visibility across machines. It does not load models or run inference.
        """)
}

private func deployVerifyCommand(_ argv: [String]) {
    var endpoints: [String] = []
    var minMachines = 2
    var timeoutSeconds = 2.0
    var path = "/api/server"
    var emitJSON = false

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
        case "--endpoint", "-e":
            endpoints.append(value(arg))
        case "--endpoints":
            endpoints += value(arg).split(separator: ",").map(String.init)
        case "--min-machines":
            let raw = value(arg)
            guard let parsed = Int(raw), parsed > 0 else {
                fail("--min-machines needs a positive integer")
            }
            minMachines = parsed
        case "--timeout":
            let raw = value(arg)
            guard let parsed = Double(raw), parsed > 0 else {
                fail("--timeout needs a positive number")
            }
            timeoutSeconds = parsed
        case "--path":
            let raw = value(arg).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !raw.isEmpty else { fail("--path cannot be empty") }
            path = raw.hasPrefix("/") ? raw : "/\(raw)"
        case "--json":
            emitJSON = true
        case "-h", "--help":
            deployUsage()
            exit(0)
        default:
            if arg.hasPrefix("-") {
                fail("unknown deploy verify option: \(arg)")
            }
            endpoints.append(arg)
        }
        i += 1
    }

    endpoints = endpoints.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    guard !endpoints.isEmpty else {
        fail("deploy verify requires at least one endpoint")
    }

    let config = DeployVerifyConfig(
        endpoints: endpoints,
        minMachines: minMachines,
        timeoutSeconds: timeoutSeconds,
        path: path,
        emitJSON: emitJSON)
    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var exitCode: Int32 = 0

    Task {
        defer { semaphore.signal() }
        do {
            let output = try await runDeployVerify(config)
            if config.emitJSON {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                FileHandle.standardOutput.write(try encoder.encode(output))
                FileHandle.standardOutput.write(Data("\n".utf8))
            } else {
                print(renderDeployVerify(output))
            }
            exitCode = output.ok ? 0 : 1
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exitCode = 2
        }
    }

    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode)
}

private func runDeployVerify(_ config: DeployVerifyConfig) async throws -> DeployVerifyOutput {
    let probes = try config.endpoints.map {
        try normalizeDeployVerifyEndpoint($0, defaultPath: config.path)
    }
    var results: [DeployVerifyEndpointResult] = []
    results.reserveCapacity(probes.count)
    for probe in probes {
        results.append(await requestDeployVerifyEndpoint(probe, timeoutSeconds: config.timeoutSeconds))
    }

    let reachable = results.filter(\.reachable)
    let reachableMachines = Set(reachable.map { $0.host.lowercased() }).count
    let ok = reachableMachines >= config.minMachines && reachable.count >= config.minMachines
    return DeployVerifyOutput(
        ok: ok,
        requiredMachines: config.minMachines,
        reachableMachines: reachableMachines,
        reachableEndpoints: reachable.count,
        endpoints: results)
}

private func normalizeDeployVerifyEndpoint(
    _ rawEndpoint: String,
    defaultPath: String
) throws -> DeployVerifyEndpoint {
    let trimmed = rawEndpoint.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { throw DeployVerifyError("empty endpoint") }
    let withScheme = trimmed.contains("://") ? trimmed : "http://\(trimmed)"
    guard var components = URLComponents(string: withScheme) else {
        throw DeployVerifyError("invalid endpoint: \(rawEndpoint)")
    }
    guard let scheme = components.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
        throw DeployVerifyError("endpoint must use http or https: \(rawEndpoint)")
    }
    guard let host = components.host, !host.isEmpty else {
        throw DeployVerifyError("endpoint needs a host: \(rawEndpoint)")
    }
    if components.port == nil {
        components.port = 1237
    }
    if components.path.isEmpty || components.path == "/" {
        components.path = defaultPath
    }
    guard let url = components.url else {
        throw DeployVerifyError("invalid endpoint URL: \(rawEndpoint)")
    }
    return DeployVerifyEndpoint(raw: trimmed, url: url, host: host, port: components.port ?? 1237)
}

private func requestDeployVerifyEndpoint(
    _ endpoint: DeployVerifyEndpoint,
    timeoutSeconds: Double
) async -> DeployVerifyEndpointResult {
    let start = Date()
    let configuration = URLSessionConfiguration.ephemeral
    configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
    configuration.timeoutIntervalForRequest = timeoutSeconds
    configuration.timeoutIntervalForResource = timeoutSeconds
    configuration.urlCache = nil
    let session = URLSession(configuration: configuration)
    defer { session.finishTasksAndInvalidate() }

    var request = URLRequest(url: endpoint.url, timeoutInterval: timeoutSeconds)
    request.httpMethod = "GET"
    request.setValue("caix/\(CaixBuildInfo.version)", forHTTPHeaderField: "User-Agent")

    do {
        let (data, response) = try await session.data(for: request)
        let elapsedMS = elapsedMilliseconds(since: start)
        guard let http = response as? HTTPURLResponse else {
            return deployVerifyResult(
                endpoint, statusCode: nil, elapsedMS: elapsedMS, error: "non-HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            return deployVerifyResult(
                endpoint,
                statusCode: http.statusCode,
                elapsedMS: elapsedMS,
                error: "HTTP \(http.statusCode)")
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return deployVerifyResult(
                endpoint,
                statusCode: http.statusCode,
                elapsedMS: elapsedMS,
                error: "invalid caix server JSON")
        }
        let serverOK = deployVerifyBool(object, keys: ["ok"])
        let name = deployVerifyString(object, keys: ["name"])
        let runtimeLinked = deployVerifyBool(object, keys: ["runtimeLinked", "runtime_linked"])
        let computeUnit = deployVerifyString(object, keys: ["computeUnit", "compute_unit"])
        guard serverOK == true else {
            return deployVerifyResult(
                endpoint,
                statusCode: http.statusCode,
                elapsedMS: elapsedMS,
                serverName: name,
                runtimeLinked: runtimeLinked,
                computeUnit: computeUnit,
                error: "caix server ok=false or missing")
        }
        return deployVerifyResult(
            endpoint,
            reachable: true,
            statusCode: http.statusCode,
            elapsedMS: elapsedMS,
            serverName: name,
            runtimeLinked: runtimeLinked,
            computeUnit: computeUnit,
            error: nil)
    } catch {
        return deployVerifyResult(
            endpoint,
            statusCode: nil,
            elapsedMS: elapsedMilliseconds(since: start),
            error: deployVerifyErrorDescription(error))
    }
}

private func deployVerifyResult(
    _ endpoint: DeployVerifyEndpoint,
    reachable: Bool = false,
    statusCode: Int?,
    elapsedMS: Int,
    serverName: String? = nil,
    runtimeLinked: Bool? = nil,
    computeUnit: String? = nil,
    error: String?
) -> DeployVerifyEndpointResult {
    DeployVerifyEndpointResult(
        endpoint: endpoint.raw,
        url: endpoint.url.absoluteString,
        host: endpoint.host,
        port: endpoint.port,
        reachable: reachable,
        statusCode: statusCode,
        elapsedMS: elapsedMS,
        serverName: serverName,
        runtimeLinked: runtimeLinked,
        computeUnit: computeUnit,
        error: error)
}

private func renderDeployVerify(_ output: DeployVerifyOutput) -> String {
    var lines = [
        "deploy verify",
        "required machines: \(output.requiredMachines)",
        "reachable machines: \(output.reachableMachines)",
        "reachable endpoints: \(output.reachableEndpoints)",
    ]
    for result in output.endpoints {
        var fields = ["- \(result.host):\(result.port)", result.reachable ? "ok" : "fail"]
        if let statusCode = result.statusCode {
            fields.append("http=\(statusCode)")
        }
        fields.append("ms=\(result.elapsedMS)")
        if let serverName = result.serverName {
            fields.append("name=\(serverName)")
        }
        if let runtimeLinked = result.runtimeLinked {
            fields.append("runtime_linked=\(runtimeLinked)")
        }
        if let computeUnit = result.computeUnit {
            fields.append("compute=\(computeUnit)")
        }
        if let error = result.error {
            fields.append("error=\(error)")
        }
        lines.append(fields.joined(separator: " "))
    }
    lines.append("status: \(output.ok ? "ok" : "fail")")
    return lines.joined(separator: "\n")
}

private func elapsedMilliseconds(since start: Date) -> Int {
    max(0, Int((Date().timeIntervalSince(start) * 1000).rounded()))
}

private func deployVerifyBool(_ object: [String: Any], keys: [String]) -> Bool? {
    for key in keys {
        if let value = object[key] as? Bool {
            return value
        }
    }
    return nil
}

private func deployVerifyString(_ object: [String: Any], keys: [String]) -> String? {
    for key in keys {
        if let value = object[key] as? String {
            return value
        }
    }
    return nil
}

private func deployVerifyErrorDescription(_ error: Error) -> String {
    if let urlError = error as? URLError {
        switch urlError.code {
        case .timedOut:
            return "timeout"
        case .cannotConnectToHost, .networkConnectionLost, .notConnectedToInternet:
            return urlError.localizedDescription
        default:
            return urlError.localizedDescription
        }
    }
    return error.localizedDescription
}

private struct DeployVerifyError: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}
