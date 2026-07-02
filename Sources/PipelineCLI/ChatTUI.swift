import Darwin
import Foundation
import CoreAIServer

func chatTUICommand(_ argv: [String]) {
    var options = ChatTUI.Options()

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
        case "--model":
            options.model = value(arg)
        case "--shell":
            let raw = value(arg).lowercased()
            guard let mode = ChatTUI.ShellMode(rawValue: raw) else {
                fail("--shell must be ask, on, or off")
            }
            options.shellMode = mode
        case "--cwd":
            options.cwd = value(arg)
        case "--max-tokens":
            guard let parsed = Int(value(arg)), parsed > 0 else { fail("--max-tokens must be positive") }
            options.maxTokens = parsed
        case "--temperature":
            guard let parsed = Double(value(arg)), parsed >= 0 else { fail("--temperature must be >= 0") }
            options.temperature = parsed
        case "--top-p":
            guard let parsed = Double(value(arg)), parsed > 0, parsed <= 1 else { fail("--top-p must be in (0, 1]") }
            options.topP = parsed
        case "--system":
            options.systemPrompt = value(arg)
        case "-h", "--help":
            print(
                """
                USAGE:
                  caix chat [--endpoint URL] [--model NAME] [--shell ask|on|off]
                  caix tui  [--endpoint URL] [--model NAME] [--shell ask|on|off]

                Terminal commands:
                  /help                 Show commands
                  /models               List served models
                  /model [name]         Switch model or open picker
                  /install              Select and install a catalog model
                  /shell ask|on|off     Control model-initiated shell tool access
                  /run <command>        Run a shell command now
                  /cwd <dir>            Change shell working directory
                  /activity             Show recent server requests
                  /dashboard            Show native dashboard command
                  /system <text>        Set system prompt for future turns
                  /temperature <x>      Set sampling temperature for future turns
                  /top-p <x|default>    Set nucleus sampling for future turns
                  /max-tokens <n>       Set generation limit for future turns
                  /params               Show current generation parameters
                  /clear                Clear chat history
                  /quit                 Exit
                """
            )
            exit(0)
        default:
            fail("unknown chat option: \(arg)")
        }
        i += 1
    }

    do {
        var chat = try ChatTUI(options: options)
        try chat.run()
    } catch {
        FileHandle.standardError.write(Data("chat error: \(error)\n".utf8))
        exit(1)
    }
}

struct ChatTUI {
    enum ShellMode: String {
        case ask
        case on
        case off
    }

    struct Options {
        var endpoint = "http://127.0.0.1:1237"
        var model: String?
        var shellMode: ShellMode = .ask
        var cwd = FileManager.default.currentDirectoryPath
        var maxTokens = 1024
        var temperature = 0.7
        var topP: Double?
        var systemPrompt: String?
    }

    struct Message {
        var role: String
        var content: String
    }

    struct ToolCall {
        var id: String
        var name: String
        var arguments: String
    }

    struct AssistantTurn {
        var content: String
        var reasoning: String
        var toolCalls: [ToolCall]
        var finishReason: String?
    }

    var options: Options
    private let endpoint: URL
    private var history: [Message] = []
    private var models: [String] = []
    private var model: String
    private var shellMode: ShellMode
    private var cwd: String
    private var sessionLog: [String] = []

    init(options: Options) throws {
        self.options = options
        let rawEndpoint = options.endpoint.hasPrefix("http://") || options.endpoint.hasPrefix("https://")
            ? options.endpoint
            : "http://\(options.endpoint)"
        guard let url = URL(string: rawEndpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))) else {
            throw ChatTUIError("invalid endpoint \(options.endpoint)")
        }
        self.endpoint = url
        self.model = options.model ?? ""
        self.shellMode = options.shellMode
        self.cwd = options.cwd
    }

    mutating func run() throws {
        models = try fetchModels()
        if model.isEmpty {
            guard let first = models.first else {
                throw ChatTUIError("no served models found at \(endpoint.absoluteString); start caix serve with an installed bundle")
            }
            model = first
        } else if !models.isEmpty, !models.contains(model) {
            throw ChatTUIError("model '\(model)' is not served; use /models to inspect available models")
        }
        if let system = options.systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines), !system.isEmpty {
            history.append(Message(role: "system", content: system))
        }

        printHeader()
        while true {
            let prompt = "\n\(model)> "
            guard let input = readLine(prompt: prompt) else {
                print("")
                return
            }
            let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            if trimmed.hasPrefix("/") {
                if try handleCommand(trimmed) { return }
                continue
            }
            history.append(Message(role: "user", content: input))
            try runAgentTurn()
        }
    }

    private mutating func printHeader() {
        log("session start endpoint=\(endpoint.absoluteString) model=\(model) shell=\(shellMode.rawValue) cwd=\(cwd)")
        let width = max(52, min(100, terminalWidth()))
        print(rule(width: width))
        print("caix tui (cakes)")
        print("server   \(endpoint.absoluteString)")
        print("model    \(model)")
        print("shell    \(shellMode.rawValue)   cwd \(cwd)")
        if let warning = ModelSuitability.chatWarning(for: model) {
            print("warning: \(warning)")
        }
        print("type /help for commands, /model for picker, /activity for requests, /quit to exit")
        print(rule(width: width))
    }

    private mutating func handleCommand(_ input: String) throws -> Bool {
        let parts = input.split(separator: " ", maxSplits: 1).map(String.init)
        let command = parts[0]
        let rest = parts.count > 1 ? parts[1] : ""
        switch command {
        case "/help":
            print(
                """
                commands:
                  /models               list served models
                  /model [name]         switch model or open picker
                  /install              select and install a catalog model
                  /shell ask|on|off     ask before shell, run directly, or disable
                  /run <command>        run a shell command now
                  /cwd <dir>            change shell working directory
                  /activity             show recent server requests
                  /dashboard            show native dashboard command
                  /system <text>        set system prompt for future turns
                  /temperature <x>      set sampling temperature for future turns
                  /top-p <x|default>    set nucleus sampling for future turns
                  /max-tokens <n>       set generation limit for future turns
                  /params               show current generation parameters
                  /clear                clear chat history
                  /log                  print redacted session log
                  /quit                 exit
                """
            )
        case "/models":
            models = try fetchModels()
            printModels()
        case "/model":
            if rest.isEmpty {
                try selectModel()
                return false
            }
            models = try fetchModels()
            guard models.contains(rest) else {
                print("model not served: \(rest)")
                return false
            }
            model = rest
            log("model switched to \(model)")
            print("model: \(model)")
            if let warning = ModelSuitability.chatWarning(for: model) {
                print("warning: \(warning)")
            }
        case "/install", "/catalog":
            let root = (try? fetchServerExportsDir()) ?? caixDefaultExportsDisplayPath
            print("install root: \(root)")
            try Catalog.installInteractiveBlocking(exportsDir: root)
            models = try fetchModels()
            print("models refreshed: \(models.count)")
        case "/dashboard":
            print("run: caix dashboard --endpoint \(endpoint.absoluteString)")
        case "/activity":
            try printActivity()
        case "/shell":
            guard let mode = ShellMode(rawValue: rest.lowercased()) else {
                print("usage: /shell ask|on|off")
                return false
            }
            shellMode = mode
            log("shell mode \(mode.rawValue)")
            print("shell: \(mode.rawValue)")
        case "/run":
            guard !rest.isEmpty else {
                print("usage: /run <command>")
                return false
            }
            let result = try runShell(command: rest, cwd: cwd, timeout: 30, initiatedByModel: false)
            print(result)
        case "/cwd":
            guard !rest.isEmpty else {
                print(cwd)
                return false
            }
            let expanded = caixExpandPath(rest)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue else {
                print("not a directory: \(expanded)")
                return false
            }
            cwd = expanded
            log("cwd \(cwd)")
            print("cwd: \(cwd)")
        case "/system":
            history.removeAll { $0.role == "system" }
            if !rest.isEmpty {
                history.insert(Message(role: "system", content: rest), at: 0)
                log("system prompt updated")
                print("system prompt set")
            } else {
                log("system prompt cleared")
                print("system prompt cleared")
            }
        case "/temperature":
            guard let parsed = Double(rest), parsed >= 0 else {
                print("usage: /temperature <value >= 0>  (current: \(options.temperature))")
                return false
            }
            options.temperature = parsed
            log("temperature \(parsed)")
            print("temperature: \(parsed)")
        case "/top-p", "/top_p":
            if rest.lowercased() == "default" {
                options.topP = nil
                log("top_p default")
                print("top_p: server default")
                return false
            }
            guard let parsed = Double(rest), parsed > 0, parsed <= 1 else {
                let current = options.topP.map { "\($0)" } ?? "server default"
                print("usage: /top-p <value in (0, 1]|default>  (current: \(current))")
                return false
            }
            options.topP = parsed
            log("top_p \(parsed)")
            print("top_p: \(parsed)")
        case "/max-tokens":
            guard let parsed = Int(rest), parsed > 0 else {
                print("usage: /max-tokens <positive integer>  (current: \(options.maxTokens))")
                return false
            }
            options.maxTokens = parsed
            log("max tokens \(parsed)")
            print("max tokens: \(parsed)")
        case "/params":
            let topP = options.topP.map { "\($0)" } ?? "server default"
            print(
                """
                generation parameters:
                  temperature  \(options.temperature)
                  top_p        \(topP)
                  max tokens   \(options.maxTokens)
                """
            )
        case "/clear":
            let system = history.first(where: { $0.role == "system" })
            history.removeAll()
            if let system { history.append(system) }
            log("history cleared")
            print("cleared")
        case "/log":
            print(sessionLog.joined(separator: "\n"))
        case "/quit", "/exit":
            log("session end")
            return true
        default:
            print("unknown command: \(command)")
        }
        return false
    }

    private func printModels() {
        guard !models.isEmpty else {
            print("no served models")
            return
        }
        for (index, name) in models.enumerated() {
            let marker = name == model ? "*" : " "
            let warning = ModelSuitability.chatWarning(for: name).map { "  warning: \($0)" } ?? ""
            print(String(format: "%@ %2d  %@%@", marker, index + 1, name, warning))
        }
    }

    private mutating func selectModel() throws {
        models = try fetchModels()
        printModels()
        print("model [\(model)]: ", terminator: "")
        fflush(stdout)
        let raw = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !raw.isEmpty else { return }
        let selected: String?
        if let index = Int(raw), index >= 1, index <= models.count {
            selected = models[index - 1]
        } else {
            selected = models.first { $0.caseInsensitiveCompare(raw) == .orderedSame }
                ?? models.first { $0.localizedCaseInsensitiveContains(raw) }
        }
        guard let selected else {
            print("model not served: \(raw)")
            return
        }
        model = selected
        log("model switched to \(model)")
        print("model: \(model)")
        if let warning = ModelSuitability.chatWarning(for: model) {
            print("warning: \(warning)")
        }
    }

    private func printActivity() throws {
        let rows = try fetchActivity()
        guard !rows.isEmpty else {
            print("no recent activity")
            return
        }
        for row in rows.prefix(12) {
            let status = row["status"].map { "\($0)" } ?? "-"
            let method = row["method"] as? String ?? "-"
            let path = row["path"] as? String ?? "-"
            let ms = intFromNumber(row["latencyMs"]) ?? intFromNumber(row["latency_ms"]) ?? 0
            let timing = activityTiming(row)
            let model = (row["model"] as? String).map { "  \($0)" } ?? ""
            let summary = row["summary"] as? String ?? "-"
            print("\(status) \(method) \(path) \(ms)ms\(timing)\(model) - \(summary)")
        }
    }

    private mutating func runAgentTurn() throws {
        for hop in 0..<6 {
            let started = Date()
            let turn = try chatCompletion()
            let elapsed = Date().timeIntervalSince(started)
            log(String(format: "turn hop=%d finish=%@ seconds=%.2f chars=%d tools=%d",
                       hop + 1, turn.finishReason ?? "unknown", elapsed, turn.content.count, turn.toolCalls.count))

            if !turn.reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                print("\nthinking:\n\(turn.reasoning)")
            }
            if !turn.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                print("\ncaix:\n\(turn.content)")
                history.append(Message(role: "assistant", content: turn.content))
            }
            guard !turn.toolCalls.isEmpty else { return }
            for call in turn.toolCalls {
                let result = try executeToolCall(call)
                history.append(Message(role: "tool", content: result))
            }
        }
        print("\n[stopped after 6 tool hops]")
        log("tool loop stopped after hop limit")
    }

    private mutating func executeToolCall(_ call: ToolCall) throws -> String {
        guard call.name == "shell_run" else {
            let message = "[tool error: unknown tool \(call.name)]"
            print(message)
            log("unknown tool \(call.name)")
            return message
        }
        let args = parseJSONObject(call.arguments)
        let command = (args["command"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let toolCWD = (args["cwd"] as? String).flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 } ?? cwd
        let timeout = min(max((args["timeout_seconds"] as? Int) ?? intFromNumber(args["timeout_seconds"]) ?? 30, 1), 60)
        guard !command.isEmpty else {
            return "[tool error: shell_run missing command]"
        }
        let display = redact(command)
        switch shellMode {
        case .off:
            print("\n[shell blocked: \(display)]")
            log("shell blocked command=\(display)")
            return "[shell disabled]"
        case .ask:
            print("\nmodel wants shell: \(display)")
            print("cwd: \(toolCWD)")
            print("run? [y/N] ", terminator: "")
            fflush(stdout)
            let answer = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
            guard answer == "y" || answer == "yes" else {
                log("shell denied command=\(display)")
                return "[shell command denied]"
            }
        case .on:
            break
        }
        let result = try runShell(command: command, cwd: toolCWD, timeout: timeout, initiatedByModel: true)
        print("\nshell:\n\(result)")
        return result
    }

    private func chatCompletion() throws -> AssistantTurn {
        let body = requestBody()
        let data = try JSONSerialization.data(withJSONObject: body, options: [])
        var request = URLRequest(url: endpoint.appendingPathComponent("v1/chat/completions"), timeoutInterval: 600)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("caix/\(CaixBuildInfo.version)", forHTTPHeaderField: "User-Agent")
        request.httpBody = data

        let (responseData, response) = try synchronousData(for: request, progress: "thinking")
        guard let http = response as? HTTPURLResponse else {
            throw ChatTUIError("invalid response")
        }
        guard (200..<300).contains(http.statusCode) else {
            let text = String(data: responseData, encoding: .utf8) ?? ""
            throw ChatTUIError("HTTP \(http.statusCode): \(redact(text))")
        }
        guard
            let object = try JSONSerialization.jsonObject(with: responseData) as? [String: Any],
            let choices = object["choices"] as? [[String: Any]],
            let choice = choices.first,
            let message = choice["message"] as? [String: Any]
        else {
            throw ChatTUIError("invalid chat response")
        }
        let content = message["content"] as? String ?? ""
        let reasoning = message["reasoning_content"] as? String ?? ""
        let calls = (message["tool_calls"] as? [[String: Any]] ?? []).compactMap { row -> ToolCall? in
            guard let function = row["function"] as? [String: Any],
                  let name = function["name"] as? String
            else { return nil }
            return ToolCall(
                id: row["id"] as? String ?? "call_\(UUID().uuidString.prefix(8))",
                name: name,
                arguments: function["arguments"] as? String ?? "{}")
        }
        return AssistantTurn(
            content: content,
            reasoning: reasoning,
            toolCalls: calls,
            finishReason: choice["finish_reason"] as? String)
    }

    private func requestBody() -> [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "messages": history.map { ["role": $0.role, "content": $0.content] },
            "max_tokens": options.maxTokens,
            "temperature": options.temperature,
            "stream": false,
        ]
        if let topP = options.topP {
            body["top_p"] = topP
        }
        if shellMode != .off {
            body["tools"] = [
                [
                    "type": "function",
                    "function": [
                        "name": "shell_run",
                        "description": "Run a local shell command in the caix terminal session. Use only when command-line access is needed.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "command": [
                                    "type": "string",
                                    "description": "Shell command to run. Do not include tokens, passwords, or authorization headers.",
                                ],
                                "cwd": [
                                    "type": "string",
                                    "description": "Working directory. Defaults to the current caix chat cwd.",
                                ],
                                "timeout_seconds": [
                                    "type": "integer",
                                    "description": "Timeout from 1 to 60 seconds.",
                                ],
                            ],
                            "required": ["command"],
                        ],
                    ],
                ],
            ]
        }
        return body
    }

    private mutating func fetchModels() throws -> [String] {
        var request = URLRequest(url: endpoint.appendingPathComponent("v1/models"), timeoutInterval: 10)
        request.setValue("caix/\(CaixBuildInfo.version)", forHTTPHeaderField: "User-Agent")
        let (data, response) = try synchronousData(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw ChatTUIError("could not reach \(endpoint.absoluteString)/v1/models")
        }
        guard
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let rows = object["data"] as? [[String: Any]]
        else { return [] }
        let names = rows.compactMap { $0["id"] as? String }.sorted()
        log("models \(names.count)")
        return names
    }

    private func fetchActivity() throws -> [[String: Any]] {
        var request = URLRequest(url: endpoint.appendingPathComponent("api/activity"), timeoutInterval: 5)
        request.setValue("caix/\(CaixBuildInfo.version)", forHTTPHeaderField: "User-Agent")
        let (data, response) = try synchronousData(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw ChatTUIError("could not reach \(endpoint.absoluteString)/api/activity")
        }
        return (try JSONSerialization.jsonObject(with: data) as? [[String: Any]]) ?? []
    }

    private func fetchServerExportsDir() throws -> String? {
        var request = URLRequest(url: endpoint.appendingPathComponent("api/server"), timeoutInterval: 5)
        request.setValue("caix/\(CaixBuildInfo.version)", forHTTPHeaderField: "User-Agent")
        let (data, response) = try synchronousData(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            return nil
        }
        guard
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let exports = object["exportsDir"] as? String,
            !exports.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        return exports
    }

    private mutating func runShell(command: String, cwd: String, timeout: Int, initiatedByModel: Bool) throws -> String {
        let redactedCommand = redact(command)
        guard !containsCredentialSyntax(command) else {
            log("shell refused credential-shaped command=\(redactedCommand)")
            throw ChatTUIError("refusing shell command with credential-shaped arguments")
        }
        let expandedCWD = caixExpandPath(cwd)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expandedCWD, isDirectory: &isDir), isDir.boolValue else {
            throw ChatTUIError("cwd is not a directory: \(expandedCWD)")
        }
        log("shell \(initiatedByModel ? "tool" : "manual") timeout=\(timeout) cwd=\(expandedCWD) command=\(redactedCommand)")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", command]
        process.currentDirectoryURL = URL(fileURLWithPath: expandedCWD, isDirectory: true)
        process.environment = ProcessInfo.processInfo.environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()

        let deadline = Date().addingTimeInterval(TimeInterval(timeout))
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            process.terminate()
            Thread.sleep(forTimeInterval: 0.5)
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
        }
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        var text = String(data: data, encoding: .utf8) ?? ""
        if text.isEmpty {
            text = "[exit \(process.terminationStatus), no output]"
        } else {
            text = "[exit \(process.terminationStatus)]\n" + text
        }
        return String(redact(text).prefix(12_000))
    }

    private func synchronousData(for request: URLRequest, progress: String? = nil) throws -> (Data, URLResponse) {
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
                box.result = .failure(ChatTUIError("empty response"))
            }
            semaphore.signal()
        }
        task.resume()
        let frames = ["|", "/", "-", "\\"]
        var frame = 0
        var lastTick = Date.distantPast
        while semaphore.wait(timeout: .now()) == .timedOut {
            if let progress, Date().timeIntervalSince(lastTick) >= 0.12 {
                print("\r\(progress) \(frames[frame])", terminator: "")
                fflush(stdout)
                frame = (frame + 1) % frames.count
                lastTick = Date()
            }
            RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
        }
        if let progress {
            print("\r\(String(repeating: " ", count: progress.count + 4))\r", terminator: "")
            fflush(stdout)
        }
        return try box.result?.get() ?? { throw ChatTUIError("empty response") }()
    }

    private mutating func log(_ line: String) {
        let stamp = ISO8601DateFormatter().string(from: Date())
        sessionLog.append("\(stamp) \(redact(line))")
        if sessionLog.count > 500 {
            sessionLog.removeFirst(sessionLog.count - 500)
        }
    }
}

struct ChatTUIError: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

private func readLine(prompt: String) -> String? {
    print(prompt, terminator: "")
    fflush(stdout)
    return Swift.readLine()
}

private func terminalWidth() -> Int {
    var size = winsize()
    guard ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0, size.ws_col > 0 else {
        return 80
    }
    return Int(size.ws_col)
}

private func rule(width: Int) -> String {
    String(repeating: "-", count: max(24, width))
}

private func parseJSONObject(_ text: String) -> [String: Any] {
    guard let data = text.data(using: .utf8),
          let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return [:] }
    return object
}

private func intFromNumber(_ value: Any?) -> Int? {
    if let int = value as? Int { return int }
    if let double = value as? Double { return Int(double) }
    if let number = value as? NSNumber { return number.intValue }
    if let string = value as? String { return Int(string) }
    return nil
}

private func redact(_ text: String) -> String {
    var out = text
    let patterns = [
        #"(?i)\b((?:[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|AUTHORIZATION|CREDENTIAL)[A-Z0-9_]*)\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s]+)"#,
        #"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/\-]+=*"#,
        #"(?i)([?&](?:access_token|token|signature|x-amz-signature)=)[^&\s]+"#,
    ]
    for pattern in patterns {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
        let range = NSRange(out.startIndex..<out.endIndex, in: out)
        out = regex.stringByReplacingMatches(in: out, options: [], range: range, withTemplate: "$1[redacted]")
    }
    return out
}

private func containsCredentialSyntax(_ command: String) -> Bool {
    let patterns = [
        #"(?i)\b(?:H(?:F)_TOKEN|TOKEN|SECRET|PASSWORD|API_KEY|AUTHORIZATION|CREDENTIAL)\s*="#,
        #"(?i)\bAuthorization\s*:"#,
        #"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]+=*"#,
        #"(?i)(?:--token|--api-key|--password)(?:=|\s+)"#,
        #"(?i)(?:access_token|x-amz-signature)="#,
    ]
    return patterns.contains { pattern in
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
        return regex.firstMatch(
            in: command, options: [], range: NSRange(command.startIndex..<command.endIndex, in: command)) != nil
    }
}
