import Foundation
import HTTPTypes
import Hummingbird
import MachineStats
import PipelineRuntime

/// Native HTTP serving layer for the Core AI pipeline.
///
/// Exposes OpenAI- and Anthropic-compatible chat endpoints (with SSE streaming), the
/// dashboard `/api/*` surface, and static hosting of `web/index.html`, all backed by the
/// `ModelManager` (persistent `.aimodel` load/offload/hot-swap on `PipelineRuntime`'s
/// `LLMEngine` handles). Generation requires the Core AI runtime (`COREAI_RUNTIME`); without
/// it the server still runs and serves stats/models/dashboard, and inference endpoints return
/// `503` with a clear "runtime unavailable" message.
public enum CoreAIServer {
    /// Start the server and block until shut down (SIGINT/SIGTERM).
    public static func serve(
        host: String = "127.0.0.1",
        port: Int = 8080,
        exportsDir: String,
        registryPath: String,
        webDir: String,
        convertScript: String,
        pythonExecutable: String = "python3",
        verbose: Bool = false,
        eagleConfig: EagleConfig? = nil,
        statsFile: String? = nil
    ) async throws {
        // Persist usage stats (totals + per-model) across restarts, ollama/omlx-style.
        let statsPath = statsFile ?? (NSHomeDirectory() + "/.caix/usage.json")
        Usage.configure(path: statsPath)
        let runtime = ServerRuntime(
            host: host,
            port: port,
            exportsDir: URL(fileURLWithPath: exportsDir, isDirectory: true),
            registryPath: URL(fileURLWithPath: registryPath),
            webDir: URL(fileURLWithPath: webDir, isDirectory: true),
            convertScript: convertScript,
            pythonExecutable: pythonExecutable,
            verbose: verbose,
            eagleConfig: eagleConfig)

        let router = Router()
        runtime.register(on: router)

        let app = Application(
            router: router,
            configuration: .init(
                address: .hostname(host, port: port),
                serverName: "coreai-pipeline"))

        let linked = CoreAIPipeline.isLinked ? "linked" : "NOT linked (set COREAI_RUNTIME=1)"
        FileHandle.standardError.write(
            Data(
                """
                caix · native Core AI runtime
                  listening   http://\(host):\(port)
                  dashboard   http://\(host):\(port)/
                  openai      POST /v1/chat/completions   GET /v1/models
                  anthropic   POST /v1/messages
                  dashboard   GET /api/stats  GET /api/models  GET /api/jobs
                              POST /api/convert  POST /api/load  POST /api/offload
                  runtime     \(linked)
                """.appending("\n").utf8))

        try await app.runService()
    }
}

/// Shared, immutable server state captured by the route handlers. `Sendable`: every stored
/// value is itself `Sendable` (the manager/job-tracker are actors).
final class ServerRuntime: Sendable {
    let manager: ModelManager
    let jobs: JobTracker
    let host: String
    let port: Int
    let exportsDir: URL
    let webDir: URL
    let registryPath: URL
    let convertScript: String
    let pythonExecutable: String
    let indexHTML: String
    let chatHTML: String
    let verbose: Bool

    init(
        host: String, port: Int, exportsDir: URL, registryPath: URL, webDir: URL, convertScript: String,
        pythonExecutable: String, verbose: Bool, eagleConfig: EagleConfig? = nil
    ) {
        self.host = host
        self.port = port
        self.exportsDir = exportsDir
        self.manager = ModelManager(
            exportsDir: exportsDir, registryPath: registryPath, verbose: verbose,
            eagleConfig: eagleConfig)
        self.jobs = JobTracker()
        self.webDir = webDir
        self.registryPath = registryPath
        self.convertScript = convertScript
        self.pythonExecutable = pythonExecutable
        self.verbose = verbose
        let indexURL = webDir.appendingPathComponent("index.html")
        self.indexHTML =
            (try? String(contentsOf: indexURL, encoding: .utf8))
            ?? "<!doctype html><title>caix</title><p>web/index.html not found at \(indexURL.path)</p>"
        let chatURL = webDir.appendingPathComponent("chat.html")
        self.chatHTML =
            (try? String(contentsOf: chatURL, encoding: .utf8))
            ?? "<!doctype html><title>caix chat</title><p>web/chat.html not found at \(chatURL.path)</p>"
    }

    // MARK: - Route registration

    func register(on router: Router<BasicRequestContext>) {
        // Dashboard (static) + dedicated chat view
        router.get("/") { _, _ in self.htmlResponse(self.indexHTML) }
        router.get("/chat") { _, _ in self.htmlResponse(self.chatHTML) }
        router.get("/assets/:file") { _, ctx in self.assetHandler(ctx) }

        // Dashboard API
        router.get("/api/stats") { _, _ in JSONResponder.encode(MachineStats.snapshot()) }
        router.get("/api/models") { _, _ in JSONResponder.encode(await self.manager.listModels()) }
        router.get("/api/jobs") { _, _ in JSONResponder.encode(await self.jobs.snapshot()) }
        router.get("/api/server") { _, _ in self.serverInfoHandler() }
        router.post("/api/load") { req, ctx in try await self.loadHandler(req, ctx) }
        router.post("/api/offload") { req, ctx in try await self.offloadHandler(req, ctx) }
        router.post("/api/delete") { req, ctx in try await self.deleteHandler(req, ctx) }
        router.post("/api/convert") { req, ctx in try await self.convertHandler(req, ctx) }
        // Model management: supported types/settings, HF support check, HF convert, live gen stats.
        router.get("/api/supported") { _, _ in self.supportedHandler() }
        router.post("/api/check-support") { req, ctx in try await self.checkSupportHandler(req, ctx) }
        router.post("/api/convert-hf") { req, ctx in try await self.convertHFHandler(req, ctx) }
        router.get("/api/genstats") { _, _ in self.genStatsHandler() }
        router.get("/api/usage") { _, _ in JSONResponder.encode(Usage.snapshot(now: Date().timeIntervalSince1970)) }
        // Tool proxies for the chat view (browser can't reach these directly: CORS).
        router.post("/api/proxy-fetch") { req, ctx in try await self.proxyFetchHandler(req, ctx) }
        router.post("/api/mcp") { req, ctx in try await self.mcpHandler(req, ctx) }

        // OpenAI-compatible
        router.get("/v1/models") { _, _ in await self.openAIModelsHandler() }
        router.post("/v1/chat/completions") { req, ctx in try await self.openAIChatHandler(req, ctx) }

        // Anthropic-compatible
        router.post("/v1/messages") { req, ctx in try await self.anthropicMessagesHandler(req, ctx) }
    }

    // MARK: - Dashboard API handlers

    private func loadHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        guard let body = try? await Self.decode(ModelActionRequest.self, request) else {
            return JSONResponder.error("invalid request body (expected {\"model\": ...})", status: .badRequest)
        }
        do {
            _ = try await manager.load(body.model)
            return JSONResponder.encode(["ok": true, "model": .string(body.model)] as [String: JSONValue])
        } catch {
            return JSONResponder.error("load failed: \(error)", status: .internalServerError)
        }
    }

    private func offloadHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        guard let body = try? await Self.decode(ModelActionRequest.self, request) else {
            return JSONResponder.error("invalid request body (expected {\"model\": ...})", status: .badRequest)
        }
        let was = await manager.offload(body.model)
        return JSONResponder.encode(["ok": .bool(true), "offloaded": .bool(was)] as [String: JSONValue])
    }

    private func deleteHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        guard let body = try? await Self.decode(ModelActionRequest.self, request) else {
            return JSONResponder.error("invalid request body (expected {\"model\": ...})", status: .badRequest)
        }
        if let err = await manager.deleteBundle(body.model) {
            return JSONResponder.error(err, status: .badRequest)
        }
        return JSONResponder.encode(["ok": .bool(true), "deleted": .string(body.model)] as [String: JSONValue])
    }

    private func convertHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        guard let body = try? await Self.decode(ModelActionRequest.self, request) else {
            return JSONResponder.error("invalid request body (expected {\"model\": ...})", status: .badRequest)
        }
        let err = await jobs.startConvert(
            model: body.model,
            script: convertScript,
            workingDir: registryPath.deletingLastPathComponent().deletingLastPathComponent(),
            pythonExecutable: pythonExecutable)
        if let err {
            return JSONResponder.error(err, status: .badRequest)
        }
        return JSONResponder.encode(["ok": .bool(true), "started": .string(body.model)] as [String: JSONValue])
    }

    // MARK: - Model management (supported types, HF support check, HF convert, gen stats)

    private var convertWorkingDir: URL {
        registryPath.deletingLastPathComponent().deletingLastPathComponent()
    }

    /// `GET /api/supported` — the Core AI export knobs the UI exposes.
    private func supportedHandler() -> Response {
        let payload: [String: JSONValue] = [
            "compression": .string("none,4bit,8bit"),
            "compute_precision": .string("float16,bfloat16,float32"),
            "generation": .string("max_tokens,temperature,top_p,top_k,seed,stop,system_prompt"),
            "context_default": .int(4096),
            "note": .string("Paste a HuggingFace repo; unsupported architectures are flagged, supported ones convert + load."),
        ]
        return JSONResponder.encode(payload)
    }

    /// `GET /api/server` — read-only runtime metadata for the dashboard's server panel.
    private func serverInfoHandler() -> Response {
        let eagle = manager.eagleSummary()
        let info = ServerInfo(
            ok: true,
            name: "coreai-pipeline",
            pid: Int(ProcessInfo.processInfo.processIdentifier),
            runtimeLinked: CoreAIPipeline.isLinked,
            host: host,
            port: port,
            baseURL: "http://\(host):\(port)",
            webDir: webDir.path,
            exportsDir: exportsDir.path,
            registryPath: registryPath.path,
            convertScript: convertScript,
            pythonExecutable: pythonExecutable,
            computeUnit: ProcessInfo.processInfo.environment["COREAI_COMPUTE"] ?? "gpu",
            verbose: verbose,
            eagle: eagle)
        return JSONResponder.encode(info)
    }

    /// `POST /api/check-support` {hf_repo} — returns the raw check JSON (supported/flagged + reason).
    private func checkSupportHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        guard let body = try? await Self.decode(CheckSupportRequest.self, request),
              !body.hf_repo.trimmingCharacters(in: .whitespaces).isEmpty else {
            return JSONResponder.error("expected {\"hf_repo\": \"org/model\"}", status: .badRequest)
        }
        let json = await jobs.runCheckSupport(
            hfRepo: body.hf_repo, script: convertScript, workingDir: convertWorkingDir,
            pythonExecutable: pythonExecutable)
        var headers = HTTPFields(); headers[.contentType] = "application/json"
        return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: ByteBuffer(string: json)))
    }

    /// `POST /api/convert-hf` {hf_repo, name?, compression?, compute_precision?, context?} —
    /// gate on support, then launch the settings-aware conversion.
    private func convertHFHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        guard let body = try? await Self.decode(HFConvertRequest.self, request),
              !body.hf_repo.trimmingCharacters(in: .whitespaces).isEmpty else {
            return JSONResponder.error("expected {\"hf_repo\": \"org/model\", ...settings}", status: .badRequest)
        }
        // Support gate (so we can return a clean flagged status before launching).
        let checkJSON = await jobs.runCheckSupport(
            hfRepo: body.hf_repo, script: convertScript, workingDir: convertWorkingDir,
            pythonExecutable: pythonExecutable)
        let checkObject = (try? JSONSerialization.jsonObject(with: Data(checkJSON.utf8)) as? [String: Any])
        let supported = (checkObject?["supported"] as? Bool) ?? false
        let ggufOnly = (checkObject?["gguf_only"] as? Bool) ?? false
        if !supported && !ggufOnly {
            var headers = HTTPFields(); headers[.contentType] = "application/json"
            // Pass the check JSON straight through (carries model_type, supported_types, reason).
            return Response(status: .ok, headers: headers,
                            body: ResponseBody(byteBuffer: ByteBuffer(string: checkJSON)))
        }
        let name = body.name?.trimmingCharacters(in: .whitespaces).nonEmpty
            ?? Self.defaultModelName(for: body.hf_repo)
        let err: String?
        if ggufOnly {
            err = await jobs.startConvertGGUF(
                name: name, ggufRepo: body.hf_repo, ggufFile: body.gguf_file?.nonEmpty,
                compression: body.compression?.nonEmpty ?? "4bit",
                precision: body.compute_precision?.nonEmpty ?? "float16",
                context: body.context, script: convertScript, workingDir: convertWorkingDir,
                pythonExecutable: pythonExecutable)
        } else {
            err = await jobs.startConvertHF(
                name: name, hfRepo: body.hf_repo,
                compression: body.compression?.nonEmpty ?? "4bit",
                precision: body.compute_precision?.nonEmpty ?? "float16",
                context: body.context, script: convertScript, workingDir: convertWorkingDir,
                pythonExecutable: pythonExecutable)
        }
        if let err {
            return JSONResponder.error(err, status: .badRequest)
        }
        return JSONResponder.encode([
            "ok": .bool(true), "supported": .bool(true), "started": .bool(true),
            "gguf": .bool(ggufOnly),
            "name": .string(name)] as [String: JSONValue])
    }

    /// `POST /api/proxy-fetch` {url} — fetch a web page server-side (the `fetch_url` tool), return
    /// its readable text (crude tag-strip). Only http/https.
    private func proxyFetchHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        struct Req: Codable { let url: String }
        guard let body = try? await Self.decode(Req.self, request),
              let url = URL(string: body.url), let scheme = url.scheme,
              scheme == "http" || scheme == "https" else {
            return JSONResponder.error("expected {\"url\":\"http(s)://…\"}", status: .badRequest)
        }
        do {
            var req = URLRequest(url: url, timeoutInterval: 20)
            req.setValue("caix/1.0", forHTTPHeaderField: "User-Agent")
            let (data, _) = try await URLSession.shared.data(for: req)
            var text = String(data: data, encoding: .utf8) ?? ""
            // crude readable-text extraction
            text = text.replacingOccurrences(of: "(?s)<script.*?</script>", with: " ", options: .regularExpression)
            text = text.replacingOccurrences(of: "(?s)<style.*?</style>", with: " ", options: .regularExpression)
            text = text.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
            text = text.replacingOccurrences(of: "&nbsp;", with: " ")
            text = text.replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)
            text = text.replacingOccurrences(of: "\n\\s*\n\\s*\n+", with: "\n\n", options: .regularExpression)
            return JSONResponder.encode(["text": String(text.prefix(8000))])
        } catch {
            return JSONResponder.error("fetch failed: \(error.localizedDescription)", status: .badGateway)
        }
    }

    /// `POST /api/mcp` {url, method, params?} — proxy a JSON-RPC call to a streamable-HTTP MCP
    /// server (does the initialize handshake first), returning `{result}` or `{error}`. Stateless:
    /// each request re-initializes (simple; fine for tools/list + tools/call).
    private func mcpHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        var req = request
        guard let buffer = try? await req.collectBody(upTo: 1024 * 1024),
              let obj = try? JSONSerialization.jsonObject(with: Data(buffer.readableBytesView)) as? [String: Any],
              let urlStr = obj["url"] as? String, let url = URL(string: urlStr), let method = obj["method"] as? String else {
            return JSONResponder.error("expected {\"url\":…,\"method\":…,\"params\":…}", status: .badRequest)
        }
        let params = obj["params"] as? [String: Any]
        do {
            let result = try await MCPProxy.call(url: url, method: method, params: params)
            let data = try JSONSerialization.data(withJSONObject: ["result": result])
            var headers = HTTPFields(); headers[.contentType] = "application/json"
            return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
        } catch {
            return JSONResponder.encode(["error": ["message": "\(error)"]])
        }
    }

    /// `GET /api/genstats` — last speculative-decoding generation metrics (tok/s, acceptance, …).
    private func genStatsHandler() -> Response {
        guard let s = LiveStats.last else {
            return JSONResponder.encode(["available": JSONValue.bool(false)])
        }
        return JSONResponder.encode(s)
    }

    // MARK: - OpenAI handlers

    private func openAIModelsHandler() async -> Response {
        let created = Int(Date().timeIntervalSince1970)
        let models = await manager.listModels().filter { $0.bundle }
        let list = OpenAIModelList(
            data: models.map { .init(id: $0.name, created: created) })
        return JSONResponder.encode(list)
    }

    private func openAIChatHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        let req: OpenAIChatRequest
        do {
            req = try await Self.decode(OpenAIChatRequest.self, request)
        } catch {
            return JSONResponder.error("invalid OpenAI chat request: \(error)", status: .badRequest)
        }
        let gen = req.toGeneration()
        let modelName = await resolveModelName(gen.model)
        let handle: ModelHandle
        do {
            handle = try await manager.handle(for: modelName)
        } catch {
            return JSONResponder.error("could not load model '\(modelName)': \(error)", status: .serviceUnavailable)
        }

        let messages = Self.messagePayload(gen.messages)
        let options = Self.options(from: gen)
        let tools = gen.toolSpecs
        let format = await manager.outputFormat(for: modelName)
        let id = "chatcmpl-" + Self.shortID()
        let created = Int(Date().timeIntervalSince1970)

        if gen.stream {
            return Self.openAIStream(
                handle: handle, messages: messages, options: options, tools: tools, format: format,
                model: modelName, id: id, created: created)
        }

        do {
            let result = try await handle.generate(messages: messages, options: options, tools: tools)
            // Normalize the raw base-format output into reasoning_content + content + tool_calls.
            let norm = StreamingNormalizer.normalizeComplete(result.text, format: format)
            let message = OpenAIResponseMessage(
                content: norm.text.isEmpty && norm.hasToolCalls ? nil : norm.text,
                reasoning_content: norm.reasoning.isEmpty ? nil : norm.reasoning,
                tool_calls: norm.hasToolCalls
                    ? norm.toolCalls.map { OpenAIToolCall(id: $0.id, name: $0.name, arguments: $0.arguments) }
                    : nil)
            let finish = norm.hasToolCalls ? "tool_calls" : Self.openAIFinish(result.stopReason)
            let response = OpenAIChatResponse(
                id: id, model: modelName, created: created, message: message, finish: finish,
                promptTokens: result.promptTokenCount, completionTokens: result.generatedTokenCount)
            return JSONResponder.encode(response)
        } catch {
            return JSONResponder.error("generation failed: \(error)", status: .internalServerError)
        }
    }

    // MARK: - Anthropic handler

    private func anthropicMessagesHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        let req: AnthropicMessagesRequest
        do {
            req = try await Self.decode(AnthropicMessagesRequest.self, request)
        } catch {
            return JSONResponder.error("invalid Anthropic messages request: \(error)", status: .badRequest)
        }
        let gen = req.toGeneration()
        let modelName = await resolveModelName(gen.model)
        let handle: ModelHandle
        do {
            handle = try await manager.handle(for: modelName)
        } catch {
            return JSONResponder.error("could not load model '\(modelName)': \(error)", status: .serviceUnavailable)
        }

        let messages = Self.messagePayload(gen.messages)
        let options = Self.options(from: gen)
        let tools = gen.toolSpecs
        let format = await manager.outputFormat(for: modelName)
        let id = "msg_" + Self.shortID()

        if gen.stream {
            return Self.anthropicStream(
                handle: handle, messages: messages, options: options, tools: tools, format: format,
                model: modelName, id: id)
        }

        do {
            let result = try await handle.generate(messages: messages, options: options, tools: tools)
            // Normalize into thinking + text + tool_use content blocks (in that order).
            let norm = StreamingNormalizer.normalizeComplete(result.text, format: format)
            var blocks: [AnthropicBlock] = []
            if !norm.reasoning.isEmpty { blocks.append(.thinking(norm.reasoning)) }
            if !norm.text.isEmpty { blocks.append(.text(norm.text)) }
            for call in norm.toolCalls {
                blocks.append(.toolUse(id: call.id, name: call.name, input: JSONAny.parse(call.arguments)))
            }
            if blocks.isEmpty { blocks.append(.text("")) }
            let stop = norm.hasToolCalls ? "tool_use" : Self.anthropicStop(result.stopReason)
            let response = AnthropicMessagesResponse(
                id: id, model: modelName, blocks: blocks, stopReason: stop,
                inputTokens: result.promptTokenCount, outputTokens: result.generatedTokenCount)
            return JSONResponder.encode(response)
        } catch {
            return JSONResponder.error("generation failed: \(error)", status: .internalServerError)
        }
    }

    // MARK: - Model resolution

    /// Map a requested model name to an available bundle, falling back to the first bundle so
    /// generic client model ids (e.g. "gpt-4") still resolve to the local model.
    private func resolveModelName(_ requested: String) async -> String {
        let bundles = await manager.listModels().filter { $0.bundle }
        if bundles.contains(where: { $0.name == requested }) { return requested }
        return bundles.first?.name ?? requested
    }

    // MARK: - Mapping helpers

    static func messagePayload(_ messages: [ChatMessage]) -> [[String: String]] {
        messages.map { ["role": $0.role, "content": $0.content] }
    }

    static func options(from gen: GenerationRequest) -> CoreAIPipeline.Options {
        CoreAIPipeline.Options(
            maxTokens: gen.maxTokens,
            temperature: gen.temperature,
            topK: gen.topK,
            topP: gen.topP,
            applyChatTemplate: true,
            stopSequences: gen.stop,
            seed: gen.seed)
    }

    static func openAIFinish(_ reason: CoreAIPipeline.StopReason) -> String {
        switch reason {
        case .eos, .stopSequence: return "stop"
        case .maxTokens, .contextLimit: return "length"
        }
    }

    static func anthropicStop(_ reason: CoreAIPipeline.StopReason) -> String {
        switch reason {
        case .eos: return "end_turn"
        case .stopSequence: return "stop_sequence"
        case .maxTokens, .contextLimit: return "max_tokens"
        }
    }

    static func shortID() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24).lowercased()
    }

    private static func defaultModelName(for repo: String) -> String {
        let base = repo.split(separator: "/").last.map(String.init) ?? "model"
        let cleaned = base
            .replacingOccurrences(of: ".gguf", with: "", options: [.caseInsensitive])
            .replacingOccurrences(of: "-GGUF", with: "", options: [.caseInsensitive])
            .replacingOccurrences(of: "_GGUF", with: "", options: [.caseInsensitive])
        return cleaned.lowercased() + "-coreai"
    }

    // MARK: - Request decoding / static responses

    static func decode<T: Decodable>(_ type: T.Type, _ request: Request) async throws -> T {
        var request = request
        let buffer = try await request.collectBody(upTo: 8 * 1024 * 1024)
        let data = Data(buffer.readableBytesView)
        return try JSONDecoder().decode(T.self, from: data)
    }

    /// Serve a file from `webDir/assets/` (local JS/CSS for the chat view). Path-traversal-safe.
    private func assetHandler(_ context: BasicRequestContext) -> Response {
        guard let name = context.parameters.get("file"), !name.contains("/"), !name.contains("..") else {
            return JSONResponder.error("bad asset", status: .badRequest)
        }
        let url = webDir.appendingPathComponent("assets").appendingPathComponent(name)
        guard let data = try? Data(contentsOf: url) else {
            return JSONResponder.error("asset not found", status: .notFound)
        }
        let ct: String = name.hasSuffix(".js") ? "application/javascript; charset=utf-8"
            : name.hasSuffix(".css") ? "text/css; charset=utf-8"
            : name.hasSuffix(".svg") ? "image/svg+xml" : "application/octet-stream"
        var headers = HTTPFields(); headers[.contentType] = ct
        headers[.cacheControl] = "public, max-age=86400"
        return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
    }

    private func htmlResponse(_ html: String) -> Response {
        var headers = HTTPFields()
        headers[.contentType] = "text/html; charset=utf-8"
        return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: ByteBuffer(string: html)))
    }
}

// MARK: - Action request DTO

extension String {
    /// `nil` when the string is empty, else self — for falling back to defaults.
    var nonEmpty: String? { isEmpty ? nil : self }
}

struct ModelActionRequest: Codable, Sendable {
    let model: String
}

/// `POST /api/check-support` body.
struct CheckSupportRequest: Codable, Sendable {
    let hf_repo: String
}

/// `POST /api/convert-hf` body — a raw HF repo + the Core AI export settings.
struct HFConvertRequest: Codable, Sendable {
    let hf_repo: String
    var name: String?
    var compression: String?        // none | 4bit | 8bit
    var compute_precision: String?  // float16 | bfloat16 | float32
    var context: Int?
    var gguf_file: String?
}

struct ServerInfo: Codable, Sendable {
    struct Eagle: Codable, Sendable {
        var enabled: Bool
        var name: String?
        var targetPath: String?
        var draftPath: String?
        var unrolledPath: String?
        var tokenizerDir: String?
    }
    var ok: Bool
    var name: String
    var pid: Int
    var runtimeLinked: Bool
    var host: String
    var port: Int
    var baseURL: String
    var webDir: String
    var exportsDir: String
    var registryPath: String
    var convertScript: String
    var pythonExecutable: String
    var computeUnit: String
    var verbose: Bool
    var eagle: Eagle
}

// MARK: - OpenAI model list DTO

struct OpenAIModelList: Codable, Sendable {
    struct Model: Codable, Sendable {
        var id: String
        var object: String = "model"
        var created: Int
        var owned_by: String = "coreai-pipeline"
    }
    var object: String = "list"
    var data: [Model]
}
