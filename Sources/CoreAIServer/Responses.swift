import Foundation
import HTTPTypes
import Hummingbird
import PipelineRuntime

// MARK: - JSON responses

/// Builds `application/json` responses by encoding `Encodable` values with Foundation, so the
/// serving layer doesn't depend on Hummingbird's content negotiation.
enum JSONResponder {
    static func encode<T: Encodable>(
        _ value: T,
        status: HTTPResponse.Status = .ok,
        headers extraHeaders: HTTPFields = HTTPFields()
    ) -> Response {
        let data = (try? JSONEncoder().encode(value)) ?? Data(#"{"error":"encoding failed"}"#.utf8)
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        for field in extraHeaders {
            headers.append(field)
        }
        return Response(
            status: status, headers: headers, body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
    }

    static func error(_ message: String, status: HTTPResponse.Status) -> Response {
        encode(["error": ["message": message, "type": "coreai_error"]], status: status)
    }

    static func conversionActive(_ decision: ConversionGuard.Decision) -> Response {
        var headers = HTTPFields()
        headers[.retryAfter] = "\(decision.retryAfterSeconds)"
        return encode(
            [
                "error": [
                    "message": JSONValue.string(
                        "generation is temporarily unavailable while a model conversion is active"),
                    "type": JSONValue.string("conversion_active"),
                    "reason": JSONValue.string(decision.reason),
                    "retry_after_seconds": JSONValue.int(decision.retryAfterSeconds),
                ],
            ] as [String: [String: JSONValue]],
            status: .serviceUnavailable,
            headers: headers)
    }
}

/// Minimal heterogeneous JSON value for ad-hoc response dictionaries.
enum JSONValue: Encodable, Sendable {
    case string(String), bool(Bool), int(Int), double(Double), null

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let s): try c.encode(s)
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let d): try c.encode(d)
        case .null: try c.encodeNil()
        }
    }
}

extension JSONValue: ExpressibleByBooleanLiteral {
    init(booleanLiteral value: Bool) { self = .bool(value) }
}
extension JSONValue: ExpressibleByStringLiteral {
    init(stringLiteral value: String) { self = .string(value) }
}

// MARK: - SSE helpers + streaming builders

extension ServerRuntime {
    private static func sseHeaders() -> HTTPFields {
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return headers
    }

    /// `data: <json>\n\n` for an `Encodable` chunk (OpenAI framing).
    private static func sseData<T: Encodable>(_ value: T) -> ByteBuffer {
        let json = (try? JSONEncoder().encode(value)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return ByteBuffer(string: "data: \(json)\n\n")
    }

    /// `event: <name>\ndata: <payload>\n\n` (Anthropic framing).
    private static func sseEvent(_ name: String, _ payload: String) -> ByteBuffer {
        ByteBuffer(string: "event: \(name)\ndata: \(payload)\n\n")
    }

    /// JSON-escape a string into a quoted literal (for hand-built Anthropic event payloads).
    static func jsonQuoted(_ s: String) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
        return out
    }

    /// OpenAI `chat.completion.chunk` SSE stream. Raw token deltas are run through the per-model
    /// ``StreamingNormalizer`` and re-emitted as `delta.reasoning_content` (during reasoning),
    /// `delta.content` (final text), and `delta.tool_calls` (one chunk per parsed call).
    static func openAIStream(
        handle: ModelHandle, messages: [[String: String]], options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]?, format: OutputFormat,
        model: String, id: String, created: Int,
        includeUsage: Bool = false,
        activity: ActivityLog? = nil, startedAt: Date? = nil
    ) -> Response {
        let requestStart = startedAt ?? Date()
        let (stream, continuation) = AsyncStream<String>.makeStream()
        let genTask = Task<CoreAIPipeline.Result, Error> {
            defer { continuation.finish() }
            return try await handle.generate(messages: messages, options: options, tools: tools) { delta in
                continuation.yield(delta)
            }
        }

        let body = ResponseBody { writer in
            let parser = StreamingNormalizer(format: format)
            var sawToolCall = false

            func emit(_ events: [StreamingNormalizer.Event]) async throws {
                for event in events {
                    switch event {
                    case .reasoning(let s):
                        try await writer.write(sseData(OpenAIChatChunk(
                            id: id, model: model, created: created, reasoningContent: s)))
                    case .text(let s):
                        try await writer.write(sseData(OpenAIChatChunk(
                            id: id, model: model, created: created, content: s)))
                    case .toolCall(let tc):
                        sawToolCall = true
                        let delta = OpenAIChatChunk.ToolCallDelta(
                            index: tc.index, id: tc.id, type: "function",
                            function: .init(name: tc.name, arguments: tc.arguments))
                        try await writer.write(sseData(OpenAIChatChunk(
                            id: id, model: model, created: created, toolCalls: [delta])))
                    }
                }
            }

            // Initial role delta.
            try await writer.write(
                sseData(OpenAIChatChunk(id: id, model: model, created: created, role: "assistant")))
            var firstDeltaAt: Date?
            for await delta in stream {
                if firstDeltaAt == nil {
                    firstDeltaAt = Date()
                }
                try await emit(parser.push(delta))
            }
            try await emit(parser.finish())

            var finish = "stop"
            let result = try? await genTask.value
            if let result { finish = openAIFinish(result.stopReason) }
            if sawToolCall { finish = "tool_calls" }
            let usage = result.map {
                OpenAIChatResponse.Usage(
                    prompt_tokens: $0.promptTokenCount,
                    completion_tokens: $0.generatedTokenCount,
                    total_tokens: $0.promptTokenCount + $0.generatedTokenCount)
            }
            await activity?.record(
                method: "POST",
                path: "/v1/chat/completions",
                status: result == nil ? 500 : 200,
                startedAt: requestStart,
                model: model,
                summary: result == nil ? "stream failed" : "stream completed (\(finish))",
                inputTokens: result?.promptTokenCount,
                outputTokens: result?.generatedTokenCount,
                firstTokenSeconds: firstDeltaAt?.timeIntervalSince(requestStart),
                loadSeconds: result?.modelLoadSeconds,
                prefillSeconds: result?.prefillSeconds,
                decodeSeconds: result?.decodeSeconds)
            try await writer.write(
                sseData(OpenAIChatChunk(
                    id: id,
                    model: model,
                    created: created,
                    finish: finish,
                    usage: includeUsage ? usage : nil)))
            try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            try await writer.finish(nil)
        }
        return Response(status: .ok, headers: sseHeaders(), body: body)
    }

    /// Anthropic Messages SSE stream. Reasoning becomes a `thinking` content block, final text a
    /// `text` block, and each tool call a `tool_use` block — emitted in order with
    /// content_block_start / _delta (thinking_delta · text_delta · input_json_delta) / _stop.
    static func anthropicStream(
        handle: ModelHandle, messages: [[String: String]], options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]?, format: OutputFormat,
        model: String, id: String,
        activity: ActivityLog? = nil, startedAt: Date? = nil
    ) -> Response {
        let requestStart = startedAt ?? Date()
        let (stream, continuation) = AsyncStream<String>.makeStream()
        let genTask = Task<CoreAIPipeline.Result, Error> {
            defer { continuation.finish() }
            return try await handle.generate(messages: messages, options: options, tools: tools) { delta in
                continuation.yield(delta)
            }
        }

        let body = ResponseBody { writer in
            let parser = StreamingNormalizer(format: format)
            var blockIndex = -1
            // Currently open streaming block: 0 none, 1 thinking, 2 text.
            var openKind = 0
            var sawToolCall = false
            var anyBlock = false

            func closeOpen() async throws {
                if openKind != 0 {
                    try await writer.write(sseEvent(
                        "content_block_stop",
                        "{\"type\":\"content_block_stop\",\"index\":\(blockIndex)}"))
                    openKind = 0
                }
            }
            func openBlock(kind: Int, startBody: String) async throws {
                try await closeOpen()
                blockIndex += 1
                anyBlock = true
                try await writer.write(sseEvent(
                    "content_block_start",
                    "{\"type\":\"content_block_start\",\"index\":\(blockIndex),\"content_block\":\(startBody)}"))
                openKind = kind
            }

            func emit(_ events: [StreamingNormalizer.Event]) async throws {
                for event in events {
                    switch event {
                    case .reasoning(let s):
                        if openKind != 1 {
                            try await openBlock(kind: 1, startBody: "{\"type\":\"thinking\",\"thinking\":\"\"}")
                        }
                        try await writer.write(sseEvent(
                            "content_block_delta",
                            "{\"type\":\"content_block_delta\",\"index\":\(blockIndex),\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\(jsonQuoted(s))}}"))
                    case .text(let s):
                        if openKind != 2 {
                            try await openBlock(kind: 2, startBody: "{\"type\":\"text\",\"text\":\"\"}")
                        }
                        try await writer.write(sseEvent(
                            "content_block_delta",
                            "{\"type\":\"content_block_delta\",\"index\":\(blockIndex),\"delta\":{\"type\":\"text_delta\",\"text\":\(jsonQuoted(s))}}"))
                    case .toolCall(let tc):
                        sawToolCall = true
                        let start = "{\"type\":\"tool_use\",\"id\":\(jsonQuoted(tc.id)),\"name\":\(jsonQuoted(tc.name)),\"input\":{}}"
                        try await openBlock(kind: 3, startBody: start)
                        try await writer.write(sseEvent(
                            "content_block_delta",
                            "{\"type\":\"content_block_delta\",\"index\":\(blockIndex),\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\(jsonQuoted(tc.arguments))}}"))
                        try await closeOpen()
                    }
                }
            }

            try await writer.write(
                sseEvent(
                    "message_start",
                    "{\"type\":\"message_start\",\"message\":{\"id\":\"\(id)\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"\(model)\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}"
                ))
            try await writer.write(sseEvent("ping", "{\"type\":\"ping\"}"))

            var firstDeltaAt: Date?
            for await delta in stream {
                if firstDeltaAt == nil {
                    firstDeltaAt = Date()
                }
                try await emit(parser.push(delta))
            }
            try await emit(parser.finish())
            try await closeOpen()

            // Ensure at least one content block exists.
            if !anyBlock {
                try await writer.write(sseEvent(
                    "content_block_start",
                    "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}"))
                try await writer.write(sseEvent(
                    "content_block_stop", "{\"type\":\"content_block_stop\",\"index\":0}"))
            }

            var stop = "end_turn"
            var outTokens = 0
            let result = try? await genTask.value
            if let result {
                stop = anthropicStop(result.stopReason)
                outTokens = result.generatedTokenCount
            }
            if sawToolCall { stop = "tool_use" }
            await activity?.record(
                method: "POST",
                path: "/v1/messages",
                status: result == nil ? 500 : 200,
                startedAt: requestStart,
                model: model,
                summary: result == nil ? "stream failed" : "stream completed (\(stop))",
                inputTokens: result?.promptTokenCount,
                outputTokens: result?.generatedTokenCount,
                firstTokenSeconds: firstDeltaAt?.timeIntervalSince(requestStart),
                loadSeconds: result?.modelLoadSeconds,
                prefillSeconds: result?.prefillSeconds,
                decodeSeconds: result?.decodeSeconds)
            try await writer.write(
                sseEvent(
                    "message_delta",
                    "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"\(stop)\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":\(outTokens)}}"
                ))
            try await writer.write(sseEvent("message_stop", "{\"type\":\"message_stop\"}"))
            try await writer.finish(nil)
        }
        return Response(status: .ok, headers: sseHeaders(), body: body)
    }
}
