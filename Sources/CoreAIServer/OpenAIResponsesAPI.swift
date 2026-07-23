import Foundation
import HTTPTypes
import Hummingbird
import PipelineRuntime

// OpenAI Responses API (`POST /v1/responses`): request/response schemas, lenient input parsing
// (string shorthand or item arrays with text/audio parts), and the route handler. Generation
// reuses the exact chat-completions path (ModelManager handle + CoreAIPipeline options +
// StreamingNormalizer); `input_audio` parts are transcribed through the resident native Whisper
// service and spliced into the message text before templating, so one request can exercise
// Whisper + the chat model together. Stateless: `store` is accepted and always echoed `false`.

// MARK: - Request schema

public struct OpenAIResponsesRequest: Decodable, Sendable {
    public var model: String
    public var input: OpenAIResponsesInput?
    /// Optional system guidance; becomes the (prepended) system message.
    public var instructions: String?
    public var max_output_tokens: Int?
    public var temperature: Double?
    public var top_p: Double?
    public var stream: Bool?
    /// Accepted and ignored — this server is stateless and always echoes `store: false`.
    public var store: Bool?
    public var metadata: [String: JSONAny]?
    /// Responses API structured-output config: `{"text":{"format":{"type":…,"schema":…}}}`.
    public var text: ResponsesTextConfig?
    /// Leniency: also accept a top-level chat-completions-style `response_format`.
    public var response_format: OpenAIResponseFormat?

    /// The structured-output request, preferring the Responses `text.format` shape and falling back
    /// to a top-level `response_format` for clients that send the chat-completions field instead.
    public var resolvedResponseFormat: OpenAIResponseFormat? {
        text?.format?.responseFormat ?? response_format
    }

    /// The input normalized to items (a bare string is one user message).
    public var items: [OpenAIResponsesInputItem] {
        switch input {
        case .none:
            return []
        case .text(let text):
            return [OpenAIResponsesInputItem(role: "user", segments: [.text(text)])]
        case .items(let items):
            return items
        }
    }

    /// Map onto the shared runtime contract, with the same defaults chat completions uses.
    /// `messages` are built by the handler (audio parts need the async Whisper splice first).
    func toGeneration(messages: [ChatMessage]) -> GenerationRequest {
        GenerationRequest(
            model: model,
            messages: messages,
            maxTokens: max_output_tokens ?? 512,
            temperature: temperature ?? 0.7,
            temperatureWasExplicit: temperature != nil,
            topP: top_p,
            stream: stream ?? false,
            responseFormat: resolvedResponseFormat)
    }
}

/// The Responses API `text` object; only its `format` is consumed here.
public struct ResponsesTextConfig: Decodable, Sendable {
    public var format: ResponsesFormatSpec?
}

/// The Responses API `text.format` object. Unlike chat completions (which nests the schema under
/// `json_schema`), the Responses shape is flat: `{"type":"json_schema","name":…,"strict":…,"schema":…}`.
/// Decoded here into the shared ``OpenAIResponseFormat``.
public struct ResponsesFormatSpec: Decodable, Sendable {
    public var responseFormat: OpenAIResponseFormat

    private enum CodingKeys: String, CodingKey { case type, name, description, strict, schema }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch (try c.decodeIfPresent(String.self, forKey: .type))?.lowercased() ?? "text" {
        case "json_schema":
            let name = try c.decodeIfPresent(String.self, forKey: .name) ?? "response"
            let description = try c.decodeIfPresent(String.self, forKey: .description)
            let strict = try c.decodeIfPresent(Bool.self, forKey: .strict)
            let schema =
                try c.decodeIfPresent(JSONAny.self, forKey: .schema)
                ?? .object(["type": .string("object")])
            responseFormat = .jsonSchema(
                OpenAIResponseFormat.JSONSchema(
                    name: name, description: description, schema: schema, strict: strict))
        case "json_object":
            responseFormat = .jsonObject
        default:
            responseFormat = .text
        }
    }
}

/// `input` is either one string (shorthand for a single user message) or an array of items.
public enum OpenAIResponsesInput: Decodable, Sendable, Equatable {
    case text(String)
    case items([OpenAIResponsesInputItem])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let text = try? c.decode(String.self) {
            self = .text(text)
        } else {
            self = .items(try c.decode([OpenAIResponsesInputItem].self))
        }
    }
}

/// One Responses input item. Only `{type:"message"}` items (and the OpenAI bare `{role, content}`
/// shorthand without `type`) are supported; content is kept as ordered text/audio segments so the
/// handler can splice audio transcripts in place.
public struct OpenAIResponsesInputItem: Decodable, Sendable, Equatable {
    public enum Segment: Sendable, Equatable {
        case text(String)
        /// A base64 `input_audio` payload (either `input_audio` or `audio` payload key).
        case audio(base64: String, format: String?)
        /// Anything we do not understand — surfaced as an explicit 400, never dropped silently.
        case unsupported(String)
    }

    public var type: String?
    public var role: String?
    public var segments: [Segment]

    init(type: String? = nil, role: String?, segments: [Segment]) {
        self.type = type
        self.role = role
        self.segments = segments
    }

    private enum CodingKeys: String, CodingKey { case type, role, content }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decodeIfPresent(String.self, forKey: .type)
        role = try c.decodeIfPresent(String.self, forKey: .role)
        if let text = try? c.decode(String.self, forKey: .content) {
            segments = [.text(text)]
        } else if let blocks = try? c.decode([JSONAny].self, forKey: .content) {
            segments = Self.parseParts(blocks)
        } else if !c.contains(.content) || (try? c.decodeNil(forKey: .content)) == true {
            segments = []
        } else {
            throw DecodingError.dataCorruptedError(
                forKey: .content, in: c,
                debugDescription: "content must be a string or an array of typed content parts")
        }
    }

    static func parseParts(_ blocks: [JSONAny]) -> [Segment] {
        blocks.map { block -> Segment in
            guard case .object(let object) = block else {
                return .unsupported("non-object content part")
            }
            let type = string(object["type"]) ?? ""
            switch type.lowercased() {
            case "input_text", "output_text", "text", "":
                guard let text = string(object["text"]) else {
                    return .unsupported(type.isEmpty ? "content part without text" : type)
                }
                return .text(text)
            case "input_audio", "audio":
                // OpenAI nests the payload under `input_audio`; also accept `audio` and a flat
                // `{data, format}` object for leniency.
                let payload: [String: JSONAny]
                if case .object(let nested)? = object["input_audio"] ?? object["audio"] {
                    payload = nested
                } else {
                    payload = object
                }
                guard let data = string(payload["data"]) else {
                    return .unsupported("input_audio part without base64 data")
                }
                return .audio(base64: data, format: string(payload["format"]))
            default:
                return .unsupported(type)
            }
        }
    }

    private static func string(_ value: JSONAny?) -> String? {
        guard case .string(let s) = value else { return nil }
        return s
    }
}

// MARK: - Response schema

public struct OpenAIResponsesResponse: Encodable, Sendable {
    public struct ContentPart: Encodable, Sendable {
        public var type: String = "output_text"
        public var text: String
        public var annotations: [JSONAny] = []

        public init(text: String) {
            self.text = text
        }
    }

    public struct OutputMessage: Encodable, Sendable {
        public var type: String = "message"
        public var id: String
        public var status: String
        public var role: String = "assistant"
        public var content: [ContentPart]

        public init(id: String, status: String, content: [ContentPart]) {
            self.id = id
            self.status = status
            self.content = content
        }
    }

    public struct IncompleteDetails: Encodable, Sendable {
        public var reason: String
    }

    public struct Usage: Encodable, Sendable {
        public var input_tokens: Int
        public var output_tokens: Int
        public var total_tokens: Int
    }

    public var id: String
    public var object: String = "response"
    public var created_at: Int
    public var status: String
    public var model: String
    public var output: [OutputMessage]
    /// `nil` while in progress (streaming snapshots); always present on the final object.
    public var usage: Usage?
    public var incomplete_details: IncompleteDetails?
    public var instructions: String?
    public var max_output_tokens: Int?
    public var temperature: Double
    public var top_p: Double
    public var store: Bool = false
    public var metadata: [String: JSONAny]
    public var parallel_tool_calls: Bool = true
    public var tool_choice: String = "auto"
    public var tools: [JSONAny] = []

    public init(
        id: String,
        createdAt: Int,
        status: String,
        model: String,
        output: [OutputMessage],
        usage: Usage?,
        incompleteDetails: IncompleteDetails? = nil,
        request: OpenAIResponsesRequest
    ) {
        self.id = id
        self.created_at = createdAt
        self.status = status
        self.model = model
        self.output = output
        self.usage = usage
        self.incomplete_details = incompleteDetails
        self.instructions = request.instructions
        self.max_output_tokens = request.max_output_tokens
        // The Responses API echoes sampling knobs, defaulting to OpenAI's documented 1.0.
        self.temperature = request.temperature ?? 1.0
        self.top_p = request.top_p ?? 1.0
        self.metadata = request.metadata ?? [:]
    }

    /// Map the runtime stop reason to the Responses API terminal status.
    static func completionStatus(
        for stopReason: CoreAIPipeline.StopReason
    ) -> (status: String, incompleteDetails: IncompleteDetails?) {
        switch stopReason {
        case .eos, .stopSequence:
            return ("completed", nil)
        case .maxTokens, .contextLimit:
            return ("incomplete", IncompleteDetails(reason: "max_output_tokens"))
        }
    }

    private enum CodingKeys: String, CodingKey {
        case id, object, created_at, status, model, output, usage, error,
             incomplete_details, instructions, max_output_tokens, temperature,
             top_p, store, metadata, parallel_tool_calls, tool_choice, tools
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(object, forKey: .object)
        try c.encode(created_at, forKey: .created_at)
        try c.encode(status, forKey: .status)
        try c.encode(model, forKey: .model)
        try c.encode(output, forKey: .output)
        // Responses API convention: absent values are explicit nulls, not omitted keys.
        // Request failures are returned as HTTP error responses, so `error` is always null here.
        if let usage { try c.encode(usage, forKey: .usage) } else { try c.encodeNil(forKey: .usage) }
        try c.encodeNil(forKey: .error)
        if let incomplete_details {
            try c.encode(incomplete_details, forKey: .incomplete_details)
        } else {
            try c.encodeNil(forKey: .incomplete_details)
        }
        if let instructions {
            try c.encode(instructions, forKey: .instructions)
        } else {
            try c.encodeNil(forKey: .instructions)
        }
        if let max_output_tokens {
            try c.encode(max_output_tokens, forKey: .max_output_tokens)
        } else {
            try c.encodeNil(forKey: .max_output_tokens)
        }
        try c.encode(temperature, forKey: .temperature)
        try c.encode(top_p, forKey: .top_p)
        try c.encode(store, forKey: .store)
        try c.encode(metadata, forKey: .metadata)
        try c.encode(parallel_tool_calls, forKey: .parallel_tool_calls)
        try c.encode(tool_choice, forKey: .tool_choice)
        try c.encode(tools, forKey: .tools)
    }
}

// MARK: - Streaming event frames

/// Builds the `event: <type>` + `data: <json>` SSE frames of the Responses API stream protocol,
/// stamping every payload with a monotonically increasing `sequence_number`. Pure (no HTTP
/// concerns) so tests can drive the exact event sequence without the runtime.
struct ResponsesStreamFramer {
    let responseID: String
    let itemID: String
    private(set) var sequence = 0

    init(responseID: String, itemID: String) {
        self.responseID = responseID
        self.itemID = itemID
    }

    struct ResponsePayload: Encodable {
        var type: String
        var response: OpenAIResponsesResponse
        var sequence_number: Int
    }
    struct ItemPayload: Encodable {
        var type: String
        var output_index: Int
        var item: OpenAIResponsesResponse.OutputMessage
        var sequence_number: Int
    }
    struct PartPayload: Encodable {
        var type: String
        var item_id: String
        var output_index: Int
        var content_index: Int
        var part: OpenAIResponsesResponse.ContentPart
        var sequence_number: Int
    }
    struct TextDeltaPayload: Encodable {
        var type: String
        var item_id: String
        var output_index: Int
        var content_index: Int
        var delta: String
        var sequence_number: Int
    }
    struct TextDonePayload: Encodable {
        var type: String
        var item_id: String
        var output_index: Int
        var content_index: Int
        var text: String
        var sequence_number: Int
    }

    private mutating func nextSequence() -> Int {
        defer { sequence += 1 }
        return sequence
    }

    private static func frame<T: Encodable>(_ event: String, _ payload: T) -> String {
        let json = (try? JSONEncoder().encode(payload))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return "event: \(event)\ndata: \(json)\n\n"
    }

    mutating func created(_ response: OpenAIResponsesResponse) -> String {
        Self.frame("response.created", ResponsePayload(
            type: "response.created", response: response, sequence_number: nextSequence()))
    }

    mutating func inProgress(_ response: OpenAIResponsesResponse) -> String {
        Self.frame("response.in_progress", ResponsePayload(
            type: "response.in_progress", response: response, sequence_number: nextSequence()))
    }

    mutating func outputItemAdded(_ item: OpenAIResponsesResponse.OutputMessage) -> String {
        Self.frame("response.output_item.added", ItemPayload(
            type: "response.output_item.added", output_index: 0, item: item,
            sequence_number: nextSequence()))
    }

    mutating func contentPartAdded() -> String {
        Self.frame("response.content_part.added", PartPayload(
            type: "response.content_part.added", item_id: itemID, output_index: 0,
            content_index: 0, part: OpenAIResponsesResponse.ContentPart(text: ""),
            sequence_number: nextSequence()))
    }

    mutating func outputTextDelta(_ delta: String) -> String {
        Self.frame("response.output_text.delta", TextDeltaPayload(
            type: "response.output_text.delta", item_id: itemID, output_index: 0,
            content_index: 0, delta: delta, sequence_number: nextSequence()))
    }

    mutating func outputTextDone(_ text: String) -> String {
        Self.frame("response.output_text.done", TextDonePayload(
            type: "response.output_text.done", item_id: itemID, output_index: 0,
            content_index: 0, text: text, sequence_number: nextSequence()))
    }

    mutating func contentPartDone(_ text: String) -> String {
        Self.frame("response.content_part.done", PartPayload(
            type: "response.content_part.done", item_id: itemID, output_index: 0,
            content_index: 0, part: OpenAIResponsesResponse.ContentPart(text: text),
            sequence_number: nextSequence()))
    }

    mutating func outputItemDone(_ item: OpenAIResponsesResponse.OutputMessage) -> String {
        Self.frame("response.output_item.done", ItemPayload(
            type: "response.output_item.done", output_index: 0, item: item,
            sequence_number: nextSequence()))
    }

    /// Terminal event: `response.completed`, `response.incomplete`, or `response.failed`,
    /// named after the final response's status and carrying the full final response object.
    mutating func finished(_ response: OpenAIResponsesResponse) -> String {
        let event = "response.\(response.status)"
        return Self.frame(event, ResponsePayload(
            type: event, response: response, sequence_number: nextSequence()))
    }
}

// MARK: - Handler

extension ServerRuntime {
    /// Errors surfaced while turning Responses API input into chat messages.
    enum ResponsesInputError: Error, Equatable {
        case emptyInput
        case unsupportedItemType(String)
        case unsupportedContentPart(String)
        case invalidAudioData
        case whisperUnavailable
        case audioRejected(String)
        case transcriptionFailed(String)
    }

    func openAIResponsesHandler(_ request: Request, _ context: BasicRequestContext) async throws -> Response {
        let started = Date()
        let path = "/v1/responses"
        if let response = await residentMemoryGuardResponse(
            method: "POST", path: path, startedAt: started)
        {
            return response
        }
        if let response = await conversionGuardResponse(
            method: "POST", path: path, startedAt: started)
        {
            return response
        }
        let req: OpenAIResponsesRequest
        do {
            req = try await Self.decode(OpenAIResponsesRequest.self, request)
        } catch {
            await activity.record(
                method: "POST", path: path, status: 400, startedAt: started,
                summary: "invalid OpenAI responses request")
            return JSONResponder.error("invalid OpenAI responses request: \(error)", status: .badRequest)
        }

        // Build chat messages, transcribing input_audio parts through the resident Whisper seam.
        let messages: [ChatMessage]
        do {
            messages = try await responsesMessages(from: req)
        } catch let error as ResponsesInputError {
            return await responsesInputErrorResponse(error, path: path, startedAt: started)
        }

        let gen = req.toGeneration(messages: messages)
        let modelName = await resolveModelName(gen.model)
        let handle: ModelHandle
        do {
            handle = try await manager.handle(for: modelName)
        } catch {
            await activity.record(
                method: "POST", path: path, status: 503, startedAt: started,
                model: modelName, summary: "model load failed: \(error)")
            return JSONResponder.error("could not load model '\(modelName)': \(error)", status: .serviceUnavailable)
        }

        // Honor a Responses `text.format` (or top-level `response_format`): true constrained decoding
        // when the backend supports it, otherwise best-effort server-side JSON coercion.
        let plan = Self.constraintPlan(
            for: gen,
            backendSupportsConstrainedDecoding:
                CoreAIPipeline.supportsConstrainedDecoding && handle.supportsConstrainedDecoding)
        if case .reject(let response) = plan {
            await activity.record(
                method: "POST", path: path, status: 400, startedAt: started,
                model: modelName, summary: "response_format unsupported by backend")
            return response
        }

        var payload = Self.messagePayload(gen.messages)
        var options = Self.options(from: gen)
        if case .coerce(let responseFormat) = plan {
            if let instruction = JSONCoercion.systemInstruction(for: responseFormat) {
                JSONCoercion.inject(instruction: instruction, into: &payload)
            }
            options.constrainedJSONSchema = nil
        }
        options.temperature = gen.resolvedTemperature(
            defaultingToGreedy: handle.defaultsToGreedyWhenTemperatureOmitted)
        options.verbose = verbose
        let format = await manager.outputFormat(for: modelName)
        let responseID = "resp_" + Self.shortID()
        let messageID = "msg_" + Self.shortID()
        let createdAt = Int(Date().timeIntervalSince1970)

        if gen.stream {
            return Self.openAIResponsesStream(
                handle: handle, messages: payload, options: options, format: format,
                request: req, model: modelName, responseID: responseID, messageID: messageID,
                createdAt: createdAt, activity: activity, startedAt: started)
        }

        do {
            let result = try await handle.generate(
                messages: payload,
                options: options)
            let completedAt = Date()
            // Same base-format normalization as chat completions; the Responses surface carries
            // only the final answer text as one output_text part.
            let norm = StreamingNormalizer.normalizeComplete(result.text, format: format)
            var outputText = norm.text
            if case .coerce(let responseFormat) = plan {
                outputText = await coerceJSON(
                    normalizedText: norm.text, responseFormat: responseFormat, handle: handle,
                    messages: payload, options: options, tools: nil,
                    additionalContext: nil, format: format, model: modelName)
            }
            let (status, incompleteDetails) = OpenAIResponsesResponse.completionStatus(
                for: result.stopReason)
            let response = OpenAIResponsesResponse(
                id: responseID,
                createdAt: createdAt,
                status: status,
                model: modelName,
                output: [OpenAIResponsesResponse.OutputMessage(
                    id: messageID,
                    status: status,
                    content: [OpenAIResponsesResponse.ContentPart(text: outputText)])],
                usage: OpenAIResponsesResponse.Usage(
                    input_tokens: result.promptTokenCount,
                    output_tokens: result.generatedTokenCount,
                    total_tokens: result.promptTokenCount + result.generatedTokenCount),
                incompleteDetails: incompleteDetails,
                request: req)
            await activity.record(
                method: "POST", path: path, status: 200, startedAt: started,
                model: modelName, summary: "completed (\(status))",
                inputTokens: result.promptTokenCount, outputTokens: result.generatedTokenCount,
                firstTokenSeconds: Self.nonStreamingFirstTokenSeconds(
                    startedAt: started, completedAt: completedAt, result: result),
                loadSeconds: result.modelLoadSeconds, prefillSeconds: result.prefillSeconds,
                decodeSeconds: result.decodeSeconds, prefixHitCount: result.prefixHitCount)
            return JSONResponder.encode(response)
        } catch {
            await activity.record(
                method: "POST", path: path, status: 500, startedAt: started,
                model: modelName, summary: "generation failed: \(error)")
            return JSONResponder.error("generation failed: \(error)", status: .internalServerError)
        }
    }

    /// Instructions become the (prepended) system message; each input item becomes one chat
    /// message with audio segments replaced by their Whisper transcript in place.
    /// Internal (not private) so request-mapping tests can exercise it without the runtime.
    func responsesMessages(from req: OpenAIResponsesRequest) async throws -> [ChatMessage] {
        var messages: [ChatMessage] = []
        if let instructions = req.instructions?.nonEmpty {
            messages.append(ChatMessage(role: "system", content: instructions))
        }
        for item in req.items {
            if let type = item.type?.lowercased(), type != "message" {
                throw ResponsesInputError.unsupportedItemType(type)
            }
            // OpenAI treats `developer` as the system-authority role.
            let rawRole = item.role ?? "user"
            let role = rawRole.lowercased() == "developer" ? "system" : rawRole
            var pieces: [String] = []
            for segment in item.segments {
                switch segment {
                case .text(let text):
                    pieces.append(text)
                case .audio(let base64, let format):
                    let transcript = try await transcribeResponsesAudio(base64: base64, format: format)
                    pieces.append("[audio transcript] \(transcript)")
                case .unsupported(let type):
                    throw ResponsesInputError.unsupportedContentPart(type)
                }
            }
            messages.append(ChatMessage(role: role, content: pieces.joined(separator: "\n")))
        }
        guard !messages.isEmpty else { throw ResponsesInputError.emptyInput }
        return messages
    }

    /// Route one base64 audio payload through the same validated in-memory Whisper service the
    /// `/v1/audio/transcriptions` endpoint uses (no shell-outs, no re-implemented decoding).
    private func transcribeResponsesAudio(base64: String, format: String?) async throws -> String {
        guard let audioTranscriptionService else { throw ResponsesInputError.whisperUnavailable }
        guard let data = Data(base64Encoded: base64, options: [.ignoreUnknownCharacters]),
              !data.isEmpty
        else {
            throw ResponsesInputError.invalidAudioData
        }
        let form = MultipartFormData(parts: [
            .init(name: "model", body: Data(OpenAIAudioAPI.modelID.utf8)),
            .init(
                name: "file",
                filename: "input." + (format?.nonEmpty ?? "wav"),
                contentType: nil,
                body: data),
        ])
        do {
            let transcriptionRequest = try OpenAIAudioTranscriptionRequest(form: form)
            let output = try await audioTranscriptionService.transcribe(transcriptionRequest)
            return output.text
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as OpenAIAudioTranscriptionService.ServiceError {
            throw ResponsesInputError.audioRejected(error.description)
        } catch let error as InMemoryAudioDecoder.DecodingError {
            throw ResponsesInputError.audioRejected(error.description)
        } catch let error as OpenAIAudioTranscriptionRequest.ValidationError {
            throw ResponsesInputError.audioRejected(error.description)
        } catch let error as WhisperDecodingPolicy.PolicyError {
            throw ResponsesInputError.audioRejected("Whisper request rejected: \(error)")
        } catch {
            throw ResponsesInputError.transcriptionFailed("\(error)")
        }
    }

    private func responsesInputErrorResponse(
        _ error: ResponsesInputError, path: String, startedAt: Date
    ) async -> Response {
        switch error {
        case .emptyInput:
            await activity.record(
                method: "POST", path: path, status: 400, startedAt: startedAt,
                summary: "responses request had no input")
            return JSONResponder.error("input (or instructions) is required", status: .badRequest)
        case .unsupportedItemType(let type):
            await activity.record(
                method: "POST", path: path, status: 400, startedAt: startedAt,
                summary: "unsupported responses input item type '\(type)'")
            return JSONResponder.error(
                "unsupported input item type '\(type)'; only 'message' items are supported",
                status: .badRequest)
        case .unsupportedContentPart(let type):
            await activity.record(
                method: "POST", path: path, status: 400, startedAt: startedAt,
                summary: "unsupported responses content part '\(type)'")
            return JSONResponder.error(
                "unsupported input content part '\(type)'; supported parts are input_text, output_text, and input_audio",
                status: .badRequest)
        case .invalidAudioData:
            await activity.record(
                method: "POST", path: path, status: 400, startedAt: startedAt,
                summary: "input_audio data was not valid base64")
            return JSONResponder.error(
                "input_audio data must be non-empty base64-encoded audio", status: .badRequest)
        case .whisperUnavailable:
            await activity.record(
                method: "POST", path: path, status: 503, startedAt: startedAt,
                model: OpenAIAudioAPI.modelID,
                summary: "native Whisper service unavailable")
            return JSONResponder.error(
                "native Whisper service is unavailable; start caix with an authenticated Whisper asset",
                status: .serviceUnavailable)
        case .audioRejected(let reason):
            await activity.record(
                method: "POST", path: path, status: 400, startedAt: startedAt,
                model: OpenAIAudioAPI.modelID,
                summary: "responses audio rejected: \(reason)")
            return JSONResponder.error(reason, status: .badRequest)
        case .transcriptionFailed(let reason):
            await activity.record(
                method: "POST", path: path, status: 500, startedAt: startedAt,
                model: OpenAIAudioAPI.modelID,
                summary: "responses transcription failed: \(reason)")
            return JSONResponder.error(
                "native Whisper transcription failed: \(reason)", status: .internalServerError)
        }
    }
}
