import Foundation

// OpenAI- and Anthropic-compatible API schemas + mapping to an internal generation request.
// Pure Foundation; the HTTP layer (Hummingbird) and ModelManager wiring are added once the
// CoreAIRuntime target lands. These map both vendor formats onto one runtime contract.

// MARK: - Internal contract (what the runtime consumes / produces)

public struct ChatMessage: Codable, Sendable {
    public var role: String          // system | user | assistant | tool
    public var content: String
    public var media: [MediaPart]

    public init(role: String, content: String, media: [MediaPart] = []) {
        self.role = role
        self.content = content
        self.media = media
    }

    private enum CodingKeys: String, CodingKey { case role, content }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.role = try c.decode(String.self, forKey: .role)
        let parsed = try ContentParts.decode(from: c, forKey: .content)
        self.content = parsed.text
        self.media = parsed.media
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(role, forKey: .role)
        try c.encode(content, forKey: .content)
    }
}

public struct MediaPart: Codable, Sendable, Equatable {
    public var type: String
    public var payload: JSONAny

    public init(type: String, payload: JSONAny) {
        self.type = type
        self.payload = payload
    }

    public var modality: String {
        let lower = type.lowercased()
        if lower.contains("image") { return "image" }
        if lower.contains("audio") { return "audio" }
        if lower.contains("video") { return "video" }
        if lower.contains("document") || lower.contains("file") { return "file" }
        return lower.isEmpty ? "unknown" : lower
    }
}

public struct GenerationRequest: Sendable {
    public var model: String
    public var messages: [ChatMessage]
    public var maxTokens: Int
    public var temperature: Double
    public var topP: Double?
    public var topK: Int?
    public var stop: [String]
    public var stream: Bool
    public var applyChatTemplate: Bool
    public var kvCapacity: Int?
    /// Optional RNG seed for reproducible temperature sampling.
    public var seed: UInt64?
    /// Tool/function specs (already normalized to OpenAI `{type:"function", function:{…}}` form)
    /// to render into the model's chat template. Empty/nil ⇒ no tools.
    public var tools: [JSONAny]?
    public init(model: String, messages: [ChatMessage], maxTokens: Int = 512, temperature: Double = 0.7,
                topP: Double? = nil, topK: Int? = nil, stop: [String] = [], stream: Bool = false,
                applyChatTemplate: Bool = true, kvCapacity: Int? = nil, seed: UInt64? = nil,
                tools: [JSONAny]? = nil) {
        self.model = model; self.messages = messages; self.maxTokens = maxTokens
        self.temperature = temperature; self.topP = topP; self.topK = topK; self.stop = stop
        self.stream = stream; self.applyChatTemplate = applyChatTemplate; self.kvCapacity = kvCapacity
        self.seed = seed; self.tools = tools
    }

    public var media: [MediaPart] { messages.flatMap(\.media) }
    public var hasMultimodalContent: Bool { !media.isEmpty }
    public var modalities: [String] {
        Array(Set(media.map(\.modality))).sorted()
    }

    /// Tool specs projected to swift-transformers `ToolSpec` (`[String: any Sendable]`).
    public var toolSpecs: [[String: any Sendable]]? {
        guard let tools, !tools.isEmpty else { return nil }
        let specs = tools.compactMap { $0.toolSpec }
        return specs.isEmpty ? nil : specs
    }
}

// MARK: - OpenAI: /v1/chat/completions

public struct OpenAIChatRequest: Codable, Sendable {
    public var model: String
    public var messages: [ChatMessage]
    public var max_tokens: Int?
    public var max_completion_tokens: Int?
    public var temperature: Double?
    public var top_k: Int?
    public var top_p: Double?
    public var kv_capacity: Int?
    public var apply_chat_template: Bool?
    public var stop: StringOrArray?
    public var stream: Bool?
    public var seed: Int?
    /// Function/tool definitions (`[{type:"function", function:{name, description, parameters}}]`).
    public var tools: [JSONAny]?
    public func toGeneration() -> GenerationRequest {
        GenerationRequest(model: model, messages: messages, maxTokens: max_tokens ?? max_completion_tokens ?? 512,
                          temperature: temperature ?? 0.7, topP: top_p, topK: top_k,
                          stop: stop?.values ?? [], stream: stream ?? false,
                          applyChatTemplate: apply_chat_template ?? true, kvCapacity: kv_capacity,
                          seed: seed.map { UInt64(bitPattern: Int64($0)) }, tools: tools)
    }
}

/// One tool call in an OpenAI response (`function.arguments` is a JSON *string*).
public struct OpenAIToolCall: Encodable, Sendable {
    public struct Function: Encodable, Sendable { public var name: String; public var arguments: String }
    public var id: String
    public var type: String = "function"
    public var function: Function
    public init(id: String, name: String, arguments: String) {
        self.id = id; self.function = Function(name: name, arguments: arguments)
    }
}

/// Assistant message in a (non-streaming) OpenAI response: final `content`, plus the normalized
/// `reasoning_content` and `tool_calls` when present.
public struct OpenAIResponseMessage: Encodable, Sendable {
    public var role: String = "assistant"
    public var content: String?
    public var reasoning_content: String?
    public var tool_calls: [OpenAIToolCall]?
}

public struct OpenAIChatResponse: Encodable, Sendable {
    public struct Choice: Encodable, Sendable { public var index: Int; public var message: OpenAIResponseMessage; public var finish_reason: String }
    public struct Usage: Encodable, Sendable { public var prompt_tokens: Int; public var completion_tokens: Int; public var total_tokens: Int }
    public var id: String; public var object: String; public var created: Int; public var model: String
    public var choices: [Choice]; public var usage: Usage
    public init(id: String, model: String, created: Int, message: OpenAIResponseMessage, finish: String,
                promptTokens: Int, completionTokens: Int) {
        self.id = id; self.object = "chat.completion"; self.created = created; self.model = model
        self.choices = [Choice(index: 0, message: message, finish_reason: finish)]
        self.usage = Usage(prompt_tokens: promptTokens, completion_tokens: completionTokens,
                           total_tokens: promptTokens + completionTokens)
    }
}

/// One SSE `data:` chunk for OpenAI streaming. The delta carries at most one of
/// `content` / `reasoning_content` / `tool_calls` per chunk.
public struct OpenAIChatChunk: Encodable, Sendable {
    public struct ToolCallDelta: Encodable, Sendable {
        public struct Function: Encodable, Sendable { public var name: String?; public var arguments: String? }
        public var index: Int
        public var id: String?
        public var type: String?
        public var function: Function?
    }
    public struct Delta: Encodable, Sendable {
        public var role: String?
        public var content: String?
        public var reasoning_content: String?
        public var tool_calls: [ToolCallDelta]?
    }
    public struct Choice: Encodable, Sendable {
        public var index: Int; public var delta: Delta; public var finish_reason: String?
        private enum CodingKeys: String, CodingKey { case index, delta, finish_reason }
        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(index, forKey: .index)
            try c.encode(delta, forKey: .delta)
            // OpenAI streaming convention: finish_reason is present (null) on every chunk.
            try c.encode(finish_reason, forKey: .finish_reason)
        }
    }
    public var id: String; public var object: String; public var created: Int; public var model: String; public var choices: [Choice]
    public init(id: String, model: String, created: Int, role: String? = nil, content: String? = nil,
                reasoningContent: String? = nil, toolCalls: [ToolCallDelta]? = nil, finish: String? = nil) {
        self.id = id; self.object = "chat.completion.chunk"; self.created = created; self.model = model
        self.choices = [Choice(
            index: 0,
            delta: Delta(role: role, content: content, reasoning_content: reasoningContent, tool_calls: toolCalls),
            finish_reason: finish)]
    }
}

// MARK: - Anthropic: /v1/messages

public struct AnthropicMessagesRequest: Codable, Sendable {
    public struct Msg: Codable, Sendable { public var role: String; public var content: AnthropicContent }
    public var model: String
    public var messages: [Msg]
    public var system: String?
    public var max_tokens: Int
    public var temperature: Double?
    public var top_p: Double?
    public var top_k: Int?
    public var stop_sequences: [String]?
    public var stream: Bool?
    public var kv_capacity: Int?
    public var apply_chat_template: Bool?
    /// Non-standard reproducibility seed accepted for local testing.
    public var seed: Int?
    /// Anthropic tool defs (`[{name, description, input_schema}]`).
    public var tools: [JSONAny]?
    public func toGeneration() -> GenerationRequest {
        var msgs: [ChatMessage] = []
        if let system { msgs.append(ChatMessage(role: "system", content: system)) }
        for m in messages {
            msgs.append(ChatMessage(role: m.role, content: m.content.text, media: m.content.media))
        }
        return GenerationRequest(model: model, messages: msgs, maxTokens: max_tokens, temperature: temperature ?? 1.0,
                                 topP: top_p, topK: top_k, stop: stop_sequences ?? [], stream: stream ?? false,
                                 applyChatTemplate: apply_chat_template ?? true, kvCapacity: kv_capacity,
                                 seed: seed.map { UInt64(bitPattern: Int64($0)) }, tools: Self.normalizeTools(tools))
    }

    /// Convert Anthropic tool defs (`{name, description, input_schema}`) into the OpenAI
    /// `{type:"function", function:{name, description, parameters}}` form the chat templates
    /// expect, so the model is prompted consistently regardless of the request vendor.
    static func normalizeTools(_ tools: [JSONAny]?) -> [JSONAny]? {
        guard let tools, !tools.isEmpty else { return nil }
        let converted: [JSONAny] = tools.map { tool in
            guard case .object(let o) = tool else { return tool }
            // Already in OpenAI function form.
            if o["type"] != nil, o["function"] != nil { return tool }
            var function: [String: JSONAny] = [:]
            if let name = o["name"] { function["name"] = name }
            if let desc = o["description"] { function["description"] = desc }
            if let schema = o["input_schema"] ?? o["parameters"] { function["parameters"] = schema }
            return .object(["type": .string("function"), "function": .object(function)])
        }
        return converted
    }
}

/// One content block in an Anthropic response: thinking, text, or tool_use (with a JSON-object
/// `input`). Encoded into the vendor-standard `{type:…}` shapes.
public enum AnthropicBlock: Encodable, Sendable {
    case thinking(String)
    case text(String)
    case toolUse(id: String, name: String, input: JSONAny)

    private enum CodingKeys: String, CodingKey {
        case type, thinking, signature, text, id, name, input
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .thinking(let t):
            try c.encode("thinking", forKey: .type)
            try c.encode(t, forKey: .thinking)
            try c.encode("", forKey: .signature)
        case .text(let t):
            try c.encode("text", forKey: .type)
            try c.encode(t, forKey: .text)
        case .toolUse(let id, let name, let input):
            try c.encode("tool_use", forKey: .type)
            try c.encode(id, forKey: .id)
            try c.encode(name, forKey: .name)
            try c.encode(input, forKey: .input)
        }
    }
}

public struct AnthropicMessagesResponse: Encodable, Sendable {
    public struct Usage: Encodable, Sendable { public var input_tokens: Int; public var output_tokens: Int }
    public var id: String; public var type: String; public var role: String; public var model: String
    public var content: [AnthropicBlock]; public var stop_reason: String; public var stop_sequence: String?
    public var usage: Usage
    public init(id: String, model: String, blocks: [AnthropicBlock], stopReason: String, inputTokens: Int, outputTokens: Int) {
        self.id = id; self.type = "message"; self.role = "assistant"; self.model = model
        self.content = blocks; self.stop_reason = stopReason; self.stop_sequence = nil
        self.usage = Usage(input_tokens: inputTokens, output_tokens: outputTokens)
    }
}

// MARK: - Helpers for the loose typing in both schemas

/// `stop` may be a string or array of strings (OpenAI).
public enum StringOrArray: Codable, Sendable {
    case one(String), many([String])
    public var values: [String] { switch self { case .one(let s): return [s]; case .many(let a): return a } }
    public init(from d: Decoder) throws {
        let c = try d.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .one(s) } else { self = .many(try c.decode([String].self)) }
    }
    public func encode(to e: Encoder) throws {
        var c = e.singleValueContainer()
        switch self { case .one(let s): try c.encode(s); case .many(let a): try c.encode(a) }
    }
}

/// Anthropic content is either a string or an array of typed blocks. Text goes to the current
/// text-only runtime path; non-text blocks are preserved so the server can reject them explicitly.
public enum AnthropicContent: Codable, Sendable {
    case text(String), blocks([JSONAny])
    public var text: String {
        switch self {
        case .text(let s): return s
        case .blocks(let blocks): return ContentParts.parse(blocks).text
        }
    }
    public var media: [MediaPart] {
        switch self {
        case .text: return []
        case .blocks(let blocks): return ContentParts.parse(blocks).media
        }
    }
    public init(from d: Decoder) throws {
        let c = try d.singleValueContainer()
        if let s = try? c.decode(String.self) {
            self = .text(s)
        } else {
            self = .blocks(try c.decode([JSONAny].self))
        }
    }
    public func encode(to e: Encoder) throws {
        var c = e.singleValueContainer()
        switch self { case .text(let s): try c.encode(s); case .blocks(let b): try c.encode(b) }
    }
}

struct ContentParts {
    var text: String
    var media: [MediaPart]

    static func decode<K: CodingKey>(
        from container: KeyedDecodingContainer<K>,
        forKey key: K
    ) throws -> ContentParts {
        if let text = try? container.decode(String.self, forKey: key) {
            return ContentParts(text: text, media: [])
        }
        if let blocks = try? container.decode([JSONAny].self, forKey: key) {
            return parse(blocks)
        }
        if (try? container.decodeNil(forKey: key)) == true {
            return ContentParts(text: "", media: [])
        }
        throw DecodingError.dataCorruptedError(
            forKey: key, in: container,
            debugDescription: "content must be a string or an array of typed content blocks")
    }

    static func parse(_ blocks: [JSONAny]) -> ContentParts {
        var texts: [String] = []
        var media: [MediaPart] = []
        for block in blocks {
            guard case .object(let object) = block else { continue }
            let type = stringValue(object["type"]) ?? ""
            if isTextBlock(type), let text = textValue(object) {
                texts.append(text)
            } else {
                media.append(MediaPart(type: type, payload: block))
            }
        }
        return ContentParts(text: texts.joined(separator: "\n"), media: media)
    }

    private static func isTextBlock(_ type: String) -> Bool {
        let lower = type.lowercased()
        return lower == "text" || lower == "input_text"
    }

    private static func textValue(_ object: [String: JSONAny]) -> String? {
        stringValue(object["text"]) ?? stringValue(object["content"])
    }

    private static func stringValue(_ value: JSONAny?) -> String? {
        guard case .string(let s) = value else { return nil }
        return s
    }
}
