import Foundation

// Per-model normalization of a model's BASE output format (reasoning + tool calls emitted by
// the chat template) into the OpenAI / Anthropic vendor standards.
//
// Models stream raw token text. Depending on the family (detected from the bundle tokenizer /
// chat_template) that text interleaves:
//   - reasoning, between `<think>`/`</think>` (Qwen/ChatML) or a `<|channel>thought … <channel|>`
//     channel (Gemma 4);
//   - tool calls, as `<tool_call>{json}</tool_call>` (Qwen), `<tool_call><function=…>…</function>
//     </tool_call>` (Qwythos), or `<|tool_call|>{json}<tool_call|>` (Gemma 4);
//   - and final answer text.
//
// `StreamingNormalizer` is an incremental state machine: feed it decoded text deltas as they
// arrive (`push`) and it emits `reasoning` / `text` / `toolCall` events, correctly handling
// markers that straddle token boundaries (it holds back any suffix that could still become a
// marker). `finish()` flushes the tail. The same machine drives both the streaming SSE paths
// and the non-streaming aggregation (`normalizeComplete`).
//
// Pure Swift — no Core AI dependency, compiles in both the standalone and `COREAI_RUNTIME`
// builds. A model whose markers aren't recognised gets `OutputFormat.passthrough`, which emits
// every token as plain `text` (today's verbatim behaviour) so nothing regresses.

// MARK: - Detected per-model format

/// The base reasoning/tool-call markers for a model family, auto-detected from its bundle.
public struct OutputFormat: Sendable, Equatable {
    public enum Family: String, Sendable { case qwen, gemma, passthrough }

    public var family: Family
    /// Markers that open a reasoning span (e.g. `<think>`). Empty ⇒ no explicit reasoning open.
    public var reasoningStarts: [String]
    /// Markers that close a reasoning span (e.g. `</think>`, `<channel|>`).
    public var reasoningEnds: [String]
    /// Markers that open a tool-call block (e.g. `<tool_call>`, `<|tool_call|>`).
    public var toolStarts: [String]
    /// Markers that close a tool-call block (e.g. `</tool_call>`, `<tool_call|>`).
    public var toolEnds: [String]
    /// Scaffolding markers stripped wherever they appear in text/reasoning (e.g. stray channel
    /// or turn tokens). Consumed silently.
    public var dropMarkers: [String]
    /// The model's output begins *inside* reasoning because the generation prompt opened the
    /// reasoning span (e.g. Qwythos appends a dangling `<think>`); there is no opening marker in
    /// the stream, only a closing one.
    public var implicitReasoningStart: Bool

    public init(
        family: Family,
        reasoningStarts: [String] = [],
        reasoningEnds: [String] = [],
        toolStarts: [String] = [],
        toolEnds: [String] = [],
        dropMarkers: [String] = [],
        implicitReasoningStart: Bool = false
    ) {
        self.family = family
        self.reasoningStarts = reasoningStarts
        self.reasoningEnds = reasoningEnds
        self.toolStarts = toolStarts
        self.toolEnds = toolEnds
        self.dropMarkers = dropMarkers
        self.implicitReasoningStart = implicitReasoningStart
    }

    /// A safe default: every token passes through as plain text.
    public static let passthrough = OutputFormat(family: .passthrough)
}

// MARK: - Format detection

extension OutputFormat {
    /// Detect the output format for a model from its `tokenizer/` directory: the chat template
    /// (`chat_template.jinja` or the `chat_template` field of `tokenizer_config.json`) plus the
    /// declared special tokens. Recognises the Qwen/ChatML and Gemma 4 families by their marker
    /// tokens; anything else falls back to ``passthrough``.
    public static func detect(modelName: String, tokenizerDir: URL) -> OutputFormat {
        let template = loadChatTemplate(tokenizerDir: tokenizerDir)
        let specials = loadSpecialTokens(tokenizerDir: tokenizerDir)
        let haystack = template + "\n" + specials.joined(separator: "\n")

        func has(_ needle: String) -> Bool { haystack.contains(needle) }

        // Gemma 4: distinctive channel / turn / tool tokens. Some published templates also use
        // slash markers such as `//thought` / `//final`.
        let gemmaMarkers = ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|think|>", "<|turn>", "//thought"]
        let looksGemma = gemmaMarkers.contains(where: has)

        // Qwen / ChatML: `<think>` reasoning and/or `<tool_call>` JSON, ChatML turn tokens.
        let looksQwen = (has("<think>") || has("<tool_call>")) && (has("<|im_start|>") || has("</think>") || has("</tool_call>"))

        if looksQwen {
            return OutputFormat(
                family: .qwen,
                reasoningStarts: ["<think>"],
                reasoningEnds: ["</think>"],
                toolStarts: ["<tool_call>"],
                toolEnds: ["</tool_call>"],
                dropMarkers: [],
                implicitReasoningStart: detectImplicitReasoningStart(
                    template: template, opens: ["<think>"], closes: ["</think>"]))
        }

        if looksGemma {
            return OutputFormat(
                family: .gemma,
                // The thought channel carries reasoning; `<|think|>` is the thinking-enable token.
                reasoningStarts: ["<|channel>thought", "<|channel>analysis", "//thought", "<|think|>"],
                reasoningEnds: ["<channel|>", "//final"],
                // Reference markers: `<|tool_call|>{json}` (Gemma4PromptTemplate) and the raw
                // start/end tokens `<|tool_call>` / `<tool_call|>`.
                toolStarts: ["<|tool_call|>", "<|tool_call>"],
                toolEnds: ["<tool_call|>", "<|tool_call|>"],
                // Strip the answer channel scaffolding and stray turn tokens from final text.
                dropMarkers: [
                    "<|channel>final\n", "<|channel>final", "//final", "<turn|>", "<|turn>",
                    "<pad>", "<bos>", "<eos>", "</s>"
                ],
                implicitReasoningStart: detectImplicitReasoningStart(
                    template: template, opens: ["<|channel>thought", "<|channel>analysis", "//thought", "<|think|>"], closes: ["<channel|>", "//final"]))
        }

        return .passthrough
    }

    /// Decide whether the generation prompt leaves an *unclosed* reasoning span — i.e. the model
    /// starts generating already inside reasoning. Heuristic: within the generation-prompt tail
    /// (everything after the last `add_generation_prompt`), count reasoning-open vs
    /// reasoning-close literal occurrences; an excess of opens means the prompt dangles one open.
    static func detectImplicitReasoningStart(template: String, opens: [String], closes: [String]) -> Bool {
        guard !template.isEmpty,
            let r = template.range(of: "add_generation_prompt", options: .backwards)
        else { return false }
        let tail = String(template[r.upperBound...])
        let openCount = opens.reduce(0) { $0 + occurrences(of: $1, in: tail) }
        let closeCount = closes.reduce(0) { $0 + occurrences(of: $1, in: tail) }
        return openCount > closeCount
    }

    private static func occurrences(of needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var idx = haystack.startIndex
        while let r = haystack.range(of: needle, range: idx..<haystack.endIndex) {
            count += 1
            idx = r.upperBound
        }
        return count
    }

    private static func loadChatTemplate(tokenizerDir: URL) -> String {
        let jinja = tokenizerDir.appendingPathComponent("chat_template.jinja")
        if let t = try? String(contentsOf: jinja, encoding: .utf8), !t.isEmpty { return t }
        let cfg = tokenizerDir.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: cfg),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let ct = obj["chat_template"] as? String { return ct }
            // Some configs store chat_template as a list of {name, template} entries.
            if let arr = obj["chat_template"] as? [[String: Any]] {
                return arr.compactMap { $0["template"] as? String }.joined(separator: "\n")
            }
        }
        return ""
    }

    /// Collect declared special-token strings from the tokenizer dir so detection can key on
    /// tokens even when the chat template doesn't spell every marker out.
    private static func loadSpecialTokens(tokenizerDir: URL) -> [String] {
        var tokens: [String] = []
        let fm = FileManager.default

        let added = tokenizerDir.appendingPathComponent("added_tokens.json")
        if fm.fileExists(atPath: added.path), let data = try? Data(contentsOf: added),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            tokens.append(contentsOf: obj.keys)
        }

        let cfg = tokenizerDir.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: cfg),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Named *_token fields (Gemma stores its channel/tool/think markers here).
            for (key, value) in obj where key.hasSuffix("_token") {
                if let s = value as? String { tokens.append(s) }
                if let d = value as? [String: Any], let s = d["content"] as? String { tokens.append(s) }
            }
            if let decoder = obj["added_tokens_decoder"] as? [String: Any] {
                for (_, v) in decoder {
                    if let d = v as? [String: Any], let s = d["content"] as? String { tokens.append(s) }
                }
            }
        }
        return tokens
    }
}

// MARK: - Streaming state machine

/// Incremental normalizer. Created per response; driven by repeated ``push(_:)`` then ``finish()``.
/// Not `Sendable`: confined to a single task (the SSE writer closure or one non-streaming call).
public final class StreamingNormalizer {
    public enum Event: Sendable, Equatable {
        /// A delta of reasoning text (maps to OpenAI `reasoning_content` / Anthropic `thinking`).
        case reasoning(String)
        /// A delta of final answer text (maps to OpenAI `content` / Anthropic `text`).
        case text(String)
        /// A fully-parsed tool call (buffered until its block closes, then emitted whole).
        case toolCall(ToolCall)
    }

    public struct ToolCall: Sendable, Equatable {
        public var index: Int
        public var id: String
        public var name: String
        /// Arguments as a compact JSON object string (OpenAI `function.arguments`).
        public var arguments: String
    }

    private enum Kind { case reasoningStart, reasoningEnd, toolStart, drop }
    private enum State { case text, reasoning, tool }

    private let format: OutputFormat
    private var state: State

    private var buffer: [Character] = []
    private var toolBuffer: [Character] = []
    private var toolIndex = 0
    private var emittedReasoning = false
    private var emittedText = false

    // Pre-tokenised watch lists (markers as Character arrays, longest first).
    private let textWatch: [(Kind, [Character])]
    private let reasoningWatch: [(Kind, [Character])]
    private let toolEndChars: [[Character]]

    public init(format: OutputFormat) {
        self.format = format
        self.state = format.implicitReasoningStart ? .reasoning : .text

        func chars(_ ss: [String]) -> [[Character]] { ss.map(Array.init) }
        func labeled(_ pairs: [(Kind, [String])]) -> [(Kind, [Character])] {
            var out: [(Kind, [Character])] = []
            for (kind, strings) in pairs {
                for s in strings where !s.isEmpty { out.append((kind, Array(s))) }
            }
            // Longest markers first so e.g. `<|channel>thought` wins over a `<|channel>` prefix.
            return out.sorted { $0.1.count > $1.1.count }
        }

        self.textWatch = labeled([
            (.reasoningStart, format.reasoningStarts),
            (.toolStart, format.toolStarts),
            (.reasoningEnd, format.reasoningEnds),  // stray close in text ⇒ drop it
            (.drop, format.dropMarkers),
        ])
        self.reasoningWatch = labeled([
            (.reasoningEnd, format.reasoningEnds),
            (.toolStart, format.toolStarts),
            (.drop, format.dropMarkers),
        ])
        self.toolEndChars = chars(format.toolEnds).sorted { $0.count > $1.count }
    }

    /// Feed a decoded text delta; returns any events that became complete.
    public func push(_ delta: String) -> [Event] {
        guard !delta.isEmpty else { return [] }
        buffer.append(contentsOf: delta)
        return drain(final: false)
    }

    /// Flush remaining buffered text at end of generation.
    public func finish() -> [Event] {
        var events = drain(final: true)
        if !buffer.isEmpty {
            let s = String(buffer)
            buffer.removeAll(keepingCapacity: false)
            switch state {
            case .reasoning: appendEmit(s, reasoning: true, into: &events)
            case .text: appendEmit(s, reasoning: false, into: &events)
            case .tool: toolBuffer.append(contentsOf: s)
            }
        }
        if state == .tool, !toolBuffer.isEmpty {
            if let tc = parseToolBlock(String(toolBuffer)) { events.append(.toolCall(tc)) }
            toolBuffer.removeAll(keepingCapacity: false)
            state = .text
        }
        return events
    }

    // MARK: Core drain loop

    private func drain(final: Bool) -> [Event] {
        var events: [Event] = []
        loop: while true {
            switch state {
            case .text, .reasoning:
                let watch = (state == .reasoning) ? reasoningWatch : textWatch
                if let (kind, range) = firstMatch(in: buffer, watch) {
                    let before = String(buffer[0..<range.lowerBound])
                    appendEmit(before, reasoning: state == .reasoning, into: &events)
                    buffer.removeSubrange(0..<range.upperBound)
                    apply(kind)
                    continue loop
                }
                // No complete marker: emit everything that can't be part of one; hold the rest.
                let hold = final ? 0 : longestPartialSuffix(buffer, watch.map { $0.1 })
                let cut = buffer.count - hold
                if cut > 0 {
                    appendEmit(String(buffer[0..<cut]), reasoning: state == .reasoning, into: &events)
                    buffer.removeFirst(cut)
                }
                break loop

            case .tool:
                if let range = firstMatchPlain(in: buffer, toolEndChars) {
                    toolBuffer.append(contentsOf: buffer[0..<range.lowerBound])
                    buffer.removeSubrange(0..<range.upperBound)
                    if let tc = parseToolBlock(String(toolBuffer)) { events.append(.toolCall(tc)) }
                    toolBuffer.removeAll(keepingCapacity: true)
                    state = .text
                    continue loop
                }
                let hold = final ? 0 : longestPartialSuffix(buffer, toolEndChars)
                let cut = buffer.count - hold
                if cut > 0 {
                    toolBuffer.append(contentsOf: buffer[0..<cut])
                    buffer.removeFirst(cut)
                }
                break loop
            }
        }
        return events
    }

    private func apply(_ kind: Kind) {
        switch kind {
        case .reasoningStart: state = .reasoning
        case .reasoningEnd: state = .text
        case .toolStart:
            state = .tool
            toolBuffer.removeAll(keepingCapacity: true)
        case .drop:
            break  // marker already consumed; stay in the current state
        }
    }

    /// Emit `s` as a reasoning or text delta, left-trimming whitespace on the first emission of
    /// each channel so the `</think>\n\n` gap doesn't leak into the answer.
    private func appendEmit(_ s: String, reasoning: Bool, into events: inout [Event]) {
        guard !s.isEmpty else { return }
        if reasoning {
            var out = s
            if !emittedReasoning {
                out = String(out.drop { $0 == "\n" || $0 == "\r" || $0 == " " || $0 == "\t" })
                if out.isEmpty { return }
                emittedReasoning = true
            }
            events.append(.reasoning(out))
        } else {
            var out = s
            if !emittedText {
                out = String(out.drop { $0 == "\n" || $0 == "\r" || $0 == " " || $0 == "\t" })
                if out.isEmpty { return }
                emittedText = true
            }
            events.append(.text(out))
        }
    }

    // MARK: Marker matching helpers

    private func matchAt(_ buf: [Character], _ pos: Int, _ marker: [Character]) -> Bool {
        guard pos + marker.count <= buf.count else { return false }
        var i = 0
        while i < marker.count {
            if buf[pos + i] != marker[i] { return false }
            i += 1
        }
        return true
    }

    /// Earliest position at which any labelled marker fully matches (longest wins at a tie).
    private func firstMatch(in buf: [Character], _ watch: [(Kind, [Character])]) -> (Kind, Range<Int>)? {
        guard !buf.isEmpty else { return nil }
        for pos in 0..<buf.count {
            for (kind, marker) in watch where matchAt(buf, pos, marker) {
                return (kind, pos..<(pos + marker.count))
            }
        }
        return nil
    }

    private func firstMatchPlain(in buf: [Character], _ markers: [[Character]]) -> Range<Int>? {
        guard !buf.isEmpty else { return nil }
        for pos in 0..<buf.count {
            for marker in markers where matchAt(buf, pos, marker) {
                return pos..<(pos + marker.count)
            }
        }
        return nil
    }

    /// Length of the longest suffix of `buf` that is a *proper prefix* of some marker — i.e. the
    /// tail we must hold back because it could still grow into a complete marker.
    private func longestPartialSuffix(_ buf: [Character], _ markers: [[Character]]) -> Int {
        let maxLen = markers.map(\.count).max() ?? 0
        guard maxLen > 1, !buf.isEmpty else { return 0 }
        let upper = min(maxLen - 1, buf.count)
        var len = upper
        while len >= 1 {
            let suffix = Array(buf.suffix(len))
            for marker in markers where marker.count > len {
                if Array(marker.prefix(len)) == suffix { return len }
            }
            len -= 1
        }
        return 0
    }

    // MARK: Tool-call block parsing

    /// Parse a captured tool-call block (the text between the start/end markers) into a
    /// ``ToolCall``. Handles JSON (`{"name":…,"arguments":{…}}`) and the Qwen/Hermes XML form
    /// (`<function=name><parameter=k>v</parameter></function>`).
    private func parseToolBlock(_ raw: String) -> ToolCall? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.hasPrefix("{"), let tc = parseJSONTool(trimmed) { return tc }
        if trimmed.contains("<function="), let tc = parseXMLTool(trimmed) { return tc }
        // Fallback: locate a JSON object embedded anywhere in the block.
        if let brace = trimmed.firstIndex(of: "{"), let tc = parseJSONTool(String(trimmed[brace...])) {
            return tc
        }
        return nil
    }

    private func parseJSONTool(_ s: String) -> ToolCall? {
        guard let data = s.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let name = obj["name"] as? String
        else { return nil }
        let argsObj = obj["arguments"] ?? obj["parameters"] ?? [String: Any]()
        return makeToolCall(name: name, arguments: Self.compactJSON(argsObj))
    }

    private func parseXMLTool(_ s: String) -> ToolCall? {
        guard let fn = s.range(of: "<function=") else { return nil }
        let afterFn = s[fn.upperBound...]
        guard let gt = afterFn.firstIndex(of: ">") else { return nil }
        let name = String(afterFn[afterFn.startIndex..<gt]).trimmingCharacters(in: .whitespaces)
        var args: [String: Any] = [:]
        var rest = afterFn[afterFn.index(after: gt)...]
        while let p = rest.range(of: "<parameter=") {
            let afterP = rest[p.upperBound...]
            guard let pgt = afterP.firstIndex(of: ">") else { break }
            let key = String(afterP[afterP.startIndex..<pgt]).trimmingCharacters(in: .whitespaces)
            let valStart = afterP.index(after: pgt)
            guard let pEnd = afterP.range(of: "</parameter>", range: valStart..<afterP.endIndex) else { break }
            let value = String(afterP[valStart..<pEnd.lowerBound]).trimmingCharacters(in: .newlines)
            args[key] = Self.coerce(value)
            rest = afterP[pEnd.upperBound...]
        }
        return makeToolCall(name: name, arguments: Self.compactJSON(args))
    }

    private func makeToolCall(name: String, arguments: String) -> ToolCall {
        let idx = toolIndex
        toolIndex += 1
        return ToolCall(index: idx, id: "call_\(idx)", name: name, arguments: arguments)
    }

    /// Compact JSON string for an arbitrary JSON-compatible value; `{}` on failure.
    static func compactJSON(_ value: Any) -> String {
        if let s = value as? String, s.hasPrefix("{") || s.hasPrefix("[") { return s }
        guard JSONSerialization.isValidJSONObject(value),
            let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
            let s = String(data: data, encoding: .utf8)
        else {
            // Wrap a scalar so we still produce a JSON object string.
            if let s = value as? String, let d = try? JSONSerialization.data(withJSONObject: ["value": s]),
                let str = String(data: d, encoding: .utf8) { return str }
            return "{}"
        }
        return s
    }

    /// Coerce an XML parameter value into a typed JSON value (bool / int / double / json) or
    /// leave it as a string.
    static func coerce(_ value: String) -> Any {
        let v = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if v == "true" { return true }
        if v == "false" { return false }
        if let i = Int(v) { return i }
        if let d = Double(v) { return d }
        if v.hasPrefix("{") || v.hasPrefix("[") {
            if let data = v.data(using: .utf8), let obj = try? JSONSerialization.jsonObject(with: data) {
                return obj
            }
        }
        return value
    }
}

// MARK: - Whole-response aggregation (non-streaming)

/// The normalized split of a whole (non-streaming) generation.
public struct NormalizedResult: Sendable {
    public var reasoning: String
    public var text: String
    public var toolCalls: [StreamingNormalizer.ToolCall]
    public var hasToolCalls: Bool { !toolCalls.isEmpty }
}

extension StreamingNormalizer {
    /// Run the state machine over a complete output string and aggregate the events.
    public static func normalizeComplete(_ raw: String, format: OutputFormat) -> NormalizedResult {
        let n = StreamingNormalizer(format: format)
        var events = n.push(raw)
        events.append(contentsOf: n.finish())
        var reasoning = ""
        var text = ""
        var calls: [ToolCall] = []
        for event in events {
            switch event {
            case .reasoning(let s): reasoning += s
            case .text(let s): text += s
            case .toolCall(let t): calls.append(t)
            }
        }
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return NormalizedResult(
            reasoning: reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            text: collapseDuplicatedInitialWord(trimmedText),
            toolCalls: calls)
    }

    /// Some tiny random/coreai test exports can duplicate the first lexical token in decoded
    /// final text (`HelloHello!`, `TheThe ...`). Keep this intentionally narrow: only collapse a
    /// contiguous, title-cased word repeated at byte zero and followed by a non-word boundary.
    public static func collapseDuplicatedInitialWord(_ text: String) -> String {
        guard text.count >= 6, let first = text.unicodeScalars.first, CharacterSet.uppercaseLetters.contains(first) else {
            return text
        }
        let scalars = Array(text.unicodeScalars)
        let letters = CharacterSet.letters
        let wordChars = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
        let maxWordLength = min(24, scalars.count / 2)
        guard maxWordLength >= 3 else { return text }
        for length in 3...maxWordLength {
            guard scalars[0..<length].allSatisfy({ letters.contains($0) }) else {
                return text
            }
            let firstWord = Array(scalars[0..<length])
            let secondWord = Array(scalars[length..<(length * 2)])
            guard firstWord == secondWord else { continue }
            if scalars.count > length * 2, wordChars.contains(scalars[length * 2]) {
                continue
            }
            let prefix = String(String.UnicodeScalarView(firstWord))
            let suffix = String(String.UnicodeScalarView(scalars[(length * 2)..<scalars.count]))
            return prefix + suffix
        }
        return text
    }
}

// MARK: - Generic JSON value (tool specs in, tool_use input out)

/// A decoded JSON value used to (a) carry request `tools` to the tokenizer's chat template and
/// (b) re-encode tool-call arguments as a nested JSON object for Anthropic `tool_use.input`.
public enum JSONAny: Codable, Sendable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case array([JSONAny])
    case object([String: JSONAny])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let i = try? c.decode(Int.self) { self = .int(i); return }
        if let d = try? c.decode(Double.self) { self = .double(d); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let a = try? c.decode([JSONAny].self) { self = .array(a); return }
        if let o = try? c.decode([String: JSONAny].self) { self = .object(o); return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "unsupported JSON value")
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    /// Recursively project to `Sendable` values for the swift-transformers `ToolSpec`
    /// (`[String: any Sendable]`) consumed by the chat template.
    public var sendable: any Sendable {
        switch self {
        case .null: return ""
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let a): return a.map { $0.sendable }
        case .object(let o): return o.mapValues { $0.sendable }
        }
    }

    /// As a `[String: any Sendable]` tool-spec dictionary, when this value is an object.
    public var toolSpec: [String: any Sendable]? {
        if case .object(let o) = self { return o.mapValues { $0.sendable } }
        return nil
    }

    /// Parse a JSON string into a `JSONAny` (used to turn `function.arguments` back into an
    /// object for Anthropic `tool_use.input`).
    public static func parse(_ jsonString: String) -> JSONAny {
        guard let data = jsonString.data(using: .utf8),
            let value = try? JSONDecoder().decode(JSONAny.self, from: data)
        else { return .object([:]) }
        return value
    }
}
