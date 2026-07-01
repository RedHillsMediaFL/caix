import XCTest

@testable import CoreAIServer

/// Unit tests for the per-model reasoning / tool-call normalizer. These exercise the parser in
/// isolation (no Core AI runtime needed) over canned base-format strings, including the
/// streaming case where markers straddle token boundaries.
final class OutputNormalizerTests: XCTestCase {
    // Qwen / ChatML: `<think>…</think>` reasoning, `<tool_call>{json}</tool_call>` tools.
    private let qwen = OutputFormat(
        family: .qwen,
        reasoningStarts: ["<think>"], reasoningEnds: ["</think>"],
        toolStarts: ["<tool_call>"], toolEnds: ["</tool_call>"])

    // MARK: Whole-string (non-streaming) normalization

    func testQwenReasoningThenText() {
        let raw = "<think>\nThe user wants a sum. 17+25=42.\n</think>\n\nThe answer is 42."
        let r = StreamingNormalizer.normalizeComplete(raw, format: qwen)
        XCTAssertEqual(r.reasoning, "The user wants a sum. 17+25=42.")
        XCTAssertEqual(r.text, "The answer is 42.")
        XCTAssertTrue(r.toolCalls.isEmpty)
    }

    func testQwenJSONToolCall() {
        let raw = """
        <think>I should call the tool.</think>
        <tool_call>
        {"name": "get_weather", "arguments": {"city": "Paris"}}
        </tool_call>
        """
        let r = StreamingNormalizer.normalizeComplete(raw, format: qwen)
        XCTAssertEqual(r.reasoning, "I should call the tool.")
        XCTAssertEqual(r.toolCalls.count, 1)
        XCTAssertEqual(r.toolCalls[0].id, "call_0")
        XCTAssertEqual(r.toolCalls[0].name, "get_weather")
        XCTAssertEqual(r.toolCalls[0].arguments, #"{"city":"Paris"}"#)
        XCTAssertTrue(r.hasToolCalls)
    }

    func testQwenMultipleToolCalls() {
        let raw = """
        <tool_call>
        {"name": "a", "arguments": {"x": 1}}
        </tool_call>
        <tool_call>
        {"name": "b", "arguments": {"y": 2}}
        </tool_call>
        """
        let r = StreamingNormalizer.normalizeComplete(raw, format: qwen)
        XCTAssertEqual(r.toolCalls.count, 2)
        XCTAssertEqual(r.toolCalls[0].id, "call_0")
        XCTAssertEqual(r.toolCalls[0].name, "a")
        XCTAssertEqual(r.toolCalls[0].arguments, #"{"x":1}"#)
        XCTAssertEqual(r.toolCalls[1].id, "call_1")
        XCTAssertEqual(r.toolCalls[1].name, "b")
    }

    func testPassthroughNoMarkers() {
        let raw = "Just a normal answer, no markers here."
        let r = StreamingNormalizer.normalizeComplete(raw, format: .passthrough)
        XCTAssertEqual(r.text, raw)
        XCTAssertEqual(r.reasoning, "")
        XCTAssertTrue(r.toolCalls.isEmpty)
    }

    /// A Qwen format applied to text with no markers must pass straight through.
    func testQwenNoMarkersPassesThrough() {
        let raw = "Hello, world."
        let r = StreamingNormalizer.normalizeComplete(raw, format: qwen)
        XCTAssertEqual(r.text, "Hello, world.")
        XCTAssertEqual(r.reasoning, "")
    }

    func testDuplicateInitialWordCleanup() {
        XCTAssertEqual(
            StreamingNormalizer.normalizeComplete("HelloHello! How can I help?", format: qwen).text,
            "Hello! How can I help?")
        XCTAssertEqual(
            StreamingNormalizer.normalizeComplete("TheThe capital is Tallahassee.", format: qwen).text,
            "The capital is Tallahassee.")
    }

    func testDuplicateInitialWordCleanupIsNarrow() {
        XCTAssertEqual(
            StreamingNormalizer.normalizeComplete("bye bye for now.", format: qwen).text,
            "bye bye for now.")
        XCTAssertEqual(
            StreamingNormalizer.normalizeComplete("murmur is a word.", format: qwen).text,
            "murmur is a word.")
        XCTAssertEqual(
            StreamingNormalizer.normalizeComplete("CanCanberra be reached?", format: qwen).text,
            "CanCanberra be reached?")
    }

    // MARK: Streaming — markers split across token boundaries

    /// Feed the raw text one Character at a time; the incremental events must reassemble into the
    /// same reasoning / text / tool-call split as the whole-string parse.
    func testStreamingCharByCharSplitsMarkers() {
        let raw = """
        <think>reason here</think>answer text<tool_call>
        {"name": "f", "arguments": {"k": "v"}}
        </tool_call>
        """
        let n = StreamingNormalizer(format: qwen)
        var reasoning = ""
        var text = ""
        var calls: [StreamingNormalizer.ToolCall] = []
        for ch in raw {
            for event in n.push(String(ch)) {
                switch event {
                case .reasoning(let s): reasoning += s
                case .text(let s): text += s
                case .toolCall(let t): calls.append(t)
                }
            }
        }
        for event in n.finish() {
            switch event {
            case .reasoning(let s): reasoning += s
            case .text(let s): text += s
            case .toolCall(let t): calls.append(t)
            }
        }
        XCTAssertEqual(reasoning, "reason here")
        XCTAssertEqual(text, "answer text")
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "f")
        XCTAssertEqual(calls[0].arguments, #"{"k":"v"}"#)
        // No marker fragment should ever leak into text.
        XCTAssertFalse(text.contains("<"))
    }

    /// Reasoning deltas must stream out incrementally (not just at the end).
    func testStreamingEmitsReasoningBeforeClose() {
        let n = StreamingNormalizer(format: qwen)
        var events = n.push("<think>hello ")
        events += n.push("world")
        // We should already have reasoning deltas before `</think>` arrives.
        let reasoningSoFar = events.compactMap { if case .reasoning(let s) = $0 { return s } else { return nil } }.joined()
        XCTAssertEqual(reasoningSoFar, "hello world")
    }

    // MARK: Qwythos — XML tool calls + implicit reasoning start

    func testImplicitReasoningStartXMLTool() {
        // Qwythos: generation prompt opens `<think>`, so output starts inside reasoning; tools
        // are emitted as `<function=…><parameter=…>…`.
        let qwythos = OutputFormat(
            family: .qwen,
            reasoningStarts: ["<think>"], reasoningEnds: ["</think>"],
            toolStarts: ["<tool_call>"], toolEnds: ["</tool_call>"],
            implicitReasoningStart: true)
        let raw = """
        I am thinking about this.</think>Here is the call.
        <tool_call>
        <function=get_weather>
        <parameter=city>
        Paris
        </parameter>
        </function>
        </tool_call>
        """
        let r = StreamingNormalizer.normalizeComplete(raw, format: qwythos)
        XCTAssertEqual(r.reasoning, "I am thinking about this.")
        XCTAssertEqual(r.text, "Here is the call.")
        XCTAssertEqual(r.toolCalls.count, 1)
        XCTAssertEqual(r.toolCalls[0].name, "get_weather")
        XCTAssertEqual(r.toolCalls[0].arguments, #"{"city":"Paris"}"#)
    }

    // MARK: Gemma — thought channel + JSON tool call

    func testGemmaThoughtChannelAndTool() {
        let gemma = OutputFormat(
            family: .gemma,
            reasoningStarts: ["<|channel>thought", "<|channel>analysis", "//thought", "<|think|>"],
            reasoningEnds: ["<channel|>", "//final"],
            toolStarts: ["<|tool_call|>", "<|tool_call>"],
            toolEnds: ["<tool_call|>", "<|tool_call|>"],
            dropMarkers: ["<|channel>final\n", "<|channel>final", "//final", "<turn|>", "<|turn>", "<pad>", "<bos>", "<eos>", "</s>"])
        let raw = "<|channel>thought\nWeighing options.<channel|>The result.<|tool_call|>{\"name\": \"lookup\", \"arguments\": {\"q\": \"x\"}}<tool_call|>"
        let r = StreamingNormalizer.normalizeComplete(raw, format: gemma)
        XCTAssertEqual(r.reasoning, "Weighing options.")
        XCTAssertEqual(r.text, "The result.")
        XCTAssertEqual(r.toolCalls.count, 1)
        XCTAssertEqual(r.toolCalls[0].name, "lookup")
        XCTAssertEqual(r.toolCalls[0].arguments, #"{"q":"x"}"#)
    }

    func testGemmaDropsChannelScaffolding() {
        let gemma = OutputFormat(
            family: .gemma,
            reasoningStarts: ["<|channel>thought"], reasoningEnds: ["<channel|>"],
            dropMarkers: ["<|channel>final\n", "<|channel>final", "//final", "<turn|>", "<|turn>", "<pad>", "<bos>", "<eos>", "</s>"])
        let raw = "<|channel>thought\nthinking<channel|><|channel>final\nThe answer.<turn|>"
        let r = StreamingNormalizer.normalizeComplete(raw, format: gemma)
        XCTAssertEqual(r.reasoning, "thinking")
        XCTAssertEqual(r.text, "The answer.")
    }

    func testGemmaDropsPadAndSlashThoughtScaffolding() {
        let gemma = OutputFormat(
            family: .gemma,
            reasoningStarts: ["<|channel>thought", "//thought"], reasoningEnds: ["<channel|>", "//final"],
            dropMarkers: ["<|channel>final\n", "<|channel>final", "//final", "<turn|>", "<|turn>", "<pad>", "<bos>", "<eos>", "</s>"])
        let raw = "Hey<pad>! there!//thought\nprivate scratch"
        let r = StreamingNormalizer.normalizeComplete(raw, format: gemma)
        XCTAssertEqual(r.text, "Hey! there!")
        XCTAssertEqual(r.reasoning, "private scratch")
    }

    // MARK: JSONAny round-trip (tool args ⇆ Anthropic input)

    func testJSONAnyParseAndEncode() {
        let value = JSONAny.parse(#"{"city":"Paris","n":3,"ok":true}"#)
        guard case .object(let o) = value else { return XCTFail("expected object") }
        XCTAssertEqual(o["city"], .string("Paris"))
        XCTAssertEqual(o["n"], .int(3))
        XCTAssertEqual(o["ok"], .bool(true))
        // toolSpec projection yields Sendable values.
        XCTAssertNotNil(value.toolSpec)
    }

    // MARK: Format detection from a real bundle

    func testDetectQwenBundle() throws {
        let dir = URL(fileURLWithPath: "/Volumes/SSD/ai-dev/coreai-pipeline/exports/qwen3-0.6b-coreai/tokenizer")
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw XCTSkip("qwen bundle not present")
        }
        let f = OutputFormat.detect(modelName: "qwen3-0.6b-coreai", tokenizerDir: dir)
        XCTAssertEqual(f.family, .qwen)
        XCTAssertEqual(f.reasoningStarts, ["<think>"])
        XCTAssertEqual(f.toolStarts, ["<tool_call>"])
        // Qwen3-0.6b generation prompt does not dangle an open `<think>`.
        XCTAssertFalse(f.implicitReasoningStart)
    }

    func testDetectQwythosImplicitStart() throws {
        let dir = URL(fileURLWithPath: "/Volumes/SSD/ai-dev/coreai-pipeline/exports/qwythos-9b-coreai/tokenizer")
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw XCTSkip("qwythos bundle not present")
        }
        let f = OutputFormat.detect(modelName: "qwythos-9b-coreai", tokenizerDir: dir)
        XCTAssertEqual(f.family, .qwen)
        XCTAssertEqual(f.reasoningStarts, ["<think>"])
        XCTAssertEqual(f.toolStarts, ["<tool_call>"])
    }

    func testQwythosDefaultTemplateDanglesThinking() {
        let template = """
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\\n\\n</think>\\n\\n' }}
            {%- else %}
                {{- '<think>\\n' }}
            {%- endif %}
        {%- endif %}
        """
        XCTAssertTrue(OutputFormat.detectImplicitReasoningStart(
            template: template, opens: ["<think>"], closes: ["</think>"]))
    }

    func testQwythosPatchedTemplateStartsInFinalText() {
        let template = """
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\\n' }}
            {{- '<think>\\n\\n</think>\\n\\n' }}
        {%- endif %}
        """
        XCTAssertFalse(OutputFormat.detectImplicitReasoningStart(
            template: template, opens: ["<think>"], closes: ["</think>"]))
    }

    func testDetectGemmaBundle() throws {
        let dir = URL(fileURLWithPath: "/Volumes/SSD/ai-dev/coreai-pipeline/exports/gemma4-31b-assistant-coreai/tokenizer")
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw XCTSkip("gemma bundle not present")
        }
        let f = OutputFormat.detect(modelName: "gemma4-31b-assistant-coreai", tokenizerDir: dir)
        XCTAssertEqual(f.family, .gemma)
        XCTAssertTrue(f.reasoningEnds.contains("<channel|>"))
    }
}
