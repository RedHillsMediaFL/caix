import XCTest

@testable import PipelineRuntime

#if COREAI_RUNTIME
final class Gemma4ChatTemplateContractTests: XCTestCase {
    func testCanonicalTemplateAuthenticationFailsClosedBeforeRendering() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-gemma4-template-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("tampered".utf8).write(
            to: directory.appendingPathComponent("chat_template.jinja"),
            options: .withoutOverwriting)

        XCTAssertThrowsError(
            try Gemma4ChatTemplateContract.compatibleTemplate(
                tokenizerDirectory: directory)
        ) { error in
            XCTAssertTrue(String(describing: error).contains("SHA-256 mismatch"))
        }
    }

    func testCanonicalDeveloperAndUserTurnsMatchTransformersFiveSix() async throws {
        let actual = try await encode([
            ["role": "developer", "content": "Be concise."],
            ["role": "user", "content": "Say hello."],
        ])

        XCTAssertEqual(actual, [
            2, 105, 9731, 107, 3912, 63510, 236761, 106, 107, 105, 2364, 107,
            37889, 29104, 236761, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
        ])
    }

    func testCanonicalToolLoopPreservesReasoningCallAndResponse() async throws {
        let tools: [[String: any Sendable]] = [[
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get weather",
                "parameters": [
                    "type": "object",
                    "properties": ["city": ["type": "string"]],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let arguments: [String: any Sendable] = ["city": "Tallahassee"]
        let function: [String: any Sendable] = [
            "name": "get_weather",
            "arguments": arguments,
        ]
        let toolCall: [String: any Sendable] = [
            "id": "call_1",
            "type": "function",
            "function": function,
        ]
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Weather?"],
            [
                "role": "assistant",
                "reasoning_content": "I should check.",
                "content": "",
                "tool_calls": [toolCall],
            ],
            [
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny, 91 F",
            ],
        ]

        let actual = try await encode(messages, tools: tools)

        XCTAssertEqual(actual, [
            2, 105, 9731, 107, 46, 163688, 236787, 828, 236779, 19323, 236782,
            7777, 236787, 52, 3407, 7606, 52, 236764, 19031, 29616, 15921, 29616,
            13319, 29616, 2084, 236787, 52, 35410, 52, 5237, 15979, 24845, 52,
            13319, 52, 1604, 2084, 236787, 52, 60688, 52, 1807, 47, 106, 107, 105,
            2364, 107, 26588, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107,
            236777, 1374, 2426, 236761, 107, 101, 48, 6639, 236787, 828, 236779,
            19323, 236782, 13319, 236787, 52, 124591, 154928, 52, 236783, 49, 50,
            6275, 236787, 828, 236779, 19323, 236782, 2394, 236787, 52, 185060,
            236764, 236743, 236819, 236770, 633, 52, 236783, 51,
        ])
    }

    func testCanonicalThinkingContextMatchesTransformersFiveSix() async throws {
        let actual = try await encode(
            [["role": "user", "content": "Solve 2+2."]],
            additionalContext: ["enable_thinking": true])

        XCTAssertEqual(actual, [
            2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 76857,
            236743, 236778, 236862, 236778, 236761, 106, 107, 105, 4368, 107,
        ])
    }

    func testCanonicalComplexToolSchemaMatchesTransformersFiveSix() async throws {
        let field: [String: any Sendable] = ["type": "string"]
        let stringItems: [String: any Sendable] = ["type": "string"]
        let values: [String: any Sendable] = ["type": "array", "items": stringItems]
        let filterProperties: [String: any Sendable] = [
            "field": field,
            "values": values,
        ]
        let filterItem: [String: any Sendable] = [
            "type": "object",
            "properties": filterProperties,
            "required": ["field"],
        ]
        let filters: [String: any Sendable] = ["type": "array", "items": filterItem]
        let limit: [String: any Sendable] = [
            "type": "integer",
            "description": "Maximum",
        ]
        let meta: [String: any Sendable] = [
            "type": "object",
            "properties": ["limit": limit] as [String: any Sendable],
            "required": ["limit"],
        ]
        let mode: [String: any Sendable] = [
            "type": "string",
            "enum": ["FAST", "DEEP"],
            "nullable": true,
        ]
        let parameters: [String: any Sendable] = [
            "type": "object",
            "properties": [
                "mode": mode,
                "filters": filters,
                "meta": meta,
            ] as [String: any Sendable],
            "required": ["mode", "filters"],
        ]
        let function: [String: any Sendable] = [
            "name": "search",
            "description": "Search records",
            "parameters": parameters,
        ]
        let tools: [[String: any Sendable]] = [[
            "type": "function",
            "function": function,
        ]]

        let actual = try await render(
            [["role": "user", "content": "Find it."]],
            tools: tools)

        XCTAssertEqual(
            actual,
            "<bos><|turn>system\n"
                + "<|tool>declaration:search{description:<|\"|>Search records<|\"|>,parameters:"
                + "{properties:{filters:{items:{properties:{field:{type:<|\"|>STRING<|\"|>},"
                + "values:{items:{type:<|\"|>STRING<|\"|>},type:<|\"|>ARRAY<|\"|>}},"
                + "required:[<|\"|>field<|\"|>],type:<|\"|>OBJECT<|\"|>},type:<|\"|>ARRAY<|\"|>},"
                + "meta:{properties:{limit:{description:<|\"|>Maximum<|\"|>,type:<|\"|>INTEGER<|\"|>}},"
                + "required:[<|\"|>limit<|\"|>],type:<|\"|>OBJECT<|\"|>},mode:{enum:[<|\"|>FAST<|\"|>,"
                + "<|\"|>DEEP<|\"|>],nullable:true,type:<|\"|>STRING<|\"|>}},required:[<|\"|>mode<|\"|>,"
                + "<|\"|>filters<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
                + "<|turn>user\nFind it.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    }

    func testCanonicalMediaPartsAndAssistantContinuationMatchTransformersFiveSix() async throws {
        let media: [[String: any Sendable]] = [
            ["type": "text", "text": "Describe:"],
            ["type": "image_url", "image_url": ["url": "data:x"] as [String: any Sendable]],
            ["type": "input_audio", "input_audio": ["data": "x"] as [String: any Sendable]],
            ["type": "video"],
        ]
        let mediaPrompt = try await render([["role": "user", "content": media]])
        XCTAssertEqual(
            mediaPrompt,
            "<bos><|turn>user\nDescribe:<|image|><|audio|><|video|><turn|>\n"
                + "<|turn>model\n<|channel>thought\n<channel|>")

        let continuation = try await render([
            ["role": "user", "content": "Start"],
            ["role": "assistant", "content": "Part one."],
            ["role": "assistant", "content": "Part two."],
        ])
        XCTAssertEqual(
            continuation,
            "<bos><|turn>user\nStart<turn|>\n<|turn>model\nPart one.Part two.<turn|>\n"
                + "<|turn>model\n<|channel>thought\n<channel|>")
    }

    func testCanonicalPreserveThinkingSwitchMatchesTransformersFiveSix() async throws {
        let function: [String: any Sendable] = [
            "name": "f",
            "description": "F",
            "parameters": [
                "type": "object",
                "properties": [String: any Sendable](),
                "required": [String](),
            ] as [String: any Sendable],
        ]
        let tools: [[String: any Sendable]] = [["type": "function", "function": function]]
        let call: [String: any Sendable] = [
            "id": "c",
            "type": "function",
            "function": ["name": "f", "arguments": [String: any Sendable]()]
                as [String: any Sendable],
        ]
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "First?"],
            [
                "role": "assistant",
                "reasoning_content": "old thought",
                "tool_calls": [call],
                "content": "",
            ],
            ["role": "tool", "tool_call_id": "c", "content": "old result"],
            ["role": "user", "content": "Now answer."],
        ]
        let withoutHistory = try await render(messages, tools: tools)
        let withHistory = try await render(
            messages,
            tools: tools,
            additionalContext: ["preserve_thinking": true])
        let prefix =
            "<bos><|turn>system\n<|tool>declaration:f{description:<|\"|>F<|\"|>,"
            + "parameters:{type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            + "<|turn>user\nFirst?<turn|>\n<|turn>model\n"
        let suffix =
            "<|tool_call>call:f{}<tool_call|><|tool_response>response:f{value:<|\"|>old result<|\"|>}"
            + "<tool_response|><turn|>\n<|turn>user\nNow answer.<turn|>\n"
            + "<|turn>model\n<|channel>thought\n<channel|>"
        XCTAssertEqual(withoutHistory, prefix + suffix)
        XCTAssertEqual(
            withHistory,
            prefix + "<|channel>thought\nold thought\n<channel|>" + suffix)
    }

    func testCanonicalRejectsStringToolArgumentsLikeTransformersFiveSix() async throws {
        let tokenizerDirectory = try tokenizerDirectory()
        let function: [String: any Sendable] = [
            "name": "f",
            "description": "F",
            "parameters": ["type": "object"] as [String: any Sendable],
        ]
        let tools: [[String: any Sendable]] = [["type": "function", "function": function]]
        let call: [String: any Sendable] = [
            "id": "c",
            "type": "function",
            "function": ["name": "f", "arguments": "{}"] as [String: any Sendable],
        ]
        do {
            _ = try await Gemma4ChatTemplateContract.render(
                tokenizerDirectory: tokenizerDirectory,
                messages: [
                    ["role": "user", "content": "x"],
                    ["role": "assistant", "content": "", "tool_calls": [call]],
                ],
                tools: tools)
            XCTFail("expected string tool arguments to be rejected")
        } catch {
            XCTAssertTrue(
                String(describing: error).contains(
                    "tool_calls[].function.arguments must be a JSON object"))
        }
    }

    private func encode(
        _ messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) async throws -> [Int] {
        let url = try tokenizerDirectory()
        return try await Gemma4ChatTemplateContract.encode(
            tokenizerDirectory: url,
            messages: messages,
            tools: tools,
            additionalContext: additionalContext)
    }

    private func render(
        _ messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) async throws -> String {
        try await Gemma4ChatTemplateContract.render(
            tokenizerDirectory: tokenizerDirectory(),
            messages: messages,
            tools: tools,
            additionalContext: additionalContext)
    }

    private func tokenizerDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["CAIX_GEMMA4_TOKENIZER"],
              !path.isEmpty
        else {
            throw XCTSkip("set CAIX_GEMMA4_TOKENIZER to the pinned July Gemma 4 snapshot")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}
#endif
