import XCTest
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import PipelineRuntime

@testable import CoreAIServer

final class JSONCoercionTests: XCTestCase {
    // MARK: - Extraction

    func testExtractsBareJSONObject() {
        let text = #"{"title":"Summary","score":3}"#
        XCTAssertEqual(
            JSONCoercion.extractJSONObject(from: text, requiredKeys: ["title"]), text)
    }

    func testExtractsFencedJSONObject() {
        let text = """
            ```json
            {"title":"Summary"}
            ```
            """
        XCTAssertEqual(
            JSONCoercion.extractJSONObject(from: text, requiredKeys: ["title"]),
            #"{"title":"Summary"}"#)
    }

    func testExtractsJSONAfterProse() {
        let text = #"Here is the summary: {"title":"Summary","done":true} — hope it helps!"#
        XCTAssertEqual(
            JSONCoercion.extractJSONObject(from: text, requiredKeys: ["title"]),
            #"{"title":"Summary","done":true}"#)
    }

    func testExtractsJSONWithNestedObjectsAndBracesInStrings() {
        // Nested object, plus braces that live inside a string value and must be ignored.
        let text = #"{"a":{"b":1},"c":"a } and { inside a string","d":"esc\"aped"}"#
        let extracted = JSONCoercion.extractJSONObject(from: text, requiredKeys: [])
        XCTAssertEqual(extracted, text)
        // And the exact substring is real, parseable JSON.
        if let extracted, let data = extracted.data(using: .utf8) {
            XCTAssertNotNil(try? JSONSerialization.jsonObject(with: data))
        } else {
            XCTFail("expected a parseable extracted object")
        }
    }

    func testReturnsNilWhenRequiredKeyMissing() {
        let text = #"{"other":"value"}"#
        XCTAssertNil(JSONCoercion.extractJSONObject(from: text, requiredKeys: ["title"]))
    }

    func testReturnsNilWhenUnparseable() {
        // Balanced braces but not valid JSON.
        XCTAssertNil(JSONCoercion.extractJSONObject(from: #"{"broken": }"#, requiredKeys: []))
        // No object at all.
        XCTAssertNil(JSONCoercion.extractJSONObject(from: "no json here", requiredKeys: []))
        // Unbalanced (never closes).
        XCTAssertNil(JSONCoercion.extractJSONObject(from: #"{"a":1"#, requiredKeys: []))
    }

    // MARK: - System instruction

    func testSystemInstructionForTextIsNil() {
        XCTAssertNil(JSONCoercion.systemInstruction(for: .text))
    }

    func testSystemInstructionForJSONObject() {
        let instruction = JSONCoercion.systemInstruction(for: .jsonObject)
        XCTAssertEqual(
            instruction,
            "Respond with ONLY a single valid JSON object. No markdown, no code fences, no explanation — output only the JSON.")
    }

    func testSystemInstructionForJSONSchemaIncludesSchemaAndName() {
        let format = OpenAIResponseFormat.jsonSchema(
            OpenAIResponseFormat.JSONSchema(
                name: "summary",
                description: "A short recap.",
                schema: .object([
                    "type": .string("object"),
                    "required": .array([.string("title")]),
                ]),
                strict: true))
        let text = JSONCoercion.systemInstruction(for: format) ?? ""
        XCTAssertTrue(text.hasPrefix("Respond with ONLY a single valid JSON object."))
        // Names the REQUIRED top-level fields and forbids wrapping (the fix for the model nesting
        // the object under the schema name).
        XCTAssertTrue(text.contains("title"), "should name the required top-level fields")
        XCTAssertTrue(text.contains("Do NOT wrap"), "should forbid wrapping in an envelope key")
        XCTAssertTrue(text.contains("A short recap."), "should include the description")
        // The schema JSON (compact, sorted keys) must be embedded verbatim.
        XCTAssertTrue(text.contains(#"{"required":["title"],"type":"object"}"#))
    }

    // MARK: - Required keys

    func testRequiredKeysParsesSchemaRequiredArray() {
        let format = OpenAIResponseFormat.jsonSchema(
            OpenAIResponseFormat.JSONSchema(
                name: "answer",
                schema: .object([
                    "type": .string("object"),
                    "required": .array([.string("title"), .string("score")]),
                ])))
        XCTAssertEqual(JSONCoercion.requiredKeys(for: format), ["title", "score"])
    }

    func testRequiredKeysEmptyForJSONObjectAndText() {
        XCTAssertEqual(JSONCoercion.requiredKeys(for: .jsonObject), [])
        XCTAssertEqual(JSONCoercion.requiredKeys(for: .text), [])
    }

    func testRetryReminderReferencesJSONOnly() {
        let reminder = JSONCoercion.retryReminder()
        XCTAssertTrue(reminder.contains("ONLY a single JSON object"))
        XCTAssertTrue(reminder.contains("top-level keys"))
    }

    // MARK: - Constraint plan

    func testConstraintPlanCoercesWhenBackendUnsupported() {
        let gen = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Return JSON.")],
            responseFormat: .jsonObject)
        guard case .coerce(let format) = ServerRuntime.constraintPlan(
            for: gen, backendSupportsConstrainedDecoding: false)
        else {
            return XCTFail("expected .coerce on a non-constrained backend")
        }
        XCTAssertEqual(format, .jsonObject)
    }

    func testConstraintPlanTrueConstrainedWhenSupported() {
        let gen = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Return JSON.")],
            responseFormat: .jsonObject)
        guard case .trueConstrained = ServerRuntime.constraintPlan(
            for: gen, backendSupportsConstrainedDecoding: true)
        else {
            return XCTFail("expected .trueConstrained when the backend supports it")
        }
    }

    func testConstraintPlanNoneForPlainText() {
        let gen = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Hi.")],
            responseFormat: .text)
        guard case ServerRuntime.ConstraintPlan.none = ServerRuntime.constraintPlan(
            for: gen, backendSupportsConstrainedDecoding: false)
        else {
            return XCTFail("expected .none for a text response_format")
        }
    }

    // MARK: - Responses text.format parsing

    func testResponsesTextFormatParsesJSONSchema() throws {
        let request = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(
                """
                {
                  "model": "local",
                  "input": "Summarize.",
                  "text": {
                    "format": {
                      "type": "json_schema",
                      "name": "summary",
                      "strict": true,
                      "schema": {"type": "object", "required": ["title"]}
                    }
                  }
                }
                """.utf8))
        guard case .jsonSchema(let schema)? = request.resolvedResponseFormat else {
            return XCTFail("expected a json_schema response format from text.format")
        }
        XCTAssertEqual(schema.name, "summary")
        XCTAssertEqual(schema.strict, true)
        XCTAssertEqual(
            JSONCoercion.requiredKeys(for: request.resolvedResponseFormat!), ["title"])
        // And it flows onto the generation contract.
        XCTAssertEqual(
            request.toGeneration(messages: []).responseFormat, request.resolvedResponseFormat)
    }

    func testResponsesAcceptsTopLevelResponseFormatForLeniency() throws {
        let request = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(
                #"{"model":"local","input":"Hi.","response_format":{"type":"json_object"}}"#.utf8))
        XCTAssertEqual(request.resolvedResponseFormat, .jsonObject)
    }

    func testResponsesWithoutFormatHasNoResponseFormat() throws {
        let request = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(#"{"model":"local","input":"Hi."}"#.utf8))
        XCTAssertNil(request.resolvedResponseFormat)
    }

    // MARK: - Handler-level: no 400 on a non-constrained backend

    func testChatJSONSchemaOnNonConstrainedBackendDoesNotReturn400() async throws {
        let runtime = try makeServerRuntime()
        let router = Router()
        runtime.register(on: router)
        let app = Application(responder: router.buildResponder())
        let body =
            """
            {
              "model": "no-such-model",
              "messages": [{"role": "user", "content": "Summarize."}],
              "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "summary", "schema": {"type": "object", "required": ["title"]}}
              }
            }
            """

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                body: ByteBuffer(string: body)
            ) { response in
                // Before coercion this returned 400 ("requires constrained decoding"). Now the
                // structured-output request is accepted and generation is attempted — which, with no
                // bundle installed, fails at model-load (503), never a 400 response_format rejection.
                XCTAssertNotEqual(response.status, .badRequest)
                XCTAssertEqual(response.status, .serviceUnavailable)
                XCTAssertTrue(String(buffer: response.body).contains("could not load model"))
            }
        }
    }

    // MARK: - Fixtures

    private func makeServerRuntime() throws -> ServerRuntime {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-coercion-\(UUID().uuidString)", isDirectory: true)
        return try ServerRuntime(
            host: "127.0.0.1",
            port: 1239,
            exportsDir: root.appendingPathComponent("exports", isDirectory: true),
            registryPath: root.appendingPathComponent("registry.json"),
            webDir: root.appendingPathComponent("web", isDirectory: true),
            convertScript: root.appendingPathComponent("convert.py").path,
            pythonExecutable: "python3",
            caixVersion: "test",
            verbose: false,
            conversionGuardEnabled: false,
            audioTranscriptionService: nil)
    }
}
