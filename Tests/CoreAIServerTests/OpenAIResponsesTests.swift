import XCTest
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import PipelineRuntime

@testable import CoreAIServer

final class OpenAIResponsesTests: XCTestCase {
    // MARK: - Request decoding

    func testResponsesInputAcceptsPlainString() throws {
        let data = Data(
            """
            {
              "model": "local",
              "input": "Say hello.",
              "max_output_tokens": 64,
              "temperature": 0.2,
              "top_p": 0.9,
              "store": true,
              "metadata": {"trace": "t1"}
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIResponsesRequest.self, from: data)
        XCTAssertEqual(request.model, "local")
        XCTAssertEqual(request.store, true)
        XCTAssertEqual(request.metadata, ["trace": .string("t1")])
        XCTAssertEqual(
            request.items,
            [OpenAIResponsesInputItem(role: "user", segments: [.text("Say hello.")])])

        let generation = request.toGeneration(
            messages: [ChatMessage(role: "user", content: "Say hello.")])
        XCTAssertEqual(generation.maxTokens, 64)
        XCTAssertEqual(generation.temperature, 0.2)
        XCTAssertTrue(generation.temperatureWasExplicit)
        XCTAssertEqual(generation.topP, 0.9)
        XCTAssertFalse(generation.stream)
    }

    func testResponsesInputAcceptsMessageItemsWithParts() throws {
        let data = Data(
            """
            {
              "model": "local",
              "input": [
                {
                  "type": "message",
                  "role": "user",
                  "content": [
                    {"type": "input_text", "text": "First part."},
                    {"type": "input_text", "text": "Second part."}
                  ]
                },
                {
                  "type": "message",
                  "role": "assistant",
                  "content": [{"type": "output_text", "text": "Earlier answer."}]
                },
                {"role": "user", "content": "Bare shorthand item."}
              ]
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIResponsesRequest.self, from: data)
        let items = request.items
        XCTAssertEqual(items.count, 3)
        XCTAssertEqual(items[0].type, "message")
        XCTAssertEqual(items[0].role, "user")
        XCTAssertEqual(items[0].segments, [.text("First part."), .text("Second part.")])
        XCTAssertEqual(items[1].role, "assistant")
        XCTAssertEqual(items[1].segments, [.text("Earlier answer.")])
        XCTAssertNil(items[2].type)
        XCTAssertEqual(items[2].segments, [.text("Bare shorthand item.")])
    }

    func testResponsesInputAudioPartsDecodeWithBothPayloadKeys() throws {
        let base64 = Data([1, 2, 3]).base64EncodedString()
        let data = Data(
            """
            {
              "model": "local",
              "input": [
                {
                  "type": "message",
                  "role": "user",
                  "content": [
                    {"type": "input_audio", "input_audio": {"data": "\(base64)", "format": "wav"}},
                    {"type": "input_audio", "audio": {"data": "\(base64)", "format": "mp3"}}
                  ]
                }
              ]
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIResponsesRequest.self, from: data)
        XCTAssertEqual(
            request.items.first?.segments,
            [
                .audio(base64: base64, format: "wav"),
                .audio(base64: base64, format: "mp3"),
            ])
    }

    // MARK: - Message building

    func testResponsesInstructionsBecomePrependedSystemMessage() async throws {
        let runtime = try makeServerRuntime(audioService: nil)
        let request = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(
                """
                {
                  "model": "local",
                  "instructions": "Answer tersely.",
                  "input": [
                    {"type": "message", "role": "developer", "content": "House rules."},
                    {"role": "user", "content": "Hi."}
                  ]
                }
                """.utf8))

        let messages = try await runtime.responsesMessages(from: request)
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages[0].role, "system")
        XCTAssertEqual(messages[0].content, "Answer tersely.")
        // OpenAI's developer role carries system authority.
        XCTAssertEqual(messages[1].role, "system")
        XCTAssertEqual(messages[1].content, "House rules.")
        XCTAssertEqual(messages[2].role, "user")
        XCTAssertEqual(messages[2].content, "Hi.")
    }

    func testResponsesRejectsUnsupportedItemsAndParts() async throws {
        let runtime = try makeServerRuntime(audioService: nil)

        let badItem = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(
                #"{"model":"local","input":[{"type":"function_call","role":"user","content":"x"}]}"#.utf8))
        do {
            _ = try await runtime.responsesMessages(from: badItem)
            XCTFail("unsupported item type unexpectedly accepted")
        } catch let error as ServerRuntime.ResponsesInputError {
            XCTAssertEqual(error, .unsupportedItemType("function_call"))
        }

        let badPart = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(
                """
                {"model":"local","input":[{"role":"user","content":[
                  {"type":"input_image","image_url":"data:image/png;base64,abc"}]}]}
                """.utf8))
        do {
            _ = try await runtime.responsesMessages(from: badPart)
            XCTFail("unsupported content part unexpectedly accepted")
        } catch let error as ServerRuntime.ResponsesInputError {
            XCTAssertEqual(error, .unsupportedContentPart("input_image"))
        }

        let empty = try JSONDecoder().decode(
            OpenAIResponsesRequest.self, from: Data(#"{"model":"local"}"#.utf8))
        do {
            _ = try await runtime.responsesMessages(from: empty)
            XCTFail("empty input unexpectedly accepted")
        } catch let error as ServerRuntime.ResponsesInputError {
            XCTAssertEqual(error, .emptyInput)
        }
    }

    func testResponsesInputAudioSplicesWhisperTranscriptIntoMessageText() async throws {
        let transcriber = ResponsesRecordingWhisperTranscriber()
        let service = OpenAIAudioTranscriptionService(
            transcriber: transcriber,
            decodeAudio: { bytes in
                XCTAssertEqual(bytes, Data([1, 2, 3]))
                return WhisperPCM(
                    samples: [0.25, -0.5, 1], sampleRate: 16_000, wasTruncated: false)
            },
            extractFeatures: { _ in [0.5, -1.25] })
        let runtime = try makeServerRuntime(audioService: service)
        let base64 = Data([1, 2, 3]).base64EncodedString()
        let request = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(
                """
                {
                  "model": "local",
                  "input": [
                    {
                      "type": "message",
                      "role": "user",
                      "content": [
                        {"type": "input_text", "text": "What was said?"},
                        {"type": "input_audio", "input_audio": {"data": "\(base64)", "format": "wav"}}
                      ]
                    }
                  ]
                }
                """.utf8))

        let messages = try await runtime.responsesMessages(from: request)
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages[0].role, "user")
        XCTAssertEqual(
            messages[0].content,
            "What was said?\n[audio transcript] native transcript")
        let call = await transcriber.lastCall()
        XCTAssertNotNil(call, "audio part must route through the native Whisper seam")
    }

    func testResponsesAudioWithoutWhisperFailsWithServiceUnavailable() async throws {
        let runtime = try makeServerRuntime(audioService: nil)
        let router = Router()
        runtime.register(on: router)
        let app = Application(responder: router.buildResponder())
        let base64 = Data([1, 2, 3]).base64EncodedString()
        let body =
            """
            {"model":"local","input":[{"role":"user","content":[
              {"type":"input_audio","input_audio":{"data":"\(base64)","format":"wav"}}]}]}
            """

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/responses",
                method: .post,
                body: ByteBuffer(string: body)
            ) { response in
                XCTAssertEqual(response.status, .serviceUnavailable)
                XCTAssertTrue(String(buffer: response.body).contains("Whisper"))
            }
        }
    }

    func testResponsesInputAudioRoutesToWhisperSeamOverHTTP() async throws {
        let transcriber = ResponsesRecordingWhisperTranscriber()
        let service = OpenAIAudioTranscriptionService(
            transcriber: transcriber,
            decodeAudio: { _ in
                WhisperPCM(samples: [0.25, -0.5, 1], sampleRate: 16_000, wasTruncated: false)
            },
            extractFeatures: { _ in [0.5, -1.25] })
        let runtime = try makeServerRuntime(audioService: service)
        let router = Router()
        runtime.register(on: router)
        let app = Application(responder: router.buildResponder())
        let base64 = Data([1, 2, 3]).base64EncodedString()
        let body =
            """
            {"model":"local","input":[{"role":"user","content":[
              {"type":"input_audio","input_audio":{"data":"\(base64)","format":"wav"}}]}]}
            """

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/responses",
                method: .post,
                body: ByteBuffer(string: body)
            ) { response in
                // No chat bundle is installed in this fixture, so generation cannot start —
                // but the audio must already have been transcribed through the resident seam.
                XCTAssertEqual(response.status, .serviceUnavailable)
                XCTAssertTrue(String(buffer: response.body).contains("could not load model"))
            }
        }
        let call = await transcriber.lastCall()
        XCTAssertNotNil(call, "HTTP request must reach the native Whisper seam before generation")
    }

    // MARK: - Error shapes

    func testResponsesUnknownModelFailsWithModelLoadErrorShape() async throws {
        let runtime = try makeServerRuntime(audioService: nil)
        let router = Router()
        runtime.register(on: router)
        let app = Application(responder: router.buildResponder())

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/responses",
                method: .post,
                body: ByteBuffer(string: #"{"model":"no-such-model","input":"Hi."}"#)
            ) { response in
                XCTAssertEqual(response.status, .serviceUnavailable)
                let object = try JSONSerialization.jsonObject(
                    with: Data(response.body.readableBytesView)) as? [String: Any]
                let error = try XCTUnwrap(object?["error"] as? [String: Any])
                let message = try XCTUnwrap(error["message"] as? String)
                XCTAssertTrue(message.contains("could not load model 'no-such-model'"))
                XCTAssertEqual(error["type"] as? String, "coreai_error")
            }
        }
    }

    func testResponsesRejectsMalformedBody() async throws {
        let runtime = try makeServerRuntime(audioService: nil)
        let router = Router()
        runtime.register(on: router)
        let app = Application(responder: router.buildResponder())

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/responses",
                method: .post,
                body: ByteBuffer(string: #"{"input": 42}"#)
            ) { response in
                XCTAssertEqual(response.status, .badRequest)
                XCTAssertTrue(
                    String(buffer: response.body).contains("invalid OpenAI responses request"))
            }
        }
    }

    // MARK: - Response object shape

    func testResponsesResponseObjectEncodesOpenAIShape() throws {
        let request = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(
                """
                {
                  "model": "local",
                  "input": "Hi.",
                  "instructions": "Answer tersely.",
                  "max_output_tokens": 32,
                  "temperature": 0.5,
                  "top_p": 0.9,
                  "store": true,
                  "metadata": {"trace": "t1"}
                }
                """.utf8))
        let response = OpenAIResponsesResponse(
            id: "resp_abc123",
            createdAt: 1_700_000_000,
            status: "completed",
            model: "gemma",
            output: [OpenAIResponsesResponse.OutputMessage(
                id: "msg_def456",
                status: "completed",
                content: [OpenAIResponsesResponse.ContentPart(text: "Hello!")])],
            usage: OpenAIResponsesResponse.Usage(
                input_tokens: 7, output_tokens: 3, total_tokens: 10),
            request: request)

        let data = try JSONEncoder().encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(object["id"] as? String, "resp_abc123")
        XCTAssertTrue((object["id"] as? String)?.hasPrefix("resp_") == true)
        XCTAssertEqual(object["object"] as? String, "response")
        XCTAssertEqual(object["created_at"] as? Int, 1_700_000_000)
        XCTAssertEqual(object["status"] as? String, "completed")
        XCTAssertEqual(object["model"] as? String, "gemma")

        let output = try XCTUnwrap(object["output"] as? [[String: Any]])
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0]["type"] as? String, "message")
        XCTAssertTrue((output[0]["id"] as? String)?.hasPrefix("msg_") == true)
        XCTAssertEqual(output[0]["status"] as? String, "completed")
        XCTAssertEqual(output[0]["role"] as? String, "assistant")
        let content = try XCTUnwrap(output[0]["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 1)
        XCTAssertEqual(content[0]["type"] as? String, "output_text")
        XCTAssertEqual(content[0]["text"] as? String, "Hello!")
        XCTAssertEqual((content[0]["annotations"] as? [Any])?.isEmpty, true)

        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual(usage["input_tokens"] as? Int, 7)
        XCTAssertEqual(usage["output_tokens"] as? Int, 3)
        XCTAssertEqual(usage["total_tokens"] as? Int, 10)

        // Absent values must be explicit nulls, and store is always false (stateless server).
        XCTAssertTrue(object["error"] is NSNull)
        XCTAssertTrue(object["incomplete_details"] is NSNull)
        XCTAssertEqual(object["instructions"] as? String, "Answer tersely.")
        XCTAssertEqual(object["max_output_tokens"] as? Int, 32)
        XCTAssertEqual(object["temperature"] as? Double, 0.5)
        XCTAssertEqual(object["top_p"] as? Double, 0.9)
        XCTAssertEqual(object["store"] as? Bool, false)
        XCTAssertEqual(object["metadata"] as? [String: String], ["trace": "t1"])
        XCTAssertEqual(object["parallel_tool_calls"] as? Bool, true)
        XCTAssertEqual(object["tool_choice"] as? String, "auto")
        XCTAssertEqual((object["tools"] as? [Any])?.isEmpty, true)
    }

    func testResponsesIncompleteStatusOnMaxTokenStop() throws {
        let maxTokens = OpenAIResponsesResponse.completionStatus(for: .maxTokens)
        XCTAssertEqual(maxTokens.status, "incomplete")
        XCTAssertEqual(maxTokens.incompleteDetails?.reason, "max_output_tokens")
        let contextLimit = OpenAIResponsesResponse.completionStatus(for: .contextLimit)
        XCTAssertEqual(contextLimit.status, "incomplete")
        XCTAssertEqual(OpenAIResponsesResponse.completionStatus(for: .eos).status, "completed")
        XCTAssertNil(OpenAIResponsesResponse.completionStatus(for: .eos).incompleteDetails)
        XCTAssertEqual(
            OpenAIResponsesResponse.completionStatus(for: .stopSequence).status, "completed")

        let request = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(#"{"model":"local","input":"Hi."}"#.utf8))
        let response = OpenAIResponsesResponse(
            id: "resp_abc123",
            createdAt: 1,
            status: maxTokens.status,
            model: "gemma",
            output: [],
            usage: OpenAIResponsesResponse.Usage(
                input_tokens: 7, output_tokens: 32, total_tokens: 39),
            incompleteDetails: maxTokens.incompleteDetails,
            request: request)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(response)) as? [String: Any])
        XCTAssertEqual(object["status"] as? String, "incomplete")
        let details = try XCTUnwrap(object["incomplete_details"] as? [String: Any])
        XCTAssertEqual(details["reason"] as? String, "max_output_tokens")
    }

    // MARK: - Streaming event framing

    func testResponsesStreamingFramesFollowEventProtocol() throws {
        let request = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data(#"{"model":"local","input":"Hi.","stream":true}"#.utf8))
        var framer = ResponsesStreamFramer(responseID: "resp_abc123", itemID: "msg_def456")
        let inProgress = OpenAIResponsesResponse(
            id: "resp_abc123", createdAt: 1, status: "in_progress", model: "gemma",
            output: [], usage: nil, request: request)
        let item = OpenAIResponsesResponse.OutputMessage(
            id: "msg_def456", status: "completed",
            content: [OpenAIResponsesResponse.ContentPart(text: "Hi!")])
        let final = OpenAIResponsesResponse(
            id: "resp_abc123", createdAt: 1, status: "completed", model: "gemma",
            output: [item],
            usage: OpenAIResponsesResponse.Usage(
                input_tokens: 4, output_tokens: 2, total_tokens: 6),
            request: request)

        let frames = [
            framer.created(inProgress),
            framer.inProgress(inProgress),
            framer.outputItemAdded(OpenAIResponsesResponse.OutputMessage(
                id: "msg_def456", status: "in_progress", content: [])),
            framer.contentPartAdded(),
            framer.outputTextDelta("Hi"),
            framer.outputTextDelta("!"),
            framer.outputTextDone("Hi!"),
            framer.contentPartDone("Hi!"),
            framer.outputItemDone(item),
            framer.finished(final),
        ]
        let expectedEvents = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]

        for (index, frame) in frames.enumerated() {
            let lines = frame.split(separator: "\n", omittingEmptySubsequences: false)
            XCTAssertEqual(lines.count, 4, "frame must be event + data + blank terminator")
            XCTAssertEqual(String(lines[0]), "event: \(expectedEvents[index])")
            XCTAssertTrue(lines[1].hasPrefix("data: "))
            XCTAssertEqual(String(lines[2]), "")
            XCTAssertEqual(String(lines[3]), "")
            let payload = try XCTUnwrap(
                JSONSerialization.jsonObject(
                    with: Data(lines[1].dropFirst("data: ".count).utf8)) as? [String: Any])
            XCTAssertEqual(payload["type"] as? String, expectedEvents[index])
            XCTAssertEqual(
                payload["sequence_number"] as? Int, index,
                "sequence numbers must increase monotonically from 0")
        }

        // Delta frames carry the delta text and addressing indices.
        let deltaPayload = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(
                    frames[4].split(separator: "\n")[1].dropFirst("data: ".count).utf8))
                as? [String: Any])
        XCTAssertEqual(deltaPayload["delta"] as? String, "Hi")
        XCTAssertEqual(deltaPayload["item_id"] as? String, "msg_def456")
        XCTAssertEqual(deltaPayload["output_index"] as? Int, 0)
        XCTAssertEqual(deltaPayload["content_index"] as? Int, 0)

        // The terminal frame carries the full final response object.
        let finalPayload = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(
                    frames[9].split(separator: "\n")[1].dropFirst("data: ".count).utf8))
                as? [String: Any])
        let finalResponse = try XCTUnwrap(finalPayload["response"] as? [String: Any])
        XCTAssertEqual(finalResponse["id"] as? String, "resp_abc123")
        XCTAssertEqual(finalResponse["status"] as? String, "completed")
        XCTAssertEqual(
            (finalResponse["usage"] as? [String: Any])?["total_tokens"] as? Int, 6)
        let outputs = try XCTUnwrap(finalResponse["output"] as? [[String: Any]])
        XCTAssertEqual(outputs.count, 1)
        XCTAssertEqual(
            ((outputs[0]["content"] as? [[String: Any]])?.first?["text"]) as? String, "Hi!")
    }

    // MARK: - Fixtures

    private func makeServerRuntime(
        audioService: OpenAIAudioTranscriptionService?
    ) throws -> ServerRuntime {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-responses-route-\(UUID().uuidString)", isDirectory: true)
        return try ServerRuntime(
            host: "127.0.0.1",
            port: 1238,
            exportsDir: root.appendingPathComponent("exports", isDirectory: true),
            registryPath: root.appendingPathComponent("registry.json"),
            webDir: root.appendingPathComponent("web", isDirectory: true),
            convertScript: root.appendingPathComponent("convert.py").path,
            pythonExecutable: "python3",
            caixVersion: "test",
            verbose: false,
            conversionGuardEnabled: false,
            audioTranscriptionService: audioService)
    }
}

private actor ResponsesRecordingWhisperTranscriber: WhisperTranscribing {
    struct Call: Sendable {
        var inputFeatures: [Float16]
        var requestedLanguage: String?
        var includeTimestamps: Bool
    }

    private var call: Call?

    func transcribe(
        inputFeatures: [Float16],
        requestedLanguage: String?,
        includeTimestamps: Bool,
        onTextToken: (@Sendable (Int32) -> Void)?
    ) async throws -> WhisperResidentEngine.Result {
        call = Call(
            inputFeatures: inputFeatures,
            requestedLanguage: requestedLanguage,
            includeTimestamps: includeTimestamps)
        return WhisperResidentEngine.Result(
            text: "native transcript",
            textTokenIDs: [100],
            language: requestedLanguage ?? "en",
            languageTokenID: 50_259,
            reachedEndToken: true,
            wasTruncated: false,
            timings: .init(
                queueSeconds: 0,
                encodeSeconds: 0.1,
                crossKVLoadSeconds: 0.01,
                decodeSeconds: 0.2,
                totalSeconds: 0.31))
    }

    func lastCall() -> Call? { call }
}
