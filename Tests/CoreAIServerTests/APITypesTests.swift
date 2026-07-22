import XCTest
import PipelineRuntime

@testable import CoreAIServer

final class APITypesTests: XCTestCase {
    func testOpenAIContentBlocksPreserveMediaParts() throws {
        let data = Data(
            """
            {
              "model": "local",
              "messages": [
                {
                  "role": "user",
                  "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,abc"}}
                  ]
                }
              ]
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let message = try XCTUnwrap(request.messages.first)
        XCTAssertEqual(message.content, "Describe this.")
        XCTAssertEqual(message.media.count, 3)
        XCTAssertEqual(message.media[0].modality, "image")
        XCTAssertEqual(message.media[1].modality, "audio")
        XCTAssertEqual(message.media[2].modality, "video")

        let generation = request.toGeneration()
        XCTAssertTrue(generation.hasMultimodalContent)
        XCTAssertEqual(generation.modalities, ["audio", "image", "video"])
    }

    func testAnthropicContentBlocksPreserveImageParts() throws {
        let data = Data(
            """
            {
              "model": "local",
              "max_tokens": 8,
              "messages": [
                {
                  "role": "user",
                  "content": [
                    {"type": "text", "text": "Read this."},
                    {
                      "type": "image",
                      "source": {"type": "base64", "media_type": "image/png", "data": "abc"}
                    }
                  ]
                }
              ]
            }
            """.utf8)

        let request = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: data)
        let generation = request.toGeneration()
        let message = try XCTUnwrap(generation.messages.first)
        XCTAssertEqual(message.content, "Read this.")
        XCTAssertEqual(message.media.count, 1)
        XCTAssertEqual(message.media[0].modality, "image")
        XCTAssertTrue(generation.hasMultimodalContent)
        XCTAssertEqual(generation.modalities, ["image"])
    }

    func testTextOnlyMessagesRemainTextOnly() throws {
        let data = Data(
            """
            {
              "model": "local",
              "messages": [
                {"role": "user", "content": "Plain text."}
              ]
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let generation = request.toGeneration()
        XCTAssertEqual(generation.messages.first?.content, "Plain text.")
        XCTAssertFalse(generation.hasMultimodalContent)
        XCTAssertEqual(generation.modalities, [])
    }

    func testOpenAIStreamOptionsDecodeIncludeUsage() throws {
        let data = Data(
            """
            {
              "model": "local",
              "messages": [
                {"role": "user", "content": "Plain text."}
              ],
              "stream": true,
              "stream_options": {"include_usage": true}
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        XCTAssertEqual(request.stream, true)
        XCTAssertEqual(request.stream_options?.include_usage, true)
    }

    func testOpenAITemperatureExplicitnessSurvivesRequestMapping() throws {
        let omitted = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(
                #"{"model":"local","messages":[{"role":"user","content":"Hi"}]}"#.utf8))
        let explicit = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(
                #"{"model":"local","messages":[{"role":"user","content":"Hi"}],"temperature":0.7}"#.utf8))

        let omittedGeneration = omitted.toGeneration()
        let explicitGeneration = explicit.toGeneration()

        XCTAssertFalse(omittedGeneration.temperatureWasExplicit)
        XCTAssertTrue(explicitGeneration.temperatureWasExplicit)
        XCTAssertEqual(
            omittedGeneration.resolvedTemperature(defaultingToGreedy: true),
            0)
        XCTAssertEqual(
            explicitGeneration.resolvedTemperature(defaultingToGreedy: true),
            0.7)
    }

    func testOpenAIRichToolLoopPayloadPreservesReasoningCallsAndToolResponse() throws {
        let data = Data(
            """
            {
              "model": "local",
              "messages": [
                {"role": "user", "content": "Weather?"},
                {
                  "role": "assistant",
                  "reasoning_content": "I should check.",
                  "content": "",
                  "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                      "name": "get_weather",
                      "arguments": {"city": "Tallahassee"}
                    }
                  }]
                },
                {
                  "role": "tool",
                  "tool_call_id": "call_1",
                  "content": "sunny, 91 F"
                }
              ]
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let payload = ServerRuntime.messagePayload(request.toGeneration().messages)

        XCTAssertEqual(payload[1]["reasoning_content"] as? String, "I should check.")
        let calls = try XCTUnwrap(payload[1]["tool_calls"] as? [any Sendable])
        XCTAssertEqual(calls.count, 1)
        let call = try XCTUnwrap(calls.first as? [String: any Sendable])
        XCTAssertEqual(call["id"] as? String, "call_1")
        let function = try XCTUnwrap(call["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "get_weather")
        let arguments = try XCTUnwrap(
            function["arguments"] as? [String: any Sendable])
        XCTAssertEqual(arguments["city"] as? String, "Tallahassee")
        XCTAssertEqual(payload[2]["tool_call_id"] as? String, "call_1")
        XCTAssertEqual(payload[2]["content"] as? String, "sunny, 91 F")
    }

    func testOpenAIResponseFormatJsonObjectMapsToGeneration() throws {
        let data = Data(
            """
            {
              "model": "local",
              "messages": [
                {"role": "user", "content": "Return JSON."}
              ],
              "response_format": {"type": "json_object"}
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        XCTAssertEqual(request.response_format, .jsonObject)

        let generation = request.toGeneration()
        XCTAssertEqual(generation.responseFormat, .jsonObject)
        XCTAssertEqual(generation.responseFormat?.requiresConstrainedDecoding, true)
        XCTAssertEqual(generation.responseFormat?.constrainedJSONSchema, #"{"type":"object"}"#)
    }

    func testOpenAIResponseFormatJsonSchemaMapsToGeneration() throws {
        let data = Data(
            """
            {
              "model": "local",
              "messages": [
                {"role": "user", "content": "Extract the answer."}
              ],
              "response_format": {
                "type": "json_schema",
                "json_schema": {
                  "name": "answer",
                  "description": "A final answer wrapper.",
                  "schema": {
                    "type": "object",
                    "properties": {
                      "answer": {"type": "string"}
                    },
                    "required": ["answer"],
                    "additionalProperties": false
                  },
                  "strict": true
                }
              }
            }
            """.utf8)

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let responseFormat = try XCTUnwrap(request.response_format)
        guard case .jsonSchema(let schema) = responseFormat else {
            XCTFail("expected json_schema response_format")
            return
        }
        XCTAssertEqual(schema.name, "answer")
        XCTAssertEqual(schema.description, "A final answer wrapper.")
        XCTAssertEqual(schema.strict, true)
        XCTAssertEqual(responseFormat.requiresConstrainedDecoding, true)
        XCTAssertEqual(
            responseFormat.constrainedJSONSchema,
            #"{"additionalProperties":false,"properties":{"answer":{"type":"string"}},"required":["answer"],"type":"object"}"#)

        let generation = request.toGeneration()
        XCTAssertEqual(generation.responseFormat, request.response_format)

        let options = ServerRuntime.options(from: generation)
        XCTAssertEqual(options.constrainedJSONSchema, responseFormat.constrainedJSONSchema)
    }

    func testOpenAIStreamChunkCanEncodeUsage() throws {
        let chunk = OpenAIChatChunk(
            id: "chatcmpl-test",
            model: "local",
            created: 1,
            finish: "stop",
            usage: OpenAIChatResponse.Usage(
                prompt_tokens: 3,
                completion_tokens: 5,
                total_tokens: 8))

        let data = try JSONEncoder().encode(chunk)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 5)
        XCTAssertEqual(usage["total_tokens"] as? Int, 8)
    }

    func testServerRejectsMultimodalGenerationOnTextOnlyBackend() throws {
        let generation = GenerationRequest(
            model: "local",
            messages: [
                ChatMessage(
                    role: "user",
                    content: "Describe this.",
                    media: [
                        MediaPart(
                            type: "image_url",
                            payload: .object([
                                "type": .string("image_url"),
                                "image_url": .object(["url": .string("data:image/png;base64,aGVsbG8=")]),
                            ]))
                    ])
            ])

        let response = try XCTUnwrap(
            ServerRuntime.rejectMultimodalIfNeeded(
                generation,
                backendSupportsMultimodalInput: false))
        XCTAssertEqual(response.status.code, 400)
    }

    func testServerAllowsSingleImageGenerationOnMultimodalBackend() {
        let generation = GenerationRequest(
            model: "local",
            messages: [
                ChatMessage(
                    role: "user",
                    content: "Describe this.",
                    media: [
                        MediaPart(
                            type: "image_url",
                            payload: .object([
                                "type": .string("image_url"),
                                "image_url": .object(["url": .string("data:image/png;base64,aGVsbG8=")]),
                            ]))
                    ])
            ])

        XCTAssertNil(
            ServerRuntime.rejectMultimodalIfNeeded(
                generation,
                backendSupportsMultimodalInput: true))
    }

    func testServerRejectsAudioVideoAndMultipleImagesOnMultimodalBackend() throws {
        let audioVideo = GenerationRequest(
            model: "local",
            messages: [
                ChatMessage(
                    role: "user",
                    content: "Describe these.",
                    media: [
                        MediaPart(
                            type: "input_audio",
                            payload: .object(["input_audio": .object(["data": .string("abc")])])),
                        MediaPart(
                            type: "video_url",
                            payload: .object(["video_url": .object(["url": .string("data:video/mp4;base64,abc")])])),
                    ])
            ])
        let audioVideoResponse = try XCTUnwrap(
            ServerRuntime.rejectMultimodalIfNeeded(
                audioVideo,
                backendMultimodalCapabilities: .gemma4ImageText()))
        XCTAssertEqual(audioVideoResponse.status.code, 400)

        let multipleImages = GenerationRequest(
            model: "local",
            messages: [
                ChatMessage(
                    role: "user",
                    content: "Compare.",
                    media: [
                        MediaPart(
                            type: "image_url",
                            payload: .object(["image_url": .object(["url": .string("data:image/png;base64,aGVsbG8=")])])),
                        MediaPart(
                            type: "image_url",
                            payload: .object(["image_url": .object(["url": .string("data:image/png;base64,aGVsbG8=")])])),
                    ])
            ])
        let multipleImageResponse = try XCTUnwrap(
            ServerRuntime.rejectMultimodalIfNeeded(
                multipleImages,
                backendMultimodalCapabilities: .gemma4ImageText()))
        XCTAssertEqual(multipleImageResponse.status.code, 400)
    }

    func testMultimodalRequestRejectsRemoteImageURLs() {
        let generation = GenerationRequest(
            model: "local",
            messages: [
                ChatMessage(
                    role: "user",
                    content: "Describe this.",
                    media: [
                        MediaPart(
                            type: "image_url",
                            payload: .object([
                                "type": .string("image_url"),
                                "image_url": .object(["url": .string("https://example.com/image.png")]),
                            ]))
                    ])
            ])

        let error = MultimodalRequestSupport.validateMinimalSingleImageRequest(
            generation,
            capabilities: .gemma4ImageText())
        XCTAssertEqual(
            error?.description,
            "remote image URLs are not supported; send a base64 data URL")
    }

    func testMultimodalRequestRejectsBlockedMonolithicGemmaRoute() {
        let generation = GenerationRequest(
            model: "local",
            messages: [
                ChatMessage(
                    role: "user",
                    content: "Describe this.",
                    media: [
                        MediaPart(
                            type: "image_url",
                            payload: .object([
                                "type": .string("image_url"),
                                "image_url": .object(["url": .string("data:image/png;base64,aGVsbG8=")]),
                            ]))
                    ])
            ])

        let error = MultimodalRequestSupport.validateMinimalSingleImageRequest(
            generation,
            capabilities: .gemma4ImageText(
                maxSoftTokensPerImage: 280,
                backend: "monolithic",
                routeAvailable: false))
        XCTAssertEqual(
            error?.description,
            "monolithic Gemma image-text bundles are discovered, but native prefill_multimodal serving is not wired yet")
    }

    func testServerRejectsStructuredResponseFormatBeforeRuntime() throws {
        let generation = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Return JSON.")],
            responseFormat: .jsonSchema(
                OpenAIResponseFormat.JSONSchema(
                    name: "answer",
                    schema: .object(["type": .string("object")]),
                    strict: true)))

        if CoreAIPipeline.supportsConstrainedDecoding {
            XCTAssertNil(ServerRuntime.rejectUnsupportedResponseFormatIfNeeded(generation))
        } else {
            let response = try XCTUnwrap(
                ServerRuntime.rejectUnsupportedResponseFormatIfNeeded(generation))
            XCTAssertEqual(response.status.code, 400)
        }
    }

    func testServerRejectsStructuredResponseFormatOnUnsupportedBackend() throws {
        let generation = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Return JSON.")],
            responseFormat: .jsonSchema(
                OpenAIResponseFormat.JSONSchema(
                    name: "answer",
                    schema: .object(["type": .string("object")]),
                    strict: true)))

        let response = try XCTUnwrap(
            ServerRuntime.rejectUnsupportedResponseFormatIfNeeded(
                generation,
                backendSupportsConstrainedDecoding: false))
        XCTAssertEqual(response.status.code, 400)
    }

    func testServerAllowsStructuredResponseFormatOnSupportedBackend() {
        let generation = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Return JSON.")],
            responseFormat: .jsonObject)

        XCTAssertNil(
            ServerRuntime.rejectUnsupportedResponseFormatIfNeeded(
                generation,
                backendSupportsConstrainedDecoding: true))
    }

    func testServerAllowsTextResponseFormatToRuntime() {
        let defaultGeneration = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Plain text.")])
        let textGeneration = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Plain text.")],
            responseFormat: .text)

        XCTAssertNil(ServerRuntime.rejectUnsupportedResponseFormatIfNeeded(defaultGeneration))
        XCTAssertNil(ServerRuntime.rejectUnsupportedResponseFormatIfNeeded(textGeneration))
    }

    func testServerAllowsTextOnlyGenerationToRuntime() {
        let generation = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Plain text.")])

        XCTAssertNil(
            ServerRuntime.rejectMultimodalIfNeeded(
                generation,
                backendSupportsMultimodalInput: false))
    }
}
