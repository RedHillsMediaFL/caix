import XCTest

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

    func testServerRejectsMultimodalGenerationBeforeRuntime() throws {
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
                                "image_url": .object(["url": .string("data:image/png;base64,abc")]),
                            ]))
                    ])
            ])

        let response = try XCTUnwrap(ServerRuntime.rejectMultimodalIfNeeded(generation))
        XCTAssertEqual(response.status.code, 400)
    }

    func testServerAllowsTextOnlyGenerationToRuntime() {
        let generation = GenerationRequest(
            model: "local",
            messages: [ChatMessage(role: "user", content: "Plain text.")])

        XCTAssertNil(ServerRuntime.rejectMultimodalIfNeeded(generation))
    }
}
