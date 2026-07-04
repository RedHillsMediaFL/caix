import Foundation

struct CaixClient {
    var baseURL: URL
    private let session: URLSession
    private let decoder = JSONDecoder()

    init(baseURL: URL, timeout: TimeInterval = 90, protocolClasses: [AnyClass]? = nil) {
        self.baseURL = ServerEndpoint.normalizedBaseURL(baseURL) ?? baseURL
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = max(timeout, 600)
        configuration.httpAdditionalHeaders = [
            "User-Agent": "caix-ios/0.1"
        ]
        configuration.protocolClasses = protocolClasses
        self.session = URLSession(configuration: configuration)
    }

    func fetchModels() async throws -> [CaixModel] {
        let apiModels = (try? await fetchAPIModels()) ?? []
        let servedIDs = (try? await fetchOpenAIModelIDs()) ?? []
        let lookup = Dictionary(uniqueKeysWithValues: apiModels.map { ($0.name, $0) })
        let ids = servedIDs.isEmpty
            ? apiModels.filter { $0.bundle != false }.map(\.name)
            : servedIDs

        return ids.map { id in
            let row = lookup[id]
            return CaixModel(
                id: id,
                status: row?.status,
                mode: row?.mode,
                supportsReasoning: row?.reasoningSupported,
                supportsImages: CaixModel.inferredImageSupport(
                    multimodalSupported: row?.multimodalSupported,
                    mode: row?.mode,
                    name: id)
            )
        }
        .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
    }

    func streamChat(
        messages: [OutboundChatMessage],
        model: String,
        maxTokens: Int = 1024,
        temperature: Double = 0.7
    ) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let body = ChatCompletionRequest(
                        model: model,
                        messages: messages,
                        max_tokens: maxTokens,
                        temperature: temperature,
                        stream: true,
                        stream_options: .init(include_usage: true)
                    )
                    var request = URLRequest(url: url(path: "v1/chat/completions"))
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.httpBody = try JSONEncoder().encode(body)

                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse else {
                        throw CaixClientError.invalidResponse
                    }
                    guard (200..<300).contains(http.statusCode) else {
                        let text = try await collectErrorBody(from: bytes)
                        throw CaixClientError.httpStatus(http.statusCode, text)
                    }

                    for try await line in bytes.lines {
                        if Task.isCancelled { break }
                        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard trimmed.hasPrefix("data:") else { continue }
                        let payload = trimmed.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
                        if payload == "[DONE]" { break }
                        guard let data = payload.data(using: .utf8) else { continue }
                        let chunk = try decoder.decode(OpenAIChatChunk.self, from: data)
                        if let usage = chunk.usage {
                            continuation.yield(.usage(usage))
                        }
                        guard let choice = chunk.choices.first else { continue }
                        if let reasoning = choice.delta.reasoning_content, !reasoning.isEmpty {
                            continuation.yield(.reasoning(reasoning))
                        }
                        if let content = choice.delta.content, !content.isEmpty {
                            continuation.yield(.content(content))
                        }
                        if choice.finish_reason != nil {
                            continuation.yield(.finished(choice.finish_reason))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func fetchOpenAIModelIDs() async throws -> [String] {
        var request = URLRequest(url: url(path: "v1/models"))
        request.httpMethod = "GET"
        let data = try await data(for: request)
        let list = try decoder.decode(OpenAIModelList.self, from: data)
        return list.data.map(\.id)
    }

    private func fetchAPIModels() async throws -> [APIModelEntry] {
        var request = URLRequest(url: url(path: "api/models"))
        request.httpMethod = "GET"
        let data = try await data(for: request)
        return try decoder.decode([APIModelEntry].self, from: data)
    }

    private func data(for request: URLRequest) async throws -> Data {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw CaixClientError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8)
            throw CaixClientError.httpStatus(http.statusCode, text)
        }
        return data
    }

    private func url(path: String) -> URL {
        baseURL.appendingPathComponent(path)
    }

    private func collectErrorBody(from bytes: URLSession.AsyncBytes) async throws -> String {
        var body = ""
        for try await line in bytes.lines {
            if !body.isEmpty { body += "\n" }
            body += line
            if body.count > 8192 { break }
        }
        return body
    }
}

enum CaixClientError: LocalizedError {
    case invalidResponse
    case httpStatus(Int, String?)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "The server returned an invalid response."
        case .httpStatus(let status, let body):
            let text = body?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let text, !text.isEmpty {
                return "HTTP \(status): \(text)"
            }
            return "HTTP \(status)"
        }
    }
}

struct OutboundChatMessage: Encodable {
    var role: String
    var content: ChatContent
}

enum ChatContent: Encodable {
    case text(String)
    case image(text: String, image: ImageAttachment)

    func encode(to encoder: Encoder) throws {
        switch self {
        case .text(let text):
            var container = encoder.singleValueContainer()
            try container.encode(text)
        case .image(let text, let image):
            var container = encoder.unkeyedContainer()
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                try container.encode(TextContentBlock(text: trimmed))
            }
            try container.encode(ImageContentBlock(url: image.dataURL))
        }
    }
}

private struct TextContentBlock: Encodable {
    var type = "input_text"
    var text: String
}

private struct ImageContentBlock: Encodable {
    struct ImageURL: Encodable {
        var url: String
    }

    var type = "image_url"
    var image_url: ImageURL

    init(url: String) {
        self.image_url = ImageURL(url: url)
    }
}

private struct ChatCompletionRequest: Encodable {
    struct StreamOptions: Encodable {
        var include_usage: Bool
    }

    var model: String
    var messages: [OutboundChatMessage]
    var max_tokens: Int
    var temperature: Double
    var stream: Bool
    var stream_options: StreamOptions
}

private struct OpenAIModelList: Decodable {
    struct Model: Decodable {
        var id: String
    }

    var data: [Model]
}

private struct APIModelEntry: Decodable {
    var name: String
    var status: String?
    var bundle: Bool?
    var mode: String?
    var reasoningSupported: Bool?
    var multimodalSupported: Bool?
}

private struct OpenAIChatChunk: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable {
            var content: String?
            var reasoning_content: String?
        }

        var delta: Delta
        var finish_reason: String?
    }

    var choices: [Choice]
    var usage: ChatUsage?
}
