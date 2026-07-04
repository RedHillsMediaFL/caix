import Network
import XCTest
import UIKit
@testable import CaixChat

final class CaixChatTests: XCTestCase {
    override func tearDown() {
        MockURLProtocol.handler = nil
        super.tearDown()
    }

    func testEndpointNormalizationAddsDefaultHTTPPortAndStripsV1() {
        let url = ServerEndpoint.normalizedURL(from: "caix.local/v1")

        XCTAssertEqual(url?.scheme, "http")
        XCTAssertEqual(url?.host, "caix.local")
        XCTAssertEqual(url?.port, 1237)
        XCTAssertEqual(url?.path, "")
    }

    func testEndpointNormalizationPreservesHTTPSWithoutForcingPort() {
        let url = ServerEndpoint.normalizedURL(from: "https://mini.tailnet.ts.net/v1")

        XCTAssertEqual(url?.scheme, "https")
        XCTAssertEqual(url?.host, "mini.tailnet.ts.net")
        XCTAssertNil(url?.port)
        XCTAssertEqual(url?.path, "")
    }

    func testModelImageSupportPrefersServerCapabilityFlag() {
        XCTAssertEqual(
            CaixModel.inferredImageSupport(
                multimodalSupported: true,
                mode: "standard",
                name: "plain-coreai"),
            true)
        XCTAssertEqual(
            CaixModel.inferredImageSupport(
                multimodalSupported: false,
                mode: "multimodal_staged",
                name: "vision-coreai"),
            false)
    }

    func testModelImageSupportFallsBackForOlderServers() {
        XCTAssertEqual(
            CaixModel.inferredImageSupport(
                multimodalSupported: nil,
                mode: "multimodal_staged",
                name: "gemma-coreai"),
            true)
        XCTAssertEqual(
            CaixModel.inferredImageSupport(
                multimodalSupported: nil,
                mode: "eagle",
                name: "gemma4-mtp"),
            false)
        XCTAssertEqual(
            CaixModel.inferredImageSupport(
                multimodalSupported: nil,
                mode: nil,
                name: "gemma4-12b-it-coreai"),
            true)
    }

    func testImageContentEncodesOpenAICompatibleBase64Block() throws {
        let image = try ImageAttachment(data: Self.pngData())
        let message = OutboundChatMessage(
            role: "user",
            content: .image(text: "describe this", image: image))

        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["role"] as? String, "user")
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 2)
        XCTAssertEqual(content[0]["type"] as? String, "input_text")
        XCTAssertEqual(content[0]["text"] as? String, "describe this")
        XCTAssertEqual(content[1]["type"] as? String, "image_url")
        let imageURL = try XCTUnwrap(content[1]["image_url"] as? [String: Any])
        XCTAssertTrue((imageURL["url"] as? String)?.hasPrefix("data:image/jpeg;base64,") == true)
    }

    func testFetchModelsMergesOpenAIAndCaixMetadata() async throws {
        MockURLProtocol.handler = { request in
            switch request.url?.path {
            case "/api/models":
                return Self.jsonResponse(for: request, body: """
                [
                  {
                    "name": "gemma-mm",
                    "params": "4B",
                    "status": "available",
                    "bundle": true,
                    "mode": "multimodal_staged",
                    "reasoningSupported": true,
                    "multimodalSupported": true
                  },
                  {
                    "name": "text-only",
                    "params": "2B",
                    "status": "available",
                    "bundle": true,
                    "mode": "staged",
                    "multimodalSupported": false
                  }
                ]
                """)
            case "/v1/models":
                return Self.jsonResponse(for: request, body: """
                {
                  "object": "list",
                  "data": [
                    { "id": "text-only", "object": "model" },
                    { "id": "gemma-mm", "object": "model" }
                  ]
                }
                """)
            default:
                return Self.jsonResponse(for: request, status: 404, body: "{}")
            }
        }

        let client = CaixClient(
            baseURL: try XCTUnwrap(ServerEndpoint.normalizedURL(from: "http://caix.test:1237")),
            protocolClasses: [MockURLProtocol.self])
        let models = try await client.fetchModels()

        XCTAssertEqual(models.map(\.id), ["gemma-mm", "text-only"])
        XCTAssertEqual(models.first { $0.id == "gemma-mm" }?.supportsImages, true)
        XCTAssertEqual(models.first { $0.id == "gemma-mm" }?.supportsReasoning, true)
        XCTAssertEqual(models.first { $0.id == "text-only" }?.supportsImages, false)
    }

    func testFetchModelsFallsBackToCaixAPIWhenOpenAIModelListIsUnavailable() async throws {
        MockURLProtocol.handler = { request in
            switch request.url?.path {
            case "/api/models":
                return Self.jsonResponse(for: request, body: """
                [
                  {
                    "name": "local-only",
                    "params": "2B",
                    "status": "available",
                    "bundle": true,
                    "mode": "staged",
                    "multimodalSupported": false
                  }
                ]
                """)
            case "/v1/models":
                return Self.jsonResponse(for: request, status: 404, body: "{}")
            default:
                return Self.jsonResponse(for: request, status: 404, body: "{}")
            }
        }

        let client = CaixClient(
            baseURL: try XCTUnwrap(ServerEndpoint.normalizedURL(from: "caix.test")),
            protocolClasses: [MockURLProtocol.self])
        let models = try await client.fetchModels()

        XCTAssertEqual(models.map(\.id), ["local-only"])
        XCTAssertEqual(models.first?.supportsImages, false)
    }

    func testNetworkDiscoveryBuildsBoundedLANSubnetCandidates() throws {
        let hosts = NetworkCandidateBuilder.hostCandidates(around: IPv4Interface(
            name: "en0",
            address: try Self.ipv4("192.168.12.24"),
            netmask: try Self.ipv4("255.255.255.0")))

        XCTAssertEqual(hosts.count, 253)
        XCTAssertTrue(hosts.contains("192.168.12.1"))
        XCTAssertTrue(hosts.contains("192.168.12.254"))
        XCTAssertFalse(hosts.contains("192.168.12.24"))
        XCTAssertFalse(hosts.contains("192.168.13.1"))
    }

    func testNetworkDiscoveryTreatsTailnetInterfacesAsBoundedLocalSlices() throws {
        let hosts = NetworkCandidateBuilder.hostCandidates(around: IPv4Interface(
            name: "utun5",
            address: try Self.ipv4("100.101.222.112"),
            netmask: try Self.ipv4("255.192.0.0")))

        XCTAssertEqual(hosts.count, 253)
        XCTAssertTrue(hosts.contains("100.101.222.1"))
        XCTAssertTrue(hosts.contains("100.101.222.254"))
        XCTAssertFalse(hosts.contains("100.101.222.112"))
        XCTAssertFalse(hosts.contains("100.101.223.1"))
    }

    func testActiveDiscoveryProbeFindsReachableCaixServer() async throws {
        let server = try await LocalCaixHTTPServer()
        defer { server.stop() }
        let port = try server.port

        let endpoints = await ActiveNetworkDiscovery.probe(
            urls: [try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))],
            source: .network,
            maxConcurrent: 1)

        XCTAssertEqual(endpoints.count, 1)
        XCTAssertEqual(endpoints.first?.source, .network)
        XCTAssertEqual(endpoints.first?.modelCount, 1)
        XCTAssertEqual(endpoints.first?.isReachable, true)
    }

    func testStreamChatParsesReasoningContentAndUsage() async throws {
        MockURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/v1/chat/completions")
            XCTAssertEqual(request.httpMethod, "POST")
            let body = try Self.requestBodyData(from: request)
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
            XCTAssertEqual(object["model"] as? String, "gemma-mm")
            XCTAssertEqual(object["stream"] as? Bool, true)
            let messages = try XCTUnwrap(object["messages"] as? [[String: Any]])
            XCTAssertEqual(messages.first?["role"] as? String, "user")

            return Self.eventStreamResponse(for: request, body: """
            data: {"choices":[{"delta":{"reasoning_content":"thinking "},"finish_reason":null}]}

            data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}

            data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

            data: [DONE]

            """)
        }

        let client = CaixClient(
            baseURL: try XCTUnwrap(ServerEndpoint.normalizedURL(from: "caix.test")),
            protocolClasses: [MockURLProtocol.self])
        var reasoning = ""
        var content = ""
        var usage: ChatUsage?
        var finished = false

        for try await delta in client.streamChat(
            messages: [OutboundChatMessage(role: "user", content: .text("hi"))],
            model: "gemma-mm")
        {
            switch delta {
            case .reasoning(let text):
                reasoning += text
            case .content(let text):
                content += text
            case .usage(let reportedUsage):
                usage = reportedUsage
            case .finished(let reason):
                finished = reason == "stop"
            }
        }

        XCTAssertEqual(reasoning, "thinking ")
        XCTAssertEqual(content, "hello")
        XCTAssertEqual(usage, ChatUsage(prompt_tokens: 3, completion_tokens: 1, total_tokens: 4))
        XCTAssertTrue(finished)
    }

    private static func pngData() throws -> Data {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 2, height: 2))
        let image = renderer.image { context in
            UIColor.systemBlue.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        }
        return try XCTUnwrap(image.pngData())
    }

    private static func jsonResponse(
        for request: URLRequest,
        status: Int = 200,
        body: String
    ) -> (HTTPURLResponse, Data) {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"])!
        return (response, Data(body.utf8))
    }

    private static func eventStreamResponse(
        for request: URLRequest,
        status: Int = 200,
        body: String
    ) -> (HTTPURLResponse, Data) {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"])!
        return (response, Data(body.utf8))
    }

    private static func requestBodyData(from request: URLRequest) throws -> Data {
        if let body = request.httpBody {
            return body
        }
        let stream = try XCTUnwrap(request.httpBodyStream)
        stream.open()
        defer { stream.close() }

        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let readCount = stream.read(&buffer, maxLength: buffer.count)
            if readCount < 0 {
                throw stream.streamError ?? CaixTestError.unreadableBody
            }
            if readCount == 0 {
                break
            }
            data.append(buffer, count: readCount)
        }
        return data
    }

    private static func ipv4(_ value: String) throws -> UInt32 {
        let parts = value.split(separator: ".").map(String.init)
        XCTAssertEqual(parts.count, 4)
        return try parts.reduce(UInt32(0)) { result, part in
            let octet = try XCTUnwrap(UInt32(part))
            XCTAssertLessThanOrEqual(octet, 255)
            return (result << 8) | octet
        }
    }
}

private final class MockURLProtocol: URLProtocol {
    static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: CaixTestError.missingHandler)
            return
        }

        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

private enum CaixTestError: Error {
    case missingHandler
    case unreadableBody
    case listenerCancelled
    case missingPort
}

private final class LocalCaixHTTPServer {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "caix.chat.tests.http")

    var port: UInt16 {
        get throws {
            guard let port = listener.port else {
                throw CaixTestError.missingPort
            }
            return port.rawValue
        }
    }

    init() async throws {
        let listener = try NWListener(using: .tcp, on: .any)
        self.listener = listener

        try await withCheckedThrowingContinuation { continuation in
            let resumeGate = ResumeGate()
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    guard resumeGate.claim() else { return }
                    continuation.resume()
                case .failed(let error):
                    guard resumeGate.claim() else { return }
                    continuation.resume(throwing: error)
                case .cancelled:
                    guard resumeGate.claim() else { return }
                    continuation.resume(throwing: CaixTestError.listenerCancelled)
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                self?.handle(connection)
            }
            listener.start(queue: queue)
        }
    }

    func stop() {
        listener.cancel()
    }

    private func handle(_ connection: NWConnection) {
        connection.start(queue: queue)
        receiveRequest(from: connection, buffer: Data())
    }

    private func receiveRequest(from connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, isComplete, _ in
            guard let self else { return }
            var nextBuffer = buffer
            if let data {
                nextBuffer.append(data)
            }
            if nextBuffer.range(of: Data("\r\n\r\n".utf8)) != nil || isComplete {
                self.respond(to: connection, requestData: nextBuffer)
            } else {
                self.receiveRequest(from: connection, buffer: nextBuffer)
            }
        }
    }

    private func respond(to connection: NWConnection, requestData: Data) {
        let request = String(decoding: requestData, as: UTF8.self)
        let path = request.split(separator: " ").dropFirst().first.map(String.init) ?? "/"
        let body: String
        let status: String
        switch path {
        case "/api/models":
            status = "200 OK"
            body = """
            [
              {
                "name": "local-probe",
                "params": "1B",
                "status": "available",
                "bundle": true,
                "mode": "staged",
                "multimodalSupported": false
              }
            ]
            """
        case "/v1/models":
            status = "200 OK"
            body = """
            {
              "object": "list",
              "data": [
                { "id": "local-probe", "object": "model" }
              ]
            }
            """
        default:
            status = "404 Not Found"
            body = "{}"
        }

        let response = """
        HTTP/1.1 \(status)\r
        Content-Type: application/json\r
        Content-Length: \(Data(body.utf8).count)\r
        Connection: close\r
        \r
        \(body)
        """
        connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}

private final class ResumeGate: @unchecked Sendable {
    private let lock = NSLock()
    private var didResume = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !didResume else { return false }
        didResume = true
        return true
    }
}
