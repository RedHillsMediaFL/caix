import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Minimal client for a streamable-HTTP MCP server: does the `initialize` handshake, then the
/// requested JSON-RPC method (`tools/list` / `tools/call`). Stateless per call — simple, and fine
/// for the chat view's tool use. Handles both `application/json` and `text/event-stream` responses.
enum MCPProxy {
    struct MCPError: Error, CustomStringConvertible { let message: String; var description: String { message } }

    static func call(url: URL, method: String, params: [String: Any]?) async throws -> Any {
        // 1. initialize (negotiate + obtain a session id if the server is stateful)
        let initParams: [String: Any] = [
            "protocolVersion": "2025-06-18",
            "capabilities": [:],
            "clientInfo": ["name": "caix", "version": "0.2.8-beta"],
        ]
        let (_, session) = try await rpc(url: url, method: "initialize", params: initParams, id: 1, session: nil)
        // 2. initialized notification (best-effort, no response)
        try? await notify(url: url, method: "notifications/initialized", session: session)
        // 3. the requested method
        let (result, _) = try await rpc(url: url, method: method, params: params, id: 2, session: session)
        return result
    }

    private static func makeRequest(url: URL, session: String?) -> URLRequest {
        var req = URLRequest(url: url, timeoutInterval: 30)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        if let s = session { req.setValue(s, forHTTPHeaderField: "Mcp-Session-Id") }
        return req
    }

    private static func rpc(url: URL, method: String, params: [String: Any]?, id: Int, session: String?)
        async throws -> (Any, String?)
    {
        var req = makeRequest(url: url, session: session)
        var body: [String: Any] = ["jsonrpc": "2.0", "method": method, "id": id]
        if let p = params { body["params"] = p }
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        let http = resp as? HTTPURLResponse
        let sid = http?.value(forHTTPHeaderField: "Mcp-Session-Id")
            ?? http?.value(forHTTPHeaderField: "mcp-session-id") ?? session
        let ct = http?.value(forHTTPHeaderField: "Content-Type") ?? ""
        let json = try parse(data: data, contentType: ct)
        if let err = json["error"] { throw MCPError(message: "MCP error: \(err)") }
        return (json["result"] ?? [String: Any](), sid)
    }

    private static func notify(url: URL, method: String, session: String?) async throws {
        var req = makeRequest(url: url, session: session)
        req.httpBody = try JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "method": method])
        _ = try? await URLSession.shared.data(for: req)
    }

    /// Parse a JSON-RPC response from either a JSON body or an SSE stream (last `data:` JSON line).
    private static func parse(data: Data, contentType: String) throws -> [String: Any] {
        if contentType.contains("text/event-stream") {
            let text = String(data: data, encoding: .utf8) ?? ""
            for line in text.split(separator: "\n").reversed() where line.hasPrefix("data:") {
                let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                if let d = payload.data(using: .utf8),
                   let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] { return o }
            }
            throw MCPError(message: "no JSON in SSE response")
        }
        guard let o = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MCPError(message: "non-JSON MCP response")
        }
        return o
    }
}
