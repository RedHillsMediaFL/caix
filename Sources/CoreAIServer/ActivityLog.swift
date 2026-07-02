import Foundation

struct ActivityEvent: Codable, Sendable, Equatable {
    var id: String
    var at: Double
    var method: String
    var path: String
    var status: Int
    var latencyMs: Int
    var model: String?
    var summary: String
    var inputTokens: Int?
    var outputTokens: Int?
    var firstTokenMs: Int?
    var loadMs: Int?
    var prefillMs: Int?
    var decodeMs: Int?
    var decodeTokensPerSecond: Double?
}

actor ActivityLog {
    private var events: [ActivityEvent] = []
    private let capacity: Int

    init(capacity: Int = 240) {
        self.capacity = max(1, capacity)
    }

    func record(
        method: String,
        path: String,
        status: Int,
        startedAt: Date,
        model: String? = nil,
        summary: String,
        inputTokens: Int? = nil,
        outputTokens: Int? = nil,
        firstTokenSeconds: Double? = nil,
        loadSeconds: Double? = nil,
        prefillSeconds: Double? = nil,
        decodeSeconds: Double? = nil
    ) {
        let output = max(0, outputTokens ?? 0)
        let decode = decodeSeconds.map { max(0, $0) }
        let event = ActivityEvent(
            id: Self.shortID(),
            at: Date().timeIntervalSince1970,
            method: method,
            path: path,
            status: status,
            latencyMs: max(0, Int(Date().timeIntervalSince(startedAt) * 1000)),
            model: model?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty,
            summary: Self.redact(summary),
            inputTokens: inputTokens,
            outputTokens: outputTokens,
            firstTokenMs: Self.milliseconds(firstTokenSeconds),
            loadMs: Self.milliseconds(loadSeconds),
            prefillMs: Self.milliseconds(prefillSeconds),
            decodeMs: Self.milliseconds(decode),
            decodeTokensPerSecond: decode.flatMap { $0 > 0 ? Double(output) / $0 : nil })
        events.append(event)
        if events.count > capacity {
            events.removeFirst(events.count - capacity)
        }
    }

    func snapshot(limit: Int = 80) -> [ActivityEvent] {
        let count = min(max(limit, 1), events.count)
        return Array(events.suffix(count).reversed())
    }

    static func redact(_ text: String) -> String {
        var out = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let patterns = [
            #"(?i)\b(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9._~+/\-]+=*"#,
            #"(?i)\b((?:[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|AUTHORIZATION|CREDENTIAL)[A-Z0-9_]*)\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s]+)"#,
            #"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/\-]+=*"#,
            #"(?i)([?&](?:access_token|token|signature|x-amz-signature)=)[^&\s]+"#,
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let range = NSRange(out.startIndex..<out.endIndex, in: out)
            out = regex.stringByReplacingMatches(in: out, options: [], range: range, withTemplate: "$1[redacted]")
        }
        if out.count > 180 {
            out = String(out.prefix(177)) + "..."
        }
        return out
    }

    private static func milliseconds(_ seconds: Double?) -> Int? {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return nil }
        return Int((seconds * 1000).rounded())
    }

    private static func shortID() -> String {
        String(UUID().uuidString.prefix(8)).lowercased()
    }
}
