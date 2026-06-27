import Foundation

/// Last-generation speculative-decoding metrics, published for the dashboard (`GET /api/genstats`).
/// Written by `EagleEngine.generate` after each completion; read by the server. Thread-safe via a
/// lock (generation runs off the main actor).
public struct SpeculativeStats: Codable, Sendable {
    public var model: String
    public var tokensPerSecond: Double
    public var acceptanceRate: Double      // 0..1
    public var tokensPerPass: Double       // mean committed tokens per target forward
    public var draftTokens: Int            // K
    public var generatedTokens: Int
    public var promptTokens: Int
    public var decodeSeconds: Double
    public var prefillSeconds: Double
    public var at: Double                  // unix epoch seconds

    public init(model: String, tokensPerSecond: Double, acceptanceRate: Double, tokensPerPass: Double,
                draftTokens: Int, generatedTokens: Int, promptTokens: Int, decodeSeconds: Double,
                prefillSeconds: Double, at: Double) {
        self.model = model; self.tokensPerSecond = tokensPerSecond; self.acceptanceRate = acceptanceRate
        self.tokensPerPass = tokensPerPass; self.draftTokens = draftTokens
        self.generatedTokens = generatedTokens; self.promptTokens = promptTokens
        self.decodeSeconds = decodeSeconds; self.prefillSeconds = prefillSeconds; self.at = at
    }
}

public enum LiveStats {
    nonisolated(unsafe) private static var _last: SpeculativeStats? = nil
    private static let lock = NSLock()

    public static func record(_ s: SpeculativeStats) {
        lock.lock(); _last = s; lock.unlock()
    }
    public static var last: SpeculativeStats? {
        lock.lock(); defer { lock.unlock() }; return _last
    }
}

// MARK: - Cumulative usage analytics (ollama/omlx-style: totals, rolling tok/s, per-model)

public struct ModelUsage: Codable, Sendable {
    public var model: String
    public var requests: Int
    public var inputTokens: Int
    public var outputTokens: Int
    public var avgTokensPerSecond: Double
}

public struct UsageSnapshot: Codable, Sendable {
    public var totalRequests: Int
    public var totalInputTokens: Int
    public var totalOutputTokens: Int
    public var avgTokensPerSecond: Double        // lifetime, output-weighted
    public var rollingTokensPerSecond: Double    // last `rollingWindowMinutes`
    public var rollingWindowMinutes: Int
    public var rollingOutputTokens: Int
    public var uptimeSeconds: Double
    public var lastModel: String
    public var byModel: [ModelUsage]
}

/// Process-global usage accumulator. `record` is called once per completed generation (any model,
/// any API). Lock-based + non-blocking; safe off the main actor. Persisted to disk (configurable
/// path) so totals + per-model stats survive server restarts, ollama/omlx-style.
public enum Usage {
    struct Rec: Codable { var model: String; var output: Int; var decode: Double; var at: Double }
    struct Agg: Codable { var requests = 0; var input = 0; var output = 0; var decode = 0.0 }
    private struct Snapshot: Codable {
        var startEpoch: Double; var totalRequests: Int; var totalInput: Int; var totalOutput: Int
        var totalDecode: Double; var lastModel: String; var perModel: [String: Agg]; var ring: [Rec]
    }

    nonisolated(unsafe) private static var startEpoch: Double = 0
    nonisolated(unsafe) private static var totalRequests = 0
    nonisolated(unsafe) private static var totalInput = 0
    nonisolated(unsafe) private static var totalOutput = 0
    nonisolated(unsafe) private static var totalDecode = 0.0
    nonisolated(unsafe) private static var lastModel = ""
    nonisolated(unsafe) private static var perModel: [String: Agg] = [:]
    nonisolated(unsafe) private static var ring: [Rec] = []   // recent gens for the rolling window
    nonisolated(unsafe) private static var persistPath: String? = nil
    private static let lock = NSLock()
    private static let windowMinutes = 10

    /// Point the accumulator at a JSON file and load any prior totals. Call once at server start.
    public static func configure(path: String) {
        lock.lock(); defer { lock.unlock() }
        persistPath = path
        guard let data = FileManager.default.contents(atPath: path),
              let s = try? JSONDecoder().decode(Snapshot.self, from: data) else { return }
        startEpoch = s.startEpoch; totalRequests = s.totalRequests; totalInput = s.totalInput
        totalOutput = s.totalOutput; totalDecode = s.totalDecode; lastModel = s.lastModel
        perModel = s.perModel; ring = s.ring
    }

    private static func saveLocked() {
        guard let path = persistPath else { return }
        let snap = Snapshot(startEpoch: startEpoch, totalRequests: totalRequests, totalInput: totalInput,
                            totalOutput: totalOutput, totalDecode: totalDecode, lastModel: lastModel,
                            perModel: perModel, ring: ring)
        guard let data = try? JSONEncoder().encode(snap) else { return }
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    public static func record(model: String, inputTokens: Int, outputTokens: Int,
                              decodeSeconds: Double, at: Double) {
        lock.lock(); defer { lock.unlock() }
        // Lifetime startEpoch persists across restarts; only set it the very first time ever.
        if startEpoch == 0 { startEpoch = at }
        totalRequests += 1; totalInput += inputTokens; totalOutput += outputTokens
        totalDecode += max(0, decodeSeconds); lastModel = model
        var a = perModel[model] ?? Agg()
        a.requests += 1; a.input += inputTokens; a.output += outputTokens; a.decode += max(0, decodeSeconds)
        perModel[model] = a
        ring.append(Rec(model: model, output: outputTokens, decode: max(0, decodeSeconds), at: at))
        let cutoff = at - Double(windowMinutes) * 60 - 1
        if ring.count > 2000 || (ring.first?.at ?? at) < cutoff {
            ring = ring.filter { $0.at >= cutoff }
        }
        // Persist every record — the file is tiny (a few hundred bytes) and generations are
        // seconds apart, so there's no point throttling and risking a lost update on restart.
        saveLocked()
    }

    public static func snapshot(now: Double) -> UsageSnapshot {
        lock.lock(); defer { lock.unlock() }
        let lifeTps = totalDecode > 0 ? Double(totalOutput) / totalDecode : 0
        let cutoff = now - Double(windowMinutes) * 60
        var rollOut = 0; var rollDec = 0.0
        for r in ring where r.at >= cutoff { rollOut += r.output; rollDec += r.decode }
        let rollTps = rollDec > 0 ? Double(rollOut) / rollDec : 0
        let models = perModel.map { (k, v) in
            ModelUsage(model: k, requests: v.requests, inputTokens: v.input, outputTokens: v.output,
                       avgTokensPerSecond: v.decode > 0 ? Double(v.output) / v.decode : 0)
        }.sorted { $0.requests > $1.requests }
        return UsageSnapshot(
            totalRequests: totalRequests, totalInputTokens: totalInput, totalOutputTokens: totalOutput,
            avgTokensPerSecond: lifeTps, rollingTokensPerSecond: rollTps,
            rollingWindowMinutes: windowMinutes, rollingOutputTokens: rollOut,
            uptimeSeconds: startEpoch > 0 ? now - startEpoch : 0, lastModel: lastModel, byModel: models)
    }
}
