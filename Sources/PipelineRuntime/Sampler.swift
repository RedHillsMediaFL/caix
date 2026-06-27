import Foundation

/// Minimal CPU sampler over Float32 logits: greedy argmax (`temperature == 0`) and
/// temperature sampling with optional top-K / top-P (nucleus) filtering.
///
/// Mirrors the algorithm order of the reference `CompositeSampler`
/// (scale → top-P → top-K → softmax → multinomial) but is self-contained so the runtime
/// core has no Accelerate dependency. Correctness-first; vocab-sized scans are fine for a CLI.
public struct Sampler {
    public var temperature: Double
    public var topK: Int?
    public var topP: Double?

    public init(temperature: Double, topK: Int? = nil, topP: Double? = nil) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }

    /// Sample a token id from `logits`. `rng` advances so callers can make sampling reproducible.
    public func sample(_ logits: [Float], using rng: inout some RandomNumberGenerator) -> Int {
        if temperature <= 0 { return Self.argmax(logits) }

        let invT = Float(1.0 / temperature)
        var scaled = logits
        for i in scaled.indices { scaled[i] *= invT }

        let needsTopK = topK != nil
        let needsTopP = (topP.map { $0 < 1.0 } ?? false)

        if !needsTopK && !needsTopP {
            var probs = scaled
            Self.softmaxInPlace(&probs)
            return Self.multinomial(probs, using: &rng)
        }

        // Restrict to an active subset, then softmax + sample within it.
        var indices = Array(scaled.indices).sorted { scaled[$0] > scaled[$1] }
        if let k = topK, k > 0, k < indices.count {
            indices = Array(indices.prefix(k))
        }
        if let p = topP, p < 1.0, !indices.isEmpty {
            // Cumulative probability cutoff over the (already sorted) candidates.
            let maxLogit = scaled[indices[0]]
            var exps = [Float](repeating: 0, count: indices.count)
            var sum: Float = 0
            for (i, idx) in indices.enumerated() {
                let e = expf(scaled[idx] - maxLogit)
                exps[i] = e
                sum += e
            }
            let invSum = 1.0 / max(sum, .leastNormalMagnitude)
            var cum: Float = 0
            var cutoff = indices.count
            for i in 0..<indices.count {
                cum += exps[i] * invSum
                if cum >= Float(p) {
                    cutoff = i + 1
                    break
                }
            }
            indices = Array(indices.prefix(cutoff))
        }
        guard !indices.isEmpty else { return Self.argmax(logits) }

        var subLogits = indices.map { scaled[$0] }
        Self.softmaxInPlace(&subLogits)
        let local = Self.multinomial(subLogits, using: &rng)
        return indices[local]
    }

    // MARK: - Primitives

    /// Index of the maximum logit (greedy). Returns 0 for an empty input.
    public static func argmax(_ logits: [Float]) -> Int {
        guard var best = logits.first else { return 0 }
        var bestIdx = 0
        for i in 1..<logits.count where logits[i] > best {
            best = logits[i]
            bestIdx = i
        }
        return bestIdx
    }

    /// Numerically stable in-place softmax.
    static func softmaxInPlace(_ values: inout [Float]) {
        guard !values.isEmpty else { return }
        var maxV = values[0]
        for v in values where v > maxV { maxV = v }
        var sum: Float = 0
        for i in values.indices {
            let e = expf(values[i] - maxV)
            values[i] = e
            sum += e
        }
        let invSum = 1.0 / max(sum, .leastNormalMagnitude)
        for i in values.indices { values[i] *= invSum }
    }

    /// Inverse-CDF multinomial sample from a normalized probability vector.
    static func multinomial(_ probs: [Float], using rng: inout some RandomNumberGenerator)
        -> Int
    {
        let r = Float.random(in: 0..<1, using: &rng)
        var cum: Float = 0
        for (i, p) in probs.enumerated() {
            cum += p
            if r < cum { return i }
        }
        return probs.count - 1
    }
}
