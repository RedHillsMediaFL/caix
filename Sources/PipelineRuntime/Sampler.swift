import Foundation

/// Minimal CPU sampler over Float32 logits: greedy argmax (`temperature == 0`) and
/// temperature sampling with optional top-K / top-P (nucleus) filtering.
///
/// Self-contained so the runtime core has no Accelerate dependency. Hot paths avoid full-vocab
/// copies and full sorts unless top-P over the whole vocabulary requires it.
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
        guard !logits.isEmpty else { return 0 }
        if temperature <= 0 { return Self.argmax(logits) }

        let invT = Float(1.0 / temperature)
        let needsTopK = topK.map { $0 > 0 && $0 < logits.count } ?? false
        let needsTopP = (topP.map { $0 < 1.0 } ?? false)

        if !needsTopK && !needsTopP {
            return Self.sampleAll(logits, invT: invT, using: &rng)
        }

        var candidates: [Candidate]
        if needsTopK, let k = topK {
            candidates = Self.topKCandidates(logits, k: k)
        } else {
            candidates = logits.indices.map { Candidate(index: $0, logit: logits[$0]) }
        }
        if let p = topP, p < 1.0 {
            candidates.sort { $0.logit > $1.logit }
            candidates = Self.topPCutoff(candidates, invT: invT, p: Float(p))
        }
        guard !candidates.isEmpty else { return Self.argmax(logits) }
        return Self.sampleCandidates(candidates, invT: invT, using: &rng)
    }

    // MARK: - Primitives

    /// Index of the maximum logit (greedy). Returns 0 for an empty input.
    public static func argmax(_ logits: [Float]) -> Int {
        logits.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress, buffer.count > 0 else { return 0 }
            var best = base.pointee
            var bestIdx = 0
            var i = 1
            while i < buffer.count {
                let value = base.advanced(by: i).pointee
                if value > best {
                    best = value
                    bestIdx = i
                }
                i += 1
            }
            return bestIdx
        }
    }

    public static func topK(_ logits: [Float], count: Int) -> [(index: Int, logit: Float)] {
        guard count > 0, !logits.isEmpty else { return [] }
        let candidates: [Candidate]
        if count < logits.count {
            candidates = topKCandidates(logits, k: count)
        } else {
            candidates = logits.indices.map { Candidate(index: $0, logit: logits[$0]) }
        }
        return candidates
            .sorted {
                if $0.logit == $1.logit { return $0.index < $1.index }
                return $0.logit > $1.logit
            }
            .map { (index: $0.index, logit: $0.logit) }
    }

    private struct Candidate {
        let index: Int
        let logit: Float
    }

    private static func topKCandidates(_ logits: [Float], k: Int) -> [Candidate] {
        precondition(k > 0)
        var heap: [Candidate] = []
        heap.reserveCapacity(k)

        logits.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            var idx = 0
            while idx < buffer.count {
                let logit = base.advanced(by: idx).pointee
                let candidate = Candidate(index: idx, logit: logit)
                if heap.count < k {
                    heap.append(candidate)
                    siftUpMinHeap(&heap, heap.count - 1)
                    idx += 1
                    continue
                }
                if logit > heap[0].logit {
                    heap[0] = candidate
                    siftDownMinHeap(&heap, 0)
                }
                idx += 1
            }
        }
        return heap
    }

    private static func siftUpMinHeap(_ heap: inout [Candidate], _ start: Int) {
        var child = start
        while child > 0 {
            let parent = (child - 1) / 2
            if heap[parent].logit <= heap[child].logit { break }
            heap.swapAt(parent, child)
            child = parent
        }
    }

    private static func siftDownMinHeap(_ heap: inout [Candidate], _ start: Int) {
        var parent = start
        while true {
            let left = parent * 2 + 1
            if left >= heap.count { return }
            let right = left + 1
            var smallest = left
            if right < heap.count && heap[right].logit < heap[left].logit {
                smallest = right
            }
            if heap[parent].logit <= heap[smallest].logit { return }
            heap.swapAt(parent, smallest)
            parent = smallest
        }
    }

    private static func topPCutoff(_ sortedCandidates: [Candidate], invT: Float, p: Float)
        -> [Candidate]
    {
        guard let first = sortedCandidates.first else { return [] }
        let maxScaled = first.logit * invT
        var exps = [Float](repeating: 0, count: sortedCandidates.count)
        var sum: Float = 0
        for i in sortedCandidates.indices {
            let e = expf(sortedCandidates[i].logit * invT - maxScaled)
            exps[i] = e
            sum += e
        }
        let invSum = 1.0 / max(sum, .leastNormalMagnitude)
        var cum: Float = 0
        var cutoff = sortedCandidates.count
        for i in sortedCandidates.indices {
            cum += exps[i] * invSum
            if cum >= p {
                cutoff = i + 1
                break
            }
        }
        return Array(sortedCandidates.prefix(cutoff))
    }

    private static func sampleAll(
        _ logits: [Float], invT: Float, using rng: inout some RandomNumberGenerator
    ) -> Int {
        logits.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress, buffer.count > 0 else { return 0 }
            var maxLogit = base.pointee
            var i = 1
            while i < buffer.count {
                let value = base.advanced(by: i).pointee
                if value > maxLogit { maxLogit = value }
                i += 1
            }

            let maxScaled = maxLogit * invT
            var sum: Float = 0
            i = 0
            while i < buffer.count {
                sum += expf(base.advanced(by: i).pointee * invT - maxScaled)
                i += 1
            }

            let target = Float.random(in: 0..<1, using: &rng) * max(sum, .leastNormalMagnitude)
            var cum: Float = 0
            i = 0
            while i < buffer.count {
                cum += expf(base.advanced(by: i).pointee * invT - maxScaled)
                if target < cum { return i }
                i += 1
            }
            return buffer.count - 1
        }
    }

    private static func sampleCandidates(
        _ candidates: [Candidate], invT: Float, using rng: inout some RandomNumberGenerator
    ) -> Int {
        candidates.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress, buffer.count > 0 else { return 0 }
            var maxLogit = base.pointee.logit
            var i = 1
            while i < buffer.count {
                let value = base.advanced(by: i).pointee.logit
                if value > maxLogit { maxLogit = value }
                i += 1
            }

            let maxScaled = maxLogit * invT
            var sum: Float = 0
            i = 0
            while i < buffer.count {
                sum += expf(base.advanced(by: i).pointee.logit * invT - maxScaled)
                i += 1
            }

            let target = Float.random(in: 0..<1, using: &rng) * max(sum, .leastNormalMagnitude)
            var cum: Float = 0
            i = 0
            while i < buffer.count {
                let candidate = base.advanced(by: i).pointee
                cum += expf(candidate.logit * invT - maxScaled)
                if target < cum { return candidate.index }
                i += 1
            }
            return base.advanced(by: buffer.count - 1).pointee.index
        }
    }
}
