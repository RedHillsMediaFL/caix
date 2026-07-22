import Foundation

/// Minimal manifest opt-in for staged Gemma EAGLE/MTP target outputs.
///
/// Tensor names are deliberately canonical rather than configurable. This keeps malformed or
/// partially converted targets from being accepted under an ambiguous manifest contract.
public struct DistributedEagleTargetContract: Codable, Hashable, Sendable {
    public let stageID: String
    public let slidingWindow: Int

    enum CodingKeys: String, CodingKey {
        case stageID = "stage_id"
        case slidingWindow = "sliding_window"
    }

    public init(stageID: String, slidingWindow: Int) {
        self.stageID = stageID
        self.slidingWindow = slidingWindow
    }

    func validate(stages: [DistributedStageManifestStage]) throws {
        guard !stageID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target stage_id must be non-empty")
        }
        guard slidingWindow > 0 else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target sliding_window must be positive")
        }
        guard let finalTransformer = stages.last(where: { $0.role == .transformerLayers }),
            finalTransformer.id == stageID
        else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target stage_id must name the final transformer_layers stage")
        }
    }

    func producesArtifacts(for stageID: String) -> Bool {
        self.stageID == stageID
    }
}

/// One compact local-only Float16 tensor emitted by a staged EAGLE target.
///
/// This type intentionally has no Codable conformance. Phase 1 does not define a remote wire
/// format for these potentially large tensors, so cluster paths must reject them instead of
/// silently copying or dropping them.
public struct DistributedEagleTargetTensor: Hashable, Sendable {
    public let shape: [Int]
    public let scalarType: DistributedStageIOScalarType
    public let float16BitPatterns: [UInt16]

    public init(
        shape: [Int],
        scalarType: DistributedStageIOScalarType,
        float16BitPatterns: [UInt16]
    ) throws {
        guard scalarType == .float16 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target tensor scalar_type must be float16")
        }
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target tensor shape \(shape) must be positive")
        }
        var elementCount = 1
        for dimension in shape {
            let (product, overflow) = elementCount.multipliedReportingOverflow(by: dimension)
            guard !overflow else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "EAGLE target tensor shape \(shape) overflows element count")
            }
            elementCount = product
        }
        guard float16BitPatterns.count == elementCount else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target tensor element count \(float16BitPatterns.count) does not match shape \(shape)")
        }
        self.shape = shape
        self.scalarType = scalarType
        self.float16BitPatterns = float16BitPatterns
    }
}

/// Local target state consumed by the future sequential EAGLE/MTP loop.
///
/// Stage handles emit one chunk whose full/sliding ranges are identical. The host accumulator
/// returns a snapshot with the complete full-attention prefix and the cropped sliding range.
public struct DistributedEagleTargetArtifacts: Hashable, Sendable {
    public let finalHidden: DistributedEagleTargetTensor
    public let fullKey: DistributedEagleTargetTensor
    public let fullValue: DistributedEagleTargetTensor
    public let slidingKey: DistributedEagleTargetTensor
    public let slidingValue: DistributedEagleTargetTensor
    public let fullPositionRange: DistributedSequenceRange
    public let slidingPositionRange: DistributedSequenceRange

    public init(
        finalHidden: DistributedEagleTargetTensor,
        fullKey: DistributedEagleTargetTensor,
        fullValue: DistributedEagleTargetTensor,
        slidingKey: DistributedEagleTargetTensor,
        slidingValue: DistributedEagleTargetTensor,
        fullPositionRange: DistributedSequenceRange,
        slidingPositionRange: DistributedSequenceRange
    ) throws {
        guard finalHidden.shape.count == 3,
            finalHidden.shape[0] == 1,
            finalHidden.shape[1] == 1
        else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target final hidden shape \(finalHidden.shape) must match [1, 1, hidden]")
        }
        try Self.validateKVPair(
            key: fullKey,
            value: fullValue,
            positionRange: fullPositionRange,
            label: "full")
        try Self.validateKVPair(
            key: slidingKey,
            value: slidingValue,
            positionRange: slidingPositionRange,
            label: "sliding")
        guard fullPositionRange.upperBound == slidingPositionRange.upperBound else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target full and sliding ranges must end at the same position")
        }
        self.finalHidden = finalHidden
        self.fullKey = fullKey
        self.fullValue = fullValue
        self.slidingKey = slidingKey
        self.slidingValue = slidingValue
        self.fullPositionRange = fullPositionRange
        self.slidingPositionRange = slidingPositionRange
    }

    private static func validateKVPair(
        key: DistributedEagleTargetTensor,
        value: DistributedEagleTargetTensor,
        positionRange: DistributedSequenceRange,
        label: String
    ) throws {
        guard positionRange.isValid else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) position range is invalid")
        }
        guard key.shape.count == 4, key.shape[0] == 1 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) KV shape \(key.shape) must match [1, heads, sequence, head_dim]")
        }
        guard key.shape == value.shape else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) key/value shapes must match")
        }
        guard key.shape[2] == positionRange.count else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) KV sequence does not match its position range")
        }
    }
}

/// Per-request, capacity-bounded host accumulation for streamed staged target outputs.
public struct DistributedEagleTargetKVAccumulator: Sendable {
    public let kvCapacity: Int
    public let slidingWindow: Int
    public private(set) var snapshot: DistributedEagleTargetArtifacts?

    public init(kvCapacity: Int, slidingWindow: Int) throws {
        guard kvCapacity > 0 else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "EAGLE target kv_capacity must be positive")
        }
        guard slidingWindow > 0 else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "EAGLE target sliding_window must be positive")
        }
        self.kvCapacity = kvCapacity
        self.slidingWindow = slidingWindow
        self.snapshot = nil
    }

    public mutating func append(_ chunk: DistributedEagleTargetArtifacts) throws {
        let processed = snapshot?.fullPositionRange.upperBound ?? 0
        guard chunk.fullPositionRange == chunk.slidingPositionRange else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target streamed chunk full and sliding ranges must match")
        }
        guard chunk.fullPositionRange.lowerBound == processed else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target chunk lower_bound \(chunk.fullPositionRange.lowerBound) does not match accumulated length \(processed)")
        }
        let nextProcessed = chunk.fullPositionRange.upperBound
        guard nextProcessed <= kvCapacity else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target accumulated length \(nextProcessed) exceeds kv_capacity \(kvCapacity)")
        }

        let fullKey = try Self.concatenating(snapshot?.fullKey, chunk.fullKey, label: "full key")
        let fullValue = try Self.concatenating(snapshot?.fullValue, chunk.fullValue, label: "full value")
        let uncroppedSlidingKey = try Self.concatenating(
            snapshot?.slidingKey, chunk.slidingKey, label: "sliding key")
        let uncroppedSlidingValue = try Self.concatenating(
            snapshot?.slidingValue, chunk.slidingValue, label: "sliding value")
        let slidingLength = min(nextProcessed, slidingWindow)
        let slidingKey = try Self.suffix(
            uncroppedSlidingKey, sequenceCount: slidingLength, label: "sliding key")
        let slidingValue = try Self.suffix(
            uncroppedSlidingValue, sequenceCount: slidingLength, label: "sliding value")

        snapshot = try DistributedEagleTargetArtifacts(
            finalHidden: chunk.finalHidden,
            fullKey: fullKey,
            fullValue: fullValue,
            slidingKey: slidingKey,
            slidingValue: slidingValue,
            fullPositionRange: DistributedSequenceRange(
                lowerBound: 0,
                upperBound: nextProcessed),
            slidingPositionRange: DistributedSequenceRange(
                lowerBound: nextProcessed - slidingLength,
                upperBound: nextProcessed))
    }

    public mutating func reset() {
        snapshot = nil
    }

    private static func concatenating(
        _ prefix: DistributedEagleTargetTensor?,
        _ suffix: DistributedEagleTargetTensor,
        label: String
    ) throws -> DistributedEagleTargetTensor {
        guard let prefix else { return suffix }
        guard prefix.shape.count == 4, suffix.shape.count == 4,
            prefix.shape[0] == suffix.shape[0],
            prefix.shape[1] == suffix.shape[1],
            prefix.shape[3] == suffix.shape[3]
        else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) layout changed between streamed chunks")
        }
        let heads = prefix.shape[1]
        let prefixSequence = prefix.shape[2]
        let suffixSequence = suffix.shape[2]
        let headDimension = prefix.shape[3]
        var values: [UInt16] = []
        values.reserveCapacity(heads * (prefixSequence + suffixSequence) * headDimension)
        for head in 0..<heads {
            let prefixStart = head * prefixSequence * headDimension
            values.append(contentsOf: prefix.float16BitPatterns[
                prefixStart..<(prefixStart + prefixSequence * headDimension)])
            let suffixStart = head * suffixSequence * headDimension
            values.append(contentsOf: suffix.float16BitPatterns[
                suffixStart..<(suffixStart + suffixSequence * headDimension)])
        }
        return try DistributedEagleTargetTensor(
            shape: [1, heads, prefixSequence + suffixSequence, headDimension],
            scalarType: .float16,
            float16BitPatterns: values)
    }

    private static func suffix(
        _ tensor: DistributedEagleTargetTensor,
        sequenceCount: Int,
        label: String
    ) throws -> DistributedEagleTargetTensor {
        guard tensor.shape.count == 4, sequenceCount > 0, sequenceCount <= tensor.shape[2] else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) cannot retain \(sequenceCount) positions from shape \(tensor.shape)")
        }
        guard sequenceCount < tensor.shape[2] else { return tensor }
        let heads = tensor.shape[1]
        let sourceSequence = tensor.shape[2]
        let headDimension = tensor.shape[3]
        let firstSequence = sourceSequence - sequenceCount
        var values: [UInt16] = []
        values.reserveCapacity(heads * sequenceCount * headDimension)
        for head in 0..<heads {
            let start = (head * sourceSequence + firstSequence) * headDimension
            values.append(contentsOf: tensor.float16BitPatterns[
                start..<(start + sequenceCount * headDimension)])
        }
        return try DistributedEagleTargetTensor(
            shape: [1, heads, sequenceCount, headDimension],
            scalarType: .float16,
            float16BitPatterns: values)
    }
}
