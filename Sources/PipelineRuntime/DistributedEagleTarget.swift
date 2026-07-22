import Foundation

/// Minimal manifest opt-in for staged Gemma EAGLE/MTP target outputs.
///
/// Tensor names are deliberately canonical rather than configurable. This keeps malformed or
/// partially converted targets from being accepted under an ambiguous manifest contract.
public struct DistributedEagleTargetContract: Codable, Hashable, Sendable {
    public let stageID: String
    public let slidingWindow: Int
    public let finalHiddenStageID: String
    public let finalHiddenTensorName: String

    enum CodingKeys: String, CodingKey {
        case stageID = "stage_id"
        case slidingWindow = "sliding_window"
        case finalHiddenStageID = "final_hidden_stage_id"
        case finalHiddenTensorName = "final_hidden_tensor_name"
    }

    public init(
        stageID: String,
        slidingWindow: Int,
        finalHiddenStageID: String = "head",
        finalHiddenTensorName: String = "hidden"
    ) {
        self.stageID = stageID
        self.slidingWindow = slidingWindow
        self.finalHiddenStageID = finalHiddenStageID
        self.finalHiddenTensorName = finalHiddenTensorName
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            stageID: try container.decode(String.self, forKey: .stageID),
            slidingWindow: try container.decode(Int.self, forKey: .slidingWindow),
            finalHiddenStageID: try container.decodeIfPresent(
                String.self, forKey: .finalHiddenStageID) ?? "head",
            finalHiddenTensorName: try container.decodeIfPresent(
                String.self, forKey: .finalHiddenTensorName) ?? "hidden")
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
        guard let finalHead = stages.last(where: { $0.role == .finalNormHead }),
            finalHead.id == finalHiddenStageID
        else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target final_hidden_stage_id must name the final_norm_head stage")
        }
        guard finalHiddenTensorName == DistributedStageIOTensorName.hidden.rawValue else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target final_hidden_tensor_name must be hidden")
        }
    }

    func producesArtifacts(for stageID: String) -> Bool {
        self.stageID == stageID
    }

    func producesFinalHidden(for stageID: String) -> Bool {
        finalHiddenStageID == stageID
    }
}

struct RawDistributedEagleTargetContract: Decodable {
    let stageID: String?
    let slidingWindow: Int?
    let finalHiddenStageID: String?
    let finalHiddenTensorName: String?
    let outputs: OutputMappings?
    let representativeKV: RepresentativeKV?

    enum CodingKeys: String, CodingKey {
        case stageID = "stage_id"
        case slidingWindow = "sliding_window"
        case finalHiddenStageID = "final_hidden_stage_id"
        case finalHiddenTensorName = "final_hidden_tensor_name"
        case outputs
        case representativeKV = "representative_kv"
    }

    struct OutputMappings: Decodable {
        let backboneHidden: Producer?

        enum CodingKeys: String, CodingKey {
            case backboneHidden = "backbone_hidden"
        }
    }

    struct Producer: Decodable {
        let stage: String?
        let tensor: String?
    }

    struct RepresentativeKV: Decodable {
        let fullAttention: KVProducer?
        let slidingAttention: KVProducer?

        enum CodingKeys: String, CodingKey {
            case fullAttention = "full_attention"
            case slidingAttention = "sliding_attention"
        }
    }

    struct KVProducer: Decodable {
        let stage: String?
        let keyOutput: String?
        let valueOutput: String?

        enum CodingKeys: String, CodingKey {
            case stage
            case keyOutput = "key_output"
            case valueOutput = "value_output"
        }
    }

    func normalized(
        stages: [DistributedStageManifestStage],
        cacheGroups: DistributedStageCacheGroups?
    ) throws -> DistributedEagleTargetContract {
        let fullProducer = representativeKV?.fullAttention
        let slidingProducer = representativeKV?.slidingAttention
        if representativeKV != nil {
            guard let fullProducer, let slidingProducer else {
                throw DistributedStageManifestError.invalidManifest(
                    "eagle_target representative_kv must declare full_attention and sliding_attention")
            }
            guard Self.firstNonEmpty(fullProducer.stage) != nil,
                Self.firstNonEmpty(slidingProducer.stage) != nil
            else {
                throw DistributedStageManifestError.invalidManifest(
                    "eagle_target representative_kv producer stages must be non-empty")
            }
            guard fullProducer.stage == slidingProducer.stage else {
                throw DistributedStageManifestError.invalidManifest(
                    "eagle_target representative_kv producers must name the same stage")
            }
            guard fullProducer.keyOutput == DistributedStageIOTensorName.kFull.rawValue,
                fullProducer.valueOutput == DistributedStageIOTensorName.vFull.rawValue,
                slidingProducer.keyOutput == DistributedStageIOTensorName.kSliding.rawValue,
                slidingProducer.valueOutput == DistributedStageIOTensorName.vSliding.rawValue
            else {
                throw DistributedStageManifestError.invalidManifest(
                    "eagle_target representative_kv outputs must be k_full/v_full and k_sliding/v_sliding")
            }
        }

        let derivedStageID = fullProducer?.stage
        if let stageID, let derivedStageID, stageID != derivedStageID {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target stage_id conflicts with representative_kv producer stage")
        }
        guard let resolvedStageID = Self.firstNonEmpty(stageID, derivedStageID) else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target stage_id is missing and representative_kv producer stage is unavailable")
        }

        let derivedSlidingWindow = cacheGroups?.groups["sliding"]?.slidingWindow
        if let slidingWindow, let derivedSlidingWindow, slidingWindow != derivedSlidingWindow {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target sliding_window conflicts with cache_groups sliding window")
        }
        guard let resolvedSlidingWindow = slidingWindow ?? derivedSlidingWindow else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target sliding_window is missing and cache_groups sliding window is unavailable")
        }

        let hiddenProducer = outputs?.backboneHidden
        if outputs != nil, hiddenProducer == nil {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target outputs must declare backbone_hidden")
        }
        if let hiddenProducer {
            guard Self.firstNonEmpty(hiddenProducer.stage) != nil,
                Self.firstNonEmpty(hiddenProducer.tensor) != nil
            else {
                throw DistributedStageManifestError.invalidManifest(
                    "eagle_target outputs.backbone_hidden must declare stage and tensor")
            }
        }
        if let finalHiddenStageID, let mappedStage = hiddenProducer?.stage,
            finalHiddenStageID != mappedStage
        {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target final_hidden_stage_id conflicts with outputs.backbone_hidden stage")
        }
        if let finalHiddenTensorName, let mappedTensor = hiddenProducer?.tensor,
            finalHiddenTensorName != mappedTensor
        {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target final_hidden_tensor_name conflicts with outputs.backbone_hidden tensor")
        }
        let defaultHeadID = stages.last(where: { $0.role == .finalNormHead })?.id
        guard let resolvedHiddenStageID = Self.firstNonEmpty(
            finalHiddenStageID, hiddenProducer?.stage, defaultHeadID)
        else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target final hidden producer stage is unavailable")
        }
        guard let resolvedHiddenTensorName = Self.firstNonEmpty(
            finalHiddenTensorName, hiddenProducer?.tensor, DistributedStageIOTensorName.hidden.rawValue)
        else {
            throw DistributedStageManifestError.invalidManifest(
                "eagle_target final hidden tensor is unavailable")
        }

        return DistributedEagleTargetContract(
            stageID: resolvedStageID,
            slidingWindow: resolvedSlidingWindow,
            finalHiddenStageID: resolvedHiddenStageID,
            finalHiddenTensorName: resolvedHiddenTensorName)
    }

    private static func firstNonEmpty(_ values: String?...) -> String? {
        values.lazy.compactMap { value -> String? in
            guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
                !trimmed.isEmpty
            else { return nil }
            return trimmed
        }.first
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
    private let storage: Storage

    public var float16BitPatterns: [UInt16] {
        switch storage {
        case .contiguous(let values):
            return values
        case .appendOnly(let sequence):
            return Self.materialize(
                sequence.segmentsChronological(),
                heads: sequence.heads,
                headDimension: sequence.headDimension,
                sequenceCount: sequence.sequenceCount)
        case .bounded(let sequence):
            return Self.materialize(
                sequence.segments,
                heads: sequence.heads,
                headDimension: sequence.headDimension,
                sequenceCount: sequence.sequenceCount)
        }
    }

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
        self.storage = .contiguous(float16BitPatterns)
    }

    public static func == (
        lhs: DistributedEagleTargetTensor,
        rhs: DistributedEagleTargetTensor
    ) -> Bool {
        lhs.shape == rhs.shape
            && lhs.scalarType == rhs.scalarType
            && lhs.float16BitPatterns == rhs.float16BitPatterns
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(shape)
        hasher.combine(scalarType)
        hasher.combine(float16BitPatterns)
    }

    var storageProfile: DistributedEagleTargetTensorStorageProfile {
        switch storage {
        case .contiguous(let values):
            return DistributedEagleTargetTensorStorageProfile(
                kind: .contiguous,
                segmentCount: 1,
                retainedElementCount: values.count,
                eagerlyMaterializedElementCount: values.count)
        case .appendOnly(let sequence):
            return DistributedEagleTargetTensorStorageProfile(
                kind: .appendOnlySequence,
                segmentCount: sequence.tail.segmentCount,
                retainedElementCount: sequence.tail.retainedElementCount,
                eagerlyMaterializedElementCount: 0)
        case .bounded(let sequence):
            return DistributedEagleTargetTensorStorageProfile(
                kind: .boundedSequence,
                segmentCount: sequence.segments.count,
                retainedElementCount: sequence.segments.reduce(0) { $0 + $1.values.count },
                eagerlyMaterializedElementCount: 0)
        }
    }

    fileprivate static func appendingAppendOnly(
        _ prefix: DistributedEagleTargetTensor?,
        _ suffix: DistributedEagleTargetTensor,
        label: String
    ) throws -> DistributedEagleTargetTensor {
        try validateSequenceLayout(prefix, suffix, label: label)
        let heads = suffix.shape[1]
        let headDimension = suffix.shape[3]
        let prefixSequence = prefix?.shape[2] ?? 0
        var tail: SequenceNode?
        if let prefix {
            switch prefix.storage {
            case .appendOnly(let sequence):
                tail = sequence.tail
            default:
                for segment in prefix.sequenceSegments() {
                    tail = SequenceNode(previous: tail, segment: segment)
                }
            }
        }
        for segment in suffix.sequenceSegments() {
            tail = SequenceNode(previous: tail, segment: segment)
        }
        guard let tail else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) cannot append an empty sequence")
        }
        return DistributedEagleTargetTensor(
            shape: [1, heads, prefixSequence + suffix.shape[2], headDimension],
            storage: .appendOnly(AppendOnlySequence(
                tail: tail,
                heads: heads,
                headDimension: headDimension,
                sequenceCount: prefixSequence + suffix.shape[2])))
    }

    fileprivate static func appendingBounded(
        _ prefix: DistributedEagleTargetTensor?,
        _ suffix: DistributedEagleTargetTensor,
        retainingSequenceCount: Int,
        label: String
    ) throws -> DistributedEagleTargetTensor {
        try validateSequenceLayout(prefix, suffix, label: label)
        let heads = suffix.shape[1]
        let headDimension = suffix.shape[3]
        var segments = prefix?.sequenceSegments() ?? []
        segments.append(contentsOf: suffix.sequenceSegments())
        var sequenceCount = segments.reduce(0) { $0 + $1.sequenceCount }
        var positionsToDrop = sequenceCount - retainingSequenceCount
        guard retainingSequenceCount > 0, positionsToDrop >= 0 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) cannot retain \(retainingSequenceCount) positions")
        }
        while positionsToDrop > 0, let first = segments.first {
            if positionsToDrop >= first.sequenceCount {
                positionsToDrop -= first.sequenceCount
                sequenceCount -= first.sequenceCount
                segments.removeFirst()
            } else {
                segments[0] = first.droppingFirst(
                    positionsToDrop,
                    heads: heads,
                    headDimension: headDimension)
                sequenceCount -= positionsToDrop
                positionsToDrop = 0
            }
        }
        guard sequenceCount == retainingSequenceCount, !segments.isEmpty else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) bounded sequence is inconsistent")
        }
        return DistributedEagleTargetTensor(
            shape: [1, heads, sequenceCount, headDimension],
            storage: .bounded(BoundedSequence(
                segments: segments,
                heads: heads,
                headDimension: headDimension,
                sequenceCount: sequenceCount)))
    }

    private init(shape: [Int], storage: Storage) {
        self.shape = shape
        self.scalarType = .float16
        self.storage = storage
    }

    private func sequenceSegments() -> [SequenceSegment] {
        switch storage {
        case .contiguous(let values):
            return [SequenceSegment(sequenceCount: shape[2], values: values)]
        case .appendOnly(let sequence):
            return sequence.segmentsChronological()
        case .bounded(let sequence):
            return sequence.segments
        }
    }

    private static func validateSequenceLayout(
        _ prefix: DistributedEagleTargetTensor?,
        _ suffix: DistributedEagleTargetTensor,
        label: String
    ) throws {
        guard suffix.shape.count == 4, suffix.shape[0] == 1 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) must use [1, heads, sequence, head_dim]")
        }
        guard let prefix else { return }
        guard prefix.shape.count == 4,
            prefix.shape[0] == suffix.shape[0],
            prefix.shape[1] == suffix.shape[1],
            prefix.shape[3] == suffix.shape[3]
        else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target \(label) layout changed between streamed chunks")
        }
    }

    private static func materialize(
        _ segments: [SequenceSegment],
        heads: Int,
        headDimension: Int,
        sequenceCount: Int
    ) -> [UInt16] {
        var values: [UInt16] = []
        values.reserveCapacity(heads * sequenceCount * headDimension)
        for head in 0..<heads {
            for segment in segments {
                let start = head * segment.sequenceCount * headDimension
                values.append(contentsOf: segment.values[
                    start..<(start + segment.sequenceCount * headDimension)])
            }
        }
        return values
    }

    private enum Storage: Sendable {
        case contiguous([UInt16])
        case appendOnly(AppendOnlySequence)
        case bounded(BoundedSequence)
    }

    private struct AppendOnlySequence: Sendable {
        let tail: SequenceNode
        let heads: Int
        let headDimension: Int
        let sequenceCount: Int

        func segmentsChronological() -> [SequenceSegment] {
            var segments: [SequenceSegment] = []
            segments.reserveCapacity(tail.segmentCount)
            var node: SequenceNode? = tail
            while let current = node {
                segments.append(current.segment)
                node = current.previous
            }
            segments.reverse()
            return segments
        }
    }

    private struct BoundedSequence: Sendable {
        let segments: [SequenceSegment]
        let heads: Int
        let headDimension: Int
        let sequenceCount: Int
    }

    private struct SequenceSegment: Sendable {
        let sequenceCount: Int
        let values: [UInt16]

        func droppingFirst(
            _ count: Int,
            heads: Int,
            headDimension: Int
        ) -> SequenceSegment {
            let retainedSequence = sequenceCount - count
            var retained: [UInt16] = []
            retained.reserveCapacity(heads * retainedSequence * headDimension)
            for head in 0..<heads {
                let start = (head * sequenceCount + count) * headDimension
                retained.append(contentsOf: values[
                    start..<(start + retainedSequence * headDimension)])
            }
            return SequenceSegment(
                sequenceCount: retainedSequence,
                values: retained)
        }
    }

    private final class SequenceNode: @unchecked Sendable {
        let previous: SequenceNode?
        let segment: SequenceSegment
        let segmentCount: Int
        let retainedElementCount: Int

        init(previous: SequenceNode?, segment: SequenceSegment) {
            self.previous = previous
            self.segment = segment
            self.segmentCount = (previous?.segmentCount ?? 0) + 1
            self.retainedElementCount = (previous?.retainedElementCount ?? 0) + segment.values.count
        }
    }
}

enum DistributedEagleTargetTensorStorageKind: Equatable, Sendable {
    case contiguous
    case appendOnlySequence
    case boundedSequence
}

struct DistributedEagleTargetTensorStorageProfile: Equatable, Sendable {
    let kind: DistributedEagleTargetTensorStorageKind
    let segmentCount: Int
    let retainedElementCount: Int
    let eagerlyMaterializedElementCount: Int
}

/// New-position representative K/V emitted by the final transformer stage.
///
/// The final post-norm hidden is emitted later by the head stage, so it deliberately is not part
/// of this chunk. The same-machine coordinator combines both producers before exposing an
/// assistant-ready ``DistributedEagleTargetArtifacts`` snapshot.
public struct DistributedEagleTargetKVChunk: Hashable, Sendable {
    public let fullKey: DistributedEagleTargetTensor
    public let fullValue: DistributedEagleTargetTensor
    public let slidingKey: DistributedEagleTargetTensor
    public let slidingValue: DistributedEagleTargetTensor
    public let fullPositionRange: DistributedSequenceRange
    public let slidingPositionRange: DistributedSequenceRange

    public init(
        fullKey: DistributedEagleTargetTensor,
        fullValue: DistributedEagleTargetTensor,
        slidingKey: DistributedEagleTargetTensor,
        slidingValue: DistributedEagleTargetTensor,
        fullPositionRange: DistributedSequenceRange,
        slidingPositionRange: DistributedSequenceRange
    ) throws {
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
        self.fullKey = fullKey
        self.fullValue = fullValue
        self.slidingKey = slidingKey
        self.slidingValue = slidingValue
        self.fullPositionRange = fullPositionRange
        self.slidingPositionRange = slidingPositionRange
    }

    fileprivate static func validateKVPair(
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
        let kvChunk = try DistributedEagleTargetKVChunk(
            fullKey: fullKey,
            fullValue: fullValue,
            slidingKey: slidingKey,
            slidingValue: slidingValue,
            fullPositionRange: fullPositionRange,
            slidingPositionRange: slidingPositionRange)
        try self.init(finalHidden: finalHidden, kvChunk: kvChunk)
    }

    public init(
        finalHidden: DistributedEagleTargetTensor,
        kvChunk: DistributedEagleTargetKVChunk
    ) throws {
        guard finalHidden.shape.count == 3,
            finalHidden.shape[0] == 1,
            finalHidden.shape[1] == 1
        else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "EAGLE target final hidden shape \(finalHidden.shape) must match [1, 1, hidden]")
        }
        self.finalHidden = finalHidden
        self.fullKey = kvChunk.fullKey
        self.fullValue = kvChunk.fullValue
        self.slidingKey = kvChunk.slidingKey
        self.slidingValue = kvChunk.slidingValue
        self.fullPositionRange = kvChunk.fullPositionRange
        self.slidingPositionRange = kvChunk.slidingPositionRange
    }
}

/// Per-request, capacity-bounded host accumulation for streamed staged target outputs.
public struct DistributedEagleTargetKVAccumulator: Sendable {
    public let kvCapacity: Int
    public let slidingWindow: Int
    public private(set) var snapshot: DistributedEagleTargetArtifacts?
    private var accumulatedKV: DistributedEagleTargetKVChunk?

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
        self.accumulatedKV = nil
    }

    public mutating func append(_ chunk: DistributedEagleTargetArtifacts) throws {
        let kvChunk = try DistributedEagleTargetKVChunk(
            fullKey: chunk.fullKey,
            fullValue: chunk.fullValue,
            slidingKey: chunk.slidingKey,
            slidingValue: chunk.slidingValue,
            fullPositionRange: chunk.fullPositionRange,
            slidingPositionRange: chunk.slidingPositionRange)
        try append(kvChunk, finalHidden: chunk.finalHidden)
    }

    mutating func append(
        _ chunk: DistributedEagleTargetKVChunk,
        finalHidden: DistributedEagleTargetTensor?
    ) throws {
        let processed = accumulatedKV?.fullPositionRange.upperBound ?? 0
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

        let fullKey = try DistributedEagleTargetTensor.appendingAppendOnly(
            accumulatedKV?.fullKey, chunk.fullKey, label: "full key")
        let fullValue = try DistributedEagleTargetTensor.appendingAppendOnly(
            accumulatedKV?.fullValue, chunk.fullValue, label: "full value")
        let slidingLength = min(nextProcessed, slidingWindow)
        let slidingKey = try DistributedEagleTargetTensor.appendingBounded(
            accumulatedKV?.slidingKey,
            chunk.slidingKey,
            retainingSequenceCount: slidingLength,
            label: "sliding key")
        let slidingValue = try DistributedEagleTargetTensor.appendingBounded(
            accumulatedKV?.slidingValue,
            chunk.slidingValue,
            retainingSequenceCount: slidingLength,
            label: "sliding value")

        let accumulatedKV = try DistributedEagleTargetKVChunk(
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
        self.accumulatedKV = accumulatedKV
        if let finalHidden {
            snapshot = try DistributedEagleTargetArtifacts(
                finalHidden: finalHidden,
                kvChunk: accumulatedKV)
        } else {
            snapshot = nil
        }
    }

    public mutating func reset() {
        snapshot = nil
        accumulatedKV = nil
    }

}
