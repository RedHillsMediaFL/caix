import Foundation

/// Role of one exported bundle in a stage-sharded LLM.
public enum DistributedStageRole: String, Codable, CaseIterable, Sendable {
    case embeddings
    case transformerLayers = "transformer_layers"
    case finalNormHead = "final_norm_head"

    public var requiresLayerRange: Bool {
        self == .transformerLayers
    }
}

/// Half-open transformer layer range: `lowerBound ..< upperBound`.
public struct DistributedLayerRange: Codable, Hashable, Sendable, CustomStringConvertible {
    public let lowerBound: Int
    public let upperBound: Int

    enum CodingKeys: String, CodingKey {
        case lowerBound = "lower_bound"
        case upperBound = "upper_bound"
    }

    public init(lowerBound: Int, upperBound: Int) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    public init(_ range: Range<Int>) {
        self.lowerBound = range.lowerBound
        self.upperBound = range.upperBound
    }

    public var count: Int {
        max(0, upperBound - lowerBound)
    }

    public var isValid: Bool {
        lowerBound >= 0 && upperBound > lowerBound
    }

    public func contains(_ layer: Int) -> Bool {
        layer >= lowerBound && layer < upperBound
    }

    public func overlaps(_ other: DistributedLayerRange) -> Bool {
        lowerBound < other.upperBound && other.lowerBound < upperBound
    }

    public func isAdjacent(to other: DistributedLayerRange) -> Bool {
        upperBound == other.lowerBound || other.upperBound == lowerBound
    }

    public var description: String {
        "\(lowerBound)..<\(upperBound)"
    }
}

/// One stage bundle assignment. `assetName` is the manifest asset key or local bundle label.
public struct DistributedStageDescriptor: Codable, Hashable, Sendable {
    public let id: String
    public let role: DistributedStageRole
    public let layerRange: DistributedLayerRange?
    public let assetName: String
    public let workerID: String?

    enum CodingKeys: String, CodingKey {
        case id
        case role
        case layerRange = "layer_range"
        case assetName = "asset_name"
        case workerID = "worker_id"
    }

    public init(
        id: String,
        role: DistributedStageRole,
        layerRange: DistributedLayerRange? = nil,
        assetName: String,
        workerID: String? = nil
    ) {
        self.id = id
        self.role = role
        self.layerRange = layerRange
        self.assetName = assetName
        self.workerID = workerID
    }
}

/// Worker address metadata. This does not define a network protocol.
public struct DistributedWorkerEndpoint: Codable, Hashable, Sendable, CustomStringConvertible {
    public let id: String
    public let host: String
    public let port: Int
    public let labels: [String: String]

    enum CodingKeys: String, CodingKey {
        case id
        case host
        case port
        case labels
    }

    public init(id: String, host: String, port: Int, labels: [String: String] = [:]) {
        self.id = id
        self.host = host
        self.port = port
        self.labels = labels
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.host = try c.decode(String.self, forKey: .host)
        self.port = try c.decode(Int.self, forKey: .port)
        self.labels = try c.decodeIfPresent([String: String].self, forKey: .labels) ?? [:]
    }

    public var description: String {
        "\(id)@\(host):\(port)"
    }
}

/// Scalar type for hidden-state packets.
public enum DistributedTensorScalarType: String, Codable, CaseIterable, Sendable {
    case float16
    case float32

    public var byteWidth: Int {
        switch self {
        case .float16: return 2
        case .float32: return 4
        }
    }
}

/// Half-open token-position range: `lowerBound ..< upperBound`.
public struct DistributedSequenceRange: Codable, Hashable, Sendable, CustomStringConvertible {
    public let lowerBound: Int
    public let upperBound: Int

    enum CodingKeys: String, CodingKey {
        case lowerBound = "lower_bound"
        case upperBound = "upper_bound"
    }

    public init(lowerBound: Int, upperBound: Int) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    public init(_ range: Range<Int>) {
        self.lowerBound = range.lowerBound
        self.upperBound = range.upperBound
    }

    public var count: Int {
        max(0, upperBound - lowerBound)
    }

    public var isValid: Bool {
        lowerBound >= 0 && upperBound > lowerBound
    }

    public var description: String {
        "\(lowerBound)..<\(upperBound)"
    }
}

/// Metadata for an activation payload sent from one stage to the next.
public struct DistributedHiddenStatePacketMetadata: Codable, Hashable, Sendable {
    public let requestID: String
    public let sourceStageID: String
    public let destinationStageID: String
    public let positionRange: DistributedSequenceRange
    /// Hidden states use `[batch, sequence, hidden]`.
    public let shape: [Int]
    public let scalarType: DistributedTensorScalarType
    public let byteCount: Int
    public let stepIndex: Int

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case sourceStageID = "source_stage_id"
        case destinationStageID = "destination_stage_id"
        case positionRange = "position_range"
        case shape
        case scalarType = "scalar_type"
        case byteCount = "byte_count"
        case stepIndex = "step_index"
    }

    public init(
        requestID: String,
        sourceStageID: String,
        destinationStageID: String,
        positionRange: DistributedSequenceRange,
        shape: [Int],
        scalarType: DistributedTensorScalarType,
        byteCount: Int,
        stepIndex: Int
    ) {
        self.requestID = requestID
        self.sourceStageID = sourceStageID
        self.destinationStageID = destinationStageID
        self.positionRange = positionRange
        self.shape = shape
        self.scalarType = scalarType
        self.byteCount = byteCount
        self.stepIndex = stepIndex
    }

    public var tokenCount: Int {
        positionRange.count
    }

    public var expectedByteCount: Int? {
        var total = scalarType.byteWidth
        for dim in shape {
            guard dim > 0 else { return nil }
            let next = total.multipliedReportingOverflow(by: dim)
            guard !next.overflow else { return nil }
            total = next.partialValue
        }
        return total
    }
}

/// Activation packet with the raw hidden-state payload.
public struct DistributedHiddenStatePacket: Hashable, Sendable {
    public let metadata: DistributedHiddenStatePacketMetadata
    public let payload: [UInt8]

    public init(metadata: DistributedHiddenStatePacketMetadata, payload: [UInt8]) throws {
        self.metadata = metadata
        self.payload = payload
        try DistributedRuntimeValidation.validate(packet: metadata)
        guard payload.count == metadata.byteCount else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "payload byte count does not match metadata byte_count")
        }
    }
}

/// Static plan for local or remote stage execution.
public struct DistributedStagePlan: Codable, Hashable, Sendable {
    public let modelName: String
    public let totalLayerCount: Int
    public let stages: [DistributedStageDescriptor]
    public let workers: [DistributedWorkerEndpoint]

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case totalLayerCount = "total_layer_count"
        case stages
        case workers
    }

    public init(
        modelName: String,
        totalLayerCount: Int,
        stages: [DistributedStageDescriptor],
        workers: [DistributedWorkerEndpoint] = []
    ) {
        self.modelName = modelName
        self.totalLayerCount = totalLayerCount
        self.stages = stages
        self.workers = workers
    }

    public func stage(id: String) -> DistributedStageDescriptor? {
        stages.first { $0.id == id }
    }

    public func nextStage(after id: String) -> DistributedStageDescriptor? {
        guard let index = stages.firstIndex(where: { $0.id == id }),
            stages.indices.contains(index + 1)
        else { return nil }
        return stages[index + 1]
    }

    public func validate() throws {
        try DistributedRuntimeValidation.validate(plan: self)
    }

    public func validate(hiddenStatePacket packet: DistributedHiddenStatePacketMetadata) throws {
        try DistributedRuntimeValidation.validate(packet: packet, in: self)
    }
}

public struct DistributedStageAllocation: Hashable, Sendable {
    public let requestID: String
    public let kvCapacity: Int

    public init(requestID: String, kvCapacity: Int) {
        self.requestID = requestID
        self.kvCapacity = kvCapacity
    }
}

public struct DistributedStageForwardInput: Hashable, Sendable {
    public let requestID: String
    public let stepIndex: Int
    public let positionRange: DistributedSequenceRange
    public let tokenIDs: [Int32]
    public let hiddenState: DistributedHiddenStatePacket?

    public init(
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        tokenIDs: [Int32] = [],
        hiddenState: DistributedHiddenStatePacket? = nil
    ) {
        self.requestID = requestID
        self.stepIndex = stepIndex
        self.positionRange = positionRange
        self.tokenIDs = tokenIDs
        self.hiddenState = hiddenState
    }
}

public struct DistributedStageForwardOutput: Hashable, Sendable {
    public let stageID: String
    public let stepIndex: Int
    public let hiddenState: DistributedHiddenStatePacket?
    public let tokenID: Int32?

    public init(
        stageID: String,
        stepIndex: Int,
        hiddenState: DistributedHiddenStatePacket? = nil,
        tokenID: Int32? = nil
    ) {
        self.stageID = stageID
        self.stepIndex = stepIndex
        self.hiddenState = hiddenState
        self.tokenID = tokenID
    }
}

public protocol DistributedStageHandle: AnyObject {
    var descriptor: DistributedStageDescriptor { get }

    func allocate(_ allocation: DistributedStageAllocation) async throws
    func forward(_ input: DistributedStageForwardInput) async throws -> DistributedStageForwardOutput
    func reset(requestID: String) async throws
    func free(requestID: String) async
}

public enum DistributedStageExecutionError: Error, Equatable, Sendable, CustomStringConvertible {
    case stageCountMismatch(expected: Int, actual: Int)
    case stageDescriptorMismatch(expected: String, actual: String)
    case duplicateStageHandle(String)
    case invalidForwardInput(String)
    case invalidStageOutput(String)

    public var description: String {
        switch self {
        case .stageCountMismatch(let expected, let actual):
            return "Stage handle count mismatch: expected \(expected), got \(actual)"
        case .stageDescriptorMismatch(let expected, let actual):
            return "Stage handle descriptor mismatch: expected \(expected), got \(actual)"
        case .duplicateStageHandle(let id):
            return "Duplicate stage handle: \(id)"
        case .invalidForwardInput(let message):
            return "Invalid distributed forward input: \(message)"
        case .invalidStageOutput(let message):
            return "Invalid distributed stage output: \(message)"
        }
    }
}

/// In-process coordinator for one staged forward. This is the same-machine milestone harness;
/// concrete handles can be fake test stages now and Core AI stage handles later.
public final class DistributedSameMachinePipeline {
    public let plan: DistributedStagePlan
    private let stages: [DistributedStageHandle]

    public init(plan: DistributedStagePlan, stages: [DistributedStageHandle]) throws {
        try plan.validate()
        guard plan.stages.count == stages.count else {
            throw DistributedStageExecutionError.stageCountMismatch(
                expected: plan.stages.count, actual: stages.count)
        }

        var seen = Set<String>()
        for (expected, handle) in zip(plan.stages, stages) {
            guard seen.insert(handle.descriptor.id).inserted else {
                throw DistributedStageExecutionError.duplicateStageHandle(handle.descriptor.id)
            }
            guard handle.descriptor == expected else {
                throw DistributedStageExecutionError.stageDescriptorMismatch(
                    expected: expected.id, actual: handle.descriptor.id)
            }
        }

        self.plan = plan
        self.stages = stages
    }

    public func allocate(requestID: String, kvCapacity: Int) async throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("request_id is empty")
        }
        guard kvCapacity > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput("kv_capacity must be positive")
        }
        let allocation = DistributedStageAllocation(requestID: requestID, kvCapacity: kvCapacity)
        for stage in stages {
            try await stage.allocate(allocation)
        }
    }

    public func forward(
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        tokenIDs: [Int32]
    ) async throws -> DistributedStageForwardOutput {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("request_id is empty")
        }
        guard !tokenIDs.isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("token_ids must be non-empty")
        }
        guard tokenIDs.count == positionRange.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "token_ids count must match position_range")
        }
        guard stepIndex >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "step_index must be non-negative")
        }

        var hiddenState: DistributedHiddenStatePacket?
        var tokenID: Int32?

        for (index, stage) in stages.enumerated() {
            let input = DistributedStageForwardInput(
                requestID: requestID,
                stepIndex: stepIndex,
                positionRange: positionRange,
                tokenIDs: index == 0 ? tokenIDs : [],
                hiddenState: hiddenState)
            let output = try await stage.forward(input)
            try validate(output: output, from: stage, at: index, stepIndex: stepIndex)
            hiddenState = output.hiddenState
            tokenID = output.tokenID
        }

        guard tokenID != nil else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "final stage did not return a token id")
        }
        return DistributedStageForwardOutput(
            stageID: stages.last!.descriptor.id,
            stepIndex: stepIndex,
            hiddenState: hiddenState,
            tokenID: tokenID)
    }

    public func reset(requestID: String) async throws {
        for stage in stages {
            try await stage.reset(requestID: requestID)
        }
    }

    public func free(requestID: String) async {
        for stage in stages {
            await stage.free(requestID: requestID)
        }
    }

    private func validate(
        output: DistributedStageForwardOutput,
        from stage: DistributedStageHandle,
        at index: Int,
        stepIndex: Int
    ) throws {
        guard output.stageID == stage.descriptor.id else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "output stage_id \(output.stageID) does not match handle \(stage.descriptor.id)")
        }
        guard output.stepIndex == stepIndex else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "output step_index does not match request")
        }

        let isFinal = index == stages.count - 1
        if isFinal {
            guard output.hiddenState == nil else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "final stage must not return a hidden state")
            }
            guard output.tokenID != nil else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "final stage must return a token id")
            }
        } else {
            guard let packet = output.hiddenState else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "non-final stage must return a hidden state")
            }
            try plan.validate(hiddenStatePacket: packet.metadata)
            guard packet.metadata.sourceStageID == stage.descriptor.id else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "hidden state source_stage_id does not match producing stage")
            }
            guard packet.metadata.destinationStageID == stages[index + 1].descriptor.id else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "hidden state destination_stage_id does not match next stage")
            }
            guard packet.payload.count == packet.metadata.byteCount else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "hidden state payload byte count does not match metadata")
            }
            guard output.tokenID == nil else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "non-final stage must not return a token id")
            }
        }
    }
}

public enum DistributedRuntimeValidationError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidIdentifier(field: String)
    case invalidTotalLayerCount(Int)
    case invalidRoleCount(role: DistributedStageRole, count: Int, expected: String)
    case duplicateStageID(String)
    case duplicateWorkerID(String)
    case invalidEndpoint(id: String, reason: String)
    case unknownWorkerID(stageID: String, workerID: String)
    case layerRangeRequired(stageID: String)
    case layerRangeNotAllowed(stageID: String, role: DistributedStageRole)
    case invalidLayerRange(stageID: String, range: DistributedLayerRange)
    case layerCoverageGap(expectedStart: Int, stageID: String, actual: DistributedLayerRange)
    case layerCoverageEnd(actualEnd: Int, expectedEnd: Int)
    case invalidStageOrder(String)
    case invalidPacket(String)
    case packetRouteMismatch(sourceStageID: String, destinationStageID: String)

    public var description: String {
        switch self {
        case .invalidIdentifier(let field):
            return "Invalid distributed runtime identifier: \(field)"
        case .invalidTotalLayerCount(let count):
            return "Invalid layer count: \(count)"
        case .invalidRoleCount(let role, let count, let expected):
            return "Invalid stage role count for \(role.rawValue): got \(count), expected \(expected)"
        case .duplicateStageID(let id):
            return "Duplicate stage id: \(id)"
        case .duplicateWorkerID(let id):
            return "Duplicate worker id: \(id)"
        case .invalidEndpoint(let id, let reason):
            return "Invalid worker endpoint \(id): \(reason)"
        case .unknownWorkerID(let stageID, let workerID):
            return "Stage \(stageID) references unknown worker \(workerID)"
        case .layerRangeRequired(let stageID):
            return "Stage \(stageID) requires a layer range"
        case .layerRangeNotAllowed(let stageID, let role):
            return "Stage \(stageID) with role \(role.rawValue) must not have a layer range"
        case .invalidLayerRange(let stageID, let range):
            return "Stage \(stageID) has invalid layer range \(range)"
        case .layerCoverageGap(let expectedStart, let stageID, let actual):
            return "Stage \(stageID) starts at \(actual.lowerBound); expected \(expectedStart)"
        case .layerCoverageEnd(let actualEnd, let expectedEnd):
            return "Layer coverage ends at \(actualEnd); expected \(expectedEnd)"
        case .invalidStageOrder(let message):
            return "Invalid stage order: \(message)"
        case .invalidPacket(let message):
            return "Invalid hidden-state packet: \(message)"
        case .packetRouteMismatch(let sourceStageID, let destinationStageID):
            return "Hidden-state packet route is not adjacent: \(sourceStageID) -> \(destinationStageID)"
        }
    }
}

public enum DistributedRuntimeValidation {
    public static func validate(endpoint: DistributedWorkerEndpoint) throws {
        guard !trimmed(endpoint.id).isEmpty else {
            throw DistributedRuntimeValidationError.invalidIdentifier(field: "worker.id")
        }
        guard !trimmed(endpoint.host).isEmpty else {
            throw DistributedRuntimeValidationError.invalidEndpoint(
                id: endpoint.id, reason: "host is empty")
        }
        guard (1...65_535).contains(endpoint.port) else {
            throw DistributedRuntimeValidationError.invalidEndpoint(
                id: endpoint.id, reason: "port must be 1...65535")
        }
        if endpoint.labels.keys.contains(where: { trimmed($0).isEmpty }) {
            throw DistributedRuntimeValidationError.invalidEndpoint(
                id: endpoint.id, reason: "label keys must be non-empty")
        }
    }

    public static func validate(stage: DistributedStageDescriptor) throws {
        guard !trimmed(stage.id).isEmpty else {
            throw DistributedRuntimeValidationError.invalidIdentifier(field: "stage.id")
        }
        guard !trimmed(stage.assetName).isEmpty else {
            throw DistributedRuntimeValidationError.invalidIdentifier(field: "stage.asset_name")
        }
        if let workerID = stage.workerID, trimmed(workerID).isEmpty {
            throw DistributedRuntimeValidationError.invalidIdentifier(field: "stage.worker_id")
        }

        if stage.role.requiresLayerRange {
            guard let range = stage.layerRange else {
                throw DistributedRuntimeValidationError.layerRangeRequired(stageID: stage.id)
            }
            guard range.isValid else {
                throw DistributedRuntimeValidationError.invalidLayerRange(
                    stageID: stage.id, range: range)
            }
        } else if stage.layerRange != nil {
            throw DistributedRuntimeValidationError.layerRangeNotAllowed(
                stageID: stage.id, role: stage.role)
        }
    }

    public static func validate(plan: DistributedStagePlan) throws {
        guard plan.totalLayerCount > 0 else {
            throw DistributedRuntimeValidationError.invalidTotalLayerCount(plan.totalLayerCount)
        }

        var workerIDs = Set<String>()
        for endpoint in plan.workers {
            try validate(endpoint: endpoint)
            guard workerIDs.insert(endpoint.id).inserted else {
                throw DistributedRuntimeValidationError.duplicateWorkerID(endpoint.id)
            }
        }

        var stageIDs = Set<String>()
        var roleCounts: [DistributedStageRole: Int] = [:]
        var layerStages: [DistributedStageDescriptor] = []

        for stage in plan.stages {
            try validate(stage: stage)
            guard stageIDs.insert(stage.id).inserted else {
                throw DistributedRuntimeValidationError.duplicateStageID(stage.id)
            }
            if let workerID = stage.workerID, !workerIDs.contains(workerID) {
                throw DistributedRuntimeValidationError.unknownWorkerID(
                    stageID: stage.id, workerID: workerID)
            }
            roleCounts[stage.role, default: 0] += 1
            if stage.role == .transformerLayers {
                layerStages.append(stage)
            }
        }

        try requireRole(.embeddings, count: roleCounts[.embeddings, default: 0], expected: "1")
        try requireRole(
            .finalNormHead, count: roleCounts[.finalNormHead, default: 0], expected: "1")
        try requireRole(
            .transformerLayers, count: layerStages.count, expected: "one or more")

        guard plan.stages.first?.role == .embeddings else {
            throw DistributedRuntimeValidationError.invalidStageOrder(
                "first stage must be embeddings")
        }
        guard plan.stages.last?.role == .finalNormHead else {
            throw DistributedRuntimeValidationError.invalidStageOrder(
                "last stage must be final_norm_head")
        }
        if plan.stages.dropFirst().dropLast().contains(where: { $0.role != .transformerLayers }) {
            throw DistributedRuntimeValidationError.invalidStageOrder(
                "only transformer_layers stages may sit between embeddings and final_norm_head")
        }

        var expectedStart = 0
        for stage in layerStages {
            guard let range = stage.layerRange else {
                throw DistributedRuntimeValidationError.layerRangeRequired(stageID: stage.id)
            }
            guard range.lowerBound == expectedStart else {
                throw DistributedRuntimeValidationError.layerCoverageGap(
                    expectedStart: expectedStart, stageID: stage.id, actual: range)
            }
            expectedStart = range.upperBound
        }
        guard expectedStart == plan.totalLayerCount else {
            throw DistributedRuntimeValidationError.layerCoverageEnd(
                actualEnd: expectedStart, expectedEnd: plan.totalLayerCount)
        }
    }

    public static func validate(packet: DistributedHiddenStatePacketMetadata) throws {
        guard !trimmed(packet.requestID).isEmpty else {
            throw DistributedRuntimeValidationError.invalidPacket("request_id is empty")
        }
        guard !trimmed(packet.sourceStageID).isEmpty else {
            throw DistributedRuntimeValidationError.invalidPacket("source_stage_id is empty")
        }
        guard !trimmed(packet.destinationStageID).isEmpty else {
            throw DistributedRuntimeValidationError.invalidPacket("destination_stage_id is empty")
        }
        guard packet.sourceStageID != packet.destinationStageID else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "source and destination stages are the same")
        }
        guard packet.positionRange.isValid else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "position_range must be non-empty and non-negative")
        }
        guard packet.shape.count == 3 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "shape must be [batch, sequence, hidden]")
        }
        guard packet.shape.allSatisfy({ $0 > 0 }) else {
            throw DistributedRuntimeValidationError.invalidPacket("shape dimensions must be positive")
        }
        guard packet.shape[1] == packet.positionRange.count else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "shape sequence does not match position_range count")
        }
        guard packet.byteCount > 0 else {
            throw DistributedRuntimeValidationError.invalidPacket("byte_count must be positive")
        }
        guard packet.expectedByteCount == packet.byteCount else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "byte_count does not match shape and scalar_type")
        }
        guard packet.stepIndex >= 0 else {
            throw DistributedRuntimeValidationError.invalidPacket("step_index must be non-negative")
        }
    }

    public static func validate(
        packet: DistributedHiddenStatePacketMetadata,
        in plan: DistributedStagePlan
    ) throws {
        try validate(plan: plan)
        try validate(packet: packet)

        guard let sourceIndex = plan.stages.firstIndex(where: { $0.id == packet.sourceStageID }) else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "unknown source stage \(packet.sourceStageID)")
        }
        guard let destinationIndex = plan.stages.firstIndex(where: { $0.id == packet.destinationStageID })
        else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "unknown destination stage \(packet.destinationStageID)")
        }
        guard destinationIndex == sourceIndex + 1 else {
            throw DistributedRuntimeValidationError.packetRouteMismatch(
                sourceStageID: packet.sourceStageID,
                destinationStageID: packet.destinationStageID)
        }
    }

    private static func requireRole(
        _ role: DistributedStageRole,
        count: Int,
        expected: String
    ) throws {
        guard count == 1 || (role == .transformerLayers && count > 0) else {
            throw DistributedRuntimeValidationError.invalidRoleCount(
                role: role, count: count, expected: expected)
        }
    }

    private static func trimmed(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
