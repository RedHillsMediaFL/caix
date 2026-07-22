import Foundation
import CryptoKit

/// Role of one exported bundle in a stage-sharded LLM.
public enum DistributedStageRole: String, Codable, CaseIterable, Sendable {
    case embeddings
    case transformerLayers = "transformer_layers"
    case finalNormHead = "final_norm_head"

    public var requiresLayerRange: Bool {
        self == .transformerLayers
    }
}

/// Position-id contract used by a staged export.
public enum DistributedPositionMode: String, Codable, CaseIterable, Sendable {
    case current
    case fullPrefix = "full_prefix"

    public func positionIDs(for positionRange: DistributedSequenceRange) throws -> [Int32] {
        guard positionRange.isValid else {
            throw DistributedStageExecutionError.invalidForwardInput("position_range is invalid")
        }
        let range: Range<Int>
        switch self {
        case .current:
            range = positionRange.lowerBound..<positionRange.upperBound
        case .fullPrefix:
            range = 0..<positionRange.upperBound
        }
        guard range.lowerBound >= Int(Int32.min),
            range.upperBound <= Int(Int32.max)
        else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_ids exceed Int32 range")
        }
        return range.map { Int32($0) }
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

/// Optional per-stage exported function names.
public struct DistributedStageFunctionMap: Codable, Hashable, Sendable {
    public let byRole: [String: [String]]

    public init(_ byRole: [String: [String]]) {
        self.byRole = byRole
    }

    public init(main: String = "main", decode: String? = nil) {
        var byRole = ["main": [main]]
        if let decode {
            byRole["decode"] = [decode]
        }
        self.byRole = byRole
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.byRole = try container.decode([String: [String]].self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(byRole)
    }

    public func name(for role: String) -> String? {
        byRole[role]?.first
    }

    public var mainFunctionName: String {
        name(for: "main") ?? "main"
    }

    public var decodeFunctionName: String? {
        name(for: "decode")
    }

    var validationErrorMessage: String? {
        for (role, names) in byRole {
            if Self.trimmed(role).isEmpty {
                return "function_map roles must be non-empty"
            }
            if names.isEmpty {
                return "function_map \(role) names must be non-empty"
            }
            if names.contains(where: { Self.trimmed($0).isEmpty }) {
                return "function_map \(role) names must be non-empty"
            }
        }
        return nil
    }

    private static func trimmed(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Optional RoPE inputs that are computed outside a staged transformer Core AI graph.
public struct DistributedStageRoPEInputSpec: Codable, Hashable, Sendable {
    public let cosInputName: String
    public let sinInputName: String
    public let headDim: Int
    public let theta: Double

    enum CodingKeys: String, CodingKey {
        case cosInputName = "cos_input"
        case sinInputName = "sin_input"
        case headDim = "head_dim"
        case theta
    }

    public init(
        cosInputName: String,
        sinInputName: String,
        headDim: Int,
        theta: Double
    ) {
        self.cosInputName = cosInputName
        self.sinInputName = sinInputName
        self.headDim = headDim
        self.theta = theta
    }

    var validationErrorMessage: String? {
        if Self.trimmed(cosInputName).isEmpty {
            return "rope cos_input must be non-empty"
        }
        if Self.trimmed(sinInputName).isEmpty {
            return "rope sin_input must be non-empty"
        }
        if cosInputName == sinInputName {
            return "rope cos_input and sin_input must differ"
        }
        if headDim <= 0 || !headDim.isMultiple(of: 2) {
            return "rope head_dim must be a positive even integer"
        }
        if !theta.isFinite || theta <= 0 {
            return "rope theta must be positive"
        }
        return nil
    }

    private static func trimmed(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// One stage bundle assignment. `assetName` is the manifest asset key or local bundle label.
public struct DistributedStageDescriptor: Codable, Hashable, Sendable {
    public let id: String
    public let role: DistributedStageRole
    public let layerRange: DistributedLayerRange?
    public let assetName: String
    public let decodeAssetName: String?
    public let functionMap: DistributedStageFunctionMap?
    public let vocabSize: Int?
    public let prefillExtraInputs: [String]
    public let workerID: String?
    public let rope: DistributedStageRoPEInputSpec?

    enum CodingKeys: String, CodingKey {
        case id
        case role
        case layerRange = "layer_range"
        case assetName = "asset_name"
        case decodeAssetName = "decode_asset_name"
        case functionMap = "function_map"
        case vocabSize = "vocab_size"
        case prefillExtraInputs = "prefill_extra_inputs"
        case workerID = "worker_id"
        case rope
    }

    public init(
        id: String,
        role: DistributedStageRole,
        layerRange: DistributedLayerRange? = nil,
        assetName: String,
        decodeAssetName: String? = nil,
        functionMap: DistributedStageFunctionMap? = nil,
        vocabSize: Int? = nil,
        prefillExtraInputs: [String] = [],
        workerID: String? = nil,
        rope: DistributedStageRoPEInputSpec? = nil
    ) {
        self.id = id
        self.role = role
        self.layerRange = layerRange
        self.assetName = assetName
        self.decodeAssetName = decodeAssetName
        self.functionMap = functionMap
        self.vocabSize = vocabSize
        self.prefillExtraInputs = prefillExtraInputs
        self.workerID = workerID
        self.rope = rope
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.role = try c.decode(DistributedStageRole.self, forKey: .role)
        self.layerRange = try c.decodeIfPresent(DistributedLayerRange.self, forKey: .layerRange)
        self.assetName = try c.decode(String.self, forKey: .assetName)
        self.decodeAssetName = try c.decodeIfPresent(String.self, forKey: .decodeAssetName)
        self.functionMap = try c.decodeIfPresent(
            DistributedStageFunctionMap.self,
            forKey: .functionMap)
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize)
        self.prefillExtraInputs = try c.decodeIfPresent(
            [String].self,
            forKey: .prefillExtraInputs) ?? []
        self.workerID = try c.decodeIfPresent(String.self, forKey: .workerID)
        self.rope = try c.decodeIfPresent(DistributedStageRoPEInputSpec.self, forKey: .rope)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(role, forKey: .role)
        try c.encodeIfPresent(layerRange, forKey: .layerRange)
        try c.encode(assetName, forKey: .assetName)
        try c.encodeIfPresent(decodeAssetName, forKey: .decodeAssetName)
        try c.encodeIfPresent(functionMap, forKey: .functionMap)
        try c.encodeIfPresent(vocabSize, forKey: .vocabSize)
        if !prefillExtraInputs.isEmpty {
            try c.encode(prefillExtraInputs, forKey: .prefillExtraInputs)
        }
        try c.encodeIfPresent(workerID, forKey: .workerID)
        try c.encodeIfPresent(rope, forKey: .rope)
    }
}

/// Stage layer metadata from a staged model manifest.
public enum DistributedStageLayerSpec: Hashable, Sendable, CustomStringConvertible {
    case label(String)
    case range(DistributedLayerRange)

    public var layerRange: DistributedLayerRange? {
        if case .range(let range) = self { return range }
        return nil
    }

    public var label: String? {
        if case .label(let label) = self { return label }
        return nil
    }

    public var description: String {
        switch self {
        case .label(let label):
            return label
        case .range(let range):
            return range.description
        }
    }
}

extension DistributedStageLayerSpec: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let values = try? container.decode([Int].self) {
            self = try .range(Self.layerRange(from: values, codingPath: decoder.codingPath))
            return
        }
        if let values = try? container.decode([String].self) {
            let ints = values.compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            guard ints.count == values.count else {
                throw DecodingError.typeMismatch(
                    DistributedStageLayerSpec.self,
                    DecodingError.Context(
                        codingPath: decoder.codingPath,
                        debugDescription: "layer range array must contain integers"))
            }
            self = try .range(Self.layerRange(from: ints, codingPath: decoder.codingPath))
            return
        }
        if let object = try? container.decode(DistributedLayerRangeObject.self) {
            self = .range(object.range)
            return
        }
        if let value = try? container.decode(String.self) {
            if let range = Self.parseRangeString(value) {
                self = .range(range)
            } else {
                self = .label(value)
            }
            return
        }
        throw DecodingError.typeMismatch(
            DistributedStageLayerSpec.self,
            DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "layers must be a label, [lower, upper], or range object"))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .label(let label):
            try container.encode(label)
        case .range(let range):
            try container.encode([range.lowerBound, range.upperBound])
        }
    }

    private static func layerRange(
        from values: [Int],
        codingPath: [CodingKey]
    ) throws -> DistributedLayerRange {
        guard values.count == 2 else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: codingPath,
                    debugDescription: "layer range must be [lower, upper]"))
        }
        let range = DistributedLayerRange(lowerBound: values[0], upperBound: values[1])
        guard range.isValid else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: codingPath,
                    debugDescription: "layer range must be non-empty and non-negative"))
        }
        return range
    }

    private static func parseRangeString(_ value: String) -> DistributedLayerRange? {
        let normalized = value
            .replacingOccurrences(of: "..<", with: ",")
            .replacingOccurrences(of: "-", with: ",")
        let parts = normalized
            .split(separator: ",", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard parts.count == 2, let lower = Int(parts[0]), let upper = Int(parts[1]) else {
            return nil
        }
        let range = DistributedLayerRange(lowerBound: lower, upperBound: upper)
        return range.isValid ? range : nil
    }
}

/// One normalized stage entry from `caix.cluster.stage_manifest.v0`.
public struct DistributedStageManifestStage: Codable, Hashable, Sendable {
    public let id: String
    public let role: DistributedStageRole
    public let layerSpec: DistributedStageLayerSpec
    public let assetName: String
    public let resolvedAssetPath: String?
    public let decodeAssetName: String?
    public let resolvedDecodeAssetPath: String?
    public let functionMap: DistributedStageFunctionMap?
    public let vocabSize: Int?
    public let prefillExtraInputs: [String]
    public let memoryGB: Double
    public let rope: DistributedStageRoPEInputSpec?

    enum CodingKeys: String, CodingKey {
        case id
        case role
        case layerSpec = "layers"
        case assetName = "asset_name"
        case resolvedAssetPath = "resolved_asset_path"
        case decodeAssetName = "decode_asset_name"
        case resolvedDecodeAssetPath = "resolved_decode_asset_path"
        case functionMap = "function_map"
        case vocabSize = "vocab_size"
        case prefillExtraInputs = "prefill_extra_inputs"
        case memoryGB = "memory_gb"
        case rope
    }

    public init(
        id: String,
        role: DistributedStageRole,
        layerSpec: DistributedStageLayerSpec,
        assetName: String,
        resolvedAssetPath: String? = nil,
        decodeAssetName: String? = nil,
        resolvedDecodeAssetPath: String? = nil,
        functionMap: DistributedStageFunctionMap? = nil,
        vocabSize: Int? = nil,
        prefillExtraInputs: [String] = [],
        memoryGB: Double,
        rope: DistributedStageRoPEInputSpec? = nil
    ) {
        self.id = id
        self.role = role
        self.layerSpec = layerSpec
        self.assetName = assetName
        self.resolvedAssetPath = resolvedAssetPath
        self.decodeAssetName = decodeAssetName
        self.resolvedDecodeAssetPath = resolvedDecodeAssetPath
        self.functionMap = functionMap
        self.vocabSize = vocabSize
        self.prefillExtraInputs = prefillExtraInputs
        self.memoryGB = memoryGB
        self.rope = rope
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.role = try c.decode(DistributedStageRole.self, forKey: .role)
        self.layerSpec = try c.decode(DistributedStageLayerSpec.self, forKey: .layerSpec)
        self.assetName = try c.decode(String.self, forKey: .assetName)
        self.resolvedAssetPath = try c.decodeIfPresent(String.self, forKey: .resolvedAssetPath)
        self.decodeAssetName = try c.decodeIfPresent(String.self, forKey: .decodeAssetName)
        self.resolvedDecodeAssetPath = try c.decodeIfPresent(
            String.self,
            forKey: .resolvedDecodeAssetPath)
        self.functionMap = try c.decodeIfPresent(
            DistributedStageFunctionMap.self,
            forKey: .functionMap)
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize)
        self.prefillExtraInputs = try c.decodeIfPresent(
            [String].self,
            forKey: .prefillExtraInputs) ?? []
        self.memoryGB = try c.decode(Double.self, forKey: .memoryGB)
        self.rope = try c.decodeIfPresent(DistributedStageRoPEInputSpec.self, forKey: .rope)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(role, forKey: .role)
        try c.encode(layerSpec, forKey: .layerSpec)
        try c.encode(assetName, forKey: .assetName)
        try c.encodeIfPresent(resolvedAssetPath, forKey: .resolvedAssetPath)
        try c.encodeIfPresent(decodeAssetName, forKey: .decodeAssetName)
        try c.encodeIfPresent(resolvedDecodeAssetPath, forKey: .resolvedDecodeAssetPath)
        try c.encodeIfPresent(functionMap, forKey: .functionMap)
        try c.encodeIfPresent(vocabSize, forKey: .vocabSize)
        if !prefillExtraInputs.isEmpty {
            try c.encode(prefillExtraInputs, forKey: .prefillExtraInputs)
        }
        try c.encode(memoryGB, forKey: .memoryGB)
        try c.encodeIfPresent(rope, forKey: .rope)
    }

    public var layerRange: DistributedLayerRange? {
        layerSpec.layerRange
    }

    public var layerLabel: String? {
        layerSpec.label
    }

    public var layerDescription: String {
        layerSpec.description
    }

    public func descriptor(workerID: String? = nil) -> DistributedStageDescriptor {
        DistributedStageDescriptor(
            id: id,
            role: role,
            layerRange: layerRange,
            assetName: assetName,
            decodeAssetName: decodeAssetName,
            functionMap: functionMap,
            vocabSize: vocabSize,
            prefillExtraInputs: prefillExtraInputs,
            workerID: workerID,
            rope: rope)
    }
}

/// Hidden-state tensor contract at stage boundaries.
public struct DistributedBoundaryTensorSpec: Codable, Hashable, Sendable {
    public let name: String
    /// Manifest shapes may use `-1` for the sequence dimension. Runtime packets use concrete sizes.
    public let shape: [Int]
    public let scalarType: DistributedTensorScalarType

    enum CodingKeys: String, CodingKey {
        case name
        case shape
        case scalarType = "scalar_type"
    }

    public init(name: String, shape: [Int], scalarType: DistributedTensorScalarType) {
        self.name = name
        self.shape = shape
        self.scalarType = scalarType
    }

    public var validationErrorMessage: String? {
        if name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "boundary hidden_state name is missing"
        }
        guard shape.count == 3 else {
            return "boundary hidden_state shape must be [batch, sequence, hidden]"
        }
        guard shape[0] > 0 else {
            return "boundary hidden_state batch dimension must be positive"
        }
        guard shape[1] == -1 || shape[1] > 0 else {
            return "boundary hidden_state sequence dimension must be positive or -1"
        }
        guard shape[2] > 0 else {
            return "boundary hidden_state hidden dimension must be positive"
        }
        return nil
    }

    public func validate() throws {
        if let message = validationErrorMessage {
            throw DistributedStageManifestError.invalidManifest(message)
        }
    }
}

/// Optional state-cache groups declared by a staged export.
///
/// Dense legacy exports use one unnamed KV capacity. Split-cache exports declare named groups, such
/// as a bounded sliding-window ring plus a full-length global cache, so the runtime can allocate each
/// Core AI state tensor at the capacity the exported graph expects.
public struct DistributedStageCacheGroups: Codable, Hashable, Sendable {
    public let strategy: String?
    public let prefillChunk: Int?
    public let groups: [String: DistributedStageCacheGroup]

    enum CodingKeys: String, CodingKey {
        case strategy
        case prefillChunk = "prefill_chunk"
        case groups
    }

    public init(
        strategy: String? = nil,
        prefillChunk: Int? = nil,
        groups: [String: DistributedStageCacheGroup]
    ) {
        self.strategy = strategy
        self.prefillChunk = prefillChunk
        self.groups = groups
    }

    public var capacities: [String: Int] {
        Dictionary(uniqueKeysWithValues: groups.map { ($0.key, $0.value.capacity) })
    }

    public func capacities(forKVCapacity kvCapacity: Int) -> [String: Int] {
        Dictionary(uniqueKeysWithValues: groups.map { name, group in
            let capacity: Int
            if group.usesFixedAllocation(groupName: name, strategy: strategy) {
                capacity = group.capacity
            } else {
                capacity = min(group.capacity, kvCapacity)
            }
            return (name, capacity)
        })
    }

    public var validationErrorMessage: String? {
        if let strategy,
            strategy.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return "cache_groups strategy is empty"
        }
        if let prefillChunk, prefillChunk <= 0 {
            return "cache_groups prefill_chunk must be positive"
        }
        guard !groups.isEmpty else {
            return "cache_groups groups must not be empty"
        }
        for (name, group) in groups {
            let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedName.isEmpty else {
                return "cache_groups contains an empty group name"
            }
            if let message = group.validationErrorMessage(groupName: trimmedName) {
                return message
            }
            if let prefillChunk,
                let slidingWindow = group.slidingWindow,
                prefillChunk > group.capacity - slidingWindow
            {
                return
                    "cache_groups \(trimmedName) capacity \(group.capacity) is too small for sliding_window \(slidingWindow) plus prefill_chunk \(prefillChunk)"
            }
        }
        return nil
    }

    public func validate() throws {
        if let message = validationErrorMessage {
            throw DistributedStageManifestError.invalidManifest(message)
        }
    }
}

public struct DistributedStageCacheGroup: Codable, Hashable, Sendable {
    public let stateNames: [String]
    public let capacity: Int
    public let slidingWindow: Int?

    enum CodingKeys: String, CodingKey {
        case stateNames = "state_names"
        case capacity
        case slidingWindow = "sliding_window"
    }

    public init(
        stateNames: [String],
        capacity: Int,
        slidingWindow: Int? = nil
    ) {
        self.stateNames = stateNames
        self.capacity = capacity
        self.slidingWindow = slidingWindow
    }

    fileprivate func validationErrorMessage(groupName: String) -> String? {
        guard capacity > 0 else {
            return "cache_groups \(groupName) capacity must be positive"
        }
        if let slidingWindow, slidingWindow <= 0 {
            return "cache_groups \(groupName) sliding_window must be positive"
        }
        guard !stateNames.isEmpty else {
            return "cache_groups \(groupName) state_names must not be empty"
        }
        var seen = Set<String>()
        for stateName in stateNames {
            let trimmed = stateName.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                return "cache_groups \(groupName) contains an empty state name"
            }
            guard seen.insert(trimmed).inserted else {
                return "cache_groups \(groupName) duplicate state name \(trimmed)"
            }
        }
        return nil
    }

    fileprivate func usesFixedAllocation(groupName: String, strategy: String?) -> Bool {
        if slidingWindow != nil { return true }
        let normalizedName = groupName.lowercased()
        if normalizedName.contains("recurrent") { return true }
        if strategy?.lowercased().contains("recurrent") == true,
           normalizedName != "full"
        {
            return true
        }
        return false
    }
}

/// Scalar types allowed in declared stage function inputs and outputs.
///
/// Boundary packets stay limited to ``DistributedTensorScalarType`` because those bytes must be
/// host-readable float tensors. Stage function contracts also need Int32 token/position inputs.
public enum DistributedStageIOScalarType: String, Codable, CaseIterable, Sendable {
    case int32
    case float16
    case bfloat16
    case float32

    public init(boundaryScalarType: DistributedTensorScalarType) {
        switch boundaryScalarType {
        case .float16:
            self = .float16
        case .float32:
            self = .float32
        }
    }

    public var isFloatingPoint: Bool {
        switch self {
        case .float16, .bfloat16, .float32:
            return true
        case .int32:
            return false
        }
    }

    public func isCompatible(withBoundaryScalarType boundaryScalarType: DistributedTensorScalarType) -> Bool {
        switch (boundaryScalarType, self) {
        case (.float16, .float16), (.float16, .bfloat16), (.float32, .float32):
            return true
        default:
            return false
        }
    }
}

/// Canonical staged-function tensor names.
public enum DistributedStageIOTensorName: String, Codable, CaseIterable, Sendable {
    case inputIDs = "input_ids"
    case positionIDs = "position_ids"
    case hiddenStates = "hidden_states"
    case blockIDsQ = "block_ids_q"
    case blockIDsKV = "block_ids_kv"
    case kFull = "k_full"
    case vFull = "v_full"
    case kSliding = "k_sliding"
    case vSliding = "v_sliding"
    case hidden
    case logits
}

/// One declared tensor in a staged Core AI function contract.
public struct DistributedStageIOTensor: Codable, Hashable, Sendable {
    public let name: String
    public let shape: [Int]
    public let scalarType: DistributedStageIOScalarType

    enum CodingKeys: String, CodingKey {
        case name
        case shape
        case scalarType = "scalar_type"
    }

    public init(
        name: String,
        shape: [Int],
        scalarType: DistributedStageIOScalarType
    ) {
        self.name = name
        self.shape = shape
        self.scalarType = scalarType
    }

    public init(
        _ name: DistributedStageIOTensorName,
        shape: [Int],
        scalarType: DistributedStageIOScalarType
    ) {
        self.init(name: name.rawValue, shape: shape, scalarType: scalarType)
    }
}

/// Declared input/output contract for one exported stage function.
///
/// This is intentionally runtime-agnostic. A Core AI stage handle can build this from function
/// descriptors, then validate it against the manifest stage role and boundary tensor before
/// accepting any hidden-state packets.
public struct DistributedStageIOContract: Codable, Hashable, Sendable {
    public let functionName: String
    public let inputs: [DistributedStageIOTensor]
    public let outputs: [DistributedStageIOTensor]
    public let stateNames: [String]

    enum CodingKeys: String, CodingKey {
        case functionName = "function_name"
        case inputs
        case outputs
        case stateNames = "state_names"
    }

    public init(
        functionName: String = "main",
        inputs: [DistributedStageIOTensor],
        outputs: [DistributedStageIOTensor],
        stateNames: [String] = []
    ) {
        self.functionName = functionName
        self.inputs = inputs
        self.outputs = outputs
        self.stateNames = stateNames
    }

    public func validate(
        for stage: DistributedStageDescriptor,
        boundaryTensor: DistributedBoundaryTensorSpec?,
        vocabSize: Int? = nil,
        eagleTarget: DistributedEagleTargetContract? = nil
    ) throws {
        guard !Self.trimmed(functionName).isEmpty else {
            throw Self.error(stageID: stage.id, "function_name is empty")
        }
        if let vocabSize, vocabSize <= 0 {
            throw Self.error(stageID: stage.id, "vocab_size must be positive")
        }
        try Self.validateStateNames(stateNames, stageID: stage.id)

        let inputByName = try Self.uniqueTensors(inputs, kind: "input", stageID: stage.id)
        let outputByName = try Self.uniqueTensors(outputs, kind: "output", stageID: stage.id)

        _ = try Self.requireTensor(
            .positionIDs,
            in: inputByName,
            kind: "input",
            stageID: stage.id,
            scalarType: .int32,
            shape: [1, -1])

        switch stage.role {
        case .embeddings:
            try Self.requireOnlyTensors(
                [.inputIDs, .positionIDs],
                in: inputByName,
                kind: "input",
                stage: stage)
            try Self.requireOnlyTensors(
                [.hiddenStates],
                in: outputByName,
                kind: "output",
                stage: stage)
            _ = try Self.requireTensor(
                .inputIDs,
                in: inputByName,
                kind: "input",
                stageID: stage.id,
                scalarType: .int32,
                shape: [1, -1])
            let hiddenOutput = try Self.requireTensor(
                .hiddenStates,
                in: outputByName,
                kind: "output",
                stageID: stage.id)
            try Self.validateHiddenStateTensor(
                hiddenOutput,
                kind: "output",
                stageID: stage.id,
                boundaryTensor: boundaryTensor)
        case .transformerLayers:
            var allowedInputs = Set([
                DistributedStageIOTensorName.hiddenStates.rawValue,
                DistributedStageIOTensorName.positionIDs.rawValue,
                DistributedStageIOTensorName.inputIDs.rawValue,
                DistributedStageIOTensorName.blockIDsQ.rawValue,
                DistributedStageIOTensorName.blockIDsKV.rawValue,
            ])
            if let rope = stage.rope {
                allowedInputs.insert(rope.cosInputName)
                allowedInputs.insert(rope.sinInputName)
            }
            try Self.requireOnlyTensorNames(
                allowedInputs,
                in: inputByName,
                kind: "input",
                stage: stage)
            let isEagleTarget = eagleTarget?.producesArtifacts(for: stage.id) == true
            try Self.requireOnlyTensors(
                isEagleTarget
                    ? [.hiddenStates, .kFull, .vFull, .kSliding, .vSliding]
                    : [.hiddenStates],
                in: outputByName,
                kind: "output",
                stage: stage)
            let hiddenInput = try Self.requireTensor(
                .hiddenStates,
                in: inputByName,
                kind: "input",
                stageID: stage.id)
            let hiddenOutput = try Self.requireTensor(
                .hiddenStates,
                in: outputByName,
                kind: "output",
                stageID: stage.id)
            if let inputIDs = inputByName[DistributedStageIOTensorName.inputIDs.rawValue] {
                try Self.validateInputIDsTensor(inputIDs, kind: "input", stageID: stage.id)
            }
            let blockIDsQ = inputByName[DistributedStageIOTensorName.blockIDsQ.rawValue]
            let blockIDsKV = inputByName[DistributedStageIOTensorName.blockIDsKV.rawValue]
            if blockIDsQ != nil || blockIDsKV != nil
                || stage.prefillExtraInputs.contains(DistributedStageIOTensorName.blockIDsQ.rawValue)
                || stage.prefillExtraInputs.contains(DistributedStageIOTensorName.blockIDsKV.rawValue)
            {
                guard let blockIDsQ, let blockIDsKV else {
                    throw Self.error(
                        stageID: stage.id,
                        "block_ids_q and block_ids_kv must be declared together")
                }
                try Self.validateBlockIDsTensor(blockIDsQ, kind: "input", stageID: stage.id)
                try Self.validateBlockIDsTensor(blockIDsKV, kind: "input", stageID: stage.id)
            }
            try Self.validateHiddenStateTensor(
                hiddenInput,
                kind: "input",
                stageID: stage.id,
                boundaryTensor: boundaryTensor)
            try Self.validateHiddenStateTensor(
                hiddenOutput,
                kind: "output",
                stageID: stage.id,
                boundaryTensor: boundaryTensor)
            if isEagleTarget {
                try Self.validateEagleTargetOutputs(
                    outputByName,
                    hiddenOutput: hiddenOutput,
                    stageID: stage.id)
            }
            if let rope = stage.rope {
                let cosInput = try Self.requireTensor(
                    rope.cosInputName,
                    in: inputByName,
                    kind: "input",
                    stageID: stage.id)
                let sinInput = try Self.requireTensor(
                    rope.sinInputName,
                    in: inputByName,
                    kind: "input",
                    stageID: stage.id)
                try Self.validateRoPETensor(cosInput, rope: rope, stageID: stage.id)
                try Self.validateRoPETensor(sinInput, rope: rope, stageID: stage.id)
            }

        case .finalNormHead:
            try Self.requireOnlyTensors(
                [.hiddenStates, .positionIDs, .inputIDs],
                in: inputByName,
                kind: "input",
                stage: stage)
            let producesFinalHidden = eagleTarget?.producesFinalHidden(for: stage.id) == true
            try Self.requireOnlyTensors(
                producesFinalHidden ? [.logits, .hidden] : [.logits],
                in: outputByName,
                kind: "output",
                stage: stage)
            let hiddenInput = try Self.requireTensor(
                .hiddenStates,
                in: inputByName,
                kind: "input",
                stageID: stage.id)
            if let inputIDs = inputByName[DistributedStageIOTensorName.inputIDs.rawValue] {
                try Self.validateInputIDsTensor(inputIDs, kind: "input", stageID: stage.id)
            }
            let logits = try Self.requireTensor(
                .logits,
                in: outputByName,
                kind: "output",
                stageID: stage.id)
            try Self.validateHiddenStateTensor(
                hiddenInput,
                kind: "input",
                stageID: stage.id,
                boundaryTensor: boundaryTensor)
            try Self.validateLogitsTensor(logits, stageID: stage.id, vocabSize: vocabSize)
            if producesFinalHidden {
                let finalHidden = try Self.requireTensor(
                    .hidden,
                    in: outputByName,
                    kind: "output",
                    stageID: stage.id)
                guard finalHidden.scalarType == .float16 else {
                    throw Self.error(
                        stageID: stage.id,
                        "output hidden scalar_type must be float16 for EAGLE target")
                }
                try Self.validateHiddenStateTensor(
                    finalHidden,
                    kind: "output",
                    stageID: stage.id,
                    boundaryTensor: boundaryTensor)
            }
        }
    }

    private static func validateHiddenStateTensor(
        _ tensor: DistributedStageIOTensor,
        kind: String,
        stageID: String,
        boundaryTensor: DistributedBoundaryTensorSpec?
    ) throws {
        if let boundaryTensor {
            let expectedScalar = DistributedStageIOScalarType(
                boundaryScalarType: boundaryTensor.scalarType)
            guard tensor.scalarType.isCompatible(withBoundaryScalarType: boundaryTensor.scalarType) else {
                throw error(
                    stageID: stageID,
                    "\(kind) hidden_states scalar_type \(tensor.scalarType.rawValue) does not match \(expectedScalar.rawValue)")
            }
        } else {
            guard tensor.scalarType.isFloatingPoint else {
                throw error(
                    stageID: stageID,
                    "\(kind) hidden_states scalar_type \(tensor.scalarType.rawValue) is not floating-point")
            }
        }

        let expectedShape = boundaryTensor?.shape ?? [1, -1, -1]
        guard shape(tensor.shape, matches: expectedShape) else {
            throw error(
                stageID: stageID,
                "\(kind) hidden_states shape \(tensor.shape) does not match \(expectedShape)")
        }
    }

    private static func validateInputIDsTensor(
        _ tensor: DistributedStageIOTensor,
        kind: String,
        stageID: String
    ) throws {
        guard tensor.scalarType == .int32 else {
            throw error(
                stageID: stageID,
                "\(kind) input_ids scalar_type \(tensor.scalarType.rawValue) does not match int32")
        }
        guard shape(tensor.shape, matches: [1, -1]) else {
            throw error(
                stageID: stageID,
                "\(kind) input_ids shape \(tensor.shape) does not match [1, -1]")
        }
    }

    private static func validateBlockIDsTensor(
        _ tensor: DistributedStageIOTensor,
        kind: String,
        stageID: String
    ) throws {
        guard tensor.scalarType == .int32 else {
            throw Self.error(
                stageID: stageID,
                "\(kind) \(tensor.name) scalar_type \(tensor.scalarType.rawValue) does not match int32")
        }
        guard tensor.shape == [1, -1] else {
            throw Self.error(
                stageID: stageID,
                "\(kind) \(tensor.name) shape \(tensor.shape) does not match [1, -1]")
        }
    }

    private static func validateLogitsTensor(
        _ tensor: DistributedStageIOTensor,
        stageID: String,
        vocabSize: Int?
    ) throws {
        guard tensor.scalarType.isFloatingPoint else {
            throw error(
                stageID: stageID,
                "output logits scalar_type \(tensor.scalarType.rawValue) is not floating-point")
        }
        let expectedShape = [1, -1, vocabSize ?? -1]
        guard shape(tensor.shape, matches: expectedShape) else {
            throw error(
                stageID: stageID,
                "output logits shape \(tensor.shape) does not match \(expectedShape)")
        }
    }

    private static func validateEagleTargetOutputs(
        _ outputByName: [String: DistributedStageIOTensor],
        hiddenOutput: DistributedStageIOTensor,
        stageID: String
    ) throws {
        guard hiddenOutput.scalarType == .float16 else {
            throw error(
                stageID: stageID,
                "output hidden_states scalar_type must be float16 for EAGLE target")
        }
        let kFull = try requireTensor(.kFull, in: outputByName, kind: "output", stageID: stageID)
        let vFull = try requireTensor(.vFull, in: outputByName, kind: "output", stageID: stageID)
        let kSliding = try requireTensor(
            .kSliding, in: outputByName, kind: "output", stageID: stageID)
        let vSliding = try requireTensor(
            .vSliding, in: outputByName, kind: "output", stageID: stageID)
        try validateEagleTargetKVPair(
            key: kFull,
            value: vFull,
            hiddenOutput: hiddenOutput,
            label: "full",
            stageID: stageID)
        try validateEagleTargetKVPair(
            key: kSliding,
            value: vSliding,
            hiddenOutput: hiddenOutput,
            label: "sliding",
            stageID: stageID)
    }

    private static func validateEagleTargetKVPair(
        key: DistributedStageIOTensor,
        value: DistributedStageIOTensor,
        hiddenOutput: DistributedStageIOTensor,
        label: String,
        stageID: String
    ) throws {
        for tensor in [key, value] {
            guard tensor.scalarType == .float16 else {
                throw error(
                    stageID: stageID,
                    "output \(tensor.name) scalar_type must be float16")
            }
            guard tensor.shape.count == 4,
                tensor.shape[0] == 1,
                tensor.shape[1] > 0,
                tensor.shape[2] == -1 || tensor.shape[2] > 0,
                tensor.shape[3] > 0
            else {
                throw error(
                    stageID: stageID,
                    "output \(tensor.name) shape \(tensor.shape) must match [1, heads, sequence, head_dim]")
            }
            let hiddenSequence = hiddenOutput.shape[1]
            let kvSequence = tensor.shape[2]
            if hiddenSequence > 0 && kvSequence > 0 && hiddenSequence != kvSequence {
                throw error(
                    stageID: stageID,
                    "output \(tensor.name) sequence dimension must match hidden_states")
            }
        }
        guard key.shape == value.shape else {
            throw error(
                stageID: stageID,
                "EAGLE target \(label) key/value output shapes must match")
        }
    }

    private static func uniqueTensors(
        _ tensors: [DistributedStageIOTensor],
        kind: String,
        stageID: String
    ) throws -> [String: DistributedStageIOTensor] {
        var byName: [String: DistributedStageIOTensor] = [:]
        for tensor in tensors {
            let name = trimmed(tensor.name)
            guard !name.isEmpty else {
                throw error(stageID: stageID, "\(kind) tensor name is empty")
            }
            guard !tensor.shape.isEmpty else {
                throw error(stageID: stageID, "\(kind) \(name) shape is empty")
            }
            guard tensor.shape.allSatisfy({ $0 == -1 || $0 > 0 }) else {
                throw error(
                    stageID: stageID,
                    "\(kind) \(name) shape dimensions must be positive or -1")
            }
            guard byName[name] == nil else {
                throw error(stageID: stageID, "\(kind) \(name) is duplicated")
            }
            byName[name] = tensor
        }
        return byName
    }

    private static func validateStateNames(
        _ stateNames: [String],
        stageID: String
    ) throws {
        var seen: Set<String> = []
        for stateName in stateNames {
            let name = trimmed(stateName)
            guard !name.isEmpty else {
                throw error(stageID: stageID, "state name is empty")
            }
            guard seen.insert(name).inserted else {
                throw error(stageID: stageID, "state \(name) is duplicated")
            }
        }
    }

    private static func requireOnlyTensors(
        _ allowedNames: [DistributedStageIOTensorName],
        in tensors: [String: DistributedStageIOTensor],
        kind: String,
        stage: DistributedStageDescriptor
    ) throws {
        try requireOnlyTensorNames(
            Set(allowedNames.map(\.rawValue)),
            in: tensors,
            kind: kind,
            stage: stage)
    }

    private static func requireOnlyTensorNames(
        _ allowed: Set<String>,
        in tensors: [String: DistributedStageIOTensor],
        kind: String,
        stage: DistributedStageDescriptor
    ) throws {
        for name in tensors.keys.sorted() where !allowed.contains(name) {
            throw error(
                stageID: stage.id,
                "\(stage.role.rawValue) stage must not declare \(kind) \(name)")
        }
    }

    private static func requireTensor(
        _ name: DistributedStageIOTensorName,
        in tensors: [String: DistributedStageIOTensor],
        kind: String,
        stageID: String,
        scalarType: DistributedStageIOScalarType? = nil,
        shape: [Int]? = nil
    ) throws -> DistributedStageIOTensor {
        guard let tensor = tensors[name.rawValue] else {
            throw error(stageID: stageID, "\(kind) \(name.rawValue) is required")
        }
        if let scalarType {
            guard tensor.scalarType == scalarType else {
                throw error(
                    stageID: stageID,
                    "\(kind) \(name.rawValue) scalar_type \(tensor.scalarType.rawValue) does not match \(scalarType.rawValue)")
            }
        }
        if let shape {
            guard Self.shape(tensor.shape, matches: shape) else {
                throw error(
                    stageID: stageID,
                    "\(kind) \(name.rawValue) shape \(tensor.shape) does not match \(shape)")
            }
        }
        return tensor
    }

    private static func requireTensor(
        _ name: String,
        in tensors: [String: DistributedStageIOTensor],
        kind: String,
        stageID: String
    ) throws -> DistributedStageIOTensor {
        guard let tensor = tensors[name] else {
            throw error(stageID: stageID, "\(kind) \(name) is required")
        }
        return tensor
    }

    private static func validateRoPETensor(
        _ tensor: DistributedStageIOTensor,
        rope: DistributedStageRoPEInputSpec,
        stageID: String
    ) throws {
        guard tensor.scalarType.isFloatingPoint else {
            throw error(
                stageID: stageID,
                "input \(tensor.name) scalar_type \(tensor.scalarType.rawValue) is not floating-point")
        }
        guard shape(tensor.shape, matches: [1, -1, rope.headDim]) else {
            throw error(
                stageID: stageID,
                "input \(tensor.name) shape \(tensor.shape) does not match [1, -1, \(rope.headDim)]")
        }
    }

    private static func shape(_ actual: [Int], matches expected: [Int]) -> Bool {
        guard actual.count == expected.count else { return false }
        for (actualDim, expectedDim) in zip(actual, expected) {
            if expectedDim == -1 {
                guard actualDim == -1 || actualDim > 0 else { return false }
            } else if actualDim != expectedDim {
                return false
            }
        }
        return true
    }

    private static func error(
        stageID: String,
        _ reason: String
    ) -> DistributedStageExecutionError {
        .invalidStageIOContract(stageID: stageID, reason: reason)
    }

    private static func trimmed(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Normalized staged manifest used by the CLI planner, same-machine harness, and future workers.
public struct DistributedStageManifest: Hashable, Sendable {
    public static let currentSchema = "caix.cluster.stage_manifest.v0"

    public let schema: String?
    public let modelName: String
    public let totalLayerCount: Int
    public let totalLayerCountDerived: Bool
    public let stages: [DistributedStageManifestStage]
    public let boundaryTensor: DistributedBoundaryTensorSpec?
    public let positionMode: DistributedPositionMode
    public let cacheGroups: DistributedStageCacheGroups?
    public let runtimeMemory: DistributedRuntimeMemoryContract?
    public let eagleTarget: DistributedEagleTargetContract?
    public let runtimePlan: DistributedStagePlan

    public init(
        schema: String? = Self.currentSchema,
        modelName: String,
        totalLayerCount: Int,
        totalLayerCountDerived: Bool = false,
        stages: [DistributedStageManifestStage],
        boundaryTensor: DistributedBoundaryTensorSpec? = nil,
        positionMode: DistributedPositionMode = .current,
        cacheGroups: DistributedStageCacheGroups? = nil,
        runtimeMemory: DistributedRuntimeMemoryContract? = nil,
        eagleTarget: DistributedEagleTargetContract? = nil
    ) throws {
        self.schema = schema
        self.modelName = modelName
        self.totalLayerCount = totalLayerCount
        self.totalLayerCountDerived = totalLayerCountDerived
        self.stages = stages
        self.boundaryTensor = boundaryTensor
        self.positionMode = positionMode
        self.cacheGroups = cacheGroups
        self.runtimeMemory = runtimeMemory
        self.eagleTarget = eagleTarget
        try boundaryTensor?.validate()
        try cacheGroups?.validate()
        try runtimeMemory?.validate(stages: stages)
        try eagleTarget?.validate(stages: stages)
        self.runtimePlan = DistributedStagePlan(
            modelName: modelName,
            totalLayerCount: totalLayerCount,
            stages: stages.map { $0.descriptor() },
            workers: [],
            boundaryTensor: boundaryTensor,
            positionMode: positionMode)
        try self.runtimePlan.validate()
    }

    public static func load(
        from url: URL,
        defaultModelName: String? = nil,
        requireClusterBlock: Bool = false
    ) throws -> DistributedStageManifest {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DistributedStageManifestError.fileNotFound(url.path)
        }
        let data = try Data(contentsOf: url)
        return try decode(
            from: data,
            sourceURL: url,
            baseURL: url.deletingLastPathComponent(),
            defaultModelName: defaultModelName,
            requireClusterBlock: requireClusterBlock)
    }

    public static func decode(
        from data: Data,
        sourceURL: URL? = nil,
        baseURL: URL? = nil,
        defaultModelName: String? = nil,
        requireClusterBlock: Bool = false
    ) throws -> DistributedStageManifest {
        let sourceDescription = sourceURL?.path ?? "<memory>"
        let root: RawDistributedStageManifestRoot
        do {
            root = try JSONDecoder().decode(RawDistributedStageManifestRoot.self, from: data)
        } catch {
            throw DistributedStageManifestError.invalidJSON("\(sourceDescription): \(error)")
        }

        let usesClusterBlock = root.cluster != nil
        let body: RawDistributedStageManifestBody
        if let cluster = root.cluster {
            body = cluster
        } else if requireClusterBlock {
            throw DistributedStageManifestError.missingClusterBlock(sourceDescription)
        } else {
            body = root.asBody
        }

        let schema = body.schema ?? (usesClusterBlock ? nil : root.schema)
        if let schema, schema != currentSchema {
            throw DistributedStageManifestError.invalidSchema(schema)
        }

        let modelName = firstNonEmpty([
            body.modelName, body.model, root.modelName, root.model, root.name, defaultModelName,
            baseURL?.lastPathComponent,
        ])
        guard let modelName else {
            throw DistributedStageManifestError.invalidManifest("model_name is missing")
        }

        guard let rawStages = body.stages, !rawStages.isEmpty else {
            throw DistributedStageManifestError.missingStages(sourceDescription)
        }
        let stages = try rawStages.enumerated().map { index, rawStage in
            try normalizeStage(rawStage, index: index, baseURL: baseURL)
        }

        let explicitTotalLayerCount = body.totalLayerCount?.value ?? body.totalLayers?.value
            ?? root.totalLayerCount?.value ?? root.totalLayers?.value
        if let explicitTotalLayerCount, explicitTotalLayerCount <= 0 {
            throw DistributedStageManifestError.invalidManifest(
                "total_layer_count must be positive")
        }

        let totalLayerCount: Int
        let totalLayerCountDerived: Bool
        if let explicitTotalLayerCount {
            totalLayerCount = explicitTotalLayerCount
            totalLayerCountDerived = false
        } else {
            totalLayerCount = try deriveTotalLayerCount(from: stages)
            totalLayerCountDerived = true
        }

        let cacheGroups = body.cacheGroups ?? root.cacheGroups
        let rawEagleTarget = body.eagleTarget ?? root.eagleTarget
        let eagleTarget = try rawEagleTarget?.normalized(
            stages: stages,
            cacheGroups: cacheGroups)

        return try DistributedStageManifest(
            schema: schema,
            modelName: modelName,
            totalLayerCount: totalLayerCount,
            totalLayerCountDerived: totalLayerCountDerived,
            stages: stages,
            boundaryTensor: body.boundary?.hiddenState ?? body.boundaryTensor
                ?? root.boundary?.hiddenState ?? root.boundaryTensor,
            positionMode: body.positionMode ?? body.positionModeCamel ?? root.positionMode
                ?? root.positionModeCamel ?? .current,
            cacheGroups: cacheGroups,
            runtimeMemory: body.runtimeMemory ?? root.runtimeMemory,
            eagleTarget: eagleTarget)
    }

    private static func normalizeStage(
        _ rawStage: RawDistributedStageManifestStage,
        index: Int,
        baseURL: URL?
    ) throws -> DistributedStageManifestStage {
        let fallbackID = "stage-\(index + 1)"
        let id = firstNonEmpty([rawStage.id, rawStage.name]) ?? fallbackID
        let assetName = firstNonEmpty([
            rawStage.path, rawStage.bundle, rawStage.bundlePath, rawStage.aimodel,
        ])
        guard let assetName else {
            throw DistributedStageManifestError.missingStageField(
                stageID: id, field: "bundle")
        }
        guard let rawRole = firstNonEmpty([rawStage.role, rawStage.kind]) else {
            throw DistributedStageManifestError.missingStageField(stageID: id, field: "role")
        }
        guard let role = DistributedStageRole(rawValue: rawRole) else {
            throw DistributedStageManifestError.invalidStageField(
                stageID: id,
                field: "role",
                reason: "must be embeddings, transformer_layers, or final_norm_head")
        }
        guard let rawLayerSpec = rawStage.layers ?? rawStage.layerRange else {
            throw DistributedStageManifestError.missingStageField(stageID: id, field: "layers")
        }

        let layerSpec: DistributedStageLayerSpec
        if role.requiresLayerRange {
            guard let range = rawLayerSpec.layerRange else {
                throw DistributedStageManifestError.invalidStageField(
                    stageID: id,
                    field: "layers",
                    reason: "transformer_layers stages need [lower, upper]")
            }
            layerSpec = .range(range)
        } else {
            guard let label = rawLayerSpec.label?.trimmingCharacters(in: .whitespacesAndNewlines),
                !label.isEmpty
            else {
                throw DistributedStageManifestError.invalidStageField(
                    stageID: id,
                    field: "layers",
                    reason: "\(role.rawValue) stages need a label")
            }
            layerSpec = .label(label)
        }

        let memoryGB =
            rawStage.memoryGB?.value
            ?? rawStage.memoryGBCamel?.value
            ?? rawStage.requiredGB?.value
            ?? rawStage.requiredMemoryGB?.value
            ?? rawStage.requiredMemoryGBCamel?.value
            ?? rawStage.estimatedMemoryGB?.value
        guard let memoryGB else {
            throw DistributedStageManifestError.missingStageField(
                stageID: id, field: "memory_gb")
        }
        guard memoryGB > 0 else {
            throw DistributedStageManifestError.invalidStageField(
                stageID: id, field: "memory_gb", reason: "must be positive")
        }

        if let reason = rawStage.functionMap?.validationErrorMessage {
            throw DistributedStageManifestError.invalidStageField(
                stageID: id, field: "function_map", reason: reason)
        }
        if let rope = rawStage.rope {
            if role != .transformerLayers {
                throw DistributedStageManifestError.invalidStageField(
                    stageID: id,
                    field: "rope",
                    reason: "rope inputs are only valid for transformer_layers stages")
            }
            if let reason = rope.validationErrorMessage {
                throw DistributedStageManifestError.invalidStageField(
                    stageID: id, field: "rope", reason: reason)
            }
        }
        let decodeAssetName = firstNonEmpty([
            rawStage.decodeAssetName, rawStage.decodeAsset, rawStage.decodeBundle,
        ])
        let vocabSize = rawStage.vocabSize?.value
        if let vocabSize, vocabSize <= 0 {
            throw DistributedStageManifestError.invalidStageField(
                stageID: id, field: "vocab_size", reason: "must be positive")
        }
        let prefillExtraInputs = try validatePrefillExtraInputs(
            rawStage.prefillExtraInputs ?? [],
            role: role,
            stageID: id)

        return DistributedStageManifestStage(
            id: id,
            role: role,
            layerSpec: layerSpec,
            assetName: assetName,
            resolvedAssetPath: resolveAssetPath(assetName, baseURL: baseURL),
            decodeAssetName: decodeAssetName,
            resolvedDecodeAssetPath: decodeAssetName.flatMap {
                resolveAssetPath($0, baseURL: baseURL)
            },
            functionMap: rawStage.functionMap,
            vocabSize: vocabSize,
            prefillExtraInputs: prefillExtraInputs,
            memoryGB: memoryGB,
            rope: rawStage.rope)
    }

    private static func validatePrefillExtraInputs(
        _ values: [String],
        role: DistributedStageRole,
        stageID: String
    ) throws -> [String] {
        guard !values.isEmpty else { return [] }
        guard role == .transformerLayers else {
            throw DistributedStageManifestError.invalidStageField(
                stageID: stageID,
                field: "prefill_extra_inputs",
                reason: "prefill_extra_inputs are only valid for transformer_layers stages")
        }

        let supported = Set([
            DistributedStageIOTensorName.blockIDsQ.rawValue,
            DistributedStageIOTensorName.blockIDsKV.rawValue,
        ])
        var seen = Set<String>()
        var normalized: [String] = []
        for value in values {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                throw DistributedStageManifestError.invalidStageField(
                    stageID: stageID,
                    field: "prefill_extra_inputs",
                    reason: "names must be non-empty")
            }
            guard supported.contains(trimmed) else {
                throw DistributedStageManifestError.invalidStageField(
                    stageID: stageID,
                    field: "prefill_extra_inputs",
                    reason: "unsupported prefill input \(trimmed)")
            }
            guard seen.insert(trimmed).inserted else {
                throw DistributedStageManifestError.invalidStageField(
                    stageID: stageID,
                    field: "prefill_extra_inputs",
                    reason: "names must be unique")
            }
            normalized.append(trimmed)
        }
        let required = supported
        guard Set(normalized) == required else {
            throw DistributedStageManifestError.invalidStageField(
                stageID: stageID,
                field: "prefill_extra_inputs",
                reason: "block_ids_q and block_ids_kv must be listed together")
        }
        return normalized
    }

    private static func deriveTotalLayerCount(
        from stages: [DistributedStageManifestStage]
    ) throws -> Int {
        let ranges = stages.compactMap { stage -> DistributedLayerRange? in
            guard stage.role == .transformerLayers else { return nil }
            return stage.layerRange
        }
        guard let last = ranges.last else {
            throw DistributedStageManifestError.invalidManifest(
                "cluster plan needs at least one transformer_layers stage")
        }
        return last.upperBound
    }

    private static func resolveAssetPath(_ assetName: String, baseURL: URL?) -> String? {
        let expanded = (assetName as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL.path
        }
        guard let baseURL else { return nil }
        return baseURL.appendingPathComponent(expanded).standardizedFileURL.path
    }

    private static func firstNonEmpty(_ values: [String?]) -> String? {
        for value in values {
            guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
                !value.isEmpty
            else { continue }
            return value
        }
        return nil
    }
}

public enum DistributedStageManifestError: Error, Equatable, Sendable, CustomStringConvertible {
    case fileNotFound(String)
    case invalidJSON(String)
    case missingClusterBlock(String)
    case missingStages(String)
    case missingStageField(stageID: String, field: String)
    case invalidStageField(stageID: String, field: String, reason: String)
    case invalidSchema(String)
    case invalidManifest(String)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "Distributed stage manifest not found: \(path)"
        case .invalidJSON(let message):
            return "Invalid distributed stage manifest JSON: \(message)"
        case .missingClusterBlock(let path):
            return "\(path) does not include cluster.stages"
        case .missingStages(let path):
            return "\(path) has no stage metadata"
        case .missingStageField(let stageID, let field):
            return "Stage \(stageID) is missing \(field) metadata"
        case .invalidStageField(let stageID, let field, let reason):
            return "Stage \(stageID) has invalid \(field) metadata: \(reason)"
        case .invalidSchema(let schema):
            return "Unsupported distributed stage manifest schema: \(schema)"
        case .invalidManifest(let message):
            return "Invalid distributed stage manifest: \(message)"
        }
    }
}

private struct RawDistributedStageManifestRoot: Decodable {
    let schema: String?
    let cluster: RawDistributedStageManifestBody?
    let model: String?
    let modelName: String?
    let name: String?
    let totalLayerCount: FlexibleInt?
    let totalLayers: FlexibleInt?
    let stages: [RawDistributedStageManifestStage]?
    let boundary: RawDistributedBoundaryBlock?
    let boundaryTensor: DistributedBoundaryTensorSpec?
    let positionMode: DistributedPositionMode?
    let positionModeCamel: DistributedPositionMode?
    let cacheGroups: DistributedStageCacheGroups?
    let runtimeMemory: DistributedRuntimeMemoryContract?
    let eagleTarget: RawDistributedEagleTargetContract?

    enum CodingKeys: String, CodingKey {
        case schema
        case cluster
        case model
        case modelName = "model_name"
        case name
        case totalLayerCount = "total_layer_count"
        case totalLayers = "total_layers"
        case stages
        case boundary
        case boundaryTensor = "boundary_tensor"
        case positionMode = "position_mode"
        case positionModeCamel = "positionMode"
        case cacheGroups = "cache_groups"
        case runtimeMemory = "runtime_memory"
        case eagleTarget = "eagle_target"
    }

    var asBody: RawDistributedStageManifestBody {
        RawDistributedStageManifestBody(
            schema: schema,
            model: model,
            modelName: modelName,
            totalLayerCount: totalLayerCount,
            totalLayers: totalLayers,
            stages: stages,
            boundary: boundary,
            boundaryTensor: boundaryTensor,
            positionMode: positionMode,
            positionModeCamel: positionModeCamel,
            cacheGroups: cacheGroups,
            runtimeMemory: runtimeMemory,
            eagleTarget: eagleTarget)
    }
}

private struct RawDistributedStageManifestBody: Decodable {
    let schema: String?
    let model: String?
    let modelName: String?
    let totalLayerCount: FlexibleInt?
    let totalLayers: FlexibleInt?
    let stages: [RawDistributedStageManifestStage]?
    let boundary: RawDistributedBoundaryBlock?
    let boundaryTensor: DistributedBoundaryTensorSpec?
    let positionMode: DistributedPositionMode?
    let positionModeCamel: DistributedPositionMode?
    let cacheGroups: DistributedStageCacheGroups?
    let runtimeMemory: DistributedRuntimeMemoryContract?
    let eagleTarget: RawDistributedEagleTargetContract?

    enum CodingKeys: String, CodingKey {
        case schema
        case model
        case modelName = "model_name"
        case totalLayerCount = "total_layer_count"
        case totalLayers = "total_layers"
        case stages
        case boundary
        case boundaryTensor = "boundary_tensor"
        case positionMode = "position_mode"
        case positionModeCamel = "positionMode"
        case cacheGroups = "cache_groups"
        case runtimeMemory = "runtime_memory"
        case eagleTarget = "eagle_target"
    }
}

private struct RawDistributedBoundaryBlock: Decodable {
    let hiddenState: DistributedBoundaryTensorSpec?

    enum CodingKeys: String, CodingKey {
        case hiddenState = "hidden_state"
    }
}

private struct RawDistributedStageManifestStage: Decodable {
    let id: String?
    let name: String?
    let path: String?
    let bundle: String?
    let bundlePath: String?
    let aimodel: String?
    let role: String?
    let kind: String?
    let layers: DistributedStageLayerSpec?
    let layerRange: DistributedStageLayerSpec?
    let memoryGB: FlexibleDouble?
    let memoryGBCamel: FlexibleDouble?
    let requiredGB: FlexibleDouble?
    let requiredMemoryGB: FlexibleDouble?
    let requiredMemoryGBCamel: FlexibleDouble?
    let estimatedMemoryGB: FlexibleDouble?
    let decodeAssetName: String?
    let decodeAsset: String?
    let decodeBundle: String?
    let functionMap: DistributedStageFunctionMap?
    let vocabSize: FlexibleInt?
    let prefillExtraInputs: [String]?
    let rope: DistributedStageRoPEInputSpec?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case path
        case bundle
        case bundlePath = "bundle_path"
        case aimodel
        case role
        case kind
        case layers
        case layerRange = "layer_range"
        case memoryGB = "memory_gb"
        case memoryGBCamel = "memoryGB"
        case requiredGB = "required_gb"
        case requiredMemoryGB = "required_memory_gb"
        case requiredMemoryGBCamel = "requiredMemoryGB"
        case estimatedMemoryGB = "estimated_memory_gb"
        case decodeAssetName = "decode_asset_name"
        case decodeAsset = "decode_asset"
        case decodeBundle = "decode_bundle"
        case functionMap = "function_map"
        case vocabSize = "vocab_size"
        case prefillExtraInputs = "prefill_extra_inputs"
        case rope
    }
}

private struct DistributedLayerRangeObject: Decodable {
    let range: DistributedLayerRange

    enum CodingKeys: String, CodingKey {
        case lowerBound = "lower_bound"
        case upperBound = "upper_bound"
        case lower
        case upper
        case start
        case end
        case from
        case to
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let lower =
            try container.decodeIfPresent(FlexibleInt.self, forKey: .lowerBound)?.value
            ?? container.decodeIfPresent(FlexibleInt.self, forKey: .lower)?.value
            ?? container.decodeIfPresent(FlexibleInt.self, forKey: .start)?.value
            ?? container.decodeIfPresent(FlexibleInt.self, forKey: .from)?.value
        let upper =
            try container.decodeIfPresent(FlexibleInt.self, forKey: .upperBound)?.value
            ?? container.decodeIfPresent(FlexibleInt.self, forKey: .upper)?.value
            ?? container.decodeIfPresent(FlexibleInt.self, forKey: .end)?.value
            ?? container.decodeIfPresent(FlexibleInt.self, forKey: .to)?.value
        guard let lower, let upper else {
            throw DecodingError.keyNotFound(
                CodingKeys.lowerBound,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "layer range object needs lower_bound and upper_bound"))
        }
        let range = DistributedLayerRange(lowerBound: lower, upperBound: upper)
        guard range.isValid else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "layer range must be non-empty and non-negative"))
        }
        self.range = range
    }
}

private struct FlexibleInt: Decodable {
    let value: Int

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self.value = value
            return
        }
        if let value = try? container.decode(String.self),
            let parsed = Int(value.trimmingCharacters(in: .whitespacesAndNewlines))
        {
            self.value = parsed
            return
        }
        throw DecodingError.typeMismatch(
            Int.self,
            DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "expected integer or integer string"))
    }
}

private struct FlexibleDouble: Decodable {
    let value: Double

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Double.self) {
            self.value = value
            return
        }
        if let value = try? container.decode(String.self) {
            let cleaned = value
                .replacingOccurrences(of: "GB", with: "", options: .caseInsensitive)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let parsed = Double(cleaned) {
                self.value = parsed
                return
            }
        }
        throw DecodingError.typeMismatch(
            Double.self,
            DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "expected number or GB string"))
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

    public var elementCount: Int? {
        expectedByteCount.map { $0 / scalarType.byteWidth }
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

    public init(metadata: DistributedHiddenStatePacketMetadata, float16Values: [Float16]) throws {
        guard metadata.scalarType == .float16 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "float16 payload requires scalar_type float16")
        }
        try Self.validateElementCount(float16Values.count, metadata: metadata)
        try self.init(metadata: metadata, payload: Self.encodeFloat16Payload(float16Values))
    }

    public init(metadata: DistributedHiddenStatePacketMetadata, float32Values: [Float]) throws {
        guard metadata.scalarType == .float32 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "float32 payload requires scalar_type float32")
        }
        try Self.validateElementCount(float32Values.count, metadata: metadata)
        try self.init(metadata: metadata, payload: Self.encodeFloat32Payload(float32Values))
    }

    public func float16Values() throws -> [Float16] {
        guard metadata.scalarType == .float16 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "float16 decode requires scalar_type float16")
        }
        return try Self.decodeFloat16Payload(payload)
    }

    public func float32Values() throws -> [Float] {
        guard metadata.scalarType == .float32 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "float32 decode requires scalar_type float32")
        }
        return try Self.decodeFloat32Payload(payload)
    }

    public func floatValuesAsFloat32() throws -> [Float] {
        switch metadata.scalarType {
        case .float16:
            return try float16Values().map(Float.init)
        case .float32:
            return try float32Values()
        }
    }

    public static func encodeFloat16Payload(_ values: [Float16]) -> [UInt8] {
        var payload: [UInt8] = []
        payload.reserveCapacity(values.count * MemoryLayout<UInt16>.size)
        for value in values {
            appendLittleEndian(value.bitPattern, to: &payload)
        }
        return payload
    }

    public static func encodeFloat32Payload(_ values: [Float]) -> [UInt8] {
        var payload: [UInt8] = []
        payload.reserveCapacity(values.count * MemoryLayout<UInt32>.size)
        for value in values {
            appendLittleEndian(value.bitPattern, to: &payload)
        }
        return payload
    }

    public static func decodeFloat16Payload(_ payload: [UInt8]) throws -> [Float16] {
        guard payload.count.isMultiple(of: MemoryLayout<UInt16>.size) else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "float16 payload byte count is not aligned")
        }
        return stride(from: 0, to: payload.count, by: MemoryLayout<UInt16>.size).map { index in
            Float16(bitPattern: UInt16(littleEndianBytes: payload[index..<(index + 2)]))
        }
    }

    public static func decodeFloat32Payload(_ payload: [UInt8]) throws -> [Float] {
        guard payload.count.isMultiple(of: MemoryLayout<UInt32>.size) else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "float32 payload byte count is not aligned")
        }
        return stride(from: 0, to: payload.count, by: MemoryLayout<UInt32>.size).map { index in
            Float(bitPattern: UInt32(littleEndianBytes: payload[index..<(index + 4)]))
        }
    }

    private static func validateElementCount(
        _ count: Int,
        metadata: DistributedHiddenStatePacketMetadata
    ) throws {
        guard metadata.elementCount == count else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "payload element count does not match metadata shape")
        }
    }

    private static func appendLittleEndian<T: FixedWidthInteger>(_ value: T, to payload: inout [UInt8]) {
        var remaining = value.littleEndian
        for _ in 0..<MemoryLayout<T>.size {
            payload.append(UInt8(truncatingIfNeeded: remaining))
            remaining >>= 8
        }
    }
}

/// Soft-token rows to overwrite into an embedding-stage hidden-state output.
///
/// `positionStart` is an absolute token position in the request. `shape` is `[rows, hidden]`.
/// Values are assumed to already be in model hidden-state scale; the runtime copies them as-is.
public struct DistributedSoftTokenSplice: Codable, Hashable, Sendable {
    public let positionStart: Int
    public let shape: [Int]
    public let scalarType: DistributedTensorScalarType
    public let payload: [UInt8]

    enum CodingKeys: String, CodingKey {
        case positionStart = "position_start"
        case shape
        case scalarType = "scalar_type"
        case payload
    }

    public init(
        positionStart: Int,
        shape: [Int],
        scalarType: DistributedTensorScalarType,
        payload: [UInt8]
    ) throws {
        self.positionStart = positionStart
        self.shape = shape
        self.scalarType = scalarType
        self.payload = payload
        try validate()
    }

    public init(
        positionStart: Int,
        rows: Int,
        hiddenSize: Int,
        float16Values: [Float16]
    ) throws {
        try self.init(
            positionStart: positionStart,
            shape: [rows, hiddenSize],
            scalarType: .float16,
            payload: DistributedHiddenStatePacket.encodeFloat16Payload(float16Values))
    }

    public init(
        positionStart: Int,
        rows: Int,
        hiddenSize: Int,
        float32Values: [Float]
    ) throws {
        try self.init(
            positionStart: positionStart,
            shape: [rows, hiddenSize],
            scalarType: .float32,
            payload: DistributedHiddenStatePacket.encodeFloat32Payload(float32Values))
    }

    public var rowCount: Int { shape.count == 2 ? shape[0] : 0 }
    public var hiddenSize: Int { shape.count == 2 ? shape[1] : 0 }
    public var positionEnd: Int { positionStart + rowCount }

    public func float16Values() throws -> [Float16] {
        guard scalarType == .float16 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "soft_token_splice float16 decode requires scalar_type float16")
        }
        return try DistributedHiddenStatePacket.decodeFloat16Payload(payload)
    }

    public func float32Values() throws -> [Float] {
        guard scalarType == .float32 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "soft_token_splice float32 decode requires scalar_type float32")
        }
        return try DistributedHiddenStatePacket.decodeFloat32Payload(payload)
    }

    private func validate() throws {
        guard positionStart >= 0 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "soft_token_splice position_start must be non-negative")
        }
        guard shape.count == 2, shape[0] > 0, shape[1] > 0 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "soft_token_splice shape must be [rows, hidden] with positive dimensions")
        }
        var expectedBytes = scalarType.byteWidth
        for dimension in shape {
            let next = expectedBytes.multipliedReportingOverflow(by: dimension)
            guard !next.overflow else {
                throw DistributedRuntimeValidationError.invalidPacket(
                    "soft_token_splice byte count overflow")
            }
            expectedBytes = next.partialValue
        }
        guard payload.count == expectedBytes else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "soft_token_splice payload byte count does not match shape")
        }
    }
}

private extension UInt16 {
    init(littleEndianBytes bytes: ArraySlice<UInt8>) {
        self = bytes.enumerated().reduce(0) { partial, byte in
            partial | (UInt16(byte.element) << UInt16(byte.offset * 8))
        }
    }
}

private extension UInt32 {
    init(littleEndianBytes bytes: ArraySlice<UInt8>) {
        self = bytes.enumerated().reduce(0) { partial, byte in
            partial | (UInt32(byte.element) << UInt32(byte.offset * 8))
        }
    }
}

/// Static plan for local or remote stage execution.
public struct DistributedStagePlan: Codable, Hashable, Sendable {
    public let modelName: String
    public let totalLayerCount: Int
    public let stages: [DistributedStageDescriptor]
    public let workers: [DistributedWorkerEndpoint]
    public let boundaryTensor: DistributedBoundaryTensorSpec?
    public let positionMode: DistributedPositionMode

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case totalLayerCount = "total_layer_count"
        case stages
        case workers
        case boundaryTensor = "boundary_tensor"
        case positionMode = "position_mode"
    }

    public init(
        modelName: String,
        totalLayerCount: Int,
        stages: [DistributedStageDescriptor],
        workers: [DistributedWorkerEndpoint] = [],
        boundaryTensor: DistributedBoundaryTensorSpec? = nil,
        positionMode: DistributedPositionMode = .current
    ) {
        self.modelName = modelName
        self.totalLayerCount = totalLayerCount
        self.stages = stages
        self.workers = workers
        self.boundaryTensor = boundaryTensor
        self.positionMode = positionMode
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelName = try container.decode(String.self, forKey: .modelName)
        self.totalLayerCount = try container.decode(Int.self, forKey: .totalLayerCount)
        self.stages = try container.decode([DistributedStageDescriptor].self, forKey: .stages)
        self.workers = try container.decodeIfPresent(
            [DistributedWorkerEndpoint].self,
            forKey: .workers) ?? []
        self.boundaryTensor = try container.decodeIfPresent(
            DistributedBoundaryTensorSpec.self,
            forKey: .boundaryTensor)
        self.positionMode = try container.decodeIfPresent(
            DistributedPositionMode.self,
            forKey: .positionMode) ?? .current
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelName, forKey: .modelName)
        try container.encode(totalLayerCount, forKey: .totalLayerCount)
        try container.encode(stages, forKey: .stages)
        try container.encode(workers, forKey: .workers)
        try container.encodeIfPresent(boundaryTensor, forKey: .boundaryTensor)
        try container.encode(positionMode, forKey: .positionMode)
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

    public func integrityHash() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

public struct DistributedWorkerHello: Codable, Hashable, Sendable {
    public let stage: DistributedStageDescriptor
    public let hiddenSize: Int?
    public let boundaryScalarType: DistributedTensorScalarType?
    public let cacheContract: String?
    public let acceptsTokenIDs: Bool?
    public let planIntegrityHash: String
    public let freeMemoryBytes: UInt64?
    public let computeUnit: String?
    public let labels: [String: String]

    enum CodingKeys: String, CodingKey {
        case stage
        case hiddenSize = "hidden_size"
        case boundaryScalarType = "boundary_scalar_type"
        case cacheContract = "cache_contract"
        case acceptsTokenIDs = "accepts_token_ids"
        case planIntegrityHash = "plan_integrity_hash"
        case freeMemoryBytes = "free_memory_bytes"
        case computeUnit = "compute_unit"
        case labels
    }

    public init(
        stage: DistributedStageDescriptor,
        hiddenSize: Int? = nil,
        boundaryScalarType: DistributedTensorScalarType? = nil,
        cacheContract: String? = nil,
        acceptsTokenIDs: Bool? = nil,
        planIntegrityHash: String,
        freeMemoryBytes: UInt64? = nil,
        computeUnit: String? = nil,
        labels: [String: String] = [:]
    ) {
        self.stage = stage
        self.hiddenSize = hiddenSize
        self.boundaryScalarType = boundaryScalarType
        self.cacheContract = cacheContract
        self.acceptsTokenIDs = acceptsTokenIDs
        self.planIntegrityHash = planIntegrityHash
        self.freeMemoryBytes = freeMemoryBytes
        self.computeUnit = computeUnit
        self.labels = labels
    }

    public func validate(
        against plan: DistributedStagePlan
    ) throws {
        try validate(against: plan, expectedPlanIntegrityHash: try plan.integrityHash())
    }

    public func validate(
        against plan: DistributedStagePlan,
        expectedPlanIntegrityHash: String
    ) throws {
        guard planIntegrityHash == expectedPlanIntegrityHash else {
            throw DistributedStageExecutionError.invalidWorkerHello(
                "plan_integrity_hash mismatch")
        }
        guard let expectedStage = plan.stage(id: stage.id) else {
            throw DistributedStageExecutionError.invalidWorkerHello(
                "unknown stage_id \(stage.id)")
        }
        guard expectedStage == stage else {
            throw DistributedStageExecutionError.invalidWorkerHello(
                "stage descriptor does not match plan for \(stage.id)")
        }

        if let hiddenSize,
            let expectedHidden = plan.boundaryTensor?.shape.last,
            expectedHidden > 0,
            hiddenSize != expectedHidden
        {
            throw DistributedStageExecutionError.invalidWorkerHello(
                "hidden_size \(hiddenSize) does not match boundary tensor hidden size \(expectedHidden)")
        }
        if let boundaryScalarType,
            let expectedScalarType = plan.boundaryTensor?.scalarType,
            boundaryScalarType != expectedScalarType
        {
            throw DistributedStageExecutionError.invalidWorkerHello(
                "boundary_scalar_type \(boundaryScalarType.rawValue) does not match boundary tensor \(expectedScalarType.rawValue)")
        }
    }
}

public struct DistributedWorkerHelloAck: Codable, Hashable, Sendable {
    public let accepted: Bool
    public let stageID: String
    public let reason: String?
    public let planIntegrityHash: String?

    enum CodingKeys: String, CodingKey {
        case accepted
        case stageID = "stage_id"
        case reason
        case planIntegrityHash = "plan_integrity_hash"
    }

    public init(
        accepted: Bool,
        stageID: String,
        reason: String? = nil,
        planIntegrityHash: String? = nil
    ) {
        self.accepted = accepted
        self.stageID = stageID
        self.reason = reason
        self.planIntegrityHash = planIntegrityHash
    }

    public func validate(against plan: DistributedStagePlan? = nil) throws {
        guard !stageID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidControlFrame("stage_id is empty")
        }
        if let plan {
            guard plan.stage(id: stageID) != nil else {
                throw DistributedStageExecutionError.invalidControlFrame("unknown stage_id \(stageID)")
            }
            if accepted {
                let expectedHash = try plan.integrityHash()
                guard planIntegrityHash == expectedHash else {
                    throw DistributedStageExecutionError.invalidControlFrame(
                        "plan_integrity_hash mismatch")
                }
            }
        }
        if !accepted {
            guard let reason, !reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "rejected hello_ack needs a reason")
            }
        }
    }
}

public struct DistributedRequestControl: Codable, Hashable, Sendable {
    public let requestID: String
    public let stageID: String?

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case stageID = "stage_id"
    }

    public init(requestID: String, stageID: String? = nil) {
        self.requestID = requestID
        self.stageID = stageID
    }

    public func validate(against plan: DistributedStagePlan? = nil) throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidControlFrame("request_id is empty")
        }
        if let stageID {
            guard !stageID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DistributedStageExecutionError.invalidControlFrame("stage_id is empty")
            }
            if let plan, plan.stage(id: stageID) == nil {
                throw DistributedStageExecutionError.invalidControlFrame("unknown stage_id \(stageID)")
            }
        }
    }
}

public struct DistributedWorkerErrorFrame: Codable, Hashable, Sendable {
    public let code: String
    public let detail: String
    public let requestID: String?
    public let stageID: String?

    enum CodingKeys: String, CodingKey {
        case code
        case detail
        case requestID = "request_id"
        case stageID = "stage_id"
    }

    public init(code: String, detail: String, requestID: String? = nil, stageID: String? = nil) {
        self.code = code
        self.detail = detail
        self.requestID = requestID
        self.stageID = stageID
    }

    public func validate(against plan: DistributedStagePlan? = nil) throws {
        guard !code.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidControlFrame("error code is empty")
        }
        guard !detail.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidControlFrame("error detail is empty")
        }
        if let requestID {
            guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DistributedStageExecutionError.invalidControlFrame("request_id is empty")
            }
        }
        if let stageID {
            guard !stageID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DistributedStageExecutionError.invalidControlFrame("stage_id is empty")
            }
            if let plan, plan.stage(id: stageID) == nil {
                throw DistributedStageExecutionError.invalidControlFrame("unknown stage_id \(stageID)")
            }
        }
    }
}

public struct DistributedStageAllocation: Codable, Hashable, Sendable {
    public let requestID: String
    public let kvCapacity: Int
    public let cacheCapacities: [String: Int]?

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case kvCapacity = "kv_capacity"
        case cacheCapacities = "cache_capacities"
    }

    public init(
        requestID: String,
        kvCapacity: Int,
        cacheCapacities: [String: Int]? = nil
    ) {
        self.requestID = requestID
        self.kvCapacity = kvCapacity
        self.cacheCapacities = cacheCapacities
    }

    public func validate() throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidControlFrame("request_id is empty")
        }
        guard kvCapacity > 0 else {
            throw DistributedStageExecutionError.invalidControlFrame("kv_capacity must be positive")
        }
        for (name, capacity) in cacheCapacities ?? [:] {
            guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "cache_capacities group name is empty")
            }
            guard capacity > 0 else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "cache_capacities[\(name)] must be positive")
            }
        }
    }

    public func capacity(forCacheGroup groupName: String) -> Int {
        cacheCapacities?[groupName] ?? kvCapacity
    }
}

public struct DistributedStageForwardFrame: Codable, Hashable, Sendable {
    public let stageID: String
    public let requestID: String
    public let stepIndex: Int
    public let positionRange: DistributedSequenceRange
    public let positionIDs: [Int32]
    public let tokenIDs: [Int32]
    public let blockIDsQ: [Int32]?
    public let blockIDsKV: [Int32]?
    public let softTokenSplice: DistributedSoftTokenSplice?
    public let hiddenState: DistributedHiddenStatePacketMetadata?

    enum CodingKeys: String, CodingKey {
        case stageID = "stage_id"
        case requestID = "request_id"
        case stepIndex = "step_index"
        case positionRange = "position_range"
        case positionIDs = "position_ids"
        case tokenIDs = "token_ids"
        case blockIDsQ = "block_ids_q"
        case blockIDsKV = "block_ids_kv"
        case softTokenSplice = "soft_token_splice"
        case hiddenState = "hidden_state"
    }

    public init(
        stageID: String,
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        positionIDs: [Int32],
        tokenIDs: [Int32] = [],
        blockIDsQ: [Int32]? = nil,
        blockIDsKV: [Int32]? = nil,
        softTokenSplice: DistributedSoftTokenSplice? = nil,
        hiddenState: DistributedHiddenStatePacketMetadata? = nil
    ) {
        self.stageID = stageID
        self.requestID = requestID
        self.stepIndex = stepIndex
        self.positionRange = positionRange
        self.positionIDs = positionIDs
        self.tokenIDs = tokenIDs
        self.blockIDsQ = blockIDsQ
        self.blockIDsKV = blockIDsKV
        self.softTokenSplice = softTokenSplice
        self.hiddenState = hiddenState
    }

    public func validate(against plan: DistributedStagePlan) throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("request_id is empty")
        }
        guard stepIndex >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput("step_index must be non-negative")
        }
        guard positionRange.isValid else {
            throw DistributedStageExecutionError.invalidForwardInput("position_range is invalid")
        }
        let expectedPositionIDs = try plan.positionMode.positionIDs(for: positionRange)
        guard positionIDs == expectedPositionIDs else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_ids do not match \(plan.positionMode.rawValue) position_mode")
        }
        guard let descriptor = plan.stage(id: stageID) else {
            throw DistributedStageExecutionError.invalidForwardInput("unknown stage_id \(stageID)")
        }

        switch descriptor.role {
        case .embeddings:
            guard blockIDsQ == nil && blockIDsKV == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "embeddings stage must not receive block_ids")
            }
            guard tokenIDs.count == positionRange.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "token_ids count must match position_range")
            }
            try Self.validateSoftTokenSplice(
                softTokenSplice,
                positionRange: positionRange,
                boundaryTensor: plan.boundaryTensor)
            guard hiddenState == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "embeddings stage must not receive a hidden state")
            }
        case .transformerLayers:
            guard softTokenSplice == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "transformer_layers stage must not receive soft_token_splice")
            }
            try Self.validateBlockIDs(
                blockIDsQ: blockIDsQ,
                blockIDsKV: blockIDsKV,
                positionRange: positionRange,
                positionIDs: positionIDs)
            guard tokenIDs.isEmpty || tokenIDs.count == positionRange.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "token_ids count must match position_range")
            }
            guard let hiddenState else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(descriptor.role.rawValue) stage requires a hidden state")
            }
            try plan.validate(hiddenStatePacket: hiddenState)
            guard hiddenState.requestID == requestID else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state request_id does not match request")
            }
            guard hiddenState.stepIndex == stepIndex else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state step_index does not match request")
            }
            guard hiddenState.positionRange == positionRange else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state position_range does not match request")
            }
            guard hiddenState.destinationStageID == stageID else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state destination_stage_id does not match stage_id")
            }
        case .finalNormHead:
            guard blockIDsQ == nil && blockIDsKV == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "final_norm_head stage must not receive block_ids")
            }
            guard softTokenSplice == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "final_norm_head stage must not receive soft_token_splice")
            }
            guard tokenIDs.isEmpty || tokenIDs.count == positionRange.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "token_ids count must match position_range")
            }
            guard let hiddenState else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(descriptor.role.rawValue) stage requires a hidden state")
            }
            try plan.validate(hiddenStatePacket: hiddenState)
            guard hiddenState.requestID == requestID else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state request_id does not match request")
            }
            guard hiddenState.stepIndex == stepIndex else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state step_index does not match request")
            }
            guard hiddenState.positionRange == positionRange else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state position_range does not match request")
            }
            guard hiddenState.destinationStageID == stageID else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state destination_stage_id does not match stage_id")
            }
        }
    }

    private static func validateBlockIDs(
        blockIDsQ: [Int32]?,
        blockIDsKV: [Int32]?,
        positionRange: DistributedSequenceRange,
        positionIDs: [Int32]
    ) throws {
        guard blockIDsQ != nil || blockIDsKV != nil else { return }
        guard let blockIDsQ, let blockIDsKV else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "block_ids_q and block_ids_kv must be supplied together")
        }
        guard blockIDsQ.count == positionRange.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "block_ids_q count must match position_range")
        }
        guard blockIDsKV.count == positionIDs.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "block_ids_kv count must match position_ids")
        }
    }

    private static func validateSoftTokenSplice(
        _ splice: DistributedSoftTokenSplice?,
        positionRange: DistributedSequenceRange,
        boundaryTensor: DistributedBoundaryTensorSpec?
    ) throws {
        guard let splice else { return }
        guard positionRange.count > 1 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "decode stage must not receive soft_token_splice")
        }
        guard splice.positionStart >= positionRange.lowerBound
            && splice.positionEnd <= positionRange.upperBound
        else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "soft_token_splice position range must be inside position_range")
        }
        if let hiddenSize = boundaryTensor?.shape.last, hiddenSize > 0 {
            guard splice.hiddenSize == hiddenSize else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "soft_token_splice hidden size must match boundary tensor")
            }
        }
    }
}

public struct DistributedStageForwardResultFrame: Codable, Hashable, Sendable {
    public let stageID: String
    public let requestID: String
    public let stepIndex: Int
    public let hiddenState: DistributedHiddenStatePacketMetadata?
    public let tokenID: Int32?

    enum CodingKeys: String, CodingKey {
        case stageID = "stage_id"
        case requestID = "request_id"
        case stepIndex = "step_index"
        case hiddenState = "hidden_state"
        case tokenID = "token_id"
    }

    public init(
        stageID: String,
        requestID: String,
        stepIndex: Int,
        hiddenState: DistributedHiddenStatePacketMetadata? = nil,
        tokenID: Int32? = nil
    ) {
        self.stageID = stageID
        self.requestID = requestID
        self.stepIndex = stepIndex
        self.hiddenState = hiddenState
        self.tokenID = tokenID
    }

    public func validate(against plan: DistributedStagePlan) throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidStageOutput("request_id is empty")
        }
        guard stepIndex >= 0 else {
            throw DistributedStageExecutionError.invalidStageOutput("step_index must be non-negative")
        }
        guard let descriptor = plan.stage(id: stageID) else {
            throw DistributedStageExecutionError.invalidStageOutput("unknown stage_id \(stageID)")
        }

        if descriptor.role == .finalNormHead {
            guard hiddenState == nil else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "final stage must not return a hidden state")
            }
            guard tokenID != nil else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "final stage must return a token id")
            }
            return
        }

        guard tokenID == nil else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "non-final stage must not return a token id")
        }
        guard let hiddenState else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "non-final stage must return a hidden state")
        }
        try plan.validate(hiddenStatePacket: hiddenState)
        guard hiddenState.requestID == requestID else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "hidden_state request_id does not match request")
        }
        guard hiddenState.stepIndex == stepIndex else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "hidden_state step_index does not match request")
        }
        guard hiddenState.sourceStageID == stageID else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "hidden_state source_stage_id does not match stage_id")
        }
    }
}

public enum DistributedWorkerMessage: Codable, Hashable, Sendable {
    case hello(DistributedWorkerHello)
    case helloAck(DistributedWorkerHelloAck)
    case allocate(DistributedStageAllocation)
    case forward(DistributedStageForwardFrame)
    case forwardResult(DistributedStageForwardResultFrame)
    case reset(DistributedRequestControl)
    case free(DistributedRequestControl)
    case error(DistributedWorkerErrorFrame)

    private enum Kind: String, Codable {
        case hello
        case helloAck = "hello_ack"
        case alloc
        case forward
        case forwardResult = "forward_result"
        case reset
        case free
        case error
    }

    private enum CodingKeys: String, CodingKey {
        case kind
        case hello
        case helloAck = "hello_ack"
        case alloc
        case forward
        case forwardResult = "forward_result"
        case reset
        case free
        case error
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .hello:
            self = .hello(try container.decode(DistributedWorkerHello.self, forKey: .hello))
        case .helloAck:
            self = .helloAck(
                try container.decode(DistributedWorkerHelloAck.self, forKey: .helloAck))
        case .alloc:
            self = .allocate(try container.decode(DistributedStageAllocation.self, forKey: .alloc))
        case .forward:
            self = .forward(try container.decode(DistributedStageForwardFrame.self, forKey: .forward))
        case .forwardResult:
            self = .forwardResult(
                try container.decode(DistributedStageForwardResultFrame.self, forKey: .forwardResult))
        case .reset:
            self = .reset(try container.decode(DistributedRequestControl.self, forKey: .reset))
        case .free:
            self = .free(try container.decode(DistributedRequestControl.self, forKey: .free))
        case .error:
            self = .error(try container.decode(DistributedWorkerErrorFrame.self, forKey: .error))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .hello(let value):
            try container.encode(Kind.hello, forKey: .kind)
            try container.encode(value, forKey: .hello)
        case .helloAck(let value):
            try container.encode(Kind.helloAck, forKey: .kind)
            try container.encode(value, forKey: .helloAck)
        case .allocate(let value):
            try container.encode(Kind.alloc, forKey: .kind)
            try container.encode(value, forKey: .alloc)
        case .forward(let value):
            try container.encode(Kind.forward, forKey: .kind)
            try container.encode(value, forKey: .forward)
        case .forwardResult(let value):
            try container.encode(Kind.forwardResult, forKey: .kind)
            try container.encode(value, forKey: .forwardResult)
        case .reset(let value):
            try container.encode(Kind.reset, forKey: .kind)
            try container.encode(value, forKey: .reset)
        case .free(let value):
            try container.encode(Kind.free, forKey: .kind)
            try container.encode(value, forKey: .free)
        case .error(let value):
            try container.encode(Kind.error, forKey: .kind)
            try container.encode(value, forKey: .error)
        }
    }

    public func validate(against plan: DistributedStagePlan) throws {
        switch self {
        case .hello(let hello):
            try hello.validate(against: plan)
        case .helloAck(let ack):
            try ack.validate(against: plan)
        case .allocate(let allocation):
            try allocation.validate()
        case .forward(let frame):
            try frame.validate(against: plan)
        case .forwardResult(let frame):
            try frame.validate(against: plan)
        case .reset(let control), .free(let control):
            try control.validate(against: plan)
        case .error(let frame):
            try frame.validate(against: plan)
        }
    }

    public var expectedPayloadByteCount: Int {
        switch self {
        case .forward(let frame):
            return frame.hiddenState?.byteCount ?? 0
        case .forwardResult(let frame):
            return frame.hiddenState?.byteCount ?? 0
        case .hello, .helloAck, .allocate, .reset, .free, .error:
            return 0
        }
    }

    public var expectsPayload: Bool {
        expectedPayloadByteCount > 0
    }

    public func validatePayloadByteCount(_ byteCount: Int) throws {
        let expected = expectedPayloadByteCount
        guard byteCount == expected else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "payload byte count \(byteCount) does not match header \(expected)")
        }
    }
}

public enum DistributedWorkerMessageCodec {
    public static func encodeJSONLine(_ message: DistributedWorkerMessage) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(message)
        guard !data.contains(0x0A) && !data.contains(0x0D) else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "worker message JSON must be a single line")
        }
        data.append(0x0A)
        return data
    }

    public static func decodeJSONLine(_ data: Data) throws -> DistributedWorkerMessage {
        var bytes = Array(data)
        if bytes.last == 0x0A {
            bytes.removeLast()
            if bytes.last == 0x0D {
                bytes.removeLast()
            }
        }
        guard !bytes.isEmpty else {
            throw DistributedStageExecutionError.invalidWireFrame("worker message line is empty")
        }
        guard !bytes.contains(0x0A) && !bytes.contains(0x0D) else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "worker message line contains multiple frames")
        }
        return try JSONDecoder().decode(DistributedWorkerMessage.self, from: Data(bytes))
    }

    public static func encodeWireFrame(_ frame: DistributedWorkerWireFrame) throws -> Data {
        try frame.message.validatePayloadByteCount(frame.payload.count)
        var data = try encodeJSONLine(frame.message)
        data.append(contentsOf: frame.payload)
        return data
    }

    public static func decodeWireFrame(_ data: Data) throws -> DistributedWorkerWireFrame {
        guard let headerEnd = data.firstIndex(of: 0x0A) else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "worker wire frame header is missing line ending")
        }
        let header = data.prefix(through: headerEnd)
        let message = try decodeJSONLine(Data(header))
        let payloadStart = data.index(after: headerEnd)
        let payload = Array(data[payloadStart..<data.endIndex])
        try message.validatePayloadByteCount(payload.count)
        return DistributedWorkerWireFrame(message: message, payload: payload)
    }
}

public struct DistributedWorkerWireFrameStreamDecoder: Sendable {
    private var buffer: [UInt8] = []

    public init() {}

    public var bufferedByteCount: Int {
        buffer.count
    }

    public mutating func append(_ data: Data) {
        buffer.append(contentsOf: data)
    }

    public mutating func nextFrame() throws -> DistributedWorkerWireFrame? {
        guard let headerEnd = buffer.firstIndex(of: 0x0A) else {
            return nil
        }
        let headerBytes = buffer[...headerEnd]
        let message = try DistributedWorkerMessageCodec.decodeJSONLine(Data(headerBytes))
        let payloadStart = headerEnd + 1
        let payloadEnd = payloadStart + message.expectedPayloadByteCount
        guard buffer.count >= payloadEnd else {
            return nil
        }
        let payload = Array(buffer[payloadStart..<payloadEnd])
        buffer.removeFirst(payloadEnd)
        let frame = DistributedWorkerWireFrame(message: message, payload: payload)
        try frame.message.validatePayloadByteCount(frame.payload.count)
        return frame
    }

    public mutating func drainFrames() throws -> [DistributedWorkerWireFrame] {
        var frames: [DistributedWorkerWireFrame] = []
        while let frame = try nextFrame() {
            frames.append(frame)
        }
        return frames
    }

    public func finish() throws {
        guard buffer.isEmpty else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "worker wire frame stream ended with \(buffer.count) buffered bytes")
        }
    }
}

public struct DistributedWorkerWireFrame: Hashable, Sendable {
    public let message: DistributedWorkerMessage
    public let payload: [UInt8]

    public init(message: DistributedWorkerMessage, payload: [UInt8] = []) {
        self.message = message
        self.payload = payload
    }

    public func validate(against plan: DistributedStagePlan) throws {
        try message.validate(against: plan)
        try message.validatePayloadByteCount(payload.count)
    }
}

public struct DistributedWorkerHandshakeCoordinator: Sendable {
    public let plan: DistributedStagePlan
    private let planIntegrityHash: String
    private var claimedStageIDs: Set<String> = []

    public init(plan: DistributedStagePlan) throws {
        try plan.validate()
        self.plan = plan
        self.planIntegrityHash = try plan.integrityHash()
    }

    public var claimedStages: Set<String> {
        claimedStageIDs
    }

    public var missingStageIDs: [String] {
        plan.stages.map(\.id).filter { !claimedStageIDs.contains($0) }
    }

    public var isReady: Bool {
        missingStageIDs.isEmpty
    }

    public mutating func processHello(
        _ wireFrame: DistributedWorkerWireFrame
    ) throws -> DistributedWorkerWireFrame {
        try wireFrame.message.validatePayloadByteCount(wireFrame.payload.count)
        guard case .hello(let hello) = wireFrame.message else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "handshake requires hello frame")
        }

        do {
            try hello.validate(
                against: plan,
                expectedPlanIntegrityHash: planIntegrityHash)
        } catch {
            return makeHelloAck(
                stageID: hello.stage.id,
                accepted: false,
                reason: rejectionReason(error))
        }

        guard !claimedStageIDs.contains(hello.stage.id) else {
            return makeHelloAck(
                stageID: hello.stage.id,
                accepted: false,
                reason: "stage already claimed")
        }

        claimedStageIDs.insert(hello.stage.id)
        return makeHelloAck(
            stageID: hello.stage.id,
            accepted: true,
            reason: nil)
    }

    public func requireReady() throws {
        let missing = missingStageIDs
        guard missing.isEmpty else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "missing worker stages: \(missing.joined(separator: ", "))")
        }
    }

    private func makeHelloAck(
        stageID: String,
        accepted: Bool,
        reason: String?
    ) -> DistributedWorkerWireFrame {
        DistributedWorkerWireFrame(message: .helloAck(
            DistributedWorkerHelloAck(
                accepted: accepted,
                stageID: stageID,
                reason: reason,
                planIntegrityHash: accepted ? planIntegrityHash : nil)))
    }

    private func rejectionReason(_ error: Error) -> String {
        guard let executionError = error as? DistributedStageExecutionError else {
            return String(describing: error)
        }
        switch executionError {
        case .invalidWorkerHello(let message),
            .invalidControlFrame(let message),
            .invalidWireFrame(let message):
            return message
        default:
            return executionError.description
        }
    }
}

public struct DistributedWorkerRequestTracker: Sendable {
    public struct RequestState: Hashable, Sendable {
        public let kvCapacity: Int
        public let processedTokenCount: Int
        public let nextStepIndex: Int

        public init(
            kvCapacity: Int,
            processedTokenCount: Int = 0,
            nextStepIndex: Int = 0
        ) {
            self.kvCapacity = kvCapacity
            self.processedTokenCount = processedTokenCount
            self.nextStepIndex = nextStepIndex
        }
    }

    private var requests: [String: RequestState] = [:]

    public init() {}

    public var activeRequestIDs: Set<String> {
        Set(requests.keys)
    }

    public func state(for requestID: String) -> RequestState? {
        requests[requestID]
    }

    public func validateAllocate(_ allocation: DistributedStageAllocation) throws {
        try allocation.validate()
        guard requests[allocation.requestID] == nil else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "request_id \(allocation.requestID) is already allocated")
        }
    }

    public mutating func commitAllocate(_ allocation: DistributedStageAllocation) {
        requests[allocation.requestID] = RequestState(kvCapacity: allocation.kvCapacity)
    }

    public func validateForward(_ frame: DistributedStageForwardFrame) throws {
        guard let state = requests[frame.requestID] else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "request_id \(frame.requestID) is not allocated")
        }
        guard frame.stepIndex == state.nextStepIndex else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "step_index \(frame.stepIndex) does not match expected \(state.nextStepIndex)")
        }
        guard frame.positionRange.lowerBound == state.processedTokenCount else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_range lower_bound \(frame.positionRange.lowerBound) does not match processed_token_count \(state.processedTokenCount)")
        }
        guard frame.positionRange.upperBound <= state.kvCapacity else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_range upper_bound \(frame.positionRange.upperBound) exceeds kv_capacity \(state.kvCapacity)")
        }
    }

    public mutating func commitForward(_ frame: DistributedStageForwardFrame) {
        guard let state = requests[frame.requestID] else { return }
        requests[frame.requestID] = RequestState(
            kvCapacity: state.kvCapacity,
            processedTokenCount: state.processedTokenCount + frame.positionRange.count,
            nextStepIndex: state.nextStepIndex + 1)
    }

    public func validateReset(_ control: DistributedRequestControl) throws {
        try control.validate()
        guard requests[control.requestID] != nil else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "request_id \(control.requestID) is not allocated")
        }
    }

    public mutating func commitReset(_ control: DistributedRequestControl) {
        guard let state = requests[control.requestID] else { return }
        requests[control.requestID] = RequestState(kvCapacity: state.kvCapacity)
    }

    public func validateFree(_ control: DistributedRequestControl) throws {
        try control.validate()
        guard requests[control.requestID] != nil else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "request_id \(control.requestID) is not allocated")
        }
    }

    public mutating func commitFree(_ control: DistributedRequestControl) {
        requests.removeValue(forKey: control.requestID)
    }
}

public final class DistributedWorkerFrameExecutor {
    public let plan: DistributedStagePlan
    public let handle: DistributedStageHandle
    private let planIntegrityHash: String
    private var requestTracker = DistributedWorkerRequestTracker()

    public init(plan: DistributedStagePlan, handle: DistributedStageHandle) throws {
        try plan.validate()
        guard let expectedDescriptor = plan.stage(id: handle.descriptor.id) else {
            throw DistributedStageExecutionError.missingStageHandle(handle.descriptor.id)
        }
        guard expectedDescriptor == handle.descriptor else {
            throw DistributedStageExecutionError.stageDescriptorMismatch(
                expected: expectedDescriptor.id, actual: handle.descriptor.id)
        }
        self.plan = plan
        self.handle = handle
        self.planIntegrityHash = try plan.integrityHash()
    }

    public func makeHello(
        cacheContract: String? = nil,
        freeMemoryBytes: UInt64? = nil,
        computeUnit: String? = nil,
        labels: [String: String] = [:]
    ) throws -> DistributedWorkerWireFrame {
        let hello = DistributedWorkerHello(
            stage: handle.descriptor,
            hiddenSize: plan.boundaryTensor?.shape.last,
            boundaryScalarType: plan.boundaryTensor?.scalarType,
            cacheContract: cacheContract,
            acceptsTokenIDs: handle.acceptsTokenIDs,
            planIntegrityHash: planIntegrityHash,
            freeMemoryBytes: freeMemoryBytes,
            computeUnit: computeUnit,
            labels: labels)
        let message = DistributedWorkerMessage.hello(hello)
        try message.validate(against: plan)
        return DistributedWorkerWireFrame(message: message)
    }

    public func process(_ wireFrame: DistributedWorkerWireFrame) async throws -> DistributedWorkerWireFrame? {
        try wireFrame.validate(against: plan)
        switch wireFrame.message {
        case .allocate(let allocation):
            try requestTracker.validateAllocate(allocation)
            try await handle.allocate(allocation)
            requestTracker.commitAllocate(allocation)
            return nil
        case .forward(let frame):
            try ensureTarget(stageID: frame.stageID)
            try requestTracker.validateForward(frame)
            let hiddenState = try frame.hiddenState.map { metadata in
                try DistributedHiddenStatePacket(metadata: metadata, payload: wireFrame.payload)
            }
            let output = try await handle.forward(DistributedStageForwardInput(
                requestID: frame.requestID,
                stepIndex: frame.stepIndex,
                positionRange: frame.positionRange,
                positionIDs: frame.positionIDs,
                tokenIDs: frame.tokenIDs,
                blockIDsQ: frame.blockIDsQ,
                blockIDsKV: frame.blockIDsKV,
                softTokenSplice: frame.softTokenSplice,
                hiddenState: hiddenState))
            guard output.stageID == handle.descriptor.id else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "output stage_id \(output.stageID) does not match worker stage \(handle.descriptor.id)")
            }
            guard output.stepIndex == frame.stepIndex else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "output step_index does not match request")
            }
            guard output.eagleTargetArtifacts == nil,
                output.eagleTargetKVChunk == nil,
                output.eagleTargetFinalHidden == nil
            else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "worker wire protocol does not support EAGLE target artifacts")
            }
            let result = DistributedStageForwardResultFrame(
                stageID: output.stageID,
                requestID: frame.requestID,
                stepIndex: output.stepIndex,
                hiddenState: output.hiddenState?.metadata,
                tokenID: output.tokenID)
            let response = DistributedWorkerWireFrame(
                message: .forwardResult(result),
                payload: output.hiddenState?.payload ?? [])
            try response.validate(against: plan)
            requestTracker.commitForward(frame)
            return response
        case .reset(let control):
            try ensureTarget(stageID: control.stageID)
            try requestTracker.validateReset(control)
            try await handle.reset(requestID: control.requestID)
            requestTracker.commitReset(control)
            return nil
        case .free(let control):
            try ensureTarget(stageID: control.stageID)
            try requestTracker.validateFree(control)
            await handle.free(requestID: control.requestID)
            requestTracker.commitFree(control)
            return nil
        case .hello, .helloAck, .forwardResult, .error:
            throw DistributedStageExecutionError.invalidControlFrame(
                "worker cannot process \(wireFrame.message.kindName) frame")
        }
    }

    public func processForTransport(
        _ wireFrame: DistributedWorkerWireFrame
    ) async throws -> DistributedWorkerWireFrame? {
        do {
            return try await process(wireFrame)
        } catch {
            let response = DistributedWorkerWireFrame(message: .error(
                DistributedWorkerErrorFrame(
                    code: errorCode(error),
                    detail: errorDetail(error),
                    requestID: wireFrame.message.nonEmptyRequestID,
                    stageID: handle.descriptor.id)))
            try response.validate(against: plan)
            return response
        }
    }

    private func ensureTarget(stageID: String?) throws {
        guard let stageID else { return }
        guard stageID == handle.descriptor.id else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "frame stage_id \(stageID) does not match worker stage \(handle.descriptor.id)")
        }
    }

    private func errorCode(_ error: Error) -> String {
        if error is DistributedRuntimeValidationError {
            return "runtime_validation"
        }
        guard let executionError = error as? DistributedStageExecutionError else {
            return "worker_error"
        }
        switch executionError {
        case .invalidWorkerHello:
            return "invalid_worker_hello"
        case .invalidControlFrame:
            return "invalid_control_frame"
        case .invalidWireFrame:
            return "invalid_wire_frame"
        case .invalidForwardInput:
            return "invalid_forward_input"
        case .invalidStageOutput:
            return "invalid_stage_output"
        case .invalidStageIOContract:
            return "invalid_stage_io_contract"
        default:
            return "worker_error"
        }
    }

    private func errorDetail(_ error: Error) -> String {
        if let validationError = error as? DistributedRuntimeValidationError {
            return validationError.description
        }
        guard let executionError = error as? DistributedStageExecutionError else {
            return String(describing: error)
        }
        switch executionError {
        case .invalidWorkerHello(let message),
            .invalidControlFrame(let message),
            .invalidWireFrame(let message),
            .invalidForwardInput(let message),
            .invalidStageOutput(let message):
            return message
        case .invalidStageIOContract(_, let reason):
            return reason
        default:
            return executionError.description
        }
    }
}

public final class DistributedLoopbackWorkerTransport {
    private let executor: DistributedWorkerFrameExecutor
    private let requestChunkSize: Int?
    private let responseChunkSize: Int?

    public init(
        executor: DistributedWorkerFrameExecutor,
        requestChunkSize: Int? = nil,
        responseChunkSize: Int? = nil
    ) {
        self.executor = executor
        self.requestChunkSize = requestChunkSize
        self.responseChunkSize = responseChunkSize
    }

    public func handshake(
        with coordinator: inout DistributedWorkerHandshakeCoordinator,
        cacheContract: String? = nil,
        freeMemoryBytes: UInt64? = nil,
        computeUnit: String? = nil,
        labels: [String: String] = [:]
    ) throws -> DistributedWorkerWireFrame {
        let hello = try executor.makeHello(
            cacheContract: cacheContract,
            freeMemoryBytes: freeMemoryBytes,
            computeUnit: computeUnit,
            labels: labels)
        let helloFrames = try decodeStream(
            DistributedWorkerMessageCodec.encodeWireFrame(hello),
            chunkSize: requestChunkSize)
        guard helloFrames.count == 1, let helloFrame = helloFrames.first else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "loopback hello must contain exactly one frame")
        }

        let response = try coordinator.processHello(helloFrame)
        let responseFrames = try decodeStream(
            DistributedWorkerMessageCodec.encodeWireFrame(response),
            chunkSize: responseChunkSize)
        guard responseFrames.count == 1, let responseFrame = responseFrames.first else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "loopback hello_ack must contain exactly one frame")
        }
        try responseFrame.validate(against: coordinator.plan)
        return responseFrame
    }

    public func roundTrip(
        _ request: DistributedWorkerWireFrame
    ) async throws -> DistributedWorkerWireFrame? {
        let requestFrames = try decodeStream(
            DistributedWorkerMessageCodec.encodeWireFrame(request),
            chunkSize: requestChunkSize)
        guard requestFrames.count == 1, let requestFrame = requestFrames.first else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "loopback request must contain exactly one frame")
        }

        guard let response = try await executor.processForTransport(requestFrame) else {
            return nil
        }
        let responseFrames = try decodeStream(
            DistributedWorkerMessageCodec.encodeWireFrame(response),
            chunkSize: responseChunkSize)
        guard responseFrames.count == 1, let responseFrame = responseFrames.first else {
            throw DistributedStageExecutionError.invalidWireFrame(
                "loopback response must contain exactly one frame")
        }
        return responseFrame
    }

    private func decodeStream(
        _ data: Data,
        chunkSize: Int?
    ) throws -> [DistributedWorkerWireFrame] {
        var decoder = DistributedWorkerWireFrameStreamDecoder()
        var frames: [DistributedWorkerWireFrame] = []
        let size = max(1, chunkSize ?? data.count)
        var index = data.startIndex

        while index < data.endIndex {
            let end = data.index(index, offsetBy: size, limitedBy: data.endIndex) ?? data.endIndex
            decoder.append(data[index..<end])
            frames.append(contentsOf: try decoder.drainFrames())
            index = end
        }

        try decoder.finish()
        return frames
    }
}

extension DistributedWorkerMessage {
    fileprivate var kindName: String {
        switch self {
        case .hello:
            return "hello"
        case .helloAck:
            return "hello_ack"
        case .allocate:
            return "alloc"
        case .forward:
            return "forward"
        case .forwardResult:
            return "forward_result"
        case .reset:
            return "reset"
        case .free:
            return "free"
        case .error:
            return "error"
        }
    }

    fileprivate var nonEmptyRequestID: String? {
        let requestID: String?
        switch self {
        case .allocate(let allocation):
            requestID = allocation.requestID
        case .forward(let frame):
            requestID = frame.requestID
        case .forwardResult(let frame):
            requestID = frame.requestID
        case .reset(let control), .free(let control):
            requestID = control.requestID
        case .error(let frame):
            requestID = frame.requestID
        case .hello, .helloAck:
            requestID = nil
        }
        guard let requestID,
            !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }
        return requestID
    }
}

public struct DistributedStageForwardInput: Hashable, Sendable {
    public let requestID: String
    public let stepIndex: Int
    public let positionRange: DistributedSequenceRange
    public let positionIDs: [Int32]
    public let tokenIDs: [Int32]
    public let blockIDsQ: [Int32]?
    public let blockIDsKV: [Int32]?
    public let softTokenSplice: DistributedSoftTokenSplice?
    public let hiddenState: DistributedHiddenStatePacket?

    public init(
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        positionIDs: [Int32],
        tokenIDs: [Int32] = [],
        blockIDsQ: [Int32]? = nil,
        blockIDsKV: [Int32]? = nil,
        softTokenSplice: DistributedSoftTokenSplice? = nil,
        hiddenState: DistributedHiddenStatePacket? = nil
    ) {
        self.requestID = requestID
        self.stepIndex = stepIndex
        self.positionRange = positionRange
        self.positionIDs = positionIDs
        self.tokenIDs = tokenIDs
        self.blockIDsQ = blockIDsQ
        self.blockIDsKV = blockIDsKV
        self.softTokenSplice = softTokenSplice
        self.hiddenState = hiddenState
    }
}

public typealias DistributedWorkerFrameRoundTrip =
    (DistributedWorkerWireFrame) async throws -> DistributedWorkerWireFrame?

public final class DistributedRemoteStageHandle: DistributedStageHandle {
    public let descriptor: DistributedStageDescriptor
    public let acceptsTokenIDs: Bool
    public let plan: DistributedStagePlan
    private let roundTrip: DistributedWorkerFrameRoundTrip

    public init(
        plan: DistributedStagePlan,
        descriptor: DistributedStageDescriptor,
        acceptsTokenIDs: Bool? = nil,
        roundTrip: @escaping DistributedWorkerFrameRoundTrip
    ) throws {
        try plan.validate()
        guard let expectedDescriptor = plan.stage(id: descriptor.id) else {
            throw DistributedStageExecutionError.missingStageHandle(descriptor.id)
        }
        guard expectedDescriptor == descriptor else {
            throw DistributedStageExecutionError.stageDescriptorMismatch(
                expected: expectedDescriptor.id, actual: descriptor.id)
        }
        self.plan = plan
        self.descriptor = descriptor
        self.acceptsTokenIDs = acceptsTokenIDs ?? (descriptor.role == .embeddings)
        self.roundTrip = roundTrip
    }

    public func allocate(_ allocation: DistributedStageAllocation) async throws {
        let response = try await roundTrip(DistributedWorkerWireFrame(
            message: .allocate(allocation)))
        try expectNoResponse(response, for: "alloc", requestID: allocation.requestID)
    }

    public func forward(
        _ input: DistributedStageForwardInput
    ) async throws -> DistributedStageForwardOutput {
        let request = DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: descriptor.id,
                requestID: input.requestID,
                stepIndex: input.stepIndex,
                positionRange: input.positionRange,
                positionIDs: input.positionIDs,
                tokenIDs: input.tokenIDs,
                blockIDsQ: input.blockIDsQ,
                blockIDsKV: input.blockIDsKV,
                softTokenSplice: input.softTokenSplice,
                hiddenState: input.hiddenState?.metadata)),
            payload: input.hiddenState?.payload ?? [])
        try request.validate(against: plan)

        guard let response = try await roundTrip(request) else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "forward response is missing")
        }
        try response.validate(against: plan)
        if case .error(let error) = response.message {
            try throwWorkerError(error, operation: "forward", requestID: input.requestID)
        }
        guard case .forwardResult(let result) = response.message else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "expected forward_result response")
        }
        guard result.stageID == descriptor.id else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "response stage_id \(result.stageID) does not match remote stage \(descriptor.id)")
        }
        guard result.requestID == input.requestID else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "response request_id does not match request")
        }
        guard result.stepIndex == input.stepIndex else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "response step_index does not match request")
        }
        let hiddenState = try result.hiddenState.map { metadata in
            try DistributedHiddenStatePacket(metadata: metadata, payload: response.payload)
        }
        return DistributedStageForwardOutput(
            stageID: result.stageID,
            stepIndex: result.stepIndex,
            hiddenState: hiddenState,
            tokenID: result.tokenID)
    }

    public func reset(requestID: String) async throws {
        let response = try await roundTrip(DistributedWorkerWireFrame(
            message: .reset(DistributedRequestControl(
                requestID: requestID, stageID: descriptor.id))))
        try expectNoResponse(response, for: "reset", requestID: requestID)
    }

    public func free(requestID: String) async {
        _ = try? await roundTrip(DistributedWorkerWireFrame(
            message: .free(DistributedRequestControl(
                requestID: requestID, stageID: descriptor.id))))
    }

    private func expectNoResponse(
        _ response: DistributedWorkerWireFrame?,
        for operation: String,
        requestID: String
    ) throws {
        guard let response else { return }
        try response.validate(against: plan)
        if case .error(let error) = response.message {
            try throwWorkerError(error, operation: operation, requestID: requestID)
        }
        throw DistributedStageExecutionError.invalidControlFrame(
            "\(operation) must not return a response")
    }

    private func throwWorkerError(
        _ error: DistributedWorkerErrorFrame,
        operation: String,
        requestID: String
    ) throws -> Never {
        if let errorRequestID = error.requestID, errorRequestID != requestID {
            throw DistributedStageExecutionError.invalidControlFrame(
                "\(operation) worker error request_id \(errorRequestID) does not match request \(requestID)")
        }
        if let errorStageID = error.stageID, errorStageID != descriptor.id {
            throw DistributedStageExecutionError.invalidControlFrame(
                "\(operation) worker error stage_id \(errorStageID) does not match remote stage \(descriptor.id)")
        }
        throw DistributedStageExecutionError.invalidControlFrame(
            "\(operation) worker error \(error.code): \(error.detail)")
    }
}

public struct DistributedLogitScore: Hashable, Sendable {
    public let tokenID: Int32
    public let logit: Float

    public init(tokenID: Int32, logit: Float) {
        self.tokenID = tokenID
        self.logit = logit
    }
}

public struct DistributedStageForwardOutput: Hashable, Sendable {
    public let stageID: String
    public let stepIndex: Int
    public let hiddenState: DistributedHiddenStatePacket?
    public let tokenID: Int32?
    public let topLogits: [DistributedLogitScore]
    public let eagleTargetKVChunk: DistributedEagleTargetKVChunk?
    public let eagleTargetFinalHidden: DistributedEagleTargetTensor?
    public let eagleTargetArtifacts: DistributedEagleTargetArtifacts?

    public init(
        stageID: String,
        stepIndex: Int,
        hiddenState: DistributedHiddenStatePacket? = nil,
        tokenID: Int32? = nil,
        topLogits: [DistributedLogitScore] = [],
        eagleTargetKVChunk: DistributedEagleTargetKVChunk? = nil,
        eagleTargetFinalHidden: DistributedEagleTargetTensor? = nil,
        eagleTargetArtifacts: DistributedEagleTargetArtifacts? = nil
    ) {
        self.stageID = stageID
        self.stepIndex = stepIndex
        self.hiddenState = hiddenState
        self.tokenID = tokenID
        self.topLogits = topLogits
        self.eagleTargetKVChunk = eagleTargetKVChunk
        self.eagleTargetFinalHidden = eagleTargetFinalHidden
        self.eagleTargetArtifacts = eagleTargetArtifacts
    }
}

public protocol DistributedStageHandle: AnyObject {
    var descriptor: DistributedStageDescriptor { get }
    var acceptsTokenIDs: Bool { get }
    var supportsEagleTargetArtifacts: Bool { get }
    var supportsEagleTargetKVChunk: Bool { get }
    var supportsEagleTargetFinalHidden: Bool { get }

    func allocate(_ allocation: DistributedStageAllocation) async throws
    func forward(_ input: DistributedStageForwardInput) async throws -> DistributedStageForwardOutput
    func reset(requestID: String) async throws
    func free(requestID: String) async
}

public extension DistributedStageHandle {
    var acceptsTokenIDs: Bool {
        descriptor.role == .embeddings
    }

    var supportsEagleTargetArtifacts: Bool { false }
    var supportsEagleTargetKVChunk: Bool { false }
    var supportsEagleTargetFinalHidden: Bool { false }
}

public struct DistributedStageHandleFactoryContext: Hashable, Sendable {
    public let stage: DistributedStageManifestStage
    public let manifest: DistributedStageManifest
    public let descriptor: DistributedStageDescriptor

    public init(
        stage: DistributedStageManifestStage,
        manifest: DistributedStageManifest,
        descriptor: DistributedStageDescriptor
    ) {
        self.stage = stage
        self.manifest = manifest
        self.descriptor = descriptor
    }

    public var boundaryTensor: DistributedBoundaryTensorSpec? {
        manifest.boundaryTensor
    }

    public var resolvedAssetURL: URL? {
        stage.resolvedAssetPath.map { URL(fileURLWithPath: $0) }
    }

    public var resolvedDecodeAssetURL: URL? {
        stage.resolvedDecodeAssetPath.map { URL(fileURLWithPath: $0) }
    }

    public var nextStage: DistributedStageDescriptor? {
        manifest.runtimePlan.nextStage(after: descriptor.id)
    }

    public var mainFunctionName: String {
        descriptor.functionMap?.mainFunctionName ?? "main"
    }

    public var decodeFunctionName: String? {
        descriptor.functionMap?.decodeFunctionName
    }

    public var vocabSize: Int? {
        descriptor.vocabSize
    }

    public var eagleTarget: DistributedEagleTargetContract? {
        manifest.eagleTarget
    }

    public func requireResolvedAssetURL() throws -> URL {
        guard let resolvedAssetURL else {
            throw DistributedStageExecutionError.missingStageAssetPath(stage.id)
        }
        return resolvedAssetURL
    }

    public func requireExistingAssetURL(fileManager: FileManager = .default) throws -> URL {
        let url = try requireResolvedAssetURL()
        guard fileManager.fileExists(atPath: url.path) else {
            throw DistributedStageExecutionError.missingStageAsset(
                stageID: stage.id, path: url.path)
        }
        return url
    }

    public func requireExistingDecodeAssetURL(fileManager: FileManager = .default) throws -> URL? {
        guard let url = resolvedDecodeAssetURL else { return nil }
        guard fileManager.fileExists(atPath: url.path) else {
            throw DistributedStageExecutionError.missingStageAsset(
                stageID: stage.id, path: url.path)
        }
        return url
    }

    public func validateStageIOContract(
        _ contract: DistributedStageIOContract,
        vocabSize: Int? = nil
    ) throws {
        try contract.validate(
            for: descriptor,
            boundaryTensor: boundaryTensor,
            vocabSize: vocabSize,
            eagleTarget: eagleTarget)
    }
}

public protocol DistributedStageHandleFactory {
    func makeStageHandle(
        for context: DistributedStageHandleFactoryContext
    ) async throws -> DistributedStageHandle
}

public enum DistributedStageExecutionError: Error, Equatable, Sendable, CustomStringConvertible {
    case stageCountMismatch(expected: Int, actual: Int)
    case stageDescriptorMismatch(expected: String, actual: String)
    case duplicateStageHandle(String)
    case missingStageHandle(String)
    case missingStageAssetPath(String)
    case missingStageAsset(stageID: String, path: String)
    case invalidWorkerHello(String)
    case invalidControlFrame(String)
    case invalidWireFrame(String)
    case invalidForwardInput(String)
    case invalidStageOutput(String)
    case invalidStageIOContract(stageID: String, reason: String)

    public var description: String {
        switch self {
        case .stageCountMismatch(let expected, let actual):
            return "Stage handle count mismatch: expected \(expected), got \(actual)"
        case .stageDescriptorMismatch(let expected, let actual):
            return "Stage handle descriptor mismatch: expected \(expected), got \(actual)"
        case .duplicateStageHandle(let id):
            return "Duplicate stage handle: \(id)"
        case .missingStageHandle(let id):
            return "Missing stage handle: \(id)"
        case .missingStageAssetPath(let id):
            return "Missing stage asset path: \(id)"
        case .missingStageAsset(let stageID, let path):
            return "Missing stage asset for \(stageID): \(path)"
        case .invalidWorkerHello(let message):
            return "Invalid distributed worker hello: \(message)"
        case .invalidControlFrame(let message):
            return "Invalid distributed control frame: \(message)"
        case .invalidWireFrame(let message):
            return "Invalid distributed worker wire frame: \(message)"
        case .invalidForwardInput(let message):
            return "Invalid distributed forward input: \(message)"
        case .invalidStageOutput(let message):
            return "Invalid distributed stage output: \(message)"
        case .invalidStageIOContract(let stageID, let reason):
            return "Invalid distributed stage IO contract for \(stageID): \(reason)"
        }
    }
}

/// In-process coordinator for one staged forward. This is the same-machine milestone harness;
/// concrete handles can be fake test stages now and Core AI stage handles later.
public final class DistributedSameMachinePipeline {
    public let plan: DistributedStagePlan
    private let stages: [DistributedStageHandle]
    private let streamedPrefillAdmission: DistributedStagedMemoryAdmission?
    private let eagleTarget: DistributedEagleTargetContract?
    private var activePrefillStageID: String?
    private var requestTracker = DistributedWorkerRequestTracker()
    private var eagleTargetAccumulators: [String: DistributedEagleTargetKVAccumulator] = [:]

    public init(
        plan: DistributedStagePlan,
        stages: [DistributedStageHandle],
        streamedPrefillAdmission: DistributedStagedMemoryAdmission? = nil,
        eagleTarget: DistributedEagleTargetContract? = nil
    ) throws {
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

        if streamedPrefillAdmission != nil {
            for stage in stages {
                guard let resident = stage as? DistributedDecodeResidentStageHandle else {
                    throw DistributedStageExecutionError.invalidControlFrame(
                        "streamed prefill requires a decode-resident handle for stage \(stage.descriptor.id)")
                }
                guard !resident.isPrefillResident else {
                    throw DistributedStageExecutionError.invalidControlFrame(
                        "decode-resident stage \(stage.descriptor.id) loaded prefill before admission")
                }
            }
        }

        if let eagleTarget {
            guard eagleTarget.slidingWindow > 0,
                plan.stages.last(where: { $0.role == .transformerLayers })?.id
                    == eagleTarget.stageID
            else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "eagle_target must name the final transformer_layers stage with a positive sliding_window")
            }
            guard let targetHandle = stages.first(where: {
                $0.descriptor.id == eagleTarget.stageID
            }), targetHandle.supportsEagleTargetKVChunk else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "eagle_target stage \(eagleTarget.stageID) requires a local auxiliary-output handle")
            }
            guard let finalHiddenHandle = stages.first(where: {
                $0.descriptor.id == eagleTarget.finalHiddenStageID
            }), finalHiddenHandle.supportsEagleTargetFinalHidden else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "eagle_target final hidden stage \(eagleTarget.finalHiddenStageID) requires a local auxiliary-output handle")
            }
        }

        self.plan = plan
        self.stages = stages
        self.streamedPrefillAdmission = streamedPrefillAdmission
        self.eagleTarget = eagleTarget
    }

    public convenience init(
        manifest: DistributedStageManifest,
        handlesByStageID: [String: DistributedStageHandle]
    ) throws {
        let orderedHandles = try manifest.runtimePlan.stages.map { descriptor in
            guard let handle = handlesByStageID[descriptor.id] else {
                throw DistributedStageExecutionError.missingStageHandle(descriptor.id)
            }
            return handle
        }
        try self.init(
            plan: manifest.runtimePlan,
            stages: orderedHandles,
            eagleTarget: manifest.eagleTarget)
    }

    public static func make(
        manifest: DistributedStageManifest,
        handleFactory: DistributedStageHandleFactory,
        streamedPrefillAdmission: DistributedStagedMemoryAdmission? = nil
    ) async throws -> DistributedSameMachinePipeline {
        let effectiveAdmission: DistributedStagedMemoryAdmission?
        if manifest.requiresStreamedPrefillResidency {
            guard let streamedPrefillAdmission else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "streamed prefill manifest requires memory admission before decode assets load")
            }
            effectiveAdmission = streamedPrefillAdmission
        } else {
            effectiveAdmission = nil
        }

        var handles: [DistributedStageHandle] = []
        handles.reserveCapacity(manifest.stages.count)
        for stage in manifest.stages {
            guard let descriptor = manifest.runtimePlan.stage(id: stage.id) else {
                throw DistributedStageExecutionError.missingStageHandle(stage.id)
            }
            let context = DistributedStageHandleFactoryContext(
                stage: stage, manifest: manifest, descriptor: descriptor)
            try effectiveAdmission?.checkBeforeAssetLoad()
            handles.append(try await handleFactory.makeStageHandle(for: context))
        }
        return try DistributedSameMachinePipeline(
            plan: manifest.runtimePlan,
            stages: handles,
            streamedPrefillAdmission: effectiveAdmission,
            eagleTarget: manifest.eagleTarget)
    }

    public func allocate(
        requestID: String,
        kvCapacity: Int,
        cacheCapacities: [String: Int]? = nil
    ) async throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("request_id is empty")
        }
        guard kvCapacity > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput("kv_capacity must be positive")
        }
        let allocation = DistributedStageAllocation(
            requestID: requestID,
            kvCapacity: kvCapacity,
            cacheCapacities: cacheCapacities)
        try requestTracker.validateAllocate(allocation)
        for stage in stages {
            try await stage.allocate(allocation)
        }
        if let eagleTarget {
            eagleTargetAccumulators[requestID] = try DistributedEagleTargetKVAccumulator(
                kvCapacity: kvCapacity,
                slidingWindow: eagleTarget.slidingWindow)
        }
        requestTracker.commitAllocate(allocation)
    }

    public func forward(
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        tokenIDs: [Int32],
        transformerTokenIDs: [Int32]? = nil,
        blockIDsQ: [Int32]? = nil,
        blockIDsKV: [Int32]? = nil,
        softTokenSplice: DistributedSoftTokenSplice? = nil,
        emitToken: Bool = true
    ) async throws -> DistributedStageForwardOutput {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("request_id is empty")
        }
        guard !tokenIDs.isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("token_ids must be non-empty")
        }
        guard positionRange.isValid else {
            throw DistributedStageExecutionError.invalidForwardInput("position_range is invalid")
        }
        guard tokenIDs.count == positionRange.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "token_ids count must match position_range")
        }
        if let transformerTokenIDs {
            guard transformerTokenIDs.count == positionRange.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "transformer_token_ids count must match position_range")
            }
        }
        guard stepIndex >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "step_index must be non-negative")
        }
        let positionIDs = try plan.positionMode.positionIDs(for: positionRange)
        if blockIDsQ != nil || blockIDsKV != nil {
            guard let blockIDsQ, let blockIDsKV else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "block_ids_q and block_ids_kv must be supplied together")
            }
            guard blockIDsQ.count == positionRange.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "block_ids_q count must match position_range")
            }
            guard blockIDsKV.count == positionIDs.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "block_ids_kv count must match position_ids")
            }
        }
        if let softTokenSplice {
            guard positionRange.count > 1 else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "decode stage must not receive soft_token_splice")
            }
            guard softTokenSplice.positionStart >= positionRange.lowerBound
                && softTokenSplice.positionEnd <= positionRange.upperBound
            else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "soft_token_splice position range must be inside position_range")
            }
            if let hiddenSize = plan.boundaryTensor?.shape.last, hiddenSize > 0 {
                guard softTokenSplice.hiddenSize == hiddenSize else {
                    throw DistributedStageExecutionError.invalidForwardInput(
                        "soft_token_splice hidden size must match boundary tensor")
                }
            }
        }

        var hiddenState: DistributedHiddenStatePacket?
        var tokenID: Int32?
        var topLogits: [DistributedLogitScore] = []
        var eagleTargetArtifacts = eagleTargetAccumulators[requestID]?.snapshot
        var eagleTargetKVChunk: DistributedEagleTargetKVChunk?
        var eagleTargetFinalHidden: DistributedEagleTargetTensor?
        let firstFrame = DistributedStageForwardFrame(
            stageID: stages.first!.descriptor.id,
            requestID: requestID,
            stepIndex: stepIndex,
            positionRange: positionRange,
            positionIDs: positionIDs,
            tokenIDs: tokenIDs,
            softTokenSplice: softTokenSplice)
        try firstFrame.validate(against: plan)
        try requestTracker.validateForward(firstFrame)

        for (index, stage) in stages.enumerated() {
            if !emitToken && stage.descriptor.role == .finalNormHead {
                break
            }
            let stageTokenIDs: [Int32]
            if stage.acceptsTokenIDs {
                stageTokenIDs = stage.descriptor.role == .transformerLayers
                    ? (transformerTokenIDs ?? tokenIDs)
                    : tokenIDs
            } else {
                stageTokenIDs = []
            }
            let input = DistributedStageForwardInput(
                requestID: requestID,
                stepIndex: stepIndex,
                positionRange: positionRange,
                positionIDs: positionIDs,
                tokenIDs: stageTokenIDs,
                blockIDsQ: stage.descriptor.role == .transformerLayers ? blockIDsQ : nil,
                blockIDsKV: stage.descriptor.role == .transformerLayers ? blockIDsKV : nil,
                softTokenSplice: stage.descriptor.role == .embeddings ? softTokenSplice : nil,
                hiddenState: hiddenState)
            let output = try await forward(stage: stage, input: input)
            try validate(output: output, from: stage, at: index, stepIndex: stepIndex)
            eagleTargetKVChunk = output.eagleTargetKVChunk ?? eagleTargetKVChunk
            eagleTargetFinalHidden = output.eagleTargetFinalHidden ?? eagleTargetFinalHidden
            hiddenState = output.hiddenState
            tokenID = output.tokenID
            topLogits = output.topLogits
        }

        if eagleTarget != nil {
            guard let eagleTargetKVChunk else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "EAGLE target forward did not return a representative KV chunk")
            }
            if emitToken, eagleTargetFinalHidden == nil {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "EAGLE target forward did not return final post-norm hidden")
            }
            guard var accumulator = eagleTargetAccumulators[requestID] else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "request_id \(requestID) has no EAGLE target accumulator")
            }
            try accumulator.append(
                eagleTargetKVChunk,
                finalHidden: eagleTargetFinalHidden)
            eagleTargetArtifacts = accumulator.snapshot
            eagleTargetAccumulators[requestID] = accumulator
        }

        if emitToken && tokenID == nil {
            throw DistributedStageExecutionError.invalidStageOutput(
                "final stage did not return a token id")
        }
        requestTracker.commitForward(firstFrame)
        return DistributedStageForwardOutput(
            stageID: emitToken ? stages.last!.descriptor.id : (hiddenState?.metadata.sourceStageID ?? stages.last!.descriptor.id),
            stepIndex: stepIndex,
            hiddenState: hiddenState,
            tokenID: tokenID,
            topLogits: topLogits,
            eagleTargetArtifacts: eagleTargetArtifacts)
    }

    private func forward(
        stage: DistributedStageHandle,
        input: DistributedStageForwardInput
    ) async throws -> DistributedStageForwardOutput {
        guard input.positionRange.count > 1, let streamedPrefillAdmission else {
            guard activePrefillStageID == nil else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "decode attempted while prefill stage \(activePrefillStageID!) is resident")
            }
            return try await stage.forward(input)
        }
        guard let resident = stage as? DistributedDecodeResidentStageHandle else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "stage \(stage.descriptor.id) does not support streamed prefill")
        }
        guard activePrefillStageID == nil else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "prefill stage \(activePrefillStageID!) is already resident")
        }

        activePrefillStageID = stage.descriptor.id
        do {
            try streamedPrefillAdmission.checkBeforeAssetLoad()
            try await resident.loadPrefill()
            guard resident.isPrefillResident else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "stage \(stage.descriptor.id) did not retain its admitted prefill asset")
            }
            let output = try await stage.forward(input)
            resident.unloadPrefill()
            activePrefillStageID = nil
            return output
        } catch {
            resident.unloadPrefill()
            activePrefillStageID = nil
            throw error
        }
    }

    public func reset(requestID: String) async throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("request_id is empty")
        }
        let control = DistributedRequestControl(requestID: requestID)
        try requestTracker.validateReset(control)
        for stage in stages {
            try await stage.reset(requestID: requestID)
        }
        if eagleTarget != nil {
            guard var accumulator = eagleTargetAccumulators[requestID] else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "request_id \(requestID) has no EAGLE target accumulator")
            }
            accumulator.reset()
            eagleTargetAccumulators[requestID] = accumulator
        }
        requestTracker.commitReset(control)
    }

    public func free(requestID: String) async {
        guard requestTracker.activeRequestIDs.contains(requestID) else { return }
        let control = DistributedRequestControl(requestID: requestID)
        for stage in stages {
            await stage.free(requestID: requestID)
        }
        eagleTargetAccumulators.removeValue(forKey: requestID)
        requestTracker.commitFree(control)
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

        let isEagleKVProducer = eagleTarget?.stageID == stage.descriptor.id
        if isEagleKVProducer {
            guard output.eagleTargetKVChunk != nil else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "eagle_target stage \(stage.descriptor.id) did not return auxiliary artifacts")
            }
        } else if output.eagleTargetKVChunk != nil {
            throw DistributedStageExecutionError.invalidStageOutput(
                "stage \(stage.descriptor.id) returned undeclared EAGLE target KV chunk")
        }

        let isEagleFinalHiddenProducer = eagleTarget?.finalHiddenStageID == stage.descriptor.id
        if isEagleFinalHiddenProducer {
            guard output.eagleTargetFinalHidden != nil else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "eagle_target final hidden stage \(stage.descriptor.id) did not return post-norm hidden")
            }
        } else if output.eagleTargetFinalHidden != nil {
            throw DistributedStageExecutionError.invalidStageOutput(
                "stage \(stage.descriptor.id) returned undeclared EAGLE target final hidden")
        }
        if output.eagleTargetArtifacts != nil {
            throw DistributedStageExecutionError.invalidStageOutput(
                "stage \(stage.descriptor.id) returned assembled EAGLE target artifacts")
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

public enum DistributedStagedStopReason: String, Sendable {
    case eos
    case maxTokens = "max_tokens"
    case contextLimit = "context_limit"
}

public struct DistributedStagedGenerationOptions: Hashable, Sendable {
    public let maxTokens: Int
    public let kvCapacity: Int?
    public let stopTokenIDs: Set<Int32>

    public init(
        maxTokens: Int = 64,
        kvCapacity: Int? = nil,
        stopTokenIDs: Set<Int32> = []
    ) {
        self.maxTokens = maxTokens
        self.kvCapacity = kvCapacity
        self.stopTokenIDs = stopTokenIDs
    }
}

public struct DistributedStagedGenerationResult: Hashable, Sendable {
    public let generatedTokenIDs: [Int32]
    public let promptTokenCount: Int
    public let stopReason: DistributedStagedStopReason
    public let kvCapacity: Int

    public var generatedTokenCount: Int {
        generatedTokenIDs.count
    }

    public init(
        generatedTokenIDs: [Int32],
        promptTokenCount: Int,
        stopReason: DistributedStagedStopReason,
        kvCapacity: Int
    ) {
        self.generatedTokenIDs = generatedTokenIDs
        self.promptTokenCount = promptTokenCount
        self.stopReason = stopReason
        self.kvCapacity = kvCapacity
    }
}

/// Thin token-loop wrapper for the same-machine staged-equivalence milestone.
///
/// This does not tokenize, sample logits, or load Core AI. The final stage is responsible for
/// returning the greedy next token; this wrapper only mirrors the coordinator-owned prefill/decode
/// request sequence and KV lifecycle.
public final class DistributedStagedEngine {
    public let pipeline: DistributedSameMachinePipeline
    public let maxContextLength: Int
    public let minKVCapacity: Int
    public let cacheGroups: DistributedStageCacheGroups?

    public init(
        pipeline: DistributedSameMachinePipeline,
        maxContextLength: Int,
        minKVCapacity: Int = 0,
        cacheGroups: DistributedStageCacheGroups? = nil
    ) throws {
        guard maxContextLength > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "max_context_length must be positive")
        }
        guard minKVCapacity >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "min_kv_capacity must be non-negative")
        }
        self.pipeline = pipeline
        self.maxContextLength = maxContextLength
        self.minKVCapacity = minKVCapacity
        self.cacheGroups = cacheGroups
        try cacheGroups?.validate()
    }

    public func generate(
        promptTokens: [Int32],
        options: DistributedStagedGenerationOptions = DistributedStagedGenerationOptions(),
        requestID: String = UUID().uuidString
    ) async throws -> DistributedStagedGenerationResult {
        guard !promptTokens.isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "prompt_tokens must be non-empty")
        }
        guard options.maxTokens >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "max_tokens must be non-negative")
        }
        let kvCapacity = try resolvedKVCapacity(
            promptCount: promptTokens.count,
            maxTokens: options.maxTokens,
            explicitKVCapacity: options.kvCapacity)

        guard options.maxTokens > 0 else {
            return DistributedStagedGenerationResult(
                generatedTokenIDs: [],
                promptTokenCount: promptTokens.count,
                stopReason: .maxTokens,
                kvCapacity: kvCapacity)
        }

        try await pipeline.allocate(
            requestID: requestID,
            kvCapacity: kvCapacity,
            cacheCapacities: resolvedCacheCapacities(kvCapacity: kvCapacity))
        do {
            var nextToken = try await pipelineNextToken(
                requestID: requestID,
                stepIndex: 0,
                positionRange: DistributedSequenceRange(
                    lowerBound: 0,
                    upperBound: promptTokens.count),
                tokenIDs: promptTokens)
            var generated: [Int32] = []
            var stopReason: DistributedStagedStopReason = .maxTokens

            while generated.count < options.maxTokens {
                if options.stopTokenIDs.contains(nextToken) {
                    stopReason = .eos
                    break
                }
                if promptTokens.count + generated.count >= maxContextLength {
                    stopReason = .contextLimit
                    break
                }

                generated.append(nextToken)
                guard generated.count < options.maxTokens else { break }

                let decodePosition = promptTokens.count + generated.count - 1
                nextToken = try await pipelineNextToken(
                    requestID: requestID,
                    stepIndex: generated.count,
                    positionRange: DistributedSequenceRange(
                        lowerBound: decodePosition,
                        upperBound: decodePosition + 1),
                    tokenIDs: [nextToken])
            }

            await pipeline.free(requestID: requestID)
            return DistributedStagedGenerationResult(
                generatedTokenIDs: generated,
                promptTokenCount: promptTokens.count,
                stopReason: stopReason,
                kvCapacity: kvCapacity)
        } catch {
            await pipeline.free(requestID: requestID)
            throw error
        }
    }

    private func pipelineNextToken(
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        tokenIDs: [Int32]
    ) async throws -> Int32 {
        let output = try await pipeline.forward(
            requestID: requestID,
            stepIndex: stepIndex,
            positionRange: positionRange,
            tokenIDs: tokenIDs)
        guard let tokenID = output.tokenID else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "staged pipeline did not return a token id")
        }
        return tokenID
    }

    private func resolvedKVCapacity(
        promptCount: Int,
        maxTokens: Int,
        explicitKVCapacity: Int?
    ) throws -> Int {
        let requested = explicitKVCapacity ?? (promptCount + maxTokens + 8)
        guard requested > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "kv_capacity must be positive")
        }
        let floored = max(requested, minKVCapacity + maxTokens)
        let capacity = min(floored, maxContextLength)
        guard capacity >= promptCount else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "kv_capacity is smaller than prompt")
        }
        return capacity
    }

    private func resolvedCacheCapacities(kvCapacity: Int) -> [String: Int]? {
        cacheGroups?.capacities(forKVCapacity: kvCapacity)
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
    case invalidBoundaryTensor(String)
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
        case .invalidBoundaryTensor(let message):
            return "Invalid boundary tensor: \(message)"
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
        if let decodeAssetName = stage.decodeAssetName, trimmed(decodeAssetName).isEmpty {
            throw DistributedRuntimeValidationError.invalidIdentifier(
                field: "stage.decode_asset_name")
        }
        if let vocabSize = stage.vocabSize, vocabSize <= 0 {
            throw DistributedRuntimeValidationError.invalidIdentifier(field: "stage.vocab_size")
        }
        if stage.functionMap?.validationErrorMessage != nil {
            throw DistributedRuntimeValidationError.invalidIdentifier(field: "stage.function_map")
        }
        try validatePrefillExtraInputs(stage.prefillExtraInputs, role: stage.role)
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

    private static func validatePrefillExtraInputs(
        _ values: [String],
        role: DistributedStageRole
    ) throws {
        guard !values.isEmpty else { return }
        guard role == .transformerLayers else {
            throw DistributedRuntimeValidationError.invalidIdentifier(
                field: "stage.prefill_extra_inputs")
        }
        let supported = Set([
            DistributedStageIOTensorName.blockIDsQ.rawValue,
            DistributedStageIOTensorName.blockIDsKV.rawValue,
        ])
        guard Set(values) == supported, values.count == supported.count else {
            throw DistributedRuntimeValidationError.invalidIdentifier(
                field: "stage.prefill_extra_inputs")
        }
    }

    public static func validate(plan: DistributedStagePlan) throws {
        guard plan.totalLayerCount > 0 else {
            throw DistributedRuntimeValidationError.invalidTotalLayerCount(plan.totalLayerCount)
        }
        if let message = plan.boundaryTensor?.validationErrorMessage {
            throw DistributedRuntimeValidationError.invalidBoundaryTensor(message)
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
        if let boundaryTensor = plan.boundaryTensor {
            try validate(packet: packet, matches: boundaryTensor)
        }
    }

    private static func validate(
        packet: DistributedHiddenStatePacketMetadata,
        matches boundaryTensor: DistributedBoundaryTensorSpec
    ) throws {
        guard packet.scalarType == boundaryTensor.scalarType else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "hidden-state packet scalar_type \(packet.scalarType.rawValue) does not match boundary tensor \(boundaryTensor.scalarType.rawValue)")
        }
        guard packet.shape.count == 3 else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "hidden-state packet shape must be [batch, sequence, hidden]")
        }
        let expected = boundaryTensor.shape
        guard packet.shape[0] == expected[0],
            (expected[1] == -1 || packet.shape[1] == expected[1]),
            packet.shape[2] == expected[2]
        else {
            throw DistributedRuntimeValidationError.invalidPacket(
                "hidden-state packet shape \(packet.shape) does not match boundary tensor shape \(expected)")
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
