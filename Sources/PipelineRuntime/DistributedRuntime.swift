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
    public let memoryGB: Double

    enum CodingKeys: String, CodingKey {
        case id
        case role
        case layerSpec = "layers"
        case assetName = "asset_name"
        case resolvedAssetPath = "resolved_asset_path"
        case memoryGB = "memory_gb"
    }

    public init(
        id: String,
        role: DistributedStageRole,
        layerSpec: DistributedStageLayerSpec,
        assetName: String,
        resolvedAssetPath: String? = nil,
        memoryGB: Double
    ) {
        self.id = id
        self.role = role
        self.layerSpec = layerSpec
        self.assetName = assetName
        self.resolvedAssetPath = resolvedAssetPath
        self.memoryGB = memoryGB
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
            workerID: workerID)
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

/// Normalized staged manifest used by the CLI planner, same-machine harness, and future workers.
public struct DistributedStageManifest: Hashable, Sendable {
    public static let currentSchema = "caix.cluster.stage_manifest.v0"

    public let schema: String?
    public let modelName: String
    public let totalLayerCount: Int
    public let totalLayerCountDerived: Bool
    public let stages: [DistributedStageManifestStage]
    public let boundaryTensor: DistributedBoundaryTensorSpec?
    public let runtimePlan: DistributedStagePlan

    public init(
        schema: String? = Self.currentSchema,
        modelName: String,
        totalLayerCount: Int,
        totalLayerCountDerived: Bool = false,
        stages: [DistributedStageManifestStage],
        boundaryTensor: DistributedBoundaryTensorSpec? = nil
    ) throws {
        self.schema = schema
        self.modelName = modelName
        self.totalLayerCount = totalLayerCount
        self.totalLayerCountDerived = totalLayerCountDerived
        self.stages = stages
        self.boundaryTensor = boundaryTensor
        try boundaryTensor?.validate()
        self.runtimePlan = DistributedStagePlan(
            modelName: modelName,
            totalLayerCount: totalLayerCount,
            stages: stages.map { $0.descriptor() },
            workers: [],
            boundaryTensor: boundaryTensor)
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

        return try DistributedStageManifest(
            schema: schema,
            modelName: modelName,
            totalLayerCount: totalLayerCount,
            totalLayerCountDerived: totalLayerCountDerived,
            stages: stages,
            boundaryTensor: body.boundary?.hiddenState ?? body.boundaryTensor
                ?? root.boundary?.hiddenState ?? root.boundaryTensor)
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

        return DistributedStageManifestStage(
            id: id,
            role: role,
            layerSpec: layerSpec,
            assetName: assetName,
            resolvedAssetPath: resolveAssetPath(assetName, baseURL: baseURL),
            memoryGB: memoryGB)
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
            boundaryTensor: boundaryTensor)
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

    enum CodingKeys: String, CodingKey {
        case schema
        case model
        case modelName = "model_name"
        case totalLayerCount = "total_layer_count"
        case totalLayers = "total_layers"
        case stages
        case boundary
        case boundaryTensor = "boundary_tensor"
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
    public let boundaryTensor: DistributedBoundaryTensorSpec?

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case totalLayerCount = "total_layer_count"
        case stages
        case workers
        case boundaryTensor = "boundary_tensor"
    }

    public init(
        modelName: String,
        totalLayerCount: Int,
        stages: [DistributedStageDescriptor],
        workers: [DistributedWorkerEndpoint] = [],
        boundaryTensor: DistributedBoundaryTensorSpec? = nil
    ) {
        self.modelName = modelName
        self.totalLayerCount = totalLayerCount
        self.stages = stages
        self.workers = workers
        self.boundaryTensor = boundaryTensor
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
    public let planIntegrityHash: String
    public let freeMemoryBytes: UInt64?
    public let computeUnit: String?
    public let labels: [String: String]

    enum CodingKeys: String, CodingKey {
        case stage
        case hiddenSize = "hidden_size"
        case boundaryScalarType = "boundary_scalar_type"
        case cacheContract = "cache_contract"
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
        planIntegrityHash: String,
        freeMemoryBytes: UInt64? = nil,
        computeUnit: String? = nil,
        labels: [String: String] = [:]
    ) {
        self.stage = stage
        self.hiddenSize = hiddenSize
        self.boundaryScalarType = boundaryScalarType
        self.cacheContract = cacheContract
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

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case kvCapacity = "kv_capacity"
    }

    public init(requestID: String, kvCapacity: Int) {
        self.requestID = requestID
        self.kvCapacity = kvCapacity
    }

    public func validate() throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidControlFrame("request_id is empty")
        }
        guard kvCapacity > 0 else {
            throw DistributedStageExecutionError.invalidControlFrame("kv_capacity must be positive")
        }
    }
}

public struct DistributedStageForwardFrame: Codable, Hashable, Sendable {
    public let stageID: String
    public let requestID: String
    public let stepIndex: Int
    public let positionRange: DistributedSequenceRange
    public let positionIDs: [Int32]
    public let tokenIDs: [Int32]
    public let hiddenState: DistributedHiddenStatePacketMetadata?

    enum CodingKeys: String, CodingKey {
        case stageID = "stage_id"
        case requestID = "request_id"
        case stepIndex = "step_index"
        case positionRange = "position_range"
        case positionIDs = "position_ids"
        case tokenIDs = "token_ids"
        case hiddenState = "hidden_state"
    }

    public init(
        stageID: String,
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        positionIDs: [Int32],
        tokenIDs: [Int32] = [],
        hiddenState: DistributedHiddenStatePacketMetadata? = nil
    ) {
        self.stageID = stageID
        self.requestID = requestID
        self.stepIndex = stepIndex
        self.positionRange = positionRange
        self.positionIDs = positionIDs
        self.tokenIDs = tokenIDs
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
        guard positionIDs.count == positionRange.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_ids count must match position_range")
        }
        guard let descriptor = plan.stage(id: stageID) else {
            throw DistributedStageExecutionError.invalidForwardInput("unknown stage_id \(stageID)")
        }

        switch descriptor.role {
        case .embeddings:
            guard tokenIDs.count == positionRange.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "token_ids count must match position_range")
            }
            guard hiddenState == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "embeddings stage must not receive a hidden state")
            }
        case .transformerLayers, .finalNormHead:
            guard tokenIDs.isEmpty else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(descriptor.role.rawValue) stage must not receive token_ids")
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
                hiddenState: hiddenState))
            guard output.stageID == handle.descriptor.id else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "output stage_id \(output.stageID) does not match worker stage \(handle.descriptor.id)")
            }
            guard output.stepIndex == frame.stepIndex else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "output step_index does not match request")
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
    public let hiddenState: DistributedHiddenStatePacket?

    public init(
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        positionIDs: [Int32],
        tokenIDs: [Int32] = [],
        hiddenState: DistributedHiddenStatePacket? = nil
    ) {
        self.requestID = requestID
        self.stepIndex = stepIndex
        self.positionRange = positionRange
        self.positionIDs = positionIDs
        self.tokenIDs = tokenIDs
        self.hiddenState = hiddenState
    }
}

public typealias DistributedWorkerFrameRoundTrip =
    (DistributedWorkerWireFrame) async throws -> DistributedWorkerWireFrame?

public final class DistributedRemoteStageHandle: DistributedStageHandle {
    public let descriptor: DistributedStageDescriptor
    public let plan: DistributedStagePlan
    private let roundTrip: DistributedWorkerFrameRoundTrip

    public init(
        plan: DistributedStagePlan,
        descriptor: DistributedStageDescriptor,
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
        }
    }
}

/// In-process coordinator for one staged forward. This is the same-machine milestone harness;
/// concrete handles can be fake test stages now and Core AI stage handles later.
public final class DistributedSameMachinePipeline {
    public let plan: DistributedStagePlan
    private let stages: [DistributedStageHandle]
    private var requestTracker = DistributedWorkerRequestTracker()

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
        try self.init(plan: manifest.runtimePlan, stages: orderedHandles)
    }

    public static func make(
        manifest: DistributedStageManifest,
        handleFactory: DistributedStageHandleFactory
    ) async throws -> DistributedSameMachinePipeline {
        var handles: [DistributedStageHandle] = []
        handles.reserveCapacity(manifest.stages.count)
        for stage in manifest.stages {
            guard let descriptor = manifest.runtimePlan.stage(id: stage.id) else {
                throw DistributedStageExecutionError.missingStageHandle(stage.id)
            }
            let context = DistributedStageHandleFactoryContext(
                stage: stage, manifest: manifest, descriptor: descriptor)
            handles.append(try await handleFactory.makeStageHandle(for: context))
        }
        return try DistributedSameMachinePipeline(plan: manifest.runtimePlan, stages: handles)
    }

    public func allocate(requestID: String, kvCapacity: Int) async throws {
        guard !requestID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("request_id is empty")
        }
        guard kvCapacity > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput("kv_capacity must be positive")
        }
        let allocation = DistributedStageAllocation(requestID: requestID, kvCapacity: kvCapacity)
        try requestTracker.validateAllocate(allocation)
        for stage in stages {
            try await stage.allocate(allocation)
        }
        requestTracker.commitAllocate(allocation)
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
        guard positionRange.isValid else {
            throw DistributedStageExecutionError.invalidForwardInput("position_range is invalid")
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
        let positionIDs = try Self.positionIDs(for: positionRange)
        let firstFrame = DistributedStageForwardFrame(
            stageID: stages.first!.descriptor.id,
            requestID: requestID,
            stepIndex: stepIndex,
            positionRange: positionRange,
            positionIDs: positionIDs,
            tokenIDs: tokenIDs)
        try firstFrame.validate(against: plan)
        try requestTracker.validateForward(firstFrame)

        for (index, stage) in stages.enumerated() {
            let input = DistributedStageForwardInput(
                requestID: requestID,
                stepIndex: stepIndex,
                positionRange: positionRange,
                positionIDs: positionIDs,
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
        requestTracker.commitForward(firstFrame)
        return DistributedStageForwardOutput(
            stageID: stages.last!.descriptor.id,
            stepIndex: stepIndex,
            hiddenState: hiddenState,
            tokenID: tokenID)
    }

    private static func positionIDs(
        for positionRange: DistributedSequenceRange
    ) throws -> [Int32] {
        guard positionRange.lowerBound >= Int(Int32.min),
            positionRange.upperBound <= Int(Int32.max)
        else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_ids exceed Int32 range")
        }
        return (positionRange.lowerBound..<positionRange.upperBound).map { Int32($0) }
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
        requestTracker.commitReset(control)
    }

    public func free(requestID: String) async {
        guard requestTracker.activeRequestIDs.contains(requestID) else { return }
        let control = DistributedRequestControl(requestID: requestID)
        for stage in stages {
            await stage.free(requestID: requestID)
        }
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

    public init(
        pipeline: DistributedSameMachinePipeline,
        maxContextLength: Int
    ) throws {
        guard maxContextLength > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "max_context_length must be positive")
        }
        self.pipeline = pipeline
        self.maxContextLength = maxContextLength
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

        try await pipeline.allocate(requestID: requestID, kvCapacity: kvCapacity)
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
        let capacity = min(requested, maxContextLength)
        guard capacity >= promptCount else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "kv_capacity is smaller than prompt")
        }
        return capacity
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
