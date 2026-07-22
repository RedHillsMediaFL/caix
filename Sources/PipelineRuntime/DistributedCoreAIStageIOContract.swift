#if COREAI_RUNTIME
import CoreAI
import Foundation

extension DistributedStageIOContract {
    init(
        coreAIFunctionName functionName: String,
        descriptor: InferenceFunctionDescriptor
    ) throws {
        try Self.validateStateDescriptors(descriptor)
        let cacheIO = try DistributedCoreAIStageCacheIO.extracted(from: descriptor)
        let inputExclusions = cacheIO.stageIOInputExclusions
        let outputExclusions = cacheIO.stageIOOutputExclusions
        self.init(
            functionName: functionName,
            inputs: try descriptor.inputNames.filter { name in
                !inputExclusions.contains(name)
            }.map { name in
                try Self.inputTensor(name: name, descriptor: descriptor)
            },
            outputs: try descriptor.outputNames.filter { name in
                !outputExclusions.contains(name)
            }.map { name in
                try Self.outputTensor(name: name, descriptor: descriptor)
            },
            stateNames: descriptor.stateNames)
    }

    static func extractedFromCoreAI(
        functionName: String = "main",
        descriptor: InferenceFunctionDescriptor
    ) throws -> DistributedStageIOContract {
        try DistributedStageIOContract(
            coreAIFunctionName: functionName,
            descriptor: descriptor)
    }

    private static func inputTensor(
        name: String,
        descriptor: InferenceFunctionDescriptor
    ) throws -> DistributedStageIOTensor {
        guard case .ndArray(let ndArrayDescriptor) = descriptor.inputDescriptor(of: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage input '\(name)' is not an NDArray")
        }
        return try tensor(name: name, ndArrayDescriptor: ndArrayDescriptor, kind: "input")
    }

    private static func outputTensor(
        name: String,
        descriptor: InferenceFunctionDescriptor
    ) throws -> DistributedStageIOTensor {
        guard case .ndArray(let ndArrayDescriptor) = descriptor.outputDescriptor(of: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage output '\(name)' is not an NDArray")
        }
        return try tensor(name: name, ndArrayDescriptor: ndArrayDescriptor, kind: "output")
    }

    private static func tensor(
        name: String,
        ndArrayDescriptor: NDArrayDescriptor,
        kind: String
    ) throws -> DistributedStageIOTensor {
        return DistributedStageIOTensor(
            name: name,
            shape: ndArrayDescriptor.shape,
            scalarType: try scalarType(
                ndArrayDescriptor.scalarType,
                tensorName: name,
                kind: kind))
    }

    private static func validateStateDescriptors(
        _ descriptor: InferenceFunctionDescriptor
    ) throws {
        for stateName in descriptor.stateNames {
            guard case .ndArray = descriptor.stateDescriptor(of: stateName) else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage state '\(stateName)' is not an NDArray")
            }
        }
    }

    private static func scalarType(
        _ scalarType: NDArray.ScalarType,
        tensorName: String,
        kind: String
    ) throws -> DistributedStageIOScalarType {
        switch scalarType {
        case .int32:
            return .int32
        case .float16:
            return .float16
        case .bfloat16:
            return .bfloat16
        case .float32:
            return .float32
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage \(kind) '\(tensorName)' scalar type \(scalarType) "
                    + "is unsupported (expected int32/float16/bfloat16/float32)")
        }
    }
}

extension DistributedStageHandleFactoryContext {
    func validateCoreAIStageIOContract(
        functionName: String = "main",
        descriptor: InferenceFunctionDescriptor,
        vocabSize: Int? = nil
    ) throws -> DistributedStageIOContract {
        let contract = try DistributedStageIOContract.extractedFromCoreAI(
            functionName: functionName,
            descriptor: descriptor)
        try validateStageIOContract(contract, vocabSize: vocabSize)
        return contract
    }
}

enum DistributedCoreAIStageCacheContract {
    case none
    case stateful
    case explicitOutputs
}

struct DistributedCoreAIStageCacheGroupIO {
    let groupName: String
    let keyCacheName: String
    let valueCacheName: String
    let keyCacheDescriptor: NDArrayDescriptor
    let valueCacheDescriptor: NDArrayDescriptor
    let keyCacheOutputDescriptor: NDArrayDescriptor
    let valueCacheOutputDescriptor: NDArrayDescriptor
}

struct DistributedCoreAIStageCacheIO {
    let contract: DistributedCoreAIStageCacheContract
    let groups: [DistributedCoreAIStageCacheGroupIO]

    var keyCacheName: String? { groups.first?.keyCacheName }
    var valueCacheName: String? { groups.first?.valueCacheName }
    var keyCacheDescriptor: NDArrayDescriptor? { groups.first?.keyCacheDescriptor }
    var valueCacheDescriptor: NDArrayDescriptor? { groups.first?.valueCacheDescriptor }
    var keyCacheOutputDescriptor: NDArrayDescriptor? { groups.first?.keyCacheOutputDescriptor }
    var valueCacheOutputDescriptor: NDArrayDescriptor? { groups.first?.valueCacheOutputDescriptor }

    var stageIOInputExclusions: Set<String> {
        switch contract {
        case .explicitOutputs:
            return Set(groups.flatMap { [$0.keyCacheName, $0.valueCacheName] })
        case .none, .stateful:
            return []
        }
    }

    var stageIOOutputExclusions: Set<String> {
        switch contract {
        case .explicitOutputs:
            return Set(groups.flatMap { [$0.keyCacheName, $0.valueCacheName] })
        case .none, .stateful:
            return []
        }
    }

    static func extracted(from descriptor: InferenceFunctionDescriptor) throws
        -> DistributedCoreAIStageCacheIO
    {
        let stateGroups = try stateCacheGroups(from: descriptor)
        let explicitGroups = try explicitCacheGroups(from: descriptor)
        if !stateGroups.isEmpty {
            return DistributedCoreAIStageCacheIO(
                contract: .stateful,
                groups: stateGroups)
        }
        if !explicitGroups.isEmpty {
            return DistributedCoreAIStageCacheIO(
                contract: .explicitOutputs,
                groups: explicitGroups)
        }
        if descriptor.stateNames.isEmpty
            && !containsCacheTensorName(descriptor.inputNames)
            && !containsCacheTensorName(descriptor.outputNames)
        {
            return DistributedCoreAIStageCacheIO(
                contract: .none,
                groups: [])
        }
        throw CoreAIPipeline.RuntimeError.modelContract(
            "expected no KV cache, stateful KV states, or explicit-cache inputs/outputs; "
                + ioSummary(descriptor))
    }

    private static func pick(_ wanted: String, _ names: [String], index: Int) -> String {
        names.contains(wanted) ? wanted : names[index]
    }

    private static func explicitCacheGroups(
        from descriptor: InferenceFunctionDescriptor
    ) throws -> [DistributedCoreAIStageCacheGroupIO] {
        guard descriptor.stateNames.isEmpty else { return [] }

        var keyByGroup: [String: String] = [:]
        var valueByGroup: [String: String] = [:]
        var groupOrder: [String] = []
        let outputNames = Set(descriptor.outputNames)

        for inputName in descriptor.inputNames {
            if let groupName = cacheGroupName(for: inputName, suffix: "keyCache") {
                guard outputNames.contains(inputName) else {
                    throw CoreAIPipeline.RuntimeError.modelContract(
                        "distributed stage explicit KV cache key input '\(inputName)' is missing a matching output")
                }
                if keyByGroup[groupName] != nil {
                    throw CoreAIPipeline.RuntimeError.modelContract(
                        "distributed stage explicit KV cache group '\(groupName)' has duplicate key input")
                }
                keyByGroup[groupName] = inputName
                groupOrder.append(groupName)
            } else if let groupName = cacheGroupName(for: inputName, suffix: "valueCache") {
                guard outputNames.contains(inputName) else {
                    throw CoreAIPipeline.RuntimeError.modelContract(
                        "distributed stage explicit KV cache value input '\(inputName)' is missing a matching output")
                }
                if valueByGroup[groupName] != nil {
                    throw CoreAIPipeline.RuntimeError.modelContract(
                        "distributed stage explicit KV cache group '\(groupName)' has duplicate value input")
                }
                valueByGroup[groupName] = inputName
                if !groupOrder.contains(groupName) {
                    groupOrder.append(groupName)
                }
            }
        }

        guard !keyByGroup.isEmpty || !valueByGroup.isEmpty else { return [] }

        var groups: [DistributedCoreAIStageCacheGroupIO] = []
        for groupName in groupOrder {
            guard let keyName = keyByGroup[groupName],
                let valueName = valueByGroup[groupName]
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage explicit KV cache group '\(groupName)' is missing a key/value pair")
            }
            guard case .ndArray(let inputKeyDesc) = descriptor.inputDescriptor(of: keyName),
                case .ndArray(let inputValueDesc) = descriptor.inputDescriptor(of: valueName)
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage explicit KV cache inputs are not NDArrays")
            }
            guard case .ndArray(let outputKeyDesc) = descriptor.outputDescriptor(of: keyName),
                case .ndArray(let outputValueDesc) = descriptor.outputDescriptor(of: valueName)
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage explicit KV cache outputs are not NDArrays")
            }
            groups.append(DistributedCoreAIStageCacheGroupIO(
                groupName: groupName,
                keyCacheName: keyName,
                valueCacheName: valueName,
                keyCacheDescriptor: inputKeyDesc,
                valueCacheDescriptor: inputValueDesc,
                keyCacheOutputDescriptor: outputKeyDesc,
                valueCacheOutputDescriptor: outputValueDesc))
        }
        return groups
    }

    private static func containsCacheTensorName(_ names: [String]) -> Bool {
        names.contains { name in
            cacheGroupName(for: name, suffix: "keyCache") != nil
                || cacheGroupName(for: name, suffix: "valueCache") != nil
        }
    }

    private static func stateCacheGroups(
        from descriptor: InferenceFunctionDescriptor
    ) throws -> [DistributedCoreAIStageCacheGroupIO] {
        guard !descriptor.stateNames.isEmpty else { return [] }
        if descriptor.stateNames.count == 2 {
            let keyName = pick("keyCache", descriptor.stateNames, index: 0)
            let valueName = pick("valueCache", descriptor.stateNames, index: 1)
            guard case .ndArray(let keyDesc) = descriptor.stateDescriptor(of: keyName),
                case .ndArray(let valueDesc) = descriptor.stateDescriptor(of: valueName)
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache states are not NDArrays")
            }
            return [
                DistributedCoreAIStageCacheGroupIO(
                    groupName: "default",
                    keyCacheName: keyName,
                    valueCacheName: valueName,
                    keyCacheDescriptor: keyDesc,
                    valueCacheDescriptor: valueDesc,
                    keyCacheOutputDescriptor: keyDesc,
                    valueCacheOutputDescriptor: valueDesc)
            ]
        }
        guard descriptor.stateNames.count.isMultiple(of: 2) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage stateful KV cache must declare key/value state pairs; "
                    + ioSummary(descriptor))
        }
        var keyByGroup: [String: String] = [:]
        var valueByGroup: [String: String] = [:]
        var groupOrder: [String] = []
        for stateName in descriptor.stateNames {
            if let groupName = cacheGroupName(for: stateName, suffix: "keyCache") {
                if keyByGroup[groupName] != nil {
                    throw CoreAIPipeline.RuntimeError.modelContract(
                        "distributed stage KV cache group '\(groupName)' has duplicate key state")
                }
                keyByGroup[groupName] = stateName
                groupOrder.append(groupName)
            } else if let groupName = cacheGroupName(for: stateName, suffix: "valueCache") {
                if valueByGroup[groupName] != nil {
                    throw CoreAIPipeline.RuntimeError.modelContract(
                        "distributed stage KV cache group '\(groupName)' has duplicate value state")
                }
                valueByGroup[groupName] = stateName
                if !groupOrder.contains(groupName) {
                    groupOrder.append(groupName)
                }
            } else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache state '\(stateName)' must end in keyCache/valueCache")
            }
        }
        var groups: [DistributedCoreAIStageCacheGroupIO] = []
        for groupName in groupOrder {
            guard let keyName = keyByGroup[groupName],
                let valueName = valueByGroup[groupName]
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache group '\(groupName)' is missing a key/value state")
            }
            guard case .ndArray(let keyDesc) = descriptor.stateDescriptor(of: keyName),
                case .ndArray(let valueDesc) = descriptor.stateDescriptor(of: valueName)
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache states are not NDArrays")
            }
            groups.append(DistributedCoreAIStageCacheGroupIO(
                groupName: groupName,
                keyCacheName: keyName,
                valueCacheName: valueName,
                keyCacheDescriptor: keyDesc,
                valueCacheDescriptor: valueDesc,
                keyCacheOutputDescriptor: keyDesc,
                valueCacheOutputDescriptor: valueDesc))
        }
        return groups
    }

    private static func cacheGroupName(for stateName: String, suffix: String) -> String? {
        guard stateName.hasSuffix(suffix) else { return nil }
        var prefix = String(stateName.dropLast(suffix.count))
        if prefix.hasSuffix("_") || prefix.hasSuffix(".") || prefix.hasSuffix("/") {
            prefix.removeLast()
        }
        return prefix.isEmpty ? "default" : prefix
    }

    private static func ioSummary(_ descriptor: InferenceFunctionDescriptor) -> String {
        "inputs=\(descriptor.inputNames), outputs=\(descriptor.outputNames), states=\(descriptor.stateNames)"
    }
}

private struct DistributedCoreAIStageNDArrayBinding {
    let name: String
    let descriptor: NDArrayDescriptor
}

private struct DistributedCoreAIStageExecutionIO {
    let inputIDs: DistributedCoreAIStageNDArrayBinding?
    let positionIDs: DistributedCoreAIStageNDArrayBinding
    let hiddenStatesInput: DistributedCoreAIStageNDArrayBinding?
    let hiddenStatesOutput: DistributedCoreAIStageNDArrayBinding?
    let logitsOutput: DistributedCoreAIStageNDArrayBinding?
    let ropeCosInput: DistributedCoreAIStageNDArrayBinding?
    let ropeSinInput: DistributedCoreAIStageNDArrayBinding?
    let blockIDsQInput: DistributedCoreAIStageNDArrayBinding?
    let blockIDsKVInput: DistributedCoreAIStageNDArrayBinding?

    static func bound(
        for stage: DistributedStageDescriptor,
        descriptor: InferenceFunctionDescriptor,
        vocabSize: Int?
    ) throws -> DistributedCoreAIStageExecutionIO {
        let positionIDs = try input(.positionIDs, descriptor: descriptor)
        switch stage.role {
        case .embeddings:
            return DistributedCoreAIStageExecutionIO(
                inputIDs: try input(.inputIDs, descriptor: descriptor),
                positionIDs: positionIDs,
                hiddenStatesInput: nil,
                hiddenStatesOutput: try output(.hiddenStates, descriptor: descriptor),
                logitsOutput: nil,
                ropeCosInput: nil,
                ropeSinInput: nil,
                blockIDsQInput: nil,
                blockIDsKVInput: nil)
        case .transformerLayers:
            return DistributedCoreAIStageExecutionIO(
                inputIDs: try optionalInput(.inputIDs, descriptor: descriptor),
                positionIDs: positionIDs,
                hiddenStatesInput: try input(.hiddenStates, descriptor: descriptor),
                hiddenStatesOutput: try output(.hiddenStates, descriptor: descriptor),
                logitsOutput: nil,
                ropeCosInput: try stage.rope.map {
                    try input(named: $0.cosInputName, descriptor: descriptor)
                },
                ropeSinInput: try stage.rope.map {
                    try input(named: $0.sinInputName, descriptor: descriptor)
                },
                blockIDsQInput: try optionalInput(.blockIDsQ, descriptor: descriptor),
                blockIDsKVInput: try optionalInput(.blockIDsKV, descriptor: descriptor))
        case .finalNormHead:
            guard vocabSize != nil else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed final_norm_head stage '\(stage.id)' requires vocab_size")
            }
            return DistributedCoreAIStageExecutionIO(
                inputIDs: try optionalInput(.inputIDs, descriptor: descriptor),
                positionIDs: positionIDs,
                hiddenStatesInput: try input(.hiddenStates, descriptor: descriptor),
                hiddenStatesOutput: nil,
                logitsOutput: try output(.logits, descriptor: descriptor),
                ropeCosInput: nil,
                ropeSinInput: nil,
                blockIDsQInput: nil,
                blockIDsKVInput: nil)
        }
    }

    private static func input(
        _ tensorName: DistributedStageIOTensorName,
        descriptor: InferenceFunctionDescriptor
    ) throws -> DistributedCoreAIStageNDArrayBinding {
        let name = tensorName.rawValue
        guard case .ndArray(let ndArrayDescriptor) = descriptor.inputDescriptor(of: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage input '\(name)' is not an NDArray")
        }
        return DistributedCoreAIStageNDArrayBinding(
            name: name,
            descriptor: ndArrayDescriptor)
    }

    private static func optionalInput(
        _ tensorName: DistributedStageIOTensorName,
        descriptor: InferenceFunctionDescriptor
    ) throws -> DistributedCoreAIStageNDArrayBinding? {
        let name = tensorName.rawValue
        guard descriptor.inputNames.contains(name) else { return nil }
        return try input(tensorName, descriptor: descriptor)
    }

    private static func input(
        named name: String,
        descriptor: InferenceFunctionDescriptor
    ) throws -> DistributedCoreAIStageNDArrayBinding {
        guard case .ndArray(let ndArrayDescriptor) = descriptor.inputDescriptor(of: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage input '\(name)' is not an NDArray")
        }
        return DistributedCoreAIStageNDArrayBinding(
            name: name,
            descriptor: ndArrayDescriptor)
    }

    private static func output(
        _ tensorName: DistributedStageIOTensorName,
        descriptor: InferenceFunctionDescriptor
    ) throws -> DistributedCoreAIStageNDArrayBinding {
        let name = tensorName.rawValue
        guard case .ndArray(let ndArrayDescriptor) = descriptor.outputDescriptor(of: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage output '\(name)' is not an NDArray")
        }
        return DistributedCoreAIStageNDArrayBinding(
            name: name,
            descriptor: ndArrayDescriptor)
    }
}

public final class DistributedCoreAIStageHandleFactory: DistributedStageHandleFactory {
    private let functionName: String?
    private let vocabSize: Int?

    public init(functionName: String? = nil, vocabSize: Int? = nil) {
        self.functionName = functionName
        self.vocabSize = vocabSize
    }

    public func makeStageHandle(
        for context: DistributedStageHandleFactoryContext
    ) async throws -> DistributedStageHandle {
        try await DistributedCoreAIStageHandle.load(
            for: context,
            functionName: functionName,
            vocabSize: vocabSize)
    }
}

/// Factory used by the streamed-prefill residency contract. It specializes only the small decode
/// asset at startup; the corresponding prefill asset remains cold until the coordinator admits it.
public final class DistributedCoreAIDecodeResidentStageHandleFactory:
    DistributedStageHandleFactory
{
    private let functionName: String?
    private let vocabSize: Int?

    public init(functionName: String? = nil, vocabSize: Int? = nil) {
        self.functionName = functionName
        self.vocabSize = vocabSize
    }

    public func makeStageHandle(
        for context: DistributedStageHandleFactoryContext
    ) async throws -> DistributedStageHandle {
        try await DistributedCoreAIStageHandle.loadDecodeResident(
            for: context,
            functionName: functionName,
            vocabSize: vocabSize)
    }
}

private struct DistributedCoreAIStageExecutionContext {
    let model: AIModel
    let assetURL: URL
    let function: InferenceFunction
    let functionName: String
    let functionDescriptor: InferenceFunctionDescriptor
    let ioContract: DistributedStageIOContract
    let executionIO: DistributedCoreAIStageExecutionIO
    let cacheIO: DistributedCoreAIStageCacheIO
}

public final class DistributedCoreAIStageHandle:
    DistributedStageHandle,
    DistributedDecodeResidentStageHandle
{
    public let descriptor: DistributedStageDescriptor
    public let assetURL: URL
    public let functionName: String
    public private(set) var ioContract: DistributedStageIOContract

    public var acceptsTokenIDs: Bool {
        decodeContext.executionIO.inputIDs != nil
    }

    public var isPrefillResident: Bool { prefillContext != nil }

    private var prefillContext: DistributedCoreAIStageExecutionContext?
    private let decodeContext: DistributedCoreAIStageExecutionContext
    private let streamsPrefill: Bool
    private let boundaryTensor: DistributedBoundaryTensorSpec?
    private let nextStageID: String?
    private let vocabSize: Int?
    private var requestStates: [String: DistributedCoreAIStageRequestState] = [:]

    private init(
        descriptor: DistributedStageDescriptor,
        assetURL: URL,
        functionName: String,
        ioContract: DistributedStageIOContract,
        prefillContext: DistributedCoreAIStageExecutionContext?,
        decodeContext: DistributedCoreAIStageExecutionContext,
        streamsPrefill: Bool,
        boundaryTensor: DistributedBoundaryTensorSpec?,
        nextStageID: String?,
        vocabSize: Int?
    ) {
        self.descriptor = descriptor
        self.assetURL = assetURL
        self.functionName = functionName
        self.ioContract = ioContract
        self.prefillContext = prefillContext
        self.decodeContext = decodeContext
        self.streamsPrefill = streamsPrefill
        self.boundaryTensor = boundaryTensor
        self.nextStageID = nextStageID
        self.vocabSize = vocabSize
    }

    public static func load(
        for context: DistributedStageHandleFactoryContext,
        functionName: String? = nil,
        vocabSize: Int? = nil
    ) async throws -> DistributedCoreAIStageHandle {
        let assetURL = try context.requireExistingAssetURL()
        let resolvedFunctionName = functionName ?? context.mainFunctionName
        let resolvedVocabSize = vocabSize ?? context.vocabSize
        let specialization = Self.specializationOptions()
        let prefillContext = try await Self.loadExecutionContext(
            assetURL: assetURL,
            functionName: resolvedFunctionName,
            descriptor: context.descriptor,
            boundaryTensor: context.boundaryTensor,
            vocabSize: resolvedVocabSize,
            specialization: specialization)
        let decodeContext = try await Self.loadDecodeExecutionContext(
            for: context,
            mainContext: prefillContext,
            specialization: specialization,
            vocabSize: resolvedVocabSize)
        try Self.validateCacheCompatibility(
            prefill: prefillContext.cacheIO,
            decode: decodeContext.cacheIO,
            stageID: context.descriptor.id)
        return DistributedCoreAIStageHandle(
            descriptor: context.descriptor,
            assetURL: assetURL,
            functionName: resolvedFunctionName,
            ioContract: prefillContext.ioContract,
            prefillContext: prefillContext,
            decodeContext: decodeContext,
            streamsPrefill: false,
            boundaryTensor: context.boundaryTensor,
            nextStageID: context.nextStage?.id,
            vocabSize: resolvedVocabSize)
    }

    public static func loadDecodeResident(
        for context: DistributedStageHandleFactoryContext,
        functionName: String? = nil,
        vocabSize: Int? = nil
    ) async throws -> DistributedCoreAIStageHandle {
        let assetURL = try context.requireExistingAssetURL()
        guard let decodeAssetURL = try context.requireExistingDecodeAssetURL() else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "streamed-prefill stage '\(context.descriptor.id)' is missing decode_asset")
        }
        guard let decodeFunctionName = context.decodeFunctionName else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "streamed-prefill stage '\(context.descriptor.id)' is missing a decode function")
        }
        let resolvedFunctionName = functionName ?? context.mainFunctionName
        let resolvedVocabSize = vocabSize ?? context.vocabSize
        let decodeDescriptor = Self.decodeDescriptor(from: context.descriptor)
        let decodeContext = try await Self.loadExecutionContext(
            assetURL: decodeAssetURL,
            functionName: decodeFunctionName,
            descriptor: decodeDescriptor,
            boundaryTensor: context.boundaryTensor,
            vocabSize: resolvedVocabSize,
            specialization: Self.specializationOptions())
        return DistributedCoreAIStageHandle(
            descriptor: context.descriptor,
            assetURL: assetURL,
            functionName: resolvedFunctionName,
            ioContract: decodeContext.ioContract,
            prefillContext: nil,
            decodeContext: decodeContext,
            streamsPrefill: true,
            boundaryTensor: context.boundaryTensor,
            nextStageID: context.nextStage?.id,
            vocabSize: resolvedVocabSize)
    }

    public func loadPrefill() async throws {
        guard streamsPrefill else { return }
        guard prefillContext == nil else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "prefill stage \(descriptor.id) is already resident")
        }
        let loaded = try await Self.loadExecutionContext(
            assetURL: assetURL,
            functionName: functionName,
            descriptor: descriptor,
            boundaryTensor: boundaryTensor,
            vocabSize: vocabSize,
            specialization: Self.specializationOptions())
        try Self.validateCacheCompatibility(
            prefill: loaded.cacheIO,
            decode: decodeContext.cacheIO,
            stageID: descriptor.id)
        ioContract = loaded.ioContract
        prefillContext = loaded
    }

    public func unloadPrefill() {
        guard streamsPrefill else { return }
        prefillContext = nil
    }

    public func allocate(_ allocation: DistributedStageAllocation) async throws {
        try allocation.validate()
        guard requestStates[allocation.requestID] == nil else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "request_id \(allocation.requestID) is already allocated")
        }
        requestStates[allocation.requestID] = try makeRequestState(for: allocation)
    }

    public func forward(
        _ input: DistributedStageForwardInput
    ) async throws -> DistributedStageForwardOutput {
        guard var requestState = requestStates[input.requestID] else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "request_id \(input.requestID) is not allocated")
        }
        let positionCount = input.positionRange.count
        let activeContext = try activeExecutionContext(positionCount: positionCount)
        let executionIO = activeContext.executionIO
        try validateForwardInput(
            input,
            requestState: requestState,
            executionIO: executionIO)

        let positionIDs = try DistributedCoreAIStageNDArrayIO.makePositionIDs(
            positionIDs: input.positionIDs,
            descriptor: executionIO.positionIDs.descriptor)
        let activeFunction = activeContext.function
        tracePositionsIfRequested(input)

        let output: DistributedStageForwardOutput
        switch descriptor.role {
        case .embeddings:
            guard let inputIDsBinding = executionIO.inputIDs,
                let hiddenOutputBinding = executionIO.hiddenStatesOutput
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed embeddings stage IO bindings are incomplete")
            }
            guard let nextStageID else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "embeddings stage has no next stage")
            }
            let inputIDs = try DistributedCoreAIStageNDArrayIO.makeInputIDs(
                tokenIDs: input.tokenIDs,
                descriptor: inputIDsBinding.descriptor)
            var hiddenStates = try DistributedCoreAIStageNDArrayIO.makeHiddenStatesOutput(
                positionCount: positionCount,
                descriptor: hiddenOutputBinding.descriptor,
                boundaryTensor: boundaryTensor)
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&hiddenStates, for: hiddenOutputBinding.name)
            try await run(
                activeFunction,
                cacheIO: activeContext.cacheIO,
                inputs: [
                    inputIDsBinding.name: inputIDs,
                    executionIO.positionIDs.name: positionIDs,
                ],
                outputViews: consume outputViews,
                requestState: &requestState)
            try DistributedCoreAIStageNDArrayIO.applySoftTokenSpliceIfPresent(
                input.softTokenSplice,
                to: &hiddenStates,
                descriptor: hiddenOutputBinding.descriptor,
                positionRange: input.positionRange)
            try traceHiddenArrayIfRequested(
                hiddenStates,
                point: "hidden_output_ndarray",
                input: input)
            let hiddenPacket = try DistributedCoreAIStageNDArrayIO.makeHiddenStatePacket(
                from: hiddenStates,
                requestID: input.requestID,
                sourceStageID: descriptor.id,
                destinationStageID: nextStageID,
                positionRange: input.positionRange,
                stepIndex: input.stepIndex)
            try traceHiddenPacketIfRequested(
                hiddenPacket,
                point: "hidden_output_packet",
                input: input)
            output = DistributedStageForwardOutput(
                stageID: descriptor.id,
                stepIndex: input.stepIndex,
                hiddenState: hiddenPacket)

        case .transformerLayers:
            guard let hiddenInputBinding = executionIO.hiddenStatesInput,
                let hiddenOutputBinding = executionIO.hiddenStatesOutput
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed transformer_layers stage IO bindings are incomplete")
            }
            guard let hiddenStatePacket = input.hiddenState else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "transformer_layers stage requires hidden_state")
            }
            guard let nextStageID else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "transformer_layers stage has no next stage")
            }
            try traceHiddenPacketIfRequested(
                hiddenStatePacket,
                point: "hidden_input_packet",
                input: input)
            let hiddenInput = try DistributedCoreAIStageNDArrayIO.makeHiddenStates(
                packet: hiddenStatePacket,
                descriptor: hiddenInputBinding.descriptor)
            try traceHiddenArrayIfRequested(
                hiddenInput,
                point: "hidden_input_ndarray",
                input: input)
            var hiddenOutput = try DistributedCoreAIStageNDArrayIO.makeHiddenStatesOutput(
                positionCount: positionCount,
                descriptor: hiddenOutputBinding.descriptor,
                boundaryTensor: boundaryTensor)
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&hiddenOutput, for: hiddenOutputBinding.name)
            var inputs = [
                hiddenInputBinding.name: hiddenInput,
                executionIO.positionIDs.name: positionIDs,
            ]
            if let inputIDsBinding = executionIO.inputIDs {
                inputs[inputIDsBinding.name] = try DistributedCoreAIStageNDArrayIO.makeInputIDs(
                    tokenIDs: input.tokenIDs,
                    descriptor: inputIDsBinding.descriptor)
            }
            if positionCount > 1 {
                try addBlockIDInputsIfNeeded(
                    to: &inputs,
                    input: input,
                    positionCount: positionCount,
                    executionIO: executionIO)
            } else if input.blockIDsQ != nil || input.blockIDsKV != nil {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "decode stage must not receive block_ids")
            }
            if let rope = descriptor.rope,
                let ropeCosInput = executionIO.ropeCosInput,
                let ropeSinInput = executionIO.ropeSinInput
            {
                inputs[ropeCosInput.name] = try DistributedCoreAIStageNDArrayIO.makeRoPEInput(
                    positionIDs: input.positionIDs,
                    positionCount: positionCount,
                    descriptor: ropeCosInput.descriptor,
                    rope: rope,
                    component: .cos)
                inputs[ropeSinInput.name] = try DistributedCoreAIStageNDArrayIO.makeRoPEInput(
                    positionIDs: input.positionIDs,
                    positionCount: positionCount,
                    descriptor: ropeSinInput.descriptor,
                    rope: rope,
                    component: .sin)
            }
            try await run(
                activeFunction,
                cacheIO: activeContext.cacheIO,
                inputs: inputs,
                outputViews: consume outputViews,
                requestState: &requestState)
            try traceHiddenArrayIfRequested(
                hiddenOutput,
                point: "hidden_output_ndarray",
                input: input)
            let hiddenPacket = try DistributedCoreAIStageNDArrayIO.makeHiddenStatePacket(
                from: hiddenOutput,
                requestID: input.requestID,
                sourceStageID: descriptor.id,
                destinationStageID: nextStageID,
                positionRange: input.positionRange,
                stepIndex: input.stepIndex)
            try traceHiddenPacketIfRequested(
                hiddenPacket,
                point: "hidden_output_packet",
                input: input)
            output = DistributedStageForwardOutput(
                stageID: descriptor.id,
                stepIndex: input.stepIndex,
                hiddenState: hiddenPacket)

        case .finalNormHead:
            guard let hiddenInputBinding = executionIO.hiddenStatesInput,
                let logitsOutputBinding = executionIO.logitsOutput
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed final_norm_head stage IO bindings are incomplete")
            }
            guard let hiddenStatePacket = input.hiddenState else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "final_norm_head stage requires hidden_state")
            }
            guard let vocabSize else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "final_norm_head stage requires vocab_size")
            }
            try traceHiddenPacketIfRequested(
                hiddenStatePacket,
                point: "hidden_input_packet",
                input: input)
            let hiddenInput = try DistributedCoreAIStageNDArrayIO.makeHiddenStates(
                packet: hiddenStatePacket,
                descriptor: hiddenInputBinding.descriptor)
            try traceHiddenArrayIfRequested(
                hiddenInput,
                point: "hidden_input_ndarray",
                input: input)
            var logits = try DistributedCoreAIStageNDArrayIO.makeLogitsOutput(
                positionCount: positionCount,
                vocabSize: vocabSize,
                descriptor: logitsOutputBinding.descriptor)
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&logits, for: logitsOutputBinding.name)
            var inputs = [
                hiddenInputBinding.name: hiddenInput,
                executionIO.positionIDs.name: positionIDs,
            ]
            if let inputIDsBinding = executionIO.inputIDs {
                inputs[inputIDsBinding.name] = try DistributedCoreAIStageNDArrayIO.makeInputIDs(
                    tokenIDs: input.tokenIDs,
                    descriptor: inputIDsBinding.descriptor)
            }
            try await run(
                activeFunction,
                cacheIO: activeContext.cacheIO,
                inputs: inputs,
                outputViews: consume outputViews,
                requestState: &requestState)
            let logitsRow = try DistributedCoreAIStageNDArrayIO.readLastLogitsRow(
                logits,
                vocabSize: vocabSize)
            traceLogitSummaryIfRequested(
                logitsRow,
                point: "logits_last_row",
                input: input)
            Self.traceLogitsIfRequested(
                logitsRow,
                label: "staged stage=\(descriptor.id) step=\(input.stepIndex)")
            let tokenID = Sampler.argmax(logitsRow)
            guard tokenID <= Int(Int32.max) else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "token id \(tokenID) exceeds Int32.max")
            }
            output = DistributedStageForwardOutput(
                stageID: descriptor.id,
                stepIndex: input.stepIndex,
                tokenID: Int32(tokenID),
                topLogits: Self.captureTopLogitsIfRequested(logitsRow))
        }

        if descriptor.role == .finalNormHead {
            requestState.processedTokenCount = input.positionRange.upperBound
        } else {
            requestState.processedTokenCount += positionCount
        }
        requestStates[input.requestID] = requestState
        return output
    }

    public func reset(requestID: String) async throws {
        guard var requestState = requestStates[requestID] else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "request_id \(requestID) is not allocated")
        }
        requestState.processedTokenCount = 0
        requestStates[requestID] = requestState
    }

    public func free(requestID: String) async {
        requestStates.removeValue(forKey: requestID)
    }

    private func activeExecutionContext(
        positionCount: Int
    ) throws -> DistributedCoreAIStageExecutionContext {
        if positionCount == 1 { return decodeContext }
        guard let prefillContext else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "prefill stage \(descriptor.id) is not resident")
        }
        return prefillContext
    }

    private static var stageSummaryTraceEnabled: Bool {
        guard let value = ProcessInfo.processInfo.environment["CAIX_DISTRIBUTED_STAGE_TRACE"] else {
            return false
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !trimmed.isEmpty && !["0", "false", "no", "off"].contains(trimmed)
    }

    private func tracePositionsIfRequested(_ input: DistributedStageForwardInput) {
        guard Self.stageSummaryTraceEnabled else { return }
        let first = input.positionIDs.first.map(String.init) ?? "nil"
        let last = input.positionIDs.last.map(String.init) ?? "nil"
        traceStageLine(
            point: "positions",
            input: input,
            fields: "position_count=\(input.positionIDs.count) first=\(first) last=\(last)")
    }

    private func traceHiddenArrayIfRequested(
        _ array: NDArray,
        point: String,
        input: DistributedStageForwardInput
    ) throws {
        guard Self.stageSummaryTraceEnabled else { return }
        let summary = try DistributedCoreAIStageNDArrayIO.hiddenStateSummary(array)
        traceStageLine(point: point, input: input, fields: summary.fields)
    }

    private func traceHiddenPacketIfRequested(
        _ packet: DistributedHiddenStatePacket,
        point: String,
        input: DistributedStageForwardInput
    ) throws {
        guard Self.stageSummaryTraceEnabled else { return }
        let summary = try DistributedCoreAIStageTensorSummary(
            scalarType: packet.metadata.scalarType.rawValue,
            shape: packet.metadata.shape,
            strides: nil,
            values: packet.floatValuesAsFloat32().map(Double.init))
        traceStageLine(
            point: point,
            input: input,
            fields: summary.fields + " payload_bytes=\(packet.payload.count)")
    }

    private func traceLogitSummaryIfRequested(
        _ logits: [Float],
        point: String,
        input: DistributedStageForwardInput
    ) {
        guard Self.stageSummaryTraceEnabled else { return }
        let summary = DistributedCoreAIStageTensorSummary(
            scalarType: "float32",
            shape: [logits.count],
            strides: nil,
            values: logits.map(Double.init))
        traceStageLine(point: point, input: input, fields: summary.fields)
    }

    private func traceStageLine(
        point: String,
        input: DistributedStageForwardInput,
        fields: String
    ) {
        let range = "\(input.positionRange.lowerBound)..<\(input.positionRange.upperBound)"
        let line = "[caix-stage-summary] stage=\(descriptor.id) "
            + "role=\(descriptor.role.rawValue) point=\(point) step=\(input.stepIndex) "
            + "positions=\(range) \(fields)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private static func traceLogitsIfRequested(_ logits: [Float], label: String) {
        guard let rawLimit = ProcessInfo.processInfo.environment["CAIX_LOGIT_TRACE_TOPK"],
            let limit = Int(rawLimit),
            limit > 0
        else {
            return
        }
        let top = Sampler.topK(logits, count: min(limit, logits.count))
        let margin = top.count > 1 ? top[0].logit - top[1].logit : .nan
        let fields = top.map { "\($0.index):\(String(format: "%.6g", Double($0.logit)))" }
            .joined(separator: ",")
        let line = "[caix-logits] \(label) top=\(fields) "
            + "margin=\(String(format: "%.6g", Double(margin)))\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private static func captureTopLogitsIfRequested(_ logits: [Float]) -> [DistributedLogitScore] {
        guard let rawLimit = ProcessInfo.processInfo.environment["CAIX_CAPTURE_STAGE_TOPK"],
            let limit = Int(rawLimit),
            limit > 0
        else {
            return []
        }
        return Sampler.topK(logits, count: min(limit, logits.count)).compactMap { item in
            guard item.index <= Int(Int32.max) else { return nil }
            return DistributedLogitScore(tokenID: Int32(item.index), logit: item.logit)
        }
    }

    private func validateForwardInput(
        _ input: DistributedStageForwardInput,
        requestState: DistributedCoreAIStageRequestState,
        executionIO: DistributedCoreAIStageExecutionIO
    ) throws {
        guard input.stepIndex >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "step_index must be non-negative")
        }
        guard input.positionRange.isValid else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_range is invalid")
        }
        if descriptor.role == .finalNormHead {
            guard input.positionRange.lowerBound >= requestState.processedTokenCount else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "position_range lower_bound \(input.positionRange.lowerBound) is behind processed_token_count \(requestState.processedTokenCount)")
            }
        } else {
            guard input.positionRange.lowerBound == requestState.processedTokenCount else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "position_range lower_bound \(input.positionRange.lowerBound) does not match processed_token_count \(requestState.processedTokenCount)")
            }
        }
        guard input.positionRange.upperBound <= requestState.kvCapacity else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_range upper_bound \(input.positionRange.upperBound) exceeds kv_capacity \(requestState.kvCapacity)")
        }
        guard input.positionIDs.count >= input.positionRange.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_ids count must cover position_range")
        }
        try validateBlockIDsIfPresent(input)

        switch descriptor.role {
        case .embeddings:
            guard input.tokenIDs.count == input.positionRange.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "token_ids count must match position_range")
            }
            guard input.hiddenState == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "embeddings stage must not receive hidden_state")
            }
            guard input.blockIDsQ == nil && input.blockIDsKV == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "embeddings stage must not receive block_ids")
            }
            try validateSoftTokenSpliceIfPresent(input)
        case .transformerLayers, .finalNormHead:
            guard input.softTokenSplice == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(descriptor.role.rawValue) stage must not receive soft_token_splice")
            }
            if executionIO.inputIDs != nil {
                guard input.tokenIDs.count == input.positionRange.count else {
                    throw DistributedStageExecutionError.invalidForwardInput(
                        "token_ids count must match position_range")
                }
            } else {
                guard input.tokenIDs.isEmpty else {
                    throw DistributedStageExecutionError.invalidForwardInput(
                        "\(descriptor.role.rawValue) stage must not receive token_ids")
                }
            }
            guard let hiddenState = input.hiddenState else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(descriptor.role.rawValue) stage requires hidden_state")
            }
            guard hiddenState.metadata.destinationStageID == descriptor.id else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state destination_stage_id does not match stage")
            }
            guard hiddenState.metadata.requestID == input.requestID else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state request_id does not match request")
            }
            guard hiddenState.metadata.stepIndex == input.stepIndex else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state step_index does not match request")
            }
            guard hiddenState.metadata.positionRange == input.positionRange else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "hidden_state position_range does not match request")
            }
            if descriptor.role == .finalNormHead {
                guard input.blockIDsQ == nil && input.blockIDsKV == nil else {
                    throw DistributedStageExecutionError.invalidForwardInput(
                        "final_norm_head stage must not receive block_ids")
                }
            }
        }
    }

    private func validateBlockIDsIfPresent(_ input: DistributedStageForwardInput) throws {
        guard input.blockIDsQ != nil || input.blockIDsKV != nil else { return }
        guard let blockIDsQ = input.blockIDsQ,
            let blockIDsKV = input.blockIDsKV
        else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "block_ids_q and block_ids_kv must be supplied together")
        }
        guard blockIDsQ.count == input.positionRange.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "block_ids_q count must match position_range")
        }
        guard blockIDsKV.count == input.positionIDs.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "block_ids_kv count must match position_ids")
        }
    }

    private func validateSoftTokenSpliceIfPresent(
        _ input: DistributedStageForwardInput
    ) throws {
        guard let splice = input.softTokenSplice else { return }
        guard input.positionRange.count > 1 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "decode stage must not receive soft_token_splice")
        }
        guard splice.positionStart >= input.positionRange.lowerBound
            && splice.positionEnd <= input.positionRange.upperBound
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

    private func addBlockIDInputsIfNeeded(
        to inputs: inout [String: NDArray],
        input: DistributedStageForwardInput,
        positionCount: Int,
        executionIO: DistributedCoreAIStageExecutionIO
    ) throws {
        let hasAnyBinding = executionIO.blockIDsQInput != nil || executionIO.blockIDsKVInput != nil
        guard hasAnyBinding else {
            guard input.blockIDsQ == nil && input.blockIDsKV == nil else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(descriptor.role.rawValue) stage has no block_ids inputs")
            }
            return
        }
        guard let blockIDsQInput = executionIO.blockIDsQInput,
            let blockIDsKVInput = executionIO.blockIDsKVInput
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed transformer_layers block_ids_q/block_ids_kv bindings are incomplete")
        }

        let blockIDsQ = input.blockIDsQ ?? Array(repeating: Int32(-1), count: positionCount)
        let blockIDsKV = input.blockIDsKV ?? Array(repeating: Int32(-1), count: input.positionIDs.count)
        inputs[blockIDsQInput.name] = try DistributedCoreAIStageNDArrayIO.makeBlockIDs(
            blockIDsQ,
            descriptor: blockIDsQInput.descriptor,
            tensorName: blockIDsQInput.name)
        inputs[blockIDsKVInput.name] = try DistributedCoreAIStageNDArrayIO.makeBlockIDs(
            blockIDsKV,
            descriptor: blockIDsKVInput.descriptor,
            tensorName: blockIDsKVInput.name)
    }

    private func run(
        _ function: InferenceFunction,
        cacheIO: DistributedCoreAIStageCacheIO,
        inputs: [String: NDArray],
        outputViews: consuming InferenceFunction.MutableViews,
        requestState: inout DistributedCoreAIStageRequestState
    ) async throws {
        switch cacheIO.contract {
        case .none:
            _ = try await function.run(
                inputs: inputs,
                outputViews: consume outputViews)
        case .stateful:
            guard !cacheIO.groups.isEmpty else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage stateful KV cache is missing")
            }
            if cacheIO.groups.count == 1 {
                let group = cacheIO.groups[0]
                var cacheGroup = try requestState.cacheGroup(named: group.groupName)
                var states = InferenceFunction.MutableViews()
                states.insert(&cacheGroup.keyCache, for: group.keyCacheName)
                states.insert(&cacheGroup.valueCache, for: group.valueCacheName)
                _ = try await function.run(
                    inputs: inputs,
                    states: consume states,
                    outputViews: consume outputViews)
                requestState.cacheGroups[group.groupName] = cacheGroup
            } else if cacheIO.groups.count == 2 {
                let firstGroupIO = cacheIO.groups[0]
                let secondGroupIO = cacheIO.groups[1]
                var firstGroup = try requestState.cacheGroup(named: firstGroupIO.groupName)
                var secondGroup = try requestState.cacheGroup(named: secondGroupIO.groupName)
                var states = InferenceFunction.MutableViews()
                states.insert(&firstGroup.keyCache, for: firstGroupIO.keyCacheName)
                states.insert(&firstGroup.valueCache, for: firstGroupIO.valueCacheName)
                states.insert(&secondGroup.keyCache, for: secondGroupIO.keyCacheName)
                states.insert(&secondGroup.valueCache, for: secondGroupIO.valueCacheName)
                _ = try await function.run(
                    inputs: inputs,
                    states: consume states,
                    outputViews: consume outputViews)
                requestState.cacheGroups[firstGroupIO.groupName] = firstGroup
                requestState.cacheGroups[secondGroupIO.groupName] = secondGroup
            } else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage stateful KV cache supports at most two cache groups")
            }
        case .explicitOutputs:
            var explicitInputs = inputs
            guard !cacheIO.groups.isEmpty else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage explicit KV cache is missing")
            }
            if cacheIO.groups.count == 1 {
                let group = cacheIO.groups[0]
                let cacheGroup = try requestState.cacheGroup(named: group.groupName)
                explicitInputs[group.keyCacheName] = cacheGroup.keyCache
                explicitInputs[group.valueCacheName] = cacheGroup.valueCache
                var nextKeyCache = try Self.makeExplicitKVCacheOutput(
                    descriptor: group.keyCacheOutputDescriptor,
                    currentCache: cacheGroup.keyCache,
                    tensorName: group.keyCacheName)
                var nextValueCache = try Self.makeExplicitKVCacheOutput(
                    descriptor: group.valueCacheOutputDescriptor,
                    currentCache: cacheGroup.valueCache,
                    tensorName: group.valueCacheName)
                outputViews.insert(&nextKeyCache, for: group.keyCacheName)
                outputViews.insert(&nextValueCache, for: group.valueCacheName)
                _ = try await function.run(
                    inputs: explicitInputs,
                    outputViews: consume outputViews)
                requestState.cacheGroups[group.groupName] =
                    DistributedCoreAIStageRequestCacheGroup(
                        keyCache: nextKeyCache,
                        valueCache: nextValueCache)
            } else if cacheIO.groups.count == 2 {
                let firstGroupIO = cacheIO.groups[0]
                let secondGroupIO = cacheIO.groups[1]
                let firstGroup = try requestState.cacheGroup(named: firstGroupIO.groupName)
                let secondGroup = try requestState.cacheGroup(named: secondGroupIO.groupName)
                explicitInputs[firstGroupIO.keyCacheName] = firstGroup.keyCache
                explicitInputs[firstGroupIO.valueCacheName] = firstGroup.valueCache
                explicitInputs[secondGroupIO.keyCacheName] = secondGroup.keyCache
                explicitInputs[secondGroupIO.valueCacheName] = secondGroup.valueCache
                var nextFirstKeyCache = try Self.makeExplicitKVCacheOutput(
                    descriptor: firstGroupIO.keyCacheOutputDescriptor,
                    currentCache: firstGroup.keyCache,
                    tensorName: firstGroupIO.keyCacheName)
                var nextFirstValueCache = try Self.makeExplicitKVCacheOutput(
                    descriptor: firstGroupIO.valueCacheOutputDescriptor,
                    currentCache: firstGroup.valueCache,
                    tensorName: firstGroupIO.valueCacheName)
                var nextSecondKeyCache = try Self.makeExplicitKVCacheOutput(
                    descriptor: secondGroupIO.keyCacheOutputDescriptor,
                    currentCache: secondGroup.keyCache,
                    tensorName: secondGroupIO.keyCacheName)
                var nextSecondValueCache = try Self.makeExplicitKVCacheOutput(
                    descriptor: secondGroupIO.valueCacheOutputDescriptor,
                    currentCache: secondGroup.valueCache,
                    tensorName: secondGroupIO.valueCacheName)
                outputViews.insert(&nextFirstKeyCache, for: firstGroupIO.keyCacheName)
                outputViews.insert(&nextFirstValueCache, for: firstGroupIO.valueCacheName)
                outputViews.insert(&nextSecondKeyCache, for: secondGroupIO.keyCacheName)
                outputViews.insert(&nextSecondValueCache, for: secondGroupIO.valueCacheName)
                _ = try await function.run(
                    inputs: explicitInputs,
                    outputViews: consume outputViews)
                requestState.cacheGroups[firstGroupIO.groupName] =
                    DistributedCoreAIStageRequestCacheGroup(
                        keyCache: nextFirstKeyCache,
                        valueCache: nextFirstValueCache)
                requestState.cacheGroups[secondGroupIO.groupName] =
                    DistributedCoreAIStageRequestCacheGroup(
                        keyCache: nextSecondKeyCache,
                        valueCache: nextSecondValueCache)
            } else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage explicit KV cache supports at most two cache groups")
            }
        }
    }

    private static func loadFunction(
        named name: String,
        from model: AIModel,
        assetURL: URL
    ) throws -> InferenceFunction {
        guard let function = try model.loadFunction(named: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "could not load distributed stage function '\(name)' from \(assetURL.lastPathComponent)")
        }
        return function
    }

    private static func specializationOptions() -> SpecializationOptions {
        var specialization = SpecializationOptions(
            preferredComputeUnitKind: LLMEngine.preferredComputeUnit())
        specialization.expectFrequentReshapes = true
        return specialization
    }

    private static func loadExecutionContext(
        assetURL: URL,
        functionName: String,
        descriptor: DistributedStageDescriptor,
        boundaryTensor: DistributedBoundaryTensorSpec?,
        vocabSize: Int?,
        specialization: SpecializationOptions
    ) async throws -> DistributedCoreAIStageExecutionContext {
        let model = try await AIModel.specialize(
            contentsOf: assetURL,
            options: specialization,
            cache: .default,
            cachePolicy: .persistent)
        return try makeExecutionContext(
            model: model,
            assetURL: assetURL,
            functionName: functionName,
            descriptor: descriptor,
            boundaryTensor: boundaryTensor,
            vocabSize: vocabSize)
    }

    private static func makeExecutionContext(
        model: AIModel,
        assetURL: URL,
        functionName: String,
        descriptor: DistributedStageDescriptor,
        boundaryTensor: DistributedBoundaryTensorSpec?,
        vocabSize: Int?
    ) throws -> DistributedCoreAIStageExecutionContext {
        guard let functionDescriptor = model.functionDescriptor(for: functionName) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage function '\(functionName)' not found in \(assetURL.lastPathComponent); have \(model.functionNames)")
        }
        let ioContract = try DistributedStageIOContract.extractedFromCoreAI(
            functionName: functionName,
            descriptor: functionDescriptor)
        try ioContract.validate(
            for: descriptor,
            boundaryTensor: boundaryTensor,
            vocabSize: vocabSize)
        let cacheIO = try DistributedCoreAIStageCacheIO.extracted(from: functionDescriptor)
        let executionIO = try DistributedCoreAIStageExecutionIO.bound(
            for: descriptor,
            descriptor: functionDescriptor,
            vocabSize: vocabSize)
        let function = try loadFunction(
            named: functionName,
            from: model,
            assetURL: assetURL)
        return DistributedCoreAIStageExecutionContext(
            model: model,
            assetURL: assetURL,
            function: function,
            functionName: functionName,
            functionDescriptor: functionDescriptor,
            ioContract: ioContract,
            executionIO: executionIO,
            cacheIO: cacheIO)
    }

    private static func loadDecodeExecutionContext(
        for context: DistributedStageHandleFactoryContext,
        mainContext: DistributedCoreAIStageExecutionContext,
        specialization: SpecializationOptions,
        vocabSize: Int?
    ) async throws -> DistributedCoreAIStageExecutionContext {
        guard let decodeFunctionName = context.decodeFunctionName,
              decodeFunctionName != mainContext.functionName
        else {
            return mainContext
        }
        let descriptor = decodeDescriptor(from: context.descriptor)
        if let decodeAssetURL = try context.requireExistingDecodeAssetURL() {
            return try await loadExecutionContext(
                assetURL: decodeAssetURL,
                functionName: decodeFunctionName,
                descriptor: descriptor,
                boundaryTensor: context.boundaryTensor,
                vocabSize: vocabSize,
                specialization: specialization)
        }
        return try makeExecutionContext(
            model: mainContext.model,
            assetURL: mainContext.assetURL,
            functionName: decodeFunctionName,
            descriptor: descriptor,
            boundaryTensor: context.boundaryTensor,
            vocabSize: vocabSize)
    }

    private static func decodeDescriptor(
        from descriptor: DistributedStageDescriptor
    ) -> DistributedStageDescriptor {
        DistributedStageDescriptor(
            id: descriptor.id,
            role: descriptor.role,
            layerRange: descriptor.layerRange,
            assetName: descriptor.assetName,
            decodeAssetName: descriptor.decodeAssetName,
            functionMap: descriptor.functionMap,
            vocabSize: descriptor.vocabSize,
            prefillExtraInputs: [],
            workerID: descriptor.workerID,
            rope: descriptor.rope)
    }

    private static func validateCacheCompatibility(
        prefill: DistributedCoreAIStageCacheIO,
        decode: DistributedCoreAIStageCacheIO,
        stageID: String
    ) throws {
        let contractsMatch: Bool
        switch (prefill.contract, decode.contract) {
        case (.none, .none), (.stateful, .stateful), (.explicitOutputs, .explicitOutputs):
            contractsMatch = true
        default:
            contractsMatch = false
        }
        guard contractsMatch, prefill.groups.count == decode.groups.count else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "streamed-prefill stage '\(stageID)' prefill/decode KV contracts differ")
        }
        for (prefillGroup, decodeGroup) in zip(prefill.groups, decode.groups) {
            guard prefillGroup.groupName == decodeGroup.groupName,
                  prefillGroup.keyCacheName == decodeGroup.keyCacheName,
                  prefillGroup.valueCacheName == decodeGroup.valueCacheName,
                  cacheDescriptor(prefillGroup.keyCacheDescriptor,
                                  matches: decodeGroup.keyCacheDescriptor),
                  cacheDescriptor(prefillGroup.valueCacheDescriptor,
                                  matches: decodeGroup.valueCacheDescriptor),
                  cacheDescriptor(prefillGroup.keyCacheOutputDescriptor,
                                  matches: decodeGroup.keyCacheOutputDescriptor),
                  cacheDescriptor(prefillGroup.valueCacheOutputDescriptor,
                                  matches: decodeGroup.valueCacheOutputDescriptor)
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "streamed-prefill stage '\(stageID)' prefill/decode KV group '\(prefillGroup.groupName)' differs")
            }
        }
    }

    private static func cacheDescriptor(
        _ lhs: NDArrayDescriptor,
        matches rhs: NDArrayDescriptor
    ) -> Bool {
        lhs.scalarType == rhs.scalarType && lhs.shape == rhs.shape
    }

    private func makeRequestState(
        for allocation: DistributedStageAllocation
    ) throws -> DistributedCoreAIStageRequestState {
        let cacheIO = decodeContext.cacheIO
        switch cacheIO.contract {
        case .none:
            return DistributedCoreAIStageRequestState(
                requestID: allocation.requestID,
                kvCapacity: allocation.kvCapacity)
        case .stateful, .explicitOutputs:
            guard !cacheIO.groups.isEmpty else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache descriptors are missing")
            }
            var cacheGroups: [String: DistributedCoreAIStageRequestCacheGroup] = [:]
            for group in cacheIO.groups {
                let capacity = allocation.capacity(forCacheGroup: group.groupName)
                cacheGroups[group.groupName] = DistributedCoreAIStageRequestCacheGroup(
                    keyCache: try Self.makeKVCache(
                        descriptor: group.keyCacheDescriptor,
                        capacity: capacity),
                    valueCache: try Self.makeKVCache(
                        descriptor: group.valueCacheDescriptor,
                        capacity: capacity))
            }
            return DistributedCoreAIStageRequestState(
                requestID: allocation.requestID,
                kvCapacity: allocation.kvCapacity,
                cacheGroups: cacheGroups)
        }
    }

    private static func makeKVCache(
        descriptor: NDArrayDescriptor,
        capacity: Int
    ) throws -> NDArray {
        let shape = try DistributedCoreAIStageKVCacheShape.resolved(
            descriptor.shape,
            capacity: capacity)
        var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
        zeroState(&array, scalarType: descriptor.scalarType)
        return array
    }

    private static func makeExplicitKVCacheOutput(
        descriptor: NDArrayDescriptor,
        currentCache: NDArray,
        tensorName: String
    ) throws -> NDArray {
        guard descriptor.shape.count == currentCache.shape.count else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage KV cache output '\(tensorName)' rank \(descriptor.shape.count) does not match current cache rank \(currentCache.shape.count)")
        }
        let shape = descriptor.shape.enumerated().map { index, dimension in
            dimension < 0 ? currentCache.shape[index] : dimension
        }
        guard shape.allSatisfy({ $0 > 0 }) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage KV cache output '\(tensorName)' resolves to invalid shape \(shape)")
        }
        return NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
    }

    private static func zeroState(_ array: inout NDArray, scalarType: NDArray.ScalarType) {
        let count = array.shape.reduce(1, *)
        switch scalarType {
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for index in 0..<count { ptr[index] = 0 }
            }
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for index in 0..<count { ptr[index] = 0 }
            }
        default:
            break
        }
    }
}

fileprivate struct DistributedCoreAIStageTensorSummary {
    let scalarType: String
    let shape: [Int]
    let strides: [Int]?
    let count: Int
    let finiteCount: Int
    let nanCount: Int
    let infCount: Int
    let minValue: Double
    let maxValue: Double
    let meanValue: Double
    let sumValue: Double
    let sumAbsValue: Double
    let l2Value: Double

    init(
        scalarType: String,
        shape: [Int],
        strides: [Int]?,
        values: [Double]
    ) {
        self.scalarType = scalarType
        self.shape = shape
        self.strides = strides
        self.count = values.count

        var finiteCount = 0
        var nanCount = 0
        var infCount = 0
        var minValue = Double.infinity
        var maxValue = -Double.infinity
        var sumValue = 0.0
        var sumAbsValue = 0.0
        var sumSquares = 0.0

        for value in values {
            if value.isNaN {
                nanCount += 1
            } else if value.isInfinite {
                infCount += 1
            } else {
                finiteCount += 1
                minValue = Swift.min(minValue, value)
                maxValue = Swift.max(maxValue, value)
                sumValue += value
                sumAbsValue += Swift.abs(value)
                sumSquares += value * value
            }
        }

        self.finiteCount = finiteCount
        self.nanCount = nanCount
        self.infCount = infCount
        self.minValue = finiteCount > 0 ? minValue : .nan
        self.maxValue = finiteCount > 0 ? maxValue : .nan
        self.meanValue = finiteCount > 0 ? sumValue / Double(finiteCount) : .nan
        self.sumValue = sumValue
        self.sumAbsValue = sumAbsValue
        self.l2Value = sumSquares.squareRoot()
    }

    var fields: String {
        let strideField = strides.map { " strides=\(Self.formatInts($0))" } ?? ""
        return "scalar=\(scalarType) shape=\(Self.formatInts(shape))\(strideField) "
            + "count=\(count) finite=\(finiteCount) nan=\(nanCount) inf=\(infCount) "
            + "min=\(Self.format(minValue)) max=\(Self.format(maxValue)) "
            + "mean=\(Self.format(meanValue)) sum=\(Self.format(sumValue)) "
            + "sum_abs=\(Self.format(sumAbsValue)) l2=\(Self.format(l2Value))"
    }

    private static func formatInts(_ values: [Int]) -> String {
        "[" + values.map(String.init).joined(separator: ",") + "]"
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.8g", value)
    }
}

enum DistributedCoreAIStageNDArrayIO {
    enum RoPEComponent {
        case cos
        case sin
    }

    static func makeInputIDs(
        tokenIDs: [Int32],
        descriptor: NDArrayDescriptor
    ) throws -> NDArray {
        try requireScalarType(.int32, descriptor: descriptor, tensorName: "input_ids")
        var array = NDArray(
            descriptor: descriptor.resolvingDynamicDimensions(
                try DistributedCoreAIStageNDArrayShape.resolvedMatrix(
                    descriptor.shape,
                    columns: tokenIDs.count,
                    tensorName: "input_ids")))
        fillInt32(&array, tokenIDs)
        return array
    }

    static func makePositionIDs(
        positionIDs: [Int32],
        descriptor: NDArrayDescriptor
    ) throws -> NDArray {
        try requireScalarType(.int32, descriptor: descriptor, tensorName: "position_ids")
        var array = NDArray(
            descriptor: descriptor.resolvingDynamicDimensions(
                try DistributedCoreAIStageNDArrayShape.resolvedMatrix(
                    descriptor.shape,
                    columns: positionIDs.count,
                    tensorName: "position_ids")))
        fillInt32(&array, positionIDs)
        return array
    }

    static func makeBlockIDs(
        _ blockIDs: [Int32],
        descriptor: NDArrayDescriptor,
        tensorName: String
    ) throws -> NDArray {
        let shape = try DistributedCoreAIStageNDArrayShape.resolvedMatrix(
            descriptor.shape,
            columns: blockIDs.count,
            tensorName: tensorName)
        var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
        switch descriptor.scalarType {
        case .int32:
            fillInt32(&array, blockIDs)
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(descriptor.scalarType) is not supported for block ids")
        }
        return array
    }

    static func makeRoPEInput(
        positionIDs: [Int32],
        positionCount: Int,
        descriptor: NDArrayDescriptor,
        rope: DistributedStageRoPEInputSpec,
        component: RoPEComponent
    ) throws -> NDArray {
        try requireFloatingScalarType(descriptor, tensorName: "rope")
        guard positionCount > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "rope position count must be positive")
        }
        guard positionIDs.count >= positionCount else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_ids count must cover rope position count")
        }
        let shape = try DistributedCoreAIStageNDArrayShape.resolvedRoPEInput(
            descriptor.shape,
            positionCount: positionCount,
            headDim: rope.headDim,
            tensorName: "rope")
        let activePositions = Array(positionIDs.suffix(positionCount))
        var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
        switch descriptor.scalarType {
        case .float16:
            fillRoPE(&array, activePositions, rope: rope, component: component, as: Float16.self)
        case .float32:
            fillRoPE(&array, activePositions, rope: rope, component: component, as: Float.self)
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "rope scalar type \(descriptor.scalarType) is not supported for distributed stage IO")
        }
        return array
    }

    fileprivate static func makeHiddenStates(
        packet: DistributedHiddenStatePacket,
        descriptor: NDArrayDescriptor
    ) throws -> NDArray {
        let shape = try DistributedCoreAIStageNDArrayShape.resolvedHiddenStates(
            descriptor.shape,
            packetShape: packet.metadata.shape)
        switch (descriptor.scalarType, packet.metadata.scalarType) {
        case (.float16, .float16):
            var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: try packet.float16Values())
            return array
        case (.bfloat16, .float16):
            var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
            try fillBFloat16(
                &array,
                bitPatterns: try packet.float16Values().map { bfloat16BitPattern(from: Float($0)) })
            return array
        case (.bfloat16, .float32):
            var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
            try fillBFloat16(
                &array,
                bitPatterns: try packet.float32Values().map(bfloat16BitPattern))
            return array
        case (.float32, .float32):
            var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: try packet.float32Values())
            return array
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "hidden_states scalar type \(descriptor.scalarType) does not match packet \(packet.metadata.scalarType.rawValue)")
        }
    }

    static func makeHiddenStatesOutput(
        positionCount: Int,
        descriptor: NDArrayDescriptor,
        boundaryTensor: DistributedBoundaryTensorSpec?
    ) throws -> NDArray {
        try requireFloatingScalarType(descriptor, tensorName: "hidden_states")
        let shape = try DistributedCoreAIStageNDArrayShape.resolvedHiddenStatesOutput(
            descriptor.shape,
            positionCount: positionCount,
            boundaryTensor: boundaryTensor)
        if let boundaryTensor {
            try requireBoundaryScalarType(
                boundaryTensor.scalarType,
                descriptor: descriptor,
                tensorName: "hidden_states")
        }
        return NDArray(
            descriptor: descriptor.resolvingDynamicDimensions(shape))
    }

    static func applySoftTokenSpliceIfPresent(
        _ splice: DistributedSoftTokenSplice?,
        to output: inout NDArray,
        descriptor: NDArrayDescriptor,
        positionRange: DistributedSequenceRange
    ) throws {
        guard let splice else { return }
        guard splice.positionStart >= positionRange.lowerBound
            && splice.positionEnd <= positionRange.upperBound
        else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "soft_token_splice position range must be inside position_range")
        }
        switch descriptor.scalarType {
        case .float16:
            let values: [Float16]
            switch splice.scalarType {
            case .float16:
                values = try splice.float16Values()
            case .float32:
                values = try splice.float32Values().map(Float16.init)
            }
            try applySoftTokenRows(
                values,
                splice: splice,
                to: &output,
                positionRange: positionRange,
                as: Float16.self)
        case .float32:
            let values: [Float]
            switch splice.scalarType {
            case .float16:
                values = try splice.float16Values().map(Float.init)
            case .float32:
                values = try splice.float32Values()
            }
            try applySoftTokenRows(
                values,
                splice: splice,
                to: &output,
                positionRange: positionRange,
                as: Float.self)
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "hidden_states scalar type \(descriptor.scalarType) is not supported for soft_token_splice")
        }
    }

    static func makeLogitsOutput(
        positionCount: Int,
        vocabSize: Int,
        descriptor: NDArrayDescriptor
    ) throws -> NDArray {
        try requireFloatingScalarType(descriptor, tensorName: "logits")
        return NDArray(
            descriptor: descriptor.resolvingDynamicDimensions(
                try DistributedCoreAIStageNDArrayShape.resolvedLogitsOutput(
                    descriptor.shape,
                    positionCount: positionCount,
                    vocabSize: vocabSize)))
    }

    static func makeHiddenStatePacket(
        from output: NDArray,
        requestID: String,
        sourceStageID: String,
        destinationStageID: String,
        positionRange: DistributedSequenceRange,
        stepIndex: Int
    ) throws -> DistributedHiddenStatePacket {
        switch output.scalarType {
        case .float16:
            let readback = try readRank3(output, as: Float16.self, tensorName: "hidden_states")
            let metadata = DistributedHiddenStatePacketMetadata(
                requestID: requestID,
                sourceStageID: sourceStageID,
                destinationStageID: destinationStageID,
                positionRange: positionRange,
                shape: readback.shape,
                scalarType: .float16,
                byteCount: readback.values.count * DistributedTensorScalarType.float16.byteWidth,
                stepIndex: stepIndex)
            return try DistributedHiddenStatePacket(metadata: metadata, float16Values: readback.values)
        case .bfloat16:
            let readback = try readRank3BFloat16AsFloat16(output, tensorName: "hidden_states")
            let metadata = DistributedHiddenStatePacketMetadata(
                requestID: requestID,
                sourceStageID: sourceStageID,
                destinationStageID: destinationStageID,
                positionRange: positionRange,
                shape: readback.shape,
                scalarType: .float16,
                byteCount: readback.values.count * DistributedTensorScalarType.float16.byteWidth,
                stepIndex: stepIndex)
            return try DistributedHiddenStatePacket(metadata: metadata, float16Values: readback.values)
        case .float32:
            let readback = try readRank3(output, as: Float.self, tensorName: "hidden_states")
            let metadata = DistributedHiddenStatePacketMetadata(
                requestID: requestID,
                sourceStageID: sourceStageID,
                destinationStageID: destinationStageID,
                positionRange: positionRange,
                shape: readback.shape,
                scalarType: .float32,
                byteCount: readback.values.count * DistributedTensorScalarType.float32.byteWidth,
                stepIndex: stepIndex)
            return try DistributedHiddenStatePacket(metadata: metadata, float32Values: readback.values)
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "hidden_states scalar type \(output.scalarType) is not supported for distributed stage output")
        }
    }

    static func readLastLogitsRow(
        _ logits: NDArray,
        vocabSize: Int
    ) throws -> [Float] {
        switch logits.scalarType {
        case .float16:
            return try logits.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
                let viewShape = copySpan(shape)
                let viewStrides = copySpan(strides)
                let offsets = try DistributedCoreAIStageTensorReadbackLayout.lastLogitsRowOffsets(
                    shape: viewShape,
                    strides: viewStrides,
                    vocabSize: vocabSize)
                return offsets.map { Float(pointer[$0]) }
            }
        case .bfloat16:
            return try readLastBFloat16LogitsRow(logits, vocabSize: vocabSize)
        case .float32:
            return try logits.view(as: Float.self).withUnsafePointer { pointer, shape, strides in
                let viewShape = copySpan(shape)
                let viewStrides = copySpan(strides)
                let offsets = try DistributedCoreAIStageTensorReadbackLayout.lastLogitsRowOffsets(
                    shape: viewShape,
                    strides: viewStrides,
                    vocabSize: vocabSize)
                return offsets.map { pointer[$0] }
            }
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "logits scalar type \(logits.scalarType) is not supported for distributed stage output")
        }
    }

    fileprivate static func hiddenStateSummary(
        _ array: NDArray
    ) throws -> DistributedCoreAIStageTensorSummary {
        switch array.scalarType {
        case .float16:
            return try rank3Summary(array, as: Float16.self, tensorName: "hidden_states")
        case .bfloat16:
            return try rank3BFloat16Summary(array, tensorName: "hidden_states")
        case .float32:
            return try rank3Summary(array, as: Float.self, tensorName: "hidden_states")
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "hidden_states scalar type \(array.scalarType) is not supported for summary tracing")
        }
    }

    private static func fillInt32(_ array: inout NDArray, _ elements: [Int32]) {
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: elements)
    }

    private static func fillRoPE<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: inout NDArray,
        _ positionIDs: [Int32],
        rope: DistributedStageRoPEInputSpec,
        component: RoPEComponent,
        as _: T.Type
    ) {
        let halfDim = rope.headDim / 2
        var values: [T] = []
        values.reserveCapacity(positionIDs.count * rope.headDim)
        for positionID in positionIDs {
            let position = Double(positionID)
            for column in 0..<rope.headDim {
                let frequencyIndex = column % halfDim
                let exponent = Double(2 * frequencyIndex) / Double(rope.headDim)
                let invFrequency = 1.0 / pow(rope.theta, exponent)
                let angle = position * invFrequency
                let value = component == .cos ? cos(angle) : sin(angle)
                values.append(T(value))
            }
        }
        var view = array.mutableView(as: T.self)
        view.copyElements(fromContentsOf: values)
    }

    private static func applySoftTokenRows<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ values: [T],
        splice: DistributedSoftTokenSplice,
        to output: inout NDArray,
        positionRange: DistributedSequenceRange,
        as _: T.Type
    ) throws {
        guard values.count == splice.rowCount * splice.hiddenSize else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "soft_token_splice element count does not match shape")
        }
        let localStart = splice.positionStart - positionRange.lowerBound
        var view = output.mutableView(as: T.self)
        try view.withUnsafeMutablePointer { pointer, shape, strides in
            let viewShape = copySpan(shape)
            let viewStrides = copySpan(strides)
            guard viewShape.count == 3, viewShape[0] == 1 else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "hidden_states shape \(viewShape) does not match [1, sequence, hidden]")
            }
            guard localStart >= 0, localStart + splice.rowCount <= viewShape[1] else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "soft_token_splice row range is outside hidden_states output")
            }
            guard splice.hiddenSize == viewShape[2] else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "soft_token_splice hidden size must match hidden_states")
            }
            for row in 0..<splice.rowCount {
                let destinationRow = localStart + row
                let sourceBase = row * splice.hiddenSize
                for column in 0..<splice.hiddenSize {
                    let destinationOffset =
                        viewStrides[0] * 0
                        + viewStrides[1] * destinationRow
                        + viewStrides[2] * column
                    pointer[destinationOffset] = values[sourceBase + column]
                }
            }
        }
    }

    private static func readRank3<T: BitwiseCopyable>(
        _ array: NDArray,
        as _: T.Type,
        tensorName: String
    ) throws -> (shape: [Int], values: [T]) {
        try array.view(as: T.self).withUnsafePointer { pointer, shape, strides in
            let viewShape = copySpan(shape)
            let viewStrides = copySpan(strides)
            let offsets = try DistributedCoreAIStageTensorReadbackLayout.rank3Offsets(
                shape: viewShape,
                strides: viewStrides,
                tensorName: tensorName)
            return (viewShape, offsets.map { pointer[$0] })
        }
    }

    private static func rank3Summary<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: NDArray,
        as _: T.Type,
        tensorName: String
    ) throws -> DistributedCoreAIStageTensorSummary {
        try array.view(as: T.self).withUnsafePointer { pointer, shape, strides in
            let viewShape = copySpan(shape)
            let viewStrides = copySpan(strides)
            let offsets = try DistributedCoreAIStageTensorReadbackLayout.rank3Offsets(
                shape: viewShape,
                strides: viewStrides,
                tensorName: tensorName)
            return DistributedCoreAIStageTensorSummary(
                scalarType: String(describing: array.scalarType),
                shape: viewShape,
                strides: viewStrides,
                values: offsets.map { Double(pointer[$0]) })
        }
    }

    private static func fillBFloat16(_ array: inout NDArray, bitPatterns: [UInt16]) throws {
        let count = array.shape.reduce(1, *)
        guard count == bitPatterns.count else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "bfloat16 hidden_states element count \(bitPatterns.count) does not match array count \(count)")
        }
        var rawView = array.mutableRawView()
        var bytes = rawView.mutableBytes
        for (index, bitPattern) in bitPatterns.enumerated() {
            bytes.storeBytes(
                of: bitPattern,
                toByteOffset: index * MemoryLayout<UInt16>.size,
                as: UInt16.self)
        }
    }

    private static func readRank3BFloat16AsFloat16(
        _ array: NDArray,
        tensorName: String
    ) throws -> (shape: [Int], values: [Float16]) {
        let shape = array.shape
        let offsets = try contiguousRank3Offsets(shape: shape, tensorName: tensorName)
        let values = try readBFloat16Values(array, offsets: offsets).map(Float16.init)
        return (shape, values)
    }

    private static func rank3BFloat16Summary(
        _ array: NDArray,
        tensorName: String
    ) throws -> DistributedCoreAIStageTensorSummary {
        let shape = array.shape
        let offsets = try contiguousRank3Offsets(shape: shape, tensorName: tensorName)
        return DistributedCoreAIStageTensorSummary(
            scalarType: String(describing: array.scalarType),
            shape: shape,
            strides: nil,
            values: try readBFloat16Values(array, offsets: offsets).map(Double.init))
    }

    private static func readLastBFloat16LogitsRow(
        _ array: NDArray,
        vocabSize: Int
    ) throws -> [Float] {
        let shape = array.shape
        let offsets: [Int]
        if shape.count == 3 {
            guard shape[0] == 1, shape[1] > 0, shape[2] == vocabSize else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "bfloat16 logits shape \(shape) does not match [1, -1, \(vocabSize)]")
            }
            let row = shape[1] - 1
            offsets = (0..<vocabSize).map { row * vocabSize + $0 }
        } else if shape.count == 2 {
            guard shape[0] > 0, shape[1] == vocabSize else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "bfloat16 logits shape \(shape) does not match [-1, \(vocabSize)]")
            }
            let row = shape[0] - 1
            offsets = (0..<vocabSize).map { row * vocabSize + $0 }
        } else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "bfloat16 logits rank \(shape.count) is unsupported")
        }
        return try readBFloat16Values(array, offsets: offsets)
    }

    private static func readBFloat16Values(_ array: NDArray, offsets: [Int]) throws -> [Float] {
        let raw = array.rawView().bytes
        let elementCount = raw.byteCount / MemoryLayout<UInt16>.size
        return try offsets.map { offset in
            guard offset >= 0 && offset < elementCount else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "bfloat16 read offset \(offset) is outside element count \(elementCount)")
            }
            let bits = raw.unsafeLoadUnaligned(
                fromByteOffset: offset * MemoryLayout<UInt16>.size,
                as: UInt16.self)
            return Float(bitPattern: UInt32(bits) << 16)
        }
    }

    private static func contiguousRank3Offsets(shape: [Int], tensorName: String) throws -> [Int] {
        guard shape.count == 3 else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) rank \(shape.count) does not match 3")
        }
        guard shape.allSatisfy({ $0 > 0 }) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) shape \(shape) has non-positive dimensions")
        }
        let count = shape.reduce(1, *)
        return Array(0..<count)
    }

    private static func bfloat16BitPattern(from value: Float) -> UInt16 {
        let bits = value.bitPattern
        let roundToNearestEven = UInt32(0x7fff) + ((bits >> 16) & 1)
        return UInt16((bits &+ roundToNearestEven) >> 16)
    }

    private static func copySpan(_ span: Span<Int>) -> [Int] {
        (0..<span.count).map { span[$0] }
    }

    private static func requireScalarType(
        _ expected: NDArray.ScalarType,
        descriptor: NDArrayDescriptor,
        tensorName: String
    ) throws {
        guard descriptor.scalarType == expected else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(descriptor.scalarType) does not match \(expected)")
        }
    }

    private static func requireFloatingScalarType(
        _ descriptor: NDArrayDescriptor,
        tensorName: String
    ) throws {
        switch descriptor.scalarType {
        case .float16, .bfloat16, .float32:
            return
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(descriptor.scalarType) is not supported for distributed stage IO")
        }
    }

    private static func requireBoundaryScalarType(
        _ expected: DistributedTensorScalarType,
        descriptor: NDArrayDescriptor,
        tensorName: String
    ) throws {
        switch (expected, descriptor.scalarType) {
        case (.float16, .float16), (.float16, .bfloat16), (.float32, .float32):
            return
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(descriptor.scalarType) does not match boundary \(expected.rawValue)")
        }
    }
}

private struct DistributedCoreAIStageRequestCacheGroup {
    var keyCache: NDArray
    var valueCache: NDArray
}

private struct DistributedCoreAIStageRequestState {
    let requestID: String
    let kvCapacity: Int
    var processedTokenCount: Int
    var cacheGroups: [String: DistributedCoreAIStageRequestCacheGroup]

    init(
        requestID: String,
        kvCapacity: Int,
        processedTokenCount: Int = 0,
        cacheGroups: [String: DistributedCoreAIStageRequestCacheGroup] = [:]
    ) {
        self.requestID = requestID
        self.kvCapacity = kvCapacity
        self.processedTokenCount = processedTokenCount
        self.cacheGroups = cacheGroups
    }

    func cacheGroup(named groupName: String) throws -> DistributedCoreAIStageRequestCacheGroup {
        guard let group = cacheGroups[groupName] else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage KV cache group '\(groupName)' is missing")
        }
        return group
    }
}
#endif

enum DistributedCoreAIStageKVCacheShape {
    static func resolved(_ descriptorShape: [Int], capacity: Int) throws -> [Int] {
        guard capacity > 0 else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "kv_capacity must be positive")
        }
        guard !descriptorShape.isEmpty else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "KV cache descriptor shape is empty")
        }
        let dynamicIndexes = descriptorShape.indices.filter { descriptorShape[$0] < 0 }
        if dynamicIndexes.isEmpty {
            guard descriptorShape.allSatisfy({ $0 > 0 }) else {
                throw DistributedStageExecutionError.invalidControlFrame(
                    "KV cache descriptor shape \(descriptorShape) resolves to invalid fixed shape")
            }
            return descriptorShape
        }
        guard dynamicIndexes.count == 1, let dynamicIndex = dynamicIndexes.first else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "KV cache descriptor shape \(descriptorShape) must have exactly one dynamic capacity dimension")
        }
        var resolved = descriptorShape
        resolved[dynamicIndex] = capacity
        guard resolved.allSatisfy({ $0 > 0 }) else {
            throw DistributedStageExecutionError.invalidControlFrame(
                "KV cache descriptor shape \(descriptorShape) resolves to invalid shape \(resolved)")
        }
        return resolved
    }
}

enum DistributedCoreAIStageTensorReadbackLayout {
    static func rank3Offsets(
        shape: [Int],
        strides: [Int],
        tensorName: String
    ) throws -> [Int] {
        try validateRank3(shape: shape, strides: strides, tensorName: tensorName)
        var offsets: [Int] = []
        offsets.reserveCapacity(shape[0] * shape[1] * shape[2])
        for batch in 0..<shape[0] {
            for row in 0..<shape[1] {
                for column in 0..<shape[2] {
                    offsets.append(
                        batch * strides[0]
                            + row * strides[1]
                            + column * strides[2])
                }
            }
        }
        return offsets
    }

    static func lastLogitsRowOffsets(
        shape: [Int],
        strides: [Int],
        vocabSize: Int
    ) throws -> [Int] {
        try validateRank3(shape: shape, strides: strides, tensorName: "logits")
        guard shape[0] == 1 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "logits shape \(shape) batch dimension must be 1")
        }
        guard vocabSize > 0 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "vocab_size must be positive")
        }
        guard shape[2] == vocabSize else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "logits shape \(shape) does not match vocab_size \(vocabSize)")
        }
        let base = (shape[1] - 1) * strides[1]
        return (0..<vocabSize).map { base + $0 * strides[2] }
    }

    private static func validateRank3(
        shape: [Int],
        strides: [Int],
        tensorName: String
    ) throws {
        guard shape.count == 3 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "\(tensorName) shape \(shape) must be rank 3")
        }
        guard strides.count == 3 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "\(tensorName) strides \(strides) must be rank 3")
        }
        guard shape.allSatisfy({ $0 > 0 }) else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "\(tensorName) shape \(shape) must be positive")
        }
        guard strides.allSatisfy({ $0 > 0 }) else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "\(tensorName) strides \(strides) must be positive")
        }
    }
}

enum DistributedCoreAIStageNDArrayShape {
    static func resolvedMatrix(
        _ descriptorShape: [Int],
        columns: Int,
        tensorName: String
    ) throws -> [Int] {
        guard columns > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "\(tensorName) count must be positive")
        }
        return try resolvedExact(
            descriptorShape,
            actualShape: [1, columns],
            tensorName: tensorName)
    }

    static func resolvedHiddenStates(
        _ descriptorShape: [Int],
        packetShape: [Int]
    ) throws -> [Int] {
        try resolvedExact(
            descriptorShape,
            actualShape: try positiveShape(packetShape, tensorName: "hidden_states"),
            tensorName: "hidden_states")
    }

    static func resolvedHiddenStatesOutput(
        _ descriptorShape: [Int],
        positionCount: Int,
        boundaryTensor: DistributedBoundaryTensorSpec?
    ) throws -> [Int] {
        guard positionCount > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "hidden_states position count must be positive")
        }
        guard let boundaryTensor else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "hidden_states output requires boundary tensor metadata")
        }
        if let message = boundaryTensor.validationErrorMessage {
            throw DistributedStageExecutionError.invalidForwardInput(message)
        }
        if boundaryTensor.shape[1] != -1 && boundaryTensor.shape[1] != positionCount {
            throw DistributedStageExecutionError.invalidForwardInput(
                "hidden_states position count \(positionCount) does not match boundary sequence dimension \(boundaryTensor.shape[1])")
        }
        return try resolvedExact(
            descriptorShape,
            actualShape: [boundaryTensor.shape[0], positionCount, boundaryTensor.shape[2]],
            tensorName: "hidden_states")
    }

    static func resolvedRoPEInput(
        _ descriptorShape: [Int],
        positionCount: Int,
        headDim: Int,
        tensorName: String
    ) throws -> [Int] {
        guard positionCount > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "\(tensorName) position count must be positive")
        }
        guard headDim > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "\(tensorName) head_dim must be positive")
        }
        return try resolvedExact(
            descriptorShape,
            actualShape: [1, positionCount, headDim],
            tensorName: tensorName)
    }

    static func resolvedLogitsOutput(
        _ descriptorShape: [Int],
        positionCount: Int,
        vocabSize: Int
    ) throws -> [Int] {
        guard positionCount > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "logits position count must be positive")
        }
        guard vocabSize > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "vocab_size must be positive")
        }
        return try resolvedExact(
            descriptorShape,
            actualShape: [1, positionCount, vocabSize],
            tensorName: "logits")
    }

    private static func resolvedExact(
        _ descriptorShape: [Int],
        actualShape: [Int],
        tensorName: String
    ) throws -> [Int] {
        guard descriptorShape.count == actualShape.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "\(tensorName) descriptor shape \(descriptorShape) rank does not match runtime shape \(actualShape)")
        }
        for dimension in descriptorShape where dimension == 0 {
            throw DistributedStageExecutionError.invalidForwardInput(
                "\(tensorName) descriptor shape \(descriptorShape) has invalid zero dimension")
        }
        for (descriptorDimension, actualDimension) in zip(descriptorShape, actualShape)
            where descriptorDimension > 0 && descriptorDimension != actualDimension
        {
            throw DistributedStageExecutionError.invalidForwardInput(
                "\(tensorName) descriptor shape \(descriptorShape) does not match runtime shape \(actualShape)")
        }
        return actualShape
    }

    private static func positiveShape(_ shape: [Int], tensorName: String) throws -> [Int] {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "\(tensorName) runtime shape \(shape) must be positive")
        }
        return shape
    }
}
