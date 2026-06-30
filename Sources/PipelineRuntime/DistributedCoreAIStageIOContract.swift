#if COREAI_RUNTIME
import CoreAI
import Foundation

extension DistributedStageIOContract {
    init(
        coreAIFunctionName functionName: String,
        descriptor: InferenceFunctionDescriptor
    ) throws {
        try Self.validateStateDescriptors(descriptor)
        self.init(
            functionName: functionName,
            inputs: try descriptor.inputNames.map { name in
                try Self.inputTensor(name: name, descriptor: descriptor)
            },
            outputs: try descriptor.outputNames.map { name in
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
        case .float32:
            return .float32
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage \(kind) '\(tensorName)' scalar type \(scalarType) "
                    + "is unsupported (expected int32/float16/float32)")
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

struct DistributedCoreAIStageCacheIO {
    let contract: DistributedCoreAIStageCacheContract
    let keyCacheName: String?
    let valueCacheName: String?
    let keyCacheDescriptor: NDArrayDescriptor?
    let valueCacheDescriptor: NDArrayDescriptor?
    let keyCacheOutputDescriptor: NDArrayDescriptor?
    let valueCacheOutputDescriptor: NDArrayDescriptor?

    static func extracted(from descriptor: InferenceFunctionDescriptor) throws
        -> DistributedCoreAIStageCacheIO
    {
        let hasStateKV = descriptor.stateNames.count == 2
        let hasExplicitKV =
            descriptor.stateNames.isEmpty
            && descriptor.inputNames.contains("keyCache")
            && descriptor.inputNames.contains("valueCache")
            && descriptor.outputNames.contains("keyCache")
            && descriptor.outputNames.contains("valueCache")
        if hasStateKV {
            let keyName = pick("keyCache", descriptor.stateNames, index: 0)
            let valueName = pick("valueCache", descriptor.stateNames, index: 1)
            guard case .ndArray(let keyDesc) = descriptor.stateDescriptor(of: keyName),
                case .ndArray(let valueDesc) = descriptor.stateDescriptor(of: valueName)
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache states are not NDArrays")
            }
            return DistributedCoreAIStageCacheIO(
                contract: .stateful,
                keyCacheName: keyName,
                valueCacheName: valueName,
                keyCacheDescriptor: keyDesc,
                valueCacheDescriptor: valueDesc,
                keyCacheOutputDescriptor: keyDesc,
                valueCacheOutputDescriptor: valueDesc)
        }
        if hasExplicitKV {
            let keyName = pick("keyCache", descriptor.inputNames, index: 2)
            let valueName = pick("valueCache", descriptor.inputNames, index: 3)
            guard case .ndArray(let inputKeyDesc) = descriptor.inputDescriptor(of: keyName),
                case .ndArray(let inputValueDesc) = descriptor.inputDescriptor(of: valueName)
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache inputs are not NDArrays")
            }
            guard case .ndArray(let outputKeyDesc) = descriptor.outputDescriptor(of: keyName),
                case .ndArray(let outputValueDesc) = descriptor.outputDescriptor(of: valueName)
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache outputs are not NDArrays")
            }
            return DistributedCoreAIStageCacheIO(
                contract: .explicitOutputs,
                keyCacheName: keyName,
                valueCacheName: valueName,
                keyCacheDescriptor: inputKeyDesc,
                valueCacheDescriptor: inputValueDesc,
                keyCacheOutputDescriptor: outputKeyDesc,
                valueCacheOutputDescriptor: outputValueDesc)
        }
        if descriptor.stateNames.isEmpty
            && !descriptor.inputNames.contains("keyCache")
            && !descriptor.inputNames.contains("valueCache")
            && !descriptor.outputNames.contains("keyCache")
            && !descriptor.outputNames.contains("valueCache")
        {
            return DistributedCoreAIStageCacheIO(
                contract: .none,
                keyCacheName: nil,
                valueCacheName: nil,
                keyCacheDescriptor: nil,
                valueCacheDescriptor: nil,
                keyCacheOutputDescriptor: nil,
                valueCacheOutputDescriptor: nil)
        }
        throw CoreAIPipeline.RuntimeError.modelContract(
            "expected no KV cache, stateful KV states, or explicit-cache inputs/outputs; "
                + ioSummary(descriptor))
    }

    private static func pick(_ wanted: String, _ names: [String], index: Int) -> String {
        names.contains(wanted) ? wanted : names[index]
    }

    private static func ioSummary(_ descriptor: InferenceFunctionDescriptor) -> String {
        "inputs=\(descriptor.inputNames), outputs=\(descriptor.outputNames), states=\(descriptor.stateNames)"
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

public final class DistributedCoreAIStageHandle: DistributedStageHandle {
    public let descriptor: DistributedStageDescriptor
    public let assetURL: URL
    public let functionName: String
    public let ioContract: DistributedStageIOContract

    private let model: AIModel
    private let functionDescriptor: InferenceFunctionDescriptor
    private let cacheIO: DistributedCoreAIStageCacheIO
    private var requestStates: [String: DistributedCoreAIStageRequestState] = [:]

    private init(
        descriptor: DistributedStageDescriptor,
        assetURL: URL,
        functionName: String,
        ioContract: DistributedStageIOContract,
        model: AIModel,
        functionDescriptor: InferenceFunctionDescriptor,
        cacheIO: DistributedCoreAIStageCacheIO
    ) {
        self.descriptor = descriptor
        self.assetURL = assetURL
        self.functionName = functionName
        self.ioContract = ioContract
        self.model = model
        self.functionDescriptor = functionDescriptor
        self.cacheIO = cacheIO
    }

    public static func load(
        for context: DistributedStageHandleFactoryContext,
        functionName: String? = nil,
        vocabSize: Int? = nil
    ) async throws -> DistributedCoreAIStageHandle {
        let assetURL = try context.requireExistingAssetURL()
        let resolvedFunctionName = functionName ?? context.mainFunctionName
        let resolvedVocabSize = vocabSize ?? context.vocabSize
        var specialization = SpecializationOptions(
            preferredComputeUnitKind: LLMEngine.preferredComputeUnit())
        specialization.expectFrequentReshapes = true
        let model = try await AIModel.specialize(
            contentsOf: assetURL,
            options: specialization,
            cache: .default,
            cachePolicy: .persistent)
        guard let functionDescriptor = model.functionDescriptor(for: resolvedFunctionName) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage function '\(resolvedFunctionName)' not found in \(assetURL.lastPathComponent); have \(model.functionNames)")
        }
        let ioContract = try context.validateCoreAIStageIOContract(
            functionName: resolvedFunctionName,
            descriptor: functionDescriptor,
            vocabSize: resolvedVocabSize)
        let cacheIO = try DistributedCoreAIStageCacheIO.extracted(from: functionDescriptor)
        return DistributedCoreAIStageHandle(
            descriptor: context.descriptor,
            assetURL: assetURL,
            functionName: resolvedFunctionName,
            ioContract: ioContract,
            model: model,
            functionDescriptor: functionDescriptor,
            cacheIO: cacheIO)
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
        guard requestStates[input.requestID] != nil else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "request_id \(input.requestID) is not allocated")
        }
        throw unimplemented("forward")
    }

    public func reset(requestID: String) async throws {
        _ = requestID
        throw unimplemented("reset")
    }

    public func free(requestID: String) async {
        requestStates.removeValue(forKey: requestID)
    }

    private func unimplemented(_ operation: String) -> CoreAIPipeline.RuntimeError {
        _ = (model, functionDescriptor)
        return .modelContract(
            "distributed Core AI stage \(operation) is not implemented yet")
    }

    private func makeRequestState(
        for allocation: DistributedStageAllocation
    ) throws -> DistributedCoreAIStageRequestState {
        switch cacheIO.contract {
        case .none:
            return DistributedCoreAIStageRequestState(
                requestID: allocation.requestID,
                kvCapacity: allocation.kvCapacity)
        case .stateful, .explicitOutputs:
            guard let keyCacheDescriptor = cacheIO.keyCacheDescriptor,
                let valueCacheDescriptor = cacheIO.valueCacheDescriptor
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage KV cache descriptors are missing")
            }
            return DistributedCoreAIStageRequestState(
                requestID: allocation.requestID,
                kvCapacity: allocation.kvCapacity,
                keyCache: try Self.makeKVCache(
                    descriptor: keyCacheDescriptor,
                    capacity: allocation.kvCapacity),
                valueCache: try Self.makeKVCache(
                    descriptor: valueCacheDescriptor,
                    capacity: allocation.kvCapacity))
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

enum DistributedCoreAIStageNDArrayIO {
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

    static func makeHiddenStates(
        packet: DistributedHiddenStatePacket,
        descriptor: NDArrayDescriptor
    ) throws -> NDArray {
        let shape = try DistributedCoreAIStageNDArrayShape.resolvedHiddenStates(
            descriptor.shape,
            packetShape: packet.metadata.shape)
        var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
        switch (descriptor.scalarType, packet.metadata.scalarType) {
        case (.float16, .float16):
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: try packet.float16Values())
        case (.float32, .float32):
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: try packet.float32Values())
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "hidden_states scalar type \(descriptor.scalarType) does not match packet \(packet.metadata.scalarType.rawValue)")
        }
        return array
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

    private static func fillInt32(_ array: inout NDArray, _ elements: [Int32]) {
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: elements)
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
        case .float16, .float32:
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
        case (.float16, .float16), (.float32, .float32):
            return
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(descriptor.scalarType) does not match boundary \(expected.rawValue)")
        }
    }
}

private struct DistributedCoreAIStageRequestState {
    let requestID: String
    let kvCapacity: Int
    var processedTokenCount: Int
    var keyCache: NDArray?
    var valueCache: NDArray?

    init(
        requestID: String,
        kvCapacity: Int,
        processedTokenCount: Int = 0,
        keyCache: NDArray? = nil,
        valueCache: NDArray? = nil
    ) {
        self.requestID = requestID
        self.kvCapacity = kvCapacity
        self.processedTokenCount = processedTokenCount
        self.keyCache = keyCache
        self.valueCache = valueCache
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
