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
        _ = allocation
        throw unimplemented("allocate")
    }

    public func forward(
        _ input: DistributedStageForwardInput
    ) async throws -> DistributedStageForwardOutput {
        _ = input
        throw unimplemented("forward")
    }

    public func reset(requestID: String) async throws {
        _ = requestID
        throw unimplemented("reset")
    }

    public func free(requestID: String) async {
        _ = requestID
        // The protocol makes free nonthrowing. This handle never allocates state until tensor
        // execution lands, so there is nothing to release here.
    }

    private func unimplemented(_ operation: String) -> CoreAIPipeline.RuntimeError {
        _ = (model, functionDescriptor, cacheIO)
        return .modelContract(
            "distributed Core AI stage \(operation) is not implemented yet")
    }
}
#endif
