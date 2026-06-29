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

public final class DistributedCoreAIStageHandleFactory: DistributedStageHandleFactory {
    private let functionName: String
    private let vocabSize: Int?

    public init(functionName: String = "main", vocabSize: Int? = nil) {
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

    private init(
        descriptor: DistributedStageDescriptor,
        assetURL: URL,
        functionName: String,
        ioContract: DistributedStageIOContract,
        model: AIModel,
        functionDescriptor: InferenceFunctionDescriptor
    ) {
        self.descriptor = descriptor
        self.assetURL = assetURL
        self.functionName = functionName
        self.ioContract = ioContract
        self.model = model
        self.functionDescriptor = functionDescriptor
    }

    public static func load(
        for context: DistributedStageHandleFactoryContext,
        functionName: String = "main",
        vocabSize: Int? = nil
    ) async throws -> DistributedCoreAIStageHandle {
        let assetURL = try context.requireExistingAssetURL()
        var specialization = SpecializationOptions(
            preferredComputeUnitKind: LLMEngine.preferredComputeUnit())
        specialization.expectFrequentReshapes = true
        let model = try await AIModel.specialize(
            contentsOf: assetURL,
            options: specialization,
            cache: .default,
            cachePolicy: .persistent)
        guard let functionDescriptor = model.functionDescriptor(for: functionName) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed stage function '\(functionName)' not found in \(assetURL.lastPathComponent); have \(model.functionNames)")
        }
        let ioContract = try context.validateCoreAIStageIOContract(
            functionName: functionName,
            descriptor: functionDescriptor,
            vocabSize: vocabSize)
        return DistributedCoreAIStageHandle(
            descriptor: context.descriptor,
            assetURL: assetURL,
            functionName: functionName,
            ioContract: ioContract,
            model: model,
            functionDescriptor: functionDescriptor)
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
        _ = (model, functionDescriptor)
        return .modelContract(
            "distributed Core AI stage \(operation) is not implemented yet")
    }
}
#endif
