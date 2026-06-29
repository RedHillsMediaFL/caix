#if COREAI_RUNTIME
import CoreAI

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
#endif
