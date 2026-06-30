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
                logitsOutput: nil)
        case .transformerLayers:
            return DistributedCoreAIStageExecutionIO(
                inputIDs: nil,
                positionIDs: positionIDs,
                hiddenStatesInput: try input(.hiddenStates, descriptor: descriptor),
                hiddenStatesOutput: try output(.hiddenStates, descriptor: descriptor),
                logitsOutput: nil)
        case .finalNormHead:
            guard vocabSize != nil else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed final_norm_head stage '\(stage.id)' requires vocab_size")
            }
            return DistributedCoreAIStageExecutionIO(
                inputIDs: nil,
                positionIDs: positionIDs,
                hiddenStatesInput: try input(.hiddenStates, descriptor: descriptor),
                hiddenStatesOutput: nil,
                logitsOutput: try output(.logits, descriptor: descriptor))
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

public final class DistributedCoreAIStageHandle: DistributedStageHandle {
    public let descriptor: DistributedStageDescriptor
    public let assetURL: URL
    public let functionName: String
    public let ioContract: DistributedStageIOContract

    private let model: AIModel
    private let prefillFunction: InferenceFunction
    private let decodeFunction: InferenceFunction
    private let decodeFunctionName: String?
    private let functionDescriptor: InferenceFunctionDescriptor
    private let executionIO: DistributedCoreAIStageExecutionIO
    private let cacheIO: DistributedCoreAIStageCacheIO
    private let boundaryTensor: DistributedBoundaryTensorSpec?
    private let nextStageID: String?
    private let vocabSize: Int?
    private var requestStates: [String: DistributedCoreAIStageRequestState] = [:]

    private init(
        descriptor: DistributedStageDescriptor,
        assetURL: URL,
        functionName: String,
        ioContract: DistributedStageIOContract,
        model: AIModel,
        prefillFunction: InferenceFunction,
        decodeFunction: InferenceFunction,
        decodeFunctionName: String?,
        functionDescriptor: InferenceFunctionDescriptor,
        executionIO: DistributedCoreAIStageExecutionIO,
        cacheIO: DistributedCoreAIStageCacheIO,
        boundaryTensor: DistributedBoundaryTensorSpec?,
        nextStageID: String?,
        vocabSize: Int?
    ) {
        self.descriptor = descriptor
        self.assetURL = assetURL
        self.functionName = functionName
        self.ioContract = ioContract
        self.model = model
        self.prefillFunction = prefillFunction
        self.decodeFunction = decodeFunction
        self.decodeFunctionName = decodeFunctionName
        self.functionDescriptor = functionDescriptor
        self.executionIO = executionIO
        self.cacheIO = cacheIO
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
        let executionIO = try DistributedCoreAIStageExecutionIO.bound(
            for: context.descriptor,
            descriptor: functionDescriptor,
            vocabSize: resolvedVocabSize)
        let prefillFunction = try Self.loadFunction(
            named: resolvedFunctionName,
            from: model,
            assetURL: assetURL)
        let resolvedDecodeFunctionName = context.decodeFunctionName
        let decodeFunction = try await Self.loadDecodeFunction(
            named: resolvedDecodeFunctionName,
            mainFunctionName: resolvedFunctionName,
            mainFunction: prefillFunction,
            mainModel: model,
            mainAssetURL: assetURL,
            decodeAssetURL: context.resolvedDecodeAssetURL,
            specialization: specialization)
        return DistributedCoreAIStageHandle(
            descriptor: context.descriptor,
            assetURL: assetURL,
            functionName: resolvedFunctionName,
            ioContract: ioContract,
            model: model,
            prefillFunction: prefillFunction,
            decodeFunction: decodeFunction,
            decodeFunctionName: resolvedDecodeFunctionName,
            functionDescriptor: functionDescriptor,
            executionIO: executionIO,
            cacheIO: cacheIO,
            boundaryTensor: context.boundaryTensor,
            nextStageID: context.nextStage?.id,
            vocabSize: resolvedVocabSize)
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
        try validateForwardInput(input, requestState: requestState)

        let positionCount = input.positionRange.count
        let positionIDs = try DistributedCoreAIStageNDArrayIO.makePositionIDs(
            positionIDs: input.positionIDs,
            descriptor: executionIO.positionIDs.descriptor)
        let activeFunction = activeFunction(positionCount: positionCount)

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
                inputs: [
                    inputIDsBinding.name: inputIDs,
                    executionIO.positionIDs.name: positionIDs,
                ],
                outputViews: consume outputViews,
                requestState: &requestState)
            output = DistributedStageForwardOutput(
                stageID: descriptor.id,
                stepIndex: input.stepIndex,
                hiddenState: try DistributedCoreAIStageNDArrayIO.makeHiddenStatePacket(
                    from: hiddenStates,
                    requestID: input.requestID,
                    sourceStageID: descriptor.id,
                    destinationStageID: nextStageID,
                    positionRange: input.positionRange,
                    stepIndex: input.stepIndex))

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
            let hiddenInput = try DistributedCoreAIStageNDArrayIO.makeHiddenStates(
                packet: hiddenStatePacket,
                descriptor: hiddenInputBinding.descriptor)
            var hiddenOutput = try DistributedCoreAIStageNDArrayIO.makeHiddenStatesOutput(
                positionCount: positionCount,
                descriptor: hiddenOutputBinding.descriptor,
                boundaryTensor: boundaryTensor)
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&hiddenOutput, for: hiddenOutputBinding.name)
            try await run(
                activeFunction,
                inputs: [
                    hiddenInputBinding.name: hiddenInput,
                    executionIO.positionIDs.name: positionIDs,
                ],
                outputViews: consume outputViews,
                requestState: &requestState)
            output = DistributedStageForwardOutput(
                stageID: descriptor.id,
                stepIndex: input.stepIndex,
                hiddenState: try DistributedCoreAIStageNDArrayIO.makeHiddenStatePacket(
                    from: hiddenOutput,
                    requestID: input.requestID,
                    sourceStageID: descriptor.id,
                    destinationStageID: nextStageID,
                    positionRange: input.positionRange,
                    stepIndex: input.stepIndex))

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
            let hiddenInput = try DistributedCoreAIStageNDArrayIO.makeHiddenStates(
                packet: hiddenStatePacket,
                descriptor: hiddenInputBinding.descriptor)
            var logits = try DistributedCoreAIStageNDArrayIO.makeLogitsOutput(
                positionCount: positionCount,
                vocabSize: vocabSize,
                descriptor: logitsOutputBinding.descriptor)
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&logits, for: logitsOutputBinding.name)
            try await run(
                activeFunction,
                inputs: [
                    hiddenInputBinding.name: hiddenInput,
                    executionIO.positionIDs.name: positionIDs,
                ],
                outputViews: consume outputViews,
                requestState: &requestState)
            let tokenID = Sampler.argmax(
                try DistributedCoreAIStageNDArrayIO.readLastLogitsRow(
                    logits,
                    vocabSize: vocabSize))
            guard tokenID <= Int(Int32.max) else {
                throw DistributedStageExecutionError.invalidStageOutput(
                    "token id \(tokenID) exceeds Int32.max")
            }
            output = DistributedStageForwardOutput(
                stageID: descriptor.id,
                stepIndex: input.stepIndex,
                tokenID: Int32(tokenID))
        }

        requestState.processedTokenCount += positionCount
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

    private func unimplemented(_ operation: String) -> CoreAIPipeline.RuntimeError {
        _ = (
            model, prefillFunction, decodeFunction, decodeFunctionName, functionDescriptor,
            executionIO, boundaryTensor, nextStageID, vocabSize
        )
        return .modelContract(
            "distributed Core AI stage \(operation) is not implemented yet")
    }

    private func activeFunction(positionCount: Int) -> InferenceFunction {
        positionCount == 1 ? decodeFunction : prefillFunction
    }

    private func validateForwardInput(
        _ input: DistributedStageForwardInput,
        requestState: DistributedCoreAIStageRequestState
    ) throws {
        guard input.stepIndex >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "step_index must be non-negative")
        }
        guard input.positionRange.isValid else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_range is invalid")
        }
        guard input.positionRange.lowerBound == requestState.processedTokenCount else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_range lower_bound \(input.positionRange.lowerBound) does not match processed_token_count \(requestState.processedTokenCount)")
        }
        guard input.positionRange.upperBound <= requestState.kvCapacity else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_range upper_bound \(input.positionRange.upperBound) exceeds kv_capacity \(requestState.kvCapacity)")
        }
        guard input.positionIDs.count >= input.positionRange.count else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "position_ids count must cover position_range")
        }

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
        case .transformerLayers, .finalNormHead:
            guard input.tokenIDs.isEmpty else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(descriptor.role.rawValue) stage must not receive token_ids")
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
        }
    }

    private func run(
        _ function: InferenceFunction,
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
            guard var keyCache = requestState.keyCache,
                var valueCache = requestState.valueCache,
                let keyCacheName = cacheIO.keyCacheName,
                let valueCacheName = cacheIO.valueCacheName
            else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "distributed stage stateful KV cache is missing")
            }
            var states = InferenceFunction.MutableViews()
            states.insert(&keyCache, for: keyCacheName)
            states.insert(&valueCache, for: valueCacheName)
            _ = try await function.run(
                inputs: inputs,
                states: consume states,
                outputViews: consume outputViews)
            requestState.keyCache = keyCache
            requestState.valueCache = valueCache
        case .explicitOutputs:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "distributed Core AI explicit-cache stage execution is not implemented yet")
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

    private static func loadDecodeFunction(
        named decodeFunctionName: String?,
        mainFunctionName: String,
        mainFunction: InferenceFunction,
        mainModel: AIModel,
        mainAssetURL: URL,
        decodeAssetURL: URL?,
        specialization: SpecializationOptions
    ) async throws -> InferenceFunction {
        guard let decodeFunctionName, decodeFunctionName != mainFunctionName else {
            return mainFunction
        }
        if let decodeAssetURL {
            let decodeModel = try await AIModel.specialize(
                contentsOf: decodeAssetURL,
                options: specialization,
                cache: .default,
                cachePolicy: .persistent)
            return try loadFunction(
                named: decodeFunctionName,
                from: decodeModel,
                assetURL: decodeAssetURL)
        }
        return try loadFunction(
            named: decodeFunctionName,
            from: mainModel,
            assetURL: mainAssetURL)
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

    private static func fillInt32(_ array: inout NDArray, _ elements: [Int32]) {
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: elements)
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
