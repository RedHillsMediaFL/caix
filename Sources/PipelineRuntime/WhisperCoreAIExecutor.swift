#if COREAI_RUNTIME

import CoreAI
import Foundation
import Tokenizers

/// One specialized, resident Core AI model and its three immutable function handles.
///
/// Asset provenance is intentionally not accepted here: the public loader must authenticate the
/// CAIX manifest and asset bytes before calling this low-level adapter.
final class WhisperCoreAIModelFactory: @unchecked Sendable, WhisperNativeSessionFactory {
    private let model: AIModel
    private let encodeFunction: InferenceFunction
    private let loadFunction: InferenceFunction
    private let decodeFunction: InferenceFunction
    private let encodeDescriptor: InferenceFunctionDescriptor
    private let loadDescriptor: InferenceFunctionDescriptor
    private let decodeDescriptor: InferenceFunctionDescriptor

    static func specialize(assetURL: URL) async throws -> WhisperCoreAIModelFactory {
        var options = SpecializationOptions(
            preferredComputeUnitKind: LLMEngine.preferredComputeUnit())
        options.expectFrequentReshapes = false
        let model = try await AIModel.specialize(
            contentsOf: assetURL,
            options: options,
            cache: .default,
            cachePolicy: .persistent)
        return try WhisperCoreAIModelFactory(model: model)
    }

    private init(model: AIModel) throws {
        let functionNames = ["encode", "load_cross_kv", "decode_step"]
        guard Set(model.functionNames) == Set(functionNames) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Whisper entrypoints must be exactly \(functionNames.sorted()); "
                    + "got \(model.functionNames.sorted())")
        }
        var projected: [String: WhisperNativeFunctionDescriptor] = [:]
        var descriptors: [String: InferenceFunctionDescriptor] = [:]
        for name in functionNames {
            guard let descriptor = model.functionDescriptor(for: name) else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Whisper function '\(name)' not found; have \(model.functionNames)")
            }
            projected[name] = try Self.project(descriptor, functionName: name)
            descriptors[name] = descriptor
        }
        try WhisperNativeContract.validate(projected)
        try Self.validateBridgeRelationships(
            encode: descriptors["encode"]!,
            load: descriptors["load_cross_kv"]!,
            decode: descriptors["decode_step"]!)

        guard let encodeFunction = try model.loadFunction(named: "encode"),
            let loadFunction = try model.loadFunction(named: "load_cross_kv"),
            let decodeFunction = try model.loadFunction(named: "decode_step")
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Whisper v2 functions passed descriptors but could not be loaded")
        }

        self.model = model
        self.encodeFunction = encodeFunction
        self.loadFunction = loadFunction
        self.decodeFunction = decodeFunction
        self.encodeDescriptor = descriptors["encode"]!
        self.loadDescriptor = descriptors["load_cross_kv"]!
        self.decodeDescriptor = descriptors["decode_step"]!
    }

    func makeSession() async throws -> any WhisperNativeSession {
        try WhisperCoreAISession(
            encodeFunction: encodeFunction,
            loadFunction: loadFunction,
            decodeFunction: decodeFunction,
            encodeDescriptor: encodeDescriptor,
            loadDescriptor: loadDescriptor,
            decodeDescriptor: decodeDescriptor)
    }

    private static func project(
        _ descriptor: InferenceFunctionDescriptor,
        functionName: String
    ) throws -> WhisperNativeFunctionDescriptor {
        func tensor(
            _ array: NDArrayDescriptor,
            name: String,
            category: String
        ) throws -> WhisperNativeTensorDescriptor {
            let scalarType: WhisperNativeScalarType
            switch array.scalarType {
            case .float16: scalarType = .float16
            case .int32: scalarType = .int32
            default:
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Whisper \(functionName) \(category) '\(name)' has unsupported "
                        + "scalar type \(array.scalarType)")
            }
            return WhisperNativeTensorDescriptor(
                scalarType: scalarType,
                shape: array.shape)
        }

        var inputs: [String: WhisperNativeTensorDescriptor] = [:]
        for name in descriptor.inputNames {
            guard case .ndArray(let array) = descriptor.inputDescriptor(of: name) else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Whisper \(functionName) input '\(name)' is not an NDArray")
            }
            inputs[name] = try tensor(
                array,
                name: name,
                category: "input")
        }
        var outputs: [String: WhisperNativeTensorDescriptor] = [:]
        for name in descriptor.outputNames {
            guard case .ndArray(let array) = descriptor.outputDescriptor(of: name) else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Whisper \(functionName) output '\(name)' is not an NDArray")
            }
            outputs[name] = try tensor(
                array,
                name: name,
                category: "output")
        }
        var states: [String: WhisperNativeTensorDescriptor] = [:]
        for name in descriptor.stateNames {
            guard case .ndArray(let array) = descriptor.stateDescriptor(of: name) else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Whisper \(functionName) state '\(name)' is not an NDArray")
            }
            states[name] = try tensor(
                array,
                name: name,
                category: "state")
        }
        return WhisperNativeFunctionDescriptor(
            inputs: inputs,
            outputs: outputs,
            states: states)
    }

    private static func validateBridgeRelationships(
        encode: InferenceFunctionDescriptor,
        load: InferenceFunctionDescriptor,
        decode: InferenceFunctionDescriptor
    ) throws {
        try WhisperNativeContract.validateExactRelationship(
            try output("cross_key_payload", from: encode),
            matches: try input("cross_key_payload", from: load),
            relationship: "encode cross_key_payload -> load_cross_kv input")
        try WhisperNativeContract.validateExactRelationship(
            try output("cross_value_payload", from: encode),
            matches: try input("cross_value_payload", from: load),
            relationship: "encode cross_value_payload -> load_cross_kv input")
        for stateName in ["cross_key_cache", "cross_value_cache", "cross_ready"] {
            try WhisperNativeContract.validateExactRelationship(
                try state(stateName, from: load),
                matches: try state(stateName, from: decode),
                relationship: "load_cross_kv \(stateName) -> decode_step state")
        }
    }

    private static func input(
        _ name: String,
        from descriptor: InferenceFunctionDescriptor
    ) throws -> NDArrayDescriptor {
        guard case .ndArray(let value) = descriptor.inputDescriptor(of: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Whisper input '\(name)' is not an NDArray")
        }
        return value
    }

    private static func output(
        _ name: String,
        from descriptor: InferenceFunctionDescriptor
    ) throws -> NDArrayDescriptor {
        guard case .ndArray(let value) = descriptor.outputDescriptor(of: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Whisper output '\(name)' is not an NDArray")
        }
        return value
    }

    private static func state(
        _ name: String,
        from descriptor: InferenceFunctionDescriptor
    ) throws -> NDArrayDescriptor {
        guard case .ndArray(let value) = descriptor.stateDescriptor(of: name) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Whisper state '\(name)' is not an NDArray")
        }
        return value
    }
}

struct WhisperCoreAISessionResources {
    var crossKeyCache: NDArray?
    var crossValueCache: NDArray?
    var selfKeyCache: NDArray?
    var selfValueCache: NDArray?
    var position: NDArray?
    var crossReady: NDArray?
    private(set) var crossKeyPayload: NDArray?
    private(set) var crossValuePayload: NDArray?

    init(
        crossKeyCache: NDArray,
        crossValueCache: NDArray,
        selfKeyCache: NDArray,
        selfValueCache: NDArray,
        position: NDArray,
        crossReady: NDArray
    ) {
        self.crossKeyCache = crossKeyCache
        self.crossValueCache = crossValueCache
        self.selfKeyCache = selfKeyCache
        self.selfValueCache = selfValueCache
        self.position = position
        self.crossReady = crossReady
    }

    mutating func installEncoderPayloads(key: NDArray, value: NDArray) {
        crossKeyPayload = key
        crossValuePayload = value
    }

    mutating func releaseEncoderPayloads() {
        crossKeyPayload = nil
        crossValuePayload = nil
    }

    mutating func dispose() {
        releaseEncoderPayloads()
        crossKeyCache = nil
        crossValueCache = nil
        selfKeyCache = nil
        selfValueCache = nil
        position = nil
        crossReady = nil
    }

    var retainedEncoderPayloadCount: Int {
        [crossKeyPayload, crossValuePayload].compactMap { $0 }.count
    }

    var retainedStateCount: Int {
        [crossKeyCache, crossValueCache, selfKeyCache, selfValueCache, position, crossReady]
            .compactMap { $0 }.count
    }
}

/// Exclusive mutable state for one audio window. `WhisperResidentEngine` is the only owner and its
/// admission gate guarantees that these methods cannot overlap with another session.
private final class WhisperCoreAISession: @unchecked Sendable, WhisperNativeSession {
    private enum Lifecycle {
        case fresh
        case encoded
        case loaded
        case finished
    }

    private let encodeFunction: InferenceFunction
    private let loadFunction: InferenceFunction
    private let decodeFunction: InferenceFunction
    private let inputFeaturesDescriptor: NDArrayDescriptor
    private let crossKeyPayloadDescriptor: NDArrayDescriptor
    private let crossValuePayloadDescriptor: NDArrayDescriptor
    private let loadStatusDescriptor: NDArrayDescriptor
    private let tokenDescriptor: NDArrayDescriptor
    private let logitsDescriptor: NDArrayDescriptor
    private let decodeStatusDescriptor: NDArrayDescriptor

    private var resources: WhisperCoreAISessionResources
    private var lifecycle = Lifecycle.fresh

    init(
        encodeFunction: InferenceFunction,
        loadFunction: InferenceFunction,
        decodeFunction: InferenceFunction,
        encodeDescriptor: InferenceFunctionDescriptor,
        loadDescriptor: InferenceFunctionDescriptor,
        decodeDescriptor: InferenceFunctionDescriptor
    ) throws {
        self.encodeFunction = encodeFunction
        self.loadFunction = loadFunction
        self.decodeFunction = decodeFunction
        self.inputFeaturesDescriptor = try Self.input(
            "input_features", from: encodeDescriptor)
        self.crossKeyPayloadDescriptor = try Self.output(
            "cross_key_payload", from: encodeDescriptor)
        self.crossValuePayloadDescriptor = try Self.output(
            "cross_value_payload", from: encodeDescriptor)
        self.loadStatusDescriptor = try Self.output("load_status", from: loadDescriptor)
        self.tokenDescriptor = try Self.input("token_id", from: decodeDescriptor)
        self.logitsDescriptor = try Self.output("logits", from: decodeDescriptor)
        self.decodeStatusDescriptor = try Self.output("decode_status", from: decodeDescriptor)

        self.resources = try WhisperCoreAISessionResources(
            crossKeyCache: Self.zeroedState("cross_key_cache", from: decodeDescriptor),
            crossValueCache: Self.zeroedState("cross_value_cache", from: decodeDescriptor),
            selfKeyCache: Self.zeroedState("self_key_cache", from: decodeDescriptor),
            selfValueCache: Self.zeroedState("self_value_cache", from: decodeDescriptor),
            position: Self.zeroedState("position", from: decodeDescriptor),
            crossReady: Self.zeroedState("cross_ready", from: decodeDescriptor))
    }

    func encode(inputFeatures: [Float16]) async throws {
        guard lifecycle == .fresh else {
            throw Self.lifecycleError("encode requires a fresh request state")
        }
        var features = NDArray(descriptor: inputFeaturesDescriptor)
        var featureView = features.mutableView(as: Float16.self)
        featureView.copyElements(fromContentsOf: inputFeatures)

        var keyPayload = NDArray(descriptor: crossKeyPayloadDescriptor)
        var valuePayload = NDArray(descriptor: crossValuePayloadDescriptor)
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&keyPayload, for: "cross_key_payload")
        outputs.insert(&valuePayload, for: "cross_value_payload")
        let noStates = InferenceFunction.MutableViews()
        _ = try await encodeFunction.run(
            inputs: ["input_features": features],
            states: consume noStates,
            outputViews: consume outputs)
        resources.installEncoderPayloads(key: keyPayload, value: valuePayload)
        lifecycle = .encoded
    }

    func loadCrossKV() async throws -> Int32 {
        guard lifecycle == .encoded,
            let crossKeyPayload = resources.crossKeyPayload,
            let crossValuePayload = resources.crossValuePayload,
            var crossKeyState = resources.crossKeyCache,
            var crossValueState = resources.crossValueCache,
            var readyState = resources.crossReady
        else {
            throw Self.lifecycleError("load_cross_kv requires exactly one completed encode")
        }
        var status = NDArray(descriptor: loadStatusDescriptor)
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&status, for: "load_status")
        var states = InferenceFunction.MutableViews()
        states.insert(&crossKeyState, for: "cross_key_cache")
        states.insert(&crossValueState, for: "cross_value_cache")
        states.insert(&readyState, for: "cross_ready")
        _ = try await loadFunction.run(
            inputs: [
                "cross_key_payload": crossKeyPayload,
                "cross_value_payload": crossValuePayload,
            ],
            states: consume states,
            outputViews: consume outputs)
        resources.crossKeyCache = crossKeyState
        resources.crossValueCache = crossValueState
        resources.crossReady = readyState
        let statusValue = Self.int32Scalar(status)
        if statusValue == 1 {
            resources.releaseEncoderPayloads()
            lifecycle = .loaded
        }
        return statusValue
    }

    func step(tokenID: Int32) async throws -> WhisperNativeStepOutput {
        guard lifecycle == .loaded,
            var crossKeyState = resources.crossKeyCache,
            var crossValueState = resources.crossValueCache,
            var selfKeyState = resources.selfKeyCache,
            var selfValueState = resources.selfValueCache,
            var positionState = resources.position,
            var readyState = resources.crossReady
        else {
            throw Self.lifecycleError("decode_step requires loaded cross-KV state")
        }
        var token = NDArray(descriptor: tokenDescriptor)
        var tokenView = token.mutableView(as: Int32.self)
        tokenView.copyElements(fromContentsOf: [tokenID])

        var logits = NDArray(descriptor: logitsDescriptor)
        var status = NDArray(descriptor: decodeStatusDescriptor)
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&logits, for: "logits")
        outputs.insert(&status, for: "decode_status")
        var states = InferenceFunction.MutableViews()
        states.insert(&crossKeyState, for: "cross_key_cache")
        states.insert(&crossValueState, for: "cross_value_cache")
        states.insert(&selfKeyState, for: "self_key_cache")
        states.insert(&selfValueState, for: "self_value_cache")
        states.insert(&positionState, for: "position")
        states.insert(&readyState, for: "cross_ready")
        _ = try await decodeFunction.run(
            inputs: ["token_id": token],
            states: consume states,
            outputViews: consume outputs)
        resources.crossKeyCache = crossKeyState
        resources.crossValueCache = crossValueState
        resources.selfKeyCache = selfKeyState
        resources.selfValueCache = selfValueState
        resources.position = positionState
        resources.crossReady = readyState
        let values = logits.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
            let vocabulary = shape[shape.count - 1]
            let stride = strides[strides.count - 1]
            return (0..<vocabulary).map { Float(pointer[$0 * stride]) }
        }
        return WhisperNativeStepOutput(
            status: Self.int32Scalar(status),
            logits: values)
    }

    func finish() async {
        resources.dispose()
        lifecycle = .finished
    }

    private static func input(
        _ name: String,
        from descriptor: InferenceFunctionDescriptor
    ) throws -> NDArrayDescriptor {
        guard case .ndArray(let value) = descriptor.inputDescriptor(of: name) else {
            throw lifecycleError("input '\(name)' is not an NDArray")
        }
        return value
    }

    private static func output(
        _ name: String,
        from descriptor: InferenceFunctionDescriptor
    ) throws -> NDArrayDescriptor {
        guard case .ndArray(let value) = descriptor.outputDescriptor(of: name) else {
            throw lifecycleError("output '\(name)' is not an NDArray")
        }
        return value
    }

    private static func zeroedState(
        _ name: String,
        from descriptor: InferenceFunctionDescriptor
    ) throws -> NDArray {
        guard case .ndArray(let value) = descriptor.stateDescriptor(of: name) else {
            throw lifecycleError("state '\(name)' is not an NDArray")
        }
        var array = NDArray(descriptor: value)
        let count = value.shape.reduce(1, *)
        switch value.scalarType {
        case .float16:
            var view = array.mutableView(as: Float16.self)
            try WhisperNativeContract.requirePackedStateStorage(
                isContiguous: view.isContiguous,
                stateName: name)
            view.withUnsafeMutablePointer { pointer, _, _ in
                for index in 0..<count { pointer[index] = 0 }
            }
        case .int32:
            var view = array.mutableView(as: Int32.self)
            try WhisperNativeContract.requirePackedStateStorage(
                isContiguous: view.isContiguous,
                stateName: name)
            view.withUnsafeMutablePointer { pointer, _, _ in
                for index in 0..<count { pointer[index] = 0 }
            }
        default:
            throw lifecycleError("state '\(name)' has unsupported scalar type")
        }
        return array
    }

    private static func int32Scalar(_ array: NDArray) -> Int32 {
        array.view(as: Int32.self).withUnsafePointer { pointer, _, _ in pointer[0] }
    }

    private static func lifecycleError(_ message: String) -> CoreAIPipeline.RuntimeError {
        .modelContract("Whisper native v2: \(message)")
    }
}

final class WhisperTokenizerDecoder: @unchecked Sendable, WhisperTextDecoding {
    private let tokenizer: any Tokenizer

    static func load(directory: URL) async throws -> WhisperTokenizerDecoder {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return WhisperTokenizerDecoder(tokenizer: tokenizer)
    }

    private init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    func decode(textTokenIDs: [Int32]) -> String {
        tokenizer.decode(tokens: textTokenIDs.map(Int.init))
    }
}

#endif
