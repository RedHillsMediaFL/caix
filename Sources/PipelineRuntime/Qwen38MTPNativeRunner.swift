#if COREAI_RUNTIME

import CoreAI
import Foundation

/// Resident executor for the one-layer Qwen3.8 MTP sidecar.
///
/// Its K/V state is purely positional and can be rolled back by moving the valid cursor; stale
/// suffix slots are overwritten before they can be read. The target's recurrent fixed state is
/// handled separately by `LLMEngine`.
final class Qwen38MTPNativeRunner {
    struct Output {
        let greedyTokens: [Int32]
        let hiddenRows: [[Float16]]
    }

    private let function: InferenceFunction
    private let inputIDsDescriptor: NDArrayDescriptor
    private let hiddenDescriptor: NDArrayDescriptor
    private let positionIDsDescriptor: NDArrayDescriptor
    private let logitsDescriptor: NDArrayDescriptor
    private let outputHiddenDescriptor: NDArrayDescriptor
    private let keyDescriptor: NDArrayDescriptor
    private let valueDescriptor: NDArrayDescriptor
    private var keyCache: NDArray
    private var valueCache: NDArray
    private(set) var processedTokenCount = 0

    static func load(aimodelURL: URL, verbose: Bool = false) async throws
        -> Qwen38MTPNativeRunner
    {
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model = try await AIModel.specialize(
            contentsOf: aimodelURL,
            options: options,
            cache: .default,
            cachePolicy: .persistent)
        guard let descriptor = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Qwen3.8 MTP sidecar has no main function")
        }
        guard Set(descriptor.inputNames) == ["input_ids", "hidden_states", "position_ids"],
            Set(descriptor.outputNames) == ["logits", "mtp_hidden_states"],
            descriptor.stateNames == ["keyCache", "valueCache"]
                || Set(descriptor.stateNames) == ["keyCache", "valueCache"]
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "invalid Qwen3.8 MTP ABI: inputs=\(descriptor.inputNames), "
                    + "outputs=\(descriptor.outputNames), states=\(descriptor.stateNames)")
        }
        guard let function = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Qwen3.8 MTP main function could not be loaded")
        }
        let runner = try Qwen38MTPNativeRunner(function: function, descriptor: descriptor)
        if verbose {
            FileHandle.standardError.write(Data(
                "[qwen-mtp] resident sidecar ready: \(aimodelURL.lastPathComponent)\n".utf8))
        }
        return runner
    }

    private init(function: InferenceFunction, descriptor: InferenceFunctionDescriptor) throws {
        func input(_ name: String) throws -> NDArrayDescriptor {
            guard case .ndArray(let value) = descriptor.inputDescriptor(of: name) else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Qwen3.8 MTP input '\(name)' is not an NDArray")
            }
            return value
        }
        func output(_ name: String) throws -> NDArrayDescriptor {
            guard case .ndArray(let value) = descriptor.outputDescriptor(of: name) else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Qwen3.8 MTP output '\(name)' is not an NDArray")
            }
            return value
        }
        func state(_ name: String) throws -> NDArrayDescriptor {
            guard case .ndArray(let value) = descriptor.stateDescriptor(of: name) else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Qwen3.8 MTP state '\(name)' is not an NDArray")
            }
            return value
        }

        let inputIDs = try input("input_ids")
        let hidden = try input("hidden_states")
        let positions = try input("position_ids")
        let logits = try output("logits")
        let outputHidden = try output("mtp_hidden_states")
        let key = try state("keyCache")
        let value = try state("valueCache")
        guard inputIDs.scalarType == .int32,
            positions.scalarType == .int32,
            hidden.scalarType == .float16,
            logits.scalarType == .float16,
            outputHidden.scalarType == .float16,
            key.scalarType == .float16,
            value.scalarType == .float16,
            hidden.shape.last == 5_120,
            logits.shape.last == 248_320,
            outputHidden.shape.last == 5_120,
            key.shape.count == 5, value.shape.count == 5,
            key.shape[2] == 4, value.shape[2] == 4,
            key.shape[4] == 256, value.shape[4] == 256
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Qwen3.8 MTP tensor geometry or dtype mismatch")
        }

        self.function = function
        self.inputIDsDescriptor = inputIDs
        self.hiddenDescriptor = hidden
        self.positionIDsDescriptor = positions
        self.logitsDescriptor = logits
        self.outputHiddenDescriptor = outputHidden
        self.keyDescriptor = key
        self.valueDescriptor = value
        self.keyCache = NDArray(
            descriptor: key.resolvingDynamicDimensions(key.shape.map { $0 < 0 ? 1 : $0 }))
        self.valueCache = NDArray(
            descriptor: value.resolvingDynamicDimensions(value.shape.map { $0 < 0 ? 1 : $0 }))
    }

    func reset(capacity: Int) {
        let keyShape = keyDescriptor.shape.map { $0 < 0 ? capacity : $0 }
        let valueShape = valueDescriptor.shape.map { $0 < 0 ? capacity : $0 }
        keyCache = NDArray(descriptor: keyDescriptor.resolvingDynamicDimensions(keyShape))
        valueCache = NDArray(descriptor: valueDescriptor.resolvingDynamicDimensions(valueShape))
        Self.zero(&keyCache)
        Self.zero(&valueCache)
        processedTokenCount = 0
    }

    func rollback(to count: Int) {
        precondition(count >= 0 && count <= processedTokenCount)
        processedTokenCount = count
    }

    func forward(tokens: [Int32], hiddenRows: [[Float16]]) async throws -> Output {
        let count = tokens.count
        precondition(count > 0 && hiddenRows.count == count)
        precondition(hiddenRows.allSatisfy { $0.count == 5_120 })

        var inputIDs = NDArray(
            descriptor: inputIDsDescriptor.resolvingDynamicDimensions([1, count]))
        var inputView = inputIDs.mutableView(as: Int32.self)
        inputView.copyElements(fromContentsOf: tokens)
        var hidden = NDArray(
            descriptor: hiddenDescriptor.resolvingDynamicDimensions([1, count, 5_120]))
        var hiddenView = hidden.mutableView(as: Float16.self)
        hiddenView.copyElements(fromContentsOf: hiddenRows.joined())

        let total = processedTokenCount + count
        let positions = (0..<total).map(Int32.init)
        var positionIDs = NDArray(
            descriptor: positionIDsDescriptor.resolvingDynamicDimensions([1, total]))
        var positionView = positionIDs.mutableView(as: Int32.self)
        positionView.copyElements(fromContentsOf: positions)

        var logits = NDArray(
            descriptor: logitsDescriptor.resolvingDynamicDimensions([1, 1, 248_320]))
        var nextHidden = NDArray(
            descriptor: outputHiddenDescriptor.resolvingDynamicDimensions([1, 1, 5_120]))
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&logits, for: "logits")
        outputs.insert(&nextHidden, for: "mtp_hidden_states")
        var states = InferenceFunction.MutableViews()
        states.insert(&keyCache, for: "keyCache")
        states.insert(&valueCache, for: "valueCache")
        _ = try await function.run(
            inputs: [
                "input_ids": inputIDs,
                "hidden_states": hidden,
                "position_ids": positionIDs,
            ],
            states: consume states,
            outputViews: consume outputs)
        processedTokenCount = total
        return Output(
            greedyTokens: Self.argmaxRows(logits, width: 248_320),
            hiddenRows: Self.rows(nextHidden, width: 5_120))
    }

    private static func zero(_ array: inout NDArray) {
        let count = array.shape.reduce(1, *)
        array.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in
            pointer.initialize(repeating: 0, count: count)
        }
    }

    private static func argmaxRows(_ array: NDArray, width: Int) -> [Int32] {
        array.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
            let rows = shape[shape.count - 2]
            let rowStride = strides[strides.count - 2]
            let columnStride = strides[strides.count - 1]
            return (0..<rows).map { row in
                let base = row * rowStride
                var best = 0
                var bestValue = pointer[base]
                for column in 1..<width {
                    let value = pointer[base + column * columnStride]
                    if value > bestValue { best = column; bestValue = value }
                }
                return Int32(best)
            }
        }
    }

    private static func rows(_ array: NDArray, width: Int) -> [[Float16]] {
        array.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
            let rows = shape[shape.count - 2]
            let rowStride = strides[strides.count - 2]
            let columnStride = strides[strides.count - 1]
            return (0..<rows).map { row in
                let base = row * rowStride
                return (0..<width).map { pointer[base + $0 * columnStride] }
            }
        }
    }
}

#endif
