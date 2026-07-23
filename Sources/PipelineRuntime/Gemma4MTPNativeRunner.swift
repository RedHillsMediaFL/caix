#if COREAI_RUNTIME

  import CoreAI
  import Foundation

  /// A single resident, stateless Gemma 4 MTP assistant `.aimodel`.
  ///
  /// This leaf adapter deliberately owns no target model, cache state, distributed runtime, model
  /// manager, or request loop. `load` validates the actual specialized Core AI descriptor before it
  /// exposes the one-step proposal API.
  final class Gemma4MTPNativeRunner {
    private let model: AIModel
    private let function: InferenceFunction
    private let inputDescriptors: [String: NDArrayDescriptor]
    private let outputDescriptors: [String: NDArrayDescriptor]

    static func load(aimodelURL: URL) async throws -> Gemma4MTPNativeRunner {
      try Gemma4MTPNativeContract.validateAssetURL(aimodelURL)

      var options = LLMEngine.eagleSpecializationOptions()
      options.expectFrequentReshapes = false
      let model = try await AIModel(contentsOf: aimodelURL, options: options)

      guard let runtimeDescriptor = model.functionDescriptor(for: "main") else {
        throw Gemma4MTPNativeContract.ContractError.invalid(
          "required main entrypoint is missing; got \(model.functionNames.sorted())")
      }
      let projectedDescriptor = try project(runtimeDescriptor)
      try Gemma4MTPNativeContract.validateModel(
        assetURL: aimodelURL,
        functionNames: model.functionNames,
        function: projectedDescriptor)

      guard let function = try model.loadFunction(named: "main") else {
        throw Gemma4MTPNativeContract.ContractError.invalid(
          "main passed descriptor validation but could not be loaded")
      }
      return try Gemma4MTPNativeRunner(
        model: model,
        function: function,
        descriptor: runtimeDescriptor)
    }

    private init(
      model: AIModel,
      function: InferenceFunction,
      descriptor: InferenceFunctionDescriptor
    ) throws {
      var inputs: [String: NDArrayDescriptor] = [:]
      for name in descriptor.inputNames {
        guard case .ndArray(let array) = descriptor.inputDescriptor(of: name) else {
          throw Gemma4MTPNativeContract.ContractError.invalid(
            "main input \(name) is not an NDArray")
        }
        inputs[name] = array
      }
      var outputs: [String: NDArrayDescriptor] = [:]
      for name in descriptor.outputNames {
        guard case .ndArray(let array) = descriptor.outputDescriptor(of: name) else {
          throw Gemma4MTPNativeContract.ContractError.invalid(
            "main output \(name) is not an NDArray")
        }
        outputs[name] = array
      }
      self.model = model
      self.function = function
      self.inputDescriptors = inputs
      self.outputDescriptors = outputs
    }

    /// Executes one Core AI forward and returns its greedy token plus recurrent backbone hidden.
    func propose(_ request: Gemma4MTPProposalRequest) async throws -> Gemma4MTPProposalResult {
      let greedy = Gemma4MTPGreedyProposalRunner { request in
        try await self.execute(request)
      }
      return try await greedy.propose(request)
    }

    private func execute(_ request: Gemma4MTPProposalRequest) async throws
      -> Gemma4MTPModelOutputs
    {
      let tokenID = try makeArray(
        .int32(shape: [1, 1], values: [request.tokenID]),
        named: "token_id")
      let hidden = try makeArray(request.hidden, named: "hidden")
      let positionIDs = try makeArray(
        .int32(shape: [1, 1], values: [request.positionID]),
        named: "position_ids")
      let kFull = try makeArray(request.kFull, named: "k_full")
      let vFull = try makeArray(request.vFull, named: "v_full")
      let kSliding = try makeArray(request.kSliding, named: "k_sliding")
      let vSliding = try makeArray(request.vSliding, named: "v_sliding")

      var logits = try outputArray(
        named: "logits",
        shape: [1, 1, Gemma4MTPNativeContract.vocabularySize])
      var nextHidden = try outputArray(
        named: "next_hidden",
        shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize])
      var outputViews = InferenceFunction.MutableViews()
      outputViews.insert(&logits, for: "logits")
      outputViews.insert(&nextHidden, for: "next_hidden")
      let noStates = InferenceFunction.MutableViews()
      _ = try await function.run(
        inputs: [
          "token_id": tokenID,
          "hidden": hidden,
          "position_ids": positionIDs,
          "k_full": kFull,
          "v_full": vFull,
          "k_sliding": kSliding,
          "v_sliding": vSliding,
        ],
        states: consume noStates,
        outputViews: consume outputViews)

      return Gemma4MTPModelOutputs(
        logits: try copyOutput(
          logits,
          name: "logits",
          width: Gemma4MTPNativeContract.vocabularySize),
        nextHidden: try copyOutput(
          nextHidden,
          name: "next_hidden",
          width: Gemma4MTPNativeContract.backboneHiddenSize))
    }

    private func makeArray(
      _ tensor: Gemma4MTPHostTensor,
      named name: String
    ) throws -> NDArray {
      guard let descriptor = inputDescriptors[name] else {
        throw Gemma4MTPNativeContract.ContractError.invalid(
          "validated main descriptor lost input \(name)")
      }
      var array = NDArray(
        descriptor: descriptor.resolvingDynamicDimensions(tensor.shape))
      switch tensor {
      case .float16(_, let values):
        var view = array.mutableView(as: Float16.self)
        view.copyElements(fromContentsOf: values)
      case .int32(_, let values):
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: values)
      }
      return array
    }

    private func outputArray(named name: String, shape: [Int]) throws -> NDArray {
      guard let descriptor = outputDescriptors[name] else {
        throw Gemma4MTPNativeContract.ContractError.invalid(
          "validated main descriptor lost output \(name)")
      }
      return NDArray(descriptor: descriptor.resolvingDynamicDimensions(shape))
    }

    private func copyOutput(
      _ array: NDArray,
      name: String,
      width: Int
    ) throws -> Gemma4MTPHostTensor {
      guard array.shape == [1, 1, width] else {
        throw Gemma4MTPNativeContract.ContractError.invalid(
          "runtime \(name) must be [1, 1, \(width)]; got \(array.shape)")
      }
      let values = array.view(as: Float16.self).withUnsafePointer { pointer, _, strides in
        let columnStride = strides[strides.count - 1]
        return (0..<width).map { pointer[$0 * columnStride] }
      }
      return .float16(shape: [1, 1, width], values: values)
    }

    private static func project(
      _ descriptor: InferenceFunctionDescriptor
    ) throws -> Gemma4MTPFunctionDescriptor {
      func tensor(
        _ descriptor: NDArrayDescriptor,
        name: String,
        category: String
      ) throws -> Gemma4MTPTensorDescriptor {
        let scalarType: Gemma4MTPScalarType
        switch descriptor.scalarType {
        case .float16: scalarType = .float16
        case .int32: scalarType = .int32
        default:
          throw Gemma4MTPNativeContract.ContractError.invalid(
            "main \(category) \(name) must be int32 or FP16; "
              + "got \(descriptor.scalarType)")
        }
        return Gemma4MTPTensorDescriptor(
          scalarType: scalarType,
          shape: descriptor.shape)
      }

      var inputs: [String: Gemma4MTPTensorDescriptor] = [:]
      for name in descriptor.inputNames {
        guard case .ndArray(let array) = descriptor.inputDescriptor(of: name) else {
          throw Gemma4MTPNativeContract.ContractError.invalid(
            "main input \(name) is not an NDArray")
        }
        inputs[name] = try tensor(array, name: name, category: "input")
      }
      var outputs: [String: Gemma4MTPTensorDescriptor] = [:]
      for name in descriptor.outputNames {
        guard case .ndArray(let array) = descriptor.outputDescriptor(of: name) else {
          throw Gemma4MTPNativeContract.ContractError.invalid(
            "main output \(name) is not an NDArray")
        }
        outputs[name] = try tensor(array, name: name, category: "output")
      }
      var states: [String: Gemma4MTPTensorDescriptor] = [:]
      for name in descriptor.stateNames {
        guard case .ndArray(let array) = descriptor.stateDescriptor(of: name) else {
          throw Gemma4MTPNativeContract.ContractError.invalid(
            "main state \(name) is not an NDArray")
        }
        states[name] = try tensor(array, name: name, category: "state")
      }
      return Gemma4MTPFunctionDescriptor(
        inputs: inputs,
        outputs: outputs,
        states: states)
    }
  }

#endif
