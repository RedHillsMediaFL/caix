import Foundation

enum Gemma4MTPScalarType: String, Sendable, Equatable {
  case float16
  case int32
}

struct Gemma4MTPTensorDescriptor: Sendable, Equatable {
  var scalarType: Gemma4MTPScalarType
  var shape: [Int]
}

struct Gemma4MTPFunctionDescriptor: Sendable, Equatable {
  var inputs: [String: Gemma4MTPTensorDescriptor]
  var outputs: [String: Gemma4MTPTensorDescriptor]
  var states: [String: Gemma4MTPTensorDescriptor]
}

enum Gemma4MTPHostTensor: Sendable, Equatable {
  case float16(shape: [Int], values: [Float16])
  case int32(shape: [Int], values: [Int32])

  var scalarType: Gemma4MTPScalarType {
    switch self {
    case .float16: return .float16
    case .int32: return .int32
    }
  }

  var shape: [Int] {
    switch self {
    case .float16(let shape, _), .int32(let shape, _): return shape
    }
  }

  var valueCount: Int {
    switch self {
    case .float16(_, let values): return values.count
    case .int32(_, let values): return values.count
    }
  }
}

struct Gemma4MTPProposalRequest: Sendable, Equatable {
  var tokenID: Int32
  var hidden: Gemma4MTPHostTensor
  var positionID: Int32
  var kFull: Gemma4MTPHostTensor
  var vFull: Gemma4MTPHostTensor
  var kSliding: Gemma4MTPHostTensor
  var vSliding: Gemma4MTPHostTensor
  var kvLength: Int32
}

struct Gemma4MTPProposalResult: Sendable, Equatable {
  var proposedToken: Int32
  var nextHidden: Gemma4MTPHostTensor
}

enum Gemma4MTPNativeContract {
  enum ContractError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalid(String)

    var description: String {
      switch self {
      case .invalid(let reason):
        return "invalid native Gemma 4 MTP assistant contract: \(reason)"
      }
    }
  }

  static let vocabularySize = 262_144
  static let backboneHiddenSize = 2_816
  static let cacheCapacity = 4_096
  static let maxContextLength = cacheCapacity
  static let slidingWindow = 1_024
  static let fullKVHeads = 2
  static let fullHeadDimension = 512
  static let slidingKVHeads = 8
  static let slidingHeadDimension = 256

  /// Stateless draft inputs for a given geometry (`.gemma26B` reproduces the fixed July ABI).
  /// The draft shares the target backbone width and full/sliding KV head geometry; its own
  /// staging tensors are pinned to the target `cacheCapacity`/`slidingWindow`.
  static func expectedInputs(
    _ geometry: Gemma4EagleGeometry
  ) -> [String: Gemma4MTPTensorDescriptor] {
    [
      "token_id": .init(scalarType: .int32, shape: [1, 1]),
      "hidden": .init(scalarType: .float16, shape: [1, 1, geometry.backboneHiddenSize]),
      "position_ids": .init(scalarType: .int32, shape: [1, 1]),
      "k_full": .init(
        scalarType: .float16,
        shape: [1, geometry.fullKVHeads, geometry.cacheCapacity, geometry.fullHeadDimension]),
      "v_full": .init(
        scalarType: .float16,
        shape: [1, geometry.fullKVHeads, geometry.cacheCapacity, geometry.fullHeadDimension]),
      "k_sliding": .init(
        scalarType: .float16,
        shape: [1, geometry.slidingKVHeads, geometry.slidingWindow, geometry.slidingHeadDimension]),
      "v_sliding": .init(
        scalarType: .float16,
        shape: [1, geometry.slidingKVHeads, geometry.slidingWindow, geometry.slidingHeadDimension]),
      "kv_length": .init(scalarType: .int32, shape: [1]),
    ]
  }

  static func expectedOutputs(
    _ geometry: Gemma4EagleGeometry
  ) -> [String: Gemma4MTPTensorDescriptor] {
    [
      "logits": .init(scalarType: .float16, shape: [1, 1, geometry.vocabularySize]),
      "next_hidden": .init(
        scalarType: .float16,
        shape: [1, 1, geometry.backboneHiddenSize]),
    ]
  }

  static func expectedUnrolledOutputs(
    _ geometry: Gemma4EagleGeometry
  ) -> [String: Gemma4MTPTensorDescriptor] {
    [
      "draft_tokens": .init(
        scalarType: .int32,
        shape: [1, geometry.draftTokens])
    ]
  }

  static func validateModel(
    assetURL: URL,
    functionNames: [String],
    function: Gemma4MTPFunctionDescriptor,
    geometry: Gemma4EagleGeometry = .gemma26B
  ) throws {
    try validateAssetURL(assetURL)
    guard functionNames == ["main"] else {
      throw ContractError.invalid(
        "entrypoints must be exactly [\"main\"]; got \(functionNames.sorted())")
    }
    try validate(function, geometry: geometry)
  }

  static func validateAssetURL(_ assetURL: URL) throws {
    guard assetURL.pathExtension.lowercased() == "aimodel" else {
      throw ContractError.invalid(
        "assistant asset must be one .aimodel; got \(assetURL.lastPathComponent)")
    }
  }

  static func validate(
    _ function: Gemma4MTPFunctionDescriptor,
    geometry: Gemma4EagleGeometry = .gemma26B
  ) throws {
    let inputs = expectedInputs(geometry)
    let outputs = expectedOutputs(geometry)
    guard Set(function.inputs.keys) == Set(inputs.keys) else {
      throw ContractError.invalid(
        "inputs must be exactly \(inputs.keys.sorted()); "
          + "got \(function.inputs.keys.sorted())")
    }
    guard Set(function.outputs.keys) == Set(outputs.keys) else {
      throw ContractError.invalid(
        "outputs must be exactly \(outputs.keys.sorted()); "
          + "got \(function.outputs.keys.sorted())")
    }
    guard function.states.isEmpty else {
      throw ContractError.invalid(
        "states must be empty; got \(function.states.keys.sorted())")
    }
    try validateDescriptors(function.inputs, expected: inputs, category: "input")
    try validateDescriptors(function.outputs, expected: outputs, category: "output")
  }

  static func validateUnrolled(
    _ function: Gemma4MTPFunctionDescriptor,
    geometry: Gemma4EagleGeometry = .gemma26B
  ) throws {
    let inputs = expectedInputs(geometry)
    let unrolledOutputs = expectedUnrolledOutputs(geometry)
    guard Set(function.inputs.keys) == Set(inputs.keys) else {
      throw ContractError.invalid(
        "unrolled inputs must be exactly \(inputs.keys.sorted()); "
          + "got \(function.inputs.keys.sorted())")
    }
    guard Set(function.outputs.keys) == Set(unrolledOutputs.keys) else {
      throw ContractError.invalid(
        "unrolled outputs must be exactly \(unrolledOutputs.keys.sorted()); "
          + "got \(function.outputs.keys.sorted())")
    }
    guard function.states.isEmpty else {
      throw ContractError.invalid(
        "unrolled states must be empty; got \(function.states.keys.sorted())")
    }
    try validateDescriptors(
      function.inputs,
      expected: inputs,
      category: "unrolled input")
    try validateDescriptors(
      function.outputs,
      expected: unrolledOutputs,
      category: "unrolled output")
  }

  static func validate(_ request: Gemma4MTPProposalRequest) throws {
    guard request.tokenID >= 0, Int(request.tokenID) < vocabularySize else {
      throw ContractError.invalid(
        "token_id must be in 0..<\(vocabularySize); got \(request.tokenID)")
    }
    try validateRuntimeInvocation(
      positionID: request.positionID,
      kvLength: request.kvLength,
      hiddenShape: request.hidden.shape,
      kFullShape: request.kFull.shape,
      vFullShape: request.vFull.shape,
      kSlidingShape: request.kSliding.shape,
      vSlidingShape: request.vSliding.shape)

    try validateHostTensor(
      request.hidden,
      name: "hidden",
      expectedType: .float16,
      expectedShape: [1, 1, backboneHiddenSize])
    try validateHostTensor(
      request.kFull,
      name: "k_full",
      expectedType: .float16,
      expectedShape: [1, fullKVHeads, cacheCapacity, fullHeadDimension])
    try validateHostTensor(
      request.vFull,
      name: "v_full",
      expectedType: .float16,
      expectedShape: [1, fullKVHeads, cacheCapacity, fullHeadDimension])
    try validateHostTensor(
      request.kSliding,
      name: "k_sliding",
      expectedType: .float16,
      expectedShape: [1, slidingKVHeads, slidingWindow, slidingHeadDimension])
    try validateHostTensor(
      request.vSliding,
      name: "v_sliding",
      expectedType: .float16,
      expectedShape: [1, slidingKVHeads, slidingWindow, slidingHeadDimension])
  }

  static func validateRuntimeInvocation(
    positionID: Int32,
    kvLength: Int32,
    hiddenShape: [Int],
    kFullShape: [Int],
    vFullShape: [Int],
    kSlidingShape: [Int],
    vSlidingShape: [Int],
    geometry: Gemma4EagleGeometry = .gemma26B
  ) throws {
    let maxContext = geometry.cacheCapacity
    guard positionID >= 0, Int(positionID) < maxContext else {
      throw ContractError.invalid(
        "position_ids must be in 0..<\(maxContext); got \(positionID)")
    }
    guard kvLength > 0, Int(kvLength) < geometry.cacheCapacity else {
      throw ContractError.invalid(
        "kv_length must be in 1..<\(geometry.cacheCapacity); got \(kvLength)")
    }
    guard positionID == kvLength else {
      throw ContractError.invalid(
        "position_ids must equal kv_length for the next absolute token; "
          + "got \(positionID) and \(kvLength)")
    }

    let expectedShapes: [(String, [Int], [Int])] = [
      ("hidden", hiddenShape, [1, 1, geometry.backboneHiddenSize]),
      ("k_full", kFullShape,
        [1, geometry.fullKVHeads, geometry.cacheCapacity, geometry.fullHeadDimension]),
      ("v_full", vFullShape,
        [1, geometry.fullKVHeads, geometry.cacheCapacity, geometry.fullHeadDimension]),
      ("k_sliding", kSlidingShape,
        [1, geometry.slidingKVHeads, geometry.slidingWindow, geometry.slidingHeadDimension]),
      ("v_sliding", vSlidingShape,
        [1, geometry.slidingKVHeads, geometry.slidingWindow, geometry.slidingHeadDimension]),
    ]
    for (name, actual, expected) in expectedShapes {
      guard actual == expected else {
        throw ContractError.invalid(
          "\(name) must have static runtime shape \(expected); got \(actual)")
      }
    }
  }

  static func validate(_ outputs: Gemma4MTPModelOutputs) throws {
    try validateHostTensor(
      outputs.logits,
      name: "logits",
      expectedType: .float16,
      expectedShape: [1, 1, vocabularySize])
    try validateHostTensor(
      outputs.nextHidden,
      name: "next_hidden",
      expectedType: .float16,
      expectedShape: [1, 1, backboneHiddenSize])
  }

  private static func validateDescriptors(
    _ actual: [String: Gemma4MTPTensorDescriptor],
    expected: [String: Gemma4MTPTensorDescriptor],
    category: String
  ) throws {
    for name in expected.keys.sorted() {
      guard actual[name] == expected[name] else {
        throw ContractError.invalid(
          "\(category) \(name) must be \(expected[name]!); "
            + "got \(String(describing: actual[name]))")
      }
    }
  }

  private static func validateHostTensor(
    _ tensor: Gemma4MTPHostTensor,
    name: String,
    expectedType: Gemma4MTPScalarType,
    expectedShape: [Int]
  ) throws {
    guard tensor.scalarType == expectedType, tensor.shape == expectedShape else {
      throw ContractError.invalid(
        "\(name) must be \(expectedType.rawValue) \(expectedShape); "
          + "got \(tensor.scalarType.rawValue) \(tensor.shape)")
    }
    try validateStorage(tensor, name: name)
  }

  private static func validateStorage(
    _ tensor: Gemma4MTPHostTensor,
    name: String
  ) throws {
    var expectedCount = 1
    for dimension in tensor.shape {
      let (product, overflow) = expectedCount.multipliedReportingOverflow(by: dimension)
      guard dimension > 0, !overflow else {
        throw ContractError.invalid("\(name) shape cannot be represented safely")
      }
      expectedCount = product
    }
    guard tensor.valueCount == expectedCount else {
      throw ContractError.invalid(
        "\(name) storage must contain \(expectedCount) values; "
          + "got \(tensor.valueCount)")
    }
  }
}
