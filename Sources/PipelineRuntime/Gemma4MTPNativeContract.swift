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
  static let backboneHiddenSize = 5_376
  static let maxContextLength = 262_144
  static let slidingWindow = 1_024
  static let fullKVHeads = 4
  static let fullHeadDimension = 512
  static let slidingKVHeads = 16
  static let slidingHeadDimension = 256

  private static let expectedInputs: [String: Gemma4MTPTensorDescriptor] = [
    "token_id": .init(scalarType: .int32, shape: [1, 1]),
    "hidden": .init(scalarType: .float16, shape: [1, 1, backboneHiddenSize]),
    "position_ids": .init(scalarType: .int32, shape: [1, 1]),
    "k_full": .init(
      scalarType: .float16,
      shape: [1, fullKVHeads, -1, fullHeadDimension]),
    "v_full": .init(
      scalarType: .float16,
      shape: [1, fullKVHeads, -1, fullHeadDimension]),
    "k_sliding": .init(
      scalarType: .float16,
      shape: [1, slidingKVHeads, -1, slidingHeadDimension]),
    "v_sliding": .init(
      scalarType: .float16,
      shape: [1, slidingKVHeads, -1, slidingHeadDimension]),
  ]

  private static let expectedOutputs: [String: Gemma4MTPTensorDescriptor] = [
    "logits": .init(scalarType: .float16, shape: [1, 1, vocabularySize]),
    "next_hidden": .init(
      scalarType: .float16,
      shape: [1, 1, backboneHiddenSize]),
  ]

  static func validateModel(
    assetURL: URL,
    functionNames: [String],
    function: Gemma4MTPFunctionDescriptor
  ) throws {
    try validateAssetURL(assetURL)
    guard functionNames == ["main"] else {
      throw ContractError.invalid(
        "entrypoints must be exactly [\"main\"]; got \(functionNames.sorted())")
    }
    try validate(function)
  }

  static func validateAssetURL(_ assetURL: URL) throws {
    guard assetURL.pathExtension.lowercased() == "aimodel" else {
      throw ContractError.invalid(
        "assistant asset must be one .aimodel; got \(assetURL.lastPathComponent)")
    }
  }

  static func validate(_ function: Gemma4MTPFunctionDescriptor) throws {
    guard Set(function.inputs.keys) == Set(expectedInputs.keys) else {
      throw ContractError.invalid(
        "inputs must be exactly \(expectedInputs.keys.sorted()); "
          + "got \(function.inputs.keys.sorted())")
    }
    guard Set(function.outputs.keys) == Set(expectedOutputs.keys) else {
      throw ContractError.invalid(
        "outputs must be exactly \(expectedOutputs.keys.sorted()); "
          + "got \(function.outputs.keys.sorted())")
    }
    guard function.states.isEmpty else {
      throw ContractError.invalid(
        "states must be empty; got \(function.states.keys.sorted())")
    }
    try validateDescriptors(function.inputs, expected: expectedInputs, category: "input")
    try validateDescriptors(function.outputs, expected: expectedOutputs, category: "output")
  }

  static func validate(_ request: Gemma4MTPProposalRequest) throws {
    guard request.tokenID >= 0, Int(request.tokenID) < vocabularySize else {
      throw ContractError.invalid(
        "token_id must be in 0..<\(vocabularySize); got \(request.tokenID)")
    }
    guard request.positionID >= 0, Int(request.positionID) < maxContextLength else {
      throw ContractError.invalid(
        "position_ids must be in 0..<\(maxContextLength); got \(request.positionID)")
    }

    try validateHostTensor(
      request.hidden,
      name: "hidden",
      expectedType: .float16,
      expectedShape: [1, 1, backboneHiddenSize])
    let fullKeyLength = try validateKV(
      request.kFull,
      name: "k_full",
      heads: fullKVHeads,
      headDimension: fullHeadDimension,
      maximumLength: maxContextLength)
    let fullValueLength = try validateKV(
      request.vFull,
      name: "v_full",
      heads: fullKVHeads,
      headDimension: fullHeadDimension,
      maximumLength: maxContextLength)
    guard fullKeyLength == fullValueLength else {
      throw ContractError.invalid(
        "full key/value sequence lengths must match; got "
          + "\(fullKeyLength) and \(fullValueLength)")
    }

    let slidingKeyLength = try validateKV(
      request.kSliding,
      name: "k_sliding",
      heads: slidingKVHeads,
      headDimension: slidingHeadDimension,
      maximumLength: slidingWindow)
    let slidingValueLength = try validateKV(
      request.vSliding,
      name: "v_sliding",
      heads: slidingKVHeads,
      headDimension: slidingHeadDimension,
      maximumLength: slidingWindow)
    guard slidingKeyLength == slidingValueLength else {
      throw ContractError.invalid(
        "sliding key/value sequence lengths must match; got "
          + "\(slidingKeyLength) and \(slidingValueLength)")
    }
    let expectedSlidingLength = min(fullKeyLength, slidingWindow)
    guard slidingKeyLength == expectedSlidingLength else {
      throw ContractError.invalid(
        "sliding sequence length must equal min(full length, \(slidingWindow)); "
          + "expected \(expectedSlidingLength), got \(slidingKeyLength)")
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

  private static func validateKV(
    _ tensor: Gemma4MTPHostTensor,
    name: String,
    heads: Int,
    headDimension: Int,
    maximumLength: Int
  ) throws -> Int {
    guard tensor.scalarType == .float16,
      tensor.shape.count == 4,
      tensor.shape[0] == 1,
      tensor.shape[1] == heads,
      tensor.shape[2] > 0,
      tensor.shape[2] <= maximumLength,
      tensor.shape[3] == headDimension
    else {
      throw ContractError.invalid(
        "\(name) must be FP16 [1, \(heads), 1...\(maximumLength), "
          + "\(headDimension)]; got \(tensor.scalarType.rawValue) \(tensor.shape)")
    }
    try validateStorage(tensor, name: name)
    return tensor.shape[2]
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
