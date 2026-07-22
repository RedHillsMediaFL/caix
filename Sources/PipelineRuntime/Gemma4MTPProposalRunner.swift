import Foundation

struct Gemma4MTPModelOutputs: Sendable, Equatable {
  var logits: Gemma4MTPHostTensor
  var nextHidden: Gemma4MTPHostTensor
}

/// Package-testable orchestration shared by the real Core AI adapter and contract-only tests.
/// The injected executor is the only operation allowed to cross the inference boundary.
struct Gemma4MTPGreedyProposalRunner {
  typealias Execute = (Gemma4MTPProposalRequest) async throws -> Gemma4MTPModelOutputs

  private let execute: Execute

  init(execute: @escaping Execute) {
    self.execute = execute
  }

  func propose(_ request: Gemma4MTPProposalRequest) async throws -> Gemma4MTPProposalResult {
    try Gemma4MTPNativeContract.validate(request)
    let outputs = try await execute(request)
    try Gemma4MTPNativeContract.validate(outputs)

    guard case .float16(_, let logits) = outputs.logits else {
      throw Gemma4MTPNativeContract.ContractError.invalid(
        "logits must use FP16 host storage")
    }
    guard logits.allSatisfy(\.isFinite) else {
      throw Gemma4MTPNativeContract.ContractError.invalid(
        "logits must contain only finite values")
    }
    guard case .float16(_, let nextHidden) = outputs.nextHidden,
      nextHidden.allSatisfy(\.isFinite)
    else {
      throw Gemma4MTPNativeContract.ContractError.invalid(
        "next_hidden must contain only finite FP16 values")
    }

    var proposedToken = 0
    var bestLogit = logits[0]
    for index in 1..<logits.count where logits[index] > bestLogit {
      proposedToken = index
      bestLogit = logits[index]
    }
    return Gemma4MTPProposalResult(
      proposedToken: Int32(proposedToken),
      nextHidden: outputs.nextHidden)
  }
}
