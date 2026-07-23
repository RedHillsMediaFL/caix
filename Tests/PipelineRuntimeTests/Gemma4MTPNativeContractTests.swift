import Foundation
import XCTest

@testable import PipelineRuntime

final class Gemma4MTPNativeContractTests: XCTestCase {
  func testAcceptsExactFixedShape26BAssistantContract() throws {
    XCTAssertNoThrow(try Gemma4MTPNativeContract.validate(Self.validFunction))
  }

  func testAcceptsExactK4UnrolledAssistantContract() throws {
    XCTAssertNoThrow(
      try Gemma4MTPNativeContract.validateUnrolled(Self.validUnrolledFunction))
  }

  func testUnrolledAssistantRejectsNonK4OutputAndMissingKVLength() {
    var function = Self.validUnrolledFunction
    function.outputs["draft_tokens"] = .init(scalarType: .int32, shape: [1, 3])
    assertUnrolledContractRejected(function, contains: "draft_tokens")

    function = Self.validUnrolledFunction
    function.inputs.removeValue(forKey: "kv_length")
    assertUnrolledContractRejected(function, contains: "inputs must be exactly")
  }

  func testModelPreflightAcceptsOneAimodelWithOnlyMain() throws {
    XCTAssertNoThrow(
      try Gemma4MTPNativeContract.validateModel(
        assetURL: URL(fileURLWithPath: "/tmp/gemma4-assistant.aimodel"),
        functionNames: ["main"],
        function: Self.validFunction))
  }

  func testModelPreflightRejectsWrongAssetExtensionAndEntrypoints() {
    XCTAssertThrowsError(
      try Gemma4MTPNativeContract.validateModel(
        assetURL: URL(fileURLWithPath: "/tmp/gemma4-assistant.mlpackage"),
        functionNames: ["main"],
        function: Self.validFunction)
    ) { error in
      XCTAssertTrue(String(describing: error).contains(".aimodel"))
    }

    XCTAssertThrowsError(
      try Gemma4MTPNativeContract.validateModel(
        assetURL: URL(fileURLWithPath: "/tmp/gemma4-assistant.aimodel"),
        functionNames: ["main", "debug"],
        function: Self.validFunction)
    ) { error in
      XCTAssertTrue(String(describing: error).contains("entrypoints"))
    }
  }

  #if COREAI_RUNTIME
    func testNativeRunnerRejectsNonAimodelBeforeSpecialization() async {
      do {
        _ = try await Gemma4MTPNativeRunner.load(
          aimodelURL: URL(fileURLWithPath: "/tmp/not-an-assistant.mlpackage"))
        XCTFail("non-.aimodel URL unexpectedly specialized")
      } catch {
        XCTAssertTrue(String(describing: error).contains(".aimodel"))
      }
    }
  #endif

  func testRejectsInputAliasesAndUnexpectedInputs() {
    var function = Self.validFunction
    function.inputs["input_ids"] = function.inputs.removeValue(forKey: "token_id")
    assertContractRejected(function, contains: "inputs must be exactly")

    function = Self.validFunction
    function.inputs["attention_mask"] = .init(scalarType: .int32, shape: [1, 1])
    assertContractRejected(function, contains: "inputs must be exactly")

    function = Self.validFunction
    function.inputs.removeValue(forKey: "kv_length")
    assertContractRejected(function, contains: "inputs must be exactly")
  }

  func testRejectsOutputAliasesAndAnyStateTensor() {
    var function = Self.validFunction
    function.outputs["hidden"] = function.outputs.removeValue(forKey: "next_hidden")
    assertContractRejected(function, contains: "outputs must be exactly")

    function = Self.validFunction
    function.states["cache"] = .init(scalarType: .float16, shape: [1])
    assertContractRejected(function, contains: "states must be empty")
  }

  func testRejectsWrongIntegerTypesAndRanks() {
    var function = Self.validFunction
    function.inputs["token_id"] = .init(scalarType: .float16, shape: [1, 1])
    assertContractRejected(function, contains: "token_id")

    function = Self.validFunction
    function.inputs["position_ids"] = .init(scalarType: .int32, shape: [1])
    assertContractRejected(function, contains: "position_ids")
  }

  func testRejectsWrongHiddenAndVocabularyGeometry() {
    var function = Self.validFunction
    function.inputs["hidden"] = .init(
      scalarType: .float16,
      shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize - 1])
    assertContractRejected(function, contains: "hidden")

    function = Self.validFunction
    function.outputs["logits"] = .init(
      scalarType: .float16,
      shape: [1, 1, Gemma4MTPNativeContract.vocabularySize - 1])
    assertContractRejected(function, contains: "logits")

    function = Self.validFunction
    function.outputs["next_hidden"] = .init(
      scalarType: .float16,
      shape: [1, Gemma4MTPNativeContract.backboneHiddenSize])
    assertContractRejected(function, contains: "next_hidden")
  }

  func testRejectsWrongFullAndSlidingKVGeometry() {
    var function = Self.validFunction
    function.inputs["k_full"] = .init(
      scalarType: .float16, shape: [1, 2, -1, 512])
    assertContractRejected(function, contains: "k_full")

    function = Self.validFunction
    function.inputs["v_full"] = .init(
      scalarType: .float16, shape: [1, 2, 4_095, 512])
    assertContractRejected(function, contains: "v_full")

    function = Self.validFunction
    function.inputs["k_sliding"] = .init(
      scalarType: .float16, shape: [1, 8, 1_024, 128])
    assertContractRejected(function, contains: "k_sliding")

    function = Self.validFunction
    function.inputs["v_sliding"] = .init(
      scalarType: .int32, shape: [1, 8, 1_024, 256])
    assertContractRejected(function, contains: "v_sliding")
  }

  func testRejectsWrongKVLengthDescriptor() {
    var function = Self.validFunction
    function.inputs["kv_length"] = .init(scalarType: .int32, shape: [1, 1])
    assertContractRejected(function, contains: "kv_length")

    function = Self.validFunction
    function.inputs["kv_length"] = .init(scalarType: .float16, shape: [1])
    assertContractRejected(function, contains: "kv_length")
  }

  func testRuntimeInvocationRequiresStaticStagingAndMatchingPosition() throws {
    XCTAssertNoThrow(
      try Gemma4MTPNativeContract.validateRuntimeInvocation(
        positionID: 1_025,
        kvLength: 1_025,
        hiddenShape: [1, 1, 2_816],
        kFullShape: [1, 2, 4_096, 512],
        vFullShape: [1, 2, 4_096, 512],
        kSlidingShape: [1, 8, 1_024, 256],
        vSlidingShape: [1, 8, 1_024, 256]))

    XCTAssertThrowsError(
      try Gemma4MTPNativeContract.validateRuntimeInvocation(
        positionID: 0,
        kvLength: 0,
        hiddenShape: [1, 1, 2_816],
        kFullShape: [1, 2, 4_096, 512],
        vFullShape: [1, 2, 4_096, 512],
        kSlidingShape: [1, 8, 1_024, 256],
        vSlidingShape: [1, 8, 1_024, 256])
    ) { error in
      XCTAssertTrue(String(describing: error).contains("kv_length"))
    }

    XCTAssertThrowsError(
      try Gemma4MTPNativeContract.validateRuntimeInvocation(
        positionID: 1_024,
        kvLength: 1_025,
        hiddenShape: [1, 1, 2_816],
        kFullShape: [1, 2, 4_096, 512],
        vFullShape: [1, 2, 4_096, 512],
        kSlidingShape: [1, 8, 1_024, 256],
        vSlidingShape: [1, 8, 1_024, 256])
    ) { error in
      XCTAssertTrue(String(describing: error).contains("position_ids"))
    }

    XCTAssertThrowsError(
      try Gemma4MTPNativeContract.validateRuntimeInvocation(
        positionID: 1_025,
        kvLength: 1_025,
        hiddenShape: [1, 1, 2_816],
        kFullShape: [1, 2, 1_025, 512],
        vFullShape: [1, 2, 4_096, 512],
        kSlidingShape: [1, 8, 1_024, 256],
        vSlidingShape: [1, 8, 1_024, 256])
    ) { error in
      XCTAssertTrue(String(describing: error).contains("k_full"))
    }
  }

  func testAcceptsWellFormedProposalRequest() throws {
    XCTAssertNoThrow(try Gemma4MTPNativeContract.validate(Self.validRequest))
  }

  func testRejectsOutOfRangeTokenAndPositionBeforeInference() {
    var request = Self.validRequest
    request.tokenID = Int32(Gemma4MTPNativeContract.vocabularySize)
    assertRequestRejected(request, contains: "token_id")

    request = Self.validRequest
    request.positionID = Int32(Gemma4MTPNativeContract.maxContextLength)
    assertRequestRejected(request, contains: "position_ids")
  }

  func testRejectsMalformedHiddenStorageBeforeInference() {
    var request = Self.validRequest
    request.hidden = .float16(
      shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize],
      values: [0])
    assertRequestRejected(request, contains: "hidden storage")

    request = Self.validRequest
    request.hidden = .int32(shape: [1, 1], values: [0])
    assertRequestRejected(request, contains: "hidden")
  }

  func testRejectsInvalidFixedKVStagingAndLogicalLengthBeforeInference() {
    var request = Self.validRequest
    request.vFull = Self.float16Tensor(shape: [1, 2, 4_095, 512])
    assertRequestRejected(request, contains: "v_full")

    request = Self.validRequest
    request.kSliding = Self.float16Tensor(shape: [1, 8, 1_023, 256])
    assertRequestRejected(request, contains: "k_sliding")

    request = Self.validRequest
    request.kvLength = Int32(Gemma4MTPNativeContract.cacheCapacity + 1)
    assertRequestRejected(request, contains: "kv_length")

    request = Self.validRequest
    request.kvLength = 1
    assertRequestRejected(request, contains: "position_ids must equal kv_length")
  }

  func testGreedyProposalExecutesOnceAndReturnsFirstMaximumWithNextHidden() async throws {
    var executionCount = 0
    var logits = [Float16](
      repeating: -2,
      count: Gemma4MTPNativeContract.vocabularySize)
    logits[7] = 3
    logits[9] = 3
    let nextHidden = Self.float16Tensor(
      shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize],
      repeating: 0.5)
    let runner = Gemma4MTPGreedyProposalRunner { request in
      executionCount += 1
      XCTAssertEqual(request, Self.validRequest)
      return Gemma4MTPModelOutputs(
        logits: .float16(
          shape: [1, 1, Gemma4MTPNativeContract.vocabularySize],
          values: logits),
        nextHidden: nextHidden)
    }

    let result = try await runner.propose(Self.validRequest)

    XCTAssertEqual(executionCount, 1)
    XCTAssertEqual(result.proposedToken, 7)
    XCTAssertEqual(result.nextHidden, nextHidden)
  }

  func testGreedyProposalDoesNotExecuteMalformedRequest() async {
    var executionCount = 0
    let runner = Gemma4MTPGreedyProposalRunner { _ in
      executionCount += 1
      return Self.validModelOutputs
    }
    var request = Self.validRequest
    request.hidden = .float16(
      shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize],
      values: [])

    do {
      _ = try await runner.propose(request)
      XCTFail("malformed request unexpectedly executed")
    } catch {
      XCTAssertTrue(String(describing: error).contains("hidden storage"))
    }
    XCTAssertEqual(executionCount, 0)
  }

  func testGreedyProposalRejectsMalformedModelOutputs() async {
    let runner = Gemma4MTPGreedyProposalRunner { _ in
      Gemma4MTPModelOutputs(
        logits: .float16(shape: [1, 1, 2], values: [0, 1]),
        nextHidden: Self.validModelOutputs.nextHidden)
    }

    do {
      _ = try await runner.propose(Self.validRequest)
      XCTFail("malformed model output unexpectedly accepted")
    } catch {
      XCTAssertTrue(String(describing: error).contains("logits"))
    }
  }

  func testGreedyProposalRejectsNonFiniteLogits() async {
    var logits = [Float16](
      repeating: 0,
      count: Gemma4MTPNativeContract.vocabularySize)
    logits[3] = .nan
    let runner = Gemma4MTPGreedyProposalRunner { _ in
      Gemma4MTPModelOutputs(
        logits: .float16(
          shape: [1, 1, Gemma4MTPNativeContract.vocabularySize],
          values: logits),
        nextHidden: Self.validModelOutputs.nextHidden)
    }

    do {
      _ = try await runner.propose(Self.validRequest)
      XCTFail("non-finite logits unexpectedly accepted")
    } catch {
      XCTAssertTrue(String(describing: error).contains("finite"))
    }
  }

  func testGreedyProposalRejectsNonFiniteNextHidden() async {
    var hidden = [Float16](
      repeating: 0,
      count: Gemma4MTPNativeContract.backboneHiddenSize)
    hidden[4] = .infinity
    let runner = Gemma4MTPGreedyProposalRunner { _ in
      Gemma4MTPModelOutputs(
        logits: Self.validModelOutputs.logits,
        nextHidden: .float16(
          shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize],
          values: hidden))
    }

    do {
      _ = try await runner.propose(Self.validRequest)
      XCTFail("non-finite next hidden unexpectedly accepted")
    } catch {
      XCTAssertTrue(String(describing: error).contains("next_hidden"))
      XCTAssertTrue(String(describing: error).contains("finite"))
    }
  }

  private func assertContractRejected(
    _ function: Gemma4MTPFunctionDescriptor,
    contains expected: String,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertThrowsError(
      try Gemma4MTPNativeContract.validate(function),
      file: file,
      line: line
    ) { error in
      XCTAssertTrue(
        String(describing: error).contains(expected),
        "\(error) does not contain \(expected)",
        file: file,
        line: line)
    }
  }

  private func assertRequestRejected(
    _ request: Gemma4MTPProposalRequest,
    contains expected: String,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertThrowsError(
      try Gemma4MTPNativeContract.validate(request),
      file: file,
      line: line
    ) { error in
      XCTAssertTrue(
        String(describing: error).contains(expected),
        "\(error) does not contain \(expected)",
        file: file,
        line: line)
    }
  }

  private func assertUnrolledContractRejected(
    _ function: Gemma4MTPFunctionDescriptor,
    contains expected: String,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertThrowsError(
      try Gemma4MTPNativeContract.validateUnrolled(function),
      file: file,
      line: line
    ) { error in
      XCTAssertTrue(
        String(describing: error).contains(expected),
        "\(error) does not contain \(expected)",
        file: file,
        line: line)
    }
  }

  private static let validFunction = Gemma4MTPFunctionDescriptor(
    inputs: [
      "token_id": .init(scalarType: .int32, shape: [1, 1]),
      "hidden": .init(
        scalarType: .float16,
        shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize]),
      "position_ids": .init(scalarType: .int32, shape: [1, 1]),
      "k_full": .init(
        scalarType: .float16,
        shape: [1, 2, Gemma4MTPNativeContract.cacheCapacity, 512]),
      "v_full": .init(
        scalarType: .float16,
        shape: [1, 2, Gemma4MTPNativeContract.cacheCapacity, 512]),
      "k_sliding": .init(
        scalarType: .float16,
        shape: [1, 8, Gemma4MTPNativeContract.slidingWindow, 256]),
      "v_sliding": .init(
        scalarType: .float16,
        shape: [1, 8, Gemma4MTPNativeContract.slidingWindow, 256]),
      "kv_length": .init(scalarType: .int32, shape: [1]),
    ],
    outputs: [
      "logits": .init(scalarType: .float16, shape: [1, 1, 262_144]),
      "next_hidden": .init(
        scalarType: .float16,
        shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize]),
    ],
    states: [:])

  private static let validRequest = Gemma4MTPProposalRequest(
    tokenID: 42,
    hidden: float16Tensor(
      shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize]),
    positionID: 2,
    kFull: float16Tensor(
      shape: [1, 2, Gemma4MTPNativeContract.cacheCapacity, 512]),
    vFull: float16Tensor(
      shape: [1, 2, Gemma4MTPNativeContract.cacheCapacity, 512]),
    kSliding: float16Tensor(
      shape: [1, 8, Gemma4MTPNativeContract.slidingWindow, 256]),
    vSliding: float16Tensor(
      shape: [1, 8, Gemma4MTPNativeContract.slidingWindow, 256]),
    kvLength: 2)

  private static let validUnrolledFunction = Gemma4MTPFunctionDescriptor(
    inputs: validFunction.inputs,
    outputs: [
      "draft_tokens": .init(scalarType: .int32, shape: [1, 4])
    ],
    states: [:])

  private static let validModelOutputs = Gemma4MTPModelOutputs(
    logits: float16Tensor(shape: [1, 1, 262_144]),
    nextHidden: float16Tensor(
      shape: [1, 1, Gemma4MTPNativeContract.backboneHiddenSize]))

  private static func float16Tensor(
    shape: [Int],
    repeating value: Float16 = 0
  ) -> Gemma4MTPHostTensor {
    .float16(
      shape: shape,
      values: [Float16](repeating: value, count: shape.reduce(1, *)))
  }
}
