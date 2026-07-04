import XCTest

@testable import PipelineRuntime

final class MonolithicPrefillPolicyTests: XCTestCase {
    func testStatefulMonolithicDefaultsToTraceWidth() {
        XCTAssertEqual(
            MonolithicPrefillPolicy.resolvedChunkSize(
                promptCount: 19, isStatefulMonolithic: true, environment: [:]),
            16)
        XCTAssertEqual(
            MonolithicPrefillPolicy.resolvedChunkSize(
                promptCount: 8, isStatefulMonolithic: true, environment: [:]),
            8)
        XCTAssertEqual(
            MonolithicPrefillPolicy.resolvedChunkSize(
                promptCount: 19, isStatefulMonolithic: false, environment: [:]),
            19)
    }

    func testExplicitPrefillOverridesArePreserved() {
        XCTAssertEqual(
            MonolithicPrefillPolicy.resolvedChunkSize(
                promptCount: 19,
                isStatefulMonolithic: true,
                environment: ["COREAI_PREFILL_CHUNK": "8"]),
            8)
        XCTAssertEqual(
            MonolithicPrefillPolicy.resolvedChunkSize(
                promptCount: 19,
                isStatefulMonolithic: true,
                environment: ["COREAI_PREFILL_MODE": "token"]),
            1)
        XCTAssertEqual(
            MonolithicPrefillPolicy.resolvedChunkSize(
                promptCount: 19,
                isStatefulMonolithic: true,
                environment: ["COREAI_PREFILL_MODE": "batch"]),
            19)
    }

    func testCoreAILanguageModelsThresholdDefaultAndOverrides() {
        XCTAssertEqual(MonolithicPrefillPolicy.coreAILanguageModelsChunkThreshold(environment: [:]), 16)
        XCTAssertNil(
            MonolithicPrefillPolicy.coreAILanguageModelsChunkThreshold(
                environment: ["COREAI_CHUNK_THRESHOLD": "32"]))
        XCTAssertEqual(
            MonolithicPrefillPolicy.coreAILanguageModelsChunkThreshold(
                environment: ["COREAI_PREFILL_CHUNK": "8"]),
            8)
        XCTAssertNil(
            MonolithicPrefillPolicy.coreAILanguageModelsChunkThreshold(
                environment: ["COREAI_PREFILL_MODE": "batch"]))
    }
}
