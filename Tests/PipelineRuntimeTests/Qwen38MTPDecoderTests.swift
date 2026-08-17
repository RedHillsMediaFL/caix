import XCTest

@testable import PipelineRuntime

final class Qwen38MTPDecoderTests: XCTestCase {
    func testMismatchReplaysOnlyGreedyMatchingPrefixThenCommitsCorrection() throws {
        var state = try Qwen38GenerationState(layout: .native)
        try state.recordForward(tokens: 40)
        let controller = Qwen38MTPDecoder()

        let result = try controller.verify(
            proposals: [11, 99, 13],
            targetGreedyTokens: [11, 12, 13],
            state: &state)

        XCTAssertEqual(result.acceptedDraftTokens, [11])
        XCTAssertEqual(result.correctionToken, 12)
        XCTAssertEqual(
            result.stateAction,
            .restoreFixedStateAndReplay(fromPosition: 40, acceptedDraftTokens: 1))
        XCTAssertEqual(state.position, 40)
        try controller.commitReplay(result, state: &state)
        XCTAssertEqual(state.position, 42)
    }

    func testAllAcceptedDraftRetainsVerifiedStateWithoutReplay() throws {
        var state = try Qwen38GenerationState(layout: .native)
        try state.recordForward(tokens: 20)
        let controller = Qwen38MTPDecoder()

        let result = try controller.verify(
            proposals: [31, 32, 33],
            targetGreedyTokens: [31, 32, 33],
            state: &state)

        XCTAssertEqual(result.acceptedDraftTokens, [31, 32, 33])
        XCTAssertNil(result.correctionToken)
        XCTAssertEqual(result.stateAction, .retainVerifiedState)
        XCTAssertEqual(state.position, 23)
    }

    func testRejectsInvalidProposalOrTargetWidthsBeforeMutatingState() throws {
        var state = try Qwen38GenerationState(layout: .native)
        let controller = Qwen38MTPDecoder()

        XCTAssertThrowsError(
            try controller.verify(proposals: [], targetGreedyTokens: [], state: &state))
        XCTAssertEqual(state.position, 0)
        XCTAssertThrowsError(
            try controller.verify(proposals: [1, 2], targetGreedyTokens: [1], state: &state))
        XCTAssertEqual(state.position, 0)
    }
}
