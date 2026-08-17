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

    func testLastProposalMismatchRetainsAlreadyCorrectTargetState() throws {
        var state = try Qwen38GenerationState(layout: .native)
        try state.recordForward(tokens: 20)
        let controller = Qwen38MTPDecoder()

        // The three target verify inputs are [anchor, proposal0, proposal1]. A mismatch at
        // proposal2 therefore requires no rollback: all three inputs already consumed by the
        // target are part of the committed greedy prefix.
        let result = try controller.verify(
            proposals: [31, 32, 99],
            targetGreedyTokens: [31, 32, 33],
            state: &state)

        XCTAssertEqual(result.acceptedDraftTokens, [31, 32])
        XCTAssertEqual(result.correctionToken, 33)
        XCTAssertEqual(result.stateAction, .retainVerifiedState)
        XCTAssertEqual(state.position, 23)
        try controller.commitReplay(result, state: &state)
        XCTAssertEqual(state.position, 23)
    }

    func testWidthTwoLastMismatchAlsoRetainsVerifiedAnchorAndFirstProposal() throws {
        var state = try Qwen38GenerationState(layout: .native)
        try state.recordForward(tokens: 8)

        let result = try Qwen38MTPDecoder().verify(
            proposals: [41, 99], targetGreedyTokens: [41, 42], state: &state)

        XCTAssertEqual(result.acceptedDraftTokens, [41])
        XCTAssertEqual(result.correctionToken, 42)
        XCTAssertEqual(result.stateAction, .retainVerifiedState)
        XCTAssertEqual(state.position, 10)
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
