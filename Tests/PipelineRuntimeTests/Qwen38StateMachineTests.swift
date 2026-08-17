import XCTest

@testable import PipelineRuntime

final class Qwen38StateMachineTests: XCTestCase {
    func testCompactHybridStateUsesOnlySixteenAttentionLayersAtFullContext() throws {
        let layout = Qwen38StateLayout.native

        XCTAssertEqual(layout.maxContextLength, 262_144)
        XCTAssertEqual(layout.fullAttentionLayers, 16)
        XCTAssertEqual(layout.keyValueCacheBytes, 16 * 1_073_741_824)
        XCTAssertEqual(layout.stateNames, ["keyCache", "valueCache", "convState", "recurrentState"])
    }

    func testMismatchRestoresOnlyFixedStateAndReplaysAcceptedPrefix() throws {
        var state = try Qwen38GenerationState(layout: .native)
        try state.recordForward(tokens: 12)
        let checkpoint = try state.beginVerification(draftTokens: 3)

        let action = try state.resolveVerification(checkpoint: checkpoint, acceptedDraftTokens: 1)

        XCTAssertEqual(
            action,
            .restoreFixedStateAndReplay(
                fromPosition: 12,
                acceptedDraftTokens: 1))
        XCTAssertEqual(state.position, 12)
        XCTAssertFalse(state.hasPartialKVRewind)
    }

    func testFullyAcceptedVerificationRetainsAdvancedCursor() throws {
        var state = try Qwen38GenerationState(layout: .native)
        try state.recordForward(tokens: 262_140)
        let checkpoint = try state.beginVerification(draftTokens: 3)

        let action = try state.resolveVerification(checkpoint: checkpoint, acceptedDraftTokens: 3)

        XCTAssertEqual(action, .retainVerifiedState)
        XCTAssertEqual(state.position, 262_143)
    }

    func testResetIsTheOnlyCursorRewindMechanism() throws {
        var state = try Qwen38GenerationState(layout: .native)
        try state.recordForward(tokens: 128)

        state.reset()

        XCTAssertEqual(state.position, 0)
        XCTAssertFalse(state.hasPartialKVRewind)
    }
}
