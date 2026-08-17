import Foundation

/// Exact-greedy control plane for the Qwen3.8 native MTP sidecar.
///
/// The sidecar proposes one to three tokens from the target's post-norm hidden state. A target
/// verify graph produces the greedy reference tokens. This controller commits only the matching
/// prefix and turns a mismatch into the explicit fixed-state restore/replay operation encoded by
/// ``Qwen38GenerationState``. GPU code supplies the proposal and verify logits; this pure layer
/// keeps its correctness rules independently testable.
public struct Qwen38MTPDecoder: Sendable {
    public enum Error: Swift.Error, Sendable, Equatable {
        case invalidDraftWidth
        case mismatchedVerificationWidth
        case invalidReplay
    }

    public struct VerificationResult: Sendable, Equatable {
        public let acceptedDraftTokens: [Int32]
        /// The target greedy token at the first rejected proposal. `nil` means all proposals
        /// matched and the target verify state is retained as-is.
        public let correctionToken: Int32?
        public let stateAction: Qwen38VerificationAction

        fileprivate init(
            acceptedDraftTokens: [Int32],
            correctionToken: Int32?,
            stateAction: Qwen38VerificationAction
        ) {
            self.acceptedDraftTokens = acceptedDraftTokens
            self.correctionToken = correctionToken
            self.stateAction = stateAction
        }
    }

    public init() {}

    /// Compare proposal tokens with the batched target-greedy outputs. This advances state by the
    /// draft width during the verify, then either retains that advance or rolls the abstract cursor
    /// back to force a GPU fixed-state restore + exact replay.
    public func verify(
        proposals: [Int32],
        targetGreedyTokens: [Int32],
        state: inout Qwen38GenerationState
    ) throws -> VerificationResult {
        guard (1...3).contains(proposals.count) else { throw Error.invalidDraftWidth }
        guard targetGreedyTokens.count == proposals.count else {
            throw Error.mismatchedVerificationWidth
        }

        let checkpoint = try state.beginVerification(draftTokens: proposals.count)
        let acceptedCount = zip(proposals, targetGreedyTokens)
            .prefix { $0 == $1 }
            .count
        let action = try state.resolveVerification(
            checkpoint: checkpoint,
            acceptedDraftTokens: acceptedCount)
        let correction = acceptedCount == proposals.count ? nil : targetGreedyTokens[acceptedCount]
        return VerificationResult(
            acceptedDraftTokens: Array(proposals.prefix(acceptedCount)),
            correctionToken: correction,
            stateAction: action)
    }

    /// Complete the replay mandated by a mismatch after the executor has restored the fixed MTL
    /// buffers. The executor replays `acceptedDraftTokens`, then feeds `correctionToken`; stale K/V
    /// bytes from rejected draft positions remain beyond the cursor and are never read.
    public func commitReplay(
        _ result: VerificationResult,
        state: inout Qwen38GenerationState
    ) throws {
        switch result.stateAction {
        case .retainVerifiedState:
            guard result.correctionToken == nil else { throw Error.invalidReplay }
        case let .restoreFixedStateAndReplay(_, acceptedDraftTokens):
            guard result.acceptedDraftTokens.count == acceptedDraftTokens,
                result.correctionToken != nil
            else { throw Error.invalidReplay }
            try state.recordForward(tokens: acceptedDraftTokens + 1)
        }
    }
}
