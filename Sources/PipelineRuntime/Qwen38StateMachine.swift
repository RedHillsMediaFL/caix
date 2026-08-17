import Foundation

/// Immutable geometry for the Qwen3.8-27B hybrid Core AI graph.
///
/// The Qwen3.8 model has 64 transformer blocks, but only every fourth block is full attention.
/// The native graph therefore retains K/V for sixteen layers and gives its 48 linear-attention
/// blocks explicit fixed convolution and recurrent state. Treating every block as a normal KV
/// layer would consume 64 GiB at the model's 262K context limit.
public struct Qwen38StateLayout: Sendable, Equatable {
    public static let native = Qwen38StateLayout(
        maxContextLength: 262_144,
        fullAttentionLayers: 16,
        kvHeads: 4,
        headDimension: 256,
        keyValueScalarBytes: 2,
        stateNames: ["keyCache", "valueCache", "convState", "recurrentState"])

    public let maxContextLength: Int
    public let fullAttentionLayers: Int
    public let kvHeads: Int
    public let headDimension: Int
    public let keyValueScalarBytes: Int
    public let stateNames: [String]

    public init(
        maxContextLength: Int,
        fullAttentionLayers: Int,
        kvHeads: Int,
        headDimension: Int,
        keyValueScalarBytes: Int,
        stateNames: [String]
    ) {
        self.maxContextLength = maxContextLength
        self.fullAttentionLayers = fullAttentionLayers
        self.kvHeads = kvHeads
        self.headDimension = headDimension
        self.keyValueScalarBytes = keyValueScalarBytes
        self.stateNames = stateNames
    }

    /// Allocation size of the compact K+V persistent state at `maxContextLength`.
    /// `keyCache` and `valueCache` are each `[16, 4, context, 256]` FP16 tensors.
    public var keyValueCacheBytes: UInt64 {
        UInt64(fullAttentionLayers)
            * 2
            * UInt64(kvHeads)
            * UInt64(maxContextLength)
            * UInt64(headDimension)
            * UInt64(keyValueScalarBytes)
    }
}

/// The only legal action after verifying a native Qwen MTP proposal.
///
/// `restoreFixedStateAndReplay` intentionally has no K/V rewind operation. K/V bytes beyond the
/// cursor are stale but unreachable because the next graph call uses the restored cursor and
/// overwrites that suffix. The fixed hybrid states must be restored and accepted tokens replayed
/// to rebuild their exact state before the correction token is run.
public enum Qwen38VerificationAction: Sendable, Equatable {
    case retainVerifiedState
    case restoreFixedStateAndReplay(fromPosition: Int, acceptedDraftTokens: Int)
}

public enum Qwen38StateError: Swift.Error, Sendable, Equatable {
    case invalidLayout
    case invalidTokenCount
    case contextLimitExceeded
    case invalidDraftWidth
    case invalidVerificationCheckpoint
    case invalidAcceptedDraftCount
}

/// Cursor-only half of the four-state GPU store.
///
/// The Core AI runtime owns the actual MTL buffers; this value type owns the invariants that make
/// them safe to reuse. It has deliberately no partial-rewind API, which prevents a future caller
/// from trying to rewind only the compact K/V state while leaving Qwen's convolution/recurrent
/// state advanced past the same token position.
public struct Qwen38GenerationState: Sendable {
    public struct VerificationCheckpoint: Sendable, Equatable {
        fileprivate let position: Int
        fileprivate let draftTokens: Int
    }

    public let layout: Qwen38StateLayout
    public private(set) var position: Int

    /// This is an invariant exposed for diagnostics/tests: native Qwen never offers a partial KV
    /// rewind. A mismatch restores fixed state and replays from a checkpoint instead.
    public var hasPartialKVRewind: Bool { false }

    public init(layout: Qwen38StateLayout = .native) throws {
        guard layout.maxContextLength > 0,
            layout.fullAttentionLayers == 16,
            layout.kvHeads == 4,
            layout.headDimension == 256,
            layout.keyValueScalarBytes == 2,
            layout.stateNames == Qwen38StateLayout.native.stateNames
        else {
            throw Qwen38StateError.invalidLayout
        }
        self.layout = layout
        self.position = 0
    }

    /// Records a completed forward pass. Position is the sole source of truth for state reads;
    /// stale storage after it is never addressable by the graph.
    public mutating func recordForward(tokens: Int) throws {
        guard tokens > 0 else { throw Qwen38StateError.invalidTokenCount }
        let (next, overflow) = position.addingReportingOverflow(tokens)
        guard !overflow, next <= layout.maxContextLength else {
            throw Qwen38StateError.contextLimitExceeded
        }
        position = next
    }

    /// Reserves a batched target verify call. Native Qwen MTP exposes one to three proposal
    /// positions; after the verify returns, `resolveVerification` either retains this advanced
    /// cursor or restores it to the checkpoint for fixed-state replay.
    public mutating func beginVerification(draftTokens: Int) throws -> VerificationCheckpoint {
        guard (1...3).contains(draftTokens) else { throw Qwen38StateError.invalidDraftWidth }
        let checkpoint = VerificationCheckpoint(position: position, draftTokens: draftTokens)
        try recordForward(tokens: draftTokens)
        return checkpoint
    }

    /// Resolves exact greedy target verification. On rejection `position` moves back only as an
    /// abstract cursor while the GPU executor restores its fixed state snapshot and replays the
    /// accepted prefix; this method never authorizes mutation of a K/V prefix.
    public mutating func resolveVerification(
        checkpoint: VerificationCheckpoint,
        acceptedDraftTokens: Int
    ) throws -> Qwen38VerificationAction {
        guard checkpoint.position >= 0,
            checkpoint.position + checkpoint.draftTokens == position
        else {
            throw Qwen38StateError.invalidVerificationCheckpoint
        }
        guard (0...checkpoint.draftTokens).contains(acceptedDraftTokens) else {
            throw Qwen38StateError.invalidAcceptedDraftCount
        }
        // A width-N target verify consumes [anchor, proposal0, … proposal(N-2)]. If only the
        // final proposal is rejected, every input already written into target state is still
        // part of the committed greedy prefix, so no restore is necessary.
        guard acceptedDraftTokens + 1 < checkpoint.draftTokens else {
            return .retainVerifiedState
        }
        position = checkpoint.position
        return .restoreFixedStateAndReplay(
            fromPosition: checkpoint.position,
            acceptedDraftTokens: acceptedDraftTokens)
    }

    /// A request boundary clears all four persistent GPU states and restarts the monotonic cursor.
    /// Prefix cache reuse, when introduced, must create a separately validated complete snapshot;
    /// it cannot call this method partially.
    public mutating func reset() {
        position = 0
    }
}
