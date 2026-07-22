import Foundation

public enum Gemma4MTPDecodeConfiguration {
    public static let defaultDraftTokens = 4
    public static let maximumDraftTokens = 8
}

/// Raw counters for staged Gemma 4 MTP decoding.
///
/// This first implementation deliberately verifies one proposal per target Q=1 forward. It never
/// writes an unverified token into target state, so rejected draft suffixes require no rollback.
public struct Gemma4MTPDecodeTelemetry: Hashable, Sendable {
    public let draftedTokens: Int
    public let acceptedDraftTokens: Int
    public let proposalBatches: Int
    public let targetDecodeForwards: Int
    public let strategy: String
    public let fastMTP: Bool
    public let exercised: Bool

    public init(
        draftedTokens: Int,
        acceptedDraftTokens: Int,
        proposalBatches: Int,
        targetDecodeForwards: Int
    ) {
        self.draftedTokens = draftedTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.proposalBatches = proposalBatches
        self.targetDecodeForwards = targetDecodeForwards
        self.strategy = "sequential_no_rollback"
        self.fastMTP = false
        self.exercised = draftedTokens > 0
    }

    static let unexercisedSequential = Gemma4MTPDecodeTelemetry(
        draftedTokens: 0,
        acceptedDraftTokens: 0,
        proposalBatches: 0,
        targetDecodeForwards: 0)
}

struct Gemma4SequentialMTPTargetStep {
    let tokenID: Int32
    let artifacts: DistributedEagleTargetArtifacts
}

struct Gemma4SequentialMTPDecodeOutcome {
    let telemetry: Gemma4MTPDecodeTelemetry
    let stoppedByCommit: Bool
}

/// Package-testable sequential Gemma 4 MTP orchestration.
///
/// The caller supplies the target's greedy anchor after prefill. Each proposal batch is built from
/// that anchor, the target's post-norm final hidden, and target KV covering every token before the
/// anchor. Verification then advances the target one committed token at a time. A mismatch emits
/// the target correction and abandons the rest of the already-drafted batch without forwarding it.
struct Gemma4SequentialMTPDecoder {
    typealias Propose = (Gemma4MTPProposalRequest) async throws -> Gemma4MTPProposalResult
    typealias TargetDecode = (Int32, Int) async throws -> Gemma4SequentialMTPTargetStep

    private let draftTokens: Int
    private let propose: Propose
    private let targetDecode: TargetDecode

    init(
        draftTokens: Int,
        propose: @escaping Propose,
        targetDecode: @escaping TargetDecode
    ) throws {
        guard (1...Gemma4MTPDecodeConfiguration.maximumDraftTokens).contains(draftTokens) else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "Gemma 4 MTP draft token count must be in 1...\(Gemma4MTPDecodeConfiguration.maximumDraftTokens)")
        }
        self.draftTokens = draftTokens
        self.propose = propose
        self.targetDecode = targetDecode
    }

    func run(
        anchorToken initialAnchor: Int32,
        targetArtifacts initialArtifacts: DistributedEagleTargetArtifacts,
        maximumAdditionalTokens: Int,
        commit: (Int32) -> Bool
    ) async throws -> Gemma4SequentialMTPDecodeOutcome {
        guard maximumAdditionalTokens >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "Gemma 4 MTP remaining token budget must be non-negative")
        }
        try Self.validateSeedArtifacts(initialArtifacts)

        var anchor = initialAnchor
        var artifacts = initialArtifacts
        var remaining = maximumAdditionalTokens
        var draftedTokens = 0
        var acceptedDraftTokens = 0
        var proposalBatches = 0
        var targetDecodeForwards = 0

        while remaining > 0 {
            let activeDraftTokens = min(draftTokens, remaining)
            let position = artifacts.fullPositionRange.upperBound
            guard let positionID = Int32(exactly: position) else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "Gemma 4 MTP position \(position) does not fit int32")
            }
            let fixedKV = Self.hostKV(from: artifacts)
            var draftInputToken = anchor
            var draftHidden = Self.hostTensor(from: artifacts.finalHidden)
            var proposals: [Int32] = []
            proposals.reserveCapacity(activeDraftTokens)

            for _ in 0..<activeDraftTokens {
                let result = try await propose(Gemma4MTPProposalRequest(
                    tokenID: draftInputToken,
                    hidden: draftHidden,
                    positionID: positionID,
                    kFull: fixedKV.kFull,
                    vFull: fixedKV.vFull,
                    kSliding: fixedKV.kSliding,
                    vSliding: fixedKV.vSliding))
                proposals.append(result.proposedToken)
                draftInputToken = result.proposedToken
                draftHidden = result.nextHidden
            }
            draftedTokens += proposals.count
            proposalBatches += 1

            var mismatched = false
            for proposal in proposals {
                let targetStep = try await targetDecode(
                    anchor,
                    Self.positionForAnchor(artifacts))
                targetDecodeForwards += 1
                try Self.validateAdvancedArtifacts(
                    targetStep.artifacts,
                    previous: artifacts)
                artifacts = targetStep.artifacts

                let committedToken: Int32
                if targetStep.tokenID == proposal {
                    acceptedDraftTokens += 1
                    committedToken = proposal
                } else {
                    committedToken = targetStep.tokenID
                    mismatched = true
                }
                anchor = committedToken
                remaining -= 1
                if !commit(committedToken) {
                    return Gemma4SequentialMTPDecodeOutcome(
                        telemetry: Self.telemetry(
                            drafted: draftedTokens,
                            accepted: acceptedDraftTokens,
                            batches: proposalBatches,
                            targetForwards: targetDecodeForwards),
                        stoppedByCommit: true)
                }
                if mismatched || remaining == 0 { break }
            }
        }

        return Gemma4SequentialMTPDecodeOutcome(
            telemetry: Self.telemetry(
                drafted: draftedTokens,
                accepted: acceptedDraftTokens,
                batches: proposalBatches,
                targetForwards: targetDecodeForwards),
            stoppedByCommit: false)
    }

    private static func positionForAnchor(
        _ artifacts: DistributedEagleTargetArtifacts
    ) -> Int {
        artifacts.fullPositionRange.upperBound
    }

    private static func validateSeedArtifacts(
        _ artifacts: DistributedEagleTargetArtifacts
    ) throws {
        guard artifacts.fullPositionRange.lowerBound == 0 else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "Gemma 4 MTP full target KV must begin at position 0")
        }
        let fullEnd = artifacts.fullPositionRange.upperBound
        let expectedSlidingCount = min(
            fullEnd,
            Gemma4MTPNativeContract.slidingWindow)
        guard artifacts.slidingPositionRange.upperBound == fullEnd,
            artifacts.slidingPositionRange.count == expectedSlidingCount
        else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "Gemma 4 MTP sliding target KV must end at \(fullEnd) and cover \(expectedSlidingCount) positions")
        }
    }

    private static func validateAdvancedArtifacts(
        _ artifacts: DistributedEagleTargetArtifacts,
        previous: DistributedEagleTargetArtifacts
    ) throws {
        let expectedEnd = previous.fullPositionRange.upperBound + 1
        try validateSeedArtifacts(artifacts)
        guard artifacts.fullPositionRange
            == DistributedSequenceRange(lowerBound: 0, upperBound: expectedEnd)
        else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "Gemma 4 MTP target Q=1 forward must advance full KV to \(expectedEnd)")
        }
    }

    private static func hostTensor(
        from tensor: DistributedEagleTargetTensor
    ) -> Gemma4MTPHostTensor {
        .float16(
            shape: tensor.shape,
            values: tensor.float16BitPatterns.map(Float16.init(bitPattern:)))
    }

    private static func hostKV(
        from artifacts: DistributedEagleTargetArtifacts
    ) -> (
        kFull: Gemma4MTPHostTensor,
        vFull: Gemma4MTPHostTensor,
        kSliding: Gemma4MTPHostTensor,
        vSliding: Gemma4MTPHostTensor
    ) {
        (
            kFull: hostTensor(from: artifacts.fullKey),
            vFull: hostTensor(from: artifacts.fullValue),
            kSliding: hostTensor(from: artifacts.slidingKey),
            vSliding: hostTensor(from: artifacts.slidingValue))
    }

    private static func telemetry(
        drafted: Int,
        accepted: Int,
        batches: Int,
        targetForwards: Int
    ) -> Gemma4MTPDecodeTelemetry {
        Gemma4MTPDecodeTelemetry(
            draftedTokens: drafted,
            acceptedDraftTokens: accepted,
            proposalBatches: batches,
            targetDecodeForwards: targetForwards)
    }
}
