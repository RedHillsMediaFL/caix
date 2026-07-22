import XCTest

@testable import PipelineRuntime

final class Gemma4SequentialMTPDecoderTests: XCTestCase {
    func testSequentialVerificationMatchesGreedyTargetAcrossAcceptanceAndMismatch() async throws {
        var proposals: [Int32] = [11, 99, 13, 14]
        var targetTokens: [Int32] = [11, 12, 13, 14]
        var targetForwardedTokens: [Int32] = []
        var proposalRequests: [Gemma4MTPProposalRequest] = []
        let decoder = try Gemma4SequentialMTPDecoder(
            draftTokens: 2,
            propose: { request in
                proposalRequests.append(request)
                return Gemma4MTPProposalResult(
                    proposedToken: proposals.removeFirst(),
                    nextHidden: .float16(shape: [1, 1, 1], values: [7]))
            },
            targetDecode: { token, position in
                targetForwardedTokens.append(token)
                return Gemma4SequentialMTPTargetStep(
                    tokenID: targetTokens.removeFirst(),
                    artifacts: try Self.artifacts(
                        length: position + 1,
                        hiddenBits: [UInt16(position + 1)]))
            })
        var generated: [Int32] = [10]

        let outcome = try await decoder.run(
            anchorToken: 10,
            targetArtifacts: Self.artifacts(length: 3, hiddenBits: [3]),
            maximumAdditionalTokens: 4
        ) { token in
            generated.append(token)
            return true
        }

        XCTAssertEqual(generated, [10, 11, 12, 13, 14])
        XCTAssertEqual(targetForwardedTokens, [10, 11, 12, 13])
        XCTAssertFalse(targetForwardedTokens.contains(99))
        XCTAssertEqual(proposalRequests.map(\.positionID), [3, 3, 5, 5])
        XCTAssertEqual(proposalRequests.map(\.tokenID), [10, 11, 12, 13])
        XCTAssertEqual(
            proposalRequests.map { Self.bitPatterns($0.hidden) },
            [[3], [Float16(7).bitPattern], [5], [Float16(7).bitPattern]])
        XCTAssertEqual(proposalRequests.map(\.kFull.shape), [
            [1, 1, 3, 1], [1, 1, 3, 1], [1, 1, 5, 1], [1, 1, 5, 1],
        ])
        XCTAssertEqual(outcome.telemetry.draftedTokens, 4)
        XCTAssertEqual(outcome.telemetry.acceptedDraftTokens, 3)
        XCTAssertEqual(outcome.telemetry.proposalBatches, 2)
        XCTAssertEqual(outcome.telemetry.targetDecodeForwards, 4)
        XCTAssertEqual(outcome.telemetry.strategy, "sequential_no_rollback")
        XCTAssertFalse(outcome.telemetry.fastMTP)
        XCTAssertTrue(outcome.telemetry.exercised)
    }

    func testMismatchEmitsCorrectionAndDiscardsUnverifiedProposalSuffix() async throws {
        var proposals: [Int32] = [99, 77, 88]
        var forwarded: [Int32] = []
        let decoder = try Gemma4SequentialMTPDecoder(
            draftTokens: 3,
            propose: { _ in
                Gemma4MTPProposalResult(
                    proposedToken: proposals.removeFirst(),
                    nextHidden: .float16(shape: [1, 1, 1], values: [0]))
            },
            targetDecode: { token, position in
                forwarded.append(token)
                return Gemma4SequentialMTPTargetStep(
                    tokenID: 11,
                    artifacts: try Self.artifacts(length: position + 1))
            })
        var committed: [Int32] = []

        let outcome = try await decoder.run(
            anchorToken: 10,
            targetArtifacts: Self.artifacts(length: 2),
            maximumAdditionalTokens: 3
        ) { token in
            committed.append(token)
            return false
        }

        XCTAssertEqual(committed, [11])
        XCTAssertEqual(forwarded, [10])
        XCTAssertEqual(outcome.telemetry.draftedTokens, 3)
        XCTAssertEqual(outcome.telemetry.acceptedDraftTokens, 0)
        XCTAssertEqual(outcome.telemetry.proposalBatches, 1)
        XCTAssertEqual(outcome.telemetry.targetDecodeForwards, 1)
        XCTAssertTrue(outcome.telemetry.exercised)
    }

    func testAllAcceptedBatchForwardsEachCommittedPredecessorExactlyOnce() async throws {
        var proposals: [Int32] = [21, 22, 23]
        var targetTokens: [Int32] = [21, 22, 23]
        var forwarded: [Int32] = []
        let decoder = try Gemma4SequentialMTPDecoder(
            draftTokens: 3,
            propose: { _ in
                Gemma4MTPProposalResult(
                    proposedToken: proposals.removeFirst(),
                    nextHidden: .float16(shape: [1, 1, 1], values: [0]))
            },
            targetDecode: { token, position in
                forwarded.append(token)
                return Gemma4SequentialMTPTargetStep(
                    tokenID: targetTokens.removeFirst(),
                    artifacts: try Self.artifacts(length: position + 1))
            })
        var committed: [Int32] = []

        let outcome = try await decoder.run(
            anchorToken: 20,
            targetArtifacts: Self.artifacts(length: 4),
            maximumAdditionalTokens: 3
        ) { token in
            committed.append(token)
            return true
        }

        XCTAssertEqual(committed, [21, 22, 23])
        XCTAssertEqual(forwarded, [20, 21, 22])
        XCTAssertEqual(outcome.telemetry.acceptedDraftTokens, 3)
        XCTAssertEqual(outcome.telemetry.targetDecodeForwards, 3)
    }

    func testTargetArtifactsBridgePreservesFloat16BitPatterns() async throws {
        let exactBits: [UInt16] = [0x0000, 0x8000, 0x3555, 0x7bff]
        let initial = try Self.artifacts(
            length: 4,
            hiddenBits: [0x8000],
            fullBits: exactBits,
            slidingBits: exactBits)
        var captured: Gemma4MTPProposalRequest?
        let decoder = try Gemma4SequentialMTPDecoder(
            draftTokens: 1,
            propose: { request in
                captured = request
                return Gemma4MTPProposalResult(
                    proposedToken: 8,
                    nextHidden: .float16(shape: [1, 1, 1], values: [0]))
            },
            targetDecode: { _, position in
                Gemma4SequentialMTPTargetStep(
                    tokenID: 9,
                    artifacts: try Self.artifacts(length: position + 1))
            })

        _ = try await decoder.run(
            anchorToken: 7,
            targetArtifacts: initial,
            maximumAdditionalTokens: 1
        ) { _ in false }

        let request = try XCTUnwrap(captured)
        XCTAssertEqual(Self.bitPatterns(request.hidden), [0x8000])
        XCTAssertEqual(Self.bitPatterns(request.kFull), exactBits)
        XCTAssertEqual(Self.bitPatterns(request.vFull), exactBits)
        XCTAssertEqual(Self.bitPatterns(request.kSliding), exactBits)
        XCTAssertEqual(Self.bitPatterns(request.vSliding), exactBits)
    }

    func testRejectsDraftWidthOutsideBoundedRuntimeRangeWithoutExecutingInference() {
        XCTAssertThrowsError(try Gemma4SequentialMTPDecoder(
            draftTokens: 0,
            propose: { _ in
                XCTFail("proposal inference must not execute")
                throw TestError.unexpectedExecution
            },
            targetDecode: { _, _ in
                XCTFail("target inference must not execute")
                throw TestError.unexpectedExecution
            }))

        XCTAssertThrowsError(try Gemma4SequentialMTPDecoder(
            draftTokens: Gemma4MTPDecodeConfiguration.maximumDraftTokens + 1,
            propose: { _ in
                XCTFail("proposal inference must not execute")
                throw TestError.unexpectedExecution
            },
            targetDecode: { _, _ in
                XCTFail("target inference must not execute")
                throw TestError.unexpectedExecution
            }))
    }

    func testRejectsTargetSnapshotWithWrongSlidingWindowCoverageBeforeInference() async throws {
        let malformed = try Self.artifacts(length: 4, slidingLength: 3)
        let decoder = try Gemma4SequentialMTPDecoder(
            draftTokens: 1,
            propose: { _ in
                XCTFail("proposal inference must not execute")
                throw TestError.unexpectedExecution
            },
            targetDecode: { _, _ in
                XCTFail("target inference must not execute")
                throw TestError.unexpectedExecution
            })

        do {
            _ = try await decoder.run(
                anchorToken: 7,
                targetArtifacts: malformed,
                maximumAdditionalTokens: 1,
                commit: { _ in true })
            XCTFail("malformed sliding target snapshot unexpectedly decoded")
        } catch {
            XCTAssertTrue(String(describing: error).contains("sliding"))
        }
    }

    private enum TestError: Error {
        case unexpectedExecution
    }

    private static func bitPatterns(_ tensor: Gemma4MTPHostTensor) -> [UInt16] {
        guard case .float16(_, let values) = tensor else {
            XCTFail("expected FP16 tensor")
            return []
        }
        return values.map(\.bitPattern)
    }

    private static func artifacts(
        length: Int,
        slidingLength: Int? = nil,
        hiddenBits: [UInt16] = [0],
        fullBits: [UInt16]? = nil,
        slidingBits: [UInt16]? = nil
    ) throws -> DistributedEagleTargetArtifacts {
        let resolvedSlidingLength = slidingLength ?? length
        let full = fullBits ?? [UInt16](repeating: 0, count: length)
        let sliding = slidingBits ?? [UInt16](repeating: 0, count: resolvedSlidingLength)
        return try DistributedEagleTargetArtifacts(
            finalHidden: DistributedEagleTargetTensor(
                shape: [1, 1, 1],
                scalarType: .float16,
                float16BitPatterns: hiddenBits),
            fullKey: DistributedEagleTargetTensor(
                shape: [1, 1, length, 1],
                scalarType: .float16,
                float16BitPatterns: full),
            fullValue: DistributedEagleTargetTensor(
                shape: [1, 1, length, 1],
                scalarType: .float16,
                float16BitPatterns: full),
            slidingKey: DistributedEagleTargetTensor(
                shape: [1, 1, resolvedSlidingLength, 1],
                scalarType: .float16,
                float16BitPatterns: sliding),
            slidingValue: DistributedEagleTargetTensor(
                shape: [1, 1, resolvedSlidingLength, 1],
                scalarType: .float16,
                float16BitPatterns: sliding),
            fullPositionRange: DistributedSequenceRange(lowerBound: 0, upperBound: length),
            slidingPositionRange: DistributedSequenceRange(
                lowerBound: length - resolvedSlidingLength,
                upperBound: length))
    }
}
