import XCTest

@testable import PipelineRuntime

final class Qwen38AccelerationPolicyTests: XCTestCase {
    func testAutoUsesNativeMTPOnlyWhenParityAndSpeedGateHold() throws {
        let decision = try Qwen38AccelerationPolicy.resolve(
            requested: .auto,
            temperature: 0,
            proof: .init(
                exactGreedy: true,
                autoregressiveTokensPerSecond: 20,
                mtpTokensPerSecond: 23))

        XCTAssertEqual(decision, .nativeMTP)
    }

    func testAutoFallsBackToAutoregressiveWhenMTPIsNotFifteenPercentFaster() throws {
        let decision = try Qwen38AccelerationPolicy.resolve(
            requested: .auto,
            temperature: 0,
            proof: .init(
                exactGreedy: true,
                autoregressiveTokensPerSecond: 20,
                mtpTokensPerSecond: 22.9))

        XCTAssertEqual(decision, .autoregressive)
    }

    func testExplicitSamplingStaysAutoregressiveAndForcedMTPFailsClosed() throws {
        XCTAssertEqual(
            try Qwen38AccelerationPolicy.resolve(
                requested: .auto,
                temperature: 0.7,
                proof: .init(
                    exactGreedy: true,
                    autoregressiveTokensPerSecond: 20,
                    mtpTokensPerSecond: 40)),
            .autoregressive)

        XCTAssertThrowsError(
            try Qwen38AccelerationPolicy.resolve(
                requested: .mtp,
                temperature: 0.7,
                proof: .init(
                    exactGreedy: true,
                    autoregressiveTokensPerSecond: 20,
                    mtpTokensPerSecond: 40))) { error in
            XCTAssertEqual(error as? Qwen38AccelerationPolicy.Error, .samplingRequiresAutoregressive)
        }
    }

    func testForcedMTPRequiresExactGreedyParityAndMeasuredSpeed() {
        XCTAssertThrowsError(
            try Qwen38AccelerationPolicy.resolve(
                requested: .mtp,
                temperature: 0,
                proof: .init(
                    exactGreedy: false,
                    autoregressiveTokensPerSecond: 20,
                    mtpTokensPerSecond: 40))) { error in
            XCTAssertEqual(error as? Qwen38AccelerationPolicy.Error, .greedyParityNotProven)
        }
    }

    func testPipelineOptionsExposeQwenAccelerationWithoutChangingLegacyDefaults() {
        var options = CoreAIPipeline.Options()

        XCTAssertEqual(options.acceleration, .auto)
        XCTAssertEqual(options.temperature, 0)
        options.acceleration = .autoregressive
        XCTAssertEqual(options.acceleration, .autoregressive)
    }

    func testExecutionPolicyNeverSilentlyRunsUnimplementedNativeMTP() throws {
        let proof = Qwen38MTPProof(
            exactGreedy: true,
            autoregressiveTokensPerSecond: 20,
            mtpTokensPerSecond: 23)

        XCTAssertEqual(
            try Qwen38ExecutionPolicy.resolve(
                requested: .auto,
                temperature: 0,
                proof: proof,
                nativeMTPAvailable: false),
            .autoregressive)

        XCTAssertThrowsError(
            try Qwen38ExecutionPolicy.resolve(
                requested: .mtp,
                temperature: 0,
                proof: proof,
                nativeMTPAvailable: false)) { error in
            XCTAssertEqual(error as? Qwen38ExecutionPolicy.Error, .nativeMTPRunnerUnavailable)
        }
    }
}
