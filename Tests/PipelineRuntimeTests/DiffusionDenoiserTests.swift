import XCTest

@testable import PipelineRuntime

/// Tests for the host block-diffusion denoise loop (``DiffusionDenoiser``). They drive the loop
/// with a synthetic ``CanvasForward`` (no model), so they validate the accept/renoise/stop
/// mechanics deterministically and run in the standalone build.
final class DiffusionDenoiserTests: XCTestCase {

    // Deterministic SplitMix64 RNG (the runtime's SeededGenerator is gated behind COREAI_RUNTIME).
    private struct TestRNG: RandomNumberGenerator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// Synthetic forward: each canvas position `p` peaks at token `p % vocab` with a
    /// caller-controlled `peak` logit (sharper peak ⇒ lower entropy). Records the
    /// self-conditioning inputs seen per call.
    private struct SyntheticForward: CanvasForward {
        let vocab: Int
        let canvasLength: Int
        /// (position, step, t) → peak logit at the position's target token.
        let peak: (Int, Int, Double) -> Float
        var calls: [(step: Int, t: Double, scUse: Bool, scTempInv: Double, hadSelfCond: Bool)] = []

        mutating func forward(
            prompt: [Int32], canvas: [Int32], step: Int, t: Double,
            selfCond: [Float]?, scUse: Bool, scTempInv: Double
        ) async throws -> [Float] {
            calls.append((step, t, scUse, scTempInv, selfCond != nil))
            var out = [Float](repeating: 0, count: canvasLength * vocab)
            for p in 0..<canvasLength {
                out[p * vocab + (p % vocab)] = peak(p, step, t)
            }
            return out
        }
    }

    private func schedule(maxSteps: Int = 12, canvas: Int = 24) -> DiffusionSchedule {
        DiffusionSchedule(
            maxDenoisingSteps: maxSteps, tMax: 0.8, tMin: 0.4, entropyBound: 0.1,
            confidenceThreshold: 0.005, stabilityThreshold: 1, canvasLength: canvas)
    }

    // MARK: - t schedule

    func testTRampMatchesSpec() {
        let s = schedule(maxSteps: 48)
        XCTAssertEqual(s.t(at: 48), 0.8, accuracy: 1e-9)  // noisiest step == t_max
        XCTAssertEqual(s.t(at: 1), 0.4 + 0.4 * (1.0 / 48.0), accuracy: 1e-9)
        // Monotonic increasing in step.
        for step in 2...48 {
            XCTAssertGreaterThan(s.t(at: step), s.t(at: step - 1))
        }
    }

    // MARK: - Convergence (peaked logits ⇒ adaptive stop before maxSteps)

    func testConvergesOnPeakedLogits() async throws {
        let s = schedule()
        let denoiser = DiffusionDenoiser(schedule: s, vocabSize: 32)
        var fwd = SyntheticForward(vocab: 32, canvasLength: s.canvasLength) { _, _, _ in 40 }
        var rng = TestRNG(seed: 1)
        let result = try await denoiser.denoiseBlock(prompt: [1, 2, 3], forward: &fwd, rng: &rng)

        XCTAssertEqual(result.stopReason, .converged)
        XCTAssertLessThan(result.steps.count, s.maxDenoisingSteps, "should stop early")
        XCTAssertEqual(result.canvas.count, s.canvasLength)
        // Peaked + low entropy ⇒ every position accepted on the final step.
        XCTAssertEqual(result.finalAcceptedCount, s.canvasLength)
        // Committed readout recovers the per-position targets (p % vocab).
        for p in 0..<s.canvasLength {
            XCTAssertEqual(Int(result.canvas[p]), p % 32, "position \(p)")
        }
    }

    // MARK: - No convergence (uniform logits ⇒ run full schedule, accept minimum progress)

    func testStepsExhaustedOnUniformLogits() async throws {
        let s = schedule()
        let denoiser = DiffusionDenoiser(schedule: s, vocabSize: 32)
        var fwd = SyntheticForward(vocab: 32, canvasLength: s.canvasLength) { _, _, _ in 0 }  // flat
        var rng = TestRNG(seed: 7)
        let result = try await denoiser.denoiseBlock(prompt: [9], forward: &fwd, rng: &rng)

        XCTAssertEqual(result.stopReason, .stepsExhausted)
        XCTAssertEqual(result.steps.count, s.maxDenoisingSteps)
        // The official entropy-bound sampler accepts the lowest-entropy position even when every
        // position is flat, because the prior cumulative entropy before that token is zero.
        for info in result.steps {
            XCTAssertEqual(info.acceptedCount, 1)
            XCTAssertEqual(info.meanEntropy, log(32.0), accuracy: 1e-3)
        }
        XCTAssertEqual(result.canvas.count, s.canvasLength)
    }

    // MARK: - Self-conditioning gating

    func testSelfConditioningGating() async throws {
        let s = schedule()
        let denoiser = DiffusionDenoiser(schedule: s, vocabSize: 32)
        var fwd = SyntheticForward(vocab: 32, canvasLength: s.canvasLength) { _, _, _ in 0 }  // never converges
        var rng = TestRNG(seed: 3)
        _ = try await denoiser.denoiseBlock(prompt: [1], forward: &fwd, rng: &rng)

        XCTAssertEqual(fwd.calls.count, s.maxDenoisingSteps)
        // Step 0: self-conditioning gated off.
        XCTAssertFalse(fwd.calls[0].scUse)
        XCTAssertFalse(fwd.calls[0].hadSelfCond)
        // Subsequent steps: self-conditioning active, fed prev logits. The official sampler uses
        // raw logits for self-conditioning, so the inverse temperature stays at 1.0.
        XCTAssertTrue(fwd.calls[1].scUse)
        XCTAssertTrue(fwd.calls[1].hadSelfCond)
        XCTAssertEqual(fwd.calls[1].scTempInv, 1.0, accuracy: 1e-9)
        XCTAssertEqual(fwd.calls[2].scTempInv, 1.0, accuracy: 1e-9)
    }

    // MARK: - Accept count grows as entropy falls (annealing)

    func testAcceptGrowsAsEntropyFalls() async throws {
        let s = schedule(maxSteps: 12, canvas: 32)
        let denoiser = DiffusionDenoiser(schedule: s, vocabSize: 32)
        // Sharper peak at later (lower) steps ⇒ entropy falls ⇒ more positions accepted.
        var fwd = SyntheticForward(vocab: 32, canvasLength: s.canvasLength) { _, step, _ in
            Float(s.maxDenoisingSteps - step) * 1.5
        }
        var rng = TestRNG(seed: 5)
        var perStepAccepts: [Int] = []
        let result = try await denoiser.denoiseBlock(
            prompt: [1], forward: &fwd, rng: &rng, onStep: { perStepAccepts.append($0.acceptedCount) })

        XCTAssertEqual(result.canvas.count, s.canvasLength)
        XCTAssertFalse(perStepAccepts.isEmpty)
        // First step (flat) accepts few; a later step accepts strictly more.
        XCTAssertGreaterThan(perStepAccepts.max()!, perStepAccepts.first!)
    }
}
