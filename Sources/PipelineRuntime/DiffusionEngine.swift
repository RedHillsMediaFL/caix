import Accelerate
import Foundation

// MARK: - Diffusion schedule (pure; parsed from bundle metadata `diffusion` block)

/// Block-diffusion denoise schedule. Mirrors `EntropyBoundSamplerConfig` from
/// `diffusiongemma`'s generation config and the bundle metadata `diffusion` block.
///
/// Defaults match the documented `diffusiongemma-26B-A4B-it` schedule
/// (`max_denoising_steps 48, t_max 0.8, t_min 0.4, entropy_bound 0.1,
/// confidence_threshold 0.005, stability_threshold 1, canvas_length 256`).
public struct DiffusionSchedule: Sendable, Equatable {
    /// Number of denoise iterations per block (the loop runs `maxDenoisingSteps → 1`).
    public var maxDenoisingSteps: Int
    /// Noise level at the first (noisiest) step.
    public var tMax: Double
    /// Noise level approached at the last step.
    public var tMin: Double
    /// Cumulative entropy (MI) budget governing how many positions are accepted per step.
    public var entropyBound: Double
    /// Mean-entropy threshold for the adaptive stop.
    public var confidenceThreshold: Double
    /// Number of consecutive stable-argmax steps required for the adaptive stop.
    public var stabilityThreshold: Int
    /// Canvas length (tokens denoised in parallel per block).
    public var canvasLength: Int
    /// Fixed prompt-window length the model was exported with. The host left-pads a shorter
    /// conditioning prefix to this length and masks the padding via `key_pad_bias`. 0 means the
    /// prompt dimension is unconstrained (no padding) — e.g. a future dynamic export.
    public var promptLength: Int

    public init(
        maxDenoisingSteps: Int = 48,
        tMax: Double = 0.8,
        tMin: Double = 0.4,
        entropyBound: Double = 0.1,
        confidenceThreshold: Double = 0.005,
        stabilityThreshold: Int = 1,
        canvasLength: Int = 256,
        promptLength: Int = 0
    ) {
        self.maxDenoisingSteps = max(1, maxDenoisingSteps)
        self.tMax = tMax
        self.tMin = tMin
        self.entropyBound = entropyBound
        self.confidenceThreshold = confidenceThreshold
        self.stabilityThreshold = max(1, stabilityThreshold)
        self.canvasLength = max(1, canvasLength)
        self.promptLength = max(0, promptLength)
    }

    /// Noise level at a given step: `t = t_min + (t_max - t_min) * (step / maxSteps)`.
    /// `step == maxSteps → t_max` (noisiest); `step == 1 → ≈ t_min` (sharpest).
    public func t(at step: Int) -> Double {
        tMin + (tMax - tMin) * (Double(step) / Double(maxDenoisingSteps))
    }
}

// MARK: - Forward abstraction (decouples the loop from the model)

/// A single stateless bidirectional forward over `[prompt | canvas]`.
///
/// The denoise loop is host-driven and model-agnostic: it only needs a function that, given
/// the current prompt + canvas token ids and the self-conditioning inputs, returns the raw
/// **canvas** logits (one distribution per canvas position). The production implementation
/// (`DiffusionEngine`, behind `#if COREAI_RUNTIME`) drives the exported `.aimodel`; tests
/// inject a synthetic forward to exercise the accept/renoise/stop mechanics without a model.
public protocol CanvasForward {
    /// Run one forward.
    /// - Returns: raw canvas logits, row-major `[canvasLength * vocabSize]` (row `p` is the
    ///   distribution for canvas position `p`).
    /// - Parameters:
    ///   - prompt: committed prefix token ids (positions `0 ..< prompt.count`).
    ///   - canvas: current canvas token ids (positions `prompt.count ..< prompt.count + L`).
    ///   - step: current denoise step (`maxDenoisingSteps … 1`).
    ///   - t: noise level for this step.
    ///   - selfCond: previous step's raw canvas logits (`nil` on the first step).
    ///   - scUse: whether self-conditioning is active (`false` on the first step).
    ///   - scTempInv: `1 / t_prev` — the inverse temperature the model applies to `selfCond`.
    mutating func forward(
        prompt: [Int32],
        canvas: [Int32],
        step: Int,
        t: Double,
        selfCond: [Float]?,
        scUse: Bool,
        scTempInv: Double
    ) async throws -> [Float]
}

// MARK: - Per-step + per-block reporting

/// Diagnostics for one denoise step.
public struct DiffusionStepInfo: Sendable {
    public let step: Int
    public let t: Double
    /// Positions accepted this step (cumulative-entropy ≤ `entropyBound`).
    public let acceptedCount: Int
    public let meanEntropy: Double
    public let minEntropy: Double
    /// Consecutive stable-argmax steps so far (drives the adaptive stop).
    public let stableCount: Int
}

/// Why a single block's denoise loop ended.
public enum DiffusionStopReason: String, Sendable {
    /// Adaptive stop: argmax canvas stable ≥ `stabilityThreshold` and mean entropy below
    /// `confidenceThreshold`.
    case converged
    /// Ran the full `maxDenoisingSteps` schedule without converging.
    case stepsExhausted
}

/// Result of denoising one canvas block.
public struct DiffusionBlockResult: Sendable {
    /// The committed block (length `canvasLength`): accepted positions keep their sampled
    /// token, the rest resolve to the final step's argmax (the model's MAP estimate).
    public let canvas: [Int32]
    public let steps: [DiffusionStepInfo]
    public let stopReason: DiffusionStopReason
    /// Accepted positions on the final step.
    public let finalAcceptedCount: Int
}

// MARK: - The host denoise loop (pure Swift; no Core AI dependency)

/// Host-side block-diffusion denoiser. Owns the iterative loop math (per-position argmax +
/// entropy + multinomial sample, cumulative-MI-bound accept, renoise, adaptive stop) and is
/// fully decoupled from the model via ``CanvasForward`` — so it compiles and is unit-testable
/// in the standalone (non-Core AI) build.
public struct DiffusionDenoiser {
    public let schedule: DiffusionSchedule
    public let vocabSize: Int
    /// Token-id range used for the random canvas init + renoise (`0 ..< vocabSize` by default).
    public let randomTokenRange: Range<Int>
    /// Absorbing-state mask token id. When set (masked discrete diffusion — e.g. diffusiongemma's
    /// `<mask>`), the canvas is initialised to, and unaccepted positions are renoised back to, this
    /// token instead of random vocab ids. The model was trained to denoise *masked* canvases, so
    /// feeding random tokens is out-of-distribution and collapses the logits to near-uniform.
    public let maskTokenId: Int32?

    public init(
        schedule: DiffusionSchedule, vocabSize: Int,
        randomTokenRange: Range<Int>? = nil, maskTokenId: Int32? = nil
    ) {
        self.schedule = schedule
        self.vocabSize = vocabSize
        self.randomTokenRange = randomTokenRange ?? (0..<max(1, vocabSize))
        self.maskTokenId = maskTokenId
    }

    /// Denoise one canvas block conditioned on `prompt`.
    ///
    /// ```
    /// canvas = random tokens[L]; prev = nil
    /// for step in 48…1:
    ///     t = t_min + (t_max - t_min)*(step/48)
    ///     logits = forward(prompt, canvas, selfCond: prev, scUse: prev != nil)
    ///     per position: argmax, entropy(softmax(raw/t)), one multinomial sample
    ///     accept the lowest-entropy positions whose cumulative entropy ≤ entropy_bound
    ///     renoise the rest with fresh random tokens; prev = logits
    ///     stop if argmax canvas stable ≥ stability_threshold steps AND mean entropy < confidence_threshold
    /// ```
    public func denoiseBlock<F: CanvasForward>(
        prompt: [Int32],
        forward: inout F,
        rng: inout some RandomNumberGenerator,
        onStep: ((DiffusionStepInfo) -> Void)? = nil
    ) async throws -> DiffusionBlockResult {
        let L = schedule.canvasLength
        let V = vocabSize

        // Official `EntropyBoundSampler.initialize_canvas`: random tokens U(vocab). (NOT `<mask>` —
        // diffusiongemma is uniform-noise discrete diffusion; the canvas init and the renoise of
        // unaccepted positions both draw fresh random ids.)
        var canvas: [Int32] = (0..<L).map { _ in Int32(Int.random(in: randomTokenRange, using: &rng)) }
        var prev: [Float]? = nil
        var prevArgmax: [Int]? = nil
        var stableCount = 0

        var steps: [DiffusionStepInfo] = []
        steps.reserveCapacity(schedule.maxDenoisingSteps)

        // Final-readout state (updated every step; used to build the committed block).
        var lastArgmax = canvas.map(Int.init)
        var lastAccepted = [Bool](repeating: false, count: L)
        var lastSampled = canvas
        var stopReason: DiffusionStopReason = .stepsExhausted

        // Reused scratch buffers (avoid a V-sized alloc per position): `scaledScratch` holds the
        // temperature-scaled, max-shifted logits; `expScratch` their exponentials.
        var scaledScratch = [Float](repeating: 0, count: V)
        var expScratch = [Float](repeating: 0, count: V)

        for step in stride(from: schedule.maxDenoisingSteps, through: 1, by: -1) {
            // The official `EntropyBoundSampler` uses raw logits (no temperature) for both the entropy
            // measure and the multinomial sample, and feeds the previous step's logits as
            // self-conditioning at temperature 1.0 (t_max/t_min are unused by this sampler).
            let t = 1.0
            let scUse = prev != nil
            let scTempInv = 1.0

            let logits = try await forward.forward(
                prompt: prompt, canvas: canvas, step: step, t: t,
                selfCond: prev, scUse: scUse, scTempInv: scTempInv)
            precondition(
                logits.count == L * V,
                "canvas forward returned \(logits.count) logits, expected \(L * V) (L=\(L) V=\(V))")

            var entropy = [Double](repeating: 0, count: L)
            var argmaxIdx = [Int](repeating: 0, count: L)
            var sampled = [Int32](repeating: 0, count: L)

            logits.withUnsafeBufferPointer { lp in
              scaledScratch.withUnsafeMutableBufferPointer { sp in
                expScratch.withUnsafeMutableBufferPointer { ep in
                    for p in 0..<L {
                        let base = p * V
                        let (amax, ent, samp) = Self.positionStats(
                            logits: lp, base: base, vocab: V, t: t, scaled: sp, exps: ep, rng: &rng)
                        argmaxIdx[p] = amax
                        entropy[p] = ent
                        sampled[p] = Int32(samp)
                    }
                }
              }
            }

            // Official `EntropyBoundSampler.accept_canvas`: sort positions by entropy ascending and
            // accept position k while the sum of all *strictly-lower* entropies stays ≤ entropy_bound
            // (`cumulative_entropy − sorted_entropy ≤ bound`). The lowest-entropy position has a
            // prior-sum of 0, so it is ALWAYS accepted — the canvas commits ≥1 token every step and
            // makes progress even when early-step logits are flat. (Our previous `cumulative ≤ bound`
            // added the current token's own entropy before the test, so when every entropy > bound it
            // committed nothing and the canvas never converged — the root cause of the garbage output.)
            var accepted = [Bool](repeating: false, count: L)
            var acceptedCount = 0
            let order = (0..<L).sorted { entropy[$0] < entropy[$1] }
            var priorSum = 0.0
            for p in order {
                if priorSum <= schedule.entropyBound {
                    accepted[p] = true
                    acceptedCount += 1
                    priorSum += entropy[p]
                } else {
                    break
                }
            }

            // Next canvas (official `renoise_canvas`): accepted positions take their sampled token,
            // every non-accepted position is renoised with a fresh random id.
            var nextCanvas = [Int32](repeating: 0, count: L)
            for p in 0..<L {
                if accepted[p] {
                    nextCanvas[p] = sampled[p]
                } else {
                    nextCanvas[p] = Int32(Int.random(in: randomTokenRange, using: &rng))
                }
            }

            // Adaptive stop bookkeeping.
            let meanEntropy = entropy.reduce(0, +) / Double(L)
            let minEntropy = entropy.min() ?? 0
            if let pa = prevArgmax, pa == argmaxIdx {
                stableCount += 1
            } else {
                stableCount = 0
            }

            let info = DiffusionStepInfo(
                step: step, t: t, acceptedCount: acceptedCount,
                meanEntropy: meanEntropy, minEntropy: minEntropy, stableCount: stableCount)
            steps.append(info)
            onStep?(info)

            lastArgmax = argmaxIdx
            lastAccepted = accepted
            lastSampled = sampled
            prev = logits
            prevArgmax = argmaxIdx
            canvas = nextCanvas

            if stableCount >= schedule.stabilityThreshold
                && meanEntropy < schedule.confidenceThreshold
            {
                stopReason = .converged
                break
            }
        }

        // Committed block: accepted positions keep their sampled token; the rest take the
        // final step's argmax (the model's denoised MAP estimate for that position).
        var committed = [Int32](repeating: 0, count: L)
        var finalAccepted = 0
        for p in 0..<L {
            if lastAccepted[p] {
                committed[p] = lastSampled[p]
                finalAccepted += 1
            } else {
                committed[p] = Int32(lastArgmax[p])
            }
        }

        return DiffusionBlockResult(
            canvas: committed, steps: steps, stopReason: stopReason,
            finalAcceptedCount: finalAccepted)
    }

    /// Per-position statistics over one vocab row: argmax, Shannon entropy (nats) of
    /// `softmax(raw / t)`, and one multinomial sample from that distribution. `exps` is a
    /// caller-owned scratch buffer of length `vocab` reused across positions.
    static func positionStats(
        logits lp: UnsafeBufferPointer<Float>,
        base: Int,
        vocab V: Int,
        t: Double,
        scaled sp: UnsafeMutableBufferPointer<Float>,
        exps ep: UnsafeMutableBufferPointer<Float>,
        rng: inout some RandomNumberGenerator
    ) -> (argmax: Int, entropy: Double, sample: Int) {
        // Vectorised with Accelerate: the per-position softmax/entropy over a 262k-wide vocab
        // dominates the host denoise loop, so the scalar passes are replaced with vDSP/vForce.
        let invT = Float(1.0 / max(t, 1e-6))
        let src = lp.baseAddress! + base
        let s = sp.baseAddress!
        let e = ep.baseAddress!
        let n = vDSP_Length(V)

        // s = logit * invT
        var invTv = invT
        vDSP_vsmul(src, 1, &invTv, s, 1, n)

        // max(s) + argmax
        var maxScaled: Float = 0
        var argU: vDSP_Length = 0
        vDSP_maxvi(s, 1, &maxScaled, &argU, n)

        // s := s - maxScaled  (≤ 0, for a stable exp)
        var negMax = -maxScaled
        vDSP_vsadd(s, 1, &negMax, s, 1, n)

        // e := exp(s)
        var cnt = Int32(V)
        vvexpf(e, s, &cnt)

        // sumExp = Σ e ; sumExpScaled = Σ e·s
        var sumExp: Float = 0
        vDSP_sve(e, 1, &sumExp, n)
        var sumExpScaled: Float = 0
        vDSP_dotpr(e, 1, s, 1, &sumExpScaled, n)

        // entropy = -Σ p·ln p = ln(sumExp) - (Σ e·s)/sumExp   (nats)
        let sE = Double(max(sumExp, .leastNormalMagnitude))
        let entropy = max(0.0, Foundation.log(sE) - Double(sumExpScaled) / sE)

        // inverse-CDF multinomial sample over e (threshold r·sumExp; early-exit).
        let r = Float(Double.random(in: 0..<1, using: &rng)) * sumExp
        var cum: Float = 0
        var sample = V - 1
        for v in 0..<V {
            cum += e[v]
            if r < cum {
                sample = v
                break
            }
        }
        return (Int(argU), entropy, sample)
    }
}

#if COREAI_RUNTIME

import CoreAI
import Tokenizers

/// Native Core AI block-diffusion denoise engine for an exported **stateless** diffusion
/// `.aimodel` bundle (e.g. `diffusiongemma`).
///
/// Unlike ``LLMEngine`` (autoregressive prefill+decode over a KV cache), the diffusion model
/// program is a single **stateless bidirectional forward** over `[prompt | canvas]` with **no
/// KV cache**. The iterative denoise loop runs host-side in ``DiffusionDenoiser``; this engine
/// supplies the per-step forward (``CanvasForward``) by driving the `.aimodel`, and wraps the
/// block-autoregressive outer loop (trim at first EOG → commit canvas to prompt → denoise the
/// next block).
///
/// ## I/O contract (resolved from the function descriptor at load)
/// Inputs: `prompt_ids` (Int32 `[1, P]`), `canvas_ids` (Int32 `[1, L]`), `position_ids`
/// (Int32 `[1, P+L]`), and self-conditioning `self_cond` (prev raw canvas logits
/// `[1, L, vocab]`), `sc_use` (0/1), `sc_temp_inv` (`1/t_prev`). Output: `canvas_logits`
/// (`[1, L, vocab]`). Names are matched canonically with fall-back to declaration order, and
/// the resolved contract is logged in `--verbose` so it can be reconciled with the real export.
///
/// ## Statelessness
/// Per Apple's diffusion runtime, a stateless `InferenceFunction` can return stale buffers on
/// alternating `run` calls, so a **fresh** function is loaded for each forward.
final class DiffusionEngine {
    private let model: AIModel
    /// The `main` forward, loaded once and reused across every denoise step. Reloading it per
    /// forward (as a guard against stale stateless buffers) re-specialises the graph each call —
    /// catastrophic for a 30-layer model. The denoise loop is sequential and copies each output
    /// (`readCanvasLogits`) before issuing the next `run`, so buffer reuse is safe.
    private let function: InferenceFunction
    let tokenizer: any Tokenizer
    let schedule: DiffusionSchedule
    let vocabSize: Int
    let maxContextLength: Int
    let loadSeconds: Double

    // Resolved I/O names.
    private let promptIdsName: String
    private let canvasIdsName: String
    private let positionIdsName: String
    private let keyPadBiasName: String?
    private let selfCondName: String?
    private let scUseName: String?
    private let scTempInvName: String?
    private let canvasLogitsName: String

    /// Fixed prompt-window length the model expects (0 = unconstrained). When > 0, the host
    /// left-pads the conditioning prefix to this length and masks the padding via `key_pad_bias`.
    let promptLength: Int

    // Resolved I/O scalar types (filled to match the graph's declared dtypes).
    private let promptIdsType: NDArray.ScalarType
    private let canvasIdsType: NDArray.ScalarType
    private let positionIdsType: NDArray.ScalarType
    private let selfCondType: NDArray.ScalarType?
    private let scUseType: NDArray.ScalarType?
    private let scTempInvType: NDArray.ScalarType?
    private let scTempInvShape: [Int]
    private let scUseShape: [Int]
    private let keyPadBiasType: NDArray.ScalarType?
    private let canvasLogitsType: NDArray.ScalarType

    private let stopIds: Set<Int>

    static func load(bundle: ResolvedBundle, verbose: Bool = false) async throws -> DiffusionEngine {
        func log(_ s: @autoclosure () -> String) {
            if verbose { FileHandle.standardError.write(Data(("[coreai] " + s() + "\n").utf8)) }
        }

        guard var schedule = bundle.diffusion else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "bundle '\(bundle.name)' is not a diffusion bundle (no `diffusion` metadata)")
        }
        // Smoke/debug escape hatches: cap the schedule for a fast end-to-end loop check without
        // re-exporting. Defaults follow the bundle metadata.
        if let s = ProcessInfo.processInfo.environment["COREAI_DIFFUSION_MAX_STEPS"], let n = Int(s),
            n > 0
        {
            schedule.maxDenoisingSteps = n
        }
        if let c = ProcessInfo.processInfo.environment["COREAI_DIFFUSION_CANVAS"], let n = Int(c),
            n > 0
        {
            schedule.canvasLength = n
        }

        let loadStart = Date()
        log("loading diffusion model \(bundle.aimodelURL.lastPathComponent) + tokenizer …")

        async let tokenizerTask = AutoTokenizer.from(modelFolder: bundle.tokenizerDir)

        let specialization = SpecializationOptions(preferredComputeUnitKind: .gpu)
        // Specialize through the on-disk compilation cache: the diffusion graph's MPS compile is
        // minutes for the 30-layer MoE, so caching it (paid once) is the difference between a
        // tens-of-minutes first run and ~instant subsequent loads.
        let model = try await AIModel.specialize(
            contentsOf: bundle.aimodelURL, options: specialization,
            cache: .default, cachePolicy: .persistent)
        let tokenizer = try await tokenizerTask

        let loadSeconds = Date().timeIntervalSince(loadStart)
        let engine = try DiffusionEngine(
            model: model, tokenizer: tokenizer, schedule: schedule,
            vocabSize: bundle.vocabSize, maxContextLength: bundle.maxContextLength,
            loadSeconds: loadSeconds, verbose: verbose)
        log(String(format: "diffusion model+tokenizer ready in %.2fs", loadSeconds))
        return engine
    }

    private init(
        model: AIModel,
        tokenizer: any Tokenizer,
        schedule: DiffusionSchedule,
        vocabSize: Int,
        maxContextLength: Int,
        loadSeconds: Double,
        verbose: Bool
    ) throws {
        self.model = model
        self.tokenizer = tokenizer
        self.schedule = schedule
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.loadSeconds = loadSeconds

        guard let descriptor = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "function 'main' not found; have \(model.functionNames)")
        }
        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("could not load function 'main'")
        }
        self.function = fn

        let inputs = descriptor.inputNames
        guard descriptor.stateNames.isEmpty else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "diffusion forward must be stateless (no KV cache); got states \(descriptor.stateNames)")
        }
        guard !descriptor.outputNames.isEmpty else {
            throw CoreAIPipeline.RuntimeError.modelContract("expected >= 1 output (canvas_logits)")
        }

        // Resolve I/O by canonical names, falling back to declaration order for the required
        // trio (prompt_ids, canvas_ids, position_ids) and matching the optional self-cond
        // inputs by name only.
        self.promptIdsName = Self.pick(["prompt_ids", "input_ids"], inputs, fallbackIndex: 0)
        self.canvasIdsName = Self.pick(["canvas_ids", "canvas"], inputs, fallbackIndex: 1)
        self.positionIdsName = Self.pick(["position_ids", "positions"], inputs, fallbackIndex: 2)
        self.keyPadBiasName = Self.find(["key_pad_bias", "pad_bias", "padding_mask"], inputs)
        self.promptLength = schedule.promptLength
        self.selfCondName = Self.find(["self_cond", "self_conditioning", "prev_logits"], inputs)
        self.scUseName = Self.find(["sc_use", "self_cond_use", "use_self_cond"], inputs)
        self.scTempInvName = Self.find(["sc_temp_inv", "self_cond_temp_inv", "sc_inv_temp"], inputs)
        self.canvasLogitsName = Self.pick(
            ["canvas_logits", "logits", "output"], descriptor.outputNames, fallbackIndex: 0)

        func ndType(_ name: String) -> (NDArray.ScalarType, [Int])? {
            guard case .ndArray(let d) = descriptor.inputDescriptor(of: name) else { return nil }
            return (d.scalarType, d.shape)
        }
        func outType(_ name: String) -> NDArray.ScalarType? {
            guard case .ndArray(let d) = descriptor.outputDescriptor(of: name) else { return nil }
            return d.scalarType
        }

        self.promptIdsType = ndType(promptIdsName)?.0 ?? .int32
        self.canvasIdsType = ndType(canvasIdsName)?.0 ?? .int32
        self.positionIdsType = ndType(positionIdsName)?.0 ?? .int32
        self.selfCondType = selfCondName.flatMap { ndType($0)?.0 }
        self.scUseType = scUseName.flatMap { ndType($0)?.0 }
        if let n = scTempInvName, let (ty, sh) = ndType(n) {
            self.scTempInvType = ty
            self.scTempInvShape = sh.map { $0 < 0 ? 1 : $0 }
        } else {
            self.scTempInvType = nil
            self.scTempInvShape = [1]
        }
        if let n = scUseName, let (_, sh) = ndType(n) {
            self.scUseShape = sh.map { $0 < 0 ? 1 : $0 }
        } else {
            self.scUseShape = [1]
        }
        self.keyPadBiasType = keyPadBiasName.flatMap { ndType($0)?.0 }

        guard let outTy = outType(canvasLogitsName) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "output '\(canvasLogitsName)' is not an NDArray")
        }
        switch outTy {
        case .float16, .float32, .bfloat16:
            break
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "unsupported canvas_logits scalar type \(outTy) (expected float16/float32/bfloat16)")
        }
        self.canvasLogitsType = outTy

        self.stopIds = Self.stopTokenIds(tokenizer: tokenizer)

        if verbose {
            var lines = ["[coreai] diffusion I/O contract:"]
            lines.append(
                "  inputs: prompt_ids='\(promptIdsName)'(\(promptIdsType)) "
                    + "canvas_ids='\(canvasIdsName)'(\(canvasIdsType)) "
                    + "position_ids='\(positionIdsName)'(\(positionIdsType))")
            let scType = selfCondType.map { "\($0)" } ?? "?"
            lines.append(
                "  self-cond: self_cond=\(selfCondName.map { "'\($0)'(\(scType))" } ?? "absent") "
                    + "sc_use=\(scUseName.map { "'\($0)'" } ?? "absent") "
                    + "sc_temp_inv=\(scTempInvName.map { "'\($0)'" } ?? "absent")")
            lines.append(
                "  key_pad_bias: \(keyPadBiasName.map { "'\($0)'(\(keyPadBiasType.map { "\($0)" } ?? "?"))" } ?? "absent")"
                    + "  promptWindow=\(promptLength > 0 ? "\(promptLength)" : "dynamic")")
            lines.append("  output: canvas_logits='\(canvasLogitsName)'(\(canvasLogitsType))")
            lines.append(
                "  schedule: steps=\(schedule.maxDenoisingSteps) t=\(schedule.tMin)…\(schedule.tMax) "
                    + "entropyBound=\(schedule.entropyBound) conf=\(schedule.confidenceThreshold) "
                    + "stability=\(schedule.stabilityThreshold) canvas=\(schedule.canvasLength) vocab=\(vocabSize)")
            FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
        }
    }

    private static func pick(_ wanted: [String], _ names: [String], fallbackIndex: Int) -> String {
        for w in wanted where names.contains(w) { return w }
        return fallbackIndex < names.count ? names[fallbackIndex] : (names.first ?? wanted[0])
    }

    private static func find(_ wanted: [String], _ names: [String]) -> String? {
        for w in wanted where names.contains(w) { return w }
        return nil
    }

    // MARK: - Prompt encoding

    func encodePrompt(messages: [[String: String]], applyChatTemplate: Bool) throws -> [Int] {
        let tokens: [Int]
        if applyChatTemplate {
            tokens = try tokenizer.applyChatTemplate(messages: messages)
        } else {
            tokens = tokenizer.encode(text: messages.map { $0["content"] ?? "" }.joined())
        }
        guard !tokens.isEmpty else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("prompt tokenized to 0 tokens")
        }
        return tokens
    }

    // MARK: - Block-autoregressive generation

    /// Run the block-autoregressive denoise: denoise a canvas block, trim at the first EOG,
    /// commit it to the prompt, and repeat until EOS/EOG or `maxTokens`.
    func generate(
        promptTokens: [Int],
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.DiffusionResult {
        func log(_ s: @autoclosure () -> String) {
            if options.verbose { FileHandle.standardError.write(Data(("[coreai] " + s() + "\n").utf8)) }
        }

        // Resolve the absorbing `<mask>` token for masked discrete diffusion (diffusiongemma).
        // The model denoises masked canvases; without this the host feeds random ids (OOD) and the
        // logits collapse to near-uniform. nil → fall back to random-token noising.
        let maskTokenId = tokenizer.convertTokenToId("<mask>").map(Int32.init)
        log("diffusion mask token <mask>=\(maskTokenId.map(String.init) ?? "absent") "
            + "(masked-diffusion=\(maskTokenId != nil))")
        let denoiser = DiffusionDenoiser(
            schedule: schedule, vocabSize: vocabSize, maskTokenId: maskTokenId)
        var rng = SeededGenerator(seed: options.seed ?? UInt64.random(in: .min ... .max))

        var committed = promptTokens.map { Int32($0) }
        var generated: [Int] = []
        var streamedText = ""
        var blocks: [CoreAIPipeline.DiffusionResult.BlockSummary] = []
        var stopReason: CoreAIPipeline.StopReason = .maxTokens

        let start = Date()
        let maxTokens = max(0, options.maxTokens)

        blockLoop: while generated.count < maxTokens {
            let blockStart = Date()
            var forward = Forward(engine: self, verbose: options.verbose)
            let block = try await denoiser.denoiseBlock(
                prompt: committed, forward: &forward, rng: &rng,
                onStep: options.verbose
                    ? { info in
                        FileHandle.standardError.write(
                            Data(
                                String(
                                    format:
                                        "[coreai]   step %3d t=%.3f accepted=%4d meanH=%.4f minH=%.4f stable=%d\n",
                                    info.step, info.t, info.acceptedCount, info.meanEntropy,
                                    info.minEntropy, info.stableCount).utf8))
                    } : nil)

            // Trim the block at the first EOG/EOS, and to the remaining token budget.
            var commitCount = block.canvas.count
            var hitEOG = false
            for (i, tok) in block.canvas.enumerated() where stopIds.contains(Int(tok)) {
                commitCount = i
                hitEOG = true
                break
            }
            commitCount = min(commitCount, maxTokens - generated.count)
            let committedTokens = Array(block.canvas.prefix(commitCount))

            blocks.append(
                CoreAIPipeline.DiffusionResult.BlockSummary(
                    stepsRun: block.steps.count,
                    stopReason: block.stopReason.rawValue,
                    finalAcceptedCount: block.finalAcceptedCount,
                    committedTokens: committedTokens.count,
                    seconds: Date().timeIntervalSince(blockStart)))

            log(
                "block \(blocks.count): \(block.steps.count) steps, stop=\(block.stopReason.rawValue), "
                    + "finalAccepted=\(block.finalAcceptedCount), committed=\(committedTokens.count) tokens"
                    + (hitEOG ? " (EOG)" : ""))

            // Commit + stream.
            for tok in committedTokens {
                generated.append(Int(tok))
                committed.append(tok)
            }
            if let onToken {
                let text = tokenizer.decode(tokens: generated)
                if text.hasPrefix(streamedText) {
                    let delta = String(text.dropFirst(streamedText.count))
                    if !delta.isEmpty { onToken(delta) }
                }
                streamedText = text
            }

            if hitEOG {
                stopReason = .eos
                break blockLoop
            }
            if committedTokens.isEmpty {
                // No forward progress (e.g. immediate EOG or budget exhausted) — avoid spinning.
                stopReason = generated.count >= maxTokens ? .maxTokens : .eos
                break blockLoop
            }
            if committed.count >= maxContextLength {
                stopReason = .contextLimit
                break blockLoop
            }
        }

        let elapsed = Date().timeIntervalSince(start)
        let finalText = tokenizer.decode(tokens: generated)
        return CoreAIPipeline.DiffusionResult(
            text: finalText,
            promptTokenCount: promptTokens.count,
            generatedTokenCount: generated.count,
            stopReason: stopReason,
            modelLoadSeconds: loadSeconds,
            generateSeconds: elapsed,
            blocks: blocks)
    }

    func isStopToken(_ id: Int) -> Bool { stopIds.contains(id) }

    // MARK: - The per-step forward (CanvasForward conformance via a thin wrapper)

    /// Adapts the engine to ``CanvasForward``. A value type so the denoiser can hold it
    /// `inout` without retaining the engine through the protocol; it just forwards to the
    /// engine's `runForward`.
    struct Forward: CanvasForward {
        unowned let engine: DiffusionEngine
        let verbose: Bool

        mutating func forward(
            prompt: [Int32], canvas: [Int32], step: Int, t: Double,
            selfCond: [Float]?, scUse: Bool, scTempInv: Double
        ) async throws -> [Float] {
            try await engine.runForward(
                prompt: prompt, canvas: canvas, selfCond: selfCond, scUse: scUse,
                scTempInv: scTempInv)
        }
    }

    /// One stateless forward over `[prompt | canvas]`: builds the input NDArrays, loads a fresh
    /// `main` function, runs it, and reads the raw `canvas_logits` as a flat `[Float]`
    /// (`canvasLength * vocab`).
    private func runForward(
        prompt: [Int32], canvas: [Int32], selfCond: [Float]?, scUse: Bool, scTempInv: Double
    ) async throws -> [Float] {
        let L = canvas.count
        let V = vocabSize

        // Fixed prompt window: the model was exported with a `promptLength`-wide prompt window.
        // Left-pad a shorter prefix (so the real tokens sit adjacent to the canvas, keeping them
        // in the sliding-window's reach) or keep the most recent `promptLength` tokens when
        // longer. `key_pad_bias` masks the pad columns out of attention.
        let promptIds: [Int32]
        let positions: [Int32]
        var padCount = 0
        if promptLength > 0 {
            let P = promptLength
            let real = prompt.count <= P ? prompt : Array(prompt.suffix(P))
            let realLen = real.count
            padCount = P - realLen
            // Pad token id 0 (masked anyway). Real tokens flush-right in the window.
            promptIds = [Int32](repeating: 0, count: padCount) + real
            // Positions: pad → 0; real prefix → 0..<realLen; canvas → realLen..<realLen+L.
            var pos = [Int32](repeating: 0, count: P + L)
            for i in 0..<realLen { pos[padCount + i] = Int32(i) }
            for j in 0..<L { pos[P + j] = Int32(realLen + j) }
            positions = pos
        } else {
            promptIds = prompt
            positions = (0..<(prompt.count + L)).map { Int32($0) }
        }
        let P = promptIds.count

        var inputs: [String: NDArray] = [:]
        inputs[promptIdsName] = Self.intArray([1, P], scalarType: promptIdsType, values: promptIds)
        inputs[canvasIdsName] = Self.intArray([1, L], scalarType: canvasIdsType, values: canvas)
        inputs[positionIdsName] = Self.intArray(
            [1, P + L], scalarType: positionIdsType, values: positions)

        // Key-padding bias: additive `[1,1,1,P+L]`, large-negative at padded prompt key columns
        // `[0, padCount)`, 0 elsewhere. Masks pad tokens out of every query's attention.
        if let name = keyPadBiasName, let ty = keyPadBiasType {
            var bias = [Float](repeating: 0, count: P + L)
            for i in 0..<padCount { bias[i] = -1.0e4 }  // f16-safe; dominates attention scores
            var arr = NDArray(shape: [1, 1, 1, P + L], scalarType: ty)
            Self.fillFloat(&arr, scalarType: ty, values: bias)
            inputs[name] = arr
        }

        // Self-conditioning (optional inputs; gated off on the first step).
        if let name = selfCondName, let ty = selfCondType {
            var arr = NDArray(shape: [1, L, V], scalarType: ty)
            if scUse, let sc = selfCond {
                Self.fillFloat(&arr, scalarType: ty, values: sc)
            } else {
                Self.fillFloat(&arr, scalarType: ty, values: nil)  // zeros
            }
            inputs[name] = arr
        }
        if let name = scUseName, let ty = scUseType {
            inputs[name] = Self.scalarArray(
                scUseShape, scalarType: ty, value: scUse ? 1.0 : 0.0)
        }
        if let name = scTempInvName, let ty = scTempInvType {
            inputs[name] = Self.scalarArray(
                scTempInvShape, scalarType: ty, value: scUse ? scTempInv : 0.0)
        }

        // Reuse the once-loaded function (see `function`); the output is copied out below before
        // the next call, so reusing it across steps is safe and avoids per-step graph re-spec.
        var outputs = try await function.run(inputs: inputs)
        guard let out = outputs.remove(canvasLogitsName)?.ndArray else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "forward produced no '\(canvasLogitsName)' output")
        }
        let flat = readCanvasLogits(out, canvasLength: L, vocab: V)
        if ProcessInfo.processInfo.environment["COREAI_DEBUG_LOGITS"] != nil {
            // Diagnostic: shape/stride of the raw output + per-row stats for the first two canvas
            // rows. Uniform (all-equal) logits ⇒ entropy = ln(vocab); this localises whether the
            // model emits constant logits or the readback mis-slices.
            let shape = out.shape
            var mn = Float.greatestFiniteMagnitude, mx = -Float.greatestFiniteMagnitude, sum = 0.0
            for v in 0..<V { let x = flat[v]; mn = min(mn, x); mx = max(mx, x); sum += Double(x) }
            let r1mn = (V..<2 * V).reduce(Float.greatestFiniteMagnitude) { min($0, flat[$1]) }
            let r1mx = (V..<2 * V).reduce(-Float.greatestFiniteMagnitude) { max($0, flat[$1]) }
            FileHandle.standardError.write(Data(
                ("[coreai] DEBUG logits: outShape=\(shape) flatCount=\(flat.count) "
                    + "row0[min=\(mn) max=\(mx) mean=\(Float(sum / Double(V)))] "
                    + "row1[min=\(r1mn) max=\(r1mx)] argmax0=\(flat.prefix(V).enumerated().max{$0.1<$1.1}?.0 ?? -1)\n").utf8))
        }
        return flat
    }

    // MARK: - NDArray helpers (dtype-aware)

    private static func intArray(
        _ shape: [Int], scalarType: NDArray.ScalarType, values: [Int32]
    ) -> NDArray {
        var arr = NDArray(shape: shape, scalarType: scalarType)
        switch scalarType {
        case .int32, .uint32:
            var view = arr.mutableView(as: Int32.self)
            view.copyElements(fromContentsOf: values)
        case .int64, .uint64:
            var view = arr.mutableView(as: Int64.self)
            view.copyElements(fromContentsOf: values.map(Int64.init))
        default:
            var view = arr.mutableView(as: Int32.self)
            view.copyElements(fromContentsOf: values)
        }
        return arr
    }

    /// A small NDArray (typically `[1]`) filled with a single broadcast `value`.
    private static func scalarArray(
        _ shape: [Int], scalarType: NDArray.ScalarType, value: Double
    ) -> NDArray {
        let count = shape.reduce(1, *)
        var arr = NDArray(shape: shape, scalarType: scalarType)
        switch scalarType {
        case .float32:
            var view = arr.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { p, _, _ in for i in 0..<count { p[i] = Float(value) } }
        case .float16:
            var view = arr.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { p, _, _ in for i in 0..<count { p[i] = Float16(value) } }
        case .int32, .uint32:
            var view = arr.mutableView(as: Int32.self)
            view.withUnsafeMutablePointer { p, _, _ in for i in 0..<count { p[i] = Int32(value) } }
        case .bool, .int8, .uint8:
            var view = arr.mutableView(as: UInt8.self)
            view.withUnsafeMutablePointer { p, _, _ in for i in 0..<count { p[i] = value != 0 ? 1 : 0 } }
        default:
            var view = arr.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { p, _, _ in for i in 0..<count { p[i] = Float(value) } }
        }
        return arr
    }

    /// Fill a float NDArray from `values` (row-major), or zero it when `values == nil`.
    private static func fillFloat(
        _ arr: inout NDArray, scalarType: NDArray.ScalarType, values: [Float]?
    ) {
        let count = arr.shape.reduce(1, *)
        switch scalarType {
        case .float32:
            var view = arr.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { p, _, _ in
                if let v = values { for i in 0..<min(count, v.count) { p[i] = v[i] } }
                else { for i in 0..<count { p[i] = 0 } }
            }
        case .float16:
            var view = arr.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { p, _, _ in
                if let v = values { for i in 0..<min(count, v.count) { p[i] = Float16(v[i]) } }
                else { for i in 0..<count { p[i] = 0 } }
            }
        default:
            break  // unsupported self-cond dtype: leave zero-initialised
        }
    }

    /// Read `canvas_logits` (`[1, rows, vocab]`) as a flat `[Float]` of `canvasLength * vocab`,
    /// taking the **last** `canvasLength` rows (the canvas span) and honoring strides.
    private func readCanvasLogits(_ array: NDArray, canvasLength L: Int, vocab V: Int) -> [Float] {
        switch canvasLogitsType {
        case .float32:
            return Self.readRows(array, as: Float.self, lastRows: L, vocab: V)
        case .bfloat16:
            return Self.readRowsBF16Bytes(array, lastRows: L, vocab: V)
        default:
            return Self.readRows(array, as: Float16.self, lastRows: L, vocab: V)
        }
    }

    /// Read **BFloat16** `canvas_logits` via the raw byte span (`NDArray.bytes`), since BFloat16
    /// has no public Swift scalar for a typed view. Each 2-byte element is a bf16 value; widen to
    /// Float by placing its bit pattern in the high 16 bits of a float32. Assumes the output is
    /// contiguous row-major `[…, rows, vocab]` (true for the fresh canvas-logits output) and takes
    /// the last `canvasLength` rows.
    private static func readRowsBF16Bytes(_ array: NDArray, lastRows L: Int, vocab V: Int) -> [Float] {
        let rv = array.rawView()
        let raw = rv.bytes  // RawSpan over the contiguous bf16 elements (2 bytes each)
        let totalElems = raw.byteCount / 2
        let rows = V > 0 ? totalElems / V : 0
        let startRow = max(0, rows - L)
        let nRows = min(L, max(0, rows))
        var out = [Float](repeating: 0, count: L * V)
        for r in 0..<nRows {
            let srcBase = (startRow + r) * V
            let dst = r * V
            for v in 0..<V {
                let bits = raw.unsafeLoadUnaligned(fromByteOffset: (srcBase + v) * 2, as: UInt16.self)
                out[dst + v] = Float(bitPattern: UInt32(bits) << 16)
            }
        }
        return out
    }

    private static func readRows<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: NDArray, as _: T.Type, lastRows L: Int, vocab V: Int
    ) -> [Float] {
        array.view(as: T.self).withUnsafePointer { ptr, shape, strides in
            let rank = shape.count
            let rows = rank >= 2 ? shape[rank - 2] : 1
            let rowStride = rank >= 2 ? strides[rank - 2] : V
            let colStride = strides[rank - 1]
            let startRow = max(0, rows - L)
            let nRows = min(L, rows)
            var out = [Float](repeating: 0, count: L * V)
            for r in 0..<nRows {
                let base = (startRow + r) * rowStride
                let dst = r * V
                for v in 0..<V { out[dst + v] = Float(ptr[base + v * colStride]) }
            }
            return out
        }
    }

    private static func stopTokenIds(tokenizer: any Tokenizer) -> Set<Int> {
        var ids = Set<Int>()
        if let eos = tokenizer.eosTokenId { ids.insert(eos) }
        for token in ["<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>", "</s>"] {
            if let id = tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        return ids
    }
}

#endif
