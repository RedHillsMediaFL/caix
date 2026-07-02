import Foundation

#if COREAI_RUNTIME
import CoreAI
#endif

/// Native Apple Core AI inference core for exported `.aimodel` LLM bundles.
///
/// Building with `COREAI_RUNTIME=1` on Xcode 27+ links the `CoreAI` system framework and
/// HuggingFace `Transformers`, and routes generation through a from-scratch prefill+decode
/// loop (`LLMEngine`) that mirrors the verified runtime contract of the exported bundle
/// (see `python/runner/run_aimodel.py`). With the flag unset everything below the facade is
/// compiled out and `run` throws `.runtimeUnavailable`, so the rest of the package keeps
/// building on stock toolchains.
public enum CoreAIPipeline {
    /// True when the package was compiled against Apple's `CoreAI` runtime.
    public static var isLinked: Bool {
        #if COREAI_RUNTIME
        true
        #else
        false
        #endif
    }

    // MARK: - Options

    /// Generation options for a single `run`.
    public struct Options: Sendable {
        /// Maximum number of tokens to generate.
        public var maxTokens: Int
        /// Sampling temperature. `0` selects deterministic greedy/argmax sampling.
        public var temperature: Double
        /// Top-K filter (optional, only used when `temperature > 0`).
        public var topK: Int?
        /// Top-P / nucleus filter (optional, only used when `temperature > 0`).
        public var topP: Double?
        /// Apply the tokenizer's chat template (`add_generation_prompt`) to the prompt.
        /// When `false` the prompt is tokenized verbatim (raw completion).
        public var applyChatTemplate: Bool
        /// Optional fixed KV-cache capacity (tokens). Defaults to
        /// `min(promptLen + maxTokens + 8, maxContextLength)`.
        public var kvCapacity: Int?
        /// Optional decoded-text stop strings. Generation stops once the continuation contains
        /// one of these strings; the matched stop text is not emitted.
        public var stopSequences: [String]
        /// RNG seed for reproducible temperature sampling. `nil` = system RNG.
        public var seed: UInt64?
        /// Emit progress/diagnostics to stderr.
        public var verbose: Bool

        public init(
            maxTokens: Int = 64,
            temperature: Double = 0,
            topK: Int? = nil,
            topP: Double? = nil,
            applyChatTemplate: Bool = true,
            kvCapacity: Int? = nil,
            stopSequences: [String] = [],
            seed: UInt64? = nil,
            verbose: Bool = false
        ) {
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topK = topK
            self.topP = topP
            self.applyChatTemplate = applyChatTemplate
            self.kvCapacity = kvCapacity
            self.stopSequences = stopSequences
            self.seed = seed
            self.verbose = verbose
        }
    }

    // MARK: - Result

    /// Why generation stopped.
    public enum StopReason: String, Sendable {
        case eos
        case maxTokens
        case contextLimit
        case stopSequence
    }

    /// Outcome of a `run`.
    public struct Result: Sendable {
        public let text: String
        public let promptTokenCount: Int
        public let generatedTokenCount: Int
        public let stopReason: StopReason
        public let modelLoadSeconds: Double
        public let prefillSeconds: Double
        public let decodeSeconds: Double

        public var decodeTokensPerSecond: Double {
            decodeSeconds > 0 ? Double(generatedTokenCount) / decodeSeconds : 0
        }

        public init(
            text: String,
            promptTokenCount: Int,
            generatedTokenCount: Int,
            stopReason: StopReason,
            modelLoadSeconds: Double,
            prefillSeconds: Double,
            decodeSeconds: Double
        ) {
            self.text = text
            self.promptTokenCount = promptTokenCount
            self.generatedTokenCount = generatedTokenCount
            self.stopReason = stopReason
            self.modelLoadSeconds = modelLoadSeconds
            self.prefillSeconds = prefillSeconds
            self.decodeSeconds = decodeSeconds
        }
    }

    // MARK: - Stop sequence helpers

    static func firstStopRange(in text: String, stopSequences: [String]) -> Range<String.Index>? {
        var best: Range<String.Index>?
        for stop in stopSequences where !stop.isEmpty {
            guard let range = text.range(of: stop) else { continue }
            if let current = best {
                if range.lowerBound < current.lowerBound { best = range }
            } else {
                best = range
            }
        }
        return best
    }

    static func visibleTextAvoidingPartialStop(_ text: String, stopSequences: [String]) -> String {
        guard !text.isEmpty else { return text }
        var holdback = 0
        for stop in stopSequences where stop.count > 1 {
            let limit = min(stop.count - 1, text.count)
            if limit <= 0 { continue }
            for n in 1...limit where stop.hasPrefix(String(text.suffix(n))) {
                holdback = max(holdback, n)
            }
        }
        return holdback > 0 ? String(text.dropLast(holdback)) : text
    }

    // MARK: - Speculative-decoding result

    /// Outcome of a speculative (draft-model) `runSpeculative`. Carries the same fields as
    /// ``Result`` plus draft-acceptance accounting. The decoded text is **identical** to what the
    /// target model would have produced greedily on its own — speculation only changes *speed*.
    public struct SpeculativeResult: Sendable {
        public let text: String
        public let promptTokenCount: Int
        public let generatedTokenCount: Int
        public let stopReason: StopReason
        /// Max load time across the target + draft handles.
        public let modelLoadSeconds: Double
        public let prefillSeconds: Double
        public let decodeSeconds: Double
        /// Number of draft tokens proposed (K) per verification step.
        public let draftTokens: Int
        /// Total draft tokens proposed across the run.
        public let draftedTokens: Int
        /// Of those, how many the target accepted (matched its own greedy choice).
        public let acceptedDraftTokens: Int
        /// Number of target verification passes (speculative iterations).
        public let iterations: Int

        public var decodeTokensPerSecond: Double {
            decodeSeconds > 0 ? Double(generatedTokenCount) / decodeSeconds : 0
        }
        /// Fraction of proposed draft tokens that were accepted, in `0...1`.
        public var acceptanceRate: Double {
            draftedTokens > 0 ? Double(acceptedDraftTokens) / Double(draftedTokens) : 0
        }
        /// Mean tokens committed per target forward pass (≈ `1 + acceptanceRate·K`); the
        /// theoretical decode speedup vs running the target alone.
        public var tokensPerTargetForward: Double {
            iterations > 0 ? Double(generatedTokenCount) / Double(iterations) : 0
        }

        public init(
            text: String,
            promptTokenCount: Int,
            generatedTokenCount: Int,
            stopReason: StopReason,
            modelLoadSeconds: Double,
            prefillSeconds: Double,
            decodeSeconds: Double,
            draftTokens: Int,
            draftedTokens: Int,
            acceptedDraftTokens: Int,
            iterations: Int
        ) {
            self.text = text
            self.promptTokenCount = promptTokenCount
            self.generatedTokenCount = generatedTokenCount
            self.stopReason = stopReason
            self.modelLoadSeconds = modelLoadSeconds
            self.prefillSeconds = prefillSeconds
            self.decodeSeconds = decodeSeconds
            self.draftTokens = draftTokens
            self.draftedTokens = draftedTokens
            self.acceptedDraftTokens = acceptedDraftTokens
            self.iterations = iterations
        }
    }

    // MARK: - Diffusion result

    /// Outcome of a block-diffusion ``runDiffusion``. The text is the detokenized committed
    /// canvas across blocks; `blocks` carries the per-block denoise diagnostics (steps run,
    /// adaptive-stop reason, accept counts) that the loop validation reports.
    public struct DiffusionResult: Sendable {
        /// Per-block denoise summary.
        public struct BlockSummary: Sendable {
            public let stepsRun: Int
            public let stopReason: String
            public let finalAcceptedCount: Int
            public let committedTokens: Int
            public let seconds: Double

            public init(
                stepsRun: Int, stopReason: String, finalAcceptedCount: Int,
                committedTokens: Int, seconds: Double
            ) {
                self.stepsRun = stepsRun
                self.stopReason = stopReason
                self.finalAcceptedCount = finalAcceptedCount
                self.committedTokens = committedTokens
                self.seconds = seconds
            }
        }

        public let text: String
        public let promptTokenCount: Int
        public let generatedTokenCount: Int
        public let stopReason: StopReason
        public let modelLoadSeconds: Double
        public let generateSeconds: Double
        public let blocks: [BlockSummary]

        /// Total denoise steps run across all blocks.
        public var totalSteps: Int { blocks.reduce(0) { $0 + $1.stepsRun } }

        public init(
            text: String,
            promptTokenCount: Int,
            generatedTokenCount: Int,
            stopReason: StopReason,
            modelLoadSeconds: Double,
            generateSeconds: Double,
            blocks: [BlockSummary]
        ) {
            self.text = text
            self.promptTokenCount = promptTokenCount
            self.generatedTokenCount = generatedTokenCount
            self.stopReason = stopReason
            self.modelLoadSeconds = modelLoadSeconds
            self.generateSeconds = generateSeconds
            self.blocks = blocks
        }
    }

    // MARK: - Errors

    public enum RuntimeError: Error, CustomStringConvertible {
        case runtimeUnavailable
        case bundleNotFound(String)
        case invalidBundle(String)
        case modelContract(String)

        public var description: String {
            switch self {
            case .runtimeUnavailable:
                return
                    "Apple Core AI runtime is not linked. Rebuild with COREAI_RUNTIME=1 on Xcode 27+ "
                    + "(macOS 27) to run native inference."
            case .bundleNotFound(let p):
                return "Model bundle not found: \(p)"
            case .invalidBundle(let m):
                return "Invalid model bundle: \(m)"
            case .modelContract(let m):
                return "Model does not match the expected LLM runtime contract: \(m)"
            }
        }
    }

    // MARK: - Entry point

    #if COREAI_RUNTIME
    /// Apple's CoreAILanguageModels fast engine currently warms language bundles with a fixed
    /// cache shape that is too small for qwen3_5-style recurrent-state packing. Keep those bundles
    /// on the explicit sequential engine unless the caller opts into the experimental fast path.
    private static func shouldTryFastLanguagePath(modelPath: String, verbose: Bool) -> Bool {
        let env = ProcessInfo.processInfo.environment
        guard env["COREAI_LEGACY_ENGINE"] == nil else { return false }
        if env["COREAI_FAST_HYBRID_ENGINE"] != nil { return true }
        guard let bundle = try? ResolvedBundle.load(at: modelPath) else { return true }
        if bundle.minKVCapacity > 0 {
            if verbose {
                FileHandle.standardError.write(
                    Data(
                        "[fast] skipping CoreAILanguageModels path: bundle requires KV floor \(bundle.minKVCapacity)\n"
                            .utf8))
            }
            return false
        }
        return true
    }
    #endif

    /// Load `modelPath` (an exported `.aimodel` bundle directory), tokenize `prompt`, run
    /// prefill + decode natively, and return the generated text. Decoded text deltas are
    /// streamed to `onToken` as they are produced.
    @discardableResult
    public static func run(
        modelPath: String,
        prompt: String,
        options: Options = Options(),
        onToken: ((String) -> Void)? = nil
    ) async throws -> Result {
        #if COREAI_RUNTIME
        // Fast path: drive LLM generation through Apple's pipelined engine. Returns nil for
        // diffusion / non-language bundles, which fall through to `LLMEngine` (diffusion denoise +
        // the legacy sequential decode). `COREAI_LEGACY_ENGINE=1` forces the old path.
        if shouldTryFastLanguagePath(modelPath: modelPath, verbose: options.verbose) {
            if let fast = try await PipelinedLLM.runIfLanguage(
                modelPath: modelPath, prompt: prompt, options: options, onToken: onToken) {
                return fast
            }
        }
        return try await LLMEngine.run(
            modelPath: modelPath, prompt: prompt, options: options, onToken: onToken)
        #else
        _ = (modelPath, prompt, options, onToken)
        throw RuntimeError.runtimeUnavailable
        #endif
    }

    /// Speculative decoding: load `targetPath` and `draftPath` as two independent engines, let
    /// the small draft propose `draftTokens` (K) tokens per step, and have the target verify them
    /// in one batched forward — accepting the longest greedy-matching prefix and correcting the
    /// first divergence. Decoded text deltas stream to `onToken`. With greedy sampling
    /// (`temperature == 0`) the output is identical to ``run`` on the target alone.
    @discardableResult
    public static func runSpeculative(
        targetPath: String,
        draftPath: String,
        prompt: String,
        options: Options = Options(),
        draftTokens: Int = 4,
        onToken: ((String) -> Void)? = nil
    ) async throws -> SpeculativeResult {
        #if COREAI_RUNTIME
        let engine = try await SpeculativeEngine.load(
            targetPath: targetPath, draftPath: draftPath, draftTokens: draftTokens,
            verbose: options.verbose)
        let promptTokens = try engine.encodePrompt(
            messages: [["role": "user", "content": prompt]],
            applyChatTemplate: options.applyChatTemplate)
        return try await engine.generate(
            promptTokens: promptTokens, options: options, onToken: onToken)
        #else
        _ = (targetPath, draftPath, prompt, options, draftTokens, onToken)
        throw RuntimeError.runtimeUnavailable
        #endif
    }

    /// Micro-benchmark: steady-state cost of one sequential-engine forward over `n` tokens, for
    /// each `n` in `seqLengths`. Each shape is warmed up (the sequential engine recompiles the MPS
    /// graph per new input length, a one-time cost that must be excluded), then timed `iters` times
    /// with `rollbackKV(to: 0)` so every iteration is a fresh offset-0 forward of the same shape.
    /// Used to settle whether an EAGLE/speculative K+1 verify amortizes on the MoE target (does
    /// cost(K+1) ≈ cost(1), or ≈ (K+1)·cost(1)?).
    @discardableResult
    public static func benchForward(
        modelPath: String,
        seqLengths: [Int] = [1, 2, 4, 7, 1],
        warmup: Int = 4,
        iters: Int = 10
    ) async throws -> [(seq: Int, medianMs: Double)] {
        #if COREAI_RUNTIME
        func emit(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        emit("[bench] forward micro-benchmark; not decode tok/s")
        let bundle = try ResolvedBundle.load(at: modelPath)
        let engine = try await LLMEngine.load(bundle: bundle, verbose: true)
        let cap = (seqLengths.max() ?? 8) + 8
        try engine.allocateKVCache(capacity: cap)

        func timeSeq(_ n: Int) async throws -> Double {
            let toks = (0..<n).map { Int32(100 + $0) }
            for _ in 0..<warmup {
                engine.rollbackKV(to: 0)
                _ = try await engine.forwardAllRows(tokens: toks)
            }
            var ms: [Double] = []
            for _ in 0..<iters {
                engine.rollbackKV(to: 0)
                let t0 = Date()
                _ = try await engine.forwardAllRows(tokens: toks)
                ms.append(Date().timeIntervalSince(t0) * 1000)
            }
            ms.sort()
            return ms[ms.count / 2]
        }

        var out: [(Int, Double)] = []
        for n in seqLengths {
            let m = try await timeSeq(n)
            out.append((n, m))
            emit(String(format: "[bench] seq=%d: %7.2f ms (median of %d)", n, m, iters))
        }
        if let base = out.first(where: { $0.0 == 1 })?.1, base > 0 {
            emit("[bench] --- steady-state forward cost vs seq=1 ---")
            for (n, m) in out where n != 1 || out.filter({ $0.0 == 1 }).count == 1 {
                emit(String(format: "[bench]   seq=%d: %.2fx  (%.1f ms)", n, m / base, m))
            }
        }
        return out.map { (seq: $0.0, medianMs: $0.1) }
        #else
        _ = (modelPath, seqLengths, warmup, iters)
        throw RuntimeError.runtimeUnavailable
        #endif
    }

    /// Describe the Core AI function IO for an exported bundle. This intentionally specializes
    /// the on-disk assets, matching the runtime path, because descriptor shape/state information
    /// is only reliable after specialization.
    public static func inspectBundle(modelPath: String) async throws -> String {
        #if COREAI_RUNTIME
        var specialization = SpecializationOptions(preferredComputeUnitKind: LLMEngine.preferredComputeUnit())
        specialization.expectFrequentReshapes = true

        func appendFunction(
            role: String, name: String, model: AIModel, lines: inout [String]
        ) throws {
            guard let desc = model.functionDescriptor(for: name) else {
                lines.append("[\(role)] function '\(name)' not found; have \(model.functionNames)")
                return
            }
            lines.append("[\(role)] function=\(name)")
            lines.append("  inputs=\(desc.inputNames)")
            for input in desc.inputNames {
                switch desc.inputDescriptor(of: input) {
                case .ndArray(let nd):
                    lines.append("    input \(input): ndarray scalar=\(nd.scalarType) shape=\(nd.shape)")
                case .some(let descriptor):
                    lines.append("    input \(input): \(String(describing: descriptor))")
                case .none:
                    lines.append("    input \(input): <missing>")
                }
            }
            lines.append("  outputs=\(desc.outputNames)")
            for output in desc.outputNames {
                switch desc.outputDescriptor(of: output) {
                case .ndArray(let nd):
                    lines.append("    output \(output): ndarray scalar=\(nd.scalarType) shape=\(nd.shape)")
                case .some(let descriptor):
                    lines.append("    output \(output): \(String(describing: descriptor))")
                case .none:
                    lines.append("    output \(output): <missing>")
                }
            }
            lines.append("  states=\(desc.stateNames)")
            for state in desc.stateNames {
                switch desc.stateDescriptor(of: state) {
                case .ndArray(let nd):
                    lines.append("    state \(state): ndarray scalar=\(nd.scalarType) shape=\(nd.shape)")
                case .some(let descriptor):
                    lines.append("    state \(state): \(String(describing: descriptor))")
                case .none:
                    lines.append("    state \(state): <missing>")
                }
            }
        }

        let expanded = (modelPath as NSString).expandingTildeInPath
        let root = URL(fileURLWithPath: expanded, isDirectory: true)
        let fm = FileManager.default

        func inspectRawAimodel(role: String, url: URL, lines: inout [String]) async throws {
            let model = try await AIModel.specialize(
                contentsOf: url, options: specialization,
                cache: .default, cachePolicy: .persistent)
            let functionName = model.functionNames.contains("main") ? "main" : (model.functionNames.first ?? "main")
            try appendFunction(role: role, name: functionName, model: model, lines: &lines)
        }

        let eagleTarget = root.appendingPathComponent("eagle_target.aimodel")
        let eagleDraft = root.appendingPathComponent("eagle_draft.aimodel")
        if fm.fileExists(atPath: eagleTarget.path), fm.fileExists(atPath: eagleDraft.path) {
            var lines = [
                "bundle=\(root.path)",
                "package=eagle-mtp",
                "target_asset=\(eagleTarget.lastPathComponent)",
                "draft_asset=\(eagleDraft.lastPathComponent)"
            ]
            try await inspectRawAimodel(role: "target", url: eagleTarget, lines: &lines)
            try await inspectRawAimodel(role: "draft", url: eagleDraft, lines: &lines)
            let unrolled = root.appendingPathComponent("eagle_draft_unrolled_k4.aimodel")
            if fm.fileExists(atPath: unrolled.path) {
                lines.append("draft_unrolled_asset=\(unrolled.lastPathComponent)")
                try await inspectRawAimodel(role: "draft_unrolled", url: unrolled, lines: &lines)
            }
            return lines.joined(separator: "\n")
        }

        if root.pathExtension == "aimodel" {
            var lines = [
                "bundle=\(root.path)",
                "main_asset=\(root.lastPathComponent)"
            ]
            try await inspectRawAimodel(role: "main", url: root, lines: &lines)
            return lines.joined(separator: "\n")
        }

        let bundle = try ResolvedBundle.load(at: modelPath)
        let mainModel = try await AIModel.specialize(
            contentsOf: bundle.aimodelURL, options: specialization,
            cache: .default, cachePolicy: .persistent)
        let decodeModel: AIModel?
        if let decodeURL = bundle.decodeAimodelURL {
            decodeModel = try await AIModel.specialize(
                contentsOf: decodeURL, options: specialization,
                cache: .default, cachePolicy: .persistent)
        } else {
            decodeModel = nil
        }

        let functionMap = bundle.manifest.language?.functionMap
        let mainName = functionMap?.name(for: "main") ?? "main"
        let decodeName = functionMap?.name(for: "decode")
        var lines = [
            "bundle=\(bundle.root.path)",
            "main_asset=\(bundle.aimodelURL.lastPathComponent)"
        ]
        if let decodeURL = bundle.decodeAimodelURL {
            lines.append("decode_asset=\(decodeURL.lastPathComponent)")
        }
        try appendFunction(role: "main", name: mainName, model: mainModel, lines: &lines)
        if let decodeName {
            try appendFunction(
                role: "decode", name: decodeName, model: decodeModel ?? mainModel,
                lines: &lines)
        }
        return lines.joined(separator: "\n")
        #else
        _ = modelPath
        throw RuntimeError.runtimeUnavailable
        #endif
    }

    /// EAGLE / MTP speculative decoding (Gemma-4). The target (`Gemma4EagleTarget`, 6-output) and
    /// draft (`Gemma4AssistantForCausalLM`, cross-attends the target KV) are raw `.aimodel`s with
    /// custom contracts, so they are driven directly (not via `ResolvedBundle`). Greedy: output is
    /// byte-identical to the target alone; metrics report acceptance and decode throughput.
    @discardableResult
    public static func runEagle(
        targetAimodel: String,
        draftAimodel: String,
        tokenizerDir: String,
        prompt: String,
        options: Options = Options(),
        draftTokens: Int = 7,
        vocabSize: Int = 262144,
        backbone: Int = 2816,
        slidingWindow: Int = 1024,
        maxContext: Int = 4096,
        targetOnly: Bool = false,
        draftUnrolledAimodel: String? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> SpeculativeResult {
        #if COREAI_RUNTIME
        let engine = try await EagleEngine.load(
            targetURL: URL(fileURLWithPath: targetAimodel),
            draftURL: URL(fileURLWithPath: draftAimodel),
            tokenizerDir: URL(fileURLWithPath: tokenizerDir),
            draftTokens: draftTokens, vocabSize: vocabSize, backbone: backbone,
            slidingWindow: slidingWindow, maxContext: maxContext, verbose: options.verbose,
            unrolledURL: draftUnrolledAimodel.map { URL(fileURLWithPath: $0) })
        let promptTokens: [Int]
        if options.applyChatTemplate {
            promptTokens = try engine.tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]])
        } else {
            promptTokens = engine.tokenizer.encode(text: prompt)
        }
        if targetOnly {
            return try await engine.generateTargetOnly(
                promptTokens: promptTokens, options: options, onToken: onToken)
        }
        return try await engine.generate(promptTokens: promptTokens, options: options, onToken: onToken)
        #else
        _ = (targetAimodel, draftAimodel, tokenizerDir, prompt, options, draftTokens,
             vocabSize, backbone, slidingWindow, maxContext, onToken)
        throw RuntimeError.runtimeUnavailable
        #endif
    }

    // MARK: - Diffusion entry point

    /// Whether `modelPath` is a block-diffusion bundle (`kind == "diffusion"` or a `diffusion`
    /// metadata block). Cheap, no-throw, and available in the standalone build — the CLI uses
    /// it to route `run` to the diffusion denoise loop instead of ``LLMEngine``.
    public static func isDiffusionBundle(modelPath: String) -> Bool {
        ResolvedBundle.isDiffusionBundle(at: modelPath)
    }

    /// Load `modelPath` (an exported **stateless** diffusion `.aimodel` bundle), tokenize
    /// `prompt`, and run the host-side block-diffusion denoise loop (random-canvas init →
    /// 48-step entropy/MI-bound accept+renoise with self-conditioning → adaptive stop → block
    /// commit). Committed text deltas stream to `onToken`.
    @discardableResult
    public static func runDiffusion(
        modelPath: String,
        prompt: String,
        options: Options = Options(),
        onToken: ((String) -> Void)? = nil
    ) async throws -> DiffusionResult {
        #if COREAI_RUNTIME
        let bundle = try ResolvedBundle.load(at: modelPath)
        let engine = try await DiffusionEngine.load(bundle: bundle, verbose: options.verbose)
        let promptTokens = try engine.encodePrompt(
            messages: [["role": "user", "content": prompt]],
            applyChatTemplate: options.applyChatTemplate)
        return try await engine.generate(
            promptTokens: promptTokens, options: options, onToken: onToken)
        #else
        _ = (modelPath, prompt, options, onToken)
        throw RuntimeError.runtimeUnavailable
        #endif
    }
}
