#if COREAI_RUNTIME

import Accelerate
import CoreAI
import Foundation
import Tokenizers

/// Native Core AI prefill + decode engine for exported dynamic LLM `.aimodel` bundles.
///
/// The runtime contract is the one verified by `python/runner/run_aimodel.py` against the
/// exported bundle (and matches Apple's `CoreAISequentialEngine`):
///
/// - function `main` with inputs `input_ids` (Int32 `[1, -1]`), `position_ids`
///   (Int32 `[1, -1]`); output `logits` (`[1, -1, vocab]`); states `keyCache`,
///   `valueCache` (dynamic sequence dim) mutated in place.
/// - Causal mask is internal to the graph. We feed only `input_ids` plus monotonically
///   increasing absolute `position_ids` of the *same length as the KV history so far*.
/// - KV cache is allocated once at a fixed capacity (`min(prompt + maxTokens + 8,
///   maxContextLength)`) and reused across steps.
/// - Prefill feeds all prompt tokens at positions `0..<n`; decode feeds one token at a time
///   at position `processedTokenCount`. The next token is sampled from the *last* row of
///   `logits`.
///
/// ## Persistent handles
/// A bundle is loaded **once** via ``load(bundle:verbose:)`` into a reusable engine instance,
/// then driven through any number of ``generate(promptTokens:options:onToken:)`` calls (the
/// `ModelManager` keeps these instances hot and serialises access per model). The legacy
/// one-shot ``run(modelPath:prompt:options:onToken:)`` is preserved and now simply composes
/// `load` + `encodePrompt` + `generate`.
///
/// ## Dtype awareness
/// Logits and KV-cache scalar types are read from the function descriptor rather than assumed
/// to be Float16. Models exported with `--compute-precision bfloat16` (the Gemma/diffusion
/// families, required for parity) carry **BFloat16 KV-cache state**; the Core AI `NDArray`
/// typed-view API has no Swift scalar for BFloat16, so we never take a Float16 view of such a
/// buffer (which previously crashed with *"Type Float16 does not match scalar type
/// BFloat16"*). Float16/Float32 buffers are handled with their native Swift scalar; other
/// float types are left to their (zero-initialised) backing storage — safe because causal
/// attention is bounded by `position_ids` length and never reads unwritten KV slots.
///
/// Not `Sendable`: an instance is created and driven entirely within a single task; the
/// `ModelManager` confines each engine to one model handle and serialises generation.
final class LLMEngine {
    private let function: InferenceFunction
    let tokenizer: any Tokenizer

    private let inputIdsName: String
    private let positionIdsName: String
    private let keyCacheName: String
    private let valueCacheName: String
    private let logitsName: String

    private let inputIdsDescriptor: NDArrayDescriptor
    private let positionIdsDescriptor: NDArrayDescriptor
    private let logitsDescriptor: NDArrayDescriptor
    private let keyCacheDescriptor: NDArrayDescriptor
    private let valueCacheDescriptor: NDArrayDescriptor

    /// Actual scalar types as declared by the compiled graph (never assumed).
    private let logitsScalarType: NDArray.ScalarType
    private let keyCacheScalarType: NDArray.ScalarType
    private let valueCacheScalarType: NDArray.ScalarType

    let vocabSize: Int
    let maxContextLength: Int
    /// Per-model minimum KV-cache capacity (tokens), 0 when there is no floor. Hybrid
    /// `qwen3_5` models pack a recurrent SSM/conv state into a fixed prefix of the KV-cache
    /// sequence dimension (`ssm_pos`, e.g. 512 for Qwythos-9B); a smaller capacity makes the
    /// native `strided_slice_update` reject the prefix write, so allocation is floored to at
    /// least this many positions. Resolved from the bundle (see ``ResolvedBundle/minKVCapacity``).
    let minKVCapacity: Int
    /// Seconds spent loading the model + tokenizer (recorded once by ``load``).
    let loadSeconds: Double

    private var keyCache: NDArray
    private var valueCache: NDArray
    private(set) var processedTokenCount: Int = 0

    private let stopIds: Set<Int>

    // MARK: - Loading a persistent handle

    /// Load an exported `.aimodel` bundle once into a reusable engine handle (model graph +
    /// tokenizer + I/O contract). Heavy: links the graph and reads the tokenizer.
    static func load(bundle: ResolvedBundle, verbose: Bool = false) async throws -> LLMEngine {
        func log(_ s: @autoclosure () -> String) {
            if verbose { FileHandle.standardError.write(Data(("[coreai] " + s() + "\n").utf8)) }
        }

        let loadStart = Date()
        log("loading model \(bundle.aimodelURL.lastPathComponent) + tokenizer …")

        async let tokenizerTask = AutoTokenizer.from(modelFolder: bundle.tokenizerDir)

        // Specialize as a dynamic graph with frequent reshapes (input length changes per step).
        // Compute unit is overridable via COREAI_COMPUTE (gpu|ane|cpu|all) for perf experiments;
        // GPU is the default for large 4-bit decoders.
        var specialization = SpecializationOptions(preferredComputeUnitKind: Self.preferredComputeUnit())
        specialization.expectFrequentReshapes = true
        // Specialize THROUGH the on-disk compilation cache so the (expensive) MPS graph compile
        // is paid once and reused across runs/loads — `AIModel(contentsOf:)` recompiles every time.
        let model = try await AIModel.specialize(
            contentsOf: bundle.aimodelURL, options: specialization,
            cache: .default, cachePolicy: .persistent)
        let tokenizer = try await tokenizerTask

        let loadSeconds = Date().timeIntervalSince(loadStart)
        let engine = try LLMEngine(
            model: model,
            tokenizer: tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            minKVCapacity: bundle.minKVCapacity,
            loadSeconds: loadSeconds,
            verbose: verbose,
            tokenizerDir: bundle.tokenizerDir)
        log(String(format: "model+tokenizer ready in %.2fs", loadSeconds))
        return engine
    }

    /// Preferred compute unit, overridable via `COREAI_COMPUTE` (gpu|ane|cpu|all). Default `.gpu`.
    static func preferredComputeUnit() -> ComputeUnitKind {
        switch ProcessInfo.processInfo.environment["COREAI_COMPUTE"]?.lowercased() {
        case "cpu": return .cpu
        case "ane", "ne", "neuralengine": return .neuralEngine
        default: return .gpu  // "gpu", "all", nil
        }
    }

    // MARK: - Legacy one-shot entry point (preserved)

    static func run(
        modelPath: String,
        prompt: String,
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result {
        let bundle = try ResolvedBundle.load(at: modelPath)
        let engine = try await load(bundle: bundle, verbose: options.verbose)
        let promptTokens = try engine.encodePrompt(
            messages: [["role": "user", "content": prompt]],
            applyChatTemplate: options.applyChatTemplate)
        return try await engine.generate(
            promptTokens: promptTokens, options: options, onToken: onToken)
    }

    // MARK: - Prompt encoding

    /// Turn chat messages into token ids. With `applyChatTemplate`, the bundle tokenizer's
    /// chat template (`add_generation_prompt`) is applied — passing `tools` (when present) so the
    /// template renders its function-calling section; otherwise the concatenated message
    /// contents are tokenized verbatim (raw completion).
    func encodePrompt(
        messages: [[String: String]],
        tools: [[String: any Sendable]]? = nil,
        applyChatTemplate: Bool
    ) throws -> [Int] {
        let tokens: [Int]
        if applyChatTemplate {
            if let tools, !tools.isEmpty {
                tokens = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
            } else {
                tokens = try tokenizer.applyChatTemplate(messages: messages)
            }
        } else {
            let text = messages.map { $0["content"] ?? "" }.joined()
            tokens = tokenizer.encode(text: text)
        }
        guard !tokens.isEmpty else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("prompt tokenized to 0 tokens")
        }
        if ProcessInfo.processInfo.environment["COREAI_DEBUG_PROMPT"] != nil {
            FileHandle.standardError.write(Data(
                ("[coreai] DEBUG prompt tokens(\(tokens.count)): "
                    + "\(tokens.prefix(40))\(tokens.count > 40 ? " …" : "")\n").utf8))
        }
        return tokens
    }

    // MARK: - Generation

    /// Run prefill + decode over `promptTokens`, streaming decoded text deltas to `onToken`.
    /// Allocates a fresh KV cache sized to this request, so sequential calls on the same hot
    /// engine are independent (no cross-request state leakage).
    func generate(
        promptTokens: [Int],
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result {
        func log(_ s: @autoclosure () -> String) {
            if options.verbose { FileHandle.standardError.write(Data(("[coreai] " + s() + "\n").utf8)) }
        }

        // Fixed KV capacity sized to this request, floored to the per-model minimum.
        let capacity = resolvedCapacity(promptCount: promptTokens.count, options: options)
        try allocateKVCache(capacity: capacity)
        log(
            "prompt -> \(promptTokens.count) tokens, KV cache capacity \(capacity)"
                + (minKVCapacity > 0 ? " (floor \(minKVCapacity))" : ""))

        let sampler = Sampler(
            temperature: options.temperature, topK: options.topK, topP: options.topP)
        var rng = SeededGenerator(seed: options.seed ?? UInt64.random(in: .min ... .max))

        // Prefill.
        let prefillStart = Date()
        let prompt32 = promptTokens.map { Int32($0) }
        var lastLogits = try await step(tokens: prompt32)
        var nextToken = sampler.sample(lastLogits, using: &rng)
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        log(
            String(
                format: "prefill %.3fs (%.1f tok/s)", prefillSeconds,
                prefillSeconds > 0 ? Double(promptTokens.count) / prefillSeconds : 0))

        // Decode.
        let decodeStart = Date()
        var generated: [Int] = []
        var streamedText = ""
        var finalTextOverride: String?
        var stopReason: CoreAIPipeline.StopReason = .maxTokens
        let stopSequences = options.stopSequences.filter { !$0.isEmpty }

        func emitVisibleText(_ text: String) {
            guard let onToken else {
                streamedText = text
                return
            }
            if text.hasPrefix(streamedText) {
                let delta = String(text.dropFirst(streamedText.count))
                if !delta.isEmpty { onToken(delta) }
            }
            streamedText = text
        }

        // Coarse per-phase profiling (gated on COREAI_PROFILE) to find the decode bottleneck.
        let profile = ProcessInfo.processInfo.environment["COREAI_PROFILE"] != nil
        var fwdT = 0.0, sampT = 0.0, detokT = 0.0

        for _ in 0..<max(0, options.maxTokens) {
            if stopIds.contains(nextToken) {
                stopReason = .eos
                break
            }
            if processedTokenCount >= maxContextLength {
                stopReason = .contextLimit
                break
            }

            generated.append(nextToken)

            // Best-effort streaming detokenization: decode the full continuation and emit the
            // newly added suffix (handles BPE merges/spacing correctly).
            if onToken != nil || !stopSequences.isEmpty {
                let t0 = profile ? Date() : decodeStart
                let text = tokenizer.decode(tokens: generated)
                if let stopRange = CoreAIPipeline.firstStopRange(
                    in: text, stopSequences: stopSequences)
                {
                    let visible = String(text[..<stopRange.lowerBound])
                    emitVisibleText(visible)
                    finalTextOverride = visible
                    stopReason = .stopSequence
                    if profile { detokT += Date().timeIntervalSince(t0) }
                    break
                } else {
                    let visible = stopSequences.isEmpty
                        ? text
                        : CoreAIPipeline.visibleTextAvoidingPartialStop(text, stopSequences: stopSequences)
                    emitVisibleText(visible)
                }
                if profile { detokT += Date().timeIntervalSince(t0) }
            }

            let t1 = profile ? Date() : decodeStart
            lastLogits = try await step(tokens: [Int32(nextToken)])
            let t2 = profile ? Date() : decodeStart
            nextToken = sampler.sample(lastLogits, using: &rng)
            if profile {
                fwdT += t2.timeIntervalSince(t1)
                sampT += Date().timeIntervalSince(t2)
            }
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        if profile {
            FileHandle.standardError.write(Data(
                (String(
                    format:
                        "[coreai] PROFILE decode: forward+readback=%.3fs (run=%.3fs readback=%.3fs) sample=%.3fs detok=%.3fs (n=%d)\n",
                    fwdT, Self.profRunT, Self.profReadT, sampT, detokT, generated.count)).utf8))
        }
        let finalText = finalTextOverride ?? tokenizer.decode(tokens: generated)
        if finalTextOverride == nil, onToken != nil {
            emitVisibleText(finalText)
        }
        log(
            String(
                format: "decode %d tokens in %.3fs (%.1f tok/s), stop=%@",
                generated.count, decodeSeconds,
                decodeSeconds > 0 ? Double(generated.count) / decodeSeconds : 0,
                stopReason.rawValue))

        return CoreAIPipeline.Result(
            text: finalText,
            promptTokenCount: promptTokens.count,
            generatedTokenCount: generated.count,
            stopReason: stopReason,
            modelLoadSeconds: loadSeconds,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds)
    }

    // MARK: - Init / model contract validation

    private init(
        model: AIModel,
        tokenizer: any Tokenizer,
        vocabSize: Int,
        maxContextLength: Int,
        minKVCapacity: Int,
        loadSeconds: Double,
        verbose: Bool,
        tokenizerDir: URL? = nil
    ) throws {
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.minKVCapacity = minKVCapacity
        self.loadSeconds = loadSeconds

        guard let descriptor = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "function 'main' not found; have \(model.functionNames)")
        }

        guard descriptor.inputNames.count == 2 else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "expected 2 inputs, got \(descriptor.inputNames)")
        }
        guard descriptor.stateNames.count == 2 else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "expected 2 states (KV cache), got \(descriptor.stateNames)")
        }
        guard descriptor.outputNames.count >= 1 else {
            throw CoreAIPipeline.RuntimeError.modelContract("expected >= 1 output")
        }

        // Resolve I/O by the contract's canonical names, falling back to declaration order.
        self.inputIdsName = Self.pick("input_ids", descriptor.inputNames, index: 0)
        self.positionIdsName = Self.pick("position_ids", descriptor.inputNames, index: 1)
        self.keyCacheName = Self.pick("keyCache", descriptor.stateNames, index: 0)
        self.valueCacheName = Self.pick("valueCache", descriptor.stateNames, index: 1)
        self.logitsName = Self.pick("logits", descriptor.outputNames, index: 0)

        guard case .ndArray(let inDesc) = descriptor.inputDescriptor(of: inputIdsName) else {
            throw CoreAIPipeline.RuntimeError.modelContract("input '\(inputIdsName)' is not an NDArray")
        }
        guard case .ndArray(let posDesc) = descriptor.inputDescriptor(of: positionIdsName) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "input '\(positionIdsName)' is not an NDArray")
        }
        guard case .ndArray(let logitsDesc) = descriptor.outputDescriptor(of: logitsName) else {
            throw CoreAIPipeline.RuntimeError.modelContract("output '\(logitsName)' is not an NDArray")
        }
        guard case .ndArray(let keyDesc) = descriptor.stateDescriptor(of: keyCacheName),
            case .ndArray(let valueDesc) = descriptor.stateDescriptor(of: valueCacheName)
        else {
            throw CoreAIPipeline.RuntimeError.modelContract("KV cache states are not NDArrays")
        }

        self.inputIdsDescriptor = inDesc
        self.positionIdsDescriptor = posDesc
        self.logitsDescriptor = logitsDesc
        self.keyCacheDescriptor = keyDesc
        self.valueCacheDescriptor = valueDesc

        // Dtype-aware: record the graph's declared scalar types instead of assuming Float16.
        self.logitsScalarType = logitsDesc.scalarType
        self.keyCacheScalarType = keyDesc.scalarType
        self.valueCacheScalarType = valueDesc.scalarType

        // Logits must be a float type we can read through the typed-view API. Float16 is the
        // norm (Apple's exporter emits Float16 logits even for bfloat16 compute precision);
        // Float32 is also supported. We reject other logit dtypes loudly rather than crash.
        switch logitsDesc.scalarType {
        case .float16, .float32:
            break
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "unsupported logits scalar type \(logitsDesc.scalarType) (expected float16/float32)")
        }

        if verbose {
            FileHandle.standardError.write(
                Data(
                    "[coreai] dtypes: logits=\(logitsDesc.scalarType) keyCache=\(keyDesc.scalarType) valueCache=\(valueDesc.scalarType)\n"
                        .utf8))
        }

        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("could not load function 'main'")
        }
        self.function = fn
        self.stopIds = Self.stopTokenIds(tokenizer: tokenizer, tokenizerDir: tokenizerDir)

        // Placeholder 1-slot caches; replaced by allocateKVCache(capacity:) before use.
        self.keyCache = NDArray(descriptor: keyDesc.resolvingDynamicDimensions(
            keyDesc.shape.map { $0 < 0 ? 1 : $0 }))
        self.valueCache = NDArray(descriptor: valueDesc.resolvingDynamicDimensions(
            valueDesc.shape.map { $0 < 0 ? 1 : $0 }))
    }

    private static func pick(_ wanted: String, _ names: [String], index: Int) -> String {
        names.contains(wanted) ? wanted : names[index]
    }

    // MARK: - KV cache

    /// The fixed KV-cache capacity (tokens) to allocate for a request: the larger of the
    /// requested/derived size and the per-model floor `minKVCapacity + maxTokens`, clamped to
    /// `maxContextLength`. For standard models `minKVCapacity == 0`, so the floor degenerates to
    /// the requested size; for hybrid `qwen3_5` models it guarantees the SSM-state prefix fits
    /// (capacity >= `minKVCapacity`) without any manual `--kv-capacity`.
    func resolvedCapacity(promptCount: Int, options: CoreAIPipeline.Options) -> Int {
        let maxTokens = max(0, options.maxTokens)
        let requested = options.kvCapacity ?? (promptCount + maxTokens + 8)
        let floored = max(requested, minKVCapacity + maxTokens)
        return min(floored, maxContextLength)
    }

    /// Allocate the keyCache/valueCache NDArrays, resolving the dynamic sequence dim to
    /// `capacity`, and zero them when the dtype is representable as a Swift scalar (Float16 /
    /// Float32). For other float dtypes (e.g. BFloat16) we rely on the freshly-allocated
    /// backing storage being zero-initialised — and on causal attention being bounded by
    /// `position_ids`, which means unwritten KV slots are never read regardless.
    func allocateKVCache(capacity: Int) throws {
        let keyShape = keyCacheDescriptor.shape.map { $0 < 0 ? capacity : $0 }
        let valueShape = valueCacheDescriptor.shape.map { $0 < 0 ? capacity : $0 }
        keyCache = NDArray(descriptor: keyCacheDescriptor.resolvingDynamicDimensions(keyShape))
        valueCache = NDArray(descriptor: valueCacheDescriptor.resolvingDynamicDimensions(valueShape))
        Self.zeroState(&keyCache, scalarType: keyCacheScalarType)
        Self.zeroState(&valueCache, scalarType: valueCacheScalarType)
        processedTokenCount = 0
    }

    // MARK: - One forward pass

    /// Run one forward pass over the `n` new `tokens`, mutating the KV cache in place, and
    /// return the *last* token's logits as Float32 (length `vocabSize`).
    ///
    /// - `input_ids` is the `n` new tokens (shape `[1, n]`);
    /// - `position_ids` carries *all* absolute positions so far,
    ///   `[0 ..< processedTokenCount + n]`. The graph uses this length to size causal
    ///   attention over the KV cache; feeding only the new positions makes decode lose context.
    /// - `logits` has one row per *input* token (shape `[1, n, vocab]`); we read the last row.
    nonisolated(unsafe) static var profRunT: Double = 0
    nonisolated(unsafe) static var profReadT: Double = 0

    func step(tokens: [Int32]) async throws -> [Float] {
        let prof = ProcessInfo.processInfo.environment["COREAI_PROFILE"] != nil
        let t0 = prof ? Date() : Date.distantPast
        let logits = try await runForward(tokens: tokens)
        let t1 = prof ? Date() : Date.distantPast
        let row = lastRowFloat32(logits)
        if prof {
            Self.profRunT += t1.timeIntervalSince(t0)
            Self.profReadT += Date().timeIntervalSince(t1)
        }
        return row
    }

    /// Like ``step(tokens:)`` but returns *every* row of `logits` (one Float32 distribution per
    /// input token, in order). Used by speculative decoding's target verification, where one
    /// batched forward over `[anchor, draft₁ … draftₖ]` yields the target's greedy prediction at
    /// each draft position in a single pass.
    func forwardAllRows(tokens: [Int32]) async throws -> [[Float]] {
        let logits = try await runForward(tokens: tokens)
        return allRowsFloat32(logits)
    }

    /// Roll the committed KV history back to `count` tokens (discarding the positions written by
    /// later forwards). Cheap: only `processedTokenCount` moves — the stale KV slots beyond
    /// `count` are overwritten by the next forward and are never read in the meantime, because
    /// causal attention is bounded by `position_ids` length (`processedTokenCount + n`). This is
    /// what lets speculative decoding feed a draft batch, then discard the rejected suffix.
    ///
    /// Valid for the positional KV of standard attention layers (the only layers exercised by
    /// the draft/verify path). It is *not* a correct rewind of a recurrent SSM state, so the
    /// speculative path is restricted to standard-attention model pairs.
    func rollbackKV(to count: Int) {
        precondition(count >= 0 && count <= processedTokenCount, "rollback target out of range")
        processedTokenCount = count
    }

    /// Whether `id` is a turn/sequence-ending token for this model (EOS or a chat special).
    func isStopToken(_ id: Int) -> Bool { stopIds.contains(id) }

    /// One forward pass over the `n` new `tokens`, mutating the KV cache in place and returning
    /// the raw `logits` NDArray (`[1, n, vocab]`). Shared by ``step`` / ``forwardAllRows``.
    private func runForward(tokens: [Int32]) async throws -> NDArray {
        let n = tokens.count
        precondition(n > 0, "forward requires >= 1 token")

        var inputIds = NDArray(descriptor: inputIdsDescriptor.resolvingDynamicDimensions([1, n]))
        Self.fillInt32(&inputIds, tokens)

        // position_ids contract differs by export: the authored gemma4/qwen3_5 graphs want the
        // FULL [0 … processedTokenCount+n-1]; a standard `coreai.llm.export` graph wants the
        // length-matched current window [processedTokenCount … +n-1] (one position per input
        // token), else the new token gets RoPE position 0 → garbage. COREAI_POS_MODE=current
        // selects the standard contract.
        let totalPositions = processedTokenCount + n
        let positionValues: [Int32]
        let positionLen: Int
        if ProcessInfo.processInfo.environment["COREAI_POS_MODE"]?.lowercased() == "current" {
            positionValues = (processedTokenCount..<totalPositions).map { Int32($0) }
            positionLen = n
        } else {
            positionValues = (0..<totalPositions).map { Int32($0) }
            positionLen = totalPositions
        }
        var positionIds = NDArray(
            descriptor: positionIdsDescriptor.resolvingDynamicDimensions([1, positionLen]))
        Self.fillInt32(&positionIds, positionValues)

        var logits = NDArray(
            descriptor: logitsDescriptor.resolvingDynamicDimensions([1, n, vocabSize]))

        var states = InferenceFunction.MutableViews()
        states.insert(&keyCache, for: keyCacheName)
        states.insert(&valueCache, for: valueCacheName)

        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&logits, for: logitsName)

        _ = try await function.run(
            inputs: [inputIdsName: inputIds, positionIdsName: positionIds],
            states: consume states,
            outputViews: consume outputViews)

        processedTokenCount += n
        return logits
    }

    // MARK: - NDArray helpers (dtype-aware; row-major, stride-aware reads)

    private static func fillInt32(_ array: inout NDArray, _ elements: [Int32]) {
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: elements)
    }

    /// Zero an NDArray when its element type has a Swift scalar we can view as. No-op for
    /// dtypes (e.g. BFloat16) that the typed-view API can't represent — see ``allocateKVCache``.
    private static func zeroState(_ array: inout NDArray, scalarType: NDArray.ScalarType) {
        let count = array.shape.reduce(1, *)
        switch scalarType {
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr, _, _ in for i in 0..<count { ptr[i] = 0 } }
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { ptr, _, _ in for i in 0..<count { ptr[i] = 0 } }
        default:
            break
        }
    }

    /// Read the last row (`[..., rows-1, :]`) of a `[1, rows, vocab]` logits array as Float32,
    /// honoring strides, dispatching on the logits scalar type (Float16 or Float32).
    private func lastRowFloat32(_ array: NDArray) -> [Float] {
        switch logitsScalarType {
        case .float32:
            return Self.lastRow(array, as: Float.self, vocab: vocabSize)
        default:
            return Self.lastRow(array, as: Float16.self, vocab: vocabSize)
        }
    }

    private static func lastRow<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: NDArray, as _: T.Type, vocab: Int
    ) -> [Float] {
        array.view(as: T.self).withUnsafePointer { ptr, shape, strides in
            let rank = shape.count
            let rows = shape[rank - 2]
            let rowStride = strides[rank - 2]
            let colStride = strides[rank - 1]
            let base = (rows - 1) * rowStride
            var out = [Float](repeating: 0, count: vocab)
            for v in 0..<vocab { out[v] = Float(ptr[base + v * colStride]) }
            return out
        }
    }

    /// Read *all* rows of a `[1, rows, vocab]` logits array as Float32 (one distribution per
    /// input token), honoring strides and dispatching on the logits scalar type.
    private func allRowsFloat32(_ array: NDArray) -> [[Float]] {
        switch logitsScalarType {
        case .float32:
            return Self.allRows(array, as: Float.self, vocab: vocabSize)
        default:
            return Self.allRows(array, as: Float16.self, vocab: vocabSize)
        }
    }

    private static func allRows<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: NDArray, as _: T.Type, vocab: Int
    ) -> [[Float]] {
        array.view(as: T.self).withUnsafePointer { ptr, shape, strides in
            let rank = shape.count
            let rows = shape[rank - 2]
            let rowStride = strides[rank - 2]
            let colStride = strides[rank - 1]
            var out = [[Float]](repeating: [Float](repeating: 0, count: vocab), count: rows)
            for r in 0..<rows {
                let base = r * rowStride
                for v in 0..<vocab { out[r][v] = Float(ptr[base + v * colStride]) }
            }
            return out
        }
    }

    // MARK: - Stop tokens

    /// Chat / turn-ending stop tokens for `tokenizer`. Shared by the standard and EAGLE engines.
    ///
    /// The authoritative source is the model's OWN published `generation_config.json` — its
    /// `eos_token_id` is often a LIST (gemma-4 publishes `[1, 106, 50]`: <eos>, <turn|>, and a
    /// third stop). We read that list from `tokenizerDir` (or its parent) when available, so each
    /// model's real stop set is honored rather than guessed. A small string fallback covers models
    /// whose generation_config we don't ship.
    static func stopTokenIds(tokenizer: any Tokenizer, tokenizerDir: URL? = nil) -> Set<Int> {
        var ids = Set<Int>()
        if let eos = tokenizer.eosTokenId { ids.insert(eos) }
        // 1) published eos_token_id (int or [int]) from generation_config.json
        if let dir = tokenizerDir {
            for cand in [dir.appendingPathComponent("generation_config.json"),
                         dir.deletingLastPathComponent().appendingPathComponent("generation_config.json")] {
                guard let data = try? Data(contentsOf: cand),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let eos = obj["eos_token_id"] else { continue }
                if let one = eos as? Int { ids.insert(one) }
                else if let many = eos as? [Int] { ids.formUnion(many) }
                break
            }
        }
        // 2) string fallback for models we don't ship a generation_config for.
        for token in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "</s>",
                      "<turn|>", "<|turn>"] {
            if let id = tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        return ids
    }
}

/// Small deterministic RNG (SplitMix64) for reproducible temperature sampling.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

#endif
