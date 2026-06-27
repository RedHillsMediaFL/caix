#if COREAI_RUNTIME

import CoreAI
import Foundation
import Tokenizers

// EAGLE / MTP speculative decoding for Gemma-4 on Core AI.
//
// Unlike `SpeculativeEngine` (two independent standard decoders), the EAGLE draft is a *dependent*
// module of the target: each draft micro-step consumes the target's last hidden state + the
// target's representative K/V (`shared_kv`, one tensor per layer-type) and a constant position,
// and recurs on (token, predicted-hidden). So the two bundles use CUSTOM contracts:
//
//   TARGET (Gemma4EagleTarget):  inputs (input_ids, position_ids) + KV-cache state ->
//     outputs (logits, hidden, k_full, v_full, k_sliding, v_sliding)  [all f16]
//   DRAFT  (Gemma4AssistantForCausalLM): inputs (token_id, hidden, position_ids, k_full, v_full,
//     k_sliding, v_sliding) -> outputs (logits, next_hidden)  [stateless]
//
// The verify/accept + cache bookkeeping mirror `SpeculativeEngine` exactly (greedy: every committed
// token is the target's argmax, so output is byte-identical to target-only). The novelty is the
// draft proposal: seeded from the target's hidden + cropped shared-KV, recurring at a constant
// position.

// MARK: - small NDArray helpers (f16; row-major, stride-aware)

enum EagleND {
    /// Copy element `[0, row, :]` of an `[1, R, D]` f16 array into a fresh `[1, 1, D]` f16 array.
    static func hiddenRow(_ src: NDArray, row: Int, dim: Int, descriptor: NDArrayDescriptor) -> NDArray {
        var out = NDArray(descriptor: descriptor.resolvingDynamicDimensions([1, 1, dim]))
        src.view(as: Float16.self).withUnsafePointer { ptr, _, strides in
            let rowStride = strides[strides.count - 2]
            let colStride = strides[strides.count - 1]
            let base = row * rowStride
            var ov = out.mutableView(as: Float16.self)
            ov.withUnsafeMutablePointer { op, _, ostr in
                let oc = ostr[ostr.count - 1]
                for d in 0..<dim { op[d * oc] = ptr[base + d * colStride] }
            }
        }
        return out
    }

    /// Slice an `[1, H, S, Dh]` f16 KV tensor along the sequence dim to positions `[start, start+len)`
    /// into a fresh `[1, H, len, Dh]` f16 array (used to drop rejected-draft KV and to crop the
    /// sliding window). Absolute-position RoPE is preserved (we only select columns).
    static func kvSlice(_ src: NDArray, start: Int, len: Int, descriptor: NDArrayDescriptor) -> NDArray {
        let shp = src.shape  // [1, H, S, Dh]
        let h = shp[1], dh = shp[3]
        var out = NDArray(descriptor: descriptor.resolvingDynamicDimensions([1, h, len, dh]))
        src.view(as: Float16.self).withUnsafePointer { ptr, _, st in
            let hS = st[1], sS = st[2], dS = st[3]
            var ov = out.mutableView(as: Float16.self)
            ov.withUnsafeMutablePointer { op, _, ost in
                let ohS = ost[1], osS = ost[2], odS = ost[3]
                for hi in 0..<h {
                    for si in 0..<len {
                        let ib = hi * hS + (start + si) * sS
                        let ob = hi * ohS + si * osS
                        for di in 0..<dh { op[ob + di * odS] = ptr[ib + (di * dS)] }
                    }
                }
            }
        }
        return out
    }

    static func fillI32(_ a: inout NDArray, _ v: [Int32]) {
        var view = a.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: v)
    }

    /// Append the `n` new-position columns of `src` (`[1, H, n, Dh]`) into the accumulator
    /// `dst` (`[1, H, cap, Dh]`) at sequence offset `off`. The EAGLE target now emits only the
    /// new positions' repr K/V each forward (reading the persistent KV-cache state back as a graph
    /// output breaks the in-place state mutation under torch.export); Swift grows the full prefix.
    static func appendKV(_ dst: inout NDArray, _ src: NDArray, atOffset off: Int) {
        let s = src.shape  // [1, H, n, Dh]
        let h = s[1], n = s[2], dh = s[3]
        src.view(as: Float16.self).withUnsafePointer { ip, _, ist in
            let ihS = ist[1], isS = ist[2], idS = ist[3]
            var dv = dst.mutableView(as: Float16.self)
            dv.withUnsafeMutablePointer { op, _, ost in
                let ohS = ost[1], osS = ost[2], odS = ost[3]
                for hi in 0..<h {
                    for si in 0..<n {
                        let ib = hi * ihS + si * isS
                        let ob = hi * ohS + (off + si) * osS
                        for di in 0..<dh { op[ob + di * odS] = ip[ib + di * idS] }
                    }
                }
            }
        }
    }
}

// MARK: - EAGLE target (6-output + KV state)

final class EagleTargetEngine {
    private let function: InferenceFunction
    private let inDesc: [String: NDArrayDescriptor]
    private let outDesc: [String: NDArrayDescriptor]
    private var keyCache: NDArray
    private var valueCache: NDArray
    private let keyName: String
    private let valueName: String
    let vocabSize: Int
    let hiddenSize: Int
    private(set) var processed: Int = 0
    // Exclusive end index physically written by the MOST RECENT forward (preserved across
    // rollback). The reference candidate generator crops shared_kv to the full committed length
    // (including the anchor position, populated by the rejected-draft KV as an anchor proxy); we
    // can read up to here even after a rollback truncates `processed`.
    private(set) var lastWrittenEnd: Int = 0
    // Host-side accumulators for the repr-layer K/V prefix (the target now emits only the new
    // positions per forward; we grow the full prefix here). Sized to capacity in `allocateCache`.
    private var accKFull: NDArray
    private var accVFull: NDArray
    private var accKSliding: NDArray
    private var accVSliding: NDArray
    private let hf: Int, df: Int, hs: Int, ds: Int  // full/sliding head counts + head dims
    private var kvCapacity: Int = 1

    struct Out {
        let logitsRows: [[Float]]
        let hidden: NDArray          // [1, N, D]
        let kFull: NDArray, vFull: NDArray, kSliding: NDArray, vSliding: NDArray
        let kvLen: Int               // sequence length of the KV tensors (= processed after fwd)
    }

    private init(model: AIModel, vocabSize: Int, hiddenSize: Int) throws {
        guard let d = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle target: no 'main'")
        }
        func nd(_ which: String, _ name: String) -> NDArrayDescriptor? {
            let kind: NDArrayDescriptor?
            switch which {
            case "in": if case .ndArray(let x) = d.inputDescriptor(of: name) { kind = x } else { kind = nil }
            case "out": if case .ndArray(let x) = d.outputDescriptor(of: name) { kind = x } else { kind = nil }
            default: if case .ndArray(let x) = d.stateDescriptor(of: name) { kind = x } else { kind = nil }
            }
            return kind
        }
        var ins: [String: NDArrayDescriptor] = [:]
        for n in d.inputNames { ins[n] = nd("in", n) }
        var outs: [String: NDArrayDescriptor] = [:]
        for n in d.outputNames { outs[n] = nd("out", n) }
        self.inDesc = ins
        self.outDesc = outs
        // repr-layer geometry from the KV output descriptors: [1, H, seq(-1), Dh].
        let kfShp = outs["k_full"]?.shape ?? [1, 2, -1, 512]
        let ksShp = outs["k_sliding"]?.shape ?? [1, 8, -1, 256]
        self.hf = kfShp[1]; self.df = kfShp[3]
        self.hs = ksShp[1]; self.ds = ksShp[3]
        // Placeholder accumulators (re-sized in allocateCache).
        self.accKFull = NDArray(descriptor: outs["k_full"]!.resolvingDynamicDimensions([1, kfShp[1], 1, kfShp[3]]))
        self.accVFull = NDArray(descriptor: outs["v_full"]!.resolvingDynamicDimensions([1, kfShp[1], 1, kfShp[3]]))
        self.accKSliding = NDArray(descriptor: outs["k_sliding"]!.resolvingDynamicDimensions([1, ksShp[1], 1, ksShp[3]]))
        self.accVSliding = NDArray(descriptor: outs["v_sliding"]!.resolvingDynamicDimensions([1, ksShp[1], 1, ksShp[3]]))
        self.keyName = d.stateNames.first(where: { $0.lowercased().contains("key") }) ?? d.stateNames[0]
        self.valueName = d.stateNames.first(where: { $0.lowercased().contains("val") }) ?? d.stateNames[1]
        guard let kd = nd("state", keyName), let vd = nd("state", valueName) else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle target: KV state not NDArray")
        }
        self.keyCache = NDArray(descriptor: kd.resolvingDynamicDimensions(kd.shape.map { $0 < 0 ? 1 : $0 }))
        self.valueCache = NDArray(descriptor: vd.resolvingDynamicDimensions(vd.shape.map { $0 < 0 ? 1 : $0 }))
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle target: load 'main' failed")
        }
        self.function = fn
        self.kd = kd
        self.vd = vd
    }
    private let kd: NDArrayDescriptor
    private let vd: NDArrayDescriptor

    static func load(aimodelURL: URL, vocabSize: Int, hiddenSize: Int) async throws -> EagleTargetEngine {
        var spec = SpecializationOptions(preferredComputeUnitKind: .gpu)
        spec.expectFrequentReshapes = true
        let model = try await AIModel.specialize(
            contentsOf: aimodelURL, options: spec, cache: .default, cachePolicy: .persistent)
        return try EagleTargetEngine(model: model, vocabSize: vocabSize, hiddenSize: hiddenSize)
    }

    func allocateCache(capacity: Int) {
        keyCache = NDArray(descriptor: kd.resolvingDynamicDimensions(kd.shape.map { $0 < 0 ? capacity : $0 }))
        valueCache = NDArray(descriptor: vd.resolvingDynamicDimensions(vd.shape.map { $0 < 0 ? capacity : $0 }))
        kvCapacity = max(1, capacity)
        accKFull = NDArray(descriptor: outDesc["k_full"]!.resolvingDynamicDimensions([1, hf, kvCapacity, df]))
        accVFull = NDArray(descriptor: outDesc["v_full"]!.resolvingDynamicDimensions([1, hf, kvCapacity, df]))
        accKSliding = NDArray(descriptor: outDesc["k_sliding"]!.resolvingDynamicDimensions([1, hs, kvCapacity, ds]))
        accVSliding = NDArray(descriptor: outDesc["v_sliding"]!.resolvingDynamicDimensions([1, hs, kvCapacity, ds]))
        processed = 0
        lastWrittenEnd = 0
    }

    func rollback(to count: Int) { processed = count }

    func forward(_ tokens: [Int32]) async throws -> Out {
        let n = tokens.count
        let offset = processed          // prefix length before this forward
        let seqLen = processed + n
        var inputIds = NDArray(descriptor: inDesc["input_ids"]!.resolvingDynamicDimensions([1, n]))
        EagleND.fillI32(&inputIds, tokens)
        var positionIds = NDArray(
            descriptor: inDesc["position_ids"]!.resolvingDynamicDimensions([1, seqLen]))
        EagleND.fillI32(&positionIds, (0..<Int32(seqLen)).map { $0 })

        // Pre-allocate outputs. logits/hidden seq=n; KV is now the NEW positions only (seq=n).
        func out(_ name: String, _ shape: [Int]) -> NDArray {
            NDArray(descriptor: outDesc[name]!.resolvingDynamicDimensions(shape))
        }
        var logits = out("logits", [1, n, vocabSize])
        var hidden = out("hidden", [1, n, hiddenSize])
        var kFullNew = out("k_full", [1, hf, n, df])
        var vFullNew = out("v_full", [1, hf, n, df])
        var kSlidingNew = out("k_sliding", [1, hs, n, ds])
        var vSlidingNew = out("v_sliding", [1, hs, n, ds])

        var states = InferenceFunction.MutableViews()
        states.insert(&keyCache, for: keyName)
        states.insert(&valueCache, for: valueName)
        var ov = InferenceFunction.MutableViews()
        ov.insert(&logits, for: "logits")
        ov.insert(&hidden, for: "hidden")
        ov.insert(&kFullNew, for: "k_full")
        ov.insert(&vFullNew, for: "v_full")
        ov.insert(&kSlidingNew, for: "k_sliding")
        ov.insert(&vSlidingNew, for: "v_sliding")
        _ = try await function.run(
            inputs: ["input_ids": inputIds, "position_ids": positionIds],
            states: consume states, outputViews: consume ov)

        // Grow the host prefix with this forward's new positions [offset, offset+n).
        EagleND.appendKV(&accKFull, kFullNew, atOffset: offset)
        EagleND.appendKV(&accVFull, vFullNew, atOffset: offset)
        EagleND.appendKV(&accKSliding, kSlidingNew, atOffset: offset)
        EagleND.appendKV(&accVSliding, vSlidingNew, atOffset: offset)
        processed += n
        lastWrittenEnd = processed   // physical extent of THIS forward (kept across rollback)

        // Return the accumulators (capacity-sized; valid up to `processed`). The loop's `sliceKV`
        // reads only [0, length) with length <= processed, so the unused tail is never seen.
        return Out(
            logitsRows: Self.allRows(logits, vocab: vocabSize),
            hidden: hidden, kFull: accKFull, vFull: accVFull, kSliding: accKSliding, vSliding: accVSliding,
            kvLen: processed)
    }

    func hiddenDescriptor() -> NDArrayDescriptor { outDesc["hidden"]! }
    func kFullDescriptor() -> NDArrayDescriptor { outDesc["k_full"]! }
    func kSlidingDescriptor() -> NDArrayDescriptor { outDesc["k_sliding"]! }

    private static func allRows(_ a: NDArray, vocab: Int) -> [[Float]] {
        a.view(as: Float16.self).withUnsafePointer { ptr, shape, st in
            let rows = shape[shape.count - 2]
            let rs = st[st.count - 2], cs = st[st.count - 1]
            var o = [[Float]](repeating: [Float](repeating: 0, count: vocab), count: rows)
            for r in 0..<rows { let b = r * rs; for v in 0..<vocab { o[r][v] = Float(ptr[b + v * cs]) } }
            return o
        }
    }
}

// MARK: - EAGLE draft (7-in / 2-out, stateless)

final class EagleDraftEngine {
    private let function: InferenceFunction
    private let inDesc: [String: NDArrayDescriptor]
    private let outDesc: [String: NDArrayDescriptor]
    let vocabSize: Int
    let hiddenSize: Int

    private init(model: AIModel, vocabSize: Int, hiddenSize: Int) throws {
        guard let d = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle draft: no 'main'")
        }
        var ins: [String: NDArrayDescriptor] = [:]
        for n in d.inputNames { if case .ndArray(let x) = d.inputDescriptor(of: n) { ins[n] = x } }
        var outs: [String: NDArrayDescriptor] = [:]
        for n in d.outputNames { if case .ndArray(let x) = d.outputDescriptor(of: n) { outs[n] = x } }
        self.inDesc = ins
        self.outDesc = outs
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle draft: load 'main' failed")
        }
        self.function = fn
    }

    static func load(aimodelURL: URL, vocabSize: Int, hiddenSize: Int) async throws -> EagleDraftEngine {
        var spec = SpecializationOptions(preferredComputeUnitKind: .gpu)
        spec.expectFrequentReshapes = true
        let model = try await AIModel.specialize(
            contentsOf: aimodelURL, options: spec, cache: .default, cachePolicy: .persistent)
        return try EagleDraftEngine(model: model, vocabSize: vocabSize, hiddenSize: hiddenSize)
    }

    struct Out { let logits: [Float]; let nextHidden: NDArray }

    /// One draft micro-step. `hidden` is `[1,1,backbone]`; the KV tensors are the (cropped) shared
    /// target KV; `position` is the constant draft position. Returns the draft's token logits and
    /// the predicted next backbone hidden (for the next micro-step).
    func step(token: Int32, hidden: NDArray, position: Int32,
              kFull: NDArray, vFull: NDArray, kSliding: NDArray, vSliding: NDArray) async throws -> Out {
        var tokenId = NDArray(descriptor: inDesc["token_id"]!.resolvingDynamicDimensions([1, 1]))
        EagleND.fillI32(&tokenId, [token])
        var pos = NDArray(descriptor: inDesc["position_ids"]!.resolvingDynamicDimensions([1, 1]))
        EagleND.fillI32(&pos, [position])
        var h = hidden, kf = kFull, vf = vFull, ks = kSliding, vs = vSliding
        var logits = NDArray(descriptor: outDesc["logits"]!.resolvingDynamicDimensions([1, 1, vocabSize]))
        var nextHidden = NDArray(
            descriptor: outDesc["next_hidden"]!.resolvingDynamicDimensions([1, 1, hiddenSize]))
        var ov = InferenceFunction.MutableViews()
        ov.insert(&logits, for: "logits")
        ov.insert(&nextHidden, for: "next_hidden")
        var noStates = InferenceFunction.MutableViews()
        _ = try await function.run(
            inputs: ["token_id": tokenId, "hidden": h, "position_ids": pos,
                     "k_full": kf, "v_full": vf, "k_sliding": ks, "v_sliding": vs],
            states: consume noStates, outputViews: consume ov)
        let row = logits.view(as: Float16.self).withUnsafePointer { ptr, _, st in
            let cs = st[st.count - 1]
            return (0..<vocabSize).map { Float(ptr[$0 * cs]) }
        }
        return Out(logits: row, nextHidden: nextHidden)
    }
}

// MARK: - EAGLE unrolled draft (7-in / 1-out, stateless; K micro-steps fused in-graph)

/// Runs all K draft micro-steps (in-graph argmax + hidden recurrence) in ONE Core AI dispatch.
/// Same inputs as `EagleDraftEngine.step`; returns the K proposed draft tokens. Removes the
/// per-micro-step GPU launch tax — the dominant draft cost once acceptance is high. `K` is baked
/// into the exported graph (read back from the `draft_tokens` output shape).
final class EagleDraftUnrolledEngine {
    private let function: InferenceFunction
    private let inDesc: [String: NDArrayDescriptor]
    private let outDesc: NDArrayDescriptor
    let numSteps: Int
    let hiddenSize: Int

    private init(model: AIModel, hiddenSize: Int) throws {
        guard let d = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle unrolled draft: no 'main'")
        }
        var ins: [String: NDArrayDescriptor] = [:]
        for n in d.inputNames { if case .ndArray(let x) = d.inputDescriptor(of: n) { ins[n] = x } }
        guard case .ndArray(let od) = d.outputDescriptor(of: d.outputNames[0]) else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle unrolled draft: output not NDArray")
        }
        self.inDesc = ins
        self.outDesc = od
        self.numSteps = od.shape.count >= 2 && od.shape[1] > 0 ? od.shape[1] : 4
        self.hiddenSize = hiddenSize
        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle unrolled draft: load 'main' failed")
        }
        self.function = fn
    }

    static func load(aimodelURL: URL, hiddenSize: Int) async throws -> EagleDraftUnrolledEngine {
        var spec = SpecializationOptions(preferredComputeUnitKind: .gpu)
        spec.expectFrequentReshapes = true
        let model = try await AIModel.specialize(
            contentsOf: aimodelURL, options: spec, cache: .default, cachePolicy: .persistent)
        return try EagleDraftUnrolledEngine(model: model, hiddenSize: hiddenSize)
    }

    private let outputName = "draft_tokens"

    /// One dispatch -> all K proposed draft token ids.
    func draftAll(token: Int32, hidden: NDArray, position: Int32,
                  kFull: NDArray, vFull: NDArray, kSliding: NDArray, vSliding: NDArray) async throws -> [Int] {
        var tokenId = NDArray(descriptor: inDesc["token_id"]!.resolvingDynamicDimensions([1, 1]))
        EagleND.fillI32(&tokenId, [token])
        var pos = NDArray(descriptor: inDesc["position_ids"]!.resolvingDynamicDimensions([1, 1]))
        EagleND.fillI32(&pos, [position])
        var h = hidden, kf = kFull, vf = vFull, ks = kSliding, vs = vSliding
        var tokens = NDArray(descriptor: outDesc.resolvingDynamicDimensions([1, numSteps]))
        var ov = InferenceFunction.MutableViews()
        ov.insert(&tokens, for: outputName)
        var noStates = InferenceFunction.MutableViews()
        _ = try await function.run(
            inputs: ["token_id": tokenId, "hidden": h, "position_ids": pos,
                     "k_full": kf, "v_full": vf, "k_sliding": ks, "v_sliding": vs],
            states: consume noStates, outputViews: consume ov)
        return tokens.view(as: Int32.self).withUnsafePointer { ptr, _, st in
            let cs = st[st.count - 1]
            return (0..<numSteps).map { Int(ptr[$0 * cs]) }
        }
    }
}

// MARK: - EAGLE loop

public final class EagleEngine {
    let target: EagleTargetEngine
    let draft: EagleDraftEngine
    let tokenizer: any Tokenizer
    let draftTokens: Int
    let backbone: Int
    let slidingWindow: Int
    let maxContext: Int
    public let loadSeconds: Double
    private let stopIds: Set<Int>

    let draftUnrolled: EagleDraftUnrolledEngine?

    private init(target: EagleTargetEngine, draft: EagleDraftEngine, tokenizer: any Tokenizer,
                 draftTokens: Int, backbone: Int, slidingWindow: Int, maxContext: Int,
                 loadSeconds: Double, stopIds: Set<Int>,
                 draftUnrolled: EagleDraftUnrolledEngine? = nil) {
        self.target = target; self.draft = draft; self.tokenizer = tokenizer
        self.draftUnrolled = draftUnrolled
        // With an unrolled draft, K is fixed by the exported graph.
        self.draftTokens = max(1, draftUnrolled?.numSteps ?? draftTokens); self.backbone = backbone
        self.slidingWindow = slidingWindow; self.maxContext = maxContext
        self.loadSeconds = loadSeconds; self.stopIds = stopIds
    }

    public static func load(targetURL: URL, draftURL: URL, tokenizerDir: URL, draftTokens: Int,
                            vocabSize: Int, backbone: Int, slidingWindow: Int, maxContext: Int,
                            verbose: Bool, unrolledURL: URL? = nil) async throws -> EagleEngine {
        let t0 = Date()
        async let tok = AutoTokenizer.from(modelFolder: tokenizerDir)
        async let tgt = EagleTargetEngine.load(aimodelURL: targetURL, vocabSize: vocabSize, hiddenSize: backbone)
        async let drf = EagleDraftEngine.load(aimodelURL: draftURL, vocabSize: vocabSize, hiddenSize: backbone)
        let tokenizer = try await tok
        let target = try await tgt
        let draft = try await drf
        var unrolled: EagleDraftUnrolledEngine? = nil
        if let u = unrolledURL {
            unrolled = try await EagleDraftUnrolledEngine.load(aimodelURL: u, hiddenSize: backbone)
        }
        // Same turn-ending stop set as the standard engine, read from the model's published
        // generation_config.json eos_token_id list (gemma-4: [1,106,50]) so greedy EAGLE halts at
        // the real turn boundary instead of overrunning and repeating.
        let stops = LLMEngine.stopTokenIds(tokenizer: tokenizer, tokenizerDir: tokenizerDir)
        return EagleEngine(
            target: target, draft: draft, tokenizer: tokenizer, draftTokens: draftTokens,
            backbone: backbone, slidingWindow: slidingWindow, maxContext: maxContext,
            loadSeconds: Date().timeIntervalSince(t0), stopIds: stops, draftUnrolled: unrolled)
    }

    /// The four (cropped) shared-KV tensors the draft attends to. Full layer keeps the whole
    /// verified prefix; sliding layers are cropped to the last `slidingWindow` positions.
    private func sliceKV(_ o: EagleTargetEngine.Out, length: Int) -> (NDArray, NDArray, NDArray, NDArray) {
        let fullStart = 0, fullLen = length
        let slideStart = max(0, length - slidingWindow)
        let slideLen = length - slideStart
        let kfd = target.kFullDescriptor(), ksd = target.kSlidingDescriptor()
        return (
            EagleND.kvSlice(o.kFull, start: fullStart, len: fullLen, descriptor: kfd),
            EagleND.kvSlice(o.vFull, start: fullStart, len: fullLen, descriptor: kfd),
            EagleND.kvSlice(o.kSliding, start: slideStart, len: slideLen, descriptor: ksd),
            EagleND.kvSlice(o.vSliding, start: slideStart, len: slideLen, descriptor: ksd))
    }

    /// Diagnostic: greedy-decode using ONLY the target (no draft). Isolates whether the EAGLE
    /// target bundle is coherent (vs a bug in the draft/verify/seeding loop).
    func generateTargetOnly(promptTokens: [Int], options: CoreAIPipeline.Options,
                            onToken: ((String) -> Void)?) async throws -> CoreAIPipeline.SpeculativeResult {
        let maxTokens = max(0, options.maxTokens)
        target.allocateCache(capacity: min(promptTokens.count + maxTokens + 8, maxContext))
        let prompt32 = promptTokens.map { Int32($0) }
        var pf: EagleTargetEngine.Out!
        var ps = 0
        while ps < prompt32.count { let pe = min(ps + 6, prompt32.count); pf = try await target.forward(Array(prompt32[ps..<pe])); ps = pe }
        var committed = promptTokens
        var generated: [Int] = []
        var streamed = ""
        func emit(_ t: Int) -> Bool {
            if stopIds.contains(t) || generated.count >= maxTokens { return false }
            generated.append(t); committed.append(t)
            if let onToken { let txt = tokenizer.decode(tokens: generated)
                if txt.hasPrefix(streamed) { let d = String(txt.dropFirst(streamed.count)); if !d.isEmpty { onToken(d) } }
                streamed = txt }
            return true
        }
        let t0 = Date()
        var running = emit(Sampler.argmax(pf.logitsRows[pf.logitsRows.count - 1]))
        while running {
            let o = try await target.forward([Int32(committed[committed.count - 1])])
            running = emit(Sampler.argmax(o.logitsRows[0]))
        }
        let dec = Date().timeIntervalSince(t0)
        return CoreAIPipeline.SpeculativeResult(
            text: tokenizer.decode(tokens: generated), promptTokenCount: promptTokens.count,
            generatedTokenCount: generated.count, stopReason: .maxTokens, modelLoadSeconds: loadSeconds,
            prefillSeconds: 0, decodeSeconds: dec, draftTokens: 0, draftedTokens: 0,
            acceptedDraftTokens: 0, iterations: generated.count)
    }

    /// Public server entry: apply the chat template (or raw-encode) to OpenAI/Anthropic-style
    /// `messages`, then run the EAGLE speculative loop. Used by `CoreAIServer` to serve the MTP
    /// model through the same `/v1/chat/completions` path as standard models.
    public func generate(messages: [[String: String]], options: CoreAIPipeline.Options,
                         tools: [[String: any Sendable]]? = nil,
                         onToken: ((String) -> Void)?) async throws -> CoreAIPipeline.SpeculativeResult {
        let promptTokens: [Int]
        if options.applyChatTemplate {
            if let tools, !tools.isEmpty {
                promptTokens = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
            } else {
                promptTokens = try tokenizer.applyChatTemplate(messages: messages)
            }
        } else {
            promptTokens = tokenizer.encode(text: messages.last?["content"] ?? "")
        }
        // Live speculative metrics are published by the caller (ModelHandle), which knows the
        // served model name.
        return try await generate(promptTokens: promptTokens, options: options, onToken: onToken)
    }

    func generate(promptTokens: [Int], options: CoreAIPipeline.Options,
                  onToken: ((String) -> Void)?) async throws -> CoreAIPipeline.SpeculativeResult {
        func log(_ s: @autoclosure () -> String) {
            if options.verbose { FileHandle.standardError.write(Data(("[coreai] " + s() + "\n").utf8)) }
        }
        let maxTokens = max(0, options.maxTokens)
        let K = draftTokens
        let capacity = min(promptTokens.count + maxTokens + K + 8, maxContext)
        target.allocateCache(capacity: capacity)
        log("eagle prompt -> \(promptTokens.count) tokens, K=\(K), cap=\(capacity)")

        // Chunk the prefill: the full-attention layer's MPS SDPA threadgroup memory scales with
        // query length and overflows (>32KB) past ~10 query tokens at head_dim 512. Feeding the
        // prompt in <=6-token chunks keeps each forward's query count small; the cache accumulates,
        // and the LAST chunk's output carries the seed hidden + full-length representative KV.
        let prefillStart = Date()
        let prompt32 = promptTokens.map { Int32($0) }
        let chunk = 6
        var pf: EagleTargetEngine.Out!
        var ps = 0
        while ps < prompt32.count {
            let pe = min(ps + chunk, prompt32.count)
            pf = try await target.forward(Array(prompt32[ps..<pe]))
            ps = pe
        }
        let lastChunkRows = pf.logitsRows.count  // rows of the final prefill chunk
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        var committed = promptTokens
        var generated: [Int] = []
        var streamed = ""
        var finalTextOverride: String?
        var stop: CoreAIPipeline.StopReason = .maxTokens
        var drafted = 0, accepted = 0, iters = 0
        let stopSequences = options.stopSequences.filter { !$0.isEmpty }

        func emitVisibleText(_ text: String) {
            guard let onToken else {
                streamed = text
                return
            }
            if text.hasPrefix(streamed) {
                let delta = String(text.dropFirst(streamed.count))
                if !delta.isEmpty { onToken(delta) }
            }
            streamed = text
        }

        func emit(_ token: Int) -> Bool {
            if stopIds.contains(token) { stop = .eos; return false }
            if generated.count >= maxTokens { stop = .maxTokens; return false }
            if committed.count >= maxContext { stop = .contextLimit; return false }
            generated.append(token); committed.append(token)
            if onToken != nil || !stopSequences.isEmpty {
                let text = tokenizer.decode(tokens: generated)
                if let stopRange = CoreAIPipeline.firstStopRange(
                    in: text, stopSequences: stopSequences)
                {
                    let visible = String(text[..<stopRange.lowerBound])
                    emitVisibleText(visible)
                    finalTextOverride = visible
                    stop = .stopSequence
                    return false
                } else {
                    let visible = stopSequences.isEmpty
                        ? text
                        : CoreAIPipeline.visibleTextAvoidingPartialStop(text, stopSequences: stopSequences)
                    emitVisibleText(visible)
                }
            }
            return true
        }

        let decodeStart = Date()
        var running = emit(Sampler.argmax(pf.logitsRows[lastChunkRows - 1]))

        // Seed the first draft: hidden that produced the just-emitted token (last row of the final
        // prefill chunk) + KV over the prompt.
        var seedHidden = EagleND.hiddenRow(pf.hidden, row: lastChunkRows - 1,
                                           dim: backbone, descriptor: target.hiddenDescriptor())
        // Crop shared_kv to the full committed length INCLUDING the anchor position (matches the
        // reference `current_length`); capped at the most-recent forward's written extent (after
        // prefill the anchor's own KV isn't computed yet, so this degrades to prompt-only).
        var seedKV = sliceKV(pf, length: min(committed.count, target.lastWrittenEnd))

        while running {
            let L = committed.count
            let anchor = committed[L - 1]
            let pos = Int32(L - 1)

            // DRAFT K tokens (EAGLE recurrence at constant position). Unrolled engine fuses all K
            // micro-steps into ONE Core AI dispatch; otherwise step K times sequentially.
            var drafts: [Int]
            if let unrolled = draftUnrolled {
                drafts = try await unrolled.draftAll(
                    token: Int32(anchor), hidden: seedHidden, position: pos,
                    kFull: seedKV.0, vFull: seedKV.1, kSliding: seedKV.2, vSliding: seedKV.3)
            } else {
                drafts = []
                drafts.reserveCapacity(K)
                var token = Int32(anchor)
                var hidden = seedHidden
                for _ in 0..<K {
                    let o = try await draft.step(
                        token: token, hidden: hidden, position: pos,
                        kFull: seedKV.0, vFull: seedKV.1, kSliding: seedKV.2, vSliding: seedKV.3)
                    let d = Sampler.argmax(o.logits)
                    drafts.append(d)
                    token = Int32(d); hidden = o.nextHidden
                }
            }

            // VERIFY: target forward over [anchor, drafts]; outputs logits/hidden/KV.
            let vf = try await target.forward([Int32(anchor)] + drafts.map { Int32($0) })
            let verdict = SpeculativeEngine.verify(drafts: drafts, targetRows: vf.logitsRows)
            let n = verdict.acceptedCount

            drafted += K; accepted += n; iters += 1
            target.rollback(to: L + n)

            for t in verdict.acceptedTokens where running { if !emit(t) { running = false } }
            if running { if !emit(verdict.correctionToken) { running = false } }
            if !running { break }

            // Reseed from the verify forward: hidden row n produced the correction; KV up to the
            // new committed length - 1 (drops rejected-draft positions + the correction itself).
            seedHidden = EagleND.hiddenRow(vf.hidden, row: n, dim: backbone,
                                           descriptor: target.hiddenDescriptor())
            seedKV = sliceKV(vf, length: min(committed.count, target.lastWrittenEnd))
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let text = finalTextOverride ?? tokenizer.decode(tokens: generated)
        if finalTextOverride == nil, onToken != nil {
            emitVisibleText(text)
        }
        let accRate = drafted > 0 ? Double(accepted) / Double(drafted) : 0
        log(String(format: "eagle decode %d tok in %.3fs (%.1f tok/s) over %d passes; "
                   + "drafts %d accepted %d (%.1f%%), %.2f tok/pass, stop=%@",
                   generated.count, decodeSeconds,
                   decodeSeconds > 0 ? Double(generated.count) / decodeSeconds : 0,
                   iters, drafted, accepted, accRate * 100,
                   iters > 0 ? Double(generated.count) / Double(iters) : 0, stop.rawValue))

        return CoreAIPipeline.SpeculativeResult(
            text: text, promptTokenCount: promptTokens.count, generatedTokenCount: generated.count,
            stopReason: stop, modelLoadSeconds: loadSeconds, prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds, draftTokens: K, draftedTokens: drafted,
            acceptedDraftTokens: accepted, iterations: iters)
    }
}
#endif
