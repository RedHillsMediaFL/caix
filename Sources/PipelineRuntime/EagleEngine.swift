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
//     k_sliding, v_sliding, kv_length) -> outputs (logits, next_hidden)  [stateless]
//
// The verify/accept + cache bookkeeping mirror `SpeculativeEngine` exactly (greedy: every committed
// token is the target's argmax, so output is byte-identical to target-only). The novelty is the
// draft proposal: seeded from the target's hidden + fixed-capacity shared-KV staging, recurring at
// a constant position.

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

    static func fillI32(_ a: inout NDArray, _ v: [Int32]) {
        var view = a.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: v)
    }

    /// Append the `n` new-position columns of `src` (`[1, H, n, Dh]`) into the accumulator
    /// `dst` (`[1, H, cap, Dh]`) at sequence offset `off`. The EAGLE target now emits only the
    /// new positions' repr K/V each forward (reading the persistent KV-cache state back as a graph
    /// output breaks the in-place state mutation under torch.export); Swift grows the full prefix.
    static func appendKV(
        _ dst: inout NDArray,
        _ src: NDArray,
        atOffset off: Int,
        count: Int
    ) {
        let s = src.shape  // [1, H, n, Dh]
        let h = s[1], dh = s[3]
        precondition(count >= 0 && count <= s[2])
        precondition(off >= 0 && off + count <= dst.shape[2])
        src.view(as: Float16.self).withUnsafePointer { ip, _, ist in
            let ihS = ist[1], isS = ist[2], idS = ist[3]
            var dv = dst.mutableView(as: Float16.self)
            dv.withUnsafeMutablePointer { op, _, ost in
                let ohS = ost[1], osS = ost[2], odS = ost[3]
                for hi in 0..<h {
                    for si in 0..<count {
                        let ib = hi * ihS + si * isS
                        let ob = hi * ohS + (off + si) * osS
                        for di in 0..<dh { op[ob + di * odS] = ip[ib + di * idS] }
                    }
                }
            }
        }
    }

    /// Appends committed target rows to the static assistant sliding staging tensor. When the
    /// window is full, the oldest rows are shifted out and the retained/new tail remains
    /// left-aligned in slots `0..<min(validBefore + count, capacity)`.
    static func appendSlidingKV(
        _ dst: inout NDArray,
        _ src: NDArray,
        validBefore: Int,
        count: Int
    ) {
        let sourceShape = src.shape
        let heads = sourceShape[1]
        let headDimension = sourceShape[3]
        let capacity = dst.shape[2]
        precondition(count >= 0 && count <= sourceShape[2] && count <= capacity)
        let oldCount = min(validBefore, capacity)
        let overflow = max(0, oldCount + count - capacity)
        let retained = oldCount - overflow

        var destinationView = dst.mutableView(as: Float16.self)
        destinationView.withUnsafeMutablePointer { destination, _, destinationStrides in
            let destinationHeadStride = destinationStrides[1]
            let destinationSequenceStride = destinationStrides[2]
            let destinationDimensionStride = destinationStrides[3]

            if overflow > 0 {
                for head in 0..<heads {
                    for position in 0..<retained {
                        let sourcePosition = position + overflow
                        let sourceBase =
                            head * destinationHeadStride
                            + sourcePosition * destinationSequenceStride
                        let destinationBase =
                            head * destinationHeadStride
                            + position * destinationSequenceStride
                        for dimension in 0..<headDimension {
                            destination[
                                destinationBase + dimension * destinationDimensionStride
                            ] = destination[
                                sourceBase + dimension * destinationDimensionStride
                            ]
                        }
                    }
                }
            }

            src.view(as: Float16.self).withUnsafePointer { source, _, sourceStrides in
                let sourceHeadStride = sourceStrides[1]
                let sourceSequenceStride = sourceStrides[2]
                let sourceDimensionStride = sourceStrides[3]
                for head in 0..<heads {
                    for position in 0..<count {
                        let sourceBase =
                            head * sourceHeadStride + position * sourceSequenceStride
                        let destinationBase =
                            head * destinationHeadStride
                            + (retained + position) * destinationSequenceStride
                        for dimension in 0..<headDimension {
                            destination[
                                destinationBase + dimension * destinationDimensionStride
                            ] = source[sourceBase + dimension * sourceDimensionStride]
                        }
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
    private(set) var hostProcessed: Int = 0
    // Static assistant staging: full KV is absolute-position aligned to 4096; sliding KV contains
    // the newest min(kv_length, 1024) positions left-aligned.
    private var accKFull: NDArray
    private var accVFull: NDArray
    private var accKSliding: NDArray
    private var accVSliding: NDArray
    private let hf: Int, df: Int, hs: Int, ds: Int  // full/sliding head counts + head dims
    let geometry: Gemma4EagleGeometry

    struct Out {
        let logitsRows: [[Float]]
        let hidden: NDArray  // [1, Q, D]
        let kFullNew: NDArray
        let vFullNew: NDArray
        let kSlidingNew: NDArray
        let vSlidingNew: NDArray
        let startOffset: Int
        let queryWidth: Int
    }

    private init(
        model: AIModel,
        assetURL: URL,
        vocabSize: Int,
        hiddenSize: Int,
        geometry: Gemma4EagleGeometry
    ) throws {
        self.geometry = geometry
        guard let d = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle target: no 'main'")
        }
        try Gemma4EagleTargetContract.validateModel(
            assetURL: assetURL,
            functionNames: model.functionNames,
            function: Gemma4MTPNativeRunner.project(d),
            geometry: geometry)
        guard vocabSize == geometry.vocabularySize else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle target: configured vocabulary \(vocabSize) must equal "
                    + "\(geometry.vocabularySize)")
        }
        guard hiddenSize == geometry.backboneHiddenSize else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle target: configured backbone \(hiddenSize) must equal "
                    + "\(geometry.backboneHiddenSize)")
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
        let resolvedHiddenSize = geometry.backboneHiddenSize
        let kfShp = outs["k_full"]!.shape
        let ksShp = outs["k_sliding"]!.shape
        self.hf = kfShp[1]; self.df = kfShp[3]
        self.hs = ksShp[1]; self.ds = ksShp[3]
        self.accKFull = NDArray(descriptor: outs["k_full"]!.resolvingDynamicDimensions(
            [1, kfShp[1], geometry.cacheCapacity, kfShp[3]]))
        self.accVFull = NDArray(descriptor: outs["v_full"]!.resolvingDynamicDimensions(
            [1, kfShp[1], geometry.cacheCapacity, kfShp[3]]))
        self.accKSliding = NDArray(descriptor: outs["k_sliding"]!.resolvingDynamicDimensions(
            [1, ksShp[1], geometry.slidingWindow, ksShp[3]]))
        self.accVSliding = NDArray(descriptor: outs["v_sliding"]!.resolvingDynamicDimensions(
            [1, ksShp[1], geometry.slidingWindow, ksShp[3]]))
        self.keyName = "k_cache"
        self.valueName = "v_cache"
        guard let kd = nd("state", keyName), let vd = nd("state", valueName) else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle target: KV state not NDArray")
        }
        self.keyCache = NDArray(descriptor: kd)
        self.valueCache = NDArray(descriptor: vd)
        self.vocabSize = vocabSize
        self.hiddenSize = resolvedHiddenSize
        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle target: load 'main' failed")
        }
        self.function = fn
        self.kd = kd
        self.vd = vd
    }
    private let kd: NDArrayDescriptor
    private let vd: NDArrayDescriptor

    static func load(
        aimodelURL: URL,
        vocabSize: Int,
        hiddenSize: Int,
        geometry: Gemma4EagleGeometry
    ) async throws -> EagleTargetEngine {
        try Gemma4MTPNativeContract.validateAssetURL(aimodelURL)
        var spec = LLMEngine.eagleSpecializationOptions()
        spec.expectFrequentReshapes = false
        let model = try await AIModel(contentsOf: aimodelURL, options: spec)
        return try EagleTargetEngine(
            model: model,
            assetURL: aimodelURL,
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            geometry: geometry)
    }

    func allocateCache() {
        keyCache = NDArray(descriptor: kd)
        valueCache = NDArray(descriptor: vd)
        accKFull = NDArray(descriptor: outDesc["k_full"]!.resolvingDynamicDimensions(
            [1, hf, geometry.cacheCapacity, df]))
        accVFull = NDArray(descriptor: outDesc["v_full"]!.resolvingDynamicDimensions(
            [1, hf, geometry.cacheCapacity, df]))
        accKSliding = NDArray(descriptor: outDesc["k_sliding"]!.resolvingDynamicDimensions(
            [1, hs, geometry.slidingWindow, ds]))
        accVSliding = NDArray(descriptor: outDesc["v_sliding"]!.resolvingDynamicDimensions(
            [1, hs, geometry.slidingWindow, ds]))
        processed = 0
        hostProcessed = 0
    }

    func rollback(to count: Int) throws {
        guard count == hostProcessed, count >= 0, count <= processed else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle target: rollback \(count) must equal committed host KV "
                    + "\(hostProcessed) within processed \(processed)")
        }
        processed = count
    }

    func forward(_ tokens: [Int32]) async throws -> Out {
        let n = tokens.count
        let offset = processed
        let absolutePositions = (offset..<(offset + n)).map(Int32.init)
        try Gemma4EagleTargetContract.validateInvocation(
            queryWidth: n,
            positionIDs: absolutePositions,
            processedTokens: offset)
        var inputIds = NDArray(descriptor: inDesc["input_ids"]!.resolvingDynamicDimensions([1, n]))
        EagleND.fillI32(&inputIds, tokens)
        var positionIds = NDArray(
            descriptor: inDesc["position_ids"]!.resolvingDynamicDimensions([1, n]))
        EagleND.fillI32(&positionIds, absolutePositions)

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

        processed += n
        return Out(
            logitsRows: Self.allRows(logits, vocab: vocabSize),
            hidden: hidden,
            kFullNew: kFullNew,
            vFullNew: vFullNew,
            kSlidingNew: kSlidingNew,
            vSlidingNew: vSlidingNew,
            startOffset: offset,
            queryWidth: n)
    }

    func commitHostKV(_ output: Out, count: Int) throws {
        guard output.startOffset == hostProcessed,
              count > 0,
              count <= output.queryWidth,
              hostProcessed + count <= geometry.cacheCapacity
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle target: invalid host KV commit start=\(output.startOffset), "
                    + "host=\(hostProcessed), count=\(count), Q=\(output.queryWidth)")
        }
        EagleND.appendKV(
            &accKFull,
            output.kFullNew,
            atOffset: hostProcessed,
            count: count)
        EagleND.appendKV(
            &accVFull,
            output.vFullNew,
            atOffset: hostProcessed,
            count: count)
        EagleND.appendSlidingKV(
            &accKSliding,
            output.kSlidingNew,
            validBefore: hostProcessed,
            count: count)
        EagleND.appendSlidingKV(
            &accVSliding,
            output.vSlidingNew,
            validBefore: hostProcessed,
            count: count)
        hostProcessed += count
    }

    func stagedKV() throws -> (
        kFull: NDArray,
        vFull: NDArray,
        kSliding: NDArray,
        vSliding: NDArray,
        kvLength: Int32
    ) {
        guard hostProcessed == processed, let kvLength = Int32(exactly: hostProcessed) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle target: staged KV \(hostProcessed) does not match state \(processed)")
        }
        return (
            accKFull,
            accVFull,
            accKSliding,
            accVSliding,
            kvLength)
    }

    func hiddenDescriptor() -> NDArrayDescriptor { outDesc["hidden"]! }

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

// MARK: - EAGLE draft (8-in / 2-out, stateless)

final class EagleDraftEngine {
    private let function: InferenceFunction
    private let inDesc: [String: NDArrayDescriptor]
    private let outDesc: [String: NDArrayDescriptor]
    private let tokenInputName: String
    private let hiddenInputName: String
    private let positionInputName: String
    private let kFullInputName: String
    private let vFullInputName: String
    private let kSlidingInputName: String
    private let vSlidingInputName: String
    private let logitsOutputName: String
    private let nextHiddenOutputName: String
    let vocabSize: Int
    let hiddenSize: Int
    let geometry: Gemma4EagleGeometry

    private init(model: AIModel, vocabSize: Int, hiddenSize: Int, geometry: Gemma4EagleGeometry) throws {
        self.geometry = geometry
        guard let d = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle draft: no 'main'")
        }
        try Gemma4MTPNativeContract.validate(Gemma4MTPNativeRunner.project(d), geometry: geometry)
        guard model.functionNames == ["main"] else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle draft: entrypoints must be exactly [\"main\"]; "
                    + "got \(model.functionNames.sorted())")
        }
        guard vocabSize == geometry.vocabularySize,
              hiddenSize == geometry.backboneHiddenSize
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle draft: configured geometry (vocab \(vocabSize), backbone \(hiddenSize)) "
                    + "does not match fixed assistant ABI "
                    + "(vocab \(geometry.vocabularySize), backbone \(geometry.backboneHiddenSize))")
        }
        var ins: [String: NDArrayDescriptor] = [:]
        for n in d.inputNames { if case .ndArray(let x) = d.inputDescriptor(of: n) { ins[n] = x } }
        var outs: [String: NDArrayDescriptor] = [:]
        for n in d.outputNames { if case .ndArray(let x) = d.outputDescriptor(of: n) { outs[n] = x } }
        self.inDesc = ins
        self.outDesc = outs
        self.tokenInputName = "token_id"
        self.hiddenInputName = "hidden"
        self.positionInputName = "position_ids"
        self.kFullInputName = "k_full"
        self.vFullInputName = "v_full"
        self.kSlidingInputName = "k_sliding"
        self.vSlidingInputName = "v_sliding"
        self.logitsOutputName = "logits"
        self.nextHiddenOutputName = "next_hidden"
        self.vocabSize = vocabSize
        self.hiddenSize = geometry.backboneHiddenSize
        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle draft: load 'main' failed")
        }
        self.function = fn
    }

    static func load(
        aimodelURL: URL,
        vocabSize: Int,
        hiddenSize: Int,
        geometry: Gemma4EagleGeometry
    ) async throws -> EagleDraftEngine {
        try Gemma4MTPNativeContract.validateAssetURL(aimodelURL)
        var spec = LLMEngine.eagleSpecializationOptions()
        spec.expectFrequentReshapes = false
        let model = try await AIModel(contentsOf: aimodelURL, options: spec)
        return try EagleDraftEngine(
            model: model, vocabSize: vocabSize, hiddenSize: hiddenSize, geometry: geometry)
    }

    struct Out { let logits: [Float]; let nextHidden: NDArray }

    /// One draft micro-step. `hidden` is `[1,1,backbone]`; KV uses the static 4096/1024 staging
    /// buffers and `kvLength` masks their valid left-aligned prefixes.
    func step(
        token: Int32,
        hidden: NDArray,
        position: Int32,
        kFull: NDArray,
        vFull: NDArray,
        kSliding: NDArray,
        vSliding: NDArray,
        kvLength: Int32
    ) async throws -> Out {
        try Gemma4MTPNativeContract.validateRuntimeInvocation(
            positionID: position,
            kvLength: kvLength,
            hiddenShape: hidden.shape,
            kFullShape: kFull.shape,
            vFullShape: vFull.shape,
            kSlidingShape: kSliding.shape,
            vSlidingShape: vSliding.shape,
            geometry: geometry)
        var tokenId = NDArray(descriptor: inDesc[tokenInputName]!.resolvingDynamicDimensions([1, 1]))
        EagleND.fillI32(&tokenId, [token])
        var pos = NDArray(descriptor: inDesc[positionInputName]!.resolvingDynamicDimensions([1, 1]))
        EagleND.fillI32(&pos, [position])
        var validLength = NDArray(descriptor: inDesc["kv_length"]!)
        EagleND.fillI32(&validLength, [kvLength])
        let h = hidden, kf = kFull, vf = vFull, ks = kSliding, vs = vSliding
        var logits = NDArray(descriptor: outDesc[logitsOutputName]!.resolvingDynamicDimensions([1, 1, vocabSize]))
        var nextHidden = NDArray(
            descriptor: outDesc[nextHiddenOutputName]!.resolvingDynamicDimensions([1, 1, hiddenSize]))
        var ov = InferenceFunction.MutableViews()
        ov.insert(&logits, for: logitsOutputName)
        ov.insert(&nextHidden, for: nextHiddenOutputName)
        let noStates = InferenceFunction.MutableViews()
        _ = try await function.run(
            inputs: [
                tokenInputName: tokenId, hiddenInputName: h, positionInputName: pos,
                kFullInputName: kf, vFullInputName: vf,
                kSlidingInputName: ks, vSlidingInputName: vs,
                "kv_length": validLength,
            ],
            states: consume noStates, outputViews: consume ov)
        let row = logits.view(as: Float16.self).withUnsafePointer { ptr, _, st in
            let cs = st[st.count - 1]
            return (0..<vocabSize).map { Float(ptr[$0 * cs]) }
        }
        return Out(logits: row, nextHidden: nextHidden)
    }
}

// MARK: - EAGLE unrolled draft (8-in / 1-out, stateless; K micro-steps fused in-graph)

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
    let geometry: Gemma4EagleGeometry

    private init(model: AIModel, hiddenSize: Int, geometry: Gemma4EagleGeometry) throws {
        self.geometry = geometry
        guard let d = model.functionDescriptor(for: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle unrolled draft: no 'main'")
        }
        try Gemma4MTPNativeContract.validateUnrolled(
            Gemma4MTPNativeRunner.project(d), geometry: geometry)
        guard model.functionNames == ["main"] else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle unrolled draft: entrypoints must be exactly [\"main\"]; "
                    + "got \(model.functionNames.sorted())")
        }
        guard hiddenSize == geometry.backboneHiddenSize else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle unrolled draft: configured backbone \(hiddenSize) does not match fixed ABI "
                    + "\(geometry.backboneHiddenSize)")
        }
        var ins: [String: NDArrayDescriptor] = [:]
        for n in d.inputNames { if case .ndArray(let x) = d.inputDescriptor(of: n) { ins[n] = x } }
        guard case .ndArray(let od) = d.outputDescriptor(of: d.outputNames[0]) else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle unrolled draft: output not NDArray")
        }
        self.inDesc = ins
        self.outDesc = od
        self.numSteps = geometry.draftTokens
        self.hiddenSize = geometry.backboneHiddenSize
        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIPipeline.RuntimeError.modelContract("eagle unrolled draft: load 'main' failed")
        }
        self.function = fn
    }

    static func load(
        aimodelURL: URL,
        hiddenSize: Int,
        geometry: Gemma4EagleGeometry
    ) async throws -> EagleDraftUnrolledEngine {
        try Gemma4MTPNativeContract.validateAssetURL(aimodelURL)
        var spec = LLMEngine.eagleSpecializationOptions()
        spec.expectFrequentReshapes = false
        let model = try await AIModel(contentsOf: aimodelURL, options: spec)
        return try EagleDraftUnrolledEngine(model: model, hiddenSize: hiddenSize, geometry: geometry)
    }

    private let outputName = "draft_tokens"

    /// One dispatch -> all K proposed draft token ids.
    func draftAll(
        token: Int32,
        hidden: NDArray,
        position: Int32,
        kFull: NDArray,
        vFull: NDArray,
        kSliding: NDArray,
        vSliding: NDArray,
        kvLength: Int32
    ) async throws -> [Int] {
        try Gemma4MTPNativeContract.validateRuntimeInvocation(
            positionID: position,
            kvLength: kvLength,
            hiddenShape: hidden.shape,
            kFullShape: kFull.shape,
            vFullShape: vFull.shape,
            kSlidingShape: kSliding.shape,
            vSlidingShape: vSliding.shape,
            geometry: geometry)
        var tokenId = NDArray(descriptor: inDesc["token_id"]!.resolvingDynamicDimensions([1, 1]))
        EagleND.fillI32(&tokenId, [token])
        var pos = NDArray(descriptor: inDesc["position_ids"]!.resolvingDynamicDimensions([1, 1]))
        EagleND.fillI32(&pos, [position])
        var validLength = NDArray(descriptor: inDesc["kv_length"]!)
        EagleND.fillI32(&validLength, [kvLength])
        let h = hidden, kf = kFull, vf = vFull, ks = kSliding, vs = vSliding
        var tokens = NDArray(descriptor: outDesc.resolvingDynamicDimensions([1, numSteps]))
        var ov = InferenceFunction.MutableViews()
        ov.insert(&tokens, for: outputName)
        let noStates = InferenceFunction.MutableViews()
        _ = try await function.run(
            inputs: ["token_id": tokenId, "hidden": h, "position_ids": pos,
                     "k_full": kf, "v_full": vf, "k_sliding": ks, "v_sliding": vs,
                     "kv_length": validLength],
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
    private let chatRenderer: Gemma4ChatTemplateContract.ResidentRenderer
    public let backbone: Int
    let maxContext: Int
    public let loadSeconds: Double
    private let stopIds: Set<Int>

    let draftUnrolled: EagleDraftUnrolledEngine?

    private init(target: EagleTargetEngine, draft: EagleDraftEngine, tokenizer: any Tokenizer,
                 chatRenderer: Gemma4ChatTemplateContract.ResidentRenderer,
                 backbone: Int, maxContext: Int,
                 loadSeconds: Double, stopIds: Set<Int>,
                 draftUnrolled: EagleDraftUnrolledEngine? = nil) {
        self.target = target; self.draft = draft; self.tokenizer = tokenizer
        self.chatRenderer = chatRenderer
        self.draftUnrolled = draftUnrolled
        self.backbone = backbone
        self.maxContext = maxContext
        self.loadSeconds = loadSeconds; self.stopIds = stopIds
    }

    public static func load(targetURL: URL, draftURL: URL, tokenizerDir: URL, draftTokens: Int,
                            vocabSize: Int, backbone: Int, slidingWindow: Int, maxContext: Int,
                            verbose: Bool, unrolledURL: URL? = nil) async throws -> EagleEngine {
        let t0 = Date()
        _ = verbose
        // The backbone width selects the whole fixed-shape family (26B or the dense 31B). Everything
        // else in the ABI (K, vocab, sliding window, context) is shared, so it is validated against
        // the resolved geometry rather than hardcoded to one model.
        guard let geometry = Gemma4EagleGeometry.forBackbone(backbone) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle requires a supported Gemma 4 EAGLE backbone; got \(backbone), "
                    + "supported \(Gemma4EagleGeometry.supportedBackbones)")
        }
        guard draftTokens == geometry.draftTokens,
              vocabSize == geometry.vocabularySize,
              slidingWindow == geometry.slidingWindow,
              maxContext == geometry.cacheCapacity
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle requires the fixed Gemma 4 ABI for backbone=\(geometry.backboneHiddenSize): "
                    + "K=\(geometry.draftTokens), vocab=\(geometry.vocabularySize), "
                    + "sliding=\(geometry.slidingWindow), context=\(geometry.cacheCapacity)")
        }
        // Authenticate and compile the July prompt contract before allocating either model.
        // EAGLE is the Gemma 4 MTP backend; stale Gemma templates must fail closed rather than
        // silently changing turn boundaries or tool-call semantics.
        let chatRenderer = try Gemma4ChatTemplateContract.ResidentRenderer(
            tokenizerDirectory: tokenizerDir)
        async let tok = AutoTokenizer.from(modelFolder: tokenizerDir)
        let target = try await EagleTargetEngine.load(
            aimodelURL: targetURL, vocabSize: vocabSize, hiddenSize: backbone, geometry: geometry)
        let resolvedBackbone = target.hiddenSize
        guard resolvedBackbone == geometry.backboneHiddenSize else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "eagle target backbone \(resolvedBackbone) does not match selected geometry "
                    + "\(geometry.backboneHiddenSize)")
        }
        async let drf = EagleDraftEngine.load(
            aimodelURL: draftURL, vocabSize: vocabSize, hiddenSize: resolvedBackbone,
            geometry: geometry)
        let tokenizer = try await tok
        let draft = try await drf
        var unrolled: EagleDraftUnrolledEngine? = nil
        if let u = unrolledURL {
            unrolled = try await EagleDraftUnrolledEngine.load(
                aimodelURL: u, hiddenSize: resolvedBackbone, geometry: geometry)
        }
        // Same turn-ending stop set as the standard engine, read from the model's published
        // generation_config.json eos_token_id list (gemma-4: [1,106,50]) so greedy EAGLE halts at
        // the real turn boundary instead of overrunning and repeating.
        let stops = LLMEngine.stopTokenIds(tokenizer: tokenizer, tokenizerDir: tokenizerDir)
        return EagleEngine(
            target: target, draft: draft, tokenizer: tokenizer,
            chatRenderer: chatRenderer,
            backbone: resolvedBackbone, maxContext: maxContext,
            loadSeconds: Date().timeIntervalSince(t0), stopIds: stops, draftUnrolled: unrolled)
    }

    /// Decode using ONLY the target. Besides isolating target-bundle correctness, this is the
    /// distribution-preserving path for sampled requests until EAGLE exposes draft probabilities
    /// and implements probabilistic acceptance plus residual correction sampling.
    func generateTargetOnly(promptTokens: [Int], options: CoreAIPipeline.Options,
                            onToken: ((String) -> Void)?) async throws -> CoreAIPipeline.SpeculativeResult {
        guard !promptTokens.isEmpty else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("prompt tokenized to 0 tokens")
        }
        guard promptTokens.count <= target.geometry.cacheCapacity else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "eagle prompt exceeds fixed \(target.geometry.cacheCapacity)-token target state")
        }
        let maxTokens = max(0, options.maxTokens)
        target.allocateCache()
        let prompt32 = promptTokens.map { Int32($0) }
        let prefillStart = Date()
        var pf: EagleTargetEngine.Out!
        var ps = 0
        for queryWidth in Gemma4EagleExecutionPolicy.prefillQueryWidths(
            tokenCount: prompt32.count)
        {
            let pe = ps + queryWidth
            pf = try await target.forward(Array(prompt32[ps..<pe]))
            try target.commitHostKV(pf, count: queryWidth)
            ps = pe
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        var committed = promptTokens
        var generated: [Int] = []
        var streamed = ""
        var finalTextOverride: String?
        var stop: CoreAIPipeline.StopReason = .maxTokens
        let stopSequences = options.stopSequences.filter { !$0.isEmpty }
        let sampler = Sampler(
            temperature: options.temperature,
            topK: options.topK,
            topP: options.topP)
        var rng = SeededGenerator(seed: options.seed ?? UInt64.random(in: .min ... .max))

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

        func emit(_ t: Int) -> Bool {
            if stopIds.contains(t) {
                stop = .eos
                return false
            }
            if generated.count >= maxTokens {
                stop = .maxTokens
                return false
            }
            if committed.count >= maxContext {
                stop = .contextLimit
                return false
            }
            generated.append(t)
            committed.append(t)
            if onToken != nil || !stopSequences.isEmpty {
                let text = tokenizer.decode(tokens: generated)
                if let stopRange = CoreAIPipeline.firstStopRange(
                    in: text,
                    stopSequences: stopSequences)
                {
                    let visible = String(text[..<stopRange.lowerBound])
                    emitVisibleText(visible)
                    finalTextOverride = visible
                    stop = .stopSequence
                    return false
                }
                let visible = stopSequences.isEmpty
                    ? text
                    : CoreAIPipeline.visibleTextAvoidingPartialStop(
                        text,
                        stopSequences: stopSequences)
                emitVisibleText(visible)
            }
            return true
        }
        let t0 = Date()
        var running = emit(sampler.sample(pf.logitsRows[pf.logitsRows.count - 1], using: &rng))
        while running && Gemma4EagleExecutionPolicy.shouldRunDecode(
            generatedTokens: generated.count,
            maximumTokens: maxTokens,
            processedTokens: target.processed)
        {
            let o = try await target.forward([Int32(committed[committed.count - 1])])
            try target.commitHostKV(o, count: 1)
            running = emit(sampler.sample(o.logitsRows[0], using: &rng))
        }
        if running && generated.count < maxTokens {
            stop = .contextLimit
        }
        let dec = Date().timeIntervalSince(t0)
        let text = finalTextOverride ?? tokenizer.decode(tokens: generated)
        if finalTextOverride == nil, onToken != nil {
            emitVisibleText(text)
        }
        return CoreAIPipeline.SpeculativeResult(
            text: text, promptTokenCount: promptTokens.count,
            generatedTokenCount: generated.count, stopReason: stop, modelLoadSeconds: loadSeconds,
            prefillSeconds: prefillSeconds, decodeSeconds: dec, draftTokens: 0, draftedTokens: 0,
            acceptedDraftTokens: 0, iterations: generated.count)
    }

    /// Public server entry: apply the chat template (or raw-encode) to OpenAI/Anthropic-style
    /// `messages`, then run the EAGLE speculative loop. Used by `CoreAIServer` to serve the MTP
    /// model through the same `/v1/chat/completions` path as standard models.
    public func generate(
        messages: [[String: any Sendable]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.SpeculativeResult {
        guard options.constrainedJSONSchema == nil else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "JSON-schema constrained decoding is not supported on EAGLE speculative decoding backends")
        }
        let promptTokens = try encodePrompt(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            applyChatTemplate: options.applyChatTemplate)
        // Live speculative metrics are published by the caller (ModelHandle), which knows the
        // served model name.
        return try await generate(promptTokens: promptTokens, options: options, onToken: onToken)
    }

    /// Compatibility entry for legacy Chat Completions callers. New Responses callers use the
    /// rich overload above so nested reasoning/tool objects are never flattened.
    public func generate(
        messages: [[String: String]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.SpeculativeResult {
        let rich: [[String: any Sendable]] = messages.map { message in
            var value: [String: any Sendable] = [:]
            for (key, item) in message { value[key] = item }
            return value
        }
        return try await generate(
            messages: rich,
            options: options,
            tools: tools,
            additionalContext: nil,
            onToken: onToken)
    }

    func encodePrompt(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        applyChatTemplate: Bool
    ) throws -> [Int] {
        if applyChatTemplate {
            return try chatRenderer.encode(
                tokenizer: tokenizer,
                messages: messages,
                tools: tools,
                additionalContext: additionalContext)
        }
        guard let text = messages.last?["content"] as? String else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "raw EAGLE prompting requires a final string content item")
        }
        return tokenizer.encode(text: text)
    }

    func generate(promptTokens: [Int], options: CoreAIPipeline.Options,
                  onToken: ((String) -> Void)?) async throws -> CoreAIPipeline.SpeculativeResult {
        guard options.constrainedJSONSchema == nil else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "JSON-schema constrained decoding is not supported on EAGLE speculative decoding backends")
        }
        if SpeculativeExecutionPolicy.route(options: options) == .targetOnlySampled {
            return try await generateTargetOnly(
                promptTokens: promptTokens,
                options: options,
                onToken: onToken)
        }
        guard !promptTokens.isEmpty else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("prompt tokenized to 0 tokens")
        }
        guard promptTokens.count <= target.geometry.cacheCapacity else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "eagle prompt exceeds fixed \(target.geometry.cacheCapacity)-token target state")
        }
        func log(_ s: @autoclosure () -> String) {
            if options.verbose { FileHandle.standardError.write(Data(("[coreai] " + s() + "\n").utf8)) }
        }
        let maxTokens = max(0, options.maxTokens)
        target.allocateCache()
        log(
            "eagle prompt -> \(promptTokens.count) tokens, K=\(target.geometry.draftTokens), "
                + "cap=\(target.geometry.cacheCapacity), prefill=Q5+Q1-tail")

        // Keep specialization bounded to the ABI's two query shapes: complete Q5 blocks followed
        // by one Q1 forward per tail token. The last forward supplies the anchor logits/hidden.
        let prefillStart = Date()
        let prompt32 = promptTokens.map { Int32($0) }
        var pf: EagleTargetEngine.Out!
        var ps = 0
        for queryWidth in Gemma4EagleExecutionPolicy.prefillQueryWidths(
            tokenCount: prompt32.count)
        {
            let pe = ps + queryWidth
            pf = try await target.forward(Array(prompt32[ps..<pe]))
            try target.commitHostKV(pf, count: queryWidth)
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
        var seedKV = try target.stagedKV()

        while running && Gemma4EagleExecutionPolicy.shouldRunDecode(
            generatedTokens: generated.count,
            maximumTokens: maxTokens,
            processedTokens: target.processed)
        {
            let L = committed.count
            let anchor = committed[L - 1]
            let pos = Int32(L - 1)
            guard let queryWidth = Gemma4EagleExecutionPolicy.decodeQueryWidth(
                processedTokens: target.processed)
            else {
                break
            }

            if queryWidth == 1 {
                let q1 = try await target.forward([Int32(anchor)])
                try target.commitHostKV(q1, count: 1)
                iters += 1
                running = emit(Sampler.argmax(q1.logitsRows[0]))
                seedHidden = EagleND.hiddenRow(
                    q1.hidden,
                    row: 0,
                    dim: backbone,
                    descriptor: target.hiddenDescriptor())
                seedKV = try target.stagedKV()
                continue
            }

            // The resident ABI always drafts K4 and verifies [anchor, d0, d1, d2, d3] as Q5.
            var drafts: [Int]
            if let unrolled = draftUnrolled {
                drafts = try await unrolled.draftAll(
                    token: Int32(anchor), hidden: seedHidden, position: pos,
                    kFull: seedKV.kFull, vFull: seedKV.vFull,
                    kSliding: seedKV.kSliding, vSliding: seedKV.vSliding,
                    kvLength: seedKV.kvLength)
            } else {
                drafts = []
                drafts.reserveCapacity(Gemma4EagleExecutionPolicy.draftTokens)
                var token = Int32(anchor)
                var hidden = seedHidden
                for _ in 0..<Gemma4EagleExecutionPolicy.draftTokens {
                    let o = try await draft.step(
                        token: token, hidden: hidden, position: pos,
                        kFull: seedKV.kFull, vFull: seedKV.vFull,
                        kSliding: seedKV.kSliding, vSliding: seedKV.vSliding,
                        kvLength: seedKV.kvLength)
                    let d = Sampler.argmax(o.logits)
                    drafts.append(d)
                    token = Int32(d); hidden = o.nextHidden
                }
            }

            // VERIFY: target forward over [anchor, drafts]; outputs logits/hidden/KV.
            let vf = try await target.forward([Int32(anchor)] + drafts.map { Int32($0) })
            let verdict = SpeculativeEngine.verify(drafts: drafts, targetRows: vf.logitsRows)
            let n = verdict.acceptedCount

            drafted += Gemma4EagleExecutionPolicy.draftTokens
            accepted += n
            iters += 1
            // Commit the anchor plus accepted proposal prefix; rejected suffix rows stay outside
            // logical target state and never enter the assistant staging buffers.
            try target.commitHostKV(vf, count: n + 1)
            try target.rollback(to: L + n)

            for t in verdict.acceptedTokens where running { if !emit(t) { running = false } }
            if running { if !emit(verdict.correctionToken) { running = false } }
            if !running { break }

            // Reseed from the verify forward: hidden row n produced the correction; KV up to the
            // new committed length - 1 (drops rejected-draft positions + the correction itself).
            seedHidden = EagleND.hiddenRow(vf.hidden, row: n, dim: backbone,
                                           descriptor: target.hiddenDescriptor())
            seedKV = try target.stagedKV()
        }
        if running && generated.count < maxTokens {
            stop = .contextLimit
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let text = finalTextOverride ?? tokenizer.decode(tokens: generated)
        if finalTextOverride == nil, onToken != nil {
            emitVisibleText(text)
        }
        let accRate = drafted > 0 ? Double(accepted) / Double(drafted) : 0
        log(String(format: "eagle decode %d tok in %.3fs (%.1f tok/s) over %d passes; "
                   + "drafts %d accepted %d (%.1f%%), %.2f tok/pass, final K=%d, stop=%@",
                   generated.count, decodeSeconds,
                   decodeSeconds > 0 ? Double(generated.count) / decodeSeconds : 0,
                   iters, drafted, accepted, accRate * 100,
                   iters > 0 ? Double(generated.count) / Double(iters) : 0,
                   Gemma4EagleExecutionPolicy.draftTokens, stop.rawValue))

        return CoreAIPipeline.SpeculativeResult(
            text: text, promptTokenCount: promptTokens.count, generatedTokenCount: generated.count,
            stopReason: stop, modelLoadSeconds: loadSeconds, prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            draftTokens: Gemma4EagleExecutionPolicy.draftTokens,
            draftedTokens: drafted,
            acceptedDraftTokens: accepted, iterations: iters)
    }
}
#endif
