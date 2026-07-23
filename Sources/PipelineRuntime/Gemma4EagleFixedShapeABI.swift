import Foundation

struct Gemma4EagleSlidingStagePlan: Equatable, Sendable {
    let source: Range<Int>
    let destination: Range<Int>
}

/// Fixed-shape geometry for one Gemma 4 EAGLE/MTP target+draft family. The runtime is compiled
/// against a closed set of query specializations, so every buffer, descriptor, and validation is
/// pinned to one of these presets. `backboneHiddenSize` is the selection key: it is the target's
/// backbone width and is shared by the dependent draft contract, so a single CLI `--backbone`
/// flag deterministically resolves the whole family.
struct Gemma4EagleGeometry: Sendable, Equatable {
    /// Number of transformer blocks (drives the leading `k_cache`/`v_cache` state dimension).
    let layers: Int
    /// Target backbone hidden width; shared verbatim by the draft's `hidden`/`next_hidden` tensors.
    let backboneHiddenSize: Int
    /// Persistent KV-cache state head count (`k_cache`/`v_cache` dim 2).
    let cacheKVHeads: Int
    /// Persistent KV-cache state per-head dimension (`k_cache`/`v_cache` dim 4).
    let cacheHeadDimension: Int
    /// Full-attention repr KV head count for the assistant staging tensors.
    let fullKVHeads: Int
    /// Full-attention repr KV per-head dimension.
    let fullHeadDimension: Int
    /// Sliding-attention repr KV head count for the assistant staging tensors.
    let slidingKVHeads: Int
    /// Sliding-attention repr KV per-head dimension.
    let slidingHeadDimension: Int
    /// Shared vocabulary size (logits width).
    let vocabularySize: Int
    /// Fixed target KV-state / assistant full-staging capacity in tokens.
    let cacheCapacity: Int
    /// Assistant sliding-staging window length in tokens.
    let slidingWindow: Int
    /// Draft depth K (draft_tokens output width and per-verify proposal count).
    let draftTokens: Int

    /// Current production model: Gemma-4-26B-A4B (MoE). EXACTLY the July resident ABI constants.
    static let gemma26B = Gemma4EagleGeometry(
        layers: 30,
        backboneHiddenSize: 2_816,
        cacheKVHeads: 8,
        cacheHeadDimension: 256,
        fullKVHeads: 2,
        fullHeadDimension: 512,
        slidingKVHeads: 8,
        slidingHeadDimension: 256,
        vocabularySize: 262_144,
        cacheCapacity: 4_096,
        slidingWindow: 1_024,
        draftTokens: 4)

    /// New dense Gemma-4-31B target: 2x the depth/heads/backbone, same cache/window/vocab/K.
    static let gemma31B = Gemma4EagleGeometry(
        layers: 60,
        backboneHiddenSize: 5_376,
        cacheKVHeads: 16,
        cacheHeadDimension: 256,
        fullKVHeads: 4,
        fullHeadDimension: 512,
        slidingKVHeads: 16,
        slidingHeadDimension: 256,
        vocabularySize: 262_144,
        cacheCapacity: 4_096,
        slidingWindow: 1_024,
        draftTokens: 4)

    /// Resolve the whole geometry family from the target backbone width (the CLI selection key).
    /// Returns `nil` for an unrecognized backbone so the load path can fail closed.
    static func forBackbone(_ hidden: Int) -> Gemma4EagleGeometry? {
        if hidden == gemma26B.backboneHiddenSize { return gemma26B }
        if hidden == gemma31B.backboneHiddenSize { return gemma31B }
        return nil
    }

    /// Human-readable list of supported backbones for error messages.
    static let supportedBackbones = [gemma26B.backboneHiddenSize, gemma31B.backboneHiddenSize]
}

/// Host-side dispatch rules for the resident Gemma 4 26B-A4B EAGLE target.
///
/// The exported target has one dynamic-Q `main`, but only Q1 and Q5 are part of the native
/// runtime ABI. Keeping that set closed prevents Core AI from compiling an unbounded collection
/// of query specializations while still providing a K4 verification forward.
enum Gemma4EagleExecutionPolicy {
    static let cacheCapacity = 4_096
    static let slidingWindow = 1_024
    static let draftTokens = 4
    static let verifyQueryWidth = draftTokens + 1

    static func prefillQueryWidths(tokenCount: Int) -> [Int] {
        guard tokenCount > 0 else { return [] }
        let q5Count = tokenCount / verifyQueryWidth
        let q1Count = tokenCount % verifyQueryWidth
        return [Int](repeating: verifyQueryWidth, count: q5Count)
            + [Int](repeating: 1, count: q1Count)
    }

    /// Returns Q5 while a complete K4 verification fits, then Q1 for safe target-only decode.
    static func decodeQueryWidth(processedTokens: Int) -> Int? {
        guard processedTokens >= 0, processedTokens < cacheCapacity else { return nil }
        return processedTokens + verifyQueryWidth <= cacheCapacity
            ? verifyQueryWidth
            : 1
    }

    static func shouldRunDecode(
        generatedTokens: Int,
        maximumTokens: Int,
        processedTokens: Int
    ) -> Bool {
        generatedTokens < maximumTokens
            && processedTokens + 1 < cacheCapacity
            && decodeQueryWidth(processedTokens: processedTokens) != nil
    }

    /// The assistant receives the newest sliding window left-aligned in its static Q-independent
    /// staging tensor. The full staging tensor remains absolute-position aligned.
    static func slidingStagePlan(validFullLength: Int) -> Gemma4EagleSlidingStagePlan {
        let boundedLength = min(max(0, validFullLength), cacheCapacity)
        let count = min(boundedLength, slidingWindow)
        return Gemma4EagleSlidingStagePlan(
            source: (boundedLength - count)..<boundedLength,
            destination: 0..<count)
    }
}

/// Exact descriptor and invocation contract for the July Gemma 4 26B-A4B resident target.
enum Gemma4EagleTargetContract {
    typealias ContractError = Gemma4MTPNativeContract.ContractError

    static let cacheCapacity = Gemma4EagleExecutionPolicy.cacheCapacity
    static let slidingWindow = Gemma4EagleExecutionPolicy.slidingWindow
    static let vocabularySize = Gemma4MTPNativeContract.vocabularySize
    static let backboneHiddenSize = Gemma4MTPNativeContract.backboneHiddenSize
    static let layers = 30
    static let cacheKVHeads = 8
    static let cacheHeadDimension = 256
    static let fullKVHeads = 2
    static let fullHeadDimension = 512
    static let slidingKVHeads = 8
    static let slidingHeadDimension = 256

    private static let expectedInputs: [String: Gemma4MTPTensorDescriptor] = [
        "input_ids": .init(scalarType: .int32, shape: [1, -1]),
        "position_ids": .init(scalarType: .int32, shape: [1, -1]),
    ]

    /// Dynamic-Q target outputs for a given geometry (`.gemma26B` reproduces the July ABI exactly).
    static func expectedOutputs(
        _ geometry: Gemma4EagleGeometry
    ) -> [String: Gemma4MTPTensorDescriptor] {
        [
            "logits": .init(scalarType: .float16, shape: [1, -1, geometry.vocabularySize]),
            "hidden": .init(scalarType: .float16, shape: [1, -1, geometry.backboneHiddenSize]),
            "k_full": .init(
                scalarType: .float16,
                shape: [1, geometry.fullKVHeads, -1, geometry.fullHeadDimension]),
            "v_full": .init(
                scalarType: .float16,
                shape: [1, geometry.fullKVHeads, -1, geometry.fullHeadDimension]),
            "k_sliding": .init(
                scalarType: .float16,
                shape: [1, geometry.slidingKVHeads, -1, geometry.slidingHeadDimension]),
            "v_sliding": .init(
                scalarType: .float16,
                shape: [1, geometry.slidingKVHeads, -1, geometry.slidingHeadDimension]),
        ]
    }

    /// Persistent KV-cache state descriptors for a given geometry.
    static func expectedStates(
        _ geometry: Gemma4EagleGeometry
    ) -> [String: Gemma4MTPTensorDescriptor] {
        [
            "k_cache": .init(
                scalarType: .float16,
                shape: [
                    geometry.layers, 1, geometry.cacheKVHeads,
                    geometry.cacheCapacity, geometry.cacheHeadDimension,
                ]),
            "v_cache": .init(
                scalarType: .float16,
                shape: [
                    geometry.layers, 1, geometry.cacheKVHeads,
                    geometry.cacheCapacity, geometry.cacheHeadDimension,
                ]),
        ]
    }

    static func validateModel(
        assetURL: URL,
        functionNames: [String],
        function: Gemma4MTPFunctionDescriptor,
        geometry: Gemma4EagleGeometry = .gemma26B
    ) throws {
        try Gemma4MTPNativeContract.validateAssetURL(assetURL)
        guard functionNames == ["main"] else {
            throw ContractError.invalid(
                "target entrypoints must be exactly [\"main\"]; got \(functionNames.sorted())")
        }
        try validate(function, geometry: geometry)
    }

    static func validate(
        _ function: Gemma4MTPFunctionDescriptor,
        geometry: Gemma4EagleGeometry = .gemma26B
    ) throws {
        try validateCategory(
            function.inputs,
            expected: expectedInputs,
            category: "input")
        try validateCategory(
            function.outputs,
            expected: expectedOutputs(geometry),
            category: "output")
        try validateCategory(
            function.states,
            expected: expectedStates(geometry),
            category: "state")
    }

    static func validateInvocation(
        queryWidth: Int,
        positionIDs: [Int32],
        processedTokens: Int
    ) throws {
        guard queryWidth == 1 || queryWidth == Gemma4EagleExecutionPolicy.verifyQueryWidth else {
            throw ContractError.invalid(
                "target Q must be exactly 1 or 5; got \(queryWidth)")
        }
        guard processedTokens >= 0, processedTokens + queryWidth <= cacheCapacity else {
            throw ContractError.invalid(
                "target query exceeds fixed 4096-token state: "
                    + "processed \(processedTokens), Q \(queryWidth)")
        }
        let expected = (processedTokens..<(processedTokens + queryWidth)).map(Int32.init)
        guard positionIDs == expected else {
            throw ContractError.invalid(
                "target position_ids must contain query-only absolute positions "
                    + "\(expected); got \(positionIDs)")
        }
    }

    private static func validateCategory(
        _ actual: [String: Gemma4MTPTensorDescriptor],
        expected: [String: Gemma4MTPTensorDescriptor],
        category: String
    ) throws {
        guard Set(actual.keys) == Set(expected.keys) else {
            throw ContractError.invalid(
                "target \(category)s must be exactly \(expected.keys.sorted()); "
                    + "got \(actual.keys.sorted())")
        }
        for name in expected.keys.sorted() where actual[name] != expected[name] {
            throw ContractError.invalid(
                "target \(category) \(name) must be \(expected[name]!); "
                    + "got \(String(describing: actual[name]))")
        }
    }
}
