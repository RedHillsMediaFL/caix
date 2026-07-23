import Foundation

struct Gemma4EagleSlidingStagePlan: Equatable, Sendable {
    let source: Range<Int>
    let destination: Range<Int>
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

    private static let expectedOutputs: [String: Gemma4MTPTensorDescriptor] = [
        "logits": .init(scalarType: .float16, shape: [1, -1, vocabularySize]),
        "hidden": .init(scalarType: .float16, shape: [1, -1, backboneHiddenSize]),
        "k_full": .init(
            scalarType: .float16,
            shape: [1, fullKVHeads, -1, fullHeadDimension]),
        "v_full": .init(
            scalarType: .float16,
            shape: [1, fullKVHeads, -1, fullHeadDimension]),
        "k_sliding": .init(
            scalarType: .float16,
            shape: [1, slidingKVHeads, -1, slidingHeadDimension]),
        "v_sliding": .init(
            scalarType: .float16,
            shape: [1, slidingKVHeads, -1, slidingHeadDimension]),
    ]

    private static let expectedStates: [String: Gemma4MTPTensorDescriptor] = [
        "k_cache": .init(
            scalarType: .float16,
            shape: [layers, 1, cacheKVHeads, cacheCapacity, cacheHeadDimension]),
        "v_cache": .init(
            scalarType: .float16,
            shape: [layers, 1, cacheKVHeads, cacheCapacity, cacheHeadDimension]),
    ]

    static func validateModel(
        assetURL: URL,
        functionNames: [String],
        function: Gemma4MTPFunctionDescriptor
    ) throws {
        try Gemma4MTPNativeContract.validateAssetURL(assetURL)
        guard functionNames == ["main"] else {
            throw ContractError.invalid(
                "target entrypoints must be exactly [\"main\"]; got \(functionNames.sorted())")
        }
        try validate(function)
    }

    static func validate(_ function: Gemma4MTPFunctionDescriptor) throws {
        try validateCategory(
            function.inputs,
            expected: expectedInputs,
            category: "input")
        try validateCategory(
            function.outputs,
            expected: expectedOutputs,
            category: "output")
        try validateCategory(
            function.states,
            expected: expectedStates,
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
