import XCTest

@testable import PipelineRuntime

final class Gemma4EagleFixedShapeABITests: XCTestCase {
    func testPrefillUsesQ5BlocksAndQ1TailOnly() {
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.prefillQueryWidths(tokenCount: 0),
            [])
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.prefillQueryWidths(tokenCount: 4),
            [1, 1, 1, 1])
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.prefillQueryWidths(tokenCount: 5),
            [5])
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.prefillQueryWidths(tokenCount: 13),
            [5, 5, 1, 1, 1])
    }

    func testVerifyUsesQ5UntilOnlyQ1CanFit() {
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.decodeQueryWidth(processedTokens: 4_091),
            5)
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.decodeQueryWidth(processedTokens: 4_092),
            1)
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.decodeQueryWidth(processedTokens: 4_095),
            1)
        XCTAssertNil(
            Gemma4EagleExecutionPolicy.decodeQueryWidth(processedTokens: 4_096))
    }

    func testDecodeIterationStopsBeforeRedundantVerifyAtTokenBudget() {
        XCTAssertTrue(
            Gemma4EagleExecutionPolicy.shouldRunDecode(
                generatedTokens: 3,
                maximumTokens: 4,
                processedTokens: 100))
        XCTAssertFalse(
            Gemma4EagleExecutionPolicy.shouldRunDecode(
                generatedTokens: 4,
                maximumTokens: 4,
                processedTokens: 100))
        XCTAssertFalse(
            Gemma4EagleExecutionPolicy.shouldRunDecode(
                generatedTokens: 0,
                maximumTokens: 4,
                processedTokens: 4_095))
        XCTAssertFalse(
            Gemma4EagleExecutionPolicy.shouldRunDecode(
                generatedTokens: 0,
                maximumTokens: 4,
                processedTokens: 4_096))
    }

    func testTargetInvocationAllowsOnlyQueryOnlyAbsoluteQ1AndQ5() throws {
        XCTAssertNoThrow(
            try Gemma4EagleTargetContract.validateInvocation(
                queryWidth: 5,
                positionIDs: [11, 12, 13, 14, 15],
                processedTokens: 11))
        XCTAssertNoThrow(
            try Gemma4EagleTargetContract.validateInvocation(
                queryWidth: 1,
                positionIDs: [4_095],
                processedTokens: 4_095))

        assertInvocationRejected(
            queryWidth: 4,
            positions: [0, 1, 2, 3],
            processed: 0,
            contains: "Q must be exactly 1 or 5")
        assertInvocationRejected(
            queryWidth: 5,
            positions: [0, 1, 2, 3, 9],
            processed: 0,
            contains: "query-only absolute")
        assertInvocationRejected(
            queryWidth: 5,
            positions: [4_092, 4_093, 4_094, 4_095, 4_096],
            processed: 4_092,
            contains: "4096")
    }

    func testTargetDescriptorAcceptsExact26BResidentABI() throws {
        XCTAssertNoThrow(
            try Gemma4EagleTargetContract.validate(Self.validTargetFunction))
    }

    func testTargetDescriptorRejectsLegacyCacheAndPositionABIs() {
        var function = Self.validTargetFunction
        function.inputs["position_ids"] = .init(
            scalarType: .int32,
            shape: [1, Gemma4EagleTargetContract.cacheCapacity])
        assertTargetRejected(function, contains: "position_ids")

        function = Self.validTargetFunction
        function.states["k_cache"] = .init(
            scalarType: .float16,
            shape: [30, 1, 8, -1, 256])
        assertTargetRejected(function, contains: "k_cache")

        function = Self.validTargetFunction
        function.outputs["hidden"] = .init(
            scalarType: .float16,
            shape: [1, -1, 5_376])
        assertTargetRejected(function, contains: "hidden")
    }

    func testSlidingStagePlanKeepsNewestWindowLeftAligned() {
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.slidingStagePlan(validFullLength: 100),
            .init(source: 0..<100, destination: 0..<100))
        XCTAssertEqual(
            Gemma4EagleExecutionPolicy.slidingStagePlan(validFullLength: 1_026),
            .init(source: 2..<1_026, destination: 0..<1_024))
    }

    // MARK: - geometry presets

    func testGeometryPresetsMatchVerifiedGemma26BAnd31BTable() {
        let g26 = Gemma4EagleGeometry.gemma26B
        XCTAssertEqual(g26.layers, 30)
        XCTAssertEqual(g26.backboneHiddenSize, 2_816)
        XCTAssertEqual(g26.cacheKVHeads, 8)
        XCTAssertEqual(g26.cacheHeadDimension, 256)
        XCTAssertEqual(g26.fullKVHeads, 2)
        XCTAssertEqual(g26.fullHeadDimension, 512)
        XCTAssertEqual(g26.slidingKVHeads, 8)
        XCTAssertEqual(g26.slidingHeadDimension, 256)
        XCTAssertEqual(g26.vocabularySize, 262_144)
        XCTAssertEqual(g26.cacheCapacity, 4_096)
        XCTAssertEqual(g26.slidingWindow, 1_024)
        XCTAssertEqual(g26.draftTokens, 4)

        let g31 = Gemma4EagleGeometry.gemma31B
        XCTAssertEqual(g31.layers, 60)
        XCTAssertEqual(g31.backboneHiddenSize, 5_376)
        XCTAssertEqual(g31.cacheKVHeads, 16)
        XCTAssertEqual(g31.cacheHeadDimension, 256)
        XCTAssertEqual(g31.fullKVHeads, 4)
        XCTAssertEqual(g31.fullHeadDimension, 512)
        XCTAssertEqual(g31.slidingKVHeads, 16)
        XCTAssertEqual(g31.slidingHeadDimension, 256)
        XCTAssertEqual(g31.vocabularySize, 262_144)
        XCTAssertEqual(g31.cacheCapacity, 4_096)
        XCTAssertEqual(g31.slidingWindow, 1_024)
        XCTAssertEqual(g31.draftTokens, 4)
    }

    func testForBackboneResolvesFamiliesAndRejectsUnknownBackbones() {
        XCTAssertEqual(Gemma4EagleGeometry.forBackbone(2_816), .gemma26B)
        XCTAssertEqual(Gemma4EagleGeometry.forBackbone(5_376), .gemma31B)
        XCTAssertNil(Gemma4EagleGeometry.forBackbone(4_096))
        XCTAssertNil(Gemma4EagleGeometry.forBackbone(0))
        XCTAssertEqual(
            Gemma4EagleGeometry.supportedBackbones,
            [2_816, 5_376])
    }

    func testTargetDescriptorsAreGeometryDrivenFor26BAnd31B() {
        // 26B reproduces the exact current resident ABI descriptors.
        let out26 = Gemma4EagleTargetContract.expectedOutputs(.gemma26B)
        XCTAssertEqual(out26["logits"], .init(scalarType: .float16, shape: [1, -1, 262_144]))
        XCTAssertEqual(out26["hidden"], .init(scalarType: .float16, shape: [1, -1, 2_816]))
        XCTAssertEqual(out26["k_full"], .init(scalarType: .float16, shape: [1, 2, -1, 512]))
        XCTAssertEqual(out26["v_full"], .init(scalarType: .float16, shape: [1, 2, -1, 512]))
        XCTAssertEqual(out26["k_sliding"], .init(scalarType: .float16, shape: [1, 8, -1, 256]))
        XCTAssertEqual(out26["v_sliding"], .init(scalarType: .float16, shape: [1, 8, -1, 256]))
        let state26 = Gemma4EagleTargetContract.expectedStates(.gemma26B)
        XCTAssertEqual(
            state26["k_cache"],
            .init(scalarType: .float16, shape: [30, 1, 8, 4_096, 256]))
        XCTAssertEqual(
            state26["v_cache"],
            .init(scalarType: .float16, shape: [30, 1, 8, 4_096, 256]))

        // 31B produces the new dense descriptors.
        let out31 = Gemma4EagleTargetContract.expectedOutputs(.gemma31B)
        XCTAssertEqual(out31["logits"], .init(scalarType: .float16, shape: [1, -1, 262_144]))
        XCTAssertEqual(out31["hidden"], .init(scalarType: .float16, shape: [1, -1, 5_376]))
        XCTAssertEqual(out31["k_full"], .init(scalarType: .float16, shape: [1, 4, -1, 512]))
        XCTAssertEqual(out31["v_full"], .init(scalarType: .float16, shape: [1, 4, -1, 512]))
        XCTAssertEqual(out31["k_sliding"], .init(scalarType: .float16, shape: [1, 16, -1, 256]))
        XCTAssertEqual(out31["v_sliding"], .init(scalarType: .float16, shape: [1, 16, -1, 256]))
        let state31 = Gemma4EagleTargetContract.expectedStates(.gemma31B)
        XCTAssertEqual(
            state31["k_cache"],
            .init(scalarType: .float16, shape: [60, 1, 16, 4_096, 256]))
        XCTAssertEqual(
            state31["v_cache"],
            .init(scalarType: .float16, shape: [60, 1, 16, 4_096, 256]))
    }

    func testTargetValidatesGemma31BResidentABIAndRejectsCrossGeometry() throws {
        XCTAssertNoThrow(
            try Gemma4EagleTargetContract.validate(
                Self.valid31BTargetFunction, geometry: .gemma31B))

        // The 31B target must fail closed under 26B geometry (default), and vice versa.
        assertTargetRejected(Self.valid31BTargetFunction, contains: "hidden")
        XCTAssertThrowsError(
            try Gemma4EagleTargetContract.validate(
                Self.validTargetFunction, geometry: .gemma31B)
        ) { error in
            XCTAssertTrue(String(describing: error).contains("hidden"))
        }
    }

    #if COREAI_RUNTIME
    func testProductionEngineConsumesFixedShapePolicyAndKVLength() throws {
        let source = try String(
            contentsOf: Self.pipelineRuntimeSource(named: "EagleEngine.swift"),
            encoding: .utf8)

        XCTAssertTrue(source.contains("Gemma4EagleTargetContract.validateModel("))
        XCTAssertTrue(source.contains("Gemma4EagleTargetContract.validateInvocation("))
        XCTAssertTrue(source.contains("Gemma4EagleExecutionPolicy.prefillQueryWidths("))
        XCTAssertTrue(source.contains("Gemma4EagleExecutionPolicy.decodeQueryWidth("))
        XCTAssertTrue(source.contains("Gemma4EagleExecutionPolicy.shouldRunDecode("))
        XCTAssertGreaterThanOrEqual(
            source.components(
                separatedBy: "Gemma4MTPNativeContract.validateRuntimeInvocation(").count - 1,
            2,
            "single-step and unrolled assistant dispatches must validate actual static staging")
        XCTAssertGreaterThanOrEqual(
            source.components(separatedBy: "\"kv_length\"").count - 1,
            2,
            "single-step and unrolled assistant dispatches must pass kv_length")
        XCTAssertFalse(source.contains("resolvingDynamicDimensions([1, seqLen])"))
        XCTAssertFalse(source.contains("allocateCache(capacity:"))
    }

    private static func pipelineRuntimeSource(named filename: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/PipelineRuntime")
            .appendingPathComponent(filename)
    }
    #endif

    private func assertInvocationRejected(
        queryWidth: Int,
        positions: [Int32],
        processed: Int,
        contains expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try Gemma4EagleTargetContract.validateInvocation(
                queryWidth: queryWidth,
                positionIDs: positions,
                processedTokens: processed),
            file: file,
            line: line
        ) { error in
            XCTAssertTrue(
                String(describing: error).contains(expected),
                "\(error) does not contain \(expected)",
                file: file,
                line: line)
        }
    }

    private func assertTargetRejected(
        _ function: Gemma4MTPFunctionDescriptor,
        contains expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try Gemma4EagleTargetContract.validate(function),
            file: file,
            line: line
        ) { error in
            XCTAssertTrue(
                String(describing: error).contains(expected),
                "\(error) does not contain \(expected)",
                file: file,
                line: line)
        }
    }

    private static let validTargetFunction = Gemma4MTPFunctionDescriptor(
        inputs: [
            "input_ids": .init(scalarType: .int32, shape: [1, -1]),
            "position_ids": .init(scalarType: .int32, shape: [1, -1]),
        ],
        outputs: [
            "logits": .init(scalarType: .float16, shape: [1, -1, 262_144]),
            "hidden": .init(scalarType: .float16, shape: [1, -1, 2_816]),
            "k_full": .init(scalarType: .float16, shape: [1, 2, -1, 512]),
            "v_full": .init(scalarType: .float16, shape: [1, 2, -1, 512]),
            "k_sliding": .init(scalarType: .float16, shape: [1, 8, -1, 256]),
            "v_sliding": .init(scalarType: .float16, shape: [1, 8, -1, 256]),
        ],
        states: [
            "k_cache": .init(
                scalarType: .float16,
                shape: [30, 1, 8, 4_096, 256]),
            "v_cache": .init(
                scalarType: .float16,
                shape: [30, 1, 8, 4_096, 256]),
        ])

    private static let valid31BTargetFunction = Gemma4MTPFunctionDescriptor(
        inputs: [
            "input_ids": .init(scalarType: .int32, shape: [1, -1]),
            "position_ids": .init(scalarType: .int32, shape: [1, -1]),
        ],
        outputs: [
            "logits": .init(scalarType: .float16, shape: [1, -1, 262_144]),
            "hidden": .init(scalarType: .float16, shape: [1, -1, 5_376]),
            "k_full": .init(scalarType: .float16, shape: [1, 4, -1, 512]),
            "v_full": .init(scalarType: .float16, shape: [1, 4, -1, 512]),
            "k_sliding": .init(scalarType: .float16, shape: [1, 16, -1, 256]),
            "v_sliding": .init(scalarType: .float16, shape: [1, 16, -1, 256]),
        ],
        states: [
            "k_cache": .init(
                scalarType: .float16,
                shape: [60, 1, 16, 4_096, 256]),
            "v_cache": .init(
                scalarType: .float16,
                shape: [60, 1, 16, 4_096, 256]),
        ])
}
