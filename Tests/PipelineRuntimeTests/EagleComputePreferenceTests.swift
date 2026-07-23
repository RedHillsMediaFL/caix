import Foundation
import XCTest

@testable import PipelineRuntime

#if COREAI_RUNTIME
import CoreAI

final class EagleComputePreferenceTests: XCTestCase {
    func testEagleOverrideWinsOverGeneralComputePreference() {
        XCTAssertEqual(
            LLMEngine.preferredEagleComputeUnit(environment: [
                "COREAI_EAGLE_COMPUTE": "ane",
                "COREAI_COMPUTE": "gpu",
            ]),
            .neuralEngine)
        XCTAssertEqual(
            LLMEngine.preferredEagleComputeUnit(environment: [
                "COREAI_EAGLE_COMPUTE": "GPU",
                "COREAI_COMPUTE": "ane",
            ]),
            .gpu)
        XCTAssertEqual(
            LLMEngine.preferredEagleComputeUnit(environment: [
                "COREAI_EAGLE_COMPUTE": "cpu",
                "COREAI_COMPUTE": "gpu",
            ]),
            .cpu)
    }

    func testEaglePreferenceFallsBackToGeneralComputePreference() {
        XCTAssertEqual(
            LLMEngine.preferredEagleComputeUnit(environment: [
                "COREAI_COMPUTE": "ane"
            ]),
            .neuralEngine)
        XCTAssertEqual(
            LLMEngine.preferredEagleComputeUnit(environment: [
                "COREAI_EAGLE_COMPUTE": "unsupported",
                "COREAI_COMPUTE": "cpu",
            ]),
            .cpu)
        XCTAssertEqual(
            LLMEngine.preferredEagleComputeUnit(environment: [:]),
            .gpu)
    }

    func testEagleANESpecializationExcludesGPU() {
        let options = LLMEngine.eagleSpecializationOptions(environment: [
            "COREAI_EAGLE_COMPUTE": "ane",
            "COREAI_COMPUTE": "gpu",
        ])

        XCTAssertEqual(options.allowedComputeUnitKinds, [.cpu, .neuralEngine])
        XCTAssertEqual(options.preferredComputeUnitKind, .neuralEngine)
    }

    func testEagleCPUSpecializationUsesCPUOnlyOptions() {
        let options = LLMEngine.eagleSpecializationOptions(environment: [
            "COREAI_EAGLE_COMPUTE": "cpu",
            "COREAI_COMPUTE": "gpu",
        ])

        XCTAssertEqual(options, .cpuOnly)
        XCTAssertEqual(options.allowedComputeUnitKinds, [.cpu])
        XCTAssertNil(options.preferredComputeUnitKind)
    }

    func testEagleGPUAndDefaultSpecializationPreservePreferredGPUOptions() {
        let expected = SpecializationOptions(preferredComputeUnitKind: .gpu)
        let explicit = LLMEngine.eagleSpecializationOptions(environment: [
            "COREAI_EAGLE_COMPUTE": "gpu",
            "COREAI_COMPUTE": "ane",
        ])
        let defaulted = LLMEngine.eagleSpecializationOptions(environment: [:])

        XCTAssertEqual(explicit, expected)
        XCTAssertEqual(defaulted, expected)
        XCTAssertEqual(expected.allowedComputeUnitKinds, [.cpu, .gpu, .neuralEngine])
        XCTAssertEqual(expected.preferredComputeUnitKind, .gpu)
    }

    func testEveryProductionEagleLoaderUsesEaglePreferenceWithoutHardcodedGPU() throws {
        let eagleSource = try source(named: "EagleEngine.swift")
        let nativeRunnerSource = try source(named: "Gemma4MTPNativeRunner.swift")
        let selection = "LLMEngine.eagleSpecializationOptions()"
        let directLoad = "AIModel(contentsOf:"

        XCTAssertEqual(
            eagleSource.components(separatedBy: selection).count - 1,
            3,
            "target, single-step draft, and unrolled draft must share the EAGLE preference")
        XCTAssertEqual(
            nativeRunnerSource.components(separatedBy: selection).count - 1,
            1,
            "the native MTP assistant runner must share the EAGLE preference")
        XCTAssertFalse(eagleSource.contains("preferredComputeUnitKind: .gpu"))
        XCTAssertFalse(nativeRunnerSource.contains("preferredComputeUnitKind: .gpu"))
        XCTAssertFalse(eagleSource.contains("LLMEngine.preferredEagleComputeUnit()"))
        XCTAssertFalse(nativeRunnerSource.contains("LLMEngine.preferredEagleComputeUnit()"))
        XCTAssertEqual(
            eagleSource.components(separatedBy: directLoad).count - 1,
            3,
            "all EAGLE engines must use the direct CoreAI load path")
        XCTAssertEqual(
            nativeRunnerSource.components(separatedBy: directLoad).count - 1,
            1,
            "the native MTP runner must use the direct CoreAI load path")
        XCTAssertFalse(eagleSource.contains("AIModel.specialize("))
        XCTAssertFalse(nativeRunnerSource.contains("AIModel.specialize("))
    }

    private func source(named filename: String) throws -> String {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: repositoryRoot
                .appendingPathComponent("Sources/PipelineRuntime")
                .appendingPathComponent(filename),
            encoding: .utf8)
    }
}
#endif
