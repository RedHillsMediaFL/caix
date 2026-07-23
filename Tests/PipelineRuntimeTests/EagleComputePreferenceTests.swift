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

    func testEveryProductionEagleLoaderUsesEaglePreferenceWithoutHardcodedGPU() throws {
        let eagleSource = try source(named: "EagleEngine.swift")
        let nativeRunnerSource = try source(named: "Gemma4MTPNativeRunner.swift")
        let selection = "preferredComputeUnitKind: LLMEngine.preferredEagleComputeUnit()"

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
