import Foundation
import XCTest

@testable import CoreAIServer

final class ConversionGuardTests: XCTestCase {
    func testDisabledGuardAllowsGenerationEvenWithLockAndProcess() {
        let lock = URL(fileURLWithPath: "/tmp/.agent-heavy-task.lock")
        let decision = ConversionGuard.evaluate(
            enabled: false,
            lockPaths: [lock],
            fileExists: { _ in true },
            processLines: ["python -m coreai_models.export.pipeline"])

        XCTAssertNil(decision)
    }

    func testLockFileBlocksGeneration() {
        let lock = URL(fileURLWithPath: "/tmp/.agent-heavy-task.lock")
        let decision = ConversionGuard.evaluate(
            enabled: true,
            lockPaths: [lock],
            retryAfterSeconds: 45,
            fileExists: { $0 == lock },
            processLines: [])

        XCTAssertEqual(decision?.retryAfterSeconds, 45)
        XCTAssertTrue(decision?.reason.contains(".agent-heavy-task.lock") == true)
    }

    func testExporterProcessBlocksGeneration() {
        let decision = ConversionGuard.evaluate(
            enabled: true,
            lockPaths: [],
            fileExists: { _ in false },
            processLines: ["20473 python -m coreai_models.export.pipeline --model example"])

        XCTAssertEqual(decision?.reason, "Core AI model conversion is active")
    }

    func testServeCommandWithConvertScriptDoesNotLookLikeActiveExporter() {
        let decision = ConversionGuard.evaluate(
            enabled: true,
            lockPaths: [],
            fileExists: { _ in false },
            processLines: ["123 caix serve --convert-script ./python/converter/convert.py"])

        XCTAssertNil(decision)
    }
}
