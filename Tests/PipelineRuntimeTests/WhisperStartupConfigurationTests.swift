import Foundation
import XCTest

@testable import PipelineRuntime

final class WhisperStartupConfigurationTests: XCTestCase {
    func testNoWhisperOptionsLeavesResidentEngineDisabled() throws {
        XCTAssertNil(try WhisperStartupConfiguration.resolve(
            assetPath: nil,
            tokenizerPath: nil,
            modelLockPath: nil,
            maximumQueuedRequests: nil))
    }

    func testCompleteWhisperOptionsUseDefaultBoundedQueue() throws {
        let configuration = try XCTUnwrap(WhisperStartupConfiguration.resolve(
            assetPath: "/models/whisper.aimodel",
            tokenizerPath: "/models/whisper-tokenizer",
            modelLockPath: "/config/resident-model-lock.json",
            maximumQueuedRequests: nil))

        XCTAssertEqual(configuration.assetURL.path, "/models/whisper.aimodel")
        XCTAssertEqual(configuration.tokenizerDirectory.path, "/models/whisper-tokenizer")
        XCTAssertEqual(configuration.modelLockURL.path, "/config/resident-model-lock.json")
        XCTAssertEqual(
            configuration.maximumQueuedRequests,
            WhisperResidentEngine.defaultMaximumQueuedRequests)
    }

    func testPartialWhisperOptionsReportEveryMissingFlag() throws {
        XCTAssertThrowsError(try WhisperStartupConfiguration.resolve(
            assetPath: "/models/whisper.aimodel",
            tokenizerPath: nil,
            modelLockPath: nil,
            maximumQueuedRequests: 3)) { error in
                XCTAssertEqual(
                    error as? WhisperStartupConfiguration.ConfigurationError,
                    .incomplete(missingFlags: ["--whisper-tokenizer", "--resident-model-lock"]))
                XCTAssertEqual(
                    String(describing: error),
                    "resident Whisper requires all of --whisper-asset, --whisper-tokenizer, and --resident-model-lock; missing --whisper-tokenizer, --resident-model-lock")
            }

        XCTAssertThrowsError(try WhisperStartupConfiguration.resolve(
            assetPath: nil,
            tokenizerPath: nil,
            modelLockPath: nil,
            maximumQueuedRequests: 3)) { error in
                XCTAssertEqual(
                    error as? WhisperStartupConfiguration.ConfigurationError,
                    .incomplete(missingFlags: [
                        "--whisper-asset", "--whisper-tokenizer", "--resident-model-lock",
                    ]))
            }
    }

    func testRejectsEmptyPathAndNegativeQueueBound() throws {
        XCTAssertThrowsError(try WhisperStartupConfiguration.resolve(
            assetPath: "",
            tokenizerPath: "/models/whisper-tokenizer",
            modelLockPath: "/config/resident-model-lock.json",
            maximumQueuedRequests: nil)) { error in
                XCTAssertEqual(
                    error as? WhisperStartupConfiguration.ConfigurationError,
                    .emptyValue(flag: "--whisper-asset"))
            }

        XCTAssertThrowsError(try WhisperStartupConfiguration.resolve(
            assetPath: "/models/whisper.aimodel",
            tokenizerPath: "/models/whisper-tokenizer",
            modelLockPath: "/config/resident-model-lock.json",
            maximumQueuedRequests: -1)) { error in
                XCTAssertEqual(
                    error as? WhisperStartupConfiguration.ConfigurationError,
                    .invalidMaximumQueuedRequests(-1))
            }
    }

    func testZeroQueuedRequestsIsAValidNoWaitPolicy() throws {
        let configuration = try XCTUnwrap(WhisperStartupConfiguration.resolve(
            assetPath: "/models/whisper.aimodel",
            tokenizerPath: "/models/whisper-tokenizer",
            modelLockPath: "/config/resident-model-lock.json",
            maximumQueuedRequests: 0))

        XCTAssertEqual(configuration.maximumQueuedRequests, 0)
    }

    #if !COREAI_RUNTIME
    func testConfiguredWhisperFailsClearlyWhenRuntimeIsNotLinked() async throws {
        let configuration = try XCTUnwrap(WhisperStartupConfiguration.resolve(
            assetPath: "/models/whisper.aimodel",
            tokenizerPath: "/models/whisper-tokenizer",
            modelLockPath: "/config/resident-model-lock.json",
            maximumQueuedRequests: nil))

        do {
            _ = try await configuration.loadResidentEngine()
            XCTFail("standalone build unexpectedly loaded resident Whisper")
        } catch {
            XCTAssertEqual(
                error as? WhisperStartupConfiguration.ConfigurationError,
                .runtimeUnavailable)
            XCTAssertEqual(
                String(describing: error),
                "resident Whisper requires a Core AI runtime-linked caix build; rebuild with COREAI_DIRECT_RUNTIME=1")
        }
    }
    #endif
}
