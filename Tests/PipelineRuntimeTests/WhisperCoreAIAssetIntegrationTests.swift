#if COREAI_RUNTIME

import Foundation
import XCTest

@testable import PipelineRuntime

final class WhisperCoreAIAssetIntegrationTests: XCTestCase {
    func testRealV2AssetEncodesLoadsAndDecodesOneFiniteStep() async throws {
        guard let rawAsset = ProcessInfo.processInfo.environment["CAIX_WHISPER_FULL_ASSET"],
            !rawAsset.isEmpty
        else {
            throw XCTSkip("set CAIX_WHISPER_FULL_ASSET only after the controller releases the full-model hold")
        }
        let assetURL = URL(fileURLWithPath: rawAsset)
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            XCTFail("CAIX_WHISPER_FULL_ASSET does not exist: \(assetURL.path)")
            return
        }

        let factory = try await WhisperCoreAIModelFactory.specialize(assetURL: assetURL)
        let session = try await factory.makeSession()
        do {
            try await session.encode(
                inputFeatures: [Float16](
                    repeating: 0,
                    count: WhisperResidentEngine.featureCount))
            let loadStatus = try await session.loadCrossKV()
            let step = try await session.step(tokenID: 50_258)

            XCTAssertEqual(loadStatus, 1)
            XCTAssertEqual(step.status, 1)
            XCTAssertEqual(step.logits.count, WhisperDecodingPolicy.vocabularySize)
            XCTAssertTrue(step.logits.allSatisfy(\.isFinite))
            await session.finish()
        } catch {
            await session.finish()
            throw error
        }
    }
}

#endif
