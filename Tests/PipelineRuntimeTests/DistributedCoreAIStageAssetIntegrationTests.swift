import XCTest
@testable import PipelineRuntime

final class DistributedCoreAIStageAssetIntegrationTests: XCTestCase {
    func testRealStageAssetsLoadAndValidateContracts() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_STAGE_MANIFEST to a real staged manifest")
        }

        #if COREAI_RUNTIME
        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        XCTAssertEqual(manifest.positionMode, .fullPrefix)
        XCTAssertEqual(manifest.stages.count, manifest.runtimePlan.stages.count)

        let factory = DistributedCoreAIStageHandleFactory()
        for stage in manifest.stages {
            let descriptor = try XCTUnwrap(manifest.runtimePlan.stage(id: stage.id))
            let context = DistributedStageHandleFactoryContext(
                stage: stage,
                manifest: manifest,
                descriptor: descriptor)
            let handle = try await factory.makeStageHandle(for: context)
            let coreAIHandle = try XCTUnwrap(handle as? DistributedCoreAIStageHandle)

            XCTAssertEqual(coreAIHandle.descriptor.id, stage.id)
            XCTAssertEqual(coreAIHandle.assetURL.path, try context.requireExistingAssetURL().path)
            XCTAssertEqual(coreAIHandle.functionName, context.mainFunctionName)
            XCTAssertFalse(coreAIHandle.ioContract.inputs.isEmpty)
            XCTAssertFalse(coreAIHandle.ioContract.outputs.isEmpty)
            XCTAssertNoThrow(
                try context.validateStageIOContract(
                    coreAIHandle.ioContract,
                    vocabSize: context.vocabSize))
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }
}
