import Foundation
import XCTest

@testable import PipelineRuntime

final class StagedMTPStartupConfigurationTests: XCTestCase {
    func testNoMTPOptionsLeaveStartupUnchanged() throws {
        XCTAssertNil(try StagedMTPStartupConfiguration.resolve(
            assistantPath: nil,
            draftTokens: nil,
            requireMTP: false,
            primaryBundleURL: nil,
            clusterMode: false,
            prewarm: "smallest"))
    }

    func testRequireMTPNeedsLocalStagedPrimary() throws {
        let assistant = try makeAssistantAsset()

        XCTAssertThrowsError(try StagedMTPStartupConfiguration.resolve(
            assistantPath: assistant.path,
            draftTokens: 4,
            requireMTP: true,
            primaryBundleURL: nil,
            clusterMode: false,
            prewarm: "smallest"))
    }

    func testRequireMTPRejectsClusterAndDisabledPrewarm() throws {
        let assistant = try makeAssistantAsset()
        let primary = try makeEaglePrimaryBundle()

        XCTAssertThrowsError(try StagedMTPStartupConfiguration.resolve(
            assistantPath: assistant.path,
            draftTokens: 4,
            requireMTP: true,
            primaryBundleURL: primary,
            clusterMode: true,
            prewarm: "smallest"))
        XCTAssertThrowsError(try StagedMTPStartupConfiguration.resolve(
            assistantPath: assistant.path,
            draftTokens: 4,
            requireMTP: true,
            primaryBundleURL: primary,
            clusterMode: false,
            prewarm: "off"))
    }

    func testRequireMTPRejectsEveryPrewarmOffSynonym() throws {
        let assistant = try makeAssistantAsset()
        let primary = try makeEaglePrimaryBundle()

        for prewarm in ["off", "none", "false", "no"] {
            XCTAssertThrowsError(try StagedMTPStartupConfiguration.resolve(
                assistantPath: assistant.path,
                draftTokens: 4,
                requireMTP: true,
                primaryBundleURL: primary,
                clusterMode: false,
                prewarm: prewarm), "expected \(prewarm) to disable required MTP startup")
        }
    }

    func testMTPRejectsWrongAssistantExtensionAndNonEaglePrimary() throws {
        let primary = try makeEaglePrimaryBundle()
        let wrongExtension = FileManager.default.temporaryDirectory
            .appendingPathComponent("assistant.mlpackage")

        XCTAssertThrowsError(try StagedMTPStartupConfiguration.resolve(
            assistantPath: wrongExtension.path,
            draftTokens: 4,
            requireMTP: false,
            primaryBundleURL: primary,
            clusterMode: false,
            prewarm: "smallest"))

        let assistant = try makeAssistantAsset()
        let nonEagle = try makeStagedPrimaryBundle(eagleTarget: false)
        XCTAssertThrowsError(try StagedMTPStartupConfiguration.resolve(
            assistantPath: assistant.path,
            draftTokens: 4,
            requireMTP: true,
            primaryBundleURL: nonEagle,
            clusterMode: false,
            prewarm: "smallest"))
    }

    func testMTPDraftTokensMustBeBoundedPositiveInteger() throws {
        let assistant = try makeAssistantAsset()
        let primary = try makeEaglePrimaryBundle()

        XCTAssertThrowsError(try StagedMTPStartupConfiguration.resolve(
            assistantPath: assistant.path,
            draftTokens: 0,
            requireMTP: false,
            primaryBundleURL: primary,
            clusterMode: false,
            prewarm: "smallest"))
        XCTAssertThrowsError(try StagedMTPStartupConfiguration.resolve(
            assistantPath: assistant.path,
            draftTokens: 9,
            requireMTP: false,
            primaryBundleURL: primary,
            clusterMode: false,
            prewarm: "smallest"))
    }

    func testRequireMTPProofNeedsPositiveDraftedTokensAndSequentialNonFastMode() async throws {
        let assistant = try makeAssistantAsset()
        let primary = try makeEaglePrimaryBundle()
        XCTAssertNotNil(try DistributedStageManifest.load(
            from: primary.appendingPathComponent("stage-manifest.json")).eagleTarget)
        let configuration = try XCTUnwrap(try StagedMTPStartupConfiguration.resolve(
            assistantPath: assistant.path,
            draftTokens: 4,
            requireMTP: true,
            primaryBundleURL: primary,
            clusterMode: false,
            prewarm: "smallest"))

        do {
            try await configuration.requireProof { _ in
                StagedMTPStartupProof(
                    draftedTokens: 0,
                    executionMode: .sequentialNoRollback,
                    fast: false)
            }
            XCTFail("expected zero drafted tokens to fail proof")
        } catch {}
        do {
            try await configuration.requireProof { _ in
                StagedMTPStartupProof(
                    draftedTokens: 1,
                    executionMode: .sequentialNoRollback,
                    fast: true)
            }
            XCTFail("expected fast MTP proof to fail")
        } catch {}
        try await configuration.requireProof { _ in
            StagedMTPStartupProof(
                draftedTokens: 1,
                executionMode: .sequentialNoRollback,
                fast: false)
        }
    }

    private func makeAssistantAsset() throws -> URL {
        let root = try temporaryDirectory()
        let asset = root.appendingPathComponent("assistant.aimodel", isDirectory: true)
        try FileManager.default.createDirectory(at: asset, withIntermediateDirectories: true)
        return asset
    }

    private func makeEaglePrimaryBundle() throws -> URL {
        try makeStagedPrimaryBundle(eagleTarget: true)
    }

    private func makeStagedPrimaryBundle(eagleTarget: Bool) throws -> URL {
        let root = try temporaryDirectory()
        let eagle = eagleTarget ? #", "eagle_target":{"stage_id":"layers","sliding_window":1024}"# : ""
        let manifest = """
            {
              "schema":"caix.cluster.stage_manifest.v0",
              "model":"google/gemma-4-31B-it",
              "total_layer_count":1,
              "boundary":{"hidden_state":{"name":"hidden_states","shape":[1,-1,2],"scalar_type":"float16"}},
              "stages":[
                {"id":"embed","role":"embeddings","layers":"embeddings","bundle":"stages/embed.aimodel","memory_gb":1},
                {"id":"layers","role":"transformer_layers","layers":[0,1],"bundle":"stages/layers.aimodel","memory_gb":1},
                {"id":"head","role":"final_norm_head","layers":"norm+lm_head","bundle":"stages/head.aimodel","memory_gb":1}
              ]\(eagle)
            }
            """
        try manifest.write(
            to: root.appendingPathComponent("stage-manifest.json"),
            atomically: true,
            encoding: .utf8)
        return root
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-staged-mtp-startup-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
