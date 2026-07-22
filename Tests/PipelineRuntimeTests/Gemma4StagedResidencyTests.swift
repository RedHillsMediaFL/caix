import XCTest

@testable import PipelineRuntime

final class Gemma4StagedResidencyTests: XCTestCase {
    func testStageManifestLoadsStreamedPrefillRuntimeMemoryContract() throws {
        let manifest = try decodeManifest(runtimeMemory: validRuntimeMemory)

        let memory = try XCTUnwrap(manifest.runtimeMemory)
        XCTAssertEqual(
            memory.assetResidencyPolicy,
            .decodeSetPlusOneStreamedPrefillStage)
        XCTAssertEqual(memory.allocationPolicy, .preflightInitialThenFallback)
        XCTAssertTrue(memory.fullPrefillAndDecodeSetsMayNotBeDualResident)
        XCTAssertEqual(memory.initial.contextTokens, 65_536)
        XCTAssertEqual(memory.initial.cacheBytes, 7_885_291_520)
        XCTAssertEqual(memory.initial.minimumPhysicalMemoryBytes, 53_000_000_000)
        XCTAssertEqual(memory.fallback.contextTokens, 32_768)
        XCTAssertEqual(memory.fallback.cacheBytes, 5_200_936_960)
        XCTAssertEqual(memory.fallback.minimumPhysicalMemoryBytes, 50_000_000_000)
        XCTAssertTrue(manifest.requiresStreamedPrefillResidency)
    }

    func testStreamedPrefillManifestRejectsDualResidencyPermission() {
        let runtimeMemory = validRuntimeMemory.replacingOccurrences(
            of: #""full_prefill_and_decode_sets_may_not_be_dual_resident": true"#,
            with: #""full_prefill_and_decode_sets_may_not_be_dual_resident": false"#)

        XCTAssertThrowsError(try decodeManifest(runtimeMemory: runtimeMemory)) { error in
            guard case .invalidManifest(let message) = error as? DistributedStageManifestError
            else { return XCTFail("expected invalidManifest, got \(error)") }
            XCTAssertTrue(message.contains("may_not_be_dual_resident"))
        }
    }

    func testStreamedPrefillManifestRejectsMissingDecodeAsset() {
        let stages = validStages.replacingOccurrences(
            of: #", "decode_asset":"stages/01-layers-decode.aimodel""#,
            with: "")

        XCTAssertThrowsError(
            try decodeManifest(runtimeMemory: validRuntimeMemory, stages: stages)
        ) { error in
            guard case .invalidManifest(let message) = error as? DistributedStageManifestError
            else { return XCTFail("expected invalidManifest, got \(error)") }
            XCTAssertTrue(message.contains("decode_asset"))
            XCTAssertTrue(message.contains("layers"))
        }
    }

    func testStreamedPrefillManifestRejectsNonDescendingFallbackTier() {
        let runtimeMemory = validRuntimeMemory.replacingOccurrences(
            of: #""context_tokens": 32768"#,
            with: #""context_tokens": 65536"#)

        XCTAssertThrowsError(try decodeManifest(runtimeMemory: runtimeMemory)) { error in
            guard case .invalidManifest(let message) = error as? DistributedStageManifestError
            else { return XCTFail("expected invalidManifest, got \(error)") }
            XCTAssertTrue(message.contains("fallback"))
        }
    }

    private func decodeManifest(
        runtimeMemory: String,
        stages: String? = nil
    ) throws -> DistributedStageManifest {
        let json =
            """
            {
              "schema": "\(DistributedStageManifest.currentSchema)",
              "model": "gemma4-31b-it-q4-0-target",
              "total_layer_count": 60,
              "position_mode": "full_prefix",
              "boundary": {
                "hidden_state": {
                  "name": "hidden_states",
                  "shape": [1, -1, 5376],
                  "scalar_type": "float16"
                }
              },
              "runtime_memory": \(runtimeMemory),
              "stages": \(stages ?? validStages)
            }
            """
        return try DistributedStageManifest.decode(
            from: Data(json.utf8),
            baseURL: URL(fileURLWithPath: "/tmp/gemma4-31b", isDirectory: true))
    }

    private var validRuntimeMemory: String {
        """
        {
          "allocation_policy": "preflight_initial_then_fallback",
          "asset_residency_policy": "decode_set_plus_one_streamed_prefill_stage",
          "full_prefill_and_decode_sets_may_not_be_dual_resident": true,
          "resident_weight_upper_bound_bytes": 25769803776,
          "runtime_overhead_reserve_bytes": 4294967296,
          "coresident_services_reserve_bytes": 6442450944,
          "system_headroom_bytes": 8589934592,
          "initial": {
            "context_tokens": 65536,
            "cache_bytes": 7885291520,
            "minimum_physical_memory_bytes": 53000000000
          },
          "fallback": {
            "context_tokens": 32768,
            "cache_bytes": 5200936960,
            "minimum_physical_memory_bytes": 50000000000
          }
        }
        """
    }

    private var validStages: String {
        """
        [
          {"id":"embed","role":"embeddings","layers":"embeddings","bundle":"stages/00-embed.aimodel", "decode_asset":"stages/00-embed-decode.aimodel","function_map":{"main":["main"],"decode":["decode"]},"vocab_size":262144,"memory_gb":1.0},
          {"id":"layers","role":"transformer_layers","layers":[0,60],"bundle":"stages/01-layers.aimodel", "decode_asset":"stages/01-layers-decode.aimodel","function_map":{"main":["main"],"decode":["decode"]},"memory_gb":6.0},
          {"id":"head","role":"final_norm_head","layers":"norm+lm_head","bundle":"stages/02-head.aimodel", "decode_asset":"stages/02-head-decode.aimodel","function_map":{"main":["main"],"decode":["decode"]},"vocab_size":262144,"memory_gb":1.0}
        ]
        """
    }
}
