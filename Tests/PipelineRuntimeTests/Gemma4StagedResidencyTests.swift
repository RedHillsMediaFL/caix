import XCTest

@testable import PipelineRuntime

final class Gemma4StagedResidencyTests: XCTestCase {
    private let gib = UInt64(1_073_741_824)

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

    func testMemoryAdmissionSelectsInitialThenFallbackWithoutAllocation() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: validRuntimeMemory).runtimeMemory)
        let initial = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: { self.safeSnapshot(total: 64 * self.gib) })
        let fallback = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: { self.safeSnapshot(total: 51_000_000_000) })

        XCTAssertEqual(try initial.selectContext().contextTokens, 65_536)
        XCTAssertEqual(try initial.selectContext().tier, .initial)
        XCTAssertEqual(try fallback.selectContext().contextTokens, 32_768)
        XCTAssertEqual(try fallback.selectContext().tier, .fallback)
    }

    func testMemoryAdmissionFailsClosedOnCurrentPressure() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: validRuntimeMemory).runtimeMemory)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                DistributedStagedMemorySnapshot(
                    totalPhysicalMemoryBytes: 64 * self.gib,
                    workerResidentBytes: 20 * self.gib,
                    availableBytes: 20 * self.gib,
                    pressure: .yellow,
                    swapGrowthBytes: 0)
            })

        XCTAssertThrowsError(try admission.selectContext()) { error in
            XCTAssertEqual(
                error as? DistributedStagedMemoryAdmissionError,
                .drain(.memoryPressure))
        }
    }

    func testPipelineLoadsAndUnloadsExactlyOnePrefillStageAtATime() async throws {
        let fixture = try makePipelineFixture()
        try await fixture.pipeline.allocate(requestID: "prefill", kvCapacity: 16)

        _ = try await fixture.pipeline.forward(
            requestID: "prefill",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
            tokenIDs: [11, 12])

        XCTAssertEqual(fixture.recorder.maximumActivePrefillCount, 1)
        XCTAssertEqual(fixture.recorder.activePrefillCount, 0)
        XCTAssertEqual(fixture.recorder.events, [
            "admit", "load:embed", "forward:embed", "unload:embed",
            "admit", "load:layers", "forward:layers", "unload:layers",
            "admit", "load:head", "forward:head", "unload:head",
        ])
        XCTAssertTrue(fixture.handles.allSatisfy { !$0.isPrefillResident })
    }

    func testPipelineDecodeUsesResidentSetWithoutLoadingPrefill() async throws {
        let fixture = try makePipelineFixture()
        try await fixture.pipeline.allocate(requestID: "decode", kvCapacity: 16)

        _ = try await fixture.pipeline.forward(
            requestID: "decode",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            tokenIDs: [11])

        XCTAssertEqual(fixture.recorder.events, [
            "forward:embed", "forward:layers", "forward:head",
        ])
        XCTAssertEqual(fixture.recorder.maximumActivePrefillCount, 0)
    }

    func testPipelineUnloadsPrefillWhenStageForwardThrows() async throws {
        let fixture = try makePipelineFixture(failingStageID: "layers")
        try await fixture.pipeline.allocate(requestID: "failure", kvCapacity: 16)

        do {
            _ = try await fixture.pipeline.forward(
                requestID: "failure",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
                tokenIDs: [11, 12])
            XCTFail("expected synthetic failure")
        } catch SyntheticError.forwardFailure {
            XCTAssertEqual(fixture.recorder.activePrefillCount, 0)
            XCTAssertTrue(fixture.handles.allSatisfy { !$0.isPrefillResident })
            XCTAssertEqual(Array(fixture.recorder.events.suffix(3)), [
                "load:layers", "forward:layers", "unload:layers",
            ])
        }
    }

    func testPipelineRejectsPressureBeforePrefillLoad() async throws {
        let fixture = try makePipelineFixture(pressure: .yellow)
        try await fixture.pipeline.allocate(requestID: "pressure", kvCapacity: 16)

        await XCTAssertThrowsErrorAsync(
            try await fixture.pipeline.forward(
                requestID: "pressure",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
                tokenIDs: [11, 12])) { error in
                    XCTAssertEqual(
                        error as? DistributedStagedMemoryAdmissionError,
                        .drain(.memoryPressure))
                }
        XCTAssertEqual(fixture.recorder.events, ["admit"])
        XCTAssertEqual(fixture.recorder.activePrefillCount, 0)
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

    private func safeSnapshot(total: UInt64) -> DistributedStagedMemorySnapshot {
        DistributedStagedMemorySnapshot(
            totalPhysicalMemoryBytes: total,
            workerResidentBytes: 20 * gib,
            availableBytes: 20 * gib,
            pressure: .green,
            swapGrowthBytes: 0)
    }

    private func makePipelineFixture(
        pressure: ResidentMemoryPressure = .green,
        failingStageID: String? = nil
    ) throws -> PipelineFixture {
        let manifest = try decodeManifest(runtimeMemory: validRuntimeMemory)
        let recorder = ResidencyRecorder()
        let handles = manifest.runtimePlan.stages.enumerated().map { index, descriptor in
            RecordingDecodeResidentStage(
                descriptor: descriptor,
                nextStageID: manifest.runtimePlan.stages.indices.contains(index + 1)
                    ? manifest.runtimePlan.stages[index + 1].id : nil,
                recorder: recorder,
                failForward: descriptor.id == failingStageID)
        }
        let contract = try XCTUnwrap(manifest.runtimeMemory)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                recorder.events.append("admit")
                return DistributedStagedMemorySnapshot(
                    totalPhysicalMemoryBytes: 64 * self.gib,
                    workerResidentBytes: 20 * self.gib,
                    availableBytes: 20 * self.gib,
                    pressure: pressure,
                    swapGrowthBytes: 0)
            })
        let pipeline = try DistributedSameMachinePipeline(
            plan: manifest.runtimePlan,
            stages: handles,
            streamedPrefillAdmission: admission)
        return PipelineFixture(
            pipeline: pipeline,
            handles: handles,
            recorder: recorder)
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

private struct PipelineFixture {
    let pipeline: DistributedSameMachinePipeline
    let handles: [RecordingDecodeResidentStage]
    let recorder: ResidencyRecorder
}

private enum SyntheticError: Error {
    case forwardFailure
}

private final class ResidencyRecorder: @unchecked Sendable {
    var events: [String] = []
    var activePrefillCount = 0
    var maximumActivePrefillCount = 0
}

private final class RecordingDecodeResidentStage:
    DistributedDecodeResidentStageHandle, @unchecked Sendable
{
    let descriptor: DistributedStageDescriptor
    var acceptsTokenIDs: Bool { descriptor.role == .embeddings }
    private(set) var isPrefillResident = false

    private let nextStageID: String?
    private let recorder: ResidencyRecorder
    private let failForward: Bool

    init(
        descriptor: DistributedStageDescriptor,
        nextStageID: String?,
        recorder: ResidencyRecorder,
        failForward: Bool
    ) {
        self.descriptor = descriptor
        self.nextStageID = nextStageID
        self.recorder = recorder
        self.failForward = failForward
    }

    func loadPrefill() async throws {
        XCTAssertFalse(isPrefillResident)
        isPrefillResident = true
        recorder.activePrefillCount += 1
        recorder.maximumActivePrefillCount = max(
            recorder.maximumActivePrefillCount,
            recorder.activePrefillCount)
        recorder.events.append("load:\(descriptor.id)")
    }

    func unloadPrefill() {
        guard isPrefillResident else { return }
        recorder.events.append("unload:\(descriptor.id)")
        recorder.activePrefillCount -= 1
        isPrefillResident = false
    }

    func allocate(_ allocation: DistributedStageAllocation) async throws {}

    func forward(
        _ input: DistributedStageForwardInput
    ) async throws -> DistributedStageForwardOutput {
        recorder.events.append("forward:\(descriptor.id)")
        if failForward { throw SyntheticError.forwardFailure }
        if descriptor.role == .finalNormHead {
            return DistributedStageForwardOutput(
                stageID: descriptor.id,
                stepIndex: input.stepIndex,
                tokenID: 42)
        }
        let shape = [1, input.positionRange.count, 5_376]
        let byteCount = shape.reduce(DistributedTensorScalarType.float16.byteWidth, *)
        let packet = try DistributedHiddenStatePacket(
            metadata: DistributedHiddenStatePacketMetadata(
                requestID: input.requestID,
                sourceStageID: descriptor.id,
                destinationStageID: nextStageID!,
                positionRange: input.positionRange,
                shape: shape,
                scalarType: .float16,
                byteCount: byteCount,
                stepIndex: input.stepIndex),
            payload: Array(repeating: 0, count: byteCount))
        return DistributedStageForwardOutput(
            stageID: descriptor.id,
            stepIndex: input.stepIndex,
            hiddenState: packet)
    }

    func reset(requestID: String) async throws {}
    func free(requestID: String) async {}
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ errorHandler: (Error) -> Void
) async {
    do {
        _ = try await expression()
        XCTFail("expected expression to throw")
    } catch {
        errorHandler(error)
    }
}
