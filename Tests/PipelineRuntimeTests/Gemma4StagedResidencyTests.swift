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
            snapshotProvider: {
                self.safeSnapshot(total: 64 * self.gib, available: 40 * self.gib)
            })
        let fallback = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                self.safeSnapshot(total: 51_000_000_000, available: 40 * self.gib)
            })

        XCTAssertEqual(try initial.selectContext().contextTokens, 65_536)
        XCTAssertEqual(try initial.selectContext().tier, .initial)
        XCTAssertEqual(try fallback.selectContext().contextTokens, 32_768)
        XCTAssertEqual(try fallback.selectContext().tier, .fallback)
    }

    func testMemoryAdmissionFallsBackWhenInitialExceedsIncrementalWorkerFootprint() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: validRuntimeMemory).runtimeMemory)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                self.safeSnapshot(total: 64 * self.gib, available: 34 * self.gib)
            })

        XCTAssertEqual(try admission.selectContext().tier, .fallback)
        XCTAssertEqual(try admission.selectContext().contextTokens, 32_768)
    }

    func testMemoryAdmissionReservesPendingAssistantBeforeItBecomesResident() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: validRuntimeMemory).runtimeMemory)
        let available = 37 * gib
        let withoutAssistant = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                self.safeSnapshot(total: 64 * self.gib, available: available)
            })
        let withAssistant = DistributedStagedMemoryAdmission(
            contract: contract,
            pendingResidentBytes: 3_757_662_899,
            snapshotProvider: {
                self.safeSnapshot(total: 64 * self.gib, available: available)
            })

        XCTAssertEqual(try withoutAssistant.selectContext().tier, .initial)
        XCTAssertEqual(try withAssistant.selectContext().tier, .fallback)
    }

    func testMemoryAdmissionRejectsBothTiersWhenCurrentAvailabilityIsUnsafe() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: validRuntimeMemory).runtimeMemory)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                self.safeSnapshot(total: 64 * self.gib, available: 32 * self.gib)
            })

        XCTAssertThrowsError(try admission.selectContext()) { error in
            guard case .insufficientAvailableMemory(let required, let actual) =
                error as? DistributedStagedMemoryAdmissionError
            else { return XCTFail("expected insufficientAvailableMemory, got \(error)") }
            XCTAssertGreaterThan(required, actual)
            XCTAssertEqual(actual, 32 * self.gib)
        }
    }

    func testProductionScaleAdmissionSelectsFallbackFromIncrementalWorkerFootprint() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: productionRuntimeMemory).runtimeMemory)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                DistributedStagedMemorySnapshot(
                    totalPhysicalMemoryBytes: 64 * self.gib,
                    workerResidentBytes: 8_979_152,
                    availableBytes: 41_494_396_928,
                    allocationCapacityBytes: 41_494_396_928,
                    pressure: .green,
                    swapGrowthBytes: 0)
            })

        XCTAssertEqual(try admission.selectContext().tier, .fallback)
    }

    func testProductionScaleAdmissionRejectsAvailabilityBelowFallbackIncrement() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: productionRuntimeMemory).runtimeMemory)
        let fallbackIncrement = UInt64(39_551_696_176)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                DistributedStagedMemorySnapshot(
                    totalPhysicalMemoryBytes: 64 * self.gib,
                    workerResidentBytes: 8_979_152,
                    availableBytes: fallbackIncrement - 1,
                    allocationCapacityBytes: fallbackIncrement - 1,
                    pressure: .green,
                    swapGrowthBytes: 0)
            })

        XCTAssertThrowsError(try admission.selectContext()) { error in
            guard case .insufficientAvailableMemory(let required, let actual) =
                error as? DistributedStagedMemoryAdmissionError
            else { return XCTFail("expected insufficientAvailableMemory, got \(error)") }
            XCTAssertEqual(required, fallbackIncrement)
            XCTAssertEqual(actual, fallbackIncrement - 1)
        }
    }

    func testProductionScaleAdmissionUsesAllocationCapacityAndProjectedResidentCap() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: productionRuntimeMemory).runtimeMemory)
        let assistantBytes = UInt64(3_757_662_899)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            pendingResidentBytes: assistantBytes,
            snapshotProvider: {
                DistributedStagedMemorySnapshot(
                    totalPhysicalMemoryBytes: 64 * self.gib,
                    workerResidentBytes: 488_066_264,
                    availableBytes: 29_600_000_000,
                    allocationCapacityBytes: 48 * self.gib,
                    pressure: .green,
                    swapGrowthBytes: 0)
            })

        XCTAssertEqual(try admission.selectContext().tier, .fallback)
    }

    func testProductionScaleAdmissionRejectsInsufficientAllocationCapacity() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: productionRuntimeMemory).runtimeMemory)
        let currentWorkerBytes = UInt64(488_066_264)
        let fallbackProjection = UInt64(39_560_675_328)
        let assistantBytes = UInt64(3_757_662_899)
        let exactRequirement = fallbackProjection - currentWorkerBytes + assistantBytes

        let rejected = DistributedStagedMemoryAdmission(
            contract: contract,
            pendingResidentBytes: assistantBytes,
            snapshotProvider: {
                DistributedStagedMemorySnapshot(
                    totalPhysicalMemoryBytes: 64 * self.gib,
                    workerResidentBytes: currentWorkerBytes,
                    availableBytes: 29_600_000_000,
                    allocationCapacityBytes: exactRequirement - 1,
                    pressure: .green,
                    swapGrowthBytes: 0)
            })
        XCTAssertThrowsError(try rejected.selectContext()) { error in
            guard case .insufficientAvailableMemory(let required, let actual) =
                error as? DistributedStagedMemoryAdmissionError
            else { return XCTFail("expected insufficientAvailableMemory, got \(error)") }
            XCTAssertEqual(required, exactRequirement)
            XCTAssertEqual(actual, exactRequirement - 1)
        }
    }

    func testPerAssetGateStillUsesLiveAvailableHeadroom() throws {
        let contract = try XCTUnwrap(
            decodeManifest(runtimeMemory: productionRuntimeMemory).runtimeMemory)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                DistributedStagedMemorySnapshot(
                    totalPhysicalMemoryBytes: 64 * self.gib,
                    workerResidentBytes: 20 * self.gib,
                    availableBytes: 7 * self.gib,
                    allocationCapacityBytes: 48 * self.gib,
                    pressure: .green,
                    swapGrowthBytes: 0)
            })

        XCTAssertThrowsError(try admission.checkBeforeAssetLoad()) { error in
            XCTAssertEqual(
                error as? DistributedStagedMemoryAdmissionError,
                .drain(.insufficientAvailableMemory))
        }
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
                    allocationCapacityBytes: 40 * self.gib,
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
            "admit", "load:embed", "admit", "forward:embed", "unload:embed",
            "admit", "load:layers", "admit", "forward:layers", "unload:layers",
            "admit", "load:head", "admit", "forward:head", "unload:head",
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
                "admit", "forward:layers", "unload:layers",
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

    func testPipelineUnloadsPrefillWhenPostLoadAdmissionFails() async throws {
        let fixture = try makePipelineFixture(admissionPressures: [.green, .yellow])
        try await fixture.pipeline.allocate(requestID: "post-load-pressure", kvCapacity: 16)

        await XCTAssertThrowsErrorAsync(
            try await fixture.pipeline.forward(
                requestID: "post-load-pressure",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
                tokenIDs: [11, 12])) { error in
                    XCTAssertEqual(
                        error as? DistributedStagedMemoryAdmissionError,
                        .drain(.memoryPressure))
                }
        XCTAssertEqual(fixture.recorder.events, [
            "admit", "load:embed", "admit", "unload:embed",
        ])
        XCTAssertEqual(fixture.recorder.activePrefillCount, 0)
        XCTAssertTrue(fixture.handles.allSatisfy { !$0.isPrefillResident })
    }

    func testMakeChecksBeforeAndAfterEveryDecodeAssetLoad() async throws {
        let manifest = try decodeManifest(runtimeMemory: validRuntimeMemory)
        let recorder = ResidencyRecorder()
        let factory = RecordingDecodeResidentFactory(recorder: recorder)
        let contract = try XCTUnwrap(manifest.runtimeMemory)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                recorder.events.append("admit")
                return self.safeSnapshot(total: 64 * self.gib)
            })

        let pipeline = try await DistributedSameMachinePipeline.make(
            manifest: manifest,
            handleFactory: factory,
            streamedPrefillAdmission: admission)

        withExtendedLifetime(pipeline) {
            XCTAssertEqual(recorder.events, [
                "admit", "factory:embed", "admit",
                "admit", "factory:layers", "admit",
                "admit", "factory:head", "admit",
            ])
        }
    }

    func testMakeReleasesResidentHandlesWhenPostLoadAdmissionFails() async throws {
        let manifest = try decodeManifest(runtimeMemory: validRuntimeMemory)
        let recorder = ResidencyRecorder(admissionPressures: [.green, .yellow])
        let factory = RecordingDecodeResidentFactory(recorder: recorder)
        let contract = try XCTUnwrap(manifest.runtimeMemory)
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: {
                recorder.events.append("admit")
                return self.safeSnapshot(
                    total: 64 * self.gib,
                    allocationCapacity: 48 * self.gib,
                    pressure: recorder.nextAdmissionPressure())
            })

        await XCTAssertThrowsErrorAsync(
            try await DistributedSameMachinePipeline.make(
                manifest: manifest,
                handleFactory: factory,
                streamedPrefillAdmission: admission)) { error in
                    XCTAssertEqual(
                        error as? DistributedStagedMemoryAdmissionError,
                        .drain(.memoryPressure))
                }
        XCTAssertEqual(recorder.events, [
            "admit", "factory:embed", "admit", "release:embed",
        ])
    }

    func testMakeRejectsStreamedManifestWithoutAdmission() async throws {
        let manifest = try decodeManifest(runtimeMemory: validRuntimeMemory)
        let recorder = ResidencyRecorder()

        await XCTAssertThrowsErrorAsync(
            try await DistributedSameMachinePipeline.make(
                manifest: manifest,
                handleFactory: RecordingDecodeResidentFactory(recorder: recorder))) { error in
                    guard case .invalidControlFrame(let message) =
                        error as? DistributedStageExecutionError
                    else { return XCTFail("expected invalidControlFrame, got \(error)") }
                    XCTAssertTrue(message.contains("memory admission"))
                }
        XCTAssertTrue(recorder.events.isEmpty)
    }

    #if COREAI_RUNTIME
    func testCoreAIDecodeResidentFactoryBindsTheStreamedStageProtocol() {
        let factory: any DistributedStageHandleFactory =
            DistributedCoreAIDecodeResidentStageHandleFactory()
        XCTAssertTrue(
            String(describing: type(of: factory))
                .contains("DistributedCoreAIDecodeResidentStageHandleFactory"))
    }

    func testTextStagedResidentLoadConfigurationSelectsFallbackBeforeAllocation() throws {
        let manifest = try decodeManifest(runtimeMemory: validRuntimeMemory)
        var snapshotCount = 0

        let configuration = try TextStagedModel.residentLoadConfiguration(
            manifest: manifest,
            metadataMaxContextLength: 262_144,
            snapshotProvider: {
                snapshotCount += 1
                return self.safeSnapshot(
                    total: 51_000_000_000,
                    available: 40 * self.gib)
            })

        XCTAssertEqual(snapshotCount, 1)
        XCTAssertEqual(configuration.contextSelection?.tier, .fallback)
        XCTAssertEqual(configuration.maxContextLength, 32_768)
        XCTAssertTrue(configuration.requiresDecodeResidentFactory)
        XCTAssertNotNil(configuration.streamedPrefillAdmission)
    }

    func testTextStagedResidentLoadConfigurationReservesPendingAssistantBytes() throws {
        let manifest = try decodeManifest(runtimeMemory: validRuntimeMemory)

        let configuration = try TextStagedModel.residentLoadConfiguration(
            manifest: manifest,
            metadataMaxContextLength: 262_144,
            snapshotProvider: {
                self.safeSnapshot(
                    total: 64 * self.gib,
                    available: 17 * self.gib,
                    allocationCapacity: 37 * self.gib)
            },
            pendingResidentBytes: 3_757_662_899)

        XCTAssertEqual(configuration.contextSelection?.tier, .fallback)
        XCTAssertEqual(configuration.maxContextLength, 32_768)
    }

    func testTextStagedGenerationValidationRejectsExplicitSampling() {
        let options = CoreAIPipeline.Options(maxTokens: 8, temperature: 0.7)

        XCTAssertThrowsError(
            try TextStagedModel.validateGenerationOptions(options)
        ) { error in
            XCTAssertTrue(String(describing: error).contains("greedy decoding only"))
        }
    }
    #endif

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

    private func safeSnapshot(
        total: UInt64,
        available: UInt64? = nil,
        allocationCapacity: UInt64? = nil,
        pressure: ResidentMemoryPressure = .green,
        workerResident: UInt64? = nil
    ) -> DistributedStagedMemorySnapshot {
        DistributedStagedMemorySnapshot(
            totalPhysicalMemoryBytes: total,
            workerResidentBytes: workerResident ?? 128 * 1_048_576,
            availableBytes: available ?? 40 * gib,
            allocationCapacityBytes: allocationCapacity ?? available ?? 40 * gib,
            pressure: pressure,
            swapGrowthBytes: 0)
    }

    private func makePipelineFixture(
        pressure: ResidentMemoryPressure = .green,
        admissionPressures: [ResidentMemoryPressure] = [],
        failingStageID: String? = nil
    ) throws -> PipelineFixture {
        let manifest = try decodeManifest(runtimeMemory: validRuntimeMemory)
        let recorder = ResidencyRecorder(admissionPressures: admissionPressures)
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
                    allocationCapacityBytes: 40 * self.gib,
                    pressure: recorder.nextAdmissionPressure(default: pressure),
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

    private var productionRuntimeMemory: String {
        """
        {
          "allocation_policy": "preflight_initial_then_fallback",
          "asset_residency_policy": "decode_set_plus_one_streamed_prefill_stage",
          "full_prefill_and_decode_sets_may_not_be_dual_resident": true,
          "resident_weight_upper_bound_bytes": 25769803776,
          "runtime_overhead_reserve_bytes": 8589934592,
          "coresident_services_reserve_bytes": 6442450944,
          "system_headroom_bytes": 12884901888,
          "initial": {
            "context_tokens": 65536,
            "cache_bytes": 7885291520,
            "minimum_physical_memory_bytes": 61572382720
          },
          "fallback": {
            "context_tokens": 32768,
            "cache_bytes": 5200936960,
            "minimum_physical_memory_bytes": 58888028160
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
    private var admissionPressures: [ResidentMemoryPressure]

    init(admissionPressures: [ResidentMemoryPressure] = []) {
        self.admissionPressures = admissionPressures
    }

    func nextAdmissionPressure(
        default fallback: ResidentMemoryPressure = .green
    ) -> ResidentMemoryPressure {
        guard !admissionPressures.isEmpty else { return fallback }
        return admissionPressures.removeFirst()
    }
}

private final class RecordingDecodeResidentFactory: DistributedStageHandleFactory {
    private let recorder: ResidencyRecorder

    init(recorder: ResidencyRecorder) {
        self.recorder = recorder
    }

    func makeStageHandle(
        for context: DistributedStageHandleFactoryContext
    ) async throws -> DistributedStageHandle {
        recorder.events.append("factory:\(context.descriptor.id)")
        return RecordingDecodeResidentStage(
            descriptor: context.descriptor,
            nextStageID: context.nextStage?.id,
            recorder: recorder,
            failForward: false)
    }
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

    deinit {
        recorder.events.append("release:\(descriptor.id)")
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
