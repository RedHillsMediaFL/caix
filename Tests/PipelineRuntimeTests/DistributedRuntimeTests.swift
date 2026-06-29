import XCTest

@testable import PipelineRuntime

final class DistributedRuntimeTests: XCTestCase {
    func testValidStagePlanCoversLayersAndWorkers() throws {
        let plan = makePlan()

        XCTAssertNoThrow(try plan.validate())
        XCTAssertEqual(plan.nextStage(after: "layers-0-16")?.id, "layers-16-32")
    }

    func testStagePlanRejectsLayerGap() {
        let plan = makePlan(stages: [
            stage("embed", .embeddings, assetName: "embeddings", workerID: "coordinator"),
            stage(
                "layers-0-8", .transformerLayers,
                range: DistributedLayerRange(lowerBound: 0, upperBound: 8),
                assetName: "layers_0_8", workerID: "worker-a"),
            stage(
                "layers-12-32", .transformerLayers,
                range: DistributedLayerRange(lowerBound: 12, upperBound: 32),
                assetName: "layers_12_32", workerID: "worker-b"),
            stage("final", .finalNormHead, assetName: "final", workerID: "coordinator"),
        ])

        XCTAssertThrowsError(try plan.validate()) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .layerCoverageGap(
                    expectedStart: 8,
                    stageID: "layers-12-32",
                    actual: DistributedLayerRange(lowerBound: 12, upperBound: 32)))
        }
    }

    func testStagePlanRejectsUnknownWorker() {
        let plan = makePlan(stages: [
            stage("embed", .embeddings, assetName: "embeddings", workerID: "coordinator"),
            stage(
                "layers-0-32", .transformerLayers,
                range: DistributedLayerRange(lowerBound: 0, upperBound: 32),
                assetName: "layers_0_32", workerID: "missing"),
            stage("final", .finalNormHead, assetName: "final", workerID: "coordinator"),
        ])

        XCTAssertThrowsError(try plan.validate()) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .unknownWorkerID(stageID: "layers-0-32", workerID: "missing"))
        }
    }

    func testStagePlanRejectsInvalidBoundaryTensor() {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, 0, 4096], scalarType: .float16))

        XCTAssertThrowsError(try plan.validate()) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .invalidBoundaryTensor(
                    "boundary hidden_state sequence dimension must be positive or -1"))
        }
    }

    func testWorkerEndpointRejectsInvalidPort() {
        let endpoint = DistributedWorkerEndpoint(id: "worker-a", host: "127.0.0.1", port: 0)

        XCTAssertThrowsError(try DistributedRuntimeValidation.validate(endpoint: endpoint)) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .invalidEndpoint(id: "worker-a", reason: "port must be 1...65535"))
        }
    }

    func testHiddenStatePacketValidatesShapeByteCountAndRoute() throws {
        let packet = DistributedHiddenStatePacketMetadata(
            requestID: "req-1",
            sourceStageID: "layers-0-16",
            destinationStageID: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
            shape: [1, 3, 4096],
            scalarType: .float16,
            byteCount: 1 * 3 * 4096 * 2,
            stepIndex: 0)

        XCTAssertEqual(packet.tokenCount, 3)
        XCTAssertEqual(packet.expectedByteCount, 24_576)
        XCTAssertNoThrow(try makePlan().validate(hiddenStatePacket: packet))
    }

    func testHiddenStatePacketRejectsMismatchedByteCount() {
        let packet = DistributedHiddenStatePacketMetadata(
            requestID: "req-1",
            sourceStageID: "layers-0-16",
            destinationStageID: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            shape: [1, 1, 4096],
            scalarType: .float32,
            byteCount: 4096,
            stepIndex: 0)

        XCTAssertThrowsError(try DistributedRuntimeValidation.validate(packet: packet)) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .invalidPacket("byte_count does not match shape and scalar_type"))
        }
    }

    func testHiddenStatePacketRejectsNonAdjacentRoute() {
        let packet = DistributedHiddenStatePacketMetadata(
            requestID: "req-1",
            sourceStageID: "embed",
            destinationStageID: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            shape: [1, 1, 4096],
            scalarType: .float16,
            byteCount: 8192,
            stepIndex: 0)

        XCTAssertThrowsError(try makePlan().validate(hiddenStatePacket: packet)) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .packetRouteMismatch(sourceStageID: "embed", destinationStageID: "layers-16-32"))
        }
    }

    func testHiddenStatePacketRejectsBoundaryTensorScalarTypeMismatch() {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let packet = DistributedHiddenStatePacketMetadata(
            requestID: "req-1",
            sourceStageID: "layers-0-16",
            destinationStageID: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            shape: [1, 1, 2],
            scalarType: .float32,
            byteCount: 8,
            stepIndex: 0)

        XCTAssertThrowsError(try plan.validate(hiddenStatePacket: packet)) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .invalidPacket(
                    "hidden-state packet scalar_type float32 does not match boundary tensor float16"))
        }
    }

    func testHiddenStatePacketRejectsBoundaryTensorMismatch() {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let packet = DistributedHiddenStatePacketMetadata(
            requestID: "req-1",
            sourceStageID: "layers-0-16",
            destinationStageID: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            shape: [1, 1, 4],
            scalarType: .float16,
            byteCount: 8,
            stepIndex: 0)

        XCTAssertThrowsError(try plan.validate(hiddenStatePacket: packet)) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .invalidPacket(
                    "hidden-state packet shape [1, 1, 4] does not match boundary tensor shape [1, -1, 2]"))
        }
    }

    func testStagePlanCodableUsesStableSnakeCaseKeys() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let data = try JSONEncoder().encode(plan)
        let json = String(decoding: data, as: UTF8.self)

        XCTAssertTrue(json.contains(#""model_name""#))
        XCTAssertTrue(json.contains(#""total_layer_count""#))
        XCTAssertTrue(json.contains(#""layer_range""#))
        XCTAssertTrue(json.contains(#""worker_id""#))
        XCTAssertTrue(json.contains(#""boundary_tensor""#))

        let decoded = try JSONDecoder().decode(DistributedStagePlan.self, from: data)
        XCTAssertEqual(decoded, plan)
    }

    func testStageManifestLoadsTopLevelManifest() throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(stageManifestJSON(modelKey: "model", includeTotalLayerCount: true).utf8),
            baseURL: URL(fileURLWithPath: "/tmp/caix-manifest", isDirectory: true))

        XCTAssertEqual(manifest.schema, DistributedStageManifest.currentSchema)
        XCTAssertEqual(manifest.modelName, "qwen3-0.6b-coreai")
        XCTAssertEqual(manifest.totalLayerCount, 28)
        XCTAssertFalse(manifest.totalLayerCountDerived)
        XCTAssertEqual(manifest.stages.count, 4)
        XCTAssertEqual(manifest.stages[1].layerRange, DistributedLayerRange(lowerBound: 0, upperBound: 14))
        XCTAssertEqual(manifest.boundaryTensor?.name, "hidden_states")
        XCTAssertEqual(manifest.boundaryTensor?.shape, [1, -1, 1024])
        XCTAssertEqual(manifest.boundaryTensor?.scalarType, .float16)
        XCTAssertEqual(manifest.runtimePlan.boundaryTensor, manifest.boundaryTensor)
        XCTAssertEqual(
            manifest.stages[1].resolvedAssetPath,
            "/tmp/caix-manifest/stages/01-layers-00-14.aimodel")
        XCTAssertEqual(manifest.runtimePlan.stages.map(\.id), [
            "embed", "layers-00-14", "layers-14-28", "head",
        ])
        XCTAssertNoThrow(try manifest.runtimePlan.validate())
    }

    func testStageManifestLoadsMetadataClusterBlockAndDerivesLayerCount() throws {
        let json =
            """
            {
              "name": "qwen3-0.6b-coreai",
              "cluster": \(stageManifestJSON(modelKey: nil, includeTotalLayerCount: false))
            }
            """

        let manifest = try DistributedStageManifest.decode(
            from: Data(json.utf8),
            baseURL: URL(fileURLWithPath: "/tmp/qwen3", isDirectory: true),
            requireClusterBlock: true)

        XCTAssertEqual(manifest.modelName, "qwen3-0.6b-coreai")
        XCTAssertEqual(manifest.totalLayerCount, 28)
        XCTAssertTrue(manifest.totalLayerCountDerived)
        XCTAssertEqual(manifest.boundaryTensor?.shape, [1, -1, 1024])
        XCTAssertEqual(manifest.runtimePlan.stages[2].layerRange, DistributedLayerRange(lowerBound: 14, upperBound: 28))
    }

    func testStageManifestRejectsLayerCoverageGap() {
        let json =
            """
            {
              "schema": "\(DistributedStageManifest.currentSchema)",
              "model": "qwen3-0.6b-coreai",
              "total_layer_count": 28,
              "stages": [
                {"id":"embed","role":"embeddings","layers":"embeddings","bundle":"embed.aimodel","memory_gb":1},
                {"id":"layers-00-10","role":"transformer_layers","layers":[0,10],"bundle":"layers-a.aimodel","memory_gb":2},
                {"id":"layers-12-28","role":"transformer_layers","layers":[12,28],"bundle":"layers-b.aimodel","memory_gb":2},
                {"id":"head","role":"final_norm_head","layers":"norm+lm_head","bundle":"head.aimodel","memory_gb":1}
              ]
            }
            """

        XCTAssertThrowsError(try DistributedStageManifest.decode(from: Data(json.utf8))) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .layerCoverageGap(
                    expectedStart: 10,
                    stageID: "layers-12-28",
                    actual: DistributedLayerRange(lowerBound: 12, upperBound: 28)))
        }
    }

    func testStageManifestRejectsBadBoundaryTensorShape() {
        let json =
            """
            {
              "schema": "\(DistributedStageManifest.currentSchema)",
              "model": "qwen3-0.6b-coreai",
              "total_layer_count": 28,
              "boundary": {
                "hidden_state": {
                  "name": "hidden_states",
                  "shape": [1, 0, 1024],
                  "scalar_type": "float16"
                }
              },
              "stages": [
                {"id":"embed","role":"embeddings","layers":"embeddings","bundle":"embed.aimodel","memory_gb":1},
                {"id":"layers-00-28","role":"transformer_layers","layers":[0,28],"bundle":"layers.aimodel","memory_gb":2},
                {"id":"head","role":"final_norm_head","layers":"norm+lm_head","bundle":"head.aimodel","memory_gb":1}
              ]
            }
            """

        XCTAssertThrowsError(try DistributedStageManifest.decode(from: Data(json.utf8))) { error in
            XCTAssertEqual(
                error as? DistributedStageManifestError,
                .invalidManifest(
                    "boundary hidden_state sequence dimension must be positive or -1"))
        }
    }

    func testSameMachinePipelineForwardsThroughOrderedStageHandles() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        try await pipeline.allocate(requestID: "req-1", kvCapacity: 16)
        let output = try await pipeline.forward(
            requestID: "req-1",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
            tokenIDs: [1, 2, 3])

        XCTAssertEqual(output.stageID, "final")
        XCTAssertEqual(output.stepIndex, 0)
        XCTAssertEqual(output.tokenID, 42)
        XCTAssertNil(output.hiddenState)
        XCTAssertEqual(handles.map(\.allocatedRequests), [["req-1"], ["req-1"], ["req-1"], ["req-1"]])
        XCTAssertEqual(handles[0].inputs.first?.tokenIDs, [1, 2, 3])
        XCTAssertNil(handles[0].inputs.first?.hiddenState)
        XCTAssertEqual(handles[1].inputs.first?.hiddenState?.metadata.sourceStageID, "embed")
        XCTAssertEqual(handles[2].inputs.first?.hiddenState?.metadata.sourceStageID, "layers-0-16")
        XCTAssertEqual(handles[3].inputs.first?.hiddenState?.metadata.sourceStageID, "layers-16-32")
    }

    func testSameMachinePipelineBuildsFromManifestHandleMap() async throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(stageManifestJSON(modelKey: "model", includeTotalLayerCount: true).utf8),
            baseURL: URL(fileURLWithPath: "/tmp/caix-manifest", isDirectory: true))
        let handles = makeFakeHandles(for: manifest.runtimePlan)
        let handlesByStageID = Dictionary(uniqueKeysWithValues: handles.reversed().map {
            ($0.descriptor.id, $0)
        })

        let pipeline = try DistributedSameMachinePipeline(
            manifest: manifest, handlesByStageID: handlesByStageID)
        try await pipeline.allocate(requestID: "req-manifest", kvCapacity: 8)
        let output = try await pipeline.forward(
            requestID: "req-manifest",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
            tokenIDs: [11, 12])

        XCTAssertEqual(output.stageID, "head")
        XCTAssertEqual(output.tokenID, 42)
        XCTAssertEqual(handles[1].inputs.first?.hiddenState?.metadata.shape, [1, 2, 1024])
    }

    func testSameMachinePipelineBuildsFromManifestFactory() async throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(stageManifestJSON(modelKey: "model", includeTotalLayerCount: true).utf8),
            baseURL: URL(fileURLWithPath: "/tmp/caix-manifest", isDirectory: true))
        let handles = makeFakeHandles(for: manifest.runtimePlan)
        let factory = FakeDistributedStageHandleFactory(handles: handles)

        let pipeline = try await DistributedSameMachinePipeline.make(
            manifest: manifest, handleFactory: factory)
        try await pipeline.allocate(requestID: "req-factory", kvCapacity: 8)
        let output = try await pipeline.forward(
            requestID: "req-factory",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
            tokenIDs: [21, 22])

        XCTAssertEqual(factory.requestedStageIDs, manifest.stages.map(\.id))
        XCTAssertEqual(factory.requestedAssetNames.first, "stages/00-embed.aimodel")
        XCTAssertEqual(
            factory.requestedAssetPaths.first,
            "/tmp/caix-manifest/stages/00-embed.aimodel")
        XCTAssertEqual(factory.requestedBoundaryShapes.first, [1, -1, 1024])
        XCTAssertEqual(output.stageID, "head")
        XCTAssertEqual(output.tokenID, 42)
    }

    func testSameMachinePipelineFactoryPropagatesMissingHandle() async throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(stageManifestJSON(modelKey: "model", includeTotalLayerCount: true).utf8),
            baseURL: URL(fileURLWithPath: "/tmp/caix-manifest", isDirectory: true))
        let handles = makeFakeHandles(for: manifest.runtimePlan).filter {
            $0.descriptor.id != "layers-14-28"
        }
        let factory = FakeDistributedStageHandleFactory(handles: handles)

        await XCTAssertThrowsErrorAsync(
            try await DistributedSameMachinePipeline.make(
                manifest: manifest, handleFactory: factory)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .missingStageHandle("layers-14-28"))
        }
    }

    func testSameMachinePipelineFactoryRequiresResolvedAssetPath() async throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(stageManifestJSON(modelKey: "model", includeTotalLayerCount: true).utf8))
        let factory = ResolvingDistributedStageHandleFactory()

        await XCTAssertThrowsErrorAsync(
            try await DistributedSameMachinePipeline.make(
                manifest: manifest, handleFactory: factory)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .missingStageAssetPath("embed"))
        }
    }

    func testSameMachinePipelineRejectsMissingManifestHandle() throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(stageManifestJSON(modelKey: "model", includeTotalLayerCount: true).utf8),
            baseURL: URL(fileURLWithPath: "/tmp/caix-manifest", isDirectory: true))
        var handlesByStageID = Dictionary(uniqueKeysWithValues: makeFakeHandles(for: manifest.runtimePlan).map {
            ($0.descriptor.id, $0)
        })
        handlesByStageID.removeValue(forKey: "layers-14-28")

        XCTAssertThrowsError(
            try DistributedSameMachinePipeline(
                manifest: manifest, handlesByStageID: handlesByStageID)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .missingStageHandle("layers-14-28"))
        }
    }

    func testSameMachinePipelineRejectsOutOfOrderHandles() throws {
        let plan = makePlan()
        var handles = makeFakeHandles(for: plan)
        handles.swapAt(1, 2)

        XCTAssertThrowsError(try DistributedSameMachinePipeline(plan: plan, stages: handles)) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .stageDescriptorMismatch(expected: "layers-0-16", actual: "layers-16-32"))
        }
    }

    func testSameMachinePipelineRejectsBadPacketRoute() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan, badFirstRoute: true)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)
        try await pipeline.allocate(requestID: "req-1", kvCapacity: 16)

        await XCTAssertThrowsErrorAsync(
            try await pipeline.forward(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                tokenIDs: [7])
        ) { error in
            XCTAssertEqual(
                error as? DistributedRuntimeValidationError,
                .packetRouteMismatch(sourceStageID: "embed", destinationStageID: "final"))
        }
    }

    func testSameMachinePipelineRejectsEmptyForwardRequestID() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        await XCTAssertThrowsErrorAsync(
            try await pipeline.forward(
                requestID: " ",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                tokenIDs: [7])
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("request_id is empty"))
        }
    }

    private func makePlan(
        stages: [DistributedStageDescriptor]? = nil,
        workers: [DistributedWorkerEndpoint]? = nil,
        boundaryTensor: DistributedBoundaryTensorSpec? = nil
    ) -> DistributedStagePlan {
        DistributedStagePlan(
            modelName: "qwen3-0.6b-coreai",
            totalLayerCount: 32,
            stages: stages ?? [
                stage("embed", .embeddings, assetName: "embeddings", workerID: "coordinator"),
                stage(
                    "layers-0-16", .transformerLayers,
                    range: DistributedLayerRange(lowerBound: 0, upperBound: 16),
                    assetName: "layers_0_16", workerID: "worker-a"),
                stage(
                    "layers-16-32", .transformerLayers,
                    range: DistributedLayerRange(lowerBound: 16, upperBound: 32),
                    assetName: "layers_16_32", workerID: "worker-b"),
                stage("final", .finalNormHead, assetName: "final", workerID: "coordinator"),
            ],
            workers: workers ?? [
                DistributedWorkerEndpoint(id: "coordinator", host: "127.0.0.1", port: 9010),
                DistributedWorkerEndpoint(id: "worker-a", host: "127.0.0.1", port: 9011),
                DistributedWorkerEndpoint(id: "worker-b", host: "127.0.0.1", port: 9012),
            ],
            boundaryTensor: boundaryTensor)
    }

    private func stage(
        _ id: String,
        _ role: DistributedStageRole,
        range: DistributedLayerRange? = nil,
        assetName: String,
        workerID: String? = nil
    ) -> DistributedStageDescriptor {
        DistributedStageDescriptor(
            id: id, role: role, layerRange: range, assetName: assetName, workerID: workerID)
    }

    private func stageManifestJSON(modelKey: String?, includeTotalLayerCount: Bool) -> String {
        let modelLine = modelKey.map { #""\#($0)": "qwen3-0.6b-coreai","# } ?? ""
        let totalLine = includeTotalLayerCount ? #""total_layer_count": 28,"# : ""
        return
            """
            {
              "schema": "\(DistributedStageManifest.currentSchema)",
              \(modelLine)
              \(totalLine)
              "boundary": {
                "hidden_state": {
                  "name": "hidden_states",
                  "shape": [1, -1, 1024],
                  "scalar_type": "float16"
                }
              },
              "stages": [
                {
                  "id": "embed",
                  "role": "embeddings",
                  "layers": "embeddings",
                  "bundle": "stages/00-embed.aimodel",
                  "memory_gb": 1.0
                },
                {
                  "id": "layers-00-14",
                  "role": "transformer_layers",
                  "layers": [0, 14],
                  "bundle": "stages/01-layers-00-14.aimodel",
                  "memory_gb": 2.0
                },
                {
                  "id": "layers-14-28",
                  "role": "transformer_layers",
                  "layers": {"lower_bound": 14, "upper_bound": 28},
                  "bundle": "stages/02-layers-14-28.aimodel",
                  "memory_gb": "2GB"
                },
                {
                  "id": "head",
                  "role": "final_norm_head",
                  "layers": "norm+lm_head",
                  "bundle": "stages/03-head.aimodel",
                  "memory_gb": 1.0
                }
              ]
            }
            """
    }

    private func makeFakeHandles(
        for plan: DistributedStagePlan,
        badFirstRoute: Bool = false
    ) -> [FakeDistributedStageHandle] {
        let hiddenWidth = plan.boundaryTensor?.shape[2] ?? 2
        return plan.stages.enumerated().map { index, descriptor in
            let nextID = plan.stages.indices.contains(index + 1) ? plan.stages[index + 1].id : nil
            let output: (DistributedStageForwardInput) throws -> DistributedStageForwardOutput = { input in
                if descriptor.role == .finalNormHead {
                    return DistributedStageForwardOutput(
                        stageID: descriptor.id, stepIndex: input.stepIndex, tokenID: 42)
                }
                let destination = badFirstRoute && descriptor.id == "embed" ? "final" : nextID!
                let packet = try self.hiddenPacket(
                    requestID: input.requestID,
                    source: descriptor.id,
                    destination: destination,
                    positionRange: input.positionRange,
                    stepIndex: input.stepIndex,
                    hiddenWidth: hiddenWidth,
                    fill: UInt8(index + 1))
                return DistributedStageForwardOutput(
                    stageID: descriptor.id, stepIndex: input.stepIndex, hiddenState: packet)
            }
            return FakeDistributedStageHandle(descriptor: descriptor, output: output)
        }
    }

    private func hiddenPacket(
        requestID: String,
        source: String,
        destination: String,
        positionRange: DistributedSequenceRange,
        stepIndex: Int,
        hiddenWidth: Int = 2,
        fill: UInt8
    ) throws -> DistributedHiddenStatePacket {
        let shape = [1, positionRange.count, hiddenWidth]
        let byteCount = shape.reduce(DistributedTensorScalarType.float16.byteWidth, *)
        return try DistributedHiddenStatePacket(
            metadata: DistributedHiddenStatePacketMetadata(
                requestID: requestID,
                sourceStageID: source,
                destinationStageID: destination,
                positionRange: positionRange,
                shape: shape,
                scalarType: .float16,
                byteCount: byteCount,
                stepIndex: stepIndex),
            payload: Array(repeating: fill, count: byteCount))
    }
}

private final class FakeDistributedStageHandle: DistributedStageHandle {
    let descriptor: DistributedStageDescriptor
    var allocatedRequests: [String] = []
    var inputs: [DistributedStageForwardInput] = []
    private let output: (DistributedStageForwardInput) throws -> DistributedStageForwardOutput

    init(
        descriptor: DistributedStageDescriptor,
        output: @escaping (DistributedStageForwardInput) throws -> DistributedStageForwardOutput
    ) {
        self.descriptor = descriptor
        self.output = output
    }

    func allocate(_ allocation: DistributedStageAllocation) async throws {
        allocatedRequests.append(allocation.requestID)
    }

    func forward(_ input: DistributedStageForwardInput) async throws -> DistributedStageForwardOutput {
        inputs.append(input)
        return try output(input)
    }

    func reset(requestID: String) async throws {}

    func free(requestID: String) async {}
}

private final class FakeDistributedStageHandleFactory: DistributedStageHandleFactory {
    var requestedStageIDs: [String] = []
    var requestedAssetNames: [String] = []
    var requestedAssetPaths: [String] = []
    var requestedBoundaryShapes: [[Int]] = []
    private let handlesByStageID: [String: FakeDistributedStageHandle]

    init(handles: [FakeDistributedStageHandle]) {
        self.handlesByStageID = Dictionary(uniqueKeysWithValues: handles.map {
            ($0.descriptor.id, $0)
        })
    }

    func makeStageHandle(
        for context: DistributedStageHandleFactoryContext
    ) async throws -> DistributedStageHandle {
        let stage = context.stage
        requestedStageIDs.append(stage.id)
        requestedAssetNames.append(stage.assetName)
        requestedAssetPaths.append(try context.requireResolvedAssetURL().path)
        requestedBoundaryShapes.append(context.boundaryTensor?.shape ?? [])
        guard let handle = handlesByStageID[stage.id] else {
            throw DistributedStageExecutionError.missingStageHandle(stage.id)
        }
        return handle
    }
}

private final class ResolvingDistributedStageHandleFactory: DistributedStageHandleFactory {
    func makeStageHandle(
        for context: DistributedStageHandleFactoryContext
    ) async throws -> DistributedStageHandle {
        _ = try context.requireResolvedAssetURL()
        throw DistributedStageExecutionError.missingStageHandle(context.stage.id)
    }
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line,
    _ errorHandler: (_ error: Error) -> Void = { _ in }
) async {
    do {
        _ = try await expression()
        XCTFail(message(), file: file, line: line)
    } catch {
        errorHandler(error)
    }
}
