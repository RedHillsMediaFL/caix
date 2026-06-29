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

    func testStagePlanCodableUsesStableSnakeCaseKeys() throws {
        let plan = makePlan()
        let data = try JSONEncoder().encode(plan)
        let json = String(decoding: data, as: UTF8.self)

        XCTAssertTrue(json.contains(#""model_name""#))
        XCTAssertTrue(json.contains(#""total_layer_count""#))
        XCTAssertTrue(json.contains(#""layer_range""#))
        XCTAssertTrue(json.contains(#""worker_id""#))

        let decoded = try JSONDecoder().decode(DistributedStagePlan.self, from: data)
        XCTAssertEqual(decoded, plan)
    }

    private func makePlan(
        stages: [DistributedStageDescriptor]? = nil,
        workers: [DistributedWorkerEndpoint]? = nil
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
            ])
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
}
