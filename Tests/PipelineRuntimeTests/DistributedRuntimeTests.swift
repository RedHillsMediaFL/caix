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

    func testWorkerMessageEnvelopeRoundTripsForwardFrame() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 5),
            stepIndex: 1,
            fill: 9)
        let frame = DistributedStageForwardFrame(
            stageID: "layers-16-32",
            requestID: "req-1",
            stepIndex: 1,
            positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 5),
            positionIDs: [4],
            hiddenState: packet.metadata)
        let message = DistributedWorkerMessage.forward(frame)

        try frame.validate(against: plan)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(message)
        let json = String(decoding: data, as: UTF8.self)

        XCTAssertTrue(json.contains(#""kind":"forward""#))
        XCTAssertTrue(json.contains(#""position_ids":[4]"#))
        XCTAssertTrue(json.contains(#""hidden_state""#))
        XCTAssertEqual(try JSONDecoder().decode(DistributedWorkerMessage.self, from: data), message)
    }

    func testWorkerMessageEnvelopeRoundTripsControlFrames() throws {
        let hello = DistributedWorkerMessage.hello(
            DistributedWorkerHello(
                stage: stage(
                    "layers-0-16", .transformerLayers,
                    range: DistributedLayerRange(lowerBound: 0, upperBound: 16),
                    assetName: "layers_0_16", workerID: "worker-a"),
                hiddenSize: 4096,
                boundaryScalarType: .float16,
                cacheContract: "stateful",
                planIntegrityHash: "abc123",
                freeMemoryBytes: 1024,
                computeUnit: "all"))
        let alloc = DistributedWorkerMessage.allocate(
            DistributedStageAllocation(requestID: "req-1", kvCapacity: 128))
        let ack = DistributedWorkerMessage.helloAck(
            DistributedWorkerHelloAck(
                accepted: true, stageID: "layers-0-16", planIntegrityHash: "abc123"))
        let reset = DistributedWorkerMessage.reset(
            DistributedRequestControl(requestID: "req-1", stageID: "layers-0-16"))
        let error = DistributedWorkerMessage.error(
            DistributedWorkerErrorFrame(
                code: "plan_mismatch",
                detail: "worker plan hash did not match",
                stageID: "layers-0-16"))

        for message in [hello, alloc, ack, reset, error] {
            let data = try JSONEncoder().encode(message)
            XCTAssertEqual(try JSONDecoder().decode(DistributedWorkerMessage.self, from: data), message)
        }
    }

    func testWorkerMessageCodecRoundTripsSingleJSONLine() throws {
        let message = DistributedWorkerMessage.allocate(
            DistributedStageAllocation(requestID: "req-1", kvCapacity: 128))

        let line = try DistributedWorkerMessageCodec.encodeJSONLine(message)

        XCTAssertEqual(line.last, 0x0A)
        XCTAssertFalse(line.dropLast().contains(0x0A))
        XCTAssertEqual(try DistributedWorkerMessageCodec.decodeJSONLine(line), message)
        XCTAssertEqual(
            try DistributedWorkerMessageCodec.decodeJSONLine(Data(line.dropLast())),
            message)
    }

    func testWorkerMessageCodecAcceptsCRLFLineEnding() throws {
        let message = DistributedWorkerMessage.free(
            DistributedRequestControl(requestID: "req-1", stageID: "layers-0-16"))
        var line = try DistributedWorkerMessageCodec.encodeJSONLine(message)
        line.insert(0x0D, at: line.count - 1)

        XCTAssertEqual(try DistributedWorkerMessageCodec.decodeJSONLine(line), message)
    }

    func testWorkerMessageCodecRejectsEmptyAndMultiFrameLines() throws {
        XCTAssertThrowsError(try DistributedWorkerMessageCodec.decodeJSONLine(Data("\n".utf8))) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWireFrame("worker message line is empty"))
        }

        let first = try DistributedWorkerMessageCodec.encodeJSONLine(
            .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 1)))
        let second = try DistributedWorkerMessageCodec.encodeJSONLine(
            .free(DistributedRequestControl(requestID: "req-1")))

        XCTAssertThrowsError(
            try DistributedWorkerMessageCodec.decodeJSONLine(first + second)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWireFrame("worker message line contains multiple frames"))
        }
    }

    func testWorkerMessagePayloadExpectationUsesHiddenStateByteCount() throws {
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 7),
            stepIndex: 1,
            fill: 9)
        let forward = DistributedWorkerMessage.forward(
            DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 1,
                positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 7),
                positionIDs: [4, 5, 6],
                hiddenState: packet.metadata))
        let forwardResult = DistributedWorkerMessage.forwardResult(
            DistributedStageForwardResultFrame(
                stageID: "layers-0-16",
                requestID: "req-1",
                stepIndex: 1,
                hiddenState: packet.metadata))
        let alloc = DistributedWorkerMessage.allocate(
            DistributedStageAllocation(requestID: "req-1", kvCapacity: 128))

        XCTAssertEqual(packet.metadata.byteCount, 12)
        XCTAssertEqual(forward.expectedPayloadByteCount, 12)
        XCTAssertTrue(forward.expectsPayload)
        XCTAssertEqual(forwardResult.expectedPayloadByteCount, 12)
        XCTAssertTrue(forwardResult.expectsPayload)
        XCTAssertEqual(alloc.expectedPayloadByteCount, 0)
        XCTAssertFalse(alloc.expectsPayload)
    }

    func testWireFrameValidatesPayloadLength() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 7),
            stepIndex: 1,
            fill: 9)
        let message = DistributedWorkerMessage.forward(
            DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 1,
                positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 7),
                positionIDs: [4, 5, 6],
                hiddenState: packet.metadata))

        XCTAssertNoThrow(
            try DistributedWorkerWireFrame(message: message, payload: packet.payload)
                .validate(against: plan))
        XCTAssertThrowsError(
            try DistributedWorkerWireFrame(message: message, payload: Array(packet.payload.dropLast()))
                .validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWireFrame("payload byte count 11 does not match header 12"))
        }
    }

    func testWireFrameRejectsUnexpectedControlPayload() throws {
        let plan = makePlan()
        let message = DistributedWorkerMessage.allocate(
            DistributedStageAllocation(requestID: "req-1", kvCapacity: 128))

        XCTAssertThrowsError(
            try DistributedWorkerWireFrame(message: message, payload: [1])
                .validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWireFrame("payload byte count 1 does not match header 0"))
        }
    }

    func testWorkerMessageCodecRoundTripsWireFrameWithPayloadBytes() throws {
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 7),
            stepIndex: 1,
            fill: 9)
        let message = DistributedWorkerMessage.forward(
            DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 1,
                positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 7),
                positionIDs: [4, 5, 6],
                hiddenState: packet.metadata))
        let payload: [UInt8] = [0, 10, 13, 1, 2, 3, 4, 5, 6, 7, 8, 255]
        let frame = DistributedWorkerWireFrame(message: message, payload: payload)

        let encoded = try DistributedWorkerMessageCodec.encodeWireFrame(frame)
        let decoded = try DistributedWorkerMessageCodec.decodeWireFrame(encoded)

        XCTAssertEqual(decoded, frame)
        XCTAssertEqual(encoded.filter { $0 == 0x0A }.count, 2)
    }

    func testWorkerMessageCodecRejectsWireFrameWithoutHeaderLineEnding() {
        XCTAssertThrowsError(
            try DistributedWorkerMessageCodec.decodeWireFrame(Data(#"{"kind":"alloc"}"#.utf8))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWireFrame("worker wire frame header is missing line ending"))
        }
    }

    func testWorkerMessageCodecRejectsExtraControlPayloadBytes() throws {
        let message = DistributedWorkerMessage.allocate(
            DistributedStageAllocation(requestID: "req-1", kvCapacity: 128))
        var encoded = try DistributedWorkerMessageCodec.encodeJSONLine(message)
        encoded.append(0x01)

        XCTAssertThrowsError(
            try DistributedWorkerMessageCodec.decodeWireFrame(encoded)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWireFrame("payload byte count 1 does not match header 0"))
        }
    }

    func testWireFrameStreamDecoderWaitsForCompleteHeaderAndPayload() throws {
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 7),
            stepIndex: 1,
            fill: 9)
        let frame = DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 1,
                positionRange: DistributedSequenceRange(lowerBound: 4, upperBound: 7),
                positionIDs: [4, 5, 6],
                hiddenState: packet.metadata)),
            payload: packet.payload)
        let encoded = try DistributedWorkerMessageCodec.encodeWireFrame(frame)
        let headerEnd = try XCTUnwrap(encoded.firstIndex(of: 0x0A))
        var decoder = DistributedWorkerWireFrameStreamDecoder()

        decoder.append(encoded.prefix(headerEnd))
        XCTAssertNil(try decoder.nextFrame())
        decoder.append(encoded[headerEnd...headerEnd])
        XCTAssertNil(try decoder.nextFrame())
        decoder.append(encoded[(headerEnd + 1)..<(encoded.count - 1)])
        XCTAssertNil(try decoder.nextFrame())
        decoder.append(encoded[(encoded.count - 1)..<encoded.count])

        XCTAssertEqual(try decoder.nextFrame(), frame)
        XCTAssertEqual(decoder.bufferedByteCount, 0)
        XCTAssertNoThrow(try decoder.finish())
    }

    func testWireFrameStreamDecoderDrainsSequentialFrames() throws {
        let alloc = DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 128)))
        let free = DistributedWorkerWireFrame(
            message: .free(DistributedRequestControl(requestID: "req-1", stageID: "layers-0-16")))
        let encoded = try DistributedWorkerMessageCodec.encodeWireFrame(alloc)
            + DistributedWorkerMessageCodec.encodeWireFrame(free)
        var decoder = DistributedWorkerWireFrameStreamDecoder()

        decoder.append(encoded)

        XCTAssertEqual(try decoder.drainFrames(), [alloc, free])
        XCTAssertNoThrow(try decoder.finish())
    }

    func testWireFrameStreamDecoderReportsPartialTrailingBytes() throws {
        let alloc = DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 128)))
        let encoded = try DistributedWorkerMessageCodec.encodeWireFrame(alloc)
        var decoder = DistributedWorkerWireFrameStreamDecoder()

        decoder.append(encoded.dropLast())

        XCTAssertNil(try decoder.nextFrame())
        XCTAssertThrowsError(try decoder.finish()) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWireFrame(
                    "worker wire frame stream ended with \(encoded.count - 1) buffered bytes"))
        }
    }

    func testWorkerFrameExecutorBuildsValidatedHello() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[1]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)

        let frame = try executor.makeHello(
            cacheContract: "stateful",
            freeMemoryBytes: 1024,
            computeUnit: "all",
            labels: ["lane": "middle"])

        XCTAssertNoThrow(try frame.validate(against: plan))
        guard case .hello(let hello) = frame.message else {
            return XCTFail("expected hello frame")
        }
        XCTAssertEqual(hello.stage, handle.descriptor)
        XCTAssertEqual(hello.hiddenSize, 2)
        XCTAssertEqual(hello.boundaryScalarType, .float16)
        XCTAssertEqual(hello.cacheContract, "stateful")
        XCTAssertEqual(hello.freeMemoryBytes, 1024)
        XCTAssertEqual(hello.computeUnit, "all")
        XCTAssertEqual(hello.labels, ["lane": "middle"])
        XCTAssertEqual(hello.planIntegrityHash, try plan.integrityHash())
    }

    func testWorkerHandshakeCoordinatorAcceptsAllStagesAndReportsReady() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handles = makeFakeHandles(for: plan)
        var coordinator = try DistributedWorkerHandshakeCoordinator(plan: plan)

        for handle in handles {
            let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
            let response = try coordinator.processHello(try executor.makeHello())
            guard case .helloAck(let ack) = response.message else {
                return XCTFail("expected hello_ack")
            }

            XCTAssertTrue(ack.accepted)
            XCTAssertEqual(ack.stageID, handle.descriptor.id)
            XCTAssertNil(ack.reason)
            XCTAssertEqual(ack.planIntegrityHash, try plan.integrityHash())
            XCTAssertTrue(response.payload.isEmpty)
        }

        XCTAssertEqual(coordinator.claimedStages, Set(plan.stages.map(\.id)))
        XCTAssertEqual(coordinator.missingStageIDs, [])
        XCTAssertTrue(coordinator.isReady)
        XCTAssertNoThrow(try coordinator.requireReady())
    }

    func testWorkerHandshakeCoordinatorRejectsDuplicateStageClaim() throws {
        let plan = makePlan()
        let handle = makeFakeHandles(for: plan)[1]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let hello = try executor.makeHello()
        var coordinator = try DistributedWorkerHandshakeCoordinator(plan: plan)

        _ = try coordinator.processHello(hello)
        let response = try coordinator.processHello(hello)

        guard case .helloAck(let ack) = response.message else {
            return XCTFail("expected hello_ack")
        }
        XCTAssertFalse(ack.accepted)
        XCTAssertEqual(ack.stageID, handle.descriptor.id)
        XCTAssertEqual(ack.reason, "stage already claimed")
        XCTAssertNil(ack.planIntegrityHash)
        XCTAssertEqual(coordinator.claimedStages, [handle.descriptor.id])
    }

    func testWorkerHandshakeCoordinatorRejectsPlanMismatch() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        var coordinator = try DistributedWorkerHandshakeCoordinator(plan: plan)
        let hello = DistributedWorkerHello(
            stage: try XCTUnwrap(plan.stage(id: "layers-0-16")),
            hiddenSize: 2,
            boundaryScalarType: .float16,
            planIntegrityHash: "stale")

        let response = try coordinator.processHello(DistributedWorkerWireFrame(
            message: .hello(hello)))

        guard case .helloAck(let ack) = response.message else {
            return XCTFail("expected hello_ack")
        }
        XCTAssertFalse(ack.accepted)
        XCTAssertEqual(ack.stageID, "layers-0-16")
        XCTAssertEqual(ack.reason, "plan_integrity_hash mismatch")
        XCTAssertNil(ack.planIntegrityHash)
        XCTAssertTrue(coordinator.claimedStages.isEmpty)
    }

    func testWorkerHandshakeCoordinatorRejectsNonHelloFrame() throws {
        let plan = makePlan()
        var coordinator = try DistributedWorkerHandshakeCoordinator(plan: plan)

        XCTAssertThrowsError(
            try coordinator.processHello(DistributedWorkerWireFrame(
                message: .allocate(DistributedStageAllocation(
                    requestID: "req-1", kvCapacity: 128))))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("handshake requires hello frame"))
        }
    }

    func testWorkerHandshakeCoordinatorRequiresAllStages() throws {
        let plan = makePlan()
        let handle = makeFakeHandles(for: plan)[0]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        var coordinator = try DistributedWorkerHandshakeCoordinator(plan: plan)

        _ = try coordinator.processHello(try executor.makeHello())

        XCTAssertFalse(coordinator.isReady)
        XCTAssertEqual(
            coordinator.missingStageIDs,
            ["layers-0-16", "layers-16-32", "final"])
        XCTAssertThrowsError(try coordinator.requireReady()) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame(
                    "missing worker stages: layers-0-16, layers-16-32, final"))
        }
    }

    func testWorkerFrameExecutorProcessesControlFrames() async throws {
        let plan = makePlan()
        let handle = makeFakeHandles(for: plan)[1]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)

        let allocateResponse = try await executor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 128))))
        let resetResponse = try await executor.process(DistributedWorkerWireFrame(
            message: .reset(DistributedRequestControl(
                requestID: "req-1", stageID: handle.descriptor.id))))
        let freeResponse = try await executor.process(DistributedWorkerWireFrame(
            message: .free(DistributedRequestControl(
                requestID: "req-1", stageID: handle.descriptor.id)))
        )

        XCTAssertNil(allocateResponse)
        XCTAssertNil(resetResponse)
        XCTAssertNil(freeResponse)

        XCTAssertEqual(handle.allocatedRequests, ["req-1"])
        XCTAssertEqual(handle.resetRequests, ["req-1"])
        XCTAssertEqual(handle.freeRequests, ["req-1"])
    }

    func testWorkerFrameExecutorRejectsDuplicateAllocateAndAllowsReallocateAfterFree() async throws {
        let plan = makePlan()
        let handle = makeFakeHandles(for: plan)[1]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)

        let firstAllocateResponse = try await executor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 128)))
        )
        XCTAssertNil(firstAllocateResponse)
        await XCTAssertThrowsErrorAsync(
            try await executor.process(DistributedWorkerWireFrame(
                message: .allocate(DistributedStageAllocation(
                    requestID: "req-1", kvCapacity: 128))))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("request_id req-1 is already allocated"))
        }
        XCTAssertEqual(handle.allocatedRequests, ["req-1"])

        let freeResponse = try await executor.process(DistributedWorkerWireFrame(
            message: .free(DistributedRequestControl(
                requestID: "req-1", stageID: handle.descriptor.id)))
        )
        let secondAllocateResponse = try await executor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 64)))
        )
        XCTAssertNil(freeResponse)
        XCTAssertNil(secondAllocateResponse)

        XCTAssertEqual(handle.freeRequests, ["req-1"])
        XCTAssertEqual(handle.allocatedRequests, ["req-1", "req-1"])
    }

    func testWorkerFrameExecutorProcessesForwardFrame() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
            stepIndex: 0,
            fill: 9)
        let request = DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
                positionIDs: [0, 1, 2],
                hiddenState: packet.metadata)),
            payload: packet.payload)

        let allocateResponse = try await executor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 8))))
        let maybeResponse = try await executor.process(request)
        let response = try XCTUnwrap(maybeResponse)

        XCTAssertNil(allocateResponse)
        XCTAssertEqual(handle.inputs.count, 1)
        XCTAssertEqual(handle.inputs[0].requestID, "req-1")
        XCTAssertEqual(handle.inputs[0].hiddenState, packet)
        XCTAssertNoThrow(try response.validate(against: plan))
        guard case .forwardResult(let result) = response.message else {
            return XCTFail("expected forward_result frame")
        }
        XCTAssertEqual(result.stageID, "layers-16-32")
        XCTAssertEqual(result.requestID, "req-1")
        XCTAssertEqual(result.stepIndex, 0)
        XCTAssertEqual(result.hiddenState?.sourceStageID, "layers-16-32")
        XCTAssertEqual(result.hiddenState?.destinationStageID, "final")
        XCTAssertEqual(response.payload.count, 12)
    }

    func testWorkerFrameExecutorRejectsForwardBeforeAllocate() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 7)

        await XCTAssertThrowsErrorAsync(
            try await executor.process(DistributedWorkerWireFrame(
                message: .forward(DistributedStageForwardFrame(
                    stageID: "layers-16-32",
                    requestID: "req-1",
                    stepIndex: 0,
                    positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                    positionIDs: [0],
                    hiddenState: packet.metadata)),
                payload: packet.payload))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("request_id req-1 is not allocated"))
        }
        XCTAssertTrue(handle.inputs.isEmpty)
    }

    func testWorkerFrameExecutorRejectsStepDrift() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let firstPacket = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 7)
        let stalePacket = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 1, upperBound: 2),
            stepIndex: 0,
            fill: 8)

        _ = try await executor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 4))))
        _ = try await executor.process(DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                hiddenState: firstPacket.metadata)),
            payload: firstPacket.payload))

        await XCTAssertThrowsErrorAsync(
            try await executor.process(DistributedWorkerWireFrame(
                message: .forward(DistributedStageForwardFrame(
                    stageID: "layers-16-32",
                    requestID: "req-1",
                    stepIndex: 0,
                    positionRange: DistributedSequenceRange(lowerBound: 1, upperBound: 2),
                    positionIDs: [1],
                    hiddenState: stalePacket.metadata)),
                payload: stalePacket.payload))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("step_index 0 does not match expected 1"))
        }
        XCTAssertEqual(handle.inputs.count, 1)
    }

    func testWorkerFrameExecutorRejectsPositionDriftAndCapacityOverflow() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handles = makeFakeHandles(for: plan)
        let driftExecutor = try DistributedWorkerFrameExecutor(plan: plan, handle: handles[2])
        let overflowExecutor = try DistributedWorkerFrameExecutor(plan: plan, handle: handles[1])
        let driftPacket = try hiddenPacket(
            requestID: "req-drift",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 1, upperBound: 2),
            stepIndex: 0,
            fill: 7)
        let overflowPacket = try hiddenPacket(
            requestID: "req-overflow",
            source: "embed",
            destination: "layers-0-16",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
            stepIndex: 0,
            fill: 8)

        _ = try await driftExecutor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(
                requestID: "req-drift", kvCapacity: 4))))
        await XCTAssertThrowsErrorAsync(
            try await driftExecutor.process(DistributedWorkerWireFrame(
                message: .forward(DistributedStageForwardFrame(
                    stageID: "layers-16-32",
                    requestID: "req-drift",
                    stepIndex: 0,
                    positionRange: DistributedSequenceRange(lowerBound: 1, upperBound: 2),
                    positionIDs: [1],
                    hiddenState: driftPacket.metadata)),
                payload: driftPacket.payload))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput(
                    "position_range lower_bound 1 does not match processed_token_count 0"))
        }

        _ = try await overflowExecutor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(
                requestID: "req-overflow", kvCapacity: 1))))
        await XCTAssertThrowsErrorAsync(
            try await overflowExecutor.process(DistributedWorkerWireFrame(
                message: .forward(DistributedStageForwardFrame(
                    stageID: "layers-0-16",
                    requestID: "req-overflow",
                    stepIndex: 0,
                    positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
                    positionIDs: [0, 1],
                    hiddenState: overflowPacket.metadata)),
                payload: overflowPacket.payload))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("position_range upper_bound 2 exceeds kv_capacity 1"))
        }
        XCTAssertTrue(handles[1].inputs.isEmpty)
        XCTAssertTrue(handles[2].inputs.isEmpty)
    }

    func testWorkerFrameExecutorResetRewindsRequestState() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 7)
        let request = DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                hiddenState: packet.metadata)),
            payload: packet.payload)

        _ = try await executor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 4))))
        _ = try await executor.process(request)
        _ = try await executor.process(DistributedWorkerWireFrame(
            message: .reset(DistributedRequestControl(
                requestID: "req-1", stageID: "layers-16-32"))))
        _ = try await executor.process(request)

        XCTAssertEqual(handle.inputs.count, 2)
        XCTAssertEqual(handle.resetRequests, ["req-1"])
    }

    func testWorkerFrameExecutorFreeDropsRequestState() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 7)

        _ = try await executor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 4))))
        _ = try await executor.process(DistributedWorkerWireFrame(
            message: .free(DistributedRequestControl(
                requestID: "req-1", stageID: "layers-16-32"))))

        await XCTAssertThrowsErrorAsync(
            try await executor.process(DistributedWorkerWireFrame(
                message: .free(DistributedRequestControl(
                    requestID: "req-1", stageID: "layers-16-32"))))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("request_id req-1 is not allocated"))
        }
        await XCTAssertThrowsErrorAsync(
            try await executor.process(DistributedWorkerWireFrame(
                message: .forward(DistributedStageForwardFrame(
                    stageID: "layers-16-32",
                    requestID: "req-1",
                    stepIndex: 0,
                    positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                    positionIDs: [0],
                    hiddenState: packet.metadata)),
                payload: packet.payload))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("request_id req-1 is not allocated"))
        }
        XCTAssertTrue(handle.inputs.isEmpty)
        XCTAssertEqual(handle.freeRequests, ["req-1"])
    }

    func testWorkerFrameExecutorRejectsFreeBeforeAllocate() async throws {
        let plan = makePlan()
        let handle = makeFakeHandles(for: plan)[1]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)

        await XCTAssertThrowsErrorAsync(
            try await executor.process(DistributedWorkerWireFrame(
                message: .free(DistributedRequestControl(
                    requestID: "req-1", stageID: handle.descriptor.id))))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("request_id req-1 is not allocated"))
        }
        XCTAssertTrue(handle.freeRequests.isEmpty)
    }

    func testWorkerFrameExecutorRejectsWrongStageFrame() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "embed",
            destination: "layers-0-16",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 7)
        let request = DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: "layers-0-16",
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                hiddenState: packet.metadata)),
            payload: packet.payload)

        await XCTAssertThrowsErrorAsync(try await executor.process(request)) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame(
                    "frame stage_id layers-0-16 does not match worker stage layers-16-32"))
        }
        XCTAssertTrue(handle.inputs.isEmpty)
    }

    func testWorkerFrameExecutorRejectsCoordinatorOnlyFrames() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[1]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let finalStage = try XCTUnwrap(plan.stage(id: "final"))
        let messages: [(DistributedWorkerMessage, String)] = [
            (
                .hello(DistributedWorkerHello(
                    stage: handle.descriptor,
                    hiddenSize: 2,
                    boundaryScalarType: .float16,
                    planIntegrityHash: try plan.integrityHash())),
                "hello"
            ),
            (
                .helloAck(DistributedWorkerHelloAck(
                    accepted: true,
                    stageID: handle.descriptor.id,
                    planIntegrityHash: try plan.integrityHash())),
                "hello_ack"
            ),
            (
                .forwardResult(DistributedStageForwardResultFrame(
                    stageID: finalStage.id,
                    requestID: "req-1",
                    stepIndex: 0,
                    hiddenState: nil,
                    tokenID: 42)),
                "forward_result"
            ),
            (
                .error(DistributedWorkerErrorFrame(
                    code: "bad_request",
                    detail: "coordinator-side error",
                    stageID: handle.descriptor.id)),
                "error"
            ),
        ]

        for (message, kind) in messages {
            await XCTAssertThrowsErrorAsync(
                try await executor.process(DistributedWorkerWireFrame(message: message))
            ) { error in
                XCTAssertEqual(
                    error as? DistributedStageExecutionError,
                    .invalidControlFrame("worker cannot process \(kind) frame"))
            }
        }
        XCTAssertTrue(handle.inputs.isEmpty)
        XCTAssertTrue(handle.allocatedRequests.isEmpty)
        XCTAssertTrue(handle.resetRequests.isEmpty)
        XCTAssertTrue(handle.freeRequests.isEmpty)
    }

    func testRemoteStageHandleRoundTripsThroughWorkerExecutor() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handles = makeFakeHandles(for: plan)
        let workerExecutor = try DistributedWorkerFrameExecutor(plan: plan, handle: handles[1])
        let loopback = DistributedLoopbackWorkerTransport(
            executor: workerExecutor,
            requestChunkSize: 5,
            responseChunkSize: 3)
        let remote = try DistributedRemoteStageHandle(
            plan: plan,
            descriptor: handles[1].descriptor
        ) { request in
            try await loopback.roundTrip(request)
        }
        let pipeline = try DistributedSameMachinePipeline(
            plan: plan,
            stages: [handles[0], remote, handles[2], handles[3]])

        try await pipeline.allocate(requestID: "req-remote", kvCapacity: 16)
        let output = try await pipeline.forward(
            requestID: "req-remote",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
            tokenIDs: [1, 2, 3])

        XCTAssertEqual(output.tokenID, 42)
        XCTAssertEqual(handles[1].allocatedRequests, ["req-remote"])
        XCTAssertEqual(handles[1].inputs.count, 1)
        XCTAssertEqual(handles[1].inputs[0].positionIDs, [0, 1, 2])
        XCTAssertEqual(handles[1].inputs[0].hiddenState?.metadata.sourceStageID, "embed")
        XCTAssertEqual(handles[2].inputs[0].hiddenState?.metadata.sourceStageID, "layers-0-16")
    }

    func testLoopbackWorkerTransportReturnsNilForControlFrames() async throws {
        let plan = makePlan()
        let handle = makeFakeHandles(for: plan)[1]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let loopback = DistributedLoopbackWorkerTransport(executor: executor, requestChunkSize: 1)

        let response = try await loopback.roundTrip(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 128))))

        XCTAssertNil(response)
        XCTAssertEqual(handle.allocatedRequests, ["req-1"])
    }

    func testLoopbackWorkerTransportRoundTripsHandshakeWithChunking() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handles = makeFakeHandles(for: plan)
        var coordinator = try DistributedWorkerHandshakeCoordinator(plan: plan)

        for handle in handles {
            let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
            let loopback = DistributedLoopbackWorkerTransport(
                executor: executor,
                requestChunkSize: 3,
                responseChunkSize: 2)
            let response = try loopback.handshake(
                with: &coordinator,
                cacheContract: "stateful",
                freeMemoryBytes: 2048,
                computeUnit: "all",
                labels: ["stage": handle.descriptor.id])

            XCTAssertNoThrow(try response.validate(against: plan))
            guard case .helloAck(let ack) = response.message else {
                return XCTFail("expected hello_ack")
            }
            XCTAssertTrue(ack.accepted)
            XCTAssertEqual(ack.stageID, handle.descriptor.id)
            XCTAssertEqual(ack.planIntegrityHash, try plan.integrityHash())
            XCTAssertNil(ack.reason)
            XCTAssertTrue(response.payload.isEmpty)
        }

        XCTAssertTrue(coordinator.isReady)
        XCTAssertNoThrow(try coordinator.requireReady())
    }

    func testLoopbackWorkerTransportRejectsDuplicateHandshake() throws {
        let plan = makePlan()
        let handle = makeFakeHandles(for: plan)[1]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let loopback = DistributedLoopbackWorkerTransport(
            executor: executor,
            requestChunkSize: 1,
            responseChunkSize: 1)
        var coordinator = try DistributedWorkerHandshakeCoordinator(plan: plan)

        _ = try loopback.handshake(with: &coordinator)
        let response = try loopback.handshake(with: &coordinator)

        guard case .helloAck(let ack) = response.message else {
            return XCTFail("expected hello_ack")
        }
        XCTAssertFalse(ack.accepted)
        XCTAssertEqual(ack.stageID, handle.descriptor.id)
        XCTAssertEqual(ack.reason, "stage already claimed")
        XCTAssertNil(ack.planIntegrityHash)
        XCTAssertEqual(coordinator.claimedStages, [handle.descriptor.id])
    }

    func testLoopbackWorkerTransportRoundTripsForwardResponseWithChunking() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let loopback = DistributedLoopbackWorkerTransport(
            executor: executor,
            requestChunkSize: 7,
            responseChunkSize: 2)
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
            stepIndex: 0,
            fill: 9)
        let request = DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
                positionIDs: [0, 1, 2],
                hiddenState: packet.metadata)),
            payload: packet.payload)

        let allocateResponse = try await loopback.roundTrip(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 8))))
        XCTAssertNil(allocateResponse)
        let maybeResponse = try await loopback.roundTrip(request)
        let response = try XCTUnwrap(maybeResponse)

        XCTAssertNoThrow(try response.validate(against: plan))
        guard case .forwardResult(let result) = response.message else {
            return XCTFail("expected forward_result frame")
        }
        XCTAssertEqual(result.stageID, "layers-16-32")
        XCTAssertEqual(result.requestID, "req-1")
        XCTAssertEqual(result.stepIndex, 0)
        XCTAssertEqual(response.payload.count, packet.payload.count)
        XCTAssertEqual(handle.inputs.first?.positionIDs, [0, 1, 2])
    }

    func testLoopbackWorkerTransportReturnsErrorFrameForWorkerRejection() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let loopback = DistributedLoopbackWorkerTransport(
            executor: executor,
            requestChunkSize: 4,
            responseChunkSize: 3)
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 9)

        let maybeResponse = try await loopback.roundTrip(DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: "layers-16-32",
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                hiddenState: packet.metadata)),
            payload: packet.payload))
        let response = try XCTUnwrap(maybeResponse)

        XCTAssertNoThrow(try response.validate(against: plan))
        guard case .error(let error) = response.message else {
            return XCTFail("expected error frame")
        }
        XCTAssertEqual(error.code, "invalid_forward_input")
        XCTAssertEqual(error.detail, "request_id req-1 is not allocated")
        XCTAssertEqual(error.requestID, "req-1")
        XCTAssertEqual(error.stageID, "layers-16-32")
        XCTAssertTrue(response.payload.isEmpty)
        XCTAssertTrue(handle.inputs.isEmpty)
    }

    func testLoopbackWorkerTransportReturnsRuntimeValidationErrorFrame() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan, badFirstRoute: true)[0]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let loopback = DistributedLoopbackWorkerTransport(executor: executor)

        let allocateResponse = try await loopback.roundTrip(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 8))))
        XCTAssertNil(allocateResponse)
        let maybeResponse = try await loopback.roundTrip(DistributedWorkerWireFrame(
            message: .forward(DistributedStageForwardFrame(
                stageID: "embed",
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                tokenIDs: [7]))))
        let response = try XCTUnwrap(maybeResponse)

        XCTAssertNoThrow(try response.validate(against: plan))
        guard case .error(let error) = response.message else {
            return XCTFail("expected error frame")
        }
        XCTAssertEqual(error.code, "runtime_validation")
        XCTAssertEqual(
            error.detail,
            "Hidden-state packet route is not adjacent: embed -> final")
        XCTAssertEqual(error.requestID, "req-1")
        XCTAssertEqual(error.stageID, "embed")
        XCTAssertTrue(response.payload.isEmpty)
        XCTAssertEqual(handle.inputs.count, 1)
    }

    func testRemoteStageHandleSurfacesLoopbackWorkerErrorFrame() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let handle = makeFakeHandles(for: plan)[2]
        let executor = try DistributedWorkerFrameExecutor(plan: plan, handle: handle)
        let loopback = DistributedLoopbackWorkerTransport(executor: executor)
        let remote = try DistributedRemoteStageHandle(
            plan: plan,
            descriptor: handle.descriptor
        ) { request in
            try await loopback.roundTrip(request)
        }
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 9)

        await XCTAssertThrowsErrorAsync(
            try await remote.forward(DistributedStageForwardInput(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                hiddenState: packet))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame(
                    "forward worker error invalid_forward_input: request_id req-1 is not allocated"))
        }
        XCTAssertTrue(handle.inputs.isEmpty)
    }

    func testRemoteStageHandleRejectsMissingForwardResponse() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let remote = try DistributedRemoteStageHandle(
            plan: plan,
            descriptor: try XCTUnwrap(plan.stage(id: "embed"))
        ) { _ in nil }

        await XCTAssertThrowsErrorAsync(
            try await remote.forward(DistributedStageForwardInput(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                tokenIDs: [7]))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidStageOutput("forward response is missing"))
        }
    }

    func testRemoteStageHandlePropagatesWorkerErrorResponse() async throws {
        let plan = makePlan()
        let remote = try DistributedRemoteStageHandle(
            plan: plan,
            descriptor: try XCTUnwrap(plan.stage(id: "layers-0-16"))
        ) { _ in
            DistributedWorkerWireFrame(message: .error(DistributedWorkerErrorFrame(
                code: "bad_request",
                detail: "unexpected response",
                requestID: "req-1",
                stageID: "layers-0-16")))
        }

        await XCTAssertThrowsErrorAsync(
            try await remote.allocate(DistributedStageAllocation(
                requestID: "req-1", kvCapacity: 16))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("alloc worker error bad_request: unexpected response"))
        }
    }

    func testRemoteStageHandlePropagatesForwardWorkerErrorResponse() async throws {
        let plan = makePlan()
        let remote = try DistributedRemoteStageHandle(
            plan: plan,
            descriptor: try XCTUnwrap(plan.stage(id: "embed"))
        ) { _ in
            DistributedWorkerWireFrame(message: .error(DistributedWorkerErrorFrame(
                code: "kv_exhausted",
                detail: "capacity exceeded",
                requestID: "req-1",
                stageID: "embed")))
        }

        await XCTAssertThrowsErrorAsync(
            try await remote.forward(DistributedStageForwardInput(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                tokenIDs: [7]))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("forward worker error kv_exhausted: capacity exceeded"))
        }
    }

    func testRemoteStageHandleRejectsMismatchedWorkerErrorEnvelope() async throws {
        let plan = makePlan()
        let wrongRequest = try DistributedRemoteStageHandle(
            plan: plan,
            descriptor: try XCTUnwrap(plan.stage(id: "layers-0-16"))
        ) { _ in
            DistributedWorkerWireFrame(message: .error(DistributedWorkerErrorFrame(
                code: "bad_request",
                detail: "wrong request",
                requestID: "other-req",
                stageID: "layers-0-16")))
        }
        let wrongStage = try DistributedRemoteStageHandle(
            plan: plan,
            descriptor: try XCTUnwrap(plan.stage(id: "layers-0-16"))
        ) { _ in
            DistributedWorkerWireFrame(message: .error(DistributedWorkerErrorFrame(
                code: "bad_request",
                detail: "wrong stage",
                requestID: "req-1",
                stageID: "layers-16-32")))
        }

        await XCTAssertThrowsErrorAsync(
            try await wrongRequest.allocate(DistributedStageAllocation(
                requestID: "req-1", kvCapacity: 16))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame(
                    "alloc worker error request_id other-req does not match request req-1"))
        }

        await XCTAssertThrowsErrorAsync(
            try await wrongStage.allocate(DistributedStageAllocation(
                requestID: "req-1", kvCapacity: 16))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame(
                    "alloc worker error stage_id layers-16-32 does not match remote stage layers-0-16"))
        }
    }

    func testRemoteStageHandleRejectsMismatchedForwardResponseRequest() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let descriptor = try XCTUnwrap(plan.stage(id: "final"))
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-16-32",
            destination: "final",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 7)
        let remote = try DistributedRemoteStageHandle(plan: plan, descriptor: descriptor) { _ in
            DistributedWorkerWireFrame(message: .forwardResult(DistributedStageForwardResultFrame(
                stageID: descriptor.id,
                requestID: "other-req",
                stepIndex: 0,
                hiddenState: nil,
                tokenID: 42)))
        }

        await XCTAssertThrowsErrorAsync(
            try await remote.forward(DistributedStageForwardInput(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                hiddenState: packet))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidStageOutput("response request_id does not match request"))
        }
    }

    func testRemoteStageHandleRejectsMismatchedForwardResponseStep() async throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let descriptor = try XCTUnwrap(plan.stage(id: "final"))
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-16-32",
            destination: "final",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 7)
        let remote = try DistributedRemoteStageHandle(plan: plan, descriptor: descriptor) { _ in
            DistributedWorkerWireFrame(message: .forwardResult(DistributedStageForwardResultFrame(
                stageID: descriptor.id,
                requestID: "req-1",
                stepIndex: 1,
                hiddenState: nil,
                tokenID: 42)))
        }

        await XCTAssertThrowsErrorAsync(
            try await remote.forward(DistributedStageForwardInput(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                positionIDs: [0],
                hiddenState: packet))
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidStageOutput("response step_index does not match request"))
        }
    }

    func testWorkerHelloValidatesPlanIntegrityAndStageClaim() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 4096], scalarType: .float16))
        let hello = DistributedWorkerHello(
            stage: try XCTUnwrap(plan.stage(id: "layers-0-16")),
            hiddenSize: 4096,
            boundaryScalarType: .float16,
            cacheContract: "stateful",
            planIntegrityHash: try plan.integrityHash())

        XCTAssertNoThrow(try hello.validate(against: plan))
    }

    func testWorkerHelloRejectsPlanIntegrityMismatch() throws {
        let plan = makePlan()
        let hello = DistributedWorkerHello(
            stage: try XCTUnwrap(plan.stage(id: "layers-0-16")),
            planIntegrityHash: "stale-plan")

        XCTAssertThrowsError(
            try hello.validate(against: plan, expectedPlanIntegrityHash: "current-plan")
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWorkerHello("plan_integrity_hash mismatch"))
        }
    }

    func testWorkerHelloRejectsStageDescriptorMismatch() throws {
        let plan = makePlan()
        let hello = DistributedWorkerHello(
            stage: stage(
                "layers-0-16", .transformerLayers,
                range: DistributedLayerRange(lowerBound: 0, upperBound: 8),
                assetName: "layers_0_16", workerID: "worker-a"),
            planIntegrityHash: "plan-hash")

        XCTAssertThrowsError(
            try hello.validate(against: plan, expectedPlanIntegrityHash: "plan-hash")
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWorkerHello("stage descriptor does not match plan for layers-0-16"))
        }
    }

    func testWorkerHelloRejectsBoundaryContractMismatch() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 4096], scalarType: .float16))
        let wrongHiddenSize = DistributedWorkerHello(
            stage: try XCTUnwrap(plan.stage(id: "layers-0-16")),
            hiddenSize: 2048,
            boundaryScalarType: .float16,
            planIntegrityHash: "plan-hash")
        let wrongScalar = DistributedWorkerHello(
            stage: try XCTUnwrap(plan.stage(id: "layers-0-16")),
            hiddenSize: 4096,
            boundaryScalarType: .float32,
            planIntegrityHash: "plan-hash")

        XCTAssertThrowsError(
            try wrongHiddenSize.validate(against: plan, expectedPlanIntegrityHash: "plan-hash")
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWorkerHello(
                    "hidden_size 2048 does not match boundary tensor hidden size 4096"))
        }
        XCTAssertThrowsError(
            try wrongScalar.validate(against: plan, expectedPlanIntegrityHash: "plan-hash")
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidWorkerHello(
                    "boundary_scalar_type float32 does not match boundary tensor float16"))
        }
    }

    func testAllocationFrameValidatesRequestAndCapacity() throws {
        XCTAssertNoThrow(try DistributedStageAllocation(requestID: "req-1", kvCapacity: 128).validate())

        XCTAssertThrowsError(
            try DistributedStageAllocation(requestID: " ", kvCapacity: 128).validate()
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("request_id is empty"))
        }
        XCTAssertThrowsError(
            try DistributedStageAllocation(requestID: "req-1", kvCapacity: 0).validate()
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("kv_capacity must be positive"))
        }
    }

    func testRequestControlFrameValidatesRequestAndOptionalStage() throws {
        let plan = makePlan()
        XCTAssertNoThrow(
            try DistributedRequestControl(requestID: "req-1").validate(against: plan))
        XCTAssertNoThrow(
            try DistributedRequestControl(requestID: "req-1", stageID: "layers-0-16")
                .validate(against: plan))

        XCTAssertThrowsError(
            try DistributedRequestControl(requestID: "").validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("request_id is empty"))
        }
        XCTAssertThrowsError(
            try DistributedRequestControl(requestID: "req-1", stageID: "missing")
                .validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("unknown stage_id missing"))
        }
    }

    func testHelloAckFrameValidatesPlanHashAndRejectedReason() throws {
        let plan = makePlan()
        let accepted = DistributedWorkerHelloAck(
            accepted: true,
            stageID: "layers-0-16",
            planIntegrityHash: try plan.integrityHash())
        let rejected = DistributedWorkerHelloAck(
            accepted: false,
            stageID: "layers-0-16",
            reason: "plan mismatch")

        XCTAssertNoThrow(try accepted.validate(against: plan))
        XCTAssertNoThrow(try rejected.validate(against: plan))

        XCTAssertThrowsError(
            try DistributedWorkerHelloAck(
                accepted: true, stageID: "layers-0-16", planIntegrityHash: "old")
                .validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("plan_integrity_hash mismatch"))
        }
        XCTAssertThrowsError(
            try DistributedWorkerHelloAck(accepted: false, stageID: "layers-0-16")
                .validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("rejected hello_ack needs a reason"))
        }
    }

    func testErrorFrameValidatesCodeDetailAndStage() throws {
        let plan = makePlan()
        XCTAssertNoThrow(
            try DistributedWorkerErrorFrame(
                code: "plan_mismatch",
                detail: "worker plan hash did not match",
                requestID: "req-1",
                stageID: "layers-0-16")
                .validate(against: plan))

        XCTAssertThrowsError(
            try DistributedWorkerErrorFrame(code: " ", detail: "bad").validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("error code is empty"))
        }
        XCTAssertThrowsError(
            try DistributedWorkerErrorFrame(code: "bad", detail: "", stageID: "missing")
                .validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("error detail is empty"))
        }
        XCTAssertThrowsError(
            try DistributedWorkerErrorFrame(code: "bad", detail: "bad", stageID: "missing")
                .validate(against: plan)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("unknown stage_id missing"))
        }
    }

    func testWorkerMessageValidateDispatchesControlAndForwardFrames() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let hidden = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 0,
            fill: 1)
        let messages: [DistributedWorkerMessage] = [
            .hello(
                DistributedWorkerHello(
                    stage: try XCTUnwrap(plan.stage(id: "layers-0-16")),
                    hiddenSize: 2,
                    boundaryScalarType: .float16,
                    planIntegrityHash: try plan.integrityHash())),
            .helloAck(
                DistributedWorkerHelloAck(
                    accepted: true,
                    stageID: "layers-0-16",
                    planIntegrityHash: try plan.integrityHash())),
            .allocate(DistributedStageAllocation(requestID: "req-1", kvCapacity: 128)),
            .forward(
                DistributedStageForwardFrame(
                    stageID: "layers-16-32",
                    requestID: "req-1",
                    stepIndex: 0,
                    positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                    positionIDs: [0],
                    hiddenState: hidden.metadata)),
            .forwardResult(
                DistributedStageForwardResultFrame(
                    stageID: "layers-0-16",
                    requestID: "req-1",
                    stepIndex: 0,
                    hiddenState: hidden.metadata)),
            .reset(DistributedRequestControl(requestID: "req-1", stageID: "layers-0-16")),
            .free(DistributedRequestControl(requestID: "req-1", stageID: "layers-0-16")),
            .error(
                DistributedWorkerErrorFrame(
                    code: "bad_request",
                    detail: "bad frame",
                    requestID: "req-1",
                    stageID: "layers-0-16")),
        ]

        for message in messages {
            XCTAssertNoThrow(try message.validate(against: plan))
        }
    }

    func testWorkerMessageValidateRejectsBadFrame() throws {
        let plan = makePlan()
        let message = DistributedWorkerMessage.allocate(
            DistributedStageAllocation(requestID: "req-1", kvCapacity: 0))

        XCTAssertThrowsError(try message.validate(against: plan)) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("kv_capacity must be positive"))
        }
    }

    func testForwardFrameValidatesRoleSpecificInputs() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let firstStage = DistributedStageForwardFrame(
            stageID: "embed",
            requestID: "req-1",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
            positionIDs: [0, 1, 2],
            tokenIDs: [10, 11, 12])
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "embed",
            destination: "layers-0-16",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
            stepIndex: 0,
            fill: 7)
        let middleStage = DistributedStageForwardFrame(
            stageID: "layers-0-16",
            requestID: "req-1",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 3),
            positionIDs: [0, 1, 2],
            hiddenState: packet.metadata)

        XCTAssertNoThrow(try firstStage.validate(against: plan))
        XCTAssertNoThrow(try middleStage.validate(against: plan))
    }

    func testForwardFrameRejectsWrongPayloadForRole() throws {
        let plan = makePlan()
        let badFirstStage = DistributedStageForwardFrame(
            stageID: "embed",
            requestID: "req-1",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            positionIDs: [0])

        XCTAssertThrowsError(try badFirstStage.validate(against: plan)) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("token_ids count must match position_range"))
        }

        let badMiddleStage = DistributedStageForwardFrame(
            stageID: "layers-0-16",
            requestID: "req-1",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            positionIDs: [0],
            tokenIDs: [7])

        XCTAssertThrowsError(try badMiddleStage.validate(against: plan)) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("transformer_layers stage must not receive token_ids"))
        }
    }

    func testForwardResultFrameValidatesStageOutput() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let packet = try hiddenPacket(
            requestID: "req-1",
            source: "layers-0-16",
            destination: "layers-16-32",
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            stepIndex: 1,
            fill: 3)
        let middleOutput = DistributedStageForwardResultFrame(
            stageID: "layers-0-16",
            requestID: "req-1",
            stepIndex: 1,
            hiddenState: packet.metadata)
        let finalOutput = DistributedStageForwardResultFrame(
            stageID: "final",
            requestID: "req-1",
            stepIndex: 1,
            tokenID: 42)

        XCTAssertNoThrow(try middleOutput.validate(against: plan))
        XCTAssertNoThrow(try finalOutput.validate(against: plan))
    }

    func testForwardResultFrameRejectsInvalidFinalOutput() throws {
        let plan = makePlan()
        let badFinal = DistributedStageForwardResultFrame(
            stageID: "final",
            requestID: "req-1",
            stepIndex: 0)

        XCTAssertThrowsError(try badFinal.validate(against: plan)) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidStageOutput("final stage must return a token id"))
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

    func testStagePlanIntegrityHashIsStableAndChangesWithPlan() throws {
        let plan = makePlan(
            boundaryTensor: DistributedBoundaryTensorSpec(
                name: "hidden_states", shape: [1, -1, 2], scalarType: .float16))
        let data = try JSONEncoder().encode(plan)
        let decoded = try JSONDecoder().decode(DistributedStagePlan.self, from: data)
        let changedPlan = DistributedStagePlan(
            modelName: plan.modelName,
            totalLayerCount: plan.totalLayerCount,
            stages: plan.stages.map { descriptor in
                descriptor.id == "layers-16-32"
                    ? DistributedStageDescriptor(
                        id: descriptor.id,
                        role: descriptor.role,
                        layerRange: descriptor.layerRange,
                        assetName: "layers_16_32_v2",
                        workerID: descriptor.workerID)
                    : descriptor
            },
            workers: plan.workers,
            boundaryTensor: plan.boundaryTensor)

        XCTAssertEqual(try plan.integrityHash(), try decoded.integrityHash())
        XCTAssertEqual(try plan.integrityHash().count, 64)
        XCTAssertNotEqual(try plan.integrityHash(), try changedPlan.integrityHash())
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

    func testStageHandleContextRequiresExistingAssetURL() throws {
        let baseURL = try makeTemporaryManifestBaseURL()
        let stageURL = baseURL.appendingPathComponent("stages/00-embed.aimodel")
        try FileManager.default.createDirectory(
            at: stageURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: baseURL) }

        let manifest = try DistributedStageManifest.decode(
            from: Data(stageManifestJSON(modelKey: "model", includeTotalLayerCount: true).utf8),
            baseURL: baseURL)
        let stage = manifest.stages[0]
        let context = try XCTUnwrap(makeContext(manifest: manifest, stage: stage))

        XCTAssertEqual(try context.requireExistingAssetURL().path, stageURL.standardizedFileURL.path)
    }

    func testStageHandleContextRejectsMissingAssetURL() throws {
        let baseURL = try makeTemporaryManifestBaseURL()
        defer { try? FileManager.default.removeItem(at: baseURL) }

        let manifest = try DistributedStageManifest.decode(
            from: Data(stageManifestJSON(modelKey: "model", includeTotalLayerCount: true).utf8),
            baseURL: baseURL)
        let stage = manifest.stages[0]
        let context = try XCTUnwrap(makeContext(manifest: manifest, stage: stage))
        let expectedPath = baseURL.appendingPathComponent("stages/00-embed.aimodel")
            .standardizedFileURL.path

        XCTAssertThrowsError(try context.requireExistingAssetURL()) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .missingStageAsset(stageID: "embed", path: expectedPath))
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

    func testSameMachinePipelineRejectsInvalidPositionRangeBeforeForwarding() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        await XCTAssertThrowsErrorAsync(
            try await pipeline.forward(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 2, upperBound: 2),
                tokenIDs: [7])
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("position_range is invalid"))
        }
        XCTAssertTrue(handles.allSatisfy { $0.inputs.isEmpty })
    }

    func testSameMachinePipelineRequiresAllocationBeforeForwarding() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        await XCTAssertThrowsErrorAsync(
            try await pipeline.forward(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                tokenIDs: [7])
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("request_id req-1 is not allocated"))
        }
        XCTAssertTrue(handles.allSatisfy { $0.inputs.isEmpty })
    }

    func testSameMachinePipelineRejectsDuplicateAllocationBeforeForwarding() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        try await pipeline.allocate(requestID: "req-1", kvCapacity: 4)
        await XCTAssertThrowsErrorAsync(
            try await pipeline.allocate(requestID: "req-1", kvCapacity: 4)
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame("request_id req-1 is already allocated"))
        }
        XCTAssertEqual(handles.map(\.allocatedRequests), [["req-1"], ["req-1"], ["req-1"], ["req-1"]])
    }

    func testSameMachinePipelineTracksForwardStepOrderAndPositions() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        try await pipeline.allocate(requestID: "req-1", kvCapacity: 4)
        _ = try await pipeline.forward(
            requestID: "req-1",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
            tokenIDs: [1, 2])

        await XCTAssertThrowsErrorAsync(
            try await pipeline.forward(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 2, upperBound: 3),
                tokenIDs: [3])
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("step_index 0 does not match expected 1"))
        }

        await XCTAssertThrowsErrorAsync(
            try await pipeline.forward(
                requestID: "req-1",
                stepIndex: 1,
                positionRange: DistributedSequenceRange(lowerBound: 3, upperBound: 4),
                tokenIDs: [4])
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput(
                    "position_range lower_bound 3 does not match processed_token_count 2"))
        }
        XCTAssertTrue(handles.allSatisfy { $0.inputs.count == 1 })

        let output = try await pipeline.forward(
            requestID: "req-1",
            stepIndex: 1,
            positionRange: DistributedSequenceRange(lowerBound: 2, upperBound: 3),
            tokenIDs: [3])
        XCTAssertEqual(output.tokenID, 42)
        XCTAssertTrue(handles.allSatisfy { $0.inputs.count == 2 })
    }

    func testSameMachinePipelineResetRewindsRequestTracker() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        try await pipeline.allocate(requestID: "req-1", kvCapacity: 4)
        _ = try await pipeline.forward(
            requestID: "req-1",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
            tokenIDs: [1, 2])
        try await pipeline.reset(requestID: "req-1")
        let output = try await pipeline.forward(
            requestID: "req-1",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            tokenIDs: [3])

        XCTAssertEqual(output.tokenID, 42)
        XCTAssertEqual(handles.map(\.resetRequests), [["req-1"], ["req-1"], ["req-1"], ["req-1"]])
        XCTAssertTrue(handles.allSatisfy { $0.inputs.count == 2 })
    }

    func testSameMachinePipelineFreeClearsRequestTracker() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        try await pipeline.allocate(requestID: "req-1", kvCapacity: 4)
        await pipeline.free(requestID: "req-1")

        await XCTAssertThrowsErrorAsync(
            try await pipeline.forward(
                requestID: "req-1",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                tokenIDs: [7])
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("request_id req-1 is not allocated"))
        }
        XCTAssertEqual(handles.map(\.freeRequests), [["req-1"], ["req-1"], ["req-1"], ["req-1"]])
    }

    func testStagedEngineGeneratesPrefillAndDecodeTokenSteps() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)
        let engine = try DistributedStagedEngine(pipeline: pipeline, maxContextLength: 16)

        let result = try await engine.generate(
            promptTokens: [10, 11],
            options: DistributedStagedGenerationOptions(maxTokens: 3),
            requestID: "req-engine")

        XCTAssertEqual(result.generatedTokenIDs, [42, 42, 42])
        XCTAssertEqual(result.generatedTokenCount, 3)
        XCTAssertEqual(result.promptTokenCount, 2)
        XCTAssertEqual(result.stopReason, .maxTokens)
        XCTAssertEqual(result.kvCapacity, 13)
        XCTAssertTrue(handles.allSatisfy { $0.inputs.count == 3 })
        XCTAssertEqual(handles[0].inputs.map(\.stepIndex), [0, 1, 2])
        XCTAssertEqual(
            handles[0].inputs.map(\.positionRange),
            [
                DistributedSequenceRange(lowerBound: 0, upperBound: 2),
                DistributedSequenceRange(lowerBound: 2, upperBound: 3),
                DistributedSequenceRange(lowerBound: 3, upperBound: 4),
            ])
        XCTAssertEqual(handles[0].inputs.map(\.positionIDs), [[0, 1], [2], [3]])
        XCTAssertEqual(handles[0].inputs.map(\.tokenIDs), [[10, 11], [42], [42]])
        XCTAssertEqual(handles.map(\.freeRequests), [
            ["req-engine"], ["req-engine"], ["req-engine"], ["req-engine"],
        ])
    }

    func testStagedEngineStopsBeforeAppendingStopToken() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)
        let engine = try DistributedStagedEngine(pipeline: pipeline, maxContextLength: 16)

        let result = try await engine.generate(
            promptTokens: [10, 11],
            options: DistributedStagedGenerationOptions(maxTokens: 3, stopTokenIDs: [42]),
            requestID: "req-stop")

        XCTAssertEqual(result.generatedTokenIDs, [])
        XCTAssertEqual(result.stopReason, .eos)
        XCTAssertTrue(handles.allSatisfy { $0.inputs.count == 1 })
        XCTAssertEqual(handles.map(\.freeRequests), [
            ["req-stop"], ["req-stop"], ["req-stop"], ["req-stop"],
        ])
    }

    func testStagedEngineStopsAtContextLimitBeforeAppendingToken() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)
        let engine = try DistributedStagedEngine(pipeline: pipeline, maxContextLength: 2)

        let result = try await engine.generate(
            promptTokens: [10, 11],
            options: DistributedStagedGenerationOptions(maxTokens: 3),
            requestID: "req-context")

        XCTAssertEqual(result.generatedTokenIDs, [])
        XCTAssertEqual(result.stopReason, .contextLimit)
        XCTAssertEqual(result.kvCapacity, 2)
        XCTAssertTrue(handles.allSatisfy { $0.inputs.count == 1 })
        XCTAssertEqual(handles.map(\.freeRequests), [
            ["req-context"], ["req-context"], ["req-context"], ["req-context"],
        ])
    }

    func testStagedEngineRejectsEmptyPromptBeforeAllocation() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)
        let engine = try DistributedStagedEngine(pipeline: pipeline, maxContextLength: 16)

        await XCTAssertThrowsErrorAsync(
            try await engine.generate(
                promptTokens: [],
                options: DistributedStagedGenerationOptions(maxTokens: 1),
                requestID: "req-empty")
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("prompt_tokens must be non-empty"))
        }
        XCTAssertTrue(handles.allSatisfy { $0.allocatedRequests.isEmpty })
        XCTAssertTrue(handles.allSatisfy { $0.inputs.isEmpty })
    }

    func testStagedEngineFreesRequestAfterForwardFailure() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)
        let engine = try DistributedStagedEngine(pipeline: pipeline, maxContextLength: 16)

        await XCTAssertThrowsErrorAsync(
            try await engine.generate(
                promptTokens: [10, 11],
                options: DistributedStagedGenerationOptions(maxTokens: 2, kvCapacity: 2),
                requestID: "req-overflow")
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("position_range upper_bound 3 exceeds kv_capacity 2"))
        }
        XCTAssertEqual(handles.map(\.freeRequests), [
            ["req-overflow"], ["req-overflow"], ["req-overflow"], ["req-overflow"],
        ])

        let result = try await engine.generate(
            promptTokens: [12],
            options: DistributedStagedGenerationOptions(maxTokens: 1),
            requestID: "req-overflow")
        XCTAssertEqual(result.generatedTokenIDs, [42])
    }

    func testSameMachinePipelineRejectsEmptyResetRequestIDBeforeForwarding() async throws {
        let plan = makePlan()
        let handles = makeFakeHandles(for: plan)
        let pipeline = try DistributedSameMachinePipeline(plan: plan, stages: handles)

        await XCTAssertThrowsErrorAsync(
            try await pipeline.reset(requestID: " ")
        ) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidForwardInput("request_id is empty"))
        }
        XCTAssertTrue(handles.allSatisfy { $0.resetRequests.isEmpty })
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

    private func makeContext(
        manifest: DistributedStageManifest,
        stage: DistributedStageManifestStage
    ) -> DistributedStageHandleFactoryContext? {
        guard let descriptor = manifest.runtimePlan.stage(id: stage.id) else { return nil }
        return DistributedStageHandleFactoryContext(
            stage: stage, manifest: manifest, descriptor: descriptor)
    }

    private func makeTemporaryManifestBaseURL() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-distributed-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
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
    var resetRequests: [String] = []
    var freeRequests: [String] = []
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

    func reset(requestID: String) async throws {
        resetRequests.append(requestID)
    }

    func free(requestID: String) async {
        freeRequests.append(requestID)
    }
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
