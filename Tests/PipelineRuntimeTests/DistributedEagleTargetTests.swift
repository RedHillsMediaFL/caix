import XCTest

@testable import PipelineRuntime

#if COREAI_RUNTIME
import CoreAI
#endif

final class DistributedEagleTargetTests: XCTestCase {
    func testManifestParsesMinimalEagleTargetContract() throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(manifestJSON(
                eagleTarget: #"{"stage_id":"layers-14-28","sliding_window":1024}"#).utf8))

        XCTAssertEqual(
            manifest.eagleTarget,
            DistributedEagleTargetContract(
                stageID: "layers-14-28",
                slidingWindow: 1_024))
    }

    func testManifestDerivesEagleProducersFromRetainedRichContract() throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(richManifestJSON().utf8))

        let target = try XCTUnwrap(manifest.eagleTarget)
        XCTAssertEqual(target.stageID, "layers-14-28")
        XCTAssertEqual(target.slidingWindow, 1_024)
        XCTAssertEqual(target.finalHiddenStageID, "head")
        XCTAssertEqual(target.finalHiddenTensorName, "hidden")
    }

    func testManifestRejectsIncompleteRichHiddenProducerMapping() {
        let json = richManifestJSON().replacingOccurrences(
            of: #""backbone_hidden": {"stage": "head", "tensor": "hidden"}"#,
            with: #""backbone_hidden": {"stage": "head"}"#)

        XCTAssertThrowsError(try DistributedStageManifest.decode(from: Data(json.utf8)))
    }

    func testManifestRejectsEagleTargetBeforeFinalTransformerStage() {
        XCTAssertThrowsError(try DistributedStageManifest.decode(
            from: Data(manifestJSON(
                eagleTarget: #"{"stage_id":"layers-00-14","sliding_window":1024}"#).utf8))) { error in
            XCTAssertEqual(
                error as? DistributedStageManifestError,
                .invalidManifest(
                    "eagle_target stage_id must name the final transformer_layers stage"))
        }
    }

    func testManifestRejectsNonPositiveEagleSlidingWindow() {
        XCTAssertThrowsError(try DistributedStageManifest.decode(
            from: Data(manifestJSON(
                eagleTarget: #"{"stage_id":"layers-14-28","sliding_window":0}"#).utf8))) { error in
            XCTAssertEqual(
                error as? DistributedStageManifestError,
                .invalidManifest("eagle_target sliding_window must be positive"))
        }
    }

    func testStageIOContractAcceptsCanonicalEagleTargetOutputsOnlyOnProducer() throws {
        let target = DistributedEagleTargetContract(
            stageID: "layers-14-28",
            slidingWindow: 1_024)
        let contract = makeStageIOContract()

        XCTAssertNoThrow(try contract.validate(
            for: transformerDescriptor(id: "layers-14-28", lower: 14, upper: 28),
            boundaryTensor: boundaryTensor,
            eagleTarget: target))

        XCTAssertThrowsError(try contract.validate(
            for: transformerDescriptor(id: "layers-00-14", lower: 0, upper: 14),
            boundaryTensor: boundaryTensor,
            eagleTarget: target))
    }

    func testStageIOContractRejectsMalformedEagleTargetKVDTypeAndShape() throws {
        let target = DistributedEagleTargetContract(
            stageID: "layers-14-28",
            slidingWindow: 1_024)
        let descriptor = transformerDescriptor(id: "layers-14-28", lower: 14, upper: 28)
        let wrongDType = makeStageIOContract(kFullScalarType: .float32)
        let wrongShape = makeStageIOContract(kSlidingShape: [1, 8, -1])

        XCTAssertThrowsError(try wrongDType.validate(
            for: descriptor,
            boundaryTensor: boundaryTensor,
            eagleTarget: target))
        XCTAssertThrowsError(try wrongShape.validate(
            for: descriptor,
            boundaryTensor: boundaryTensor,
            eagleTarget: target))
    }

    func testStageIOContractRequiresPostNormHiddenOnDeclaredHead() throws {
        let target = DistributedEagleTargetContract(
            stageID: "layers-14-28",
            slidingWindow: 1_024,
            finalHiddenStageID: "head",
            finalHiddenTensorName: "hidden")
        let descriptor = DistributedStageDescriptor(
            id: "head",
            role: .finalNormHead,
            layerRange: nil,
            assetName: "head.aimodel")
        let inputs = [
            DistributedStageIOTensor(.hiddenStates, shape: [1, -1, 1_024], scalarType: .float16),
            DistributedStageIOTensor(.positionIDs, shape: [1, -1], scalarType: .int32),
        ]
        let logits = DistributedStageIOTensor(.logits, shape: [1, -1, 256], scalarType: .float16)
        let hidden = DistributedStageIOTensor(.hidden, shape: [1, -1, 1_024], scalarType: .float16)

        XCTAssertNoThrow(try DistributedStageIOContract(
            inputs: inputs,
            outputs: [logits, hidden]
        ).validate(
            for: descriptor,
            boundaryTensor: boundaryTensor,
            vocabSize: 256,
            eagleTarget: target))

        XCTAssertThrowsError(try DistributedStageIOContract(
            inputs: inputs,
            outputs: [logits]
        ).validate(
            for: descriptor,
            boundaryTensor: boundaryTensor,
            vocabSize: 256,
            eagleTarget: target))
    }

    func testCoreAIRank4ReadbackLayoutPreservesHeadSequenceOrder() throws {
        XCTAssertEqual(
            try DistributedCoreAIStageTensorReadbackLayout.rank4Offsets(
                shape: [1, 2, 2, 2],
                strides: [100, 20, 3, 1],
                tensorName: "k_full"),
            [0, 1, 3, 4, 20, 21, 23, 24])
    }

    #if COREAI_RUNTIME
    func testCoreAIReadbackSeparatesHeadHiddenFromTransformerKVColumns() throws {
        let hidden = float16Array(shape: [1, 2, 2], values: [1, 2, 3, 4])
        let fullKey = float16Array(shape: [1, 2, 2, 1], values: [10, 11, 20, 21])
        let fullValue = float16Array(shape: [1, 2, 2, 1], values: [30, 31, 40, 41])
        let slidingKey = float16Array(shape: [1, 2, 2, 1], values: [50, 51, 60, 61])
        let slidingValue = float16Array(shape: [1, 2, 2, 1], values: [70, 71, 80, 81])
        let range = DistributedSequenceRange(lowerBound: 4, upperBound: 6)

        let chunk = try DistributedCoreAIStageNDArrayIO.makeEagleTargetKVChunk(
            fullKey: fullKey,
            fullValue: fullValue,
            slidingKey: slidingKey,
            slidingValue: slidingValue,
            positionRange: range)
        let finalHidden = try DistributedCoreAIStageNDArrayIO.makeEagleTargetFinalHidden(
            hidden,
            positionRange: range,
            tensorName: "hidden")

        XCTAssertEqual(
            finalHidden.float16BitPatterns,
            [Float16(3).bitPattern, Float16(4).bitPattern])
        XCTAssertEqual(
            chunk.fullKey.float16BitPatterns,
            [Float16(10).bitPattern, Float16(11).bitPattern,
             Float16(20).bitPattern, Float16(21).bitPattern])
        XCTAssertEqual(chunk.fullPositionRange, range)
        XCTAssertEqual(chunk.slidingPositionRange, range)
    }

    func testCoreAIReadbackRejectsNonFloat16EagleTensor() throws {
        let wrongKey = NDArray(shape: [1, 1, 1, 1], scalarType: .float32)
        let goodKV = float16Array(shape: [1, 1, 1, 1], values: [1])

        XCTAssertThrowsError(try DistributedCoreAIStageNDArrayIO.makeEagleTargetKVChunk(
            fullKey: wrongKey,
            fullValue: goodKV,
            slidingKey: goodKV,
            slidingValue: goodKV,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1)))
    }
    #endif

    func testKVAccumulatorAppendsStreamedRangesAndKeepsChronologicalSlidingWindow() throws {
        var accumulator = try DistributedEagleTargetKVAccumulator(
            kvCapacity: 8,
            slidingWindow: 3)

        try accumulator.append(artifacts(
            range: 0..<2,
            hidden: [1, 2],
            fullKey: [10, 11, 20, 21],
            fullValue: [30, 31, 40, 41],
            slidingKey: [110, 111, 210, 211],
            slidingValue: [130, 131, 230, 231]))
        try accumulator.append(artifacts(
            range: 2..<4,
            hidden: [3, 4],
            fullKey: [12, 13, 22, 23],
            fullValue: [32, 33, 42, 43],
            slidingKey: [112, 113, 212, 213],
            slidingValue: [132, 133, 232, 233]))

        let snapshot = try XCTUnwrap(accumulator.snapshot)
        XCTAssertEqual(snapshot.fullPositionRange, DistributedSequenceRange(lowerBound: 0, upperBound: 4))
        XCTAssertEqual(snapshot.slidingPositionRange, DistributedSequenceRange(lowerBound: 1, upperBound: 4))
        XCTAssertEqual(snapshot.finalHidden.float16BitPatterns, [3, 4])
        XCTAssertEqual(snapshot.fullKey.shape, [1, 2, 4, 1])
        XCTAssertEqual(snapshot.fullKey.float16BitPatterns, [10, 11, 12, 13, 20, 21, 22, 23])
        XCTAssertEqual(snapshot.fullValue.float16BitPatterns, [30, 31, 32, 33, 40, 41, 42, 43])
        XCTAssertEqual(snapshot.slidingKey.shape, [1, 2, 3, 1])
        XCTAssertEqual(snapshot.slidingKey.float16BitPatterns, [111, 112, 113, 211, 212, 213])
        XCTAssertEqual(snapshot.slidingValue.float16BitPatterns, [131, 132, 133, 231, 232, 233])
    }

    func testKVAccumulatorKeepsFullPrefixChunkBackedAndSlidingPayloadBounded() throws {
        var accumulator = try DistributedEagleTargetKVAccumulator(
            kvCapacity: 64,
            slidingWindow: 3)

        for position in 0..<64 {
            try accumulator.append(artifacts(
                range: position..<(position + 1),
                hidden: [UInt16(position), UInt16(position + 1)],
                fullKey: [UInt16(10 + position), UInt16(110 + position)],
                fullValue: [UInt16(210 + position), UInt16(310 + position)],
                slidingKey: [UInt16(410 + position), UInt16(510 + position)],
                slidingValue: [UInt16(610 + position), UInt16(710 + position)]))
        }

        let snapshot = try XCTUnwrap(accumulator.snapshot)
        XCTAssertEqual(snapshot.fullKey.storageProfile.kind, .appendOnlySequence)
        XCTAssertEqual(snapshot.fullKey.storageProfile.segmentCount, 64)
        XCTAssertEqual(snapshot.fullKey.storageProfile.retainedElementCount, 128)
        XCTAssertEqual(snapshot.fullKey.storageProfile.eagerlyMaterializedElementCount, 0)
        XCTAssertEqual(snapshot.slidingKey.storageProfile.kind, .boundedSequence)
        XCTAssertEqual(snapshot.slidingKey.storageProfile.segmentCount, 3)
        XCTAssertEqual(snapshot.slidingKey.storageProfile.retainedElementCount, 6)
        XCTAssertEqual(snapshot.slidingKey.float16BitPatterns, [471, 472, 473, 571, 572, 573])
    }

    func testEagleTargetArtifactsRejectMalformedTensorDTypeAndPairShape() throws {
        XCTAssertThrowsError(try DistributedEagleTargetTensor(
            shape: [1, 1, 1, 1],
            scalarType: .float32,
            float16BitPatterns: [1]))

        let hidden = try tensor(shape: [1, 1, 2], values: [1, 2])
        let key = try tensor(shape: [1, 2, 1, 1], values: [3, 4])
        let mismatchedValue = try tensor(shape: [1, 1, 1, 2], values: [5, 6])
        let sliding = try tensor(shape: [1, 2, 1, 1], values: [7, 8])
        XCTAssertThrowsError(try DistributedEagleTargetArtifacts(
            finalHidden: hidden,
            fullKey: key,
            fullValue: mismatchedValue,
            slidingKey: sliding,
            slidingValue: sliding,
            fullPositionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            slidingPositionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1)))
    }

    func testKVAccumulatorRejectsPositionDriftAndCapacityOverflow() throws {
        var accumulator = try DistributedEagleTargetKVAccumulator(
            kvCapacity: 3,
            slidingWindow: 2)
        try accumulator.append(artifacts(
            range: 0..<2,
            hidden: [1, 2],
            fullKey: [10, 11, 20, 21],
            fullValue: [30, 31, 40, 41],
            slidingKey: [110, 111, 210, 211],
            slidingValue: [130, 131, 230, 231]))

        XCTAssertThrowsError(try accumulator.append(artifacts(
            range: 1..<2,
            hidden: [3, 4],
            fullKey: [12, 22],
            fullValue: [32, 42],
            slidingKey: [112, 212],
            slidingValue: [132, 232])))
        XCTAssertThrowsError(try accumulator.append(artifacts(
            range: 2..<4,
            hidden: [3, 4],
            fullKey: [12, 13, 22, 23],
            fullValue: [32, 33, 42, 43],
            slidingKey: [112, 113, 212, 213],
            slidingValue: [132, 133, 232, 233])))
    }

    func testSameMachinePipelineAccumulatesAndReturnsEagleTargetArtifacts() async throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(manifestJSON(
                eagleTarget: #"{"stage_id":"layers-14-28","sliding_window":2}"#).utf8))
        let handles = try makePipelineHandles(manifest: manifest)
        let pipeline = try DistributedSameMachinePipeline(
            manifest: manifest,
            handlesByStageID: Dictionary(uniqueKeysWithValues: handles.map {
                ($0.descriptor.id, $0 as DistributedStageHandle)
            }))
        try await pipeline.allocate(requestID: "req-mtp", kvCapacity: 8)

        _ = try await pipeline.forward(
            requestID: "req-mtp",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 2),
            tokenIDs: [1, 2],
            emitToken: false)
        let output = try await pipeline.forward(
            requestID: "req-mtp",
            stepIndex: 1,
            positionRange: DistributedSequenceRange(lowerBound: 2, upperBound: 3),
            tokenIDs: [3])

        let artifacts = try XCTUnwrap(output.eagleTargetArtifacts)
        XCTAssertEqual(
            artifacts.fullPositionRange,
            DistributedSequenceRange(lowerBound: 0, upperBound: 3))
        XCTAssertEqual(
            artifacts.slidingPositionRange,
            DistributedSequenceRange(lowerBound: 1, upperBound: 3))
        XCTAssertEqual(artifacts.fullKey.float16BitPatterns, [10, 11, 12])
        XCTAssertEqual(artifacts.slidingKey.float16BitPatterns, [111, 112])
        XCTAssertEqual(artifacts.finalHidden.float16BitPatterns, [902, 903])

        try await pipeline.reset(requestID: "req-mtp")
        let resetOutput = try await pipeline.forward(
            requestID: "req-mtp",
            stepIndex: 0,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
            tokenIDs: [4])
        XCTAssertEqual(
            resetOutput.eagleTargetArtifacts?.fullPositionRange,
            DistributedSequenceRange(lowerBound: 0, upperBound: 1))
    }

    func testSameMachinePipelineRejectsRemoteEagleTargetHandle() throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(manifestJSON(
                eagleTarget: #"{"stage_id":"layers-14-28","sliding_window":2}"#).utf8))
        let handles = try makePipelineHandles(manifest: manifest)
        let targetDescriptor = try XCTUnwrap(
            manifest.runtimePlan.stage(id: "layers-14-28"))
        let remote = try DistributedRemoteStageHandle(
            plan: manifest.runtimePlan,
            descriptor: targetDescriptor
        ) { _ in nil }
        var byID = Dictionary(uniqueKeysWithValues: handles.map {
            ($0.descriptor.id, $0 as DistributedStageHandle)
        })
        byID[targetDescriptor.id] = remote

        XCTAssertThrowsError(try DistributedSameMachinePipeline(
            manifest: manifest,
            handlesByStageID: byID)) { error in
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame(
                    "eagle_target stage layers-14-28 requires a local auxiliary-output handle"))
        }
    }

    func testSameMachinePipelineRequiresArtifactsFromEagleTargetProducer() async throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(manifestJSON(
                eagleTarget: #"{"stage_id":"layers-14-28","sliding_window":2}"#).utf8))
        let handles = try makePipelineHandles(manifest: manifest, emitArtifacts: false)
        let pipeline = try DistributedSameMachinePipeline(
            manifest: manifest,
            handlesByStageID: Dictionary(uniqueKeysWithValues: handles.map {
                ($0.descriptor.id, $0 as DistributedStageHandle)
            }))
        try await pipeline.allocate(requestID: "req-missing", kvCapacity: 8)

        do {
            _ = try await pipeline.forward(
                requestID: "req-missing",
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: 1),
                tokenIDs: [1])
            XCTFail("expected the missing EAGLE target artifacts to be rejected")
        } catch {
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidStageOutput(
                    "eagle_target stage layers-14-28 did not return auxiliary artifacts"))
        }
    }

    func testWorkerExecutorRejectsLocalEagleArtifactsBeforeWireEncoding() async throws {
        let manifest = try DistributedStageManifest.decode(
            from: Data(manifestJSON(
                eagleTarget: #"{"stage_id":"layers-14-28","sliding_window":2}"#).utf8))
        let target = try XCTUnwrap(
            try makePipelineHandles(manifest: manifest).first {
                $0.descriptor.id == "layers-14-28"
            })
        let executor = try DistributedWorkerFrameExecutor(
            plan: manifest.runtimePlan,
            handle: target)
        let requestID = "req-wire-reject"
        _ = try await executor.process(DistributedWorkerWireFrame(
            message: .allocate(DistributedStageAllocation(
                requestID: requestID,
                kvCapacity: 8))))
        let range = DistributedSequenceRange(lowerBound: 0, upperBound: 1)
        let inputPacket = try DistributedHiddenStatePacket(
            metadata: DistributedHiddenStatePacketMetadata(
                requestID: requestID,
                sourceStageID: "layers-00-14",
                destinationStageID: "layers-14-28",
                positionRange: range,
                shape: [1, 1, 2],
                scalarType: .float16,
                byteCount: 4,
                stepIndex: 0),
            payload: [0, 0, 0, 0])
        let frame = DistributedStageForwardFrame(
            stageID: "layers-14-28",
            requestID: requestID,
            stepIndex: 0,
            positionRange: range,
            positionIDs: [0],
            hiddenState: inputPacket.metadata)

        do {
            _ = try await executor.process(DistributedWorkerWireFrame(
                message: .forward(frame),
                payload: inputPacket.payload))
            XCTFail("expected local EAGLE artifacts to be rejected by the worker wire path")
        } catch {
            XCTAssertEqual(
                error as? DistributedStageExecutionError,
                .invalidControlFrame(
                    "worker wire protocol does not support EAGLE target artifacts"))
        }
    }

    private var boundaryTensor: DistributedBoundaryTensorSpec {
        DistributedBoundaryTensorSpec(
            name: "hidden_states",
            shape: [1, -1, 1_024],
            scalarType: .float16)
    }

    private func transformerDescriptor(
        id: String,
        lower: Int,
        upper: Int
    ) -> DistributedStageDescriptor {
        DistributedStageDescriptor(
            id: id,
            role: .transformerLayers,
            layerRange: DistributedLayerRange(lowerBound: lower, upperBound: upper),
            assetName: "\(id).aimodel")
    }

    private func makeStageIOContract(
        kFullScalarType: DistributedStageIOScalarType = .float16,
        kSlidingShape: [Int] = [1, 8, -1, 128]
    ) -> DistributedStageIOContract {
        DistributedStageIOContract(
            inputs: [
                DistributedStageIOTensor(.hiddenStates, shape: [1, -1, 1_024], scalarType: .float16),
                DistributedStageIOTensor(.positionIDs, shape: [1, -1], scalarType: .int32),
            ],
            outputs: [
                DistributedStageIOTensor(.hiddenStates, shape: [1, -1, 1_024], scalarType: .float16),
                DistributedStageIOTensor(name: "k_full", shape: [1, 2, -1, 512], scalarType: kFullScalarType),
                DistributedStageIOTensor(name: "v_full", shape: [1, 2, -1, 512], scalarType: .float16),
                DistributedStageIOTensor(name: "k_sliding", shape: kSlidingShape, scalarType: .float16),
                DistributedStageIOTensor(name: "v_sliding", shape: [1, 8, -1, 128], scalarType: .float16),
            ])
    }

    private func tensor(shape: [Int], values: [UInt16]) throws -> DistributedEagleTargetTensor {
        try DistributedEagleTargetTensor(
            shape: shape,
            scalarType: .float16,
            float16BitPatterns: values)
    }

    #if COREAI_RUNTIME
    private func float16Array(shape: [Int], values: [Float16]) -> NDArray {
        var array = NDArray(shape: shape, scalarType: .float16)
        var view = array.mutableView(as: Float16.self)
        view.copyElements(fromContentsOf: values)
        return array
    }
    #endif

    private func artifacts(
        range: Range<Int>,
        hidden: [UInt16],
        fullKey: [UInt16],
        fullValue: [UInt16],
        slidingKey: [UInt16],
        slidingValue: [UInt16]
    ) throws -> DistributedEagleTargetArtifacts {
        let sequenceRange = DistributedSequenceRange(
            lowerBound: range.lowerBound,
            upperBound: range.upperBound)
        return try DistributedEagleTargetArtifacts(
            finalHidden: tensor(shape: [1, 1, hidden.count], values: hidden),
            fullKey: tensor(shape: [1, 2, range.count, 1], values: fullKey),
            fullValue: tensor(shape: [1, 2, range.count, 1], values: fullValue),
            slidingKey: tensor(shape: [1, 2, range.count, 1], values: slidingKey),
            slidingValue: tensor(shape: [1, 2, range.count, 1], values: slidingValue),
            fullPositionRange: sequenceRange,
            slidingPositionRange: sequenceRange)
    }

    private func kvChunk(
        range: Range<Int>,
        fullKey: [UInt16],
        fullValue: [UInt16],
        slidingKey: [UInt16],
        slidingValue: [UInt16]
    ) throws -> DistributedEagleTargetKVChunk {
        let sequenceRange = DistributedSequenceRange(
            lowerBound: range.lowerBound,
            upperBound: range.upperBound)
        return try DistributedEagleTargetKVChunk(
            fullKey: tensor(shape: [1, 1, range.count, 1], values: fullKey),
            fullValue: tensor(shape: [1, 1, range.count, 1], values: fullValue),
            slidingKey: tensor(shape: [1, 1, range.count, 1], values: slidingKey),
            slidingValue: tensor(shape: [1, 1, range.count, 1], values: slidingValue),
            fullPositionRange: sequenceRange,
            slidingPositionRange: sequenceRange)
    }

    private func makePipelineHandles(
        manifest: DistributedStageManifest,
        emitArtifacts: Bool = true
    ) throws -> [EagleTargetFakeStageHandle] {
        manifest.runtimePlan.stages.enumerated().map { index, descriptor in
            let nextStageID = manifest.runtimePlan.stages.indices.contains(index + 1)
                ? manifest.runtimePlan.stages[index + 1].id
                : nil
            return EagleTargetFakeStageHandle(
                descriptor: descriptor,
                supportsEagleTargetKVChunk: descriptor.id == manifest.eagleTarget?.stageID,
                supportsEagleTargetFinalHidden: descriptor.id
                    == manifest.eagleTarget?.finalHiddenStageID
            ) { input in
                if descriptor.role == .finalNormHead {
                    return DistributedStageForwardOutput(
                        stageID: descriptor.id,
                        stepIndex: input.stepIndex,
                        tokenID: 42,
                        eagleTargetFinalHidden: try self.tensor(
                            shape: [1, 1, 2],
                            values: [
                                UInt16(900 + input.positionRange.upperBound - 1),
                                UInt16(900 + input.positionRange.upperBound),
                            ]))
                }
                let width = 2
                let shape = [1, input.positionRange.count, width]
                let hidden = try DistributedHiddenStatePacket(
                    metadata: DistributedHiddenStatePacketMetadata(
                        requestID: input.requestID,
                        sourceStageID: descriptor.id,
                        destinationStageID: nextStageID!,
                        positionRange: input.positionRange,
                        shape: shape,
                        scalarType: .float16,
                        byteCount: shape.reduce(DistributedTensorScalarType.float16.byteWidth, *),
                        stepIndex: input.stepIndex),
                    payload: Array(
                        repeating: UInt8(index + 1),
                        count: shape.reduce(DistributedTensorScalarType.float16.byteWidth, *)))
                var eagleKVChunk: DistributedEagleTargetKVChunk?
                if emitArtifacts && descriptor.id == manifest.eagleTarget?.stageID {
                    let positions = Array(input.positionRange.lowerBound..<input.positionRange.upperBound)
                    eagleKVChunk = try self.kvChunk(
                        range: input.positionRange.lowerBound..<input.positionRange.upperBound,
                        fullKey: positions.map { UInt16(10 + $0) },
                        fullValue: positions.map { UInt16(20 + $0) },
                        slidingKey: positions.map { UInt16(110 + $0) },
                        slidingValue: positions.map { UInt16(120 + $0) })
                }
                return DistributedStageForwardOutput(
                    stageID: descriptor.id,
                    stepIndex: input.stepIndex,
                    hiddenState: hidden,
                    eagleTargetKVChunk: eagleKVChunk)
            }
        }
    }

    private func manifestJSON(eagleTarget: String) -> String {
        """
        {
          "schema": "\(DistributedStageManifest.currentSchema)",
          "model": "gemma-4-staged",
          "total_layer_count": 28,
          "boundary": {
            "hidden_state": {
              "name": "hidden_states",
              "shape": [1, -1, 2],
              "scalar_type": "float16"
            }
          },
          "eagle_target": \(eagleTarget),
          "stages": [
            {"id":"embed","role":"embeddings","layers":"embeddings","bundle":"embed.aimodel","memory_gb":1},
            {"id":"layers-00-14","role":"transformer_layers","layers":[0,14],"bundle":"layers-a.aimodel","memory_gb":2},
            {"id":"layers-14-28","role":"transformer_layers","layers":[14,28],"bundle":"layers-b.aimodel","memory_gb":2},
            {"id":"head","role":"final_norm_head","layers":"norm+lm_head","bundle":"head.aimodel","memory_gb":1}
          ]
        }
        """
    }

    private func richManifestJSON() -> String {
        """
        {
          "schema": "\(DistributedStageManifest.currentSchema)",
          "model": "gemma-4-staged",
          "total_layer_count": 28,
          "cache_groups": {
            "strategy": "gemma4_split_sliding_global_v0",
            "groups": {
              "sliding": {
                "state_names": ["sliding_keyCache", "sliding_valueCache"],
                "capacity": 3072,
                "sliding_window": 1024
              },
              "global": {
                "state_names": ["global_keyCache", "global_valueCache"],
                "capacity": 131072
              }
            }
          },
          "eagle_target": {
            "status": "target_contract_only",
            "outputs": {
              "logits": {"stage": "head", "tensor": "logits"},
              "backbone_hidden": {"stage": "head", "tensor": "hidden"}
            },
            "representative_kv": {
              "full_attention": {
                "stage": "layers-14-28",
                "key_output": "k_full",
                "value_output": "v_full"
              },
              "sliding_attention": {
                "stage": "layers-14-28",
                "key_output": "k_sliding",
                "value_output": "v_sliding"
              }
            },
            "kv_output_semantics": "new_positions_only"
          },
          "stages": [
            {"id":"embed","role":"embeddings","layers":"embeddings","bundle":"embed.aimodel","memory_gb":1},
            {"id":"layers-00-14","role":"transformer_layers","layers":[0,14],"bundle":"layers-a.aimodel","memory_gb":2},
            {"id":"layers-14-28","role":"transformer_layers","layers":[14,28],"bundle":"layers-b.aimodel","memory_gb":2},
            {"id":"head","role":"final_norm_head","layers":"norm+lm_head","bundle":"head.aimodel","memory_gb":1}
          ]
        }
        """
    }
}

private final class EagleTargetFakeStageHandle: DistributedStageHandle {
    let descriptor: DistributedStageDescriptor
    let acceptsTokenIDs: Bool
    let supportsEagleTargetKVChunk: Bool
    let supportsEagleTargetFinalHidden: Bool
    private let output: (DistributedStageForwardInput) throws -> DistributedStageForwardOutput

    init(
        descriptor: DistributedStageDescriptor,
        supportsEagleTargetKVChunk: Bool,
        supportsEagleTargetFinalHidden: Bool,
        output: @escaping (DistributedStageForwardInput) throws -> DistributedStageForwardOutput
    ) {
        self.descriptor = descriptor
        self.acceptsTokenIDs = descriptor.role == .embeddings
        self.supportsEagleTargetKVChunk = supportsEagleTargetKVChunk
        self.supportsEagleTargetFinalHidden = supportsEagleTargetFinalHidden
        self.output = output
    }

    func allocate(_ allocation: DistributedStageAllocation) async throws {}

    func forward(_ input: DistributedStageForwardInput) async throws -> DistributedStageForwardOutput {
        try output(input)
    }

    func reset(requestID: String) async throws {}

    func free(requestID: String) async {}
}
