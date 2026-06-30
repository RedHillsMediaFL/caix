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

    func testRealStageAssetsMatchMonolithicGreedyTokens() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_STAGE_MANIFEST to a real staged manifest")
        }
        guard let baselinePath = ProcessInfo.processInfo.environment["CAIX_BASELINE_MODEL"],
            !baselinePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_BASELINE_MODEL to a monolithic caix bundle")
        }

        #if COREAI_RUNTIME
        let promptFile = ProcessInfo.processInfo.environment["CAIX_TOKEN_MATCH_PROMPTS"]
        let prompts = try loadPrompts(path: promptFile)
        let maxTokens = max(
            1,
            Int(ProcessInfo.processInfo.environment["CAIX_TOKEN_MATCH_MAX_TOKENS"] ?? "") ?? 8)
        let baselineBundle = try ResolvedBundle.load(at: baselinePath)
        let baseline = try await LLMEngine.load(bundle: baselineBundle)
        let manifest = try DistributedStageManifest.load(
            from: URL(fileURLWithPath: manifestPath).standardizedFileURL)
        let pipeline = try await DistributedSameMachinePipeline.make(
            manifest: manifest,
            handleFactory: DistributedCoreAIStageHandleFactory())

        for (index, prompt) in prompts.enumerated() {
            let promptTokens = try baseline.encodePrompt(
                messages: [["role": "user", "content": prompt]],
                applyChatTemplate: false)
            let capacity = min(
                baseline.maxContextLength,
                max(promptTokens.count + maxTokens + 8, maxTokens + 1))
            try baseline.allocateKVCache(capacity: capacity)
            let requestID = "token-match-\(index)"
            try await pipeline.allocate(requestID: requestID, kvCapacity: capacity)

            let prompt32 = promptTokens.map(Int32.init)
            let baselinePrefillRows = try await baseline.forwardAllRows(tokens: prompt32)
            traceLogitRowsIfRequested(
                baselinePrefillRows,
                label: "baseline prompt=\(index) prefill")
            let baselinePrefillLogits = try XCTUnwrap(baselinePrefillRows.last)
            traceLogitsIfRequested(
                baselinePrefillLogits,
                label: "baseline prompt=\(index) step=0")
            var baselineNext = Int32(Sampler.argmax(baselinePrefillLogits))
            var stagedNext = try await nextStagedToken(
                pipeline: pipeline,
                requestID: requestID,
                stepIndex: 0,
                lowerBound: 0,
                tokenIDs: prompt32)
            guard stagedNext == baselineNext else {
                XCTFail(
                    "prompt \(index) prefill token mismatch: staged \(stagedNext), "
                        + "baseline \(baselineNext)")
                await pipeline.free(requestID: requestID)
                return
            }

            for step in 1..<maxTokens {
                let baselineDecodeLogits = try await baseline.step(tokens: [baselineNext])
                traceLogitsIfRequested(
                    baselineDecodeLogits,
                    label: "baseline prompt=\(index) step=\(step)")
                baselineNext = Int32(Sampler.argmax(baselineDecodeLogits))
                stagedNext = try await nextStagedToken(
                    pipeline: pipeline,
                    requestID: requestID,
                    stepIndex: step,
                    lowerBound: promptTokens.count + step - 1,
                    tokenIDs: [stagedNext])
                guard stagedNext == baselineNext else {
                    XCTFail(
                        "prompt \(index) decode step \(step) mismatch: staged \(stagedNext), "
                            + "baseline \(baselineNext)")
                    await pipeline.free(requestID: requestID)
                    return
                }
            }
            await pipeline.free(requestID: requestID)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealStageAssetsReplayPrefillMatchesMonolithicSecondToken() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_STAGE_MANIFEST to a real staged manifest")
        }
        guard let baselinePath = ProcessInfo.processInfo.environment["CAIX_BASELINE_MODEL"],
            !baselinePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_BASELINE_MODEL to a monolithic caix bundle")
        }

        #if COREAI_RUNTIME
        let prompt = try loadPrompts(path: ProcessInfo.processInfo.environment["CAIX_TOKEN_MATCH_PROMPTS"])[0]
        let baselineBundle = try ResolvedBundle.load(at: baselinePath)
        let baseline = try await LLMEngine.load(bundle: baselineBundle)
        let manifest = try DistributedStageManifest.load(
            from: URL(fileURLWithPath: manifestPath).standardizedFileURL)
        let pipeline = try await DistributedSameMachinePipeline.make(
            manifest: manifest,
            handleFactory: DistributedCoreAIStageHandleFactory())

        let promptTokens = try baseline.encodePrompt(
            messages: [["role": "user", "content": prompt]],
            applyChatTemplate: false)
        let prompt32 = promptTokens.map(Int32.init)
        let capacity = min(baseline.maxContextLength, promptTokens.count + 16)

        try baseline.allocateKVCache(capacity: capacity)
        let firstBaseline = Int32(Sampler.argmax(try await baseline.step(tokens: prompt32)))
        let secondBaseline = Int32(Sampler.argmax(try await baseline.step(tokens: [firstBaseline])))

        let incrementalRequestID = "replay-prefill-incremental"
        try await pipeline.allocate(requestID: incrementalRequestID, kvCapacity: capacity)
        let firstStaged = try await nextStagedToken(
            pipeline: pipeline,
            requestID: incrementalRequestID,
            stepIndex: 0,
            lowerBound: 0,
            tokenIDs: prompt32)
        await pipeline.free(requestID: incrementalRequestID)
        guard firstStaged == firstBaseline else {
            XCTFail(
                "prefill token mismatch: staged \(firstStaged), baseline \(firstBaseline)")
            return
        }

        let replayRequestID = "replay-prefill-fresh"
        try await pipeline.allocate(requestID: replayRequestID, kvCapacity: capacity)
        let secondStaged = try await nextStagedToken(
            pipeline: pipeline,
            requestID: replayRequestID,
            stepIndex: 0,
            lowerBound: 0,
            tokenIDs: prompt32 + [firstBaseline])
        await pipeline.free(requestID: replayRequestID)
        XCTAssertEqual(
            secondStaged,
            secondBaseline,
            "fresh staged prefill over prompt+first token should match monolithic second token")
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealHeadStageMatchesHiddenFixtureGreedyToken() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_STAGE_MANIFEST to a real staged manifest")
        }
        guard let hiddenPath = ProcessInfo.processInfo.environment["CAIX_HEAD_HIDDEN_F16"],
            !hiddenPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_HEAD_HIDDEN_F16 to a float16 hidden-state fixture")
        }
        guard let rawExpected = ProcessInfo.processInfo.environment["CAIX_HEAD_EXPECTED_TOKEN"],
            let expectedToken = Int32(rawExpected)
        else {
            throw XCTSkip("set CAIX_HEAD_EXPECTED_TOKEN to the expected greedy token")
        }

        #if COREAI_RUNTIME
        let manifest = try DistributedStageManifest.load(
            from: URL(fileURLWithPath: manifestPath).standardizedFileURL)
        guard let boundaryTensor = manifest.boundaryTensor else {
            throw XCTSkip("manifest has no boundary tensor metadata")
        }
        let hiddenSize = try XCTUnwrap(boundaryTensor.shape.last)
        let hiddenData = try Data(contentsOf: URL(fileURLWithPath: hiddenPath))
        XCTAssertEqual(hiddenData.count % boundaryTensor.scalarType.byteWidth, 0)
        XCTAssertEqual(boundaryTensor.scalarType, .float16)
        let elementCount = hiddenData.count / boundaryTensor.scalarType.byteWidth
        XCTAssertEqual(elementCount % hiddenSize, 0)
        let positionCount = elementCount / hiddenSize
        XCTAssertGreaterThan(positionCount, 0)

        let headIndex = try XCTUnwrap(
            manifest.runtimePlan.stages.firstIndex { $0.role == .finalNormHead })
        let headDescriptor = manifest.runtimePlan.stages[headIndex]
        let headStage = try XCTUnwrap(manifest.stages.first { $0.id == headDescriptor.id })
        let sourceStageID = headIndex > 0 ? manifest.runtimePlan.stages[headIndex - 1].id : "fixture"
        let requestID = "head-fixture"
        let positionRange = DistributedSequenceRange(lowerBound: 0, upperBound: positionCount)
        let metadata = DistributedHiddenStatePacketMetadata(
            requestID: requestID,
            sourceStageID: sourceStageID,
            destinationStageID: headDescriptor.id,
            positionRange: positionRange,
            shape: [1, positionCount, hiddenSize],
            scalarType: .float16,
            byteCount: hiddenData.count,
            stepIndex: 0)
        let packet = try DistributedHiddenStatePacket(
            metadata: metadata,
            payload: Array(hiddenData))

        let context = DistributedStageHandleFactoryContext(
            stage: headStage,
            manifest: manifest,
            descriptor: headDescriptor)
        let handle = try await DistributedCoreAIStageHandleFactory()
            .makeStageHandle(for: context)
        try await handle.allocate(DistributedStageAllocation(
            requestID: requestID,
            kvCapacity: positionCount + 1))
        let output = try await handle.forward(DistributedStageForwardInput(
            requestID: requestID,
            stepIndex: 0,
            positionRange: positionRange,
            positionIDs: manifest.positionMode.positionIDs(for: positionRange),
            hiddenState: packet))
        await handle.free(requestID: requestID)
        XCTAssertEqual(output.tokenID, expectedToken)
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    private func loadPrompts(path: String?) throws -> [String] {
        guard let path, !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return ["The future of local inference is"]
        }
        let text = try String(contentsOfFile: path, encoding: .utf8)
        let prompts = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !prompts.isEmpty else {
            throw XCTSkip("CAIX_TOKEN_MATCH_PROMPTS has no prompts")
        }
        return prompts
    }

    private func nextStagedToken(
        pipeline: DistributedSameMachinePipeline,
        requestID: String,
        stepIndex: Int,
        lowerBound: Int,
        tokenIDs: [Int32]
    ) async throws -> Int32 {
        let output = try await pipeline.forward(
            requestID: requestID,
            stepIndex: stepIndex,
            positionRange: DistributedSequenceRange(
                lowerBound: lowerBound,
                upperBound: lowerBound + tokenIDs.count),
            tokenIDs: tokenIDs)
        return try XCTUnwrap(output.tokenID)
    }

    private func traceLogitsIfRequested(_ logits: [Float], label: String) {
        guard let rawLimit = ProcessInfo.processInfo.environment["CAIX_LOGIT_TRACE_TOPK"],
            let limit = Int(rawLimit),
            limit > 0
        else {
            return
        }
        let top = Sampler.topK(logits, count: min(limit, logits.count))
        let margin = top.count > 1 ? top[0].logit - top[1].logit : .nan
        let fields = top.map { "\($0.index):\(String(format: "%.6g", Double($0.logit)))" }
            .joined(separator: ",")
        let line = "[caix-logits] \(label) top=\(fields) "
            + "margin=\(String(format: "%.6g", Double(margin)))\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func traceLogitRowsIfRequested(_ rows: [[Float]], label: String) {
        guard ProcessInfo.processInfo.environment["CAIX_TRACE_BASELINE_ROWS"] != nil else {
            return
        }
        for (rowIndex, row) in rows.enumerated() {
            traceLogitsIfRequested(row, label: "\(label) row=\(rowIndex)")
        }
    }
}
