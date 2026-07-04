import XCTest
@testable import PipelineRuntime

#if COREAI_RUNTIME
import CoreAI
import Tokenizers
#endif

final class DistributedCoreAIStageAssetIntegrationTests: XCTestCase {
    func testRealTwoCacheGroupToyStageBindsFourStateTensors() async throws {
        guard let assetPath = ProcessInfo.processInfo.environment["CAIX_TWO_CACHE_GROUP_TOY_ASSET"],
            !assetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_TWO_CACHE_GROUP_TOY_ASSET to the two-cache-group toy .aimodel")
        }

        #if COREAI_RUNTIME
        let assetURL = URL(fileURLWithPath: assetPath).standardizedFileURL
        let boundary = DistributedBoundaryTensorSpec(
            name: "hidden_states",
            shape: [1, -1, 4],
            scalarType: .float32)
        let embedStage = DistributedStageManifestStage(
            id: "toy-embed",
            role: .embeddings,
            layerSpec: .label("embed"),
            assetName: "unused-embed.aimodel",
            resolvedAssetPath: "/tmp/unused-embed.aimodel",
            memoryGB: 0.001)
        let toyStage = DistributedStageManifestStage(
            id: "toy-layers",
            role: .transformerLayers,
            layerSpec: .range(DistributedLayerRange(lowerBound: 0, upperBound: 1)),
            assetName: assetURL.lastPathComponent,
            resolvedAssetPath: assetURL.path,
            memoryGB: 0.001)
        let headStage = DistributedStageManifestStage(
            id: "toy-head",
            role: .finalNormHead,
            layerSpec: .label("head"),
            assetName: "unused-head.aimodel",
            resolvedAssetPath: "/tmp/unused-head.aimodel",
            vocabSize: 8,
            memoryGB: 0.001)
        let manifest = try DistributedStageManifest(
            modelName: "two-cache-group-toy",
            totalLayerCount: 1,
            stages: [embedStage, toyStage, headStage],
            boundaryTensor: boundary,
            positionMode: .fullPrefix)
        let descriptor = try XCTUnwrap(manifest.runtimePlan.stage(id: toyStage.id))
        let context = DistributedStageHandleFactoryContext(
            stage: toyStage,
            manifest: manifest,
            descriptor: descriptor)
        let handle = try await DistributedCoreAIStageHandleFactory().makeStageHandle(for: context)
        let coreAIHandle = try XCTUnwrap(handle as? DistributedCoreAIStageHandle)
        XCTAssertEqual(
            coreAIHandle.ioContract.stateNames,
            [
                "sliding_keyCache",
                "sliding_valueCache",
                "global_keyCache",
                "global_valueCache",
            ])

        let requestID = "two-cache-group-\(UUID().uuidString)"
        try await coreAIHandle.allocate(DistributedStageAllocation(
            requestID: requestID,
            kvCapacity: 12,
            cacheCapacities: ["sliding": 6, "global": 12]))
        do {
            let firstRange = DistributedSequenceRange(lowerBound: 0, upperBound: 2)
            let firstInput = try DistributedHiddenStatePacket(
                metadata: DistributedHiddenStatePacketMetadata(
                    requestID: requestID,
                    sourceStageID: "toy-embed",
                    destinationStageID: toyStage.id,
                    positionRange: firstRange,
                    shape: [1, 2, 4],
                    scalarType: .float32,
                    byteCount: 8 * MemoryLayout<Float>.size,
                    stepIndex: 0),
                float32Values: [0, 1, 2, 3, 4, 5, 6, 7])
            let firstOutput = try await coreAIHandle.forward(DistributedStageForwardInput(
                requestID: requestID,
                stepIndex: 0,
                positionRange: firstRange,
                positionIDs: [0, 1],
                hiddenState: firstInput))
            let firstHidden = try XCTUnwrap(firstOutput.hiddenState)
            XCTAssertEqual(firstHidden.metadata.destinationStageID, headStage.id)
            XCTAssertEqual(firstHidden.metadata.shape, [1, 2, 4])
            XCTAssertEqual(try firstHidden.float32Values(), [0, 2, 4, 6, 8, 10, 12, 14])

            let secondRange = DistributedSequenceRange(lowerBound: 2, upperBound: 3)
            let secondInput = try DistributedHiddenStatePacket(
                metadata: DistributedHiddenStatePacketMetadata(
                    requestID: requestID,
                    sourceStageID: "toy-embed",
                    destinationStageID: toyStage.id,
                    positionRange: secondRange,
                    shape: [1, 1, 4],
                    scalarType: .float32,
                    byteCount: 4 * MemoryLayout<Float>.size,
                    stepIndex: 1),
                float32Values: [10, 11, 12, 13])
            let secondOutput = try await coreAIHandle.forward(DistributedStageForwardInput(
                requestID: requestID,
                stepIndex: 1,
                positionRange: secondRange,
                positionIDs: [0, 1, 2],
                hiddenState: secondInput))
            let secondHidden = try XCTUnwrap(secondOutput.hiddenState)
            XCTAssertEqual(secondHidden.metadata.shape, [1, 1, 4])
            XCTAssertEqual(try secondHidden.float32Values(), [20, 22, 24, 26])
        } catch {
            await coreAIHandle.free(requestID: requestID)
            throw error
        }
        await coreAIHandle.free(requestID: requestID)
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealStageMainAndDecodeAssetsReportResidentDeltas() async throws {
        guard let mainAssetPath = ProcessInfo.processInfo.environment["CAIX_RSS_MAIN_ASSET"],
            !mainAssetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            let decodeAssetPath = ProcessInfo.processInfo.environment["CAIX_RSS_DECODE_ASSET"],
            !decodeAssetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_RSS_MAIN_ASSET and CAIX_RSS_DECODE_ASSET")
        }

        #if COREAI_RUNTIME
        let mainAssetURL = URL(fileURLWithPath: mainAssetPath).standardizedFileURL
        let decodeAssetURL = URL(fileURLWithPath: decodeAssetPath).standardizedFileURL
        let mainFunctionName = ProcessInfo.processInfo.environment["CAIX_RSS_MAIN_FUNCTION"] ?? "main"
        let decodeFunctionName = ProcessInfo.processInfo.environment["CAIX_RSS_DECODE_FUNCTION"] ?? "decode"
        let summaryURL = ProcessInfo.processInfo.environment["CAIX_RSS_SUMMARY_FILE"]
            .flatMap { raw -> URL? in
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : URL(fileURLWithPath: trimmed)
            }

        let baselineRSSMiB = try currentProcessRSSMiB()
        var specialization = SpecializationOptions(
            preferredComputeUnitKind: LLMEngine.preferredComputeUnit())
        specialization.expectFrequentReshapes = true

        let mainModel = try await AIModel.specialize(
            contentsOf: mainAssetURL,
            options: specialization,
            cache: .default,
            cachePolicy: .persistent)
        let mainFunction = try XCTUnwrap(
            try mainModel.loadFunction(named: mainFunctionName),
            "main function \(mainFunctionName) missing from \(mainAssetURL.path)")
        try await Task.sleep(nanoseconds: 750_000_000)
        let afterMainRSSMiB = try currentProcessRSSMiB()

        let decodeModel = try await AIModel.specialize(
            contentsOf: decodeAssetURL,
            options: specialization,
            cache: .default,
            cachePolicy: .persistent)
        let decodeFunction = try XCTUnwrap(
            try decodeModel.loadFunction(named: decodeFunctionName),
            "decode function \(decodeFunctionName) missing from \(decodeAssetURL.path)")
        try await Task.sleep(nanoseconds: 750_000_000)
        let afterDecodeRSSMiB = try currentProcessRSSMiB()

        let payload: [String: Any] = [
            "baseline_rss_mib": baselineRSSMiB,
            "after_main_rss_mib": afterMainRSSMiB,
            "after_decode_rss_mib": afterDecodeRSSMiB,
            "main_delta_mib": afterMainRSSMiB - baselineRSSMiB,
            "decode_incremental_delta_mib": afterDecodeRSSMiB - afterMainRSSMiB,
            "decode_vs_main_delta_ratio": (afterDecodeRSSMiB - afterMainRSSMiB)
                / max(afterMainRSSMiB - baselineRSSMiB, 1),
            "main_asset": mainAssetURL.path,
            "decode_asset": decodeAssetURL.path,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys])
        let json = String(decoding: data, as: UTF8.self)
        let line = "[caix-weight-rss] \(json.replacingOccurrences(of: "\n", with: " "))\n"
        FileHandle.standardError.write(Data(line.utf8))
        if let summaryURL {
            try data.write(to: summaryURL, options: .atomic)
        }
        withExtendedLifetime((mainFunction, decodeFunction)) {}
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

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

    func testRealMultimodalGemmaStageBundleGeneratesTextOnlyWithBlockIDs() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }

        #if COREAI_RUNTIME
        let minAvailableRAMGB = loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_AVAILABLE_RAM_GB")
            ?? loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_FREE_RAM_GB")
            ?? 20
        let availableRAMGB = try currentAvailableMemoryGB()
        guard availableRAMGB >= minAvailableRAMGB else {
            throw XCTSkip(
                "available RAM \(String(format: "%.2f", availableRAMGB)) GiB is below "
                    + "\(String(format: "%.2f", minAvailableRAMGB)) GiB floor")
        }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        XCTAssertEqual(manifest.positionMode, .fullPrefix)
        XCTAssertEqual(manifest.stages.count, 4)
        XCTAssertEqual(
            manifest.runtimePlan.stages.filter { $0.role == .transformerLayers }.count,
            2)
        let blockIDsRequired = try MultimodalStagedModel.blockIDsRequired(manifestURL: manifestURL)
        for stage in manifest.runtimePlan.stages where stage.role == .transformerLayers {
            if blockIDsRequired {
                XCTAssertEqual(stage.prefillExtraInputs, ["block_ids_q", "block_ids_kv"])
            } else {
                XCTAssertTrue(stage.prefillExtraInputs.isEmpty)
            }
        }

        let tokenizerURL = manifestURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer", isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        XCTAssertTrue(tokenizer.hasChatTemplate, "multimodal Gemma tokenizer must expose bundled chat_template.jinja")
        let promptFile = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_TEXT_PROMPTS_FILE"]
        let prompts: [String]
        if let promptFile, !promptFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            prompts = try loadPrompts(path: promptFile)
        } else {
            prompts = [
                ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_TEXT_PROMPT"]
                    ?? "What is the capital of France?"
            ]
        }
        let maxTokens = max(
            1,
            loadIntEnvironment("CAIX_MM_GEMMA_MAX_TOKENS") ?? 48)

        let maxContextLength = stagedMaxContextLength(manifestURL: manifestURL)
        let pipeline = try await DistributedSameMachinePipeline.make(
            manifest: manifest,
            handleFactory: DistributedCoreAIStageHandleFactory())
        var summaries: [String] = []
        var jsonLines: [String] = []
        summaries.append("[caix-mm-gemma-text] manifest=\(manifestURL.path)")
        summaries.append(
            "[caix-mm-gemma-text] template_applied=true available_ram_gib=\(String(format: "%.2f", availableRAMGB)) prompts=\(prompts.count) max_tokens=\(maxTokens)")

        for (promptIndex, prompt) in prompts.enumerated() {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]])
            guard !promptTokens.isEmpty else {
                XCTFail("multimodal Gemma text-only prompt \(promptIndex) tokenized to 0 tokens")
                return
            }
            let prompt32 = promptTokens.map(Int32.init)
            let capacity = min(maxContextLength, max(promptTokens.count + maxTokens + 8, 2))
            let requestID = "mm-gemma-text-\(promptIndex)-\(UUID().uuidString)"
            try await pipeline.allocate(requestID: requestID, kvCapacity: capacity)

            let prefillPositionRange = DistributedSequenceRange(
                lowerBound: 0,
                upperBound: prompt32.count)
            let prefillPositionIDs = try manifest.positionMode.positionIDs(for: prefillPositionRange)
            let textOnlyBlockIDsQ = blockIDsRequired
                ? Array(repeating: Int32(-1), count: prefillPositionRange.count)
                : nil
            let textOnlyBlockIDsKV = blockIDsRequired
                ? Array(repeating: Int32(-1), count: prefillPositionIDs.count)
                : nil

            var generated: [Int32] = []
            var stepTopLogits: [[DistributedLogitScore]] = []
            do {
                var observation = try await nextStagedTokenObservation(
                    pipeline: pipeline,
                    requestID: requestID,
                    stepIndex: 0,
                    lowerBound: 0,
                    tokenIDs: prompt32,
                    blockIDsQ: textOnlyBlockIDsQ,
                    blockIDsKV: textOnlyBlockIDsKV)
                generated.append(observation.tokenID)
                stepTopLogits.append(observation.topLogits)

                let eosTokenID = tokenizer.convertTokenToId("<eos>").map(Int32.init)
                while generated.count < maxTokens {
                    if let eosTokenID, generated.last == eosTokenID {
                        break
                    }
                    let lowerBound = promptTokens.count + generated.count - 1
                    observation = try await nextStagedTokenObservation(
                        pipeline: pipeline,
                        requestID: requestID,
                        stepIndex: generated.count,
                        lowerBound: lowerBound,
                        tokenIDs: [observation.tokenID])
                    generated.append(observation.tokenID)
                    stepTopLogits.append(observation.topLogits)
                }
                await pipeline.free(requestID: requestID)
            } catch {
                await pipeline.free(requestID: requestID)
                throw error
            }

            let generatedTextWithSpecials = tokenizer.decode(
                tokens: generated.map(Int.init),
                skipSpecialTokens: false)
            let generatedText = tokenizer.decode(
                tokens: generated.map(Int.init),
                skipSpecialTokens: true)
            let trimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            let letterCount = trimmed.unicodeScalars
                .filter { CharacterSet.letters.contains($0) }
                .count
            XCTAssertGreaterThanOrEqual(generated.count, min(8, maxTokens))
            XCTAssertGreaterThanOrEqual(Set(generated).count, 2)
            XCTAssertGreaterThanOrEqual(letterCount, 1)
            XCTAssertFalse(trimmed.contains("\u{FFFD}"))

            summaries.append("[caix-mm-gemma-text] prompt_index=\(promptIndex) prompt_tokens=\(promptTokens.count) generated_tokens=\(generated.count)")
            summaries.append("[caix-mm-gemma-text] prompt_index=\(promptIndex) prompt_token_ids=\(promptTokens.map(String.init).joined(separator: ","))")
            summaries.append("[caix-mm-gemma-text] prompt_index=\(promptIndex) generated_token_ids=\(generated.map(String.init).joined(separator: ","))")
            summaries.append("[caix-mm-gemma-text] prompt_index=\(promptIndex) decoded_with_specials=\(generatedTextWithSpecials.trimmingCharacters(in: .whitespacesAndNewlines))")
            summaries.append("[caix-mm-gemma-text] prompt_index=\(promptIndex) decoded_human=\(trimmed)")

            jsonLines.append(try jsonLine([
                "prompt_index": promptIndex,
                "prompt": prompt,
                "template_applied": true,
                "prompt_token_ids": promptTokens,
                "generated_token_ids": generated.map(Int.init),
                "first_top_logits": (stepTopLogits.first ?? []).map { [
                    "token_id": Int($0.tokenID),
                    "logit": Double($0.logit),
                ] },
                "step_top_logits": stepTopLogits.map { topLogits in
                    topLogits.map { [
                        "token_id": Int($0.tokenID),
                        "logit": Double($0.logit),
                    ] }
                },
                "decoded_with_specials": generatedTextWithSpecials,
                "decoded_human": generatedText,
            ]))
        }

        let summary = summaries.joined(separator: "\n")
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_TEXT_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
        if let jsonlPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_TEXT_JSONL_OUTPUT"],
            !jsonlPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (jsonLines.joined(separator: "\n") + "\n").write(
                toFile: jsonlPath,
                atomically: true,
                encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealMultimodalGemmaStageBundleSyntheticSoftTokenSpliceLocalizesDivergence() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }

        #if COREAI_RUNTIME
        let minAvailableRAMGB = loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_AVAILABLE_RAM_GB")
            ?? loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_FREE_RAM_GB")
            ?? 20
        let availableRAMGB = try currentAvailableMemoryGB()
        guard availableRAMGB >= minAvailableRAMGB else {
            throw XCTSkip(
                "available RAM \(String(format: "%.2f", availableRAMGB)) GiB is below "
                    + "\(String(format: "%.2f", minAvailableRAMGB)) GiB floor")
        }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        let tokenizerURL = manifestURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer", isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let prompt = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_SYNTHETIC_PROMPT"]
            ?? "Q: What is shown in this image? Context before image tokens should stay stable. A:"
        let promptTokens = tokenizer.encode(text: prompt).map(Int32.init)
        let imageStart = loadIntEnvironment("CAIX_MM_GEMMA_SYNTHETIC_IMAGE_START") ?? 6
        let imageRows = loadIntEnvironment("CAIX_MM_GEMMA_SYNTHETIC_IMAGE_ROWS") ?? 4
        guard promptTokens.count >= imageStart + imageRows + 2 else {
            throw XCTSkip(
                "synthetic prompt has \(promptTokens.count) tokens; need at least \(imageStart + imageRows + 2)")
        }
        let hiddenSize = try XCTUnwrap(manifest.boundaryTensor?.shape.last)
        let blockIDsRequired = try MultimodalStagedModel.blockIDsRequired(manifestURL: manifestURL)
        let splice = try DistributedSoftTokenSplice(
            positionStart: imageStart,
            rows: imageRows,
            hiddenSize: hiddenSize,
            float16Values: syntheticSoftTokenRows(rows: imageRows, hiddenSize: hiddenSize))
        let imageBlockIDs = imageSpanBlockIDs(
            count: promptTokens.count,
            imageStart: imageStart,
            imageRows: imageRows)
        let textOnlyBlockIDs = Array(repeating: Int32(-1), count: promptTokens.count)
        let capacity = min(
            stagedMaxContextLength(manifestURL: manifestURL),
            max(promptTokens.count + 10, 2))
        let handles = try await makeCoreAIStageHandles(manifest: manifest)

        let baseline = try await runManualStagedPrefillAndOneDecode(
            handles: handles,
            manifest: manifest,
            requestID: "mm-splice-baseline-\(UUID().uuidString)",
            capacity: capacity,
            tokenIDs: promptTokens,
            blockIDsQ: blockIDsRequired ? textOnlyBlockIDs : nil,
            blockIDsKV: blockIDsRequired ? textOnlyBlockIDs : nil,
            softTokenSplice: nil)
        let spliced = try await runManualStagedPrefillAndOneDecode(
            handles: handles,
            manifest: manifest,
            requestID: "mm-splice-active-\(UUID().uuidString)",
            capacity: capacity,
            tokenIDs: promptTokens,
            blockIDsQ: blockIDsRequired ? imageBlockIDs : nil,
            blockIDsKV: blockIDsRequired ? imageBlockIDs : nil,
            softTokenSplice: splice)

        let embedDiff = try rowByteDifferences(
            baseline.embedPacket,
            spliced.embedPacket)
        XCTAssertTrue(embedDiff[..<imageStart].allSatisfy { !$0 })
        XCTAssertTrue(embedDiff[imageStart..<(imageStart + imageRows)].allSatisfy { $0 })

        let transformerDiff = try rowByteDifferences(
            baseline.lastTransformerPacket,
            spliced.lastTransformerPacket)
        XCTAssertTrue(
            transformerDiff[..<imageStart].allSatisfy { !$0 },
            "rows before synthetic image span must stay byte-identical")
        XCTAssertTrue(
            transformerDiff[imageStart...].allSatisfy { $0 },
            "rows from synthetic image span onward must diverge")

        let summary = """
            [caix-mm-gemma-splice] manifest=\(manifestURL.path)
            [caix-mm-gemma-splice] available_ram_gib=\(String(format: "%.2f", availableRAMGB)) prompt_tokens=\(promptTokens.count) image_start=\(imageStart) image_rows=\(imageRows) hidden=\(hiddenSize)
            [caix-mm-gemma-splice] embed_diff=\(boolVector(embedDiff))
            [caix-mm-gemma-splice] transformer_diff=\(boolVector(transformerDiff))
            [caix-mm-gemma-splice] prefill_tokens baseline=\(baseline.prefillToken) spliced=\(spliced.prefillToken)
            [caix-mm-gemma-splice] decode_tokens baseline=\(baseline.decodeToken) spliced=\(spliced.decodeToken)
            """
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_SPLICE_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealMultimodalGemmaStageBundleSyntheticBlockIDsEnableBidirectionalImageAttention() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }

        #if COREAI_RUNTIME
        let minAvailableRAMGB = loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_AVAILABLE_RAM_GB")
            ?? loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_FREE_RAM_GB")
            ?? 20
        let availableRAMGB = try currentAvailableMemoryGB()
        guard availableRAMGB >= minAvailableRAMGB else {
            throw XCTSkip(
                "available RAM \(String(format: "%.2f", availableRAMGB)) GiB is below "
                    + "\(String(format: "%.2f", minAvailableRAMGB)) GiB floor")
        }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        let blockIDsRequired = try MultimodalStagedModel.blockIDsRequired(manifestURL: manifestURL)
        guard blockIDsRequired else {
            throw XCTSkip("this multimodal staged bundle uses causal image attention and has no block_ids inputs")
        }
        let tokenizerURL = manifestURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer", isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let prompt = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_SYNTHETIC_PROMPT"]
            ?? "Q: What is shown in this image? Context before image tokens should stay stable. A:"
        let promptTokens = tokenizer.encode(text: prompt).map(Int32.init)
        let imageStart = loadIntEnvironment("CAIX_MM_GEMMA_SYNTHETIC_IMAGE_START") ?? 6
        let imageRows = loadIntEnvironment("CAIX_MM_GEMMA_SYNTHETIC_IMAGE_ROWS") ?? 4
        guard promptTokens.count >= imageStart + imageRows + 2 else {
            throw XCTSkip(
                "synthetic prompt has \(promptTokens.count) tokens; need at least \(imageStart + imageRows + 2)")
        }

        let hiddenSize = try XCTUnwrap(manifest.boundaryTensor?.shape.last)
        let splice = try DistributedSoftTokenSplice(
            positionStart: imageStart,
            rows: imageRows,
            hiddenSize: hiddenSize,
            float16Values: syntheticSoftTokenRows(rows: imageRows, hiddenSize: hiddenSize))
        let bidirectionalImageBlockIDs = imageSpanBlockIDs(
            count: promptTokens.count,
            imageStart: imageStart,
            imageRows: imageRows)
        let causalBlockIDs = Array(repeating: Int32(-1), count: promptTokens.count)
        let capacity = min(
            stagedMaxContextLength(manifestURL: manifestURL),
            max(promptTokens.count + 10, 2))
        let handles = try await makeCoreAIStageHandles(manifest: manifest)

        let bidirectional = try await runManualStagedPrefillAndOneDecode(
            handles: handles,
            manifest: manifest,
            requestID: "mm-blockids-bidir-\(UUID().uuidString)",
            capacity: capacity,
            tokenIDs: promptTokens,
            blockIDsQ: bidirectionalImageBlockIDs,
            blockIDsKV: bidirectionalImageBlockIDs,
            softTokenSplice: splice)
        let causal = try await runManualStagedPrefillAndOneDecode(
            handles: handles,
            manifest: manifest,
            requestID: "mm-blockids-causal-\(UUID().uuidString)",
            capacity: capacity,
            tokenIDs: promptTokens,
            blockIDsQ: causalBlockIDs,
            blockIDsKV: causalBlockIDs,
            softTokenSplice: splice)

        let embedDiff = try rowByteDifferences(
            bidirectional.embedPacket,
            causal.embedPacket)
        XCTAssertTrue(
            embedDiff.allSatisfy { !$0 },
            "block_ids must not affect the embed stage")

        let transformerDiff = try rowByteDifferences(
            bidirectional.lastTransformerPacket,
            causal.lastTransformerPacket)
        XCTAssertTrue(
            transformerDiff[..<imageStart].allSatisfy { !$0 },
            "rows before synthetic image span must stay byte-identical")
        XCTAssertTrue(
            transformerDiff[imageStart..<(imageStart + imageRows)].allSatisfy { $0 },
            "image rows must differ when shared block_ids enable bidirectional image attention")

        let summary = """
            [caix-mm-gemma-blockids] manifest=\(manifestURL.path)
            [caix-mm-gemma-blockids] available_ram_gib=\(String(format: "%.2f", availableRAMGB)) prompt_tokens=\(promptTokens.count) image_start=\(imageStart) image_rows=\(imageRows) hidden=\(hiddenSize)
            [caix-mm-gemma-blockids] embed_diff=\(boolVector(embedDiff)) differing_rows=\(rowIndices(embedDiff))
            [caix-mm-gemma-blockids] transformer_diff=\(boolVector(transformerDiff)) differing_rows=\(rowIndices(transformerDiff))
            [caix-mm-gemma-blockids] prefill_tokens bidirectional=\(bidirectional.prefillToken) causal=\(causal.prefillToken)
            [caix-mm-gemma-blockids] decode_tokens bidirectional=\(bidirectional.decodeToken) causal=\(causal.decodeToken)
            """
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_BLOCK_IDS_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealMultimodalGemmaStageBundleConsumesFrozenFixture() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }
        guard let fixturePath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_DIR"],
            !fixturePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set COREAI_MM_FIXTURE_DIR to a frozen multimodal fixture directory")
        }

        #if COREAI_RUNTIME
        let minAvailableRAMGB = loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_AVAILABLE_RAM_GB")
            ?? loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_FREE_RAM_GB")
            ?? 20
        let availableRAMGB = try currentAvailableMemoryGB()
        guard availableRAMGB >= minAvailableRAMGB else {
            throw XCTSkip(
                "available RAM \(String(format: "%.2f", availableRAMGB)) GiB is below "
                    + "\(String(format: "%.2f", minAvailableRAMGB)) GiB floor")
        }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        let blockIDsRequired = try MultimodalStagedModel.blockIDsRequired(manifestURL: manifestURL)
        let hiddenSize = try XCTUnwrap(manifest.boundaryTensor?.shape.last)
        let fixture = try loadMultimodalFixture(
            directory: URL(fileURLWithPath: fixturePath).standardizedFileURL,
            hiddenSize: hiddenSize)
        let embedderObservation = try await loadCaixEmbedderObservationIfRequested(
            fixture: fixture,
            fixtureDirectory: URL(fileURLWithPath: fixturePath).standardizedFileURL,
            hiddenSize: hiddenSize)
        if loadBoolEnvironment("CAIX_MM_EMBEDDER_COMPARE_ONLY") {
            let summary = """
                [caix-mm-gemma-fixture] manifest=\(manifestURL.path)
                [caix-mm-gemma-fixture] fixture_dir=\(fixturePath)
                [caix-mm-gemma-fixture] available_ram_gib=\(String(format: "%.2f", availableRAMGB)) tokens=\(fixture.inputIDs.count) image_start=\(fixture.imageStart) image_rows=\(fixture.imageRows) hidden=\(hiddenSize) soft_source=\(embedderObservation == nil ? "fixture_hf" : "caix_embedder")
                \(caixEmbedderSummaryLine(embedderObservation, imageRows: fixture.imageRows))
                [caix-mm-gemma-fixture] compare_only=true
                """
            FileHandle.standardError.write(Data((summary + "\n").utf8))
            if let outputPath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_OUTPUT"],
                !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
            }
            return
        }
        let activeSplice = embedderObservation?.splice ?? fixture.splice
        let maxContextLength = stagedMaxContextLength(manifestURL: manifestURL)
        let capacity = min(
            maxContextLength,
            max(fixture.inputIDs.count + max(2, fixture.reference.generatedTokenIDs.count) + 8, 2))
        let handles = try await makeCoreAIStageHandles(manifest: manifest)
        let active = try await runManualStagedPrefillAndOneDecode(
            handles: handles,
            manifest: manifest,
            requestID: "mm-fixture-active-\(UUID().uuidString)",
            capacity: capacity,
            tokenIDs: fixture.inputIDs,
            blockIDsQ: blockIDsRequired ? fixture.blockIDsQ : nil,
            blockIDsKV: blockIDsRequired ? fixture.blockIDsKV : nil,
            softTokenSplice: activeSplice,
            decodeInputToken: fixture.reference.generatedTokenIDs.first)

        if let bootstrapPath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_BOOTSTRAP_REFERENCE_OUTPUT"],
            !bootstrapPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try writeFixtureReference(
                topLogits: active.prefillTopLogits,
                generatedTokenIDs: [active.prefillToken, active.decodeToken],
                to: URL(fileURLWithPath: bootstrapPath).standardizedFileURL)
            if loadBoolEnvironment("COREAI_MM_FIXTURE_BOOTSTRAP_ONLY") {
                return
            }
        }

        let inactiveBlockIDs = blockIDsRequired
            ? Array(repeating: Int32(-1), count: fixture.inputIDs.count)
            : nil
        let inactive = try await runManualStagedPrefillAndOneDecode(
            handles: handles,
            manifest: manifest,
            requestID: "mm-fixture-inactive-\(UUID().uuidString)",
            capacity: capacity,
            tokenIDs: fixture.inputIDs,
            blockIDsQ: inactiveBlockIDs,
            blockIDsKV: inactiveBlockIDs,
            softTokenSplice: nil,
            decodeInputToken: fixture.reference.generatedTokenIDs.first)

        let referenceTop1 = try XCTUnwrap(fixture.reference.firstTopLogits.first?.tokenID)
        let observedTop1 = try XCTUnwrap(active.prefillTopLogits.first?.tokenID)
        XCTAssertEqual(
            observedTop1,
            referenceTop1,
            "fixture first-token top-1 must match frozen reference")
        let topKKL = truncatedTopKKL(
            reference: fixture.reference.firstTopLogits,
            observed: active.prefillTopLogits)
        let topKOverlap = topKTokenOverlap(
            fixture.reference.firstTopLogits,
            active.prefillTopLogits)
        let maxTopKKL = loadDoubleEnvironment("COREAI_MM_FIXTURE_MAX_TOPK_KL")
            ?? (blockIDsRequired ? 0.2 : 0.8)
        XCTAssertLessThanOrEqual(
            topKKL,
            maxTopKKL,
            "fixture first-token top-k KL exceeded threshold")
        let inactiveTop1 = try XCTUnwrap(inactive.prefillTopLogits.first?.tokenID)
        let activeInactiveTopKKL = truncatedTopKKL(
            reference: active.prefillTopLogits,
            observed: inactive.prefillTopLogits)
        let activeInactiveTopKOverlap = topKTokenOverlap(
            active.prefillTopLogits,
            inactive.prefillTopLogits)
        let teacherForcedActive = try await runManualStagedTeacherForcedTokens(
            handles: handles,
            manifest: manifest,
            requestID: "mm-fixture-tf-active-\(UUID().uuidString)",
            capacity: capacity,
            tokenIDs: fixture.inputIDs,
            blockIDsQ: blockIDsRequired ? fixture.blockIDsQ : nil,
            blockIDsKV: blockIDsRequired ? fixture.blockIDsKV : nil,
            softTokenSplice: activeSplice,
            referenceGeneratedTokenIDs: fixture.reference.generatedTokenIDs)
        let teacherForcedInactive = try await runManualStagedTeacherForcedTokens(
            handles: handles,
            manifest: manifest,
            requestID: "mm-fixture-tf-inactive-\(UUID().uuidString)",
            capacity: capacity,
            tokenIDs: fixture.inputIDs,
            blockIDsQ: inactiveBlockIDs,
            blockIDsKV: inactiveBlockIDs,
            softTokenSplice: nil,
            referenceGeneratedTokenIDs: fixture.reference.generatedTokenIDs)
        XCTAssertEqual(
            teacherForcedActive.count,
            fixture.reference.generatedTokenIDs.count,
            "mm-on teacher-forced observations must cover every fixture reference token")
        XCTAssertEqual(
            teacherForcedInactive.count,
            fixture.reference.generatedTokenIDs.count,
            "mm-off teacher-forced observations must cover every fixture reference token")
        let activeHFMatches = countTokenMatches(
            teacherForcedActive.map(\.tokenID),
            fixture.reference.generatedTokenIDs)
        let inactiveHFMatches = countTokenMatches(
            teacherForcedInactive.map(\.tokenID),
            fixture.reference.generatedTokenIDs)
        let firstActiveInactiveDivergence = firstTokenMismatch(
            teacherForcedActive.map(\.tokenID),
            teacherForcedInactive.map(\.tokenID))
        let minActiveHFMatches = loadIntEnvironment("COREAI_MM_FIXTURE_MIN_ON_HF_MATCHES")
            ?? (blockIDsRequired ? 28 : max(1, Int(Double(fixture.reference.generatedTokenIDs.count) * 0.80)))
        let maxInactiveHFMatches = loadIntEnvironment("COREAI_MM_FIXTURE_MAX_OFF_HF_MATCHES")
            ?? (blockIDsRequired ? 24 : Int(Double(fixture.reference.generatedTokenIDs.count) * 0.75))
        let minActiveInactiveMatchMargin =
            loadIntEnvironment("COREAI_MM_FIXTURE_MIN_ON_OFF_MATCH_MARGIN")
                ?? (blockIDsRequired ? 4 : max(4, fixture.reference.generatedTokenIDs.count / 8))
        let maxFirstDivergence = loadIntEnvironment("COREAI_MM_FIXTURE_MAX_FIRST_ON_OFF_DIVERGENCE")
            ?? 8
        let minActiveInactiveTopKKL =
            loadDoubleEnvironment("COREAI_MM_FIXTURE_MIN_ON_OFF_TOPK_KL") ?? 0.02
        XCTAssertGreaterThanOrEqual(
            activeHFMatches,
            minActiveHFMatches,
            "mm-on teacher-forced HF agreement is below the multimodal fixture floor")
        XCTAssertLessThanOrEqual(
            inactiveHFMatches,
            maxInactiveHFMatches,
            "mm-off teacher-forced HF agreement is too high; image-conditioned path is not discriminating")
        XCTAssertGreaterThan(
            activeHFMatches,
            inactiveHFMatches,
            "mm-on must agree with image-grounded HF more often than mm-off")
        XCTAssertGreaterThanOrEqual(
            activeHFMatches - inactiveHFMatches,
            minActiveInactiveMatchMargin,
            "mm-on vs mm-off HF agreement margin is too small")
        let firstDivergence = try XCTUnwrap(
            firstActiveInactiveDivergence,
            "mm-on and mm-off teacher-forced continuations must diverge")
        if blockIDsRequired {
            XCTAssertGreaterThanOrEqual(
                firstDivergence,
                1,
                "mm-on vs mm-off should not diverge before generated content starts")
        } else {
            XCTAssertGreaterThanOrEqual(
                firstDivergence,
                0,
                "causal E-series mm-on vs mm-off divergence may start at the structural opener")
        }
        XCTAssertLessThanOrEqual(
            firstDivergence,
            maxFirstDivergence,
            "mm-on vs mm-off must diverge near the first visual content tokens")
        XCTAssertGreaterThanOrEqual(
            activeInactiveTopKKL,
            minActiveInactiveTopKKL,
            "mm-on vs mm-off top-k KL must show image-conditioned logit movement")
        let teacherForcedTable = teacherForcedTokenTable(
            reference: fixture.reference.generatedTokenIDs,
            active: teacherForcedActive.map(\.tokenID),
            inactive: teacherForcedInactive.map(\.tokenID))
        let tokenizerURL = manifestURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer", isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let teacherForcedExamples = decodedTeacherForcedExamples(
            reference: fixture.reference.generatedTokenIDs,
            active: teacherForcedActive.map(\.tokenID),
            inactive: teacherForcedInactive.map(\.tokenID),
            start: firstActiveInactiveDivergence ?? 0,
            count: 3,
            tokenizer: tokenizer)

        let embedDiff = try rowByteDifferences(
            inactive.embedPacket,
            active.embedPacket)
        XCTAssertTrue(
            embedDiff[..<fixture.imageStart].allSatisfy { !$0 },
            "rows before fixture image span must stay byte-identical at embed")
        XCTAssertTrue(
            embedDiff[fixture.imageStart..<(fixture.imageStart + fixture.imageRows)].allSatisfy { $0 },
            "fixture image rows must differ after soft-token splice")
        try assertSoftTokenSpliceRows(
            activeSplice,
            areEmbeddedIn: active.embedPacket)

        let transformerDiff = try rowByteDifferences(
            inactive.lastTransformerPacket,
            active.lastTransformerPacket)
        XCTAssertTrue(
            transformerDiff[..<fixture.imageStart].allSatisfy { !$0 },
            "rows before fixture image span must stay byte-identical after transformers")
        XCTAssertTrue(
            transformerDiff[fixture.imageStart..<(fixture.imageStart + fixture.imageRows)].allSatisfy { $0 },
            "fixture image rows must differ after splice plus block_ids")

        let summary = """
            [caix-mm-gemma-fixture] manifest=\(manifestURL.path)
            [caix-mm-gemma-fixture] fixture_dir=\(fixturePath)
            [caix-mm-gemma-fixture] available_ram_gib=\(String(format: "%.2f", availableRAMGB)) tokens=\(fixture.inputIDs.count) image_start=\(fixture.imageStart) image_rows=\(fixture.imageRows) hidden=\(hiddenSize) soft_source=\(embedderObservation == nil ? "fixture_hf" : "caix_embedder")
            \(caixEmbedderSummaryLine(embedderObservation, imageRows: fixture.imageRows))
            [caix-mm-gemma-fixture] first_top1 observed=\(observedTop1) reference=\(referenceTop1) topk_overlap=\(topKOverlap)/\(fixture.reference.firstTopLogits.count) topk_kl=\(String(format: "%.6f", topKKL)) max_topk_kl=\(String(format: "%.6f", maxTopKKL))
            [caix-mm-gemma-fixture] mm_on_vs_off first_top1 active=\(observedTop1) inactive=\(inactiveTop1) topk_overlap=\(activeInactiveTopKOverlap)/\(active.prefillTopLogits.count) topk_kl=\(String(format: "%.6f", activeInactiveTopKKL))
            [caix-mm-gemma-fixture] teacher_forced steps=\(fixture.reference.generatedTokenIDs.count) mm_on_hf_top1=\(activeHFMatches)/\(fixture.reference.generatedTokenIDs.count) mm_off_hf_top1=\(inactiveHFMatches)/\(fixture.reference.generatedTokenIDs.count) first_on_off_divergence=\(firstActiveInactiveDivergence.map(String.init) ?? "none")
            [caix-mm-gemma-fixture] teacher_forced_tokens=\(teacherForcedTable)
            [caix-mm-gemma-fixture] teacher_forced_examples=\(teacherForcedExamples)
            [caix-mm-gemma-fixture] embed_diff=\(boolVector(embedDiff)) differing_rows=\(rowIndices(embedDiff))
            [caix-mm-gemma-fixture] transformer_diff=\(boolVector(transformerDiff)) differing_rows=\(rowIndices(transformerDiff))
            [caix-mm-gemma-fixture] prefill_token active=\(active.prefillToken) inactive=\(inactive.prefillToken)
            [caix-mm-gemma-fixture] decode_token active=\(active.decodeToken) inactive=\(inactive.decodeToken)
            """
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealMultimodalGemmaProductionLoopConsumesFrozenFixture() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }
        guard let fixturePath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_DIR"],
            !fixturePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set COREAI_MM_FIXTURE_DIR to a frozen multimodal fixture directory")
        }

        #if COREAI_RUNTIME
        let minAvailableRAMGB = loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_AVAILABLE_RAM_GB")
            ?? loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_FREE_RAM_GB")
            ?? 20
        let availableRAMGB = try currentAvailableMemoryGB()
        guard availableRAMGB >= minAvailableRAMGB else {
            throw XCTSkip(
                "available RAM \(String(format: "%.2f", availableRAMGB)) GiB is below "
                    + "\(String(format: "%.2f", minAvailableRAMGB)) GiB floor")
        }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        let hiddenSize = try XCTUnwrap(manifest.boundaryTensor?.shape.last)
        let fixtureDirectory = URL(fileURLWithPath: fixturePath).standardizedFileURL
        let fixture = try loadMultimodalFixture(directory: fixtureDirectory, hiddenSize: hiddenSize)
        let model = try await MultimodalStagedModel.load(manifestURL: manifestURL)
        let blockIDsRequired = try MultimodalStagedModel.blockIDsRequired(manifestURL: manifestURL)
        let active = try await model.teacherForcedTop1Tokens(
            inputIDs: fixture.inputIDs,
            blockIDsQ: blockIDsRequired ? fixture.blockIDsQ : nil,
            blockIDsKV: blockIDsRequired ? fixture.blockIDsKV : nil,
            softTokenSplice: fixture.splice,
            referenceTokenIDs: fixture.reference.generatedTokenIDs)
        let inactiveBlockIDs = blockIDsRequired
            ? Array(repeating: Int32(-1), count: fixture.inputIDs.count)
            : nil
        let inactive = try await model.teacherForcedTop1Tokens(
            inputIDs: fixture.inputIDs,
            blockIDsQ: inactiveBlockIDs,
            blockIDsKV: inactiveBlockIDs,
            softTokenSplice: nil,
            referenceTokenIDs: fixture.reference.generatedTokenIDs)

        let referenceTop1 = try XCTUnwrap(fixture.reference.firstTopLogits.first?.tokenID)
        let activeTop1 = try XCTUnwrap(active.first?.tokenID)
        XCTAssertEqual(activeTop1, referenceTop1)
        let activeHFMatches = countTokenMatches(
            active.map(\.tokenID),
            fixture.reference.generatedTokenIDs)
        let inactiveHFMatches = countTokenMatches(
            inactive.map(\.tokenID),
            fixture.reference.generatedTokenIDs)
        let firstDivergence = firstTokenMismatch(
            active.map(\.tokenID),
            inactive.map(\.tokenID))
        let minActiveHFMatches = loadIntEnvironment("COREAI_MM_FIXTURE_MIN_ON_HF_MATCHES")
            ?? (blockIDsRequired ? 28 : max(1, Int(Double(fixture.reference.generatedTokenIDs.count) * 0.80)))
        let maxInactiveHFMatches = loadIntEnvironment("COREAI_MM_FIXTURE_MAX_OFF_HF_MATCHES")
            ?? (blockIDsRequired ? 24 : Int(Double(fixture.reference.generatedTokenIDs.count) * 0.75))
        let minActiveInactiveMatchMargin =
            loadIntEnvironment("COREAI_MM_FIXTURE_MIN_ON_OFF_MATCH_MARGIN")
                ?? (blockIDsRequired ? 4 : max(4, fixture.reference.generatedTokenIDs.count / 8))
        let maxFirstDivergence = loadIntEnvironment("COREAI_MM_FIXTURE_MAX_FIRST_ON_OFF_DIVERGENCE") ?? 8
        XCTAssertGreaterThanOrEqual(activeHFMatches, minActiveHFMatches)
        XCTAssertLessThanOrEqual(inactiveHFMatches, maxInactiveHFMatches)
        XCTAssertGreaterThanOrEqual(activeHFMatches - inactiveHFMatches, minActiveInactiveMatchMargin)
        XCTAssertNotNil(firstDivergence)
        if let firstDivergence {
            XCTAssertLessThanOrEqual(firstDivergence, maxFirstDivergence)
        }

        let summary = """
            [caix-mm-gemma-production-loop] manifest=\(manifestURL.path)
            [caix-mm-gemma-production-loop] fixture_dir=\(fixturePath)
            [caix-mm-gemma-production-loop] available_ram_gib=\(String(format: "%.2f", availableRAMGB)) tokens=\(fixture.inputIDs.count) image_start=\(fixture.imageStart) image_rows=\(fixture.imageRows)
            [caix-mm-gemma-production-loop] first_top1 observed=\(activeTop1) reference=\(referenceTop1)
            [caix-mm-gemma-production-loop] teacher_forced steps=\(fixture.reference.generatedTokenIDs.count) mm_on_hf_top1=\(activeHFMatches)/\(fixture.reference.generatedTokenIDs.count) mm_off_hf_top1=\(inactiveHFMatches)/\(fixture.reference.generatedTokenIDs.count) first_on_off_divergence=\(firstDivergence.map(String.init) ?? "none")
            """
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["COREAI_MM_PRODUCTION_LOOP_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealMultimodalGemmaProductionLoopFreeRunsFrozenFixture() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }
        guard let fixturePath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_DIR"],
            !fixturePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set COREAI_MM_FIXTURE_DIR to a frozen multimodal fixture directory")
        }

        #if COREAI_RUNTIME
        let minAvailableRAMGB = loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_AVAILABLE_RAM_GB")
            ?? loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_FREE_RAM_GB")
            ?? 20
        let availableRAMGB = try currentAvailableMemoryGB()
        guard availableRAMGB >= minAvailableRAMGB else {
            throw XCTSkip(
                "available RAM \(String(format: "%.2f", availableRAMGB)) GiB is below "
                    + "\(String(format: "%.2f", minAvailableRAMGB)) GiB floor")
        }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        let hiddenSize = try XCTUnwrap(manifest.boundaryTensor?.shape.last)
        let fixtureDirectory = URL(fileURLWithPath: fixturePath).standardizedFileURL
        let fixture = try loadMultimodalFixture(directory: fixtureDirectory, hiddenSize: hiddenSize)
        let model = try await MultimodalStagedModel.load(manifestURL: manifestURL)
        let blockIDsRequired = try MultimodalStagedModel.blockIDsRequired(manifestURL: manifestURL)
        let embedderObservation = try await loadCaixEmbedderObservationIfRequested(
            fixture: fixture,
            fixtureDirectory: fixtureDirectory,
            hiddenSize: hiddenSize)
        let splice = embedderObservation?.splice ?? fixture.splice
        let maxTokens = loadIntEnvironment("COREAI_MM_FIXTURE_FREE_RUN_MAX_TOKENS") ?? 64
        let result = try await model.generate(
            input: MultimodalStagedGenerationInput(
                inputIDs: fixture.inputIDs,
                blockIDsQ: blockIDsRequired ? fixture.blockIDsQ : nil,
                blockIDsKV: blockIDsRequired ? fixture.blockIDsKV : nil,
                softTokenSplice: splice),
            options: CoreAIPipeline.Options(
                maxTokens: maxTokens,
                temperature: 0,
                applyChatTemplate: false,
                kvCapacity: fixture.inputIDs.count + maxTokens + 8),
            requestID: "mm-fixture-free-run-\(UUID().uuidString)",
            onToken: nil)
        XCTAssertEqual(result.promptTokenCount, fixture.inputIDs.count)
        XCTAssertGreaterThan(result.generatedTokenCount, 0)

        let summary = """
            [caix-mm-gemma-free-run] manifest=\(manifestURL.path)
            [caix-mm-gemma-free-run] fixture_dir=\(fixturePath)
            [caix-mm-gemma-free-run] available_ram_gib=\(String(format: "%.2f", availableRAMGB)) tokens=\(fixture.inputIDs.count) image_start=\(fixture.imageStart) image_rows=\(fixture.imageRows) soft_source=\(embedderObservation == nil ? "fixture_hf" : "caix_embedder")
            \(caixEmbedderSummaryLine(embedderObservation, imageRows: fixture.imageRows))
            [caix-mm-gemma-free-run] generated_tokens=\(result.generatedTokenIDs.map(String.init).joined(separator: ","))
            [caix-mm-gemma-free-run] text=\(result.text.replacingOccurrences(of: "\n", with: "\\n"))
            """
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_FREE_RUN_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealMultimodalGemmaLayerZeroMagnitudeForFrozenFixture() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }
        guard let fixturePath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_DIR"],
            !fixturePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set COREAI_MM_FIXTURE_DIR to a frozen multimodal fixture directory")
        }

        #if COREAI_RUNTIME
        let minAvailableRAMGB = loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_AVAILABLE_RAM_GB")
            ?? loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_FREE_RAM_GB")
            ?? 20
        let availableRAMGB = try currentAvailableMemoryGB()
        guard availableRAMGB >= minAvailableRAMGB else {
            throw XCTSkip(
                "available RAM \(String(format: "%.2f", availableRAMGB)) GiB is below "
                    + "\(String(format: "%.2f", minAvailableRAMGB)) GiB floor")
        }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        let hiddenSize = try XCTUnwrap(manifest.boundaryTensor?.shape.last)
        let fixtureDirectory = URL(fileURLWithPath: fixturePath).standardizedFileURL
        let fixture = try loadMultimodalFixture(directory: fixtureDirectory, hiddenSize: hiddenSize)
        let embedderObservation = try await loadCaixEmbedderObservationIfRequested(
            fixture: fixture,
            fixtureDirectory: fixtureDirectory,
            hiddenSize: hiddenSize)
        let splice = embedderObservation?.splice ?? fixture.splice
        let blockIDsRequired = try MultimodalStagedModel.blockIDsRequired(manifestURL: manifestURL)
        let handles = try await makeCoreAIStageHandles(manifest: manifest)
        let observation = try await runManualStagedPrefillAndOneDecode(
            handles: handles,
            manifest: manifest,
            requestID: "mm-layer0-magnitude-\(UUID().uuidString)",
            capacity: fixture.inputIDs.count + 8,
            tokenIDs: fixture.inputIDs,
            blockIDsQ: blockIDsRequired ? fixture.blockIDsQ : nil,
            blockIDsKV: blockIDsRequired ? fixture.blockIDsKV : nil,
            softTokenSplice: splice,
            decodeInputToken: fixture.reference.generatedTokenIDs.first)

        let shape = observation.embedPacket.metadata.shape
        guard shape.count == 3, shape[0] == 1, shape[1] == fixture.inputIDs.count, shape[2] == hiddenSize else {
            XCTFail("unexpected embed packet shape \(shape)")
            return
        }
        let embeddedValues: [Float]
        switch observation.embedPacket.metadata.scalarType {
        case .float16:
            embeddedValues = try observation.embedPacket.float16Values().map(Float.init)
        case .float32:
            embeddedValues = try observation.embedPacket.float32Values()
        default:
            throw XCTSkip("unsupported embed packet scalar type \(observation.embedPacket.metadata.scalarType)")
        }

        let spliceValues: [Float]
        switch splice.scalarType {
        case .float16:
            spliceValues = try splice.float16Values().map(Float.init)
        case .float32:
            spliceValues = try splice.float32Values()
        }

        func norm(values: [Float], row: Int, baseRows: Int, hidden: Int) -> Double {
            XCTAssertGreaterThanOrEqual(row, 0)
            XCTAssertLessThan(row, baseRows)
            var sum = 0.0
            let offset = row * hidden
            for column in 0..<hidden {
                let value = Double(values[offset + column])
                sum += value * value
            }
            return sqrt(sum)
        }

        func normSummary(_ norms: [Double]) -> (min: Double, mean: Double, max: Double) {
            guard let first = norms.first else { return (.nan, .nan, .nan) }
            var minValue = first
            var maxValue = first
            var sum = 0.0
            for value in norms {
                minValue = min(minValue, value)
                maxValue = max(maxValue, value)
                sum += value
            }
            return (minValue, sum / Double(norms.count), maxValue)
        }

        let imageRows = fixture.imageStart..<(fixture.imageStart + fixture.imageRows)
        let caixImageNorms = imageRows.map {
            norm(values: embeddedValues, row: $0, baseRows: shape[1], hidden: hiddenSize)
        }
        let spliceImageNorms = (0..<fixture.imageRows).map {
            norm(values: spliceValues, row: $0, baseRows: fixture.imageRows, hidden: hiddenSize)
        }
        let caixImageSummary = normSummary(caixImageNorms)
        let spliceImageSummary = normSummary(spliceImageNorms)
        let textBeforeImageRow = max(0, fixture.imageStart - 1)
        let textAfterImageRow = min(fixture.inputIDs.count - 1, fixture.imageStart + fixture.imageRows + 1)
        let textBeforeImageNorm = norm(
            values: embeddedValues,
            row: textBeforeImageRow,
            baseRows: shape[1],
            hidden: hiddenSize)
        let textAfterImageNorm = norm(
            values: embeddedValues,
            row: textAfterImageRow,
            baseRows: shape[1],
            hidden: hiddenSize)

        let summary = """
            [caix-mm-layer0-magnitude] manifest=\(manifestURL.path)
            [caix-mm-layer0-magnitude] fixture_dir=\(fixturePath)
            [caix-mm-layer0-magnitude] available_ram_gib=\(String(format: "%.2f", availableRAMGB)) tokens=\(fixture.inputIDs.count) image_start=\(fixture.imageStart) image_rows=\(fixture.imageRows) hidden=\(hiddenSize) soft_source=\(embedderObservation == nil ? "fixture_hf" : "caix_embedder")
            \(caixEmbedderSummaryLine(embedderObservation, imageRows: fixture.imageRows))
            [caix-mm-layer0-magnitude] caix_image_norm_min=\(String(format: "%.9g", caixImageSummary.min)) caix_image_norm_mean=\(String(format: "%.9g", caixImageSummary.mean)) caix_image_norm_max=\(String(format: "%.9g", caixImageSummary.max))
            [caix-mm-layer0-magnitude] splice_image_norm_min=\(String(format: "%.9g", spliceImageSummary.min)) splice_image_norm_mean=\(String(format: "%.9g", spliceImageSummary.mean)) splice_image_norm_max=\(String(format: "%.9g", spliceImageSummary.max))
            [caix-mm-layer0-magnitude] text_before_image_row=\(textBeforeImageRow) text_before_image_norm=\(String(format: "%.9g", textBeforeImageNorm)) text_after_image_row=\(textAfterImageRow) text_after_image_norm=\(String(format: "%.9g", textAfterImageNorm))
            """
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["COREAI_MM_MAGNITUDE_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealMultimodalGemmaStageBundleRunsNearMaxContextPrefillAndDecode() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }
        guard let requestedTokens = loadIntEnvironment("CAIX_MM_GEMMA_LONG_CONTEXT_TOKENS") else {
            throw XCTSkip("set CAIX_MM_GEMMA_LONG_CONTEXT_TOKENS to run the long-context allocation gate")
        }

        #if COREAI_RUNTIME
        let minAvailableRAMGB = loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_AVAILABLE_RAM_GB")
            ?? loadDoubleEnvironment("CAIX_MM_GEMMA_MIN_FREE_RAM_GB")
            ?? 20
        let availableBeforeGB = try currentAvailableMemoryGB()
        guard availableBeforeGB >= minAvailableRAMGB else {
            throw XCTSkip(
                "available RAM \(String(format: "%.2f", availableBeforeGB)) GiB is below "
                    + "\(String(format: "%.2f", minAvailableRAMGB)) GiB floor")
        }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        _ = try DistributedStageManifest.load(from: manifestURL)
        let maxContextLength = stagedMaxContextLength(manifestURL: manifestURL)
        let tokenCount = min(max(2, requestedTokens), maxContextLength - 1)
        let requestedGenerateTokens =
            loadIntEnvironment("CAIX_MM_GEMMA_LONG_CONTEXT_GENERATE_TOKENS") ?? 2
        let requestedCapacity = loadIntEnvironment("CAIX_MM_GEMMA_LONG_CONTEXT_KV_CAPACITY")
            ?? (tokenCount + max(1, requestedGenerateTokens))
        let capacity = min(max(requestedCapacity, tokenCount + 1), maxContextLength)
        let blockIDsRequired = try MultimodalStagedModel.blockIDsRequired(manifestURL: manifestURL)
        let blockIDs = blockIDsRequired ? Array(repeating: Int32(-1), count: tokenCount) : nil
        var tokenIDs = Array(repeating: Int32(818), count: tokenCount)
        tokenIDs[0] = 2
        let markers = try loadTokenMarkerEnvironment(
            "CAIX_MM_GEMMA_LONG_CONTEXT_MARKERS",
            tokenCount: tokenCount)
        for marker in markers {
            tokenIDs[marker.index] = marker.tokenID
        }

        let started = Date()
        let model = try await MultimodalStagedModel.load(manifestURL: manifestURL)
        let expectedGeneratedTokens = min(requestedGenerateTokens, maxContextLength - tokenCount)
        XCTAssertGreaterThan(expectedGeneratedTokens, 0)
        let result = try await model.generate(
            input: MultimodalStagedGenerationInput(
                inputIDs: tokenIDs,
                blockIDsQ: blockIDs,
                blockIDsKV: blockIDs,
                softTokenSplice: nil),
            options: CoreAIPipeline.Options(
                maxTokens: expectedGeneratedTokens,
                temperature: 0,
                applyChatTemplate: false,
                kvCapacity: capacity),
            requestID: "mm-gemma-long-context-\(UUID().uuidString)",
            onToken: nil)
        let elapsed = Date().timeIntervalSince(started)
        let availableAfterGB = try currentAvailableMemoryGB()
        XCTAssertGreaterThanOrEqual(tokenCount, requestedTokens)
        XCTAssertGreaterThanOrEqual(capacity, tokenCount + 1)
        XCTAssertEqual(result.generatedTokenIDs.count, expectedGeneratedTokens)
        for tokenID in result.generatedTokenIDs {
            XCTAssertGreaterThanOrEqual(tokenID, 0)
        }

        let summary = """
            [caix-mm-gemma-long-context] manifest=\(manifestURL.path)
            [caix-mm-gemma-long-context] requested_tokens=\(requestedTokens) prefill_tokens=\(tokenCount) kv_capacity=\(capacity) max_context_length=\(maxContextLength)
            [caix-mm-gemma-long-context] markers=\(markers.map { "\($0.index):\($0.tokenID)" }.joined(separator: ","))
            [caix-mm-gemma-long-context] available_ram_gib_before=\(String(format: "%.2f", availableBeforeGB)) available_ram_gib_after=\(String(format: "%.2f", availableAfterGB)) elapsed_s=\(String(format: "%.3f", elapsed))
            [caix-mm-gemma-long-context] generated_count=\(result.generatedTokenIDs.count) generated_tokens=\(result.generatedTokenIDs.map(String.init).joined(separator: ","))
            """
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_LONG_CONTEXT_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testGemma4SwiftImagePreprocessingAndTokenBuilderMatchFrozenFixture() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_MM_GEMMA_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_MM_GEMMA_STAGE_MANIFEST to the multimodal Gemma staged manifest")
        }
        guard let fixturePath = ProcessInfo.processInfo.environment["COREAI_MM_FIXTURE_DIR"],
            !fixturePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set COREAI_MM_FIXTURE_DIR to a frozen multimodal fixture directory")
        }

        #if COREAI_RUNTIME
        let fixtureDirectory = URL(fileURLWithPath: fixturePath).standardizedFileURL
        let imageData = try Data(contentsOf: fixtureDirectory.appendingPathComponent("procedural_input.png"))
        let reference = try loadFloat32NPY(fixtureDirectory.appendingPathComponent("pixel_values.npy"))
        let referencePositionIDs = try loadInt32NPY(
            fixtureDirectory.appendingPathComponent("image_position_ids.npy"))
        let observed = try Gemma4MultimodalProcessor.preprocessImage(
            data: imageData,
            pixelValuesShape: reference.shape,
            imagePositionIDsShape: referencePositionIDs.shape)
        XCTAssertEqual(observed.pixelValuesShape, reference.shape)
        XCTAssertEqual(observed.imagePositionIDsShape, referencePositionIDs.shape)
        XCTAssertEqual(observed.imagePositionIDs, referencePositionIDs.values)
        let pixelDelta = compareFloatArrays(observed.pixelValues, reference.values)

        let tokens = try loadFixtureTokensJSON(fixtureDirectory.appendingPathComponent("tokens.json"))
        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let config = try MultimodalStagedModel.configuration(manifestURL: manifestURL)
        let imageTextSeparator = config.kind == "gemma4" ? "" : "\n"
        let tokenizerURL = manifestURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer", isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let prompt = try Gemma4MultimodalProcessor.buildSingleImagePrompt(
            messages: [["role": "user", "content": "<|image|>\(imageTextSeparator)\(tokens.prompt)"]],
            tokenizer: tokenizer,
            imageRows: tokens.imageRows,
            blockIDsRequired: tokens.blockIDsQ.contains { $0 >= 0 } || tokens.blockIDsKV.contains { $0 >= 0 })
        XCTAssertEqual(prompt.inputIDs, tokens.inputIDs)
        XCTAssertEqual(prompt.imageStart, tokens.imageStart)
        XCTAssertEqual(prompt.imageRows, tokens.imageRows)
        XCTAssertEqual(prompt.blockIDsQ, tokens.blockIDsQ)
        XCTAssertEqual(prompt.blockIDsKV, tokens.blockIDsKV)

        let maxPixelDelta = loadDoubleEnvironment("COREAI_MM_SWIFT_PREPROCESS_MAX_PIXEL_DELTA") ?? 0.1
        XCTAssertLessThanOrEqual(
            pixelDelta.maxAbsDelta,
            maxPixelDelta,
            "Swift Gemma image preprocessing must stay within the validated PIL-bicubic-family bound")

        let summary = """
            [caix-mm-gemma-swift-preprocess] fixture_dir=\(fixturePath)
            [caix-mm-gemma-swift-preprocess] pixel_shape=\(observed.pixelValuesShape) position_shape=\(observed.imagePositionIDsShape) valid_rows=\(observed.validRows)
            [caix-mm-gemma-swift-preprocess] pixel_max_abs_delta=\(String(format: "%.9g", pixelDelta.maxAbsDelta)) pixel_mean_abs_delta=\(String(format: "%.9g", pixelDelta.meanAbsDelta)) threshold=\(String(format: "%.9g", maxPixelDelta))
            [caix-mm-gemma-swift-preprocess] token_exact=true imgStart=\(prompt.imageStart) N=\(prompt.imageRows) input_ids=\(prompt.inputIDs.count)
            """
        FileHandle.standardError.write(Data((summary + "\n").utf8))
        if let outputPath = ProcessInfo.processInfo.environment["COREAI_MM_SWIFT_PREPROCESS_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            try (summary + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
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

    func testRealMonolithicGreedyTokenSequencesAreDeterministic() async throws {
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
        let repeats = max(
            2,
            Int(ProcessInfo.processInfo.environment["CAIX_MONOLITHIC_DETERMINISM_REPEATS"] ?? "")
                ?? 3)
        let baselineBundle = try ResolvedBundle.load(at: baselinePath)
        var reference: [[Int32]]?

        for repeatIndex in 0..<repeats {
            let baseline = try await LLMEngine.load(bundle: baselineBundle)
            var sequences: [[Int32]] = []
            for (promptIndex, prompt) in prompts.enumerated() {
                let promptTokens = try baseline.encodePrompt(
                    messages: [["role": "user", "content": prompt]],
                    applyChatTemplate: false)
                guard !promptTokens.isEmpty else {
                    XCTFail("prompt \(promptIndex) tokenized to 0 tokens")
                    return
                }
                let sequence = try await monolithicGreedyTokenSequence(
                    baseline,
                    promptTokens: promptTokens,
                    maxTokens: maxTokens)
                sequences.append(sequence)
            }
            if reference == nil {
                reference = sequences
                continue
            }
            guard let reference else {
                XCTFail("missing monolithic determinism reference")
                return
            }
            if let mismatch = firstSequenceMismatch(reference, sequences) {
                XCTFail(
                    "monolithic greedy sequence changed at repeat \(repeatIndex), "
                        + "prompt \(mismatch.prompt), token \(mismatch.token): "
                        + "reference \(mismatch.expected), observed \(mismatch.observed)")
                return
            }
        }
        if let reference {
            try writeGreedyTokenSequencesIfRequested(reference)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealPersistentPipelinedGreedyTokenSequencesAreDeterministic() async throws {
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
        let repeats = max(
            1,
            Int(ProcessInfo.processInfo.environment["CAIX_MONOLITHIC_DETERMINISM_REPEATS"] ?? "")
                ?? 3)
        var reference: [[Int32]]?

        for repeatIndex in 0..<repeats {
            guard let handle = try await PipelinedLLM.loadPersistent(
                bundlePath: baselinePath,
                verbose: true)
            else {
                throw XCTSkip("bundle is not accepted by the CoreAILM persistent pipelined engine")
            }

            var sequences: [[Int32]] = []
            for prompt in prompts {
                let result = try await handle.generate(
                    [["role": "user", "content": prompt]],
                    CoreAIPipeline.Options(
                        maxTokens: maxTokens,
                        temperature: 0,
                        applyChatTemplate: false,
                        verbose: true),
                    nil,
                    nil)
                XCTAssertEqual(result.generatedTokenIDs.count, result.generatedTokenCount)
                sequences.append(result.generatedTokenIDs)
            }

            if reference == nil {
                reference = sequences
                continue
            }
            guard let reference else {
                XCTFail("missing persistent pipelined determinism reference")
                return
            }
            if let mismatch = firstSequenceMismatch(reference, sequences) {
                XCTFail(
                    "persistent pipelined greedy sequence changed at repeat \(repeatIndex), "
                        + "prompt \(mismatch.prompt), token \(mismatch.token): "
                        + "reference \(mismatch.expected), observed \(mismatch.observed)")
                return
            }
        }
        if let reference {
            try writeGreedyTokenSequencesIfRequested(reference)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealStageAssetsMatchExpectedGreedyTokens() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_STAGE_MANIFEST to a real staged manifest")
        }

        #if COREAI_RUNTIME
        let promptFile = ProcessInfo.processInfo.environment["CAIX_TOKEN_MATCH_PROMPTS"]
        let prompts = try loadPrompts(path: promptFile)
        let expectedTokens = try loadExpectedGreedyTokens()
        XCTAssertEqual(
            expectedTokens.count,
            prompts.count,
            "expected greedy token count must match prompt count")
        guard expectedTokens.count == prompts.count else { return }

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        let tokenizerURL = manifestURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer", isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let maxContextLength = stagedMaxContextLength(manifestURL: manifestURL)
        let pipeline = try await DistributedSameMachinePipeline.make(
            manifest: manifest,
            handleFactory: DistributedCoreAIStageHandleFactory())

        for (index, prompt) in prompts.enumerated() {
            let promptTokens = tokenizer.encode(text: prompt)
            guard !promptTokens.isEmpty else {
                XCTFail("prompt \(index) tokenized to 0 tokens")
                return
            }
            let capacity = min(maxContextLength, max(promptTokens.count + 9, 2))
            let requestID = "expected-token-\(index)"
            try await pipeline.allocate(requestID: requestID, kvCapacity: capacity)
            let stagedNext = try await nextStagedToken(
                pipeline: pipeline,
                requestID: requestID,
                stepIndex: 0,
                lowerBound: 0,
                tokenIDs: promptTokens.map(Int32.init))
            await pipeline.free(requestID: requestID)
            XCTAssertEqual(
                stagedNext,
                expectedTokens[index],
                "prompt \(index) prefill token should match reference")
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRealStageAssetsMatchExpectedGreedyTokenSequences() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_STAGE_MANIFEST to a real staged manifest")
        }

        #if COREAI_RUNTIME
        let promptFile = ProcessInfo.processInfo.environment["CAIX_TOKEN_MATCH_PROMPTS"]
        let prompts = try loadPrompts(path: promptFile)
        let expectedSequences = try loadExpectedGreedyTokenSequences()
        XCTAssertEqual(
            expectedSequences.count,
            prompts.count,
            "expected greedy sequence count must match prompt count")
        guard expectedSequences.count == prompts.count else { return }
        let tieTolerance = loadDoubleEnvironment("CAIX_EXPECTED_GREEDY_TIE_TOLERANCE") ?? 0.02

        let manifestURL = URL(fileURLWithPath: manifestPath).standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        let tokenizerURL = manifestURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer", isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let maxContextLength = stagedMaxContextLength(manifestURL: manifestURL)
        let pipeline = try await DistributedSameMachinePipeline.make(
            manifest: manifest,
            handleFactory: DistributedCoreAIStageHandleFactory())

        for (index, prompt) in prompts.enumerated() {
            let promptTokens = tokenizer.encode(text: prompt)
            guard !promptTokens.isEmpty else {
                XCTFail("prompt \(index) tokenized to 0 tokens")
                return
            }
            let expected = expectedSequences[index]
            guard !expected.isEmpty else {
                XCTFail("prompt \(index) expected sequence is empty")
                return
            }
            let capacity = min(maxContextLength, max(promptTokens.count + expected.count + 8, 2))
            let requestID = "expected-sequence-\(index)"
            try await pipeline.allocate(requestID: requestID, kvCapacity: capacity)

            var lowerBound = 0
            var tokenIDs = promptTokens.map(Int32.init)
            for (step, expectedToken) in expected.enumerated() {
                let observation = try await nextStagedTokenObservation(
                    pipeline: pipeline,
                    requestID: requestID,
                    stepIndex: step,
                    lowerBound: lowerBound,
                    tokenIDs: tokenIDs)
                if observation.tokenID != expectedToken,
                    let tie = acceptedTeacherForcedTie(
                        observation: observation,
                        expectedToken: expectedToken,
                        tolerance: tieTolerance)
                {
                    traceAcceptedTie(
                        tie,
                        promptIndex: index,
                        step: step,
                        expectedToken: expectedToken,
                        observedToken: observation.tokenID)
                } else if observation.tokenID != expectedToken {
                    XCTFail(
                        "prompt \(index) greedy token \(step) mismatch under teacher forcing: "
                            + "staged \(observation.tokenID), expected \(expectedToken); "
                            + "top logits {\(describeTopLogits(observation.topLogits))}")
                    await pipeline.free(requestID: requestID)
                    return
                }
                lowerBound = promptTokens.count + step
                tokenIDs = [expectedToken]
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

    func testRealTransformerStageMatchesHiddenFixtureSummary() async throws {
        guard let manifestPath = ProcessInfo.processInfo.environment["CAIX_STAGE_MANIFEST"],
            !manifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_STAGE_MANIFEST to a real staged manifest")
        }
        guard let stageID = ProcessInfo.processInfo.environment["CAIX_TRANSFORMER_STAGE_ID"],
            !stageID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_TRANSFORMER_STAGE_ID to a transformer stage id")
        }
        guard let hiddenPath = ProcessInfo.processInfo.environment["CAIX_TRANSFORMER_HIDDEN_F16"],
            !hiddenPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_TRANSFORMER_HIDDEN_F16 to a float16 hidden-state fixture")
        }
        guard let expectedL2 = loadDoubleEnvironment("CAIX_TRANSFORMER_EXPECTED_L2"),
            let expectedSumAbs = loadDoubleEnvironment("CAIX_TRANSFORMER_EXPECTED_SUM_ABS")
        else {
            throw XCTSkip("set CAIX_TRANSFORMER_EXPECTED_L2 and CAIX_TRANSFORMER_EXPECTED_SUM_ABS")
        }
        let tolerance = loadDoubleEnvironment("CAIX_TRANSFORMER_SUMMARY_TOLERANCE") ?? 0.02

        #if COREAI_RUNTIME
        let manifest = try DistributedStageManifest.load(
            from: URL(fileURLWithPath: manifestPath).standardizedFileURL)
        guard let boundaryTensor = manifest.boundaryTensor else {
            throw XCTSkip("manifest has no boundary tensor metadata")
        }
        XCTAssertEqual(boundaryTensor.scalarType, .float16)
        let hiddenSize = try XCTUnwrap(boundaryTensor.shape.last)
        let hiddenData = try Data(contentsOf: URL(fileURLWithPath: hiddenPath))
        XCTAssertEqual(hiddenData.count % boundaryTensor.scalarType.byteWidth, 0)
        let elementCount = hiddenData.count / boundaryTensor.scalarType.byteWidth
        XCTAssertEqual(elementCount % hiddenSize, 0)
        let positionCount = elementCount / hiddenSize
        XCTAssertGreaterThan(positionCount, 0)

        let stageIndex = try XCTUnwrap(manifest.runtimePlan.stages.firstIndex { $0.id == stageID })
        let descriptor = manifest.runtimePlan.stages[stageIndex]
        XCTAssertEqual(descriptor.role, .transformerLayers)
        let stage = try XCTUnwrap(manifest.stages.first { $0.id == descriptor.id })
        let sourceStageID = stageIndex > 0 ? manifest.runtimePlan.stages[stageIndex - 1].id : "fixture"
        let requestID = "transformer-fixture"
        let positionRange = DistributedSequenceRange(lowerBound: 0, upperBound: positionCount)
        let metadata = DistributedHiddenStatePacketMetadata(
            requestID: requestID,
            sourceStageID: sourceStageID,
            destinationStageID: descriptor.id,
            positionRange: positionRange,
            shape: [1, positionCount, hiddenSize],
            scalarType: .float16,
            byteCount: hiddenData.count,
            stepIndex: 0)
        let packet = try DistributedHiddenStatePacket(
            metadata: metadata,
            payload: Array(hiddenData))

        let context = DistributedStageHandleFactoryContext(
            stage: stage,
            manifest: manifest,
            descriptor: descriptor)
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

        let outputPacket = try XCTUnwrap(output.hiddenState)
        XCTAssertEqual(outputPacket.metadata.sourceStageID, descriptor.id)
        let summary = try summarizeFloat16Payload(outputPacket.payload)
        XCTAssertEqual(summary.count, elementCount)
        assertClose(summary.l2, expectedL2, tolerance: tolerance, label: "l2")
        assertClose(summary.sumAbs, expectedSumAbs, tolerance: tolerance, label: "sum_abs")
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

    private func loadExpectedGreedyTokens() throws -> [Int32] {
        let environment = ProcessInfo.processInfo.environment
        let raw: String?
        if let path = environment["CAIX_EXPECTED_GREEDY_TOKENS_FILE"],
            !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            raw = try String(contentsOfFile: path, encoding: .utf8)
        } else {
            raw = environment["CAIX_EXPECTED_GREEDY_TOKENS"]
                ?? environment["CAIX_EXPECTED_GREEDY_TOKEN"]
        }
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("set CAIX_EXPECTED_GREEDY_TOKENS to reference token ids")
        }
        let fields = raw.split { character in
            character == "," || character == ";" || character.isWhitespace
        }
        let tokens = fields.compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        guard tokens.count == fields.count, !tokens.isEmpty else {
            throw XCTSkip("CAIX_EXPECTED_GREEDY_TOKENS must contain Int32 token ids")
        }
        return tokens
    }

    private func loadExpectedGreedyTokenSequences() throws -> [[Int32]] {
        let environment = ProcessInfo.processInfo.environment
        let raw: String?
        if let path = environment["CAIX_EXPECTED_GREEDY_TOKEN_SEQUENCES_FILE"],
            !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            raw = try String(contentsOfFile: path, encoding: .utf8)
        } else {
            raw = environment["CAIX_EXPECTED_GREEDY_TOKEN_SEQUENCES"]
        }
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("set CAIX_EXPECTED_GREEDY_TOKEN_SEQUENCES to reference token ids")
        }
        let sequences = raw.split(whereSeparator: \.isNewline).map { line -> [Int32]? in
            let fields = line.split { character in
                character == "," || character == ";" || character.isWhitespace
            }
            let tokens = fields.compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            guard tokens.count == fields.count else { return nil }
            return tokens
        }
        guard !sequences.isEmpty, sequences.allSatisfy({ $0?.isEmpty == false }) else {
            throw XCTSkip("CAIX_EXPECTED_GREEDY_TOKEN_SEQUENCES must contain complete Int32 token ids")
        }
        return sequences.compactMap(\.self)
    }

    private func writeGreedyTokenSequencesIfRequested(_ sequences: [[Int32]]) throws {
        guard let path = ProcessInfo.processInfo.environment["CAIX_OBSERVED_GREEDY_TOKEN_SEQUENCES_FILE"],
            !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return
        }
        let text = sequences
            .map { sequence in sequence.map(String.init).joined(separator: ",") }
            .joined(separator: "\n") + "\n"
        try text.write(toFile: path, atomically: true, encoding: .utf8)
    }

    private func stagedMaxContextLength(manifestURL: URL) -> Int {
        let metadataURL = manifestURL
            .deletingLastPathComponent()
            .appendingPathComponent("metadata.json")
        guard let data = try? Data(contentsOf: metadataURL),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let language = object["language"] as? [String: Any]
        else {
            return 8192
        }
        if let value = language["max_context_length"] as? Int {
            return value
        }
        if let value = language["max_context_length"] as? Double {
            return Int(value)
        }
        return 8192
    }

    #if COREAI_RUNTIME
    private func monolithicGreedyTokenSequence(
        _ baseline: LLMEngine,
        promptTokens: [Int],
        maxTokens: Int
    ) async throws -> [Int32] {
        let result = try await baseline.generate(
            promptTokens: promptTokens,
            options: CoreAIPipeline.Options(
                maxTokens: maxTokens,
                temperature: 0,
                applyChatTemplate: false),
            onToken: nil)
        return result.generatedTokenIDs
    }

    private struct ManualStagedObservation {
        let embedPacket: DistributedHiddenStatePacket
        let lastTransformerPacket: DistributedHiddenStatePacket
        let prefillToken: Int32
        let decodeToken: Int32
        let prefillTopLogits: [DistributedLogitScore]
        let decodeTopLogits: [DistributedLogitScore]
    }

    private struct MultimodalFixture {
        let inputIDs: [Int32]
        let imageStart: Int
        let imageRows: Int
        let splice: DistributedSoftTokenSplice
        let blockIDsQ: [Int32]
        let blockIDsKV: [Int32]
        let reference: MultimodalFixtureReference
    }

    private struct MultimodalFixtureReference {
        let firstTopLogits: [DistributedLogitScore]
        let generatedTokenIDs: [Int32]
        let stepTopLogits: [[DistributedLogitScore]]
    }

    private struct CaixEmbedderObservation {
        let assetPath: String
        let computeMode: String
        let preprocessingSource: String
        let outputShape: [Int]
        let validRows: Int
        let splice: DistributedSoftTokenSplice
        let comparison: SoftTokenComparison
    }

    private struct SoftTokenComparison {
        let observedNonFiniteCount: Int
        let referenceNonFiniteCount: Int
        let finitePairCount: Int
        let maxAbsDelta: Double
        let meanAbsDelta: Double
        let rowCosineValidCount: Int
        let rowCosineMin: Double
        let rowCosineMean: Double
        let rowCosineMax: Double
    }

    private struct NPYArray {
        let descr: String
        let shape: [Int]
        let payload: [UInt8]
    }

    private struct FixtureTokenJSON {
        let prompt: String
        let inputIDs: [Int32]
        let imageStart: Int
        let imageRows: Int
        let blockIDsQ: [Int32]
        let blockIDsKV: [Int32]
    }

    private struct FloatArrayDelta {
        let maxAbsDelta: Double
        let meanAbsDelta: Double
    }

    private func makeCoreAIStageHandles(
        manifest: DistributedStageManifest
    ) async throws -> [DistributedStageHandle] {
        var handles: [DistributedStageHandle] = []
        let factory = DistributedCoreAIStageHandleFactory()
        for stage in manifest.stages {
            let descriptor = try XCTUnwrap(manifest.runtimePlan.stage(id: stage.id))
            handles.append(try await factory.makeStageHandle(for: DistributedStageHandleFactoryContext(
                stage: stage,
                manifest: manifest,
                descriptor: descriptor)))
        }
        return handles
    }

    private func runManualStagedPrefillAndOneDecode(
        handles: [DistributedStageHandle],
        manifest: DistributedStageManifest,
        requestID: String,
        capacity: Int,
        tokenIDs: [Int32],
        blockIDsQ: [Int32]?,
        blockIDsKV: [Int32]?,
        softTokenSplice: DistributedSoftTokenSplice?,
        decodeInputToken: Int32? = nil
    ) async throws -> ManualStagedObservation {
        let allocation = DistributedStageAllocation(requestID: requestID, kvCapacity: capacity)
        for handle in handles {
            try await handle.allocate(allocation)
        }
        do {
            let prefill = try await runManualStagedStep(
                handles: handles,
                manifest: manifest,
                requestID: requestID,
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: tokenIDs.count),
                tokenIDs: tokenIDs,
                blockIDsQ: blockIDsQ,
                blockIDsKV: blockIDsKV,
                softTokenSplice: softTokenSplice)
            let prefillToken = try XCTUnwrap(prefill.tokenID)
            let decodeTokenID = decodeInputToken ?? prefillToken
            let decode = try await runManualStagedStep(
                handles: handles,
                manifest: manifest,
                requestID: requestID,
                stepIndex: 1,
                positionRange: DistributedSequenceRange(
                    lowerBound: tokenIDs.count,
                    upperBound: tokenIDs.count + 1),
                tokenIDs: [decodeTokenID],
                blockIDsQ: nil,
                blockIDsKV: nil,
                softTokenSplice: nil)
            let decodeToken = try XCTUnwrap(decode.tokenID)
            for handle in handles {
                await handle.free(requestID: requestID)
            }
            return ManualStagedObservation(
                embedPacket: prefill.embedPacket,
                lastTransformerPacket: prefill.lastTransformerPacket,
                prefillToken: prefillToken,
                decodeToken: decodeToken,
                prefillTopLogits: prefill.topLogits,
                decodeTopLogits: decode.topLogits)
        } catch {
            for handle in handles {
                await handle.free(requestID: requestID)
            }
            throw error
        }
    }

    private func runManualStagedTeacherForcedTokens(
        handles: [DistributedStageHandle],
        manifest: DistributedStageManifest,
        requestID: String,
        capacity: Int,
        tokenIDs: [Int32],
        blockIDsQ: [Int32]?,
        blockIDsKV: [Int32]?,
        softTokenSplice: DistributedSoftTokenSplice?,
        referenceGeneratedTokenIDs: [Int32]
    ) async throws -> [StagedTokenObservation] {
        guard !referenceGeneratedTokenIDs.isEmpty else { return [] }
        let allocation = DistributedStageAllocation(requestID: requestID, kvCapacity: capacity)
        for handle in handles {
            try await handle.allocate(allocation)
        }
        do {
            var observations: [StagedTokenObservation] = []
            observations.reserveCapacity(referenceGeneratedTokenIDs.count)
            let prefill = try await runManualStagedStep(
                handles: handles,
                manifest: manifest,
                requestID: requestID,
                stepIndex: 0,
                positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: tokenIDs.count),
                tokenIDs: tokenIDs,
                blockIDsQ: blockIDsQ,
                blockIDsKV: blockIDsKV,
                softTokenSplice: softTokenSplice)
            observations.append(StagedTokenObservation(
                tokenID: try XCTUnwrap(prefill.tokenID),
                topLogits: prefill.topLogits))

            for step in 1..<referenceGeneratedTokenIDs.count {
                let decode = try await runManualStagedStep(
                    handles: handles,
                    manifest: manifest,
                    requestID: requestID,
                    stepIndex: step,
                    positionRange: DistributedSequenceRange(
                        lowerBound: tokenIDs.count + step - 1,
                        upperBound: tokenIDs.count + step),
                    tokenIDs: [referenceGeneratedTokenIDs[step - 1]],
                    blockIDsQ: nil,
                    blockIDsKV: nil,
                    softTokenSplice: nil)
                observations.append(StagedTokenObservation(
                    tokenID: try XCTUnwrap(decode.tokenID),
                    topLogits: decode.topLogits))
            }
            for handle in handles {
                await handle.free(requestID: requestID)
            }
            return observations
        } catch {
            for handle in handles {
                await handle.free(requestID: requestID)
            }
            throw error
        }
    }

    private func loadCaixEmbedderObservationIfRequested(
        fixture: MultimodalFixture,
        fixtureDirectory: URL,
        hiddenSize: Int
    ) async throws -> CaixEmbedderObservation? {
        guard let embedderPath = ProcessInfo.processInfo.environment["CAIX_MM_EMBEDDER_ASSET"],
            !embedderPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }
        let assetURL = URL(fileURLWithPath: embedderPath).standardizedFileURL
        let useSwiftPreprocess = loadBoolEnvironment("CAIX_MM_EMBEDDER_USE_SWIFT_PREPROCESS")
        let preprocessingSource: String
        let pixelValues: (shape: [Int], values: [Float])
        let imagePositionIDs: [Int32]
        if useSwiftPreprocess {
            let imageData = try Data(
                contentsOf: fixtureDirectory.appendingPathComponent("procedural_input.png"))
            let referencePixels = try loadFloat32NPY(
                fixtureDirectory.appendingPathComponent("pixel_values.npy"))
            let referencePositionIDs = try loadInt32NPY(
                fixtureDirectory.appendingPathComponent("image_position_ids.npy"))
            let processed = try Gemma4MultimodalProcessor.preprocessImage(
                data: imageData,
                pixelValuesShape: referencePixels.shape,
                imagePositionIDsShape: referencePositionIDs.shape)
            preprocessingSource = "swift"
            pixelValues = (shape: processed.pixelValuesShape, values: processed.pixelValues)
            imagePositionIDs = processed.imagePositionIDs
        } else {
            preprocessingSource = "fixture_hf"
            pixelValues = try loadFloat32NPY(
                fixtureDirectory.appendingPathComponent("pixel_values.npy"))
            imagePositionIDs = try loadInt32NPY(
                fixtureDirectory.appendingPathComponent("image_position_ids.npy")).values
        }
        let caixRows = try await runCaixVisionEmbedder(
            assetURL: assetURL,
            pixelValues: pixelValues,
            imagePositionIDs: imagePositionIDs)
        XCTAssertEqual(caixRows.shape.count, 3)
        XCTAssertEqual(caixRows.shape.first, 1)
        XCTAssertEqual(caixRows.shape.last, hiddenSize)
        XCTAssertGreaterThanOrEqual(caixRows.shape[1], fixture.imageRows)
        guard caixRows.shape.count == 3,
            caixRows.shape[0] == 1,
            caixRows.shape[1] >= fixture.imageRows,
            caixRows.shape[2] == hiddenSize
        else {
            throw XCTSkip("caix embedder output shape \(caixRows.shape) is incompatible with fixture")
        }
        let validRows = fixture.imageRows
        let rowValues = Array(caixRows.values.prefix(fixture.imageRows * hiddenSize))
        let comparison = try compareSoftTokens(
            observed: rowValues,
            reference: fixture.splice.float16Values(),
            rows: fixture.imageRows,
            hiddenSize: hiddenSize)
        return CaixEmbedderObservation(
            assetPath: assetURL.path,
            computeMode: caixEmbedderComputeMode(),
            preprocessingSource: preprocessingSource,
            outputShape: caixRows.shape,
            validRows: validRows,
            splice: try DistributedSoftTokenSplice(
                positionStart: fixture.imageStart,
                rows: fixture.imageRows,
                hiddenSize: hiddenSize,
                float16Values: rowValues),
            comparison: comparison)
    }

    private func runCaixVisionEmbedder(
        assetURL: URL,
        pixelValues: (shape: [Int], values: [Float]),
        imagePositionIDs: [Int32]
    ) async throws -> (shape: [Int], values: [Float16]) {
        var specialization = caixEmbedderSpecializationOptions()
        specialization.expectFrequentReshapes = true
        let model = try await AIModel.specialize(
            contentsOf: assetURL,
            options: specialization,
            cache: .default,
            cachePolicy: .persistent)
        guard let descriptor = model.functionDescriptor(for: "vision") else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "caix mm embedder function 'vision' not found in \(assetURL.lastPathComponent); have \(model.functionNames)")
        }
        guard case .ndArray(let pixelDescriptor) = descriptor.inputDescriptor(of: "pixel_values"),
            case .ndArray(let positionDescriptor) = descriptor.inputDescriptor(of: "image_position_ids"),
            case .ndArray = descriptor.outputDescriptor(of: "mm_embeds")
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "caix mm embedder must expose pixel_values, image_position_ids, and mm_embeds NDArrays")
        }
        guard pixelValues.shape == pixelDescriptor.shape else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "pixel_values shape \(pixelValues.shape) does not match embedder \(pixelDescriptor.shape)")
        }
        guard imagePositionIDs.count == positionDescriptor.shape.reduce(1, *) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "image_position_ids count \(imagePositionIDs.count) does not match embedder \(positionDescriptor.shape)")
        }
        guard let function = try model.loadFunction(named: "vision") else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "could not load caix mm embedder function 'vision'")
        }
        var inputs: [String: NDArray] = [:]
        inputs["pixel_values"] = try makeFloatNDArray(
            shape: pixelValues.shape,
            scalarType: pixelDescriptor.scalarType,
            values: pixelValues.values,
            tensorName: "pixel_values")
        inputs["image_position_ids"] = try makeInt32NDArray(
            shape: positionDescriptor.shape,
            scalarType: positionDescriptor.scalarType,
            values: imagePositionIDs,
            tensorName: "image_position_ids")
        var outputs = try await function.run(inputs: inputs)
        guard let output = outputs.remove("mm_embeds")?.ndArray else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "caix mm embedder did not produce mm_embeds")
        }
        return try readRank3AsFloat16(output, tensorName: "mm_embeds")
    }

    private func caixEmbedderComputeMode() -> String {
        ProcessInfo.processInfo.environment["CAIX_MM_EMBEDDER_COMPUTE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            ?? "gpu"
    }

    private func caixEmbedderSpecializationOptions() -> SpecializationOptions {
        switch caixEmbedderComputeMode() {
        case "cpu", "cpuonly", "cpu_only":
            return .cpuOnly
        case "default", "all":
            return .default
        case "cpu_gpu", "cpugpu", "cpuandgpu", "cpu_and_gpu":
            return .default
        case "ane", "neuralengine", "neural_engine":
            return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        default:
            return SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
    }

    private func caixEmbedderSummaryLine(
        _ observation: CaixEmbedderObservation?,
        imageRows: Int
    ) -> String {
        guard let observation else {
            return "[caix-mm-gemma-fixture] caix_embedder skipped"
        }
        let comparison = observation.comparison
        return "[caix-mm-gemma-fixture] caix_embedder asset=\(observation.assetPath) "
            + "compute=\(observation.computeMode) "
            + "preprocess=\(observation.preprocessingSource) "
            + "output_shape=\(observation.outputShape) valid_rows=\(observation.validRows) "
            + "observed_nonfinite=\(comparison.observedNonFiniteCount) "
            + "reference_nonfinite=\(comparison.referenceNonFiniteCount) "
            + "finite_pairs=\(comparison.finitePairCount) "
            + "max_abs_delta=\(String(format: "%.6g", comparison.maxAbsDelta)) "
            + "mean_abs_delta=\(String(format: "%.6g", comparison.meanAbsDelta)) "
            + "row_cosine_valid=\(comparison.rowCosineValidCount)/\(imageRows) "
            + "row_cosine_min=\(String(format: "%.6g", comparison.rowCosineMin)) "
            + "row_cosine_mean=\(String(format: "%.6g", comparison.rowCosineMean)) "
            + "row_cosine_max=\(String(format: "%.6g", comparison.rowCosineMax))"
    }

    private struct ManualStagedStepOutput {
        let embedPacket: DistributedHiddenStatePacket
        let lastTransformerPacket: DistributedHiddenStatePacket
        let tokenID: Int32?
        let topLogits: [DistributedLogitScore]
    }

    private func runManualStagedStep(
        handles: [DistributedStageHandle],
        manifest: DistributedStageManifest,
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        tokenIDs: [Int32],
        blockIDsQ: [Int32]?,
        blockIDsKV: [Int32]?,
        softTokenSplice: DistributedSoftTokenSplice?
    ) async throws -> ManualStagedStepOutput {
        let positionIDs = try manifest.positionMode.positionIDs(for: positionRange)
        var hiddenState: DistributedHiddenStatePacket?
        var embedPacket: DistributedHiddenStatePacket?
        var lastTransformerPacket: DistributedHiddenStatePacket?
        var tokenID: Int32?
        var topLogits: [DistributedLogitScore] = []

        for handle in handles {
            let output = try await handle.forward(DistributedStageForwardInput(
                requestID: requestID,
                stepIndex: stepIndex,
                positionRange: positionRange,
                positionIDs: positionIDs,
                tokenIDs: handle.acceptsTokenIDs ? tokenIDs : [],
                blockIDsQ: handle.descriptor.role == .transformerLayers ? blockIDsQ : nil,
                blockIDsKV: handle.descriptor.role == .transformerLayers ? blockIDsKV : nil,
                softTokenSplice: handle.descriptor.role == .embeddings ? softTokenSplice : nil,
                hiddenState: hiddenState))
            if handle.descriptor.role == .embeddings {
                embedPacket = output.hiddenState
            }
            if handle.descriptor.role == .transformerLayers {
                lastTransformerPacket = output.hiddenState
            }
            hiddenState = output.hiddenState
            tokenID = output.tokenID
            if !output.topLogits.isEmpty {
                topLogits = output.topLogits
            }
        }

        return ManualStagedStepOutput(
            embedPacket: try XCTUnwrap(embedPacket),
            lastTransformerPacket: try XCTUnwrap(lastTransformerPacket),
            tokenID: tokenID,
            topLogits: topLogits)
    }

    private func loadMultimodalFixture(
        directory: URL,
        hiddenSize: Int
    ) throws -> MultimodalFixture {
        let tokens = try loadFixtureTokens(
            directory.appendingPathComponent("tokens.json"))
        let softImage = try loadFloat16NPY(
            directory.appendingPathComponent("soft_image.npy"))
        guard softImage.shape.count == 2 else {
            throw XCTSkip("soft_image.npy must have shape [N, hidden]")
        }
        let imageRows = tokens.imageRows ?? softImage.shape[0]
        XCTAssertEqual(softImage.shape[0], imageRows)
        XCTAssertEqual(softImage.shape[1], hiddenSize)
        XCTAssertEqual(softImage.values.count, imageRows * hiddenSize)
        guard softImage.shape[0] == imageRows,
            softImage.shape[1] == hiddenSize,
            softImage.values.count == imageRows * hiddenSize
        else {
            throw XCTSkip("soft_image.npy shape does not match tokens.json or manifest hidden size")
        }
        let imageStart = tokens.imageStart
        guard imageStart >= 0,
            imageRows > 0,
            tokens.inputIDs.count >= imageStart + imageRows
        else {
            throw XCTSkip("tokens.json image span is outside input_ids")
        }

        let blockIDsQ = try loadInt32NPY(
            directory.appendingPathComponent("block_ids_q.npy"),
            expectedCount: tokens.inputIDs.count)
        let blockIDsKV = try loadInt32NPY(
            directory.appendingPathComponent("block_ids_kv.npy"),
            expectedCount: tokens.inputIDs.count)
        if blockIDsQ.contains(where: { $0 >= 0 }) || blockIDsKV.contains(where: { $0 >= 0 }) {
            try validateImageBlockIDs(
                blockIDsQ,
                imageStart: imageStart,
                imageRows: imageRows,
                label: "block_ids_q.npy")
            try validateImageBlockIDs(
                blockIDsKV,
                imageStart: imageStart,
                imageRows: imageRows,
                label: "block_ids_kv.npy")
        } else {
            XCTAssertTrue(blockIDsQ.allSatisfy { $0 == -1 })
            XCTAssertTrue(blockIDsKV.allSatisfy { $0 == -1 })
        }

        return MultimodalFixture(
            inputIDs: tokens.inputIDs,
            imageStart: imageStart,
            imageRows: imageRows,
            splice: try DistributedSoftTokenSplice(
                positionStart: imageStart,
                rows: imageRows,
                hiddenSize: hiddenSize,
                float16Values: softImage.values),
            blockIDsQ: blockIDsQ,
            blockIDsKV: blockIDsKV,
            reference: try loadFixtureReference(
                directory.appendingPathComponent("hf_reference.json")))
    }

    private func loadFixtureTokens(
        _ url: URL
    ) throws -> (inputIDs: [Int32], imageStart: Int, imageRows: Int?) {
        let object = try loadJSONObject(url)
        let inputIDs = try intArray(
            object["input_ids"] ?? object["inputIDs"],
            label: "tokens.json input_ids")
        let imageStart = try intValue(
            object["imgStart"] ?? object["img_start"] ?? object["image_start"],
            label: "tokens.json imgStart")
        let imageRowsValue = try optionalIntValue(
            object["N"] ?? object["n"] ?? object["image_rows"] ?? object["imageRows"],
            label: "tokens.json N")
        return (inputIDs.map(Int32.init), imageStart, imageRowsValue)
    }

    private func loadFixtureTokensJSON(_ url: URL) throws -> FixtureTokenJSON {
        let object = try loadJSONObject(url)
        let tokens = try loadFixtureTokens(url)
        let imageRows = try XCTUnwrap(tokens.imageRows, "tokens.json must carry N for mm fixture")
        let prompt = try stringValue(object["prompt"], label: "tokens.json prompt")
        let directory = url.deletingLastPathComponent()
        let blockIDsQ = try loadInt32NPY(
            directory.appendingPathComponent("block_ids_q.npy"),
            expectedCount: tokens.inputIDs.count)
        let blockIDsKV = try loadInt32NPY(
            directory.appendingPathComponent("block_ids_kv.npy"),
            expectedCount: tokens.inputIDs.count)
        return FixtureTokenJSON(
            prompt: prompt,
            inputIDs: tokens.inputIDs,
            imageStart: tokens.imageStart,
            imageRows: imageRows,
            blockIDsQ: blockIDsQ,
            blockIDsKV: blockIDsKV)
    }

    private func loadFixtureReference(_ url: URL) throws -> MultimodalFixtureReference {
        let object = try loadJSONObject(url)
        let topLogitsValue = object["first_top_logits"] ?? object["top_logits"]
        let firstTopLogits = try logitScores(
            topLogitsValue,
            label: "hf_reference.json first_top_logits")
        let generated = try optionalIntArray(
            object["generated_token_ids"] ?? object["greedy_token_ids"],
            label: "hf_reference.json generated_token_ids")
        let stepTopLogits = try optionalLogitScoreSequences(
            object["step_top_logits"],
            label: "hf_reference.json step_top_logits")
        guard !firstTopLogits.isEmpty else {
            throw XCTSkip("hf_reference.json first_top_logits must not be empty")
        }
        return MultimodalFixtureReference(
            firstTopLogits: firstTopLogits,
            generatedTokenIDs: generated.map(Int32.init),
            stepTopLogits: stepTopLogits)
    }

    private func loadJSONObject(_ url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("\(url.lastPathComponent) must contain a JSON object")
        }
        return object
    }

    private func loadFloat16NPY(_ url: URL) throws -> (shape: [Int], values: [Float16]) {
        let array = try loadNPY(url)
        guard array.descr == "<f2" || array.descr == "|f2" else {
            throw XCTSkip("\(url.lastPathComponent) must be float16, got \(array.descr)")
        }
        guard array.payload.count.isMultiple(of: 2) else {
            throw XCTSkip("\(url.lastPathComponent) float16 payload has odd byte count")
        }
        var values: [Float16] = []
        values.reserveCapacity(array.payload.count / 2)
        var offset = 0
        while offset < array.payload.count {
            let bits = UInt16(array.payload[offset])
                | (UInt16(array.payload[offset + 1]) << 8)
            values.append(Float16(bitPattern: bits))
            offset += 2
        }
        return (array.shape, values)
    }

    private func loadFloat32NPY(_ url: URL) throws -> (shape: [Int], values: [Float]) {
        let array = try loadNPY(url)
        guard array.descr == "<f4" || array.descr == "|f4" else {
            throw XCTSkip("\(url.lastPathComponent) must be float32, got \(array.descr)")
        }
        guard array.payload.count.isMultiple(of: 4) else {
            throw XCTSkip("\(url.lastPathComponent) float32 payload has invalid byte count")
        }
        var values: [Float] = []
        values.reserveCapacity(array.payload.count / 4)
        var offset = 0
        while offset < array.payload.count {
            let bits = UInt32(array.payload[offset])
                | (UInt32(array.payload[offset + 1]) << 8)
                | (UInt32(array.payload[offset + 2]) << 16)
                | (UInt32(array.payload[offset + 3]) << 24)
            values.append(Float(bitPattern: bits))
            offset += 4
        }
        return (array.shape, values)
    }

    private func loadInt32NPY(
        _ url: URL,
        expectedCount: Int
    ) throws -> [Int32] {
        let array = try loadNPY(url)
        guard array.descr == "<i4" || array.descr == "|i4" else {
            throw XCTSkip("\(url.lastPathComponent) must be int32, got \(array.descr)")
        }
        guard array.payload.count == expectedCount * 4 else {
            throw XCTSkip(
                "\(url.lastPathComponent) payload count \(array.payload.count / 4) "
                    + "does not match expected \(expectedCount)")
        }
        let elementCount = array.shape.reduce(1, *)
        guard elementCount == expectedCount else {
            throw XCTSkip(
                "\(url.lastPathComponent) shape \(array.shape) does not match expected \(expectedCount)")
        }
        var values: [Int32] = []
        values.reserveCapacity(expectedCount)
        var offset = 0
        while offset < array.payload.count {
            let bits = UInt32(array.payload[offset])
                | (UInt32(array.payload[offset + 1]) << 8)
                | (UInt32(array.payload[offset + 2]) << 16)
                | (UInt32(array.payload[offset + 3]) << 24)
            values.append(Int32(bitPattern: bits))
            offset += 4
        }
        return values
    }

    private func loadInt32NPY(
        _ url: URL
    ) throws -> (shape: [Int], values: [Int32]) {
        let array = try loadNPY(url)
        guard array.descr == "<i4" || array.descr == "|i4" else {
            throw XCTSkip("\(url.lastPathComponent) must be int32, got \(array.descr)")
        }
        guard array.payload.count.isMultiple(of: 4) else {
            throw XCTSkip("\(url.lastPathComponent) int32 payload has invalid byte count")
        }
        let expectedCount = array.shape.reduce(1, *)
        guard array.payload.count == expectedCount * 4 else {
            throw XCTSkip(
                "\(url.lastPathComponent) payload count \(array.payload.count / 4) "
                    + "does not match shape \(array.shape)")
        }
        var values: [Int32] = []
        values.reserveCapacity(expectedCount)
        var offset = 0
        while offset < array.payload.count {
            let bits = UInt32(array.payload[offset])
                | (UInt32(array.payload[offset + 1]) << 8)
                | (UInt32(array.payload[offset + 2]) << 16)
                | (UInt32(array.payload[offset + 3]) << 24)
            values.append(Int32(bitPattern: bits))
            offset += 4
        }
        return (array.shape, values)
    }

    private func loadNPY(_ url: URL) throws -> NPYArray {
        let data = try Data(contentsOf: url)
        guard data.count >= 10,
            data[0] == 0x93,
            data[1] == UInt8(ascii: "N"),
            data[2] == UInt8(ascii: "U"),
            data[3] == UInt8(ascii: "M"),
            data[4] == UInt8(ascii: "P"),
            data[5] == UInt8(ascii: "Y")
        else {
            throw XCTSkip("\(url.lastPathComponent) is not a NumPy .npy file")
        }
        let major = data[6]
        let headerLength: Int
        let headerStart: Int
        if major == 1 {
            headerLength = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else if major == 2 || major == 3 {
            guard data.count >= 12 else {
                throw XCTSkip("\(url.lastPathComponent) has a truncated .npy v\(major) header")
            }
            headerLength = Int(data[8])
                | (Int(data[9]) << 8)
                | (Int(data[10]) << 16)
                | (Int(data[11]) << 24)
            headerStart = 12
        } else {
            throw XCTSkip("\(url.lastPathComponent) has unsupported .npy version \(major)")
        }
        let payloadStart = headerStart + headerLength
        guard data.count >= payloadStart else {
            throw XCTSkip("\(url.lastPathComponent) has a truncated .npy header")
        }
        guard let header = String(
            data: data.subdata(in: headerStart..<payloadStart),
            encoding: .ascii)
        else {
            throw XCTSkip("\(url.lastPathComponent) has a non-ASCII .npy header")
        }
        let descr = try parseNPYStringValue(header, key: "descr", fileName: url.lastPathComponent)
        let fortranOrder = try parseNPYBoolValue(
            header,
            key: "fortran_order",
            fileName: url.lastPathComponent)
        guard !fortranOrder else {
            throw XCTSkip("\(url.lastPathComponent) must be C-order, not Fortran-order")
        }
        let shape = try parseNPYShape(header, fileName: url.lastPathComponent)
        return NPYArray(
            descr: descr,
            shape: shape,
            payload: Array(data[payloadStart..<data.count]))
    }

    private func parseNPYStringValue(
        _ header: String,
        key: String,
        fileName: String
    ) throws -> String {
        guard let keyRange = header.range(of: "'\(key)'") ?? header.range(of: "\"\(key)\""),
            let colon = header[keyRange.upperBound...].firstIndex(of: ":")
        else {
            throw XCTSkip("\(fileName) .npy header missing \(key)")
        }
        let tail = header[header.index(after: colon)...]
        guard let quote = tail.firstIndex(where: { $0 == "'" || $0 == "\"" }) else {
            throw XCTSkip("\(fileName) .npy header has invalid \(key)")
        }
        let quoteChar = tail[quote]
        let valueStart = header.index(after: quote)
        guard let valueEnd = header[valueStart...].firstIndex(of: quoteChar) else {
            throw XCTSkip("\(fileName) .npy header has unterminated \(key)")
        }
        return String(header[valueStart..<valueEnd])
    }

    private func parseNPYBoolValue(
        _ header: String,
        key: String,
        fileName: String
    ) throws -> Bool {
        guard let keyRange = header.range(of: "'\(key)'") ?? header.range(of: "\"\(key)\""),
            let colon = header[keyRange.upperBound...].firstIndex(of: ":")
        else {
            throw XCTSkip("\(fileName) .npy header missing \(key)")
        }
        let tail = header[header.index(after: colon)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if tail.hasPrefix("False") || tail.hasPrefix("false") { return false }
        if tail.hasPrefix("True") || tail.hasPrefix("true") { return true }
        throw XCTSkip("\(fileName) .npy header has invalid \(key)")
    }

    private func parseNPYShape(
        _ header: String,
        fileName: String
    ) throws -> [Int] {
        guard let keyRange = header.range(of: "'shape'") ?? header.range(of: "\"shape\""),
            let openParen = header[keyRange.upperBound...].firstIndex(of: "("),
            let closeParen = header[openParen...].firstIndex(of: ")")
        else {
            throw XCTSkip("\(fileName) .npy header missing shape")
        }
        let body = header[header.index(after: openParen)..<closeParen]
        let values = body.split(separator: ",").compactMap { raw -> Int? in
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            return Int(trimmed)
        }
        guard !values.isEmpty else {
            throw XCTSkip("\(fileName) .npy header has empty shape")
        }
        return values
    }

    private func intArray(_ value: Any?, label: String) throws -> [Int] {
        guard let array = value as? [Any] else {
            throw XCTSkip("\(label) must be an array")
        }
        return try array.map { try intValue($0, label: label) }
    }

    private func optionalIntArray(_ value: Any?, label: String) throws -> [Int] {
        guard let value else { return [] }
        return try intArray(value, label: label)
    }

    private func intValue(_ value: Any?, label: String) throws -> Int {
        if let value = value as? Int { return value }
        if let value = value as? Int32 { return Int(value) }
        if let value = value as? Double, value.rounded() == value { return Int(value) }
        if let value = value as? String, let parsed = Int(value) { return parsed }
        throw XCTSkip("\(label) must be an integer")
    }

    private func optionalIntValue(_ value: Any?, label: String) throws -> Int? {
        guard let value else { return nil }
        return try intValue(value, label: label)
    }

    private func stringValue(_ value: Any?, label: String) throws -> String {
        guard let value = value as? String else {
            throw XCTSkip("\(label) must be a string")
        }
        return value
    }

    private func logitScores(_ value: Any?, label: String) throws -> [DistributedLogitScore] {
        guard let array = value as? [Any] else {
            throw XCTSkip("\(label) must be an array")
        }
        return try array.map { item in
            guard let object = item as? [String: Any] else {
                throw XCTSkip("\(label) entries must be objects")
            }
            let tokenID = try intValue(
                object["token_id"] ?? object["tokenID"],
                label: "\(label) token_id")
            let logit = try floatValue(
                object["logit"],
                label: "\(label) logit")
            return DistributedLogitScore(tokenID: Int32(tokenID), logit: logit)
        }
    }

    private func optionalLogitScoreSequences(
        _ value: Any?,
        label: String
    ) throws -> [[DistributedLogitScore]] {
        guard let value else { return [] }
        guard let array = value as? [Any] else {
            throw XCTSkip("\(label) must be an array")
        }
        return try array.enumerated().map { index, item in
            try logitScores(item, label: "\(label)[\(index)]")
        }
    }

    private func floatValue(_ value: Any?, label: String) throws -> Float {
        if let value = value as? Float { return value }
        if let value = value as? Double { return Float(value) }
        if let value = value as? Int { return Float(value) }
        if let value = value as? String, let parsed = Float(value) { return parsed }
        throw XCTSkip("\(label) must be numeric")
    }

    private func validateImageBlockIDs(
        _ values: [Int32],
        imageStart: Int,
        imageRows: Int,
        label: String
    ) throws {
        guard imageStart + imageRows <= values.count else {
            throw XCTSkip("\(label) image span is outside block id vector")
        }
        let imageIDs = Array(values[imageStart..<(imageStart + imageRows)])
        guard let blockID = imageIDs.first,
            blockID >= 0,
            imageIDs.allSatisfy({ $0 == blockID })
        else {
            throw XCTSkip("\(label) image span must share one non-negative id")
        }
        let outsideIDs = Array(values[..<imageStart])
            + Array(values[(imageStart + imageRows)..<values.count])
        guard outsideIDs.allSatisfy({ $0 == -1 }) else {
            throw XCTSkip("\(label) non-image rows must be -1")
        }
    }

    private func truncatedTopKKL(
        reference: [DistributedLogitScore],
        observed: [DistributedLogitScore]
    ) -> Double {
        let referenceMap = Dictionary(uniqueKeysWithValues: reference.map { ($0.tokenID, Double($0.logit)) })
        let observedMap = Dictionary(uniqueKeysWithValues: observed.map { ($0.tokenID, Double($0.logit)) })
        let tokens = Set(referenceMap.keys).union(observedMap.keys)
        let referenceFloor = (referenceMap.values.min() ?? 0) - 20
        let observedFloor = (observedMap.values.min() ?? 0) - 20
        let referenceLogits = Dictionary(uniqueKeysWithValues: tokens.map {
            ($0, referenceMap[$0] ?? referenceFloor)
        })
        let observedLogits = Dictionary(uniqueKeysWithValues: tokens.map {
            ($0, observedMap[$0] ?? observedFloor)
        })
        let referenceProbs = softmax(referenceLogits)
        let observedProbs = softmax(observedLogits)
        return tokens.reduce(0) { partial, tokenID in
            let p = max(referenceProbs[tokenID] ?? 0, 1e-45)
            let q = max(observedProbs[tokenID] ?? 0, 1e-45)
            return partial + p * log(p / q)
        }
    }

    private func topKTokenOverlap(
        _ lhs: [DistributedLogitScore],
        _ rhs: [DistributedLogitScore]
    ) -> Int {
        Set(lhs.map(\.tokenID)).intersection(Set(rhs.map(\.tokenID))).count
    }

    private func countTokenMatches(_ lhs: [Int32], _ rhs: [Int32]) -> Int {
        zip(lhs, rhs).filter { $0 == $1 }.count
    }

    private func firstTokenMismatch(_ lhs: [Int32], _ rhs: [Int32]) -> Int? {
        let count = min(lhs.count, rhs.count)
        for index in 0..<count where lhs[index] != rhs[index] {
            return index
        }
        return lhs.count == rhs.count ? nil : count
    }

    private func teacherForcedTokenTable(
        reference: [Int32],
        active: [Int32],
        inactive: [Int32]
    ) -> String {
        let count = min(reference.count, active.count, inactive.count)
        return (0..<count)
            .map { index in
                "\(index):hf=\(reference[index]),on=\(active[index]),off=\(inactive[index])"
            }
            .joined(separator: ";")
    }

    private func decodedTeacherForcedExamples(
        reference: [Int32],
        active: [Int32],
        inactive: [Int32],
        start: Int,
        count: Int,
        tokenizer: any Tokenizer
    ) -> String {
        let end = min(max(start, 0) + count, reference.count, active.count, inactive.count)
        guard start < end else { return "none" }
        return (start..<end)
            .map { index in
                let hf = decodedToken(reference[index], tokenizer: tokenizer)
                let on = decodedToken(active[index], tokenizer: tokenizer)
                let off = decodedToken(inactive[index], tokenizer: tokenizer)
                return "\(index):hf=\(reference[index])/\(hf),on=\(active[index])/\(on),off=\(inactive[index])/\(off)"
            }
            .joined(separator: ";")
    }

    private func decodedToken(_ tokenID: Int32, tokenizer: any Tokenizer) -> String {
        let decoded = tokenizer.decode(tokens: [Int(tokenID)], skipSpecialTokens: false)
        if decoded.isEmpty { return "<empty>" }
        return decoded
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
    }

    private func makeFloatNDArray(
        shape: [Int],
        scalarType: NDArray.ScalarType,
        values: [Float],
        tensorName: String
    ) throws -> NDArray {
        guard shape.reduce(1, *) == values.count else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) element count \(values.count) does not match shape \(shape)")
        }
        var array = NDArray(shape: shape, scalarType: scalarType)
        switch scalarType {
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: values.map(Float16.init))
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: values)
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(scalarType) is not supported")
        }
        return array
    }

    private func makeInt32NDArray(
        shape: [Int],
        scalarType: NDArray.ScalarType,
        values: [Int32],
        tensorName: String
    ) throws -> NDArray {
        guard shape.reduce(1, *) == values.count else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) element count \(values.count) does not match shape \(shape)")
        }
        var array = NDArray(shape: shape, scalarType: scalarType)
        switch scalarType {
        case .int32:
            var view = array.mutableView(as: Int32.self)
            view.copyElements(fromContentsOf: values)
        case .int64:
            var view = array.mutableView(as: Int64.self)
            view.copyElements(fromContentsOf: values.map(Int64.init))
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(scalarType) is not supported")
        }
        return array
    }

    private func readRank3AsFloat16(
        _ array: NDArray,
        tensorName: String
    ) throws -> (shape: [Int], values: [Float16]) {
        switch array.scalarType {
        case .float16:
            return try readRank3FloatingPoint(
                array,
                as: Float16.self,
                tensorName: tensorName) { $0 }
        case .float32:
            return try readRank3FloatingPoint(
                array,
                as: Float.self,
                tensorName: tensorName) { Float16($0) }
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(array.scalarType) is not float16 or float32")
        }
    }

    private func readRank3FloatingPoint<T: BinaryFloatingPoint & BitwiseCopyable>(
        _ array: NDArray,
        as _: T.Type,
        tensorName: String,
        convert: (T) -> Float16
    ) throws -> (shape: [Int], values: [Float16]) {
        return try array.view(as: T.self).withUnsafePointer { pointer, shape, strides in
            let viewShape = copySpan(shape)
            let viewStrides = copySpan(strides)
            guard viewShape.count == 3 else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "\(tensorName) must be rank 3, got \(viewShape)")
            }
            var values: [Float16] = []
            values.reserveCapacity(viewShape.reduce(1, *))
            for i in 0..<viewShape[0] {
                for j in 0..<viewShape[1] {
                    for k in 0..<viewShape[2] {
                        values.append(convert(pointer[
                            viewStrides[0] * i
                                + viewStrides[1] * j
                                + viewStrides[2] * k]))
                    }
                }
            }
            return (viewShape, values)
        }
    }

    private func validImagePositionRowCount(_ values: [Int32]) -> Int {
        var count = 0
        var index = 0
        while index + 1 < values.count {
            if values[index] >= 0 && values[index + 1] >= 0 {
                count += 1
            }
            index += 2
        }
        return count
    }

    private func compareSoftTokens(
        observed: [Float16],
        reference: [Float16],
        rows: Int,
        hiddenSize: Int
    ) throws -> SoftTokenComparison {
        guard observed.count == reference.count, observed.count == rows * hiddenSize else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "soft-token comparison shape mismatch observed=\(observed.count) reference=\(reference.count) rows=\(rows) hidden=\(hiddenSize)")
        }
        var maxAbsDelta = 0.0
        var sumAbsDelta = 0.0
        var finitePairCount = 0
        var observedNonFiniteCount = 0
        var referenceNonFiniteCount = 0
        var rowCosines: [Double] = []
        rowCosines.reserveCapacity(rows)
        for row in 0..<rows {
            var dot = 0.0
            var observedNorm = 0.0
            var referenceNorm = 0.0
            var rowIsFinite = true
            for column in 0..<hiddenSize {
                let index = row * hiddenSize + column
                let observedValue = Double(Float(observed[index]))
                let referenceValue = Double(Float(reference[index]))
                if !observedValue.isFinite {
                    observedNonFiniteCount += 1
                    rowIsFinite = false
                }
                if !referenceValue.isFinite {
                    referenceNonFiniteCount += 1
                    rowIsFinite = false
                }
                guard observedValue.isFinite, referenceValue.isFinite else {
                    continue
                }
                let delta = abs(observedValue - referenceValue)
                maxAbsDelta = max(maxAbsDelta, delta)
                sumAbsDelta += delta
                finitePairCount += 1
                dot += observedValue * referenceValue
                observedNorm += observedValue * observedValue
                referenceNorm += referenceValue * referenceValue
            }
            let denominator = sqrt(observedNorm) * sqrt(referenceNorm)
            if rowIsFinite, denominator > 0 {
                rowCosines.append(dot / denominator)
            }
        }
        return SoftTokenComparison(
            observedNonFiniteCount: observedNonFiniteCount,
            referenceNonFiniteCount: referenceNonFiniteCount,
            finitePairCount: finitePairCount,
            maxAbsDelta: maxAbsDelta,
            meanAbsDelta: finitePairCount > 0 ? sumAbsDelta / Double(finitePairCount) : .nan,
            rowCosineValidCount: rowCosines.count,
            rowCosineMin: rowCosines.min() ?? 0,
            rowCosineMean: rowCosines.reduce(0, +) / Double(max(rowCosines.count, 1)),
            rowCosineMax: rowCosines.max() ?? 0)
    }

    private func compareFloatArrays(_ observed: [Float], _ reference: [Float]) -> FloatArrayDelta {
        XCTAssertEqual(observed.count, reference.count)
        guard observed.count == reference.count, !observed.isEmpty else {
            return FloatArrayDelta(maxAbsDelta: .infinity, meanAbsDelta: .infinity)
        }
        var maxAbsDelta = 0.0
        var sumAbsDelta = 0.0
        for (observedValue, referenceValue) in zip(observed, reference) {
            let delta = abs(Double(observedValue) - Double(referenceValue))
            maxAbsDelta = max(maxAbsDelta, delta)
            sumAbsDelta += delta
        }
        return FloatArrayDelta(
            maxAbsDelta: maxAbsDelta,
            meanAbsDelta: sumAbsDelta / Double(observed.count))
    }

    private func copySpan(_ span: Span<Int>) -> [Int] {
        (0..<span.count).map { span[$0] }
    }

    private func softmax(_ logits: [Int32: Double]) -> [Int32: Double] {
        guard let maxLogit = logits.values.max() else { return [:] }
        let expValues = logits.mapValues { exp($0 - maxLogit) }
        let denominator = expValues.values.reduce(0, +)
        guard denominator > 0 else { return [:] }
        return expValues.mapValues { $0 / denominator }
    }

    private func writeFixtureReference(
        topLogits: [DistributedLogitScore],
        generatedTokenIDs: [Int32],
        to url: URL
    ) throws {
        let object: [String: Any] = [
            "first_top_logits": topLogits.map { [
                "token_id": Int($0.tokenID),
                "logit": Double($0.logit),
            ] },
            "generated_token_ids": generatedTokenIDs.map(Int.init),
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }

    private func syntheticSoftTokenRows(rows: Int, hiddenSize: Int) -> [Float16] {
        var values: [Float16] = []
        values.reserveCapacity(rows * hiddenSize)
        for row in 0..<rows {
            for column in 0..<hiddenSize {
                let sign: Float = ((row + column).isMultiple(of: 2)) ? 1 : -1
                let bucket = Float((column % 31) + 1) / 31.0
                values.append(Float16(sign * (0.125 + bucket * 0.25)))
            }
        }
        return values
    }

    private func imageSpanBlockIDs(
        count: Int,
        imageStart: Int,
        imageRows: Int
    ) -> [Int32] {
        var ids = Array(repeating: Int32(-1), count: count)
        for index in imageStart..<(imageStart + imageRows) {
            ids[index] = 0
        }
        return ids
    }

    private func assertSoftTokenSpliceRows(
        _ splice: DistributedSoftTokenSplice,
        areEmbeddedIn packet: DistributedHiddenStatePacket
    ) throws {
        let shape = packet.metadata.shape
        guard shape.count == 3, shape[0] == 1 else {
            XCTFail("embed packet shape \(shape) does not match [1, sequence, hidden]")
            return
        }
        let localStart = splice.positionStart - packet.metadata.positionRange.lowerBound
        guard localStart >= 0,
            localStart + splice.rowCount <= shape[1],
            splice.hiddenSize == shape[2]
        else {
            XCTFail(
                "soft-token splice rows \(splice.positionStart)..<\(splice.positionEnd) "
                    + "are outside embed packet shape \(shape) range \(packet.metadata.positionRange)")
            return
        }

        switch packet.metadata.scalarType {
        case .float16:
            let embedded = try packet.float16Values()
            let expected: [Float16]
            switch splice.scalarType {
            case .float16:
                expected = try splice.float16Values()
            case .float32:
                expected = try splice.float32Values().map(Float16.init)
            }
            try assertSpliceValues(
                embedded: embedded,
                expected: expected,
                localStart: localStart,
                sequenceLength: shape[1],
                hiddenSize: shape[2])
        case .float32:
            let embedded = try packet.float32Values()
            let expected: [Float]
            switch splice.scalarType {
            case .float16:
                expected = try splice.float16Values().map(Float.init)
            case .float32:
                expected = try splice.float32Values()
            }
            try assertSpliceValues(
                embedded: embedded,
                expected: expected,
                localStart: localStart,
                sequenceLength: shape[1],
                hiddenSize: shape[2])
        default:
            XCTFail("embed packet scalar type \(packet.metadata.scalarType.rawValue) is unsupported")
        }
    }

    private func assertSpliceValues<T: BinaryFloatingPoint & Equatable>(
        embedded: [T],
        expected: [T],
        localStart: Int,
        sequenceLength: Int,
        hiddenSize: Int
    ) throws {
        guard expected.count.isMultiple(of: hiddenSize) else {
            XCTFail("soft-token expected value count \(expected.count) is not row-aligned")
            return
        }
        let rows = expected.count / hiddenSize
        XCTAssertEqual(
            embedded.count,
            sequenceLength * hiddenSize,
            "embed packet value count must match shape")
        guard embedded.count == sequenceLength * hiddenSize else { return }
        for row in 0..<rows {
            for column in 0..<hiddenSize {
                let expectedOffset = row * hiddenSize + column
                let embeddedOffset = (localStart + row) * hiddenSize + column
                guard embedded[embeddedOffset] == expected[expectedOffset] else {
                    XCTFail(
                        "soft-token splice value mismatch at row \(row), column \(column): "
                            + "embedded \(embedded[embeddedOffset]), expected \(expected[expectedOffset])")
                    return
                }
            }
        }
    }

    private func rowByteDifferences(
        _ expected: DistributedHiddenStatePacket,
        _ observed: DistributedHiddenStatePacket
    ) throws -> [Bool] {
        XCTAssertEqual(expected.metadata.shape, observed.metadata.shape)
        XCTAssertEqual(expected.metadata.scalarType, observed.metadata.scalarType)
        guard expected.metadata.shape == observed.metadata.shape,
            expected.metadata.scalarType == observed.metadata.scalarType,
            expected.metadata.shape.count == 3
        else {
            throw XCTSkip("cannot compare hidden-state rows with mismatched metadata")
        }
        let sequenceLength = expected.metadata.shape[1]
        let hiddenSize = expected.metadata.shape[2]
        let rowBytes = hiddenSize * expected.metadata.scalarType.byteWidth
        XCTAssertEqual(expected.payload.count, observed.payload.count)
        XCTAssertEqual(expected.payload.count, sequenceLength * rowBytes)
        return (0..<sequenceLength).map { row in
            let start = row * rowBytes
            let end = start + rowBytes
            return expected.payload[start..<end] != observed.payload[start..<end]
        }
    }

    private func boolVector(_ values: [Bool]) -> String {
        values.map { $0 ? "1" : "0" }.joined()
    }

    private func rowIndices(_ values: [Bool]) -> String {
        let indices = values.enumerated()
            .filter { $0.element }
            .map { String($0.offset) }
        return indices.isEmpty ? "none" : indices.joined(separator: ",")
    }
    #endif

    private func jsonLine(_ object: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys])
        guard let line = String(data: data, encoding: .utf8) else {
            throw XCTSkip("could not encode JSONL record")
        }
        return line
    }

    private struct GreedySequenceMismatch {
        let prompt: Int
        let token: Int
        let expected: String
        let observed: String
    }

    private func firstSequenceMismatch(
        _ expected: [[Int32]],
        _ observed: [[Int32]]
    ) -> GreedySequenceMismatch? {
        let promptCount = min(expected.count, observed.count)
        for promptIndex in 0..<promptCount {
            let expectedTokens = expected[promptIndex]
            let observedTokens = observed[promptIndex]
            let tokenCount = min(expectedTokens.count, observedTokens.count)
            for tokenIndex in 0..<tokenCount {
                if expectedTokens[tokenIndex] != observedTokens[tokenIndex] {
                    return GreedySequenceMismatch(
                        prompt: promptIndex,
                        token: tokenIndex,
                        expected: "\(expectedTokens[tokenIndex])",
                        observed: "\(observedTokens[tokenIndex])")
                }
            }
            if expectedTokens.count != observedTokens.count {
                return GreedySequenceMismatch(
                    prompt: promptIndex,
                    token: tokenCount,
                    expected: tokenCount < expectedTokens.count
                        ? "\(expectedTokens[tokenCount])"
                        : "<end>",
                    observed: tokenCount < observedTokens.count
                        ? "\(observedTokens[tokenCount])"
                        : "<end>")
            }
        }
        if expected.count != observed.count {
            return GreedySequenceMismatch(
                prompt: promptCount,
                token: 0,
                expected: "prompt-count \(expected.count)",
                observed: "prompt-count \(observed.count)")
        }
        return nil
    }

    private func nextStagedToken(
        pipeline: DistributedSameMachinePipeline,
        requestID: String,
        stepIndex: Int,
        lowerBound: Int,
        tokenIDs: [Int32]
    ) async throws -> Int32 {
        let observation = try await nextStagedTokenObservation(
            pipeline: pipeline,
            requestID: requestID,
            stepIndex: stepIndex,
            lowerBound: lowerBound,
            tokenIDs: tokenIDs)
        return observation.tokenID
    }

    private struct StagedTokenObservation {
        let tokenID: Int32
        let topLogits: [DistributedLogitScore]
    }

    private struct TeacherForcedTie {
        let margin: Float
        let expectedLogit: Float
        let observedLogit: Float
        let topLogits: [DistributedLogitScore]
    }

    private func nextStagedTokenObservation(
        pipeline: DistributedSameMachinePipeline,
        requestID: String,
        stepIndex: Int,
        lowerBound: Int,
        tokenIDs: [Int32],
        blockIDsQ: [Int32]? = nil,
        blockIDsKV: [Int32]? = nil
    ) async throws -> StagedTokenObservation {
        let output = try await pipeline.forward(
            requestID: requestID,
            stepIndex: stepIndex,
            positionRange: DistributedSequenceRange(
                lowerBound: lowerBound,
                upperBound: lowerBound + tokenIDs.count),
            tokenIDs: tokenIDs,
            blockIDsQ: blockIDsQ,
            blockIDsKV: blockIDsKV)
        return StagedTokenObservation(
            tokenID: try XCTUnwrap(output.tokenID),
            topLogits: output.topLogits)
    }

    private func acceptedTeacherForcedTie(
        observation: StagedTokenObservation,
        expectedToken: Int32,
        tolerance: Double
    ) -> TeacherForcedTie? {
        guard let observed = observation.topLogits.first(where: { $0.tokenID == observation.tokenID }),
            let expected = observation.topLogits.first(where: { $0.tokenID == expectedToken })
        else {
            return nil
        }
        let topLogit = observation.topLogits.first?.logit ?? observed.logit
        let margin = max(0, topLogit - expected.logit)
        guard Double(margin) <= tolerance else { return nil }
        return TeacherForcedTie(
            margin: margin,
            expectedLogit: expected.logit,
            observedLogit: observed.logit,
            topLogits: observation.topLogits)
    }

    private func traceAcceptedTie(
        _ tie: TeacherForcedTie,
        promptIndex: Int,
        step: Int,
        expectedToken: Int32,
        observedToken: Int32
    ) {
        let line = "[caix-token-tie] prompt=\(promptIndex) step=\(step) "
            + "expected=\(expectedToken) staged=\(observedToken) "
            + "margin=\(String(format: "%.6g", Double(tie.margin))) "
            + "expected_logit=\(String(format: "%.6g", Double(tie.expectedLogit))) "
            + "staged_logit=\(String(format: "%.6g", Double(tie.observedLogit))) "
            + "top={\(describeTopLogits(tie.topLogits))}\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func describeTopLogits(_ topLogits: [DistributedLogitScore]) -> String {
        topLogits.map { item in
            "\(item.tokenID):\(String(format: "%.6g", Double(item.logit)))"
        }
        .joined(separator: ",")
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

    private func loadDoubleEnvironment(_ name: String) -> Double? {
        guard let raw = ProcessInfo.processInfo.environment[name],
            !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }
        return Double(raw)
    }

    private func loadIntEnvironment(_ name: String) -> Int? {
        guard let raw = ProcessInfo.processInfo.environment[name],
            !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }
        return Int(raw)
    }

    private struct TokenMarker {
        let index: Int
        let tokenID: Int32
    }

    private func loadTokenMarkerEnvironment(
        _ name: String,
        tokenCount: Int
    ) throws -> [TokenMarker] {
        guard let raw = ProcessInfo.processInfo.environment[name]?.trimmingCharacters(
            in: .whitespacesAndNewlines
        ), !raw.isEmpty else {
            return []
        }
        return try raw.split(separator: ",").map { item in
            let parts = item.split(separator: ":", omittingEmptySubsequences: false)
            guard parts.count == 2,
                let rawIndex = Int(parts[0]),
                let tokenID = Int32(parts[1])
            else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(name) marker \(item) must be index:token_id")
            }
            let index = rawIndex < 0 ? tokenCount + rawIndex : rawIndex
            guard index >= 0, index < tokenCount else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "\(name) marker index \(rawIndex) resolves outside 0..<\(tokenCount)")
            }
            return TokenMarker(index: index, tokenID: tokenID)
        }
    }

    private func loadBoolEnvironment(_ name: String) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[name]?.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).lowercased(), !raw.isEmpty else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    private func currentAvailableMemoryGB() throws -> Double {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/vm_stat")
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0,
            let output = String(data: data, encoding: .utf8)
        else {
            throw XCTSkip("could not read vm_stat before loading multimodal Gemma bundle")
        }
        let pageSize = parseVMStatPageSize(output) ?? 16_384
        let freePages = try parseVMStatPages(output, label: "Pages free")
        let inactivePages = try parseVMStatPages(output, label: "Pages inactive")
        let purgeablePages = try parseVMStatPages(output, label: "Pages purgeable")
        let speculativePages = try parseVMStatPages(output, label: "Pages speculative")
        let availablePages = freePages + inactivePages + purgeablePages + speculativePages
        let bytes = Double(availablePages) * Double(pageSize)
        return bytes / 1_073_741_824.0
    }

    private func currentProcessRSSMiB() throws -> Double {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-o", "rss=", "-p", String(ProcessInfo.processInfo.processIdentifier)]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0,
            let output = String(data: data, encoding: .utf8),
            let rssKiB = Double(output.trimmingCharacters(in: .whitespacesAndNewlines))
        else {
            throw XCTSkip("could not read current process RSS")
        }
        return rssKiB / 1024.0
    }

    private func parseVMStatPageSize(_ output: String) -> Int? {
        guard let range = output.range(of: "page size of ") else { return nil }
        let suffix = output[range.upperBound...]
        let digits = suffix.prefix { $0.isNumber }
        return Int(digits)
    }

    private func parseVMStatPages(_ output: String, label: String) throws -> Int {
        guard let line = output.split(whereSeparator: \.isNewline)
            .first(where: { $0.trimmingCharacters(in: .whitespaces).hasPrefix(label) })
        else {
            throw XCTSkip("vm_stat output did not contain \(label)")
        }
        let digits = line.split(separator: ":").dropFirst().joined()
            .filter(\.isNumber)
        guard let pages = Int(digits) else {
            throw XCTSkip("could not parse \(label) from vm_stat")
        }
        return pages
    }

    private struct Float16Summary {
        let count: Int
        let sumAbs: Double
        let l2: Double
    }

    private func summarizeFloat16Payload(_ payload: [UInt8]) throws -> Float16Summary {
        guard payload.count % MemoryLayout<UInt16>.size == 0 else {
            throw XCTSkip("float16 payload byte count is not even")
        }
        var sumAbs = 0.0
        var sumSquares = 0.0
        var count = 0
        var index = 0
        while index < payload.count {
            let bitPattern = UInt16(payload[index]) | (UInt16(payload[index + 1]) << 8)
            let value = Double(Float(Float16(bitPattern: bitPattern)))
            guard value.isFinite else {
                XCTFail("float16 payload contains non-finite value")
                break
            }
            sumAbs += abs(value)
            sumSquares += value * value
            count += 1
            index += MemoryLayout<UInt16>.size
        }
        return Float16Summary(count: count, sumAbs: sumAbs, l2: sqrt(sumSquares))
    }

    private func assertClose(
        _ actual: Double,
        _ expected: Double,
        tolerance: Double,
        label: String
    ) {
        let scale = max(abs(expected), 1.0)
        let relativeError = abs(actual - expected) / scale
        XCTAssertLessThanOrEqual(
            relativeError,
            tolerance,
            "\(label) relative error \(relativeError) exceeds \(tolerance); actual=\(actual) expected=\(expected)")
    }
}
