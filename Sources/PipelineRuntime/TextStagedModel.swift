#if COREAI_RUNTIME

import Foundation
import Tokenizers

/// Persistent same-machine staged text generation for model families whose export contract is
/// expressed as a caix `stage-manifest.json` rather than a single monolithic Core AI language bundle.
public final class TextStagedModel {
    struct ResidentLoadConfiguration {
        let maxContextLength: Int
        let contextSelection: DistributedStagedContextSelection?
        let streamedPrefillAdmission: DistributedStagedMemoryAdmission?
        let requiresDecodeResidentFactory: Bool
    }

    public let manifestURL: URL
    public let manifest: DistributedStageManifest
    public let name: String
    public let maxContextLength: Int
    public let bundleByteSize: UInt64
    public let loadSeconds: Double

    private let pipeline: DistributedSameMachinePipeline
    private let tokenizer: any Tokenizer
    private let tokenizerDir: URL
    private let chatRenderer: Gemma4ChatTemplateContract.ResidentRenderer?
    private let stopTokenIDs: Set<Int32>
    private let mtpAssistant: Gemma4MTPNativeRunner?
    private let mtpDraftTokens: Int

    private init(
        manifestURL: URL,
        manifest: DistributedStageManifest,
        pipeline: DistributedSameMachinePipeline,
        tokenizer: any Tokenizer,
        tokenizerDir: URL,
        chatRenderer: Gemma4ChatTemplateContract.ResidentRenderer?,
        mtpAssistant: Gemma4MTPNativeRunner?,
        mtpDraftTokens: Int,
        maxContextLength: Int,
        bundleByteSize: UInt64,
        loadSeconds: Double
    ) {
        self.manifestURL = manifestURL
        self.manifest = manifest
        self.name = manifest.modelName
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.tokenizerDir = tokenizerDir
        self.chatRenderer = chatRenderer
        self.mtpAssistant = mtpAssistant
        self.mtpDraftTokens = mtpDraftTokens
        self.maxContextLength = maxContextLength
        self.bundleByteSize = bundleByteSize
        self.loadSeconds = loadSeconds
        self.stopTokenIDs = Set(
            LLMEngine.stopTokenIds(tokenizer: tokenizer, tokenizerDir: tokenizerDir).map(Int32.init))
    }

    public static func load(
        manifestURL: URL,
        verbose: Bool = false,
        stagedMemorySnapshotProvider: (() throws -> DistributedStagedMemorySnapshot)? = nil,
        mtpAssistantURL: URL? = nil,
        mtpDraftTokens: Int = Gemma4MTPDecodeConfiguration.defaultDraftTokens
    ) async throws -> TextStagedModel {
        let started = Date()
        let resolvedManifestURL = manifestURL.standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: resolvedManifestURL)
        let root = resolvedManifestURL.deletingLastPathComponent()
        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        let metadataMaxContextLength = try readMaxContextLength(bundleRoot: root)
        let configuration = try residentLoadConfiguration(
            manifest: manifest,
            metadataMaxContextLength: metadataMaxContextLength,
            snapshotProvider: stagedMemorySnapshotProvider)
        let mtpAssistant: Gemma4MTPNativeRunner?
        if let mtpAssistantURL {
            guard manifest.eagleTarget != nil else {
                throw CoreAIPipeline.RuntimeError.invalidBundle(
                    "Gemma 4 MTP assistant requires an eagle_target staged manifest contract")
            }
            guard (1...Gemma4MTPDecodeConfiguration.maximumDraftTokens).contains(mtpDraftTokens)
            else {
                throw CoreAIPipeline.RuntimeError.invalidBundle(
                    "Gemma 4 MTP draft token count must be in 1...\(Gemma4MTPDecodeConfiguration.maximumDraftTokens)")
            }
            mtpAssistant = try await Gemma4MTPNativeRunner.load(
                aimodelURL: mtpAssistantURL.standardizedFileURL)
        } else {
            mtpAssistant = nil
        }
        let chatRenderer = manifest.requiresStreamedPrefillResidency
            ? try Gemma4ChatTemplateContract.ResidentRenderer(
                tokenizerDirectory: tokenizerDir)
            : nil
        if verbose {
            FileHandle.standardError.write(
                Data("[text-staged] loading \(resolvedManifestURL.path)\n".utf8))
        }

        let handleFactory: any DistributedStageHandleFactory =
            configuration.requiresDecodeResidentFactory
                ? DistributedCoreAIDecodeResidentStageHandleFactory()
                : DistributedCoreAIStageHandleFactory()
        let pipeline = try await DistributedSameMachinePipeline.make(
            manifest: manifest,
            handleFactory: handleFactory,
            streamedPrefillAdmission: configuration.streamedPrefillAdmission)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
        return TextStagedModel(
            manifestURL: resolvedManifestURL,
            manifest: manifest,
            pipeline: pipeline,
            tokenizer: tokenizer,
            tokenizerDir: tokenizerDir,
            chatRenderer: chatRenderer,
            mtpAssistant: mtpAssistant,
            mtpDraftTokens: mtpDraftTokens,
            maxContextLength: configuration.maxContextLength,
            bundleByteSize: directorySize(root),
            loadSeconds: Date().timeIntervalSince(started))
    }

    static func residentLoadConfiguration(
        manifest: DistributedStageManifest,
        metadataMaxContextLength: Int,
        snapshotProvider: (() throws -> DistributedStagedMemorySnapshot)?
    ) throws -> ResidentLoadConfiguration {
        guard metadataMaxContextLength > 0 else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "staged text metadata max context must be positive")
        }
        guard let contract = manifest.runtimeMemory else {
            return ResidentLoadConfiguration(
                maxContextLength: metadataMaxContextLength,
                contextSelection: nil,
                streamedPrefillAdmission: nil,
                requiresDecodeResidentFactory: false)
        }
        guard let snapshotProvider else {
            throw DistributedStagedMemoryAdmissionError.telemetryUnavailable
        }
        let admission = DistributedStagedMemoryAdmission(
            contract: contract,
            snapshotProvider: snapshotProvider)
        let selection = try admission.selectContext()
        guard selection.contextTokens <= metadataMaxContextLength else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "runtime_memory selected context \(selection.contextTokens) exceeds metadata max context \(metadataMaxContextLength)")
        }
        return ResidentLoadConfiguration(
            maxContextLength: selection.contextTokens,
            contextSelection: selection,
            streamedPrefillAdmission: admission,
            requiresDecodeResidentFactory: true)
    }

    @discardableResult
    public func generate(
        messages: [[String: String]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        let rich = messages.map { message -> [String: any Sendable] in
            var promoted: [String: any Sendable] = [:]
            for (key, value) in message {
                promoted[key] = value
            }
            return promoted
        }
        return try await generate(
            messages: rich,
            options: options,
            tools: tools,
            additionalContext: nil,
            onToken: onToken)
    }

    @discardableResult
    public func generate(
        messages: [[String: any Sendable]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        let promptTokens = try encodePrompt(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            applyChatTemplate: options.applyChatTemplate)
        return try await generate(
            promptTokenIDs: promptTokens.map(Int32.init),
            options: options,
            onToken: onToken)
    }

    @discardableResult
    public func generate(
        promptTokenIDs: [Int32],
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        guard !promptTokenIDs.isEmpty else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("prompt tokenized to 0 tokens")
        }
        try Self.validateGenerationOptions(options)
        guard options.maxTokens > 0 else {
            return CoreAIPipeline.Result(
                text: "",
                promptTokenCount: promptTokenIDs.count,
                generatedTokenCount: 0,
                stopReason: .maxTokens,
                modelLoadSeconds: loadSeconds,
                prefillSeconds: 0,
                decodeSeconds: 0,
                mtpTelemetry: mtpAssistant == nil ? nil : .unexercisedSequential,
                generatedTokenIDs: [])
        }

        let requestID = UUID().uuidString
        let capacity = try resolvedKVCapacity(
            promptCount: promptTokenIDs.count,
            maxTokens: options.maxTokens,
            explicitKVCapacity: options.kvCapacity)
        let cacheCapacities = resolvedCacheCapacities(kvCapacity: capacity)
        try await pipeline.allocate(
            requestID: requestID,
            kvCapacity: capacity,
            cacheCapacities: cacheCapacities)
        do {
            let prefillStart = Date()
            let prefill = try await prefillNextToken(
                promptTokenIDs: promptTokenIDs,
                requestID: requestID)
            let prefillSeconds = Date().timeIntervalSince(prefillStart)

            var generated: [Int32] = []
            var streamedText = ""
            var finalTextOverride: String?
            var stopReason = CoreAIPipeline.StopReason.maxTokens
            var mtpTelemetry: Gemma4MTPDecodeTelemetry? = mtpAssistant == nil
                ? nil
                : .unexercisedSequential
            let decodeStart = Date()

            func emitIfNeeded() {
                guard let onToken else { return }
                let text = tokenizer.decode(tokens: generated.map(Int.init))
                if text.hasPrefix(streamedText) {
                    let delta = String(text.dropFirst(streamedText.count))
                    if !delta.isEmpty { onToken(delta) }
                }
                streamedText = text
            }

            func commit(_ token: Int32) -> Bool {
                if stopTokenIDs.contains(token) {
                    stopReason = .eos
                    return false
                }
                if promptTokenIDs.count + generated.count >= maxContextLength {
                    stopReason = .contextLimit
                    return false
                }

                generated.append(token)
                emitIfNeeded()

                let decoded = tokenizer.decode(tokens: generated.map(Int.init))
                if let stopRange = CoreAIPipeline.firstStopRange(
                    in: decoded,
                    stopSequences: options.stopSequences)
                {
                    stopReason = .stopSequence
                    finalTextOverride = String(decoded[..<stopRange.lowerBound])
                    return false
                }
                return generated.count < options.maxTokens
            }

            if mtpAssistant != nil {
                let artifacts = try Self.requirePrefillMTPArtifacts(
                    prefill.eagleTargetArtifacts,
                    promptTokenCount: promptTokenIDs.count)
                let decoder = try sequentialMTPDecoder(
                    requestID: requestID,
                    firstTargetStepIndex: prefill.nextStepIndex)
                if commit(prefill.tokenID) {
                    let remainingTokens = min(
                        options.maxTokens - generated.count,
                        maxContextLength - promptTokenIDs.count - generated.count)
                    let outcome = try await decoder.run(
                        anchorToken: prefill.tokenID,
                        targetArtifacts: artifacts,
                        maximumAdditionalTokens: max(0, remainingTokens),
                        commit: commit)
                    mtpTelemetry = outcome.telemetry
                }
            } else {
                var nextToken = prefill.tokenID
                var running = commit(nextToken)
                while running {
                    let decodePosition = promptTokenIDs.count + generated.count - 1
                    nextToken = try await nextTokenID(
                        requestID: requestID,
                        stepIndex: prefill.nextStepIndex + generated.count - 1,
                        positionRange: DistributedSequenceRange(
                            lowerBound: decodePosition,
                            upperBound: decodePosition + 1),
                        tokenIDs: [nextToken])
                    running = commit(nextToken)
                }
            }

            await pipeline.free(requestID: requestID)
            let decodeSeconds = Date().timeIntervalSince(decodeStart)
            let finalText = finalTextOverride ?? tokenizer.decode(tokens: generated.map(Int.init))
            return CoreAIPipeline.Result(
                text: finalText,
                promptTokenCount: promptTokenIDs.count,
                generatedTokenCount: generated.count,
                stopReason: stopReason,
                modelLoadSeconds: loadSeconds,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeSeconds,
                mtpTelemetry: mtpTelemetry,
                generatedTokenIDs: generated)
        } catch {
            await pipeline.free(requestID: requestID)
            throw error
        }
    }

    /// Runs one private target/assistant verification pass to prove resident staged MTP readiness.
    /// The request is freed before this method returns and produces no user-visible output.
    public func prewarmMTPProof() async throws -> Gemma4MTPDecodeTelemetry {
        guard mtpAssistant != nil else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "staged MTP assistant is not configured")
        }
        let promptTokenIDs = tokenizer.encode(text: "MTP readiness probe").map(Int32.init)
        guard !promptTokenIDs.isEmpty else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "MTP readiness prompt tokenized to 0 tokens")
        }

        let requestID = UUID().uuidString
        let capacity = try resolvedKVCapacity(
            promptCount: promptTokenIDs.count,
            maxTokens: 2,
            explicitKVCapacity: nil)
        try await pipeline.allocate(
            requestID: requestID,
            kvCapacity: capacity,
            cacheCapacities: resolvedCacheCapacities(kvCapacity: capacity))
        do {
            let prefill = try await prefillNextToken(
                promptTokenIDs: promptTokenIDs,
                requestID: requestID)
            let artifacts = try Self.requirePrefillMTPArtifacts(
                prefill.eagleTargetArtifacts,
                promptTokenCount: promptTokenIDs.count)
            let decoder = try sequentialMTPDecoder(
                requestID: requestID,
                firstTargetStepIndex: prefill.nextStepIndex)
            let outcome = try await decoder.run(
                anchorToken: prefill.tokenID,
                targetArtifacts: artifacts,
                maximumAdditionalTokens: 1,
                commit: { _ in true })
            await pipeline.free(requestID: requestID)
            return outcome.telemetry
        } catch {
            await pipeline.free(requestID: requestID)
            throw error
        }
    }

    static func validateGenerationOptions(_ options: CoreAIPipeline.Options) throws {
        guard options.temperature <= 0 else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "staged text serving currently supports greedy decoding only")
        }
        guard options.constrainedJSONSchema == nil else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "staged text serving does not support constrained decoding yet")
        }
        guard options.maxTokens >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "max_tokens must be non-negative")
        }
    }

    private func encodePrompt(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        applyChatTemplate: Bool
    ) throws -> [Int] {
        let tokens: [Int]
        if applyChatTemplate {
            if let chatRenderer {
                tokens = try chatRenderer.encode(
                    tokenizer: tokenizer,
                    messages: messages,
                    tools: tools,
                    additionalContext: additionalContext)
            } else {
                let stringMessages = try Self.stringMessages(messages)
                if let tools, !tools.isEmpty {
                    tokens = try tokenizer.applyChatTemplate(
                        messages: stringMessages,
                        tools: tools)
                } else {
                    tokens = try tokenizer.applyChatTemplate(messages: stringMessages)
                }
            }
        } else {
            let stringMessages = try Self.stringMessages(messages)
            tokens = tokenizer.encode(
                text: stringMessages.map { $0["content"] ?? "" }.joined())
        }
        guard !tokens.isEmpty else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("prompt tokenized to 0 tokens")
        }
        if ProcessInfo.processInfo.environment["COREAI_DEBUG_PROMPT"] != nil {
            FileHandle.standardError.write(Data(
                ("[text-staged] DEBUG prompt tokens(\(tokens.count)): "
                    + "\(tokens.prefix(40))\(tokens.count > 40 ? " …" : "")\n").utf8))
        }
        return tokens
    }

    private static func stringMessages(
        _ messages: [[String: any Sendable]]
    ) throws -> [[String: String]] {
        try messages.map { message in
            guard let role = message["role"] as? String else {
                throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                    "staged text message role must be a string")
            }
            guard let content = message["content"] as? String else {
                throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                    "generic staged text templates require string message content")
            }
            return ["role": role, "content": content]
        }
    }

    private struct PrefillResult {
        let tokenID: Int32
        let nextStepIndex: Int
        let eagleTargetArtifacts: DistributedEagleTargetArtifacts?
    }

    private func prefillNextToken(
        promptTokenIDs: [Int32],
        requestID: String
    ) async throws -> PrefillResult {
        if let chunkSize = resolvedPrefillChunkSize(promptCount: promptTokenIDs.count) {
            return try await chunkedPrefillNextToken(
                promptTokenIDs: promptTokenIDs,
                requestID: requestID,
                chunkSize: chunkSize)
        }
        let output = try await pipeline.forward(
            requestID: requestID,
            stepIndex: 0,
            positionRange: DistributedSequenceRange(
                lowerBound: 0,
                upperBound: promptTokenIDs.count),
            tokenIDs: promptTokenIDs)
        guard let tokenID = output.tokenID else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "staged prefill did not return a token id")
        }
        return PrefillResult(
            tokenID: tokenID,
            nextStepIndex: 1,
            eagleTargetArtifacts: output.eagleTargetArtifacts)
    }

    private func chunkedPrefillNextToken(
        promptTokenIDs: [Int32],
        requestID: String,
        chunkSize: Int
    ) async throws -> PrefillResult {
        guard chunkSize > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "staged prefill chunk must be positive")
        }
        var lowerBound = 0
        var stepIndex = 0
        var tokenID: Int32?
        var eagleTargetArtifacts: DistributedEagleTargetArtifacts?
        while lowerBound < promptTokenIDs.count {
            let upperBound = min(promptTokenIDs.count, lowerBound + chunkSize)
            let range = lowerBound..<upperBound
            let emitToken = upperBound == promptTokenIDs.count
            let output = try await pipeline.forward(
                requestID: requestID,
                stepIndex: stepIndex,
                positionRange: DistributedSequenceRange(
                    lowerBound: lowerBound,
                    upperBound: upperBound),
                tokenIDs: Array(promptTokenIDs[range]),
                emitToken: emitToken)
            if emitToken {
                guard let emittedTokenID = output.tokenID else {
                    throw DistributedStageExecutionError.invalidStageOutput(
                        "final staged prefill chunk did not return a token id")
                }
                tokenID = emittedTokenID
                eagleTargetArtifacts = output.eagleTargetArtifacts
            }
            lowerBound = upperBound
            stepIndex += 1
        }
        guard let tokenID else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "chunked staged prefill produced no observations")
        }
        return PrefillResult(
            tokenID: tokenID,
            nextStepIndex: stepIndex,
            eagleTargetArtifacts: eagleTargetArtifacts)
    }

    private static func requirePrefillMTPArtifacts(
        _ artifacts: DistributedEagleTargetArtifacts?,
        promptTokenCount: Int
    ) throws -> DistributedEagleTargetArtifacts {
        guard let artifacts else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "staged Gemma 4 MTP prefill did not return target artifacts")
        }
        guard artifacts.fullPositionRange
            == DistributedSequenceRange(lowerBound: 0, upperBound: promptTokenCount)
        else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "staged Gemma 4 MTP prefill target KV must cover the full prompt")
        }
        return artifacts
    }

    private func sequentialMTPDecoder(
        requestID: String,
        firstTargetStepIndex: Int
    ) throws -> Gemma4SequentialMTPDecoder {
        guard let mtpAssistant else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "staged MTP assistant is not configured")
        }
        var targetStepIndex = firstTargetStepIndex
        return try Gemma4SequentialMTPDecoder(
            draftTokens: mtpDraftTokens,
            propose: { request in
                try await mtpAssistant.propose(request)
            },
            targetDecode: { token, position in
                let output = try await self.pipeline.forward(
                    requestID: requestID,
                    stepIndex: targetStepIndex,
                    positionRange: DistributedSequenceRange(
                        lowerBound: position,
                        upperBound: position + 1),
                    tokenIDs: [token])
                targetStepIndex += 1
                guard let tokenID = output.tokenID else {
                    throw DistributedStageExecutionError.invalidStageOutput(
                        "staged MTP target Q=1 forward did not return a token id")
                }
                guard let artifacts = output.eagleTargetArtifacts else {
                    throw DistributedStageExecutionError.invalidStageOutput(
                        "staged MTP target Q=1 forward did not return target artifacts")
                }
                return Gemma4SequentialMTPTargetStep(
                    tokenID: tokenID,
                    artifacts: artifacts)
            })
    }

    private func nextTokenID(
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        tokenIDs: [Int32]
    ) async throws -> Int32 {
        let output = try await pipeline.forward(
            requestID: requestID,
            stepIndex: stepIndex,
            positionRange: positionRange,
            tokenIDs: tokenIDs)
        guard let tokenID = output.tokenID else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "staged pipeline did not return a token id")
        }
        return tokenID
    }

    private func resolvedPrefillChunkSize(promptCount: Int) -> Int? {
        guard promptCount > 1 else { return nil }
        if let raw = ProcessInfo.processInfo.environment["COREAI_STAGED_PREFILL_CHUNK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty
        {
            guard let value = Int(raw), value > 0 else { return nil }
            return max(1, value)
        }
        if let chunk = manifest.cacheGroups?.prefillChunk, chunk > 0 {
            return chunk
        }
        if manifest.cacheGroups?.strategy?.lowercased().contains("qwen3_5") == true {
            // qwen3_5 split-state bundles keep full-attention KV at native scale while
            // linear layers carry fixed recurrent state. Larger chunks can compile and run
            // short prompts but trigger severe Core AI activation pressure on token-dense
            // long-context prefill; keep old manifests safe unless explicitly overridden.
            return 128
        }
        return nil
    }

    private func resolvedKVCapacity(
        promptCount: Int,
        maxTokens: Int,
        explicitKVCapacity: Int?
    ) throws -> Int {
        let requested = explicitKVCapacity ?? (promptCount + maxTokens + 8)
        guard requested > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "kv_capacity must be positive")
        }
        let capacity = min(requested, maxContextLength)
        guard capacity >= promptCount else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "kv_capacity is smaller than prompt")
        }
        return capacity
    }

    private func resolvedCacheCapacities(kvCapacity: Int) -> [String: Int]? {
        manifest.cacheGroups?.capacities(forKVCapacity: kvCapacity)
    }

    private static func readMaxContextLength(bundleRoot: URL) throws -> Int {
        let metadataURL = bundleRoot.appendingPathComponent("metadata.json")
        guard let data = try? Data(contentsOf: metadataURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let language = object["language"] as? [String: Any]
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "staged text bundle is missing metadata.json language block")
        }
        if let value = language["max_context_length"] as? Int { return value }
        if let value = language["max_context_length"] as? Double { return Int(value) }
        throw CoreAIPipeline.RuntimeError.invalidBundle(
            "staged text bundle is missing language.max_context_length")
    }

    private static func directorySize(_ root: URL) -> UInt64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        else { return 0 }
        var total: UInt64 = 0
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true, let size = values?.fileSize {
                total += UInt64(size)
            }
        }
        return total
    }
}

#endif
