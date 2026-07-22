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

    private init(
        manifestURL: URL,
        manifest: DistributedStageManifest,
        pipeline: DistributedSameMachinePipeline,
        tokenizer: any Tokenizer,
        tokenizerDir: URL,
        chatRenderer: Gemma4ChatTemplateContract.ResidentRenderer?,
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
        self.maxContextLength = maxContextLength
        self.bundleByteSize = bundleByteSize
        self.loadSeconds = loadSeconds
        self.stopTokenIDs = Set(
            LLMEngine.stopTokenIds(tokenizer: tokenizer, tokenizerDir: tokenizerDir).map(Int32.init))
    }

    public static func load(
        manifestURL: URL,
        verbose: Bool = false,
        stagedMemorySnapshotProvider: (() throws -> DistributedStagedMemorySnapshot)? = nil
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
            var nextToken = prefill.tokenID
            let prefillSeconds = Date().timeIntervalSince(prefillStart)

            var generated: [Int32] = []
            var streamedText = ""
            var finalTextOverride: String?
            var stopReason = CoreAIPipeline.StopReason.maxTokens
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

            while generated.count < options.maxTokens {
                if stopTokenIDs.contains(nextToken) {
                    stopReason = .eos
                    break
                }
                if promptTokenIDs.count + generated.count >= maxContextLength {
                    stopReason = .contextLimit
                    break
                }

                generated.append(nextToken)
                emitIfNeeded()

                let decoded = tokenizer.decode(tokens: generated.map(Int.init))
                if let stopRange = CoreAIPipeline.firstStopRange(
                    in: decoded,
                    stopSequences: options.stopSequences)
                {
                    stopReason = .stopSequence
                    finalTextOverride = String(decoded[..<stopRange.lowerBound])
                    break
                }

                guard generated.count < options.maxTokens else { break }
                let decodePosition = promptTokenIDs.count + generated.count - 1
                nextToken = try await nextTokenID(
                    requestID: requestID,
                    stepIndex: prefill.nextStepIndex + generated.count - 1,
                    positionRange: DistributedSequenceRange(
                        lowerBound: decodePosition,
                        upperBound: decodePosition + 1),
                    tokenIDs: [nextToken])
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
                generatedTokenIDs: generated)
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
        let tokenID = try await nextTokenID(
            requestID: requestID,
            stepIndex: 0,
            positionRange: DistributedSequenceRange(
                lowerBound: 0,
                upperBound: promptTokenIDs.count),
            tokenIDs: promptTokenIDs)
        return PrefillResult(tokenID: tokenID, nextStepIndex: 1)
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
            }
            lowerBound = upperBound
            stepIndex += 1
        }
        guard let tokenID else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "chunked staged prefill produced no observations")
        }
        return PrefillResult(tokenID: tokenID, nextStepIndex: stepIndex)
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
