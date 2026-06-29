// Fast LLM generation backed by Apple's `CoreAILanguageModels` engines.
//
// WHY THIS EXISTS: our hand-rolled `LLMEngine` mirrors Apple's sequential reference engine
// (blocking `function.run()` plus host-side sampling). Apple's `EngineFactory` can select the
// pipelined engine for compatible dynamic-shape bundles, so this path drives generation through
// that engine while preserving caix prompt handling and result accounting.
#if COREAI_RUNTIME

import CoreAI
import CoreAILanguageModels
import Foundation
import Tokenizers

struct PipelinedLanguageHandle {
    typealias Generate = (
        _ messages: [[String: String]],
        _ options: CoreAIPipeline.Options,
        _ tools: [[String: any Sendable]]?,
        _ onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result

    let name: String
    let vocabSize: Int
    let maxContextLength: Int
    let loadSeconds: Double
    let generate: Generate
}

enum PipelinedLLM {
    /// Load a reusable CoreAILanguageModels engine for server hot handles. This mirrors
    /// ``runIfLanguage`` but keeps the engine/tokenizer resident so `/api/load` and OpenAI clients
    /// do not fall back to the older sequential `LLMEngine` path.
    static func loadPersistent(
        bundlePath: String,
        verbose: Bool = false
    ) async throws -> PipelinedLanguageHandle? {
        let bundle: LanguageBundle
        do {
            bundle = try LanguageBundle(from: bundlePath)
        } catch {
            return nil
        }

        func dbg(_ s: String) {
            if verbose {
                FileHandle.standardError.write(Data("[fast] \(s)\n".utf8))
            }
        }

        dbg("language bundle ok: \(bundle.name)")
        let loadStart = Date()
        let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main")
        let configData = try JSONEncoder().encode(config)

        async let engineResult = EngineFactory.createEngine(
            config: configData, modelURL: modelURL, options: EngineOptions())
        async let tokenizerResult = bundle.loadTokenizer()
        dbg("creating persistent engine (auto->pipelined) ...")
        let engine = try await engineResult
        let tokenizer = try await tokenizerResult
        dbg("warming persistent engine ...")
        try await engine.warmup(queryLength: 0, sampling: .greedy)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        dbg("persistent engine + tokenizer ready")

        var warmedSamplingKeys: Set<String> = ["greedy"]
        let generate: PipelinedLanguageHandle.Generate = { messages, options, tools, onToken in
            func requestDbg(_ s: String) {
                if options.verbose {
                    FileHandle.standardError.write(Data("[fast] \(s)\n".utf8))
                }
            }

            let samplingKey = Self.samplingWarmupKey(options)
            let sampling: SamplingConfiguration =
                options.temperature <= 0
                ? .greedy
                : SamplingConfiguration(temperature: options.temperature, topK: options.topK, topP: nil)

            let prepared = try makeInput(
                messages: messages, tools: tools, options: options, tokenizer: tokenizer)

            if !warmedSamplingKeys.contains(samplingKey) {
                requestDbg("warming up engine ...")
                try await engine.warmup(queryLength: 0, sampling: sampling)
                warmedSamplingKeys.insert(samplingKey)
                requestDbg("warmup done; generating \(options.maxTokens) tokens ...")
            } else {
                requestDbg("generating \(options.maxTokens) tokens ...")
            }

            let result = try await decodeWithVanillaStrategy(
                input: prepared.input,
                tokenizer: tokenizer,
                inferenceEngine: engine,
                samplingConfiguration: sampling,
                options: options,
                promptTokenCount: prepared.promptTokenCount,
                modelLoadSeconds: loadSeconds,
                onToken: onToken)
            requestDbg("generate done in \(String(format: "%.2f", result.decodeSeconds))s")
            return result
        }

        return PipelinedLanguageHandle(
            name: bundle.name,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            loadSeconds: loadSeconds,
            generate: generate)
    }

    private static func samplingWarmupKey(_ options: CoreAIPipeline.Options) -> String {
        let topK = options.topK.map(String.init) ?? "nil"
        return options.temperature <= 0 ? "greedy" : "temperature=\(options.temperature);topK=\(topK)"
    }

    private static func makeInput(
        messages: [[String: String]],
        tools: [[String: any Sendable]]?,
        options: CoreAIPipeline.Options,
        tokenizer: any Tokenizer
    ) throws -> (input: Input, promptTokenCount: Int) {
        let ids: [Int]
        if options.applyChatTemplate {
            if let tools, !tools.isEmpty {
                ids = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
            } else {
                ids = try tokenizer.applyChatTemplate(messages: messages)
            }
        } else {
            let text = messages.map { $0["content"] ?? "" }.joined()
            ids = tokenizer.encode(text: text)
        }
        return (.tokens(ids), ids.count)
    }

    /// Generate via Apple's pipelined engine if `modelPath` is a language bundle. Returns `nil`
    /// for non-language bundles (e.g. diffusion), so the caller falls back to ``LLMEngine``.
    static func runIfLanguage(
        modelPath: String,
        prompt: String,
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result? {
        try await runIfLanguageMessages(
            modelPath: modelPath,
            messages: [["role": "user", "content": prompt]],
            tools: nil,
            options: options,
            onToken: onToken)
    }

    /// One-shot fast generation for server requests. This intentionally creates a fresh Apple
    /// pipelined engine per call; the `/api/load` path uses ``loadPersistent`` when callers want a
    /// kept-hot handle.
    static func runIfLanguageMessages(
        modelPath: String,
        messages: [[String: String]],
        tools: [[String: any Sendable]]?,
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result? {
        // Diffusion / non-LLM bundles have no `language` block → LanguageBundle throws → fall back.
        let bundle: LanguageBundle
        do {
            bundle = try LanguageBundle(from: modelPath)
        } catch {
            return nil
        }

        func dbg(_ s: String) {
            if options.verbose {
                FileHandle.standardError.write(Data("[fast] \(s)\n".utf8))
            }
        }
        dbg("language bundle ok: \(bundle.name)")
        let loadStart = Date()
        let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main")
        let configData = try JSONEncoder().encode(config)

        // EngineFactory auto-detects: dynamic bundle → `.pipelined` (the fast engine).
        async let engineResult = EngineFactory.createEngine(
            config: configData, modelURL: modelURL, options: EngineOptions())
        async let tokenizerResult = bundle.loadTokenizer()
        dbg("creating engine (auto→pipelined) …")
        let engine = try await engineResult
        let tokenizer = try await tokenizerResult
        dbg("engine + tokenizer ready")

        // The pipelined GPU sampler supports greedy + temperature + topK, but NOT topP — drop topP
        // so we stay on the fast engine (matches Apple's `validateSamplingStrategy`).
        let sampling: SamplingConfiguration =
            options.temperature <= 0
            ? .greedy
            : SamplingConfiguration(temperature: options.temperature, topK: options.topK, topP: nil)

        let modelLoadSeconds = Date().timeIntervalSince(loadStart)

        // Pass pre-tokenized ids so the decoder does not tokenize the same prompt again.
        let prepared = try makeInput(
            messages: messages, tools: tools, options: options, tokenizer: tokenizer)

        // REQUIRED for the pipelined engine: warm up the decode/prefill graph shapes. Without this
        // the first encode loop deadlocks waiting on uncompiled shapes (llm-runner does the same).
        dbg("warming up engine …")
        try await engine.warmup(queryLength: 0, sampling: sampling)
        dbg("warmup done; generating \(options.maxTokens) tokens …")
        let result = try await decodeWithVanillaStrategy(
            input: prepared.input,
            tokenizer: tokenizer,
            inferenceEngine: engine,
            samplingConfiguration: sampling,
            options: options,
            promptTokenCount: prepared.promptTokenCount,
            modelLoadSeconds: modelLoadSeconds,
            onToken: onToken)
        dbg("generate done in \(String(format: "%.2f", result.decodeSeconds))s")
        return result
    }

    private static func decodeWithVanillaStrategy(
        input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: CoreAIPipeline.Options,
        promptTokenCount: Int,
        modelLoadSeconds: Double,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result {
        let maxTokens = max(0, options.maxTokens)
        let stopSequences = options.stopSequences.filter { !$0.isEmpty }
        let stream = try VanillaDecodingStrategy().decode(
            from: input,
            tokenizer: tokenizer,
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            options: InferenceOptions(maxTokens: maxTokens),
            stopSequences: StopSequences(for: tokenizer))

        let decodeStart = Date()
        var text = ""
        var streamedText = ""
        var finalTextOverride: String?
        var stopReason: CoreAIPipeline.StopReason = .maxTokens

        func emitVisibleText(_ visible: String) {
            guard let onToken else {
                streamedText = visible
                return
            }
            if visible.hasPrefix(streamedText) {
                let delta = String(visible.dropFirst(streamedText.count))
                if !delta.isEmpty { onToken(delta) }
            }
            streamedText = visible
        }

        for try await chunk in stream {
            text += chunk.text

            if let stopRange = CoreAIPipeline.firstStopRange(
                in: text, stopSequences: stopSequences)
            {
                let visible = String(text[..<stopRange.lowerBound])
                emitVisibleText(visible)
                finalTextOverride = visible
                stopReason = .stopSequence
                break
            }

            if onToken != nil || !stopSequences.isEmpty {
                let visible = stopSequences.isEmpty
                    ? text
                    : CoreAIPipeline.visibleTextAvoidingPartialStop(
                        text, stopSequences: stopSequences)
                emitVisibleText(visible)
            }
        }

        let finalText = finalTextOverride ?? text
        if finalTextOverride == nil, onToken != nil {
            emitVisibleText(finalText)
        }

        return CoreAIPipeline.Result(
            text: finalText,
            promptTokenCount: promptTokenCount,
            generatedTokenCount: tokenizer.encode(text: finalText).count,
            stopReason: stopReason,
            modelLoadSeconds: modelLoadSeconds,
            prefillSeconds: 0,
            decodeSeconds: Date().timeIntervalSince(decodeStart))
    }
}

#endif
