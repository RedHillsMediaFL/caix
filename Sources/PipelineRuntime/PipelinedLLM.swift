// Fast LLM generation backed by Apple's `CoreAILanguageModels` engines.
//
// WHY THIS EXISTS: our hand-rolled `LLMEngine` mirrors Apple's *sequential* reference engine
// (blocking `function.run()` + host-side sampling), which pays a CPU↔GPU sync round-trip every
// token (~10 ms even for a 0.6B model). Apple's own `EngineFactory` auto-selects the
// `CoreAIPipelinedEngine` for dynamic-shape bundles — non-blocking `encode`, GPU-side sampling,
// double-buffered — which on the *same* bundle runs qwen3-4b at ~101 tok/s vs ~60. So the fast
// path is to drive generation through Apple's documented engine, not to micro-optimize ours.
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
        let loadSeconds = Date().timeIntervalSince(loadStart)
        dbg("persistent engine + tokenizer ready")

        let generate: PipelinedLanguageHandle.Generate = { messages, options, tools, onToken in
            func requestDbg(_ s: String) {
                if options.verbose {
                    FileHandle.standardError.write(Data("[fast] \(s)\n".utf8))
                }
            }

            let sampling: SamplingConfiguration =
                options.temperature <= 0
                ? .greedy
                : SamplingConfiguration(temperature: options.temperature, topK: options.topK, topP: nil)

            let generator = try await TextGeneratorBuilder()
                .withInferenceEngine(engine)
                .withSampling(configuration: sampling)
                .withDecoding(type: .vanilla, parameters: DecodingParameters())
                .withTokenizer(tokenizer)
                .build()

            let input: Input
            let promptTokenCount: Int
            if options.applyChatTemplate {
                let ids: [Int]
                if let tools, !tools.isEmpty {
                    ids = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
                } else {
                    ids = try tokenizer.applyChatTemplate(messages: messages)
                }
                input = .tokens(ids)
                promptTokenCount = ids.count
            } else {
                let text = messages.map { $0["content"] ?? "" }.joined()
                input = .rawText(text)
                promptTokenCount = tokenizer.encode(text: text).count
            }

            requestDbg("warming up engine ...")
            try await engine.warmup(queryLength: 0, sampling: sampling)
            requestDbg("warmup done; generating \(options.maxTokens) tokens ...")

            let genStart = Date()
            var text = try await generator.generate(input: input, maxTokens: options.maxTokens)
            var stopReason: CoreAIPipeline.StopReason = .maxTokens
            if let stopRange = CoreAIPipeline.firstStopRange(
                in: text, stopSequences: options.stopSequences)
            {
                text = String(text[..<stopRange.lowerBound])
                stopReason = .stopSequence
            }
            let decodeSeconds = Date().timeIntervalSince(genStart)
            onToken?(text)
            let generatedTokenCount = tokenizer.encode(text: text).count
            requestDbg("generate done in \(String(format: "%.2f", decodeSeconds))s")

            return CoreAIPipeline.Result(
                text: text,
                promptTokenCount: promptTokenCount,
                generatedTokenCount: generatedTokenCount,
                stopReason: stopReason,
                modelLoadSeconds: loadSeconds,
                prefillSeconds: 0,
                decodeSeconds: decodeSeconds)
        }

        return PipelinedLanguageHandle(
            name: bundle.name,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            loadSeconds: loadSeconds,
            generate: generate)
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
    /// pipelined engine per call; on current macOS 27 betas, reusing a kept-hot
    /// `TextGenerator`/engine can suspend indefinitely under server workloads, while the same
    /// one-shot path is the stable path used by the CLI.
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

        let generator = try await TextGeneratorBuilder()
            .withInferenceEngine(engine)
            .withSampling(configuration: sampling)
            .withDecoding(type: .vanilla, parameters: DecodingParameters())
            .withTokenizer(tokenizer)
            .build()
        let modelLoadSeconds = Date().timeIntervalSince(loadStart)

        // Apply the chat template ourselves via a proper messages array and pass PRE-TOKENIZED ids
        // (`.tokens`), bypassing the engine's string-based `.prompt` templating.
        let input: Input
        let promptTokenCount: Int
        if options.applyChatTemplate {
            let ids: [Int]
            if let tools, !tools.isEmpty {
                ids = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
            } else {
                ids = try tokenizer.applyChatTemplate(messages: messages)
            }
            input = .tokens(ids)
            promptTokenCount = ids.count
        } else {
            let text = messages.map { $0["content"] ?? "" }.joined()
            input = .rawText(text)
            promptTokenCount = tokenizer.encode(text: text).count
        }

        // REQUIRED for the pipelined engine: warm up the decode/prefill graph shapes. Without this
        // the first encode loop deadlocks waiting on uncompiled shapes (llm-runner does the same).
        dbg("warming up engine …")
        try await engine.warmup(queryLength: 0, sampling: sampling)
        dbg("warmup done; generating \(options.maxTokens) tokens …")
        let genStart = Date()
        var text = try await generator.generate(input: input, maxTokens: options.maxTokens)
        var stopReason: CoreAIPipeline.StopReason = .maxTokens
        if let stopRange = CoreAIPipeline.firstStopRange(
            in: text, stopSequences: options.stopSequences)
        {
            text = String(text[..<stopRange.lowerBound])
            stopReason = .stopSequence
        }
        let decodeSeconds = Date().timeIntervalSince(genStart)
        dbg("generate done in \(String(format: "%.2f", decodeSeconds))s")
        onToken?(text)

        let generatedTokenCount = tokenizer.encode(text: text).count
        return CoreAIPipeline.Result(
            text: text,
            promptTokenCount: promptTokenCount,
            generatedTokenCount: generatedTokenCount,
            stopReason: stopReason,
            modelLoadSeconds: modelLoadSeconds,
            prefillSeconds: 0,
            decodeSeconds: decodeSeconds)
    }
}

#endif
