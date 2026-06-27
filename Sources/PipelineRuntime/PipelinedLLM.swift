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

enum PipelinedLLM {
    /// Generate via Apple's pipelined engine if `modelPath` is a language bundle. Returns `nil`
    /// for non-language bundles (e.g. diffusion), so the caller falls back to ``LLMEngine``.
    static func runIfLanguage(
        modelPath: String,
        prompt: String,
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
        // (`.tokens`), bypassing the engine's string-based `.prompt` templating — which is malformed
        // for these bundles (the model hallucinates a different prompt). Raw completions are fine, so
        // this is purely the template path.
        let input: Input
        let promptTokenCount: Int
        if options.applyChatTemplate,
            let ids = try? tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]]) {
            input = .tokens(ids)
            promptTokenCount = ids.count
        } else {
            input = .rawText(prompt)
            promptTokenCount = (try? tokenizer.encode(text: prompt).count) ?? 0
        }

        // REQUIRED for the pipelined engine: warm up the decode/prefill graph shapes. Without this
        // the first encode loop deadlocks waiting on uncompiled shapes (llm-runner does the same).
        dbg("warming up engine …")
        try await engine.warmup(queryLength: 0, sampling: sampling)
        dbg("warmup done; generating \(options.maxTokens) tokens …")
        let genStart = Date()
        let text = try await generator.generate(input: input, maxTokens: options.maxTokens)
        let decodeSeconds = Date().timeIntervalSince(genStart)
        dbg("generate done in \(String(format: "%.2f", decodeSeconds))s")
        onToken?(text)

        let generatedTokenCount = max(1, (try? tokenizer.encode(text: text).count) ?? options.maxTokens)
        return CoreAIPipeline.Result(
            text: text,
            promptTokenCount: promptTokenCount,
            generatedTokenCount: generatedTokenCount,
            stopReason: generatedTokenCount >= options.maxTokens ? .maxTokens : .eos,
            modelLoadSeconds: modelLoadSeconds,
            prefillSeconds: 0,
            decodeSeconds: decodeSeconds)
    }
}

#endif
