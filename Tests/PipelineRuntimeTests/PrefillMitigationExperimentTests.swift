import XCTest
@testable import PipelineRuntime

final class PrefillMitigationExperimentTests: XCTestCase {
    func testRunMonolithicPrefillMitigationPrompt() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["CAIX_PREFILL_MITIGATION_MODEL"],
            !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_PREFILL_MITIGATION_MODEL to a monolithic caix bundle")
        }
        guard let promptPath = ProcessInfo.processInfo.environment["CAIX_PREFILL_MITIGATION_PROMPT"],
            !promptPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_PREFILL_MITIGATION_PROMPT to a prompt file")
        }
        guard let outputPath = ProcessInfo.processInfo.environment["CAIX_PREFILL_MITIGATION_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_PREFILL_MITIGATION_OUTPUT to a token output path")
        }

        #if COREAI_RUNTIME
        let prompt = try String(contentsOfFile: promptPath, encoding: .utf8)
            .trimmingCharacters(in: .newlines)
        let maxTokens = max(
            1,
            Int(ProcessInfo.processInfo.environment["CAIX_PREFILL_MITIGATION_MAX_TOKENS"] ?? "")
                ?? 8)
        let result = try await CoreAIPipeline.run(
            modelPath: modelPath,
            prompt: prompt,
            options: CoreAIPipeline.Options(
                maxTokens: maxTokens,
                temperature: 0,
                applyChatTemplate: false,
                verbose: true),
            onToken: nil)
        XCTAssertEqual(result.generatedTokenIDs.count, result.generatedTokenCount)
        XCTAssertEqual(result.generatedTokenIDs.count, maxTokens)

        let tokenText = result.generatedTokenIDs.map(String.init).joined(separator: ",") + "\n"
        try tokenText.write(toFile: outputPath, atomically: true, encoding: .utf8)

        if let metaPath = ProcessInfo.processInfo.environment["CAIX_PREFILL_MITIGATION_META"],
            !metaPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            let meta = [
                "prompt_token_count=\(result.promptTokenCount)",
                "generated_token_count=\(result.generatedTokenCount)",
                String(format: "prefill_seconds=%.6f", result.prefillSeconds),
                String(format: "decode_seconds=%.6f", result.decodeSeconds),
                String(format: "decode_tokens_per_second=%.6f", result.decodeTokensPerSecond),
                "stop_reason=\(result.stopReason.rawValue)",
            ].joined(separator: "\n") + "\n"
            try meta.write(toFile: metaPath, atomically: true, encoding: .utf8)
        }
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testTraceMonolithicPrefillMitigationLogits() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["CAIX_PREFILL_MITIGATION_MODEL"],
            !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_PREFILL_MITIGATION_MODEL to a monolithic caix bundle")
        }
        guard let promptPath = ProcessInfo.processInfo.environment["CAIX_PREFILL_MITIGATION_PROMPT"],
            !promptPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_PREFILL_MITIGATION_PROMPT to a prompt file")
        }
        guard let outputPath = ProcessInfo.processInfo.environment["CAIX_PREFILL_TRACE_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_PREFILL_TRACE_OUTPUT to a trace output path")
        }

        #if COREAI_RUNTIME
        let prompt = try String(contentsOfFile: promptPath, encoding: .utf8)
            .trimmingCharacters(in: .newlines)
        let maxTokens = max(
            1,
            Int(ProcessInfo.processInfo.environment["CAIX_PREFILL_MITIGATION_MAX_TOKENS"] ?? "")
                ?? 8)
        let topK = max(2, Int(ProcessInfo.processInfo.environment["CAIX_PREFILL_TRACE_TOPK"] ?? "") ?? 8)
        let bundle = try ResolvedBundle.load(at: modelPath)
        let baseline = try await LLMEngine.load(bundle: bundle, verbose: true)
        let promptTokens = try baseline.encodePrompt(
            messages: [["role": "user", "content": prompt]],
            applyChatTemplate: false)
        let capacity = min(
            baseline.maxContextLength,
            max(promptTokens.count + maxTokens + 8, maxTokens + 1))
        try baseline.allocateKVCache(capacity: capacity)

        var lastLogits: [Float] = []
        let prefillChunkSize = baseline.resolvedPrefillChunkSize(promptCount: promptTokens.count)
        var prefillChunks: [[Int]] = []
        if prefillChunkSize < promptTokens.count {
            var offset = 0
            while offset < promptTokens.count {
                let end = min(offset + prefillChunkSize, promptTokens.count)
                let chunk = Array(promptTokens[offset..<end])
                prefillChunks.append(chunk)
                lastLogits = try await baseline.step(tokens: chunk.map(Int32.init))
                offset = end
            }
        } else {
            prefillChunks = [promptTokens]
            lastLogits = try await baseline.step(tokens: promptTokens.map(Int32.init))
        }

        var nextToken = Int32(Sampler.argmax(lastLogits))
        var generated: [Int32] = []
        var steps: [[String: Any]] = []
        for step in 0..<maxTokens {
            let top = Sampler.topK(lastLogits, count: min(topK, lastLogits.count))
            let margin = top.count > 1 ? top[0].logit - top[1].logit : .nan
            steps.append([
                "step": step,
                "token": nextToken,
                "margin": margin,
                "top": top.map { ["token": Int32($0.index), "logit": $0.logit] },
            ])
            generated.append(nextToken)
            guard generated.count < maxTokens else { break }
            lastLogits = try await baseline.step(tokens: [nextToken])
            nextToken = Int32(Sampler.argmax(lastLogits))
        }

        let object: [String: Any] = [
            "prompt": prompt,
            "prompt_tokens": promptTokens,
            "prefill_chunk_size": prefillChunkSize,
            "prefill_chunks": prefillChunks,
            "generated_tokens": generated,
            "steps": steps,
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outputPath))
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }

    func testRunQwen06DirectionalSpeedProbe() async throws {
        guard let monolithicPath = ProcessInfo.processInfo.environment["CAIX_SPEED_PROBE_MONOLITHIC"],
            !monolithicPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_SPEED_PROBE_MONOLITHIC to a monolithic bundle")
        }
        guard let stagedManifestPath = ProcessInfo.processInfo.environment["CAIX_SPEED_PROBE_STAGED_MANIFEST"],
            !stagedManifestPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_SPEED_PROBE_STAGED_MANIFEST to a staged manifest")
        }
        guard let promptPath = ProcessInfo.processInfo.environment["CAIX_SPEED_PROBE_PROMPT"],
            !promptPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_SPEED_PROBE_PROMPT to a prompt file")
        }
        guard let outputPath = ProcessInfo.processInfo.environment["CAIX_SPEED_PROBE_OUTPUT"],
            !outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw XCTSkip("set CAIX_SPEED_PROBE_OUTPUT to a result path")
        }

        #if COREAI_RUNTIME
        let prompt = try String(contentsOfFile: promptPath, encoding: .utf8)
            .trimmingCharacters(in: .newlines)
        let maxTokens = max(
            1,
            Int(ProcessInfo.processInfo.environment["CAIX_SPEED_PROBE_MAX_TOKENS"] ?? "")
                ?? 256)
        let runs = max(
            1,
            Int(ProcessInfo.processInfo.environment["CAIX_SPEED_PROBE_RUNS"] ?? "") ?? 3)

        var monolithicRows: [[String: Any]] = []
        for run in 1...runs {
            let result = try await CoreAIPipeline.run(
                modelPath: monolithicPath,
                prompt: prompt,
                options: CoreAIPipeline.Options(
                    maxTokens: maxTokens,
                    temperature: 0,
                    applyChatTemplate: false,
                    verbose: false),
                onToken: nil)
            monolithicRows.append([
                "run": run,
                "generated_token_count": result.generatedTokenCount,
                "load_seconds": result.modelLoadSeconds,
                "prefill_seconds": result.prefillSeconds,
                "decode_seconds": result.decodeSeconds,
                "decode_tokens_per_second": result.decodeTokensPerSecond,
            ])
        }

        let tokenizerBundle = try ResolvedBundle.load(at: monolithicPath)
        let tokenizerEngine = try await LLMEngine.load(bundle: tokenizerBundle)
        let promptTokens = try tokenizerEngine.encodePrompt(
            messages: [["role": "user", "content": prompt]],
            applyChatTemplate: false)

        let manifestURL = URL(fileURLWithPath: stagedManifestPath)
        let manifest = try DistributedStageManifest.load(from: manifestURL)
        var stagedRows: [[String: Any]] = []
        for run in 1...runs {
            let loadStart = Date()
            let pipeline = try await DistributedSameMachinePipeline.make(
                manifest: manifest,
                handleFactory: DistributedCoreAIStageHandleFactory())
            let engine = try DistributedStagedEngine(
                pipeline: pipeline,
                maxContextLength: tokenizerBundle.maxContextLength)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            let started = Date()
            let result = try await engine.generate(
                promptTokens: promptTokens.map(Int32.init),
                options: DistributedStagedGenerationOptions(maxTokens: maxTokens),
                requestID: "speed-probe-\(UUID().uuidString)")
            let generateSeconds = Date().timeIntervalSince(started)
            stagedRows.append([
                "run": run,
                "generated_token_count": result.generatedTokenCount,
                "load_seconds": loadSeconds,
                "generate_seconds": generateSeconds,
                "tokens_per_second": generateSeconds > 0
                    ? Double(result.generatedTokenCount) / generateSeconds
                    : 0,
            ])
        }

        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            guard !sorted.isEmpty else { return 0 }
            return sorted[sorted.count / 2]
        }

        let monolithicMedian = median(
            monolithicRows.compactMap { $0["decode_tokens_per_second"] as? Double })
        let stagedMedian = median(
            stagedRows.compactMap { $0["tokens_per_second"] as? Double })
        let object: [String: Any] = [
            "monolithic": monolithicRows,
            "staged": stagedRows,
            "summary": [
                "max_tokens": maxTokens,
                "runs": runs,
                "monolithic_median_decode_tokens_per_second": monolithicMedian,
                "staged_median_tokens_per_second": stagedMedian,
                "staged_over_monolithic": monolithicMedian > 0
                    ? stagedMedian / monolithicMedian
                    : 0,
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outputPath))
        #else
        throw XCTSkip("requires COREAI_RUNTIME=1")
        #endif
    }
}
