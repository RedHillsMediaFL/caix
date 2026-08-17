#if COREAI_RUNTIME

import Foundation

/// End-to-end exact-greedy Qwen3.8 MTP decode loop.
final class Qwen38NativeMTPEngine {
    private let target: LLMEngine
    private let draft: Qwen38MTPNativeRunner
    private let loadSeconds: Double

    static func load(bundle: ResolvedBundle, verbose: Bool) async throws
        -> Qwen38NativeMTPEngine
    {
        guard let mtpURL = bundle.mtpAimodelURL else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Qwen3.8 MTP metadata has no sidecar asset")
        }
        let started = Date()
        async let target = LLMEngine.load(bundle: bundle, verbose: verbose)
        async let draft = Qwen38MTPNativeRunner.load(aimodelURL: mtpURL, verbose: verbose)
        return try await Qwen38NativeMTPEngine(
            target: target,
            draft: draft,
            loadSeconds: Date().timeIntervalSince(started))
    }

    private init(target: LLMEngine, draft: Qwen38MTPNativeRunner, loadSeconds: Double) {
        self.target = target
        self.draft = draft
        self.loadSeconds = loadSeconds
    }

    func encodePrompt(
        messages: [[String: String]],
        tools: [[String: any Sendable]]?,
        applyChatTemplate: Bool
    ) throws -> [Int] {
        try target.encodePrompt(
            messages: messages, tools: tools, applyChatTemplate: applyChatTemplate)
    }

    func generateAutoregressive(
        promptTokens: [Int],
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result {
        try await target.generate(
            promptTokens: promptTokens, options: options, onToken: onToken)
    }

    static func run(
        modelPath: String,
        prompt: String,
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result {
        let bundle = try ResolvedBundle.load(at: modelPath)
        let engine = try await load(bundle: bundle, verbose: options.verbose)
        let promptTokens = try engine.target.encodePrompt(
            messages: [["role": "user", "content": prompt]],
            applyChatTemplate: options.applyChatTemplate)
        return try await engine.generate(
            promptTokens: promptTokens, options: options, onToken: onToken)
    }

    func generate(
        promptTokens: [Int],
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.Result {
        guard options.temperature == 0 else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "Qwen3.8 MTP preserves exact greedy decoding only")
        }
        let capacity = target.resolvedCapacity(promptCount: promptTokens.count, options: options)
        try target.allocateKVCache(capacity: capacity)
        draft.reset(capacity: capacity)

        let prefillStart = Date()
        let prompt = promptTokens.map(Int32.init)
        let targetChunk = target.resolvedPrefillChunkSize(promptCount: prompt.count)
        var targetHidden: [[Float16]] = []
        var lastTargetLogits: [Float] = []
        var offset = 0
        while offset < prompt.count {
            let end = min(offset + targetChunk, prompt.count)
            let outputs = try await target.forwardWithHidden(
                tokens: Array(prompt[offset..<end]))
            targetHidden.append(contentsOf: outputs.hidden)
            lastTargetLogits = outputs.lastLogits
            offset = end
        }
        guard let bonus = Self.argmax(lastTargetLogits) else {
            throw CoreAIPipeline.RuntimeError.modelContract("empty Qwen3.8 target logits")
        }

        // Qwen MTP training alignment: [prompt1 ... promptN-1, target bonus] is paired with the
        // target hidden rows for [prompt0 ... promptN-1].
        let shifted = Array(prompt.dropFirst()) + [bonus]
        var seedToken: Int32 = 0
        var seedHidden: [Float16] = []
        offset = 0
        while offset < shifted.count {
            let end = min(offset + 16, shifted.count)
            let output = try await draft.forward(
                tokens: Array(shifted[offset..<end]),
                hiddenRows: Array(targetHidden[offset..<end]))
            seedToken = output.greedyTokens.last!
            seedHidden = output.hiddenRows.last!
            offset = end
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        var state = try Qwen38GenerationState(layout: .native)
        try state.recordForward(tokens: prompt.count)
        let controller = Qwen38MTPDecoder()
        var currentBonus = bonus
        var generated: [Int32] = []
        var streamedText = ""
        var stopReason: CoreAIPipeline.StopReason = .maxTokens
        let stopSequences = options.stopSequences.filter { !$0.isEmpty }
        var stopText: String?

        func appendVisible(_ token: Int32) -> Bool {
            if target.isStopToken(Int(token)) {
                stopReason = .eos
                return false
            }
            generated.append(token)
            guard onToken != nil || !stopSequences.isEmpty else { return true }
            let text = target.tokenizer.decode(tokens: generated.map(Int.init))
            if let range = CoreAIPipeline.firstStopRange(in: text, stopSequences: stopSequences) {
                let visible = String(text[..<range.lowerBound])
                if let onToken, visible.hasPrefix(streamedText) {
                    let delta = String(visible.dropFirst(streamedText.count))
                    if !delta.isEmpty { onToken(delta) }
                }
                streamedText = visible
                stopText = visible
                stopReason = .stopSequence
                return false
            }
            let visible = stopSequences.isEmpty
                ? text
                : CoreAIPipeline.visibleTextAvoidingPartialStop(text, stopSequences: stopSequences)
            if let onToken, visible.hasPrefix(streamedText) {
                let delta = String(visible.dropFirst(streamedText.count))
                if !delta.isEmpty { onToken(delta) }
            }
            streamedText = visible
            return true
        }

        let decodeStart = Date()
        var draftedTokens = 0
        var acceptedTokens = 0
        var verificationRounds = 0
        var restoredRounds = 0
        var draftWidth = 3
        var recentAcceptance: [(accepted: Int, drafted: Int)] = []
        guard max(0, options.maxTokens) > 0 else {
            return CoreAIPipeline.Result(
                text: "", promptTokenCount: prompt.count, generatedTokenCount: 0,
                stopReason: .maxTokens, modelLoadSeconds: loadSeconds,
                prefillSeconds: prefillSeconds, decodeSeconds: 0, generatedTokenIDs: [])
        }
        guard appendVisible(currentBonus) else {
            return CoreAIPipeline.Result(
                text: stopText ?? "", promptTokenCount: prompt.count,
                generatedTokenCount: generated.count, stopReason: stopReason,
                modelLoadSeconds: loadSeconds, prefillSeconds: prefillSeconds,
                decodeSeconds: Date().timeIntervalSince(decodeStart),
                generatedTokenIDs: generated)
        }
        generation: while generated.count < max(0, options.maxTokens) {
            if target.processedTokenCount >= target.maxContextLength {
                stopReason = .contextLimit
                break
            }
            let draftPosition = draft.processedTokenCount
            var proposals = [seedToken]
            var draftHidden = seedHidden
            while proposals.count < draftWidth {
                let step = try await draft.forward(
                    tokens: [proposals.last!], hiddenRows: [draftHidden])
                proposals.append(step.greedyTokens[0])
                draftHidden = step.hiddenRows[0]
            }
            draftedTokens += proposals.count
            verificationRounds += 1

            let fixedSnapshot = try target.snapshotQwen38FixedState()
            let verify = try await target.forwardGreedyWithHidden(
                tokens: [currentBonus] + proposals.dropLast())
            let targetGreedy = verify.tokens
            guard targetGreedy.count == proposals.count else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Qwen3.8 verify row count does not match its draft width")
            }
            let result = try controller.verify(
                proposals: proposals, targetGreedyTokens: targetGreedy, state: &state)
            let accepted = result.acceptedDraftTokens.count
            acceptedTokens += accepted
            recentAcceptance.append((accepted, proposals.count))
            if recentAcceptance.count > 4 { recentAcceptance.removeFirst() }

            if case .restoreFixedStateAndReplay = result.stateAction {
                restoredRounds += 1
                try target.restoreQwen38FixedState(fixedSnapshot)
                let replay = [currentBonus] + Array(proposals.prefix(accepted))
                _ = try await target.forwardWithHidden(tokens: replay)
            }
            try controller.commitReplay(result, state: &state)

            // Preserve the MTP layer's own exact cache alignment, then condition the next seed on
            // the target-verified hidden row corresponding to the new bonus token.
            let keepDrafted = min(accepted, proposals.count - 1)
            draft.rollback(to: draftPosition + keepDrafted)
            var syncOutput: Qwen38MTPNativeRunner.Output?
            if accepted > keepDrafted {
                for index in keepDrafted..<accepted {
                    syncOutput = try await draft.forward(
                        tokens: [proposals[index]], hiddenRows: [verify.hidden[index]])
                }
            }
            if let correction = result.correctionToken {
                syncOutput = try await draft.forward(
                    tokens: [correction], hiddenRows: [verify.hidden[accepted]])
                currentBonus = correction
            } else {
                currentBonus = proposals.last!
            }
            guard let synchronized = syncOutput else {
                throw CoreAIPipeline.RuntimeError.modelContract(
                    "Qwen3.8 MTP failed to synchronize its next seed")
            }
            seedToken = synchronized.greedyTokens[0]
            seedHidden = synchronized.hiddenRows[0]

            if recentAcceptance.count == 4 {
                let accepted = recentAcceptance.reduce(0) { $0 + $1.accepted }
                let drafted = recentAcceptance.reduce(0) { $0 + $1.drafted }
                let rate = Double(accepted) / Double(drafted)
                if rate < 0.65 { draftWidth = 2 }
                if rate > 0.82 { draftWidth = 3 }
            }

            let committed = result.acceptedDraftTokens
                + (result.correctionToken.map { [$0] } ?? [])
            for token in committed {
                if generated.count >= options.maxTokens { break generation }
                guard appendVisible(token) else { break generation }
            }
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let text = stopText ?? target.tokenizer.decode(tokens: generated.map(Int.init))
        if stopText == nil, let onToken, text.hasPrefix(streamedText) {
            let delta = String(text.dropFirst(streamedText.count))
            if !delta.isEmpty { onToken(delta) }
        }
        if options.verbose {
            FileHandle.standardError.write(Data(String(
                format: "[qwen-mtp] decode %d tokens in %.3fs (%.1f tok/s), acceptance=%d/%d (%.1f%%), rounds=%d restores=%d\n",
                generated.count, decodeSeconds,
                decodeSeconds > 0 ? Double(generated.count) / decodeSeconds : 0,
                acceptedTokens, draftedTokens,
                draftedTokens > 0 ? 100 * Double(acceptedTokens) / Double(draftedTokens) : 0,
                verificationRounds, restoredRounds).utf8))
        }
        if ProcessInfo.processInfo.environment["CAIX_DEBUG_TOKEN_IDS"] != nil {
            FileHandle.standardError.write(Data("[qwen-mtp] token_ids=\(generated)\n".utf8))
        }
        return CoreAIPipeline.Result(
            text: text,
            promptTokenCount: prompt.count,
            generatedTokenCount: generated.count,
            stopReason: stopReason,
            modelLoadSeconds: loadSeconds,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            generatedTokenIDs: generated)
    }

    private static func argmax(_ values: [Float]) -> Int32? {
        guard var bestValue = values.first else { return nil }
        var best = 0
        for index in values.indices.dropFirst() where values[index] > bestValue {
            best = index
            bestValue = values[index]
        }
        return Int32(best)
    }
}

#endif
