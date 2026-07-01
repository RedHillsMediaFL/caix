#if COREAI_RUNTIME

import CoreAI
import Foundation
import Tokenizers

/// Speculative decoding (a.k.a. MTP / draft-model assisted decoding) over two native
/// ``LLMEngine`` handles: a large **target** and a small **draft**, each with its own
/// independent KV cache.
///
/// ## The loop (greedy)
/// Maintain a `committed` token sequence (prompt + everything emitted). Both engines keep their
/// KV caches consistent with `committed`, holding everything **except** the last token (the
/// *anchor*) — the anchor is fed together with the next batch. Each iteration:
///
/// 1. **Draft.** From the anchor, the draft autoregressively proposes `K` tokens
///    `d₀ … d_{K-1}` (one cheap forward each).
/// 2. **Verify.** The target runs **one** batched forward over `[anchor, d₀ … d_{K-1}]`
///    (`K+1` tokens), yielding its greedy prediction at every draft slot plus one *bonus*
///    slot after the last draft.
/// 3. **Accept longest prefix.** Walk the drafts: accept `dᵢ` while it equals the target's
///    greedy token at slot `i`; at the first mismatch, take the target's token as the
///    *correction* and stop. If all `K` match, append the bonus token. So each iteration
///    commits between 1 and `K+1` tokens — and **every committed token is exactly the
///    target's greedy choice**, so the decoded text is identical to running the target alone.
/// 4. **Commit + roll back.** Append the accepted prefix + correction to `committed`, then roll
///    both KV caches back to drop the rejected suffix (the target wrote `K` draft positions; the
///    draft wrote up to `K-1`). Rollback is just a counter move — stale slots are overwritten by
///    the next forward and never read (causal attention is bounded by `position_ids` length).
///
/// This ports the accept-longest-greedy-prefix / first-mismatch-correction logic of
/// `Gemma4SpeculativeDecoder` (coreai-gemma4) onto real batched target forwards + KV commits.
///
/// ## Constraints
/// Target and draft must share a tokenizer/vocabulary (the draft proposes token ids the target
/// verifies). The rollback is a correct rewind only for **standard attention** KV state, so the
/// pair must be standard-attention models (e.g. the Qwen3 family) — not the hybrid `qwen3_5`
/// SSM models, whose recurrent state cannot be position-truncated.
///
/// Not `Sendable`: like ``LLMEngine``, an instance is driven by a single task.
final class SpeculativeEngine {
    let target: LLMEngine
    let draft: LLMEngine
    let draftTokens: Int

    private init(target: LLMEngine, draft: LLMEngine, draftTokens: Int) {
        self.target = target
        self.draft = draft
        self.draftTokens = max(1, draftTokens)
    }

    /// Load a target + draft bundle pair into a speculative engine (two persistent handles with
    /// independent KV). Throws if the two models do not share a vocabulary.
    static func load(
        targetPath: String,
        draftPath: String,
        draftTokens: Int,
        verbose: Bool = false
    ) async throws -> SpeculativeEngine {
        let targetBundle = try ResolvedBundle.load(at: targetPath)
        let draftBundle = try ResolvedBundle.load(at: draftPath)
        guard targetBundle.minKVCapacity == 0, draftBundle.minKVCapacity == 0 else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "classic speculative decoding is only supported for standard-attention pairs")
        }
        // Load concurrently — two independent graph links + tokenizer reads.
        async let targetTask = LLMEngine.load(bundle: targetBundle, verbose: verbose)
        async let draftTask = LLMEngine.load(bundle: draftBundle, verbose: verbose)
        let target = try await targetTask
        let draft = try await draftTask

        guard target.vocabSize == draft.vocabSize else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "speculative decoding requires a shared vocabulary: target vocab "
                    + "\(target.vocabSize) != draft vocab \(draft.vocabSize)")
        }
        if verbose {
            FileHandle.standardError.write(
                Data(
                    "[coreai] speculative: target=\(targetBundle.name) draft=\(draftBundle.name) K=\(draftTokens)\n"
                        .utf8))
        }
        return SpeculativeEngine(target: target, draft: draft, draftTokens: draftTokens)
    }

    /// Encode a prompt with the target tokenizer (the target defines the output contract).
    func encodePrompt(
        messages: [[String: String]],
        tools: [[String: any Sendable]]? = nil,
        applyChatTemplate: Bool
    ) throws -> [Int] {
        try target.encodePrompt(messages: messages, tools: tools, applyChatTemplate: applyChatTemplate)
    }

    // MARK: - Generation

    /// Run speculative prefill + decode over `promptTokens`, streaming decoded deltas to
    /// `onToken`. Greedy: `temperature` is ignored (verification is argmax-based).
    func generate(
        promptTokens: [Int],
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)?
    ) async throws -> CoreAIPipeline.SpeculativeResult {
        func log(_ s: @autoclosure () -> String) {
            if options.verbose { FileHandle.standardError.write(Data(("[coreai] " + s() + "\n").utf8)) }
        }

        let maxTokens = max(0, options.maxTokens)
        let maxDraftTokens = draftTokens
        var tuner = DraftTokenTuner(initial: min(4, maxDraftTokens), maximum: maxDraftTokens)

        // Size both KV caches: room for prompt + all generated + a full draft batch of headroom,
        // floored to each model's per-model minimum (no-op for standard models).
        let need = promptTokens.count + maxTokens + maxDraftTokens + 8
        let requested = max(options.kvCapacity ?? need, need)
        func capacity(_ engine: LLMEngine) -> Int {
            min(max(requested, engine.minKVCapacity + maxTokens), engine.maxContextLength)
        }
        try target.allocateKVCache(capacity: capacity(target))
        try draft.allocateKVCache(capacity: capacity(draft))
        log(
            "speculative prompt -> \(promptTokens.count) tokens, K<=\(maxDraftTokens), "
                + "KV target=\(capacity(target)) draft=\(capacity(draft))")

        // Prefill both models on the prompt (target's last logits seed the first token). The two
        // engines are independent non-Sendable handles confined to this task, so prefill runs
        // sequentially (it is cheap relative to decode).
        let prefillStart = Date()
        let prompt32 = promptTokens.map { Int32($0) }
        let targetPrefillLogits = try await target.step(tokens: prompt32)
        _ = try await draft.step(tokens: prompt32)
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        // `committed` = prompt + everything emitted. Invariant entering each iteration: both
        // engines have processed `committed` minus its last token (the anchor).
        var committed = promptTokens
        var generated: [Int] = []
        var streamedText = ""
        var stopReason: CoreAIPipeline.StopReason = .maxTokens
        var draftedTokens = 0
        var acceptedDraftTokens = 0
        var iterations = 0

        // Emit one committed token (apply the same stop/maxTokens gate as the target-only path).
        // Returns false when generation should stop (token withheld).
        func emit(_ token: Int) -> Bool {
            if target.isStopToken(token) {
                stopReason = .eos
                return false
            }
            if generated.count >= maxTokens {
                stopReason = .maxTokens
                return false
            }
            if committed.count >= target.maxContextLength {
                stopReason = .contextLimit
                return false
            }
            generated.append(token)
            committed.append(token)
            if let onToken {
                let text = target.tokenizer.decode(tokens: generated)
                if text.hasPrefix(streamedText) {
                    let delta = String(text.dropFirst(streamedText.count))
                    if !delta.isEmpty { onToken(delta) }
                }
                streamedText = text
            }
            return true
        }

        let decodeStart = Date()

        // First token: the target's greedy choice from prefill (pure target, no draft involved).
        // Once emitted it is `committed.last` — the anchor that seeds the first draft batch.
        var running = emit(Sampler.argmax(targetPrefillLogits))

        // Speculative loop. Invariant entering each pass: both engines have processed `committed`
        // minus its last token (the anchor), which is fed together with the next draft batch.
        while running {
            let L = committed.count
            let anchor = committed[L - 1]
            let activeK = min(tuner.current, max(1, maxTokens - generated.count))

            // 1. DRAFT: propose activeK tokens. Feed the unconsumed committed tail (>= the anchor)
            //    to catch the draft up and produce d₀, then activeK-1 single-token forwards.
            var drafts: [Int] = []
            drafts.reserveCapacity(activeK)
            let tail = committed[draft.processedTokenCount..<L].map { Int32($0) }
            var dLogits = try await draft.step(tokens: tail)
            var d = Sampler.argmax(dLogits)
            drafts.append(d)
            for _ in 1..<activeK {
                dLogits = try await draft.step(tokens: [Int32(d)])
                d = Sampler.argmax(dLogits)
                drafts.append(d)
            }

            // 2. VERIFY: one batched target forward over [anchor, d₀ … d_{K-1}].
            let verifyInput = [Int32(anchor)] + drafts.map { Int32($0) }
            let rows = try await target.forwardAllRows(tokens: verifyInput)
            // rows[i] is the target's distribution for draft slot i; rows[K] is the bonus slot.

            // 3. ACCEPT longest matching greedy prefix; correct the first mismatch.
            let verdict = Self.verify(drafts: drafts, targetRows: rows)
            let numAccepted = verdict.acceptedCount
            let correction = verdict.correctionToken  // target's token at the divergence/bonus

            draftedTokens += activeK
            acceptedDraftTokens += numAccepted
            iterations += 1
            tuner.observe(accepted: numAccepted, drafted: activeK)

            // 4. COMMIT accepted prefix + correction (gated), then roll both caches back.
            //    Roll back BEFORE breaking so the (unused) post-stop state stays valid anyway.
            target.rollbackKV(to: L + numAccepted)
            draft.rollbackKV(to: min(L + numAccepted, L + activeK - 1))

            for t in verdict.acceptedTokens {
                if !emit(t) { running = false; break }
            }
            if running {
                if !emit(correction) { running = false }
            }
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let finalText = target.tokenizer.decode(tokens: generated)
        let accRate = draftedTokens > 0 ? Double(acceptedDraftTokens) / Double(draftedTokens) : 0
        log(
            String(
                format:
                    "speculative decode %d tokens in %.3fs (%.1f tok/s) over %d target passes; "
                    + "drafts %d accepted %d (%.1f%% accept), final K=%d, stop=%@",
                generated.count, decodeSeconds,
                decodeSeconds > 0 ? Double(generated.count) / decodeSeconds : 0,
                iterations, draftedTokens, acceptedDraftTokens, accRate * 100,
                tuner.current, stopReason.rawValue))

        return CoreAIPipeline.SpeculativeResult(
            text: finalText,
            promptTokenCount: promptTokens.count,
            generatedTokenCount: generated.count,
            stopReason: stopReason,
            modelLoadSeconds: max(target.loadSeconds, draft.loadSeconds),
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            draftTokens: tuner.current,
            draftedTokens: draftedTokens,
            acceptedDraftTokens: acceptedDraftTokens,
            iterations: iterations)
    }

    // MARK: - Verification (ported from Gemma4SpeculativeDecoder)

    /// Outcome of verifying one draft batch against the target's greedy predictions: the longest
    /// matching prefix, plus the target's token to commit next (the correction at the first
    /// mismatch, or the bonus token when every draft matched).
    struct Verdict {
        let acceptedTokens: [Int]
        let correctionToken: Int
        var acceptedCount: Int { acceptedTokens.count }
    }

    /// Accept-longest-prefix: `targetRows[i]` is the target's logits for draft slot `i`
    /// (`drafts.count` of them) and `targetRows[last]` is the bonus slot. Accept `drafts[i]`
    /// while it equals `argmax(targetRows[i])`; the commit token is the target's token at the
    /// first divergence, or the bonus token if all drafts were accepted.
    static func verify(drafts: [Int], targetRows: [[Float]]) -> Verdict {
        var accepted: [Int] = []
        accepted.reserveCapacity(drafts.count)
        for i in 0..<drafts.count {
            let targetToken = Sampler.argmax(targetRows[i])
            if drafts[i] == targetToken {
                accepted.append(drafts[i])
            } else {
                return Verdict(acceptedTokens: accepted, correctionToken: targetToken)
            }
        }
        // Every draft matched — commit the bonus token from the final (post-draft) slot.
        return Verdict(acceptedTokens: accepted, correctionToken: Sampler.argmax(targetRows[drafts.count]))
    }
}

#endif
