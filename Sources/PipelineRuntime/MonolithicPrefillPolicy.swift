import Foundation

enum MonolithicPrefillPolicy {
    static let tracedQueryWidth = 16

    static func resolvedChunkSize(
        promptCount: Int,
        isStatefulMonolithic: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        let prompt = max(1, promptCount)
        if let explicit = explicitPrefillChunk(promptCount: prompt, environment: environment) {
            return explicit
        }
        if let mode = normalized(environment["COREAI_PREFILL_MODE"]) {
            switch mode {
            case "token", "tokenwise", "single", "1", "true", "yes":
                return 1
            case "batch", "batched", "0", "false", "no", "off", "disable", "disabled":
                return prompt
            default:
                break
            }
        }
        if isStatefulMonolithic {
            return min(tracedQueryWidth, prompt)
        }
        return prompt
    }

    static func coreAILanguageModelsChunkThreshold(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        if environment["COREAI_CHUNK_THRESHOLD"] != nil {
            return nil
        }
        if let explicit = explicitPrefillChunk(promptCount: Int.max, environment: environment) {
            return explicit == Int.max ? nil : max(1, explicit)
        }
        if let mode = normalized(environment["COREAI_PREFILL_MODE"]) {
            switch mode {
            case "token", "tokenwise", "single", "1", "true", "yes":
                return 1
            case "batch", "batched", "0", "false", "no", "off", "disable", "disabled":
                return nil
            default:
                break
            }
        }
        return tracedQueryWidth
    }

    static func isMonolithicLanguageBundle(_ bundle: ResolvedBundle) -> Bool {
        bundle.manifest.kind == "llm"
            && !bundle.isDiffusion
            && bundle.decodeAimodelURL == nil
            && bundle.manifest.assets.byName.keys.allSatisfy { $0 == "main" }
    }

    private static func explicitPrefillChunk(
        promptCount: Int,
        environment: [String: String]
    ) -> Int? {
        guard let raw = normalized(environment["COREAI_PREFILL_CHUNK"]) else {
            return nil
        }
        if let parsed = Int(raw), parsed > 0 {
            return min(parsed, max(1, promptCount))
        }
        if ["0", "false", "no", "off", "disable", "disabled", "batch", "batched"].contains(raw) {
            return max(1, promptCount)
        }
        return nil
    }

    private static func normalized(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return trimmed.isEmpty ? nil : trimmed
    }
}
