import Foundation

// Best-effort, server-side JSON coercion for backends that cannot do true constrained decoding
// (e.g. the EAGLE speculative backend, or any non-CoreAILM build). Instead of rejecting a
// `response_format: json_schema|json_object` request or silently returning free-form prose, the
// server injects a strict "JSON-only" system instruction, generates normally, and then extracts
// the first balanced top-level JSON object from the model's text. This is NOT a hard guarantee —
// it just raises the odds that a downstream strict-JSON validator accepts the local answer instead
// of falling back to cloud. When extraction fails after one retry the raw text is returned
// unchanged, so there is no regression relative to today's free-form output.
//
// Pure Foundation (`JSONSerialization`) so it is unit-testable without the runtime.
public enum JSONCoercion {
    /// A strict system instruction telling the model to emit ONLY a conforming JSON object.
    /// Returns `nil` for `.text` (no coercion needed).
    public static func systemInstruction(for format: OpenAIResponseFormat) -> String? {
        switch format {
        case .text:
            return nil
        case .jsonObject:
            return jsonOnlyPreamble
        case .jsonSchema(let schema):
            var text = jsonOnlyPreamble
            if let description = schema.description, !description.isEmpty {
                text += " \(description)"
            }
            let keys = requiredKeys(for: format)
            if !keys.isEmpty {
                // Name the required fields as the TOP-LEVEL keys and explicitly forbid wrapping —
                // otherwise the model tends to nest the object under the schema name.
                text +=
                    " The JSON object's top-level keys must be exactly these fields: "
                    + keys.joined(separator: ", ")
                    + ". Do NOT wrap them inside any other key or envelope."
            }
            text +=
                " It must conform exactly to this JSON Schema (correct types, all required fields at the top level): "
                + schema.schema.compactJSONString
            return text
        }
    }

    /// A short, stricter user reminder appended before the single coercion retry.
    public static func retryReminder() -> String {
        "Your previous reply did not match the required JSON shape. Output ONLY a single JSON "
            + "object with the required fields as its top-level keys — no wrapper key, no markdown, "
            + "no extra text."
    }

    /// Place the coercion instruction where the model will actually see it. The Gemma-4 chat
    /// template only renders a system block for `messages[0]` (role system/developer); a system
    /// message appended anywhere else is dropped. So merge into a leading system/developer message
    /// when present, otherwise insert a fresh system message at index 0.
    public static func inject(
        instruction: String,
        into messages: inout [[String: any Sendable]]
    ) {
        if let first = messages.first,
           let role = first["role"] as? String,
           role == "system" || role == "developer",
           let content = first["content"] as? String {
            messages[0]["content"] = content + "\n\n" + instruction
        } else {
            messages.insert(["role": "system", "content": instruction], at: 0)
        }
    }

    /// Top-level `required` keys declared by a `.jsonSchema` format; empty for `.jsonObject`/`.text`.
    public static func requiredKeys(for format: OpenAIResponseFormat) -> [String] {
        guard case .jsonSchema(let schema) = format else { return [] }
        guard case .object(let object) = schema.schema else { return [] }
        guard case .array(let required)? = object["required"] else { return [] }
        return required.compactMap { value in
            if case .string(let key) = value { return key }
            return nil
        }
    }

    /// Extract the first balanced top-level JSON object from `text`. Strips a surrounding
    /// ```` ```json … ``` ```` (or bare ```` ``` ```` ) fence, scans for the first `{` and its
    /// matching `}` (respecting strings and escapes), parses the candidate, and — when
    /// `requiredKeys` is non-empty — verifies every required key is present at the top level.
    /// Returns the exact candidate substring on success, or `nil` when no conforming object is found.
    public static func extractJSONObject(from text: String, requiredKeys: [String]) -> String? {
        let unfenced = stripCodeFence(text)
        guard let candidate = firstBalancedObject(in: unfenced) else { return nil }
        guard let data = candidate.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data),
              let object = parsed as? [String: Any]
        else {
            return nil
        }
        if requiredKeys.isEmpty {
            return candidate
        }
        if requiredKeys.allSatisfy({ object[$0] != nil }) {
            return candidate
        }
        // Common failure mode: the model wraps the real object under a single envelope key
        // (e.g. `{"note": {…required fields…}}`). Unwrap when the sole value carries every
        // required top-level key, and re-serialize that inner object.
        if object.count == 1,
           let inner = object.values.first as? [String: Any],
           requiredKeys.allSatisfy({ inner[$0] != nil }),
           let innerData = try? JSONSerialization.data(
               withJSONObject: inner, options: [.sortedKeys]),
           let innerString = String(data: innerData, encoding: .utf8) {
            return innerString
        }
        return nil
    }

    // MARK: - Internals

    private static let jsonOnlyPreamble =
        "Respond with ONLY a single valid JSON object. No markdown, no code fences, no explanation — output only the JSON."

    /// Drop a leading ```` ``` ```` / ```` ```json ```` fence line and the trailing ```` ``` ````
    /// fence, if the (trimmed) text is fenced. A no-op for unfenced text.
    static func stripCodeFence(_ text: String) -> String {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("```") else { return trimmed }
        if let firstNewline = trimmed.firstIndex(of: "\n") {
            trimmed = String(trimmed[trimmed.index(after: firstNewline)...])
        } else {
            trimmed = String(trimmed.dropFirst(3))
        }
        if let closing = trimmed.range(of: "```", options: .backwards) {
            trimmed = String(trimmed[..<closing.lowerBound])
        }
        return trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Return the substring from the first `{` to its matching `}`, tracking nesting depth while
    /// ignoring braces and quotes that appear inside JSON strings. `nil` when unbalanced/absent.
    static func firstBalancedObject(in text: String) -> String? {
        let characters = Array(text)
        guard let start = characters.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < characters.count {
            let character = characters[index]
            if inString {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == "\"" {
                    inString = false
                }
            } else {
                switch character {
                case "\"":
                    inString = true
                case "{":
                    depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        return String(characters[start...index])
                    }
                default:
                    break
                }
            }
            index += 1
        }
        return nil
    }
}
