#if COREAI_RUNTIME

import CryptoKit
import Foundation
import Jinja
import Tokenizers

/// Canonical July 2026 Gemma 4 prompt rendering used by conversion/runtime parity tests.
///
/// The updated Gemma repository publishes its authoritative template as a standalone Jinja file.
/// Production bundles inject that exact authenticated text into `tokenizer_config.json` for
/// swift-transformers; this helper deliberately renders the standalone source so a golden test
/// can catch either template drift or Swift Jinja incompatibility before model allocation.
public enum Gemma4ChatTemplateContract {
    /// SHA-256 of Google's 2026-07-09 Gemma 4 canonical template. The compatibility rewrite is
    /// deliberately unavailable to any other template: template drift must be reviewed against
    /// Transformers before CAIX will render it as the locked production model.
    public static let canonicalSHA256 =
        "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"

    /// Authenticated, compiled renderer retained by a resident model handle. Jinja compilation
    /// and template file I/O happen once at model load, never on the request hot path.
    struct ResidentRenderer: Sendable {
        private let template: Jinja.Template

        init(tokenizerDirectory: URL) throws {
            let source = try Gemma4ChatTemplateContract.compatibleTemplate(
                tokenizerDirectory: tokenizerDirectory)
            self.template = try Jinja.Template(
                source,
                with: .init(lstripBlocks: true, trimBlocks: true))
        }

        func encode(
            tokenizer: any Tokenizer,
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]? = nil,
            additionalContext: [String: any Sendable]? = nil
        ) throws -> [Int] {
            let rendered = try render(
                tokenizer: tokenizer,
                messages: messages,
                tools: tools,
                additionalContext: additionalContext)
            return tokenizer.encode(text: rendered, addSpecialTokens: false)
        }

        func render(
            tokenizer: any Tokenizer,
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]? = nil,
            additionalContext: [String: any Sendable]? = nil
        ) throws -> String {
            var context: [String: Jinja.Value] = [
                "messages": try .array(messages.map { try Jinja.Value(any: $0) }),
                "add_generation_prompt": .boolean(true),
                "bos_token": .string(tokenizer.bosToken ?? ""),
            ]
            if let tools {
                context["tools"] = try .array(tools.map { try Jinja.Value(any: $0) })
            }
            if let additionalContext {
                for (key, value) in additionalContext {
                    context[key] = try Jinja.Value(any: value)
                }
            }
            return try template.render(context)
        }
    }

    public static func encode(
        tokenizerDirectory: URL,
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) async throws -> [Int] {
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        let renderer = try ResidentRenderer(tokenizerDirectory: tokenizerDirectory)
        return try renderer.encode(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: additionalContext)
    }

    /// Render rich OpenAI/Google message objects with the authenticated Gemma 4 template.
    /// Nested tool calls, tool responses, and reasoning fields remain typed all the way into
    /// Jinja rather than being flattened into lossy `[String: String]` dictionaries.
    public static func encode(
        tokenizer: any Tokenizer,
        tokenizerDirectory: URL,
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        let renderer = try ResidentRenderer(tokenizerDirectory: tokenizerDirectory)
        return try renderer.encode(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: additionalContext)
    }

    /// Return the exact prompt text supplied to the tokenizer. Keeping rendering visible here
    /// lets contract tests compare both text and tokens with Transformers, and avoids relying on
    /// an opaque tokenizer-internal compiled-template cache.
    public static func render(
        tokenizerDirectory: URL,
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) async throws -> String {
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        let renderer = try ResidentRenderer(tokenizerDirectory: tokenizerDirectory)
        return try renderer.render(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: additionalContext)
    }

    /// Load and authenticate the standalone template, then make the source-level rewrites
    /// needed by swift-jinja 2.x. Python Jinja consumes whitespace around a left-trim expression
    /// after a literal `{`; swift-jinja preserves one protective space to avoid lexing `{{{`.
    /// It also leaves the second newline after the system block where Python's trim-block rules
    /// remove it. Re-expressing the literal brace as Jinja data and making that block's right trim
    /// explicit produces the exact Transformers 5.6.2 token stream without modifying the model
    /// repository or accepting a different template.
    public static func compatibleTemplate(tokenizerDirectory: URL) throws -> String {
        let templateURL = tokenizerDirectory
            .appendingPathComponent("chat_template.jinja")
            .resolvingSymlinksInPath()
        let data = try BoundedRegularFileReader.read(
            templateURL,
            maximumBytes: 256 * 1_024)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == canonicalSHA256 else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Gemma 4 chat_template.jinja SHA-256 mismatch: expected "
                    + "\(canonicalSHA256), got \(digest)")
        }
        guard let template = String(data: data, encoding: .utf8) else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Gemma 4 chat_template.jinja is not valid UTF-8")
        }

        var compatible = try replacingExactlyOnce(
            in: template,
            target:
                "properties:{ {{- format_parameters(params['properties'], params['required']) -}} },",
            replacement:
                "properties:{{- '{' + format_parameters(params['properties'], params['required']) + '},' -}}")
        compatible = try replacingExactlyOnce(
            in: compatible,
            target: "{%- endif %}\n\n{#- Pre-scan: find last user message index for reasoning guard -#}",
            replacement:
                "{%- endif -%}\n\n{#- Pre-scan: find last user message index for reasoning guard -#}")
        // The format_parameters macro's eight plain interpolation tags rely on Python Jinja
        // consuming whitespace left behind by surrounding `-%}` tags. Make that intent explicit
        // for swift-jinja. These replacements change no Python render, but prevent leading-space
        // tokenizer pieces such as ` properties` and ` type` in nested JSON schemas.
        let trimmedInterpolations: [(String, String)] = [
            ("{{ key }}", "{{- key -}}"),
            ("{{ value['description'] }}", "{{- value['description'] -}}"),
            ("{{ format_argument(value['enum']) }}", "{{- format_argument(value['enum']) -}}"),
            ("{{ format_argument(item_value | upper) }}", "{{- format_argument(item_value | upper) -}}"),
            (
                "{{ format_argument(item_value | map('upper') | list) }}",
                "{{- format_argument(item_value | map('upper') | list) -}}"
            ),
            ("{{ item_key }}", "{{- item_key -}}"),
            ("{{ format_argument(item_value) }}", "{{- format_argument(item_value) -}}"),
            ("{{ value['type'] | upper }}", "{{- value['type'] | upper -}}"),
        ]
        for (target, replacement) in trimmedInterpolations {
            compatible = try replacingExactlyOnce(
                in: compatible, target: target, replacement: replacement)
        }
        // swift-jinja protects a literal `{` followed by whitespace and another Jinja opening
        // delimiter from becoming the ambiguous source sequence `{{{`; that protection is a
        // rendered space. Emit each affected schema brace from an expression instead. There are
        // seven remaining sites after the inline `parameters.properties` rewrite above.
        let structuralBraces: [(String, String)] = [
            (
                "{{- key -}}:{\n            {%- if value['description'] -%}",
                "{{- key + ':{' -}}\n            {%- if value['description'] -%}"
            ),
            (
                "items:{\n                    {%- set ns_items = namespace(found_first=false) -%}",
                "{{- 'items:{' -}}\n                    {%- set ns_items = namespace(found_first=false) -%}"
            ),
            (
                "properties:{\n                                {%- if item_value is mapping -%}",
                "{{- 'properties:{' -}}\n                                {%- if item_value is mapping -%}"
            ),
            (
                "properties:{\n                    {{- format_parameters(value['properties'], value['required'] | default([])) -}}",
                "{{- 'properties:{' -}}\n                    {{- format_parameters(value['properties'], value['required'] | default([])) -}}"
            ),
            (
                "properties:{\n                    {{- format_parameters(value, value['required'] | default([]), filter_keys=true) -}}",
                "{{- 'properties:{' -}}\n                    {{- format_parameters(value, value['required'] | default([]), filter_keys=true) -}}"
            ),
            (
                ",parameters:{\n        {%- if params['properties'] -%}",
                "{{- ',parameters:{' -}}\n        {%- if params['properties'] -%}"
            ),
            (
                ",response:{\n        {%- if response_declaration['description'] -%}",
                "{{- ',response:{' -}}\n        {%- if response_declaration['description'] -%}"
            ),
        ]
        for (target, replacement) in structuralBraces {
            compatible = try replacingExactlyOnce(
                in: compatible, target: target, replacement: replacement)
        }
        return compatible
    }

    private static func replacingExactlyOnce(
        in source: String,
        target: String,
        replacement: String
    ) throws -> String {
        let pieces = source.components(separatedBy: target)
        guard pieces.count == 2 else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Gemma 4 canonical-template compatibility pattern count was \(pieces.count - 1), expected 1")
        }
        return pieces[0] + replacement + pieces[1]
    }
}

#endif
