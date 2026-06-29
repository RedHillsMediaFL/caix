import Foundation

/// A persistently-loaded `.aimodel` LLM bundle: the model graph + tokenizer stay resident so a
/// server can serve many requests against one hot handle (no per-request load/release).
///
/// This is the reusable handle the `ModelManager` keeps in its registry. It is intentionally
/// **not** `Sendable`: a handle must be driven by one task at a time. The `ModelManager` is
/// responsible for confining each handle and serialising `generate` calls against it.
///
/// With the Core AI runtime compiled out (no `COREAI_RUNTIME`), ``load(bundlePath:verbose:)``
/// throws ``CoreAIPipeline/RuntimeError/runtimeUnavailable`` so the serving layer still builds
/// and reports the runtime as unavailable rather than failing to compile.
public final class PersistentModel {
    /// Bundle name (from `metadata.json`).
    public let name: String
    /// Resolved bundle root directory.
    public let bundlePath: String
    public let vocabSize: Int
    public let maxContextLength: Int
    /// Approximate on-disk footprint of the `.aimodel` package (used for memory accounting).
    public let bundleByteSize: UInt64
    /// Seconds spent loading the graph + tokenizer.
    public let loadSeconds: Double

    #if COREAI_RUNTIME
    private let generateImpl: PipelinedLanguageHandle.Generate

    private init(
        bundle: ResolvedBundle,
        loadSeconds: Double,
        generate: @escaping PipelinedLanguageHandle.Generate
    ) {
        self.generateImpl = generate
        self.name = bundle.name
        self.bundlePath = bundle.root.path
        self.vocabSize = bundle.vocabSize
        self.maxContextLength = bundle.maxContextLength
        self.bundleByteSize = Self.directorySize(bundle.aimodelURL)
        self.loadSeconds = loadSeconds
    }
    #else
    // Unreachable in the standalone build: `load` throws `.runtimeUnavailable` before any
    // instance is constructed. Present only so the class is initializable when compiled out.
    private init() {
        self.name = ""
        self.bundlePath = ""
        self.vocabSize = 0
        self.maxContextLength = 0
        self.bundleByteSize = 0
        self.loadSeconds = 0
    }
    #endif

    /// Load a bundle directory into a hot handle.
    public static func load(bundlePath: String, verbose: Bool = false) async throws -> PersistentModel {
        #if COREAI_RUNTIME
        let bundle = try ResolvedBundle.load(at: bundlePath)
        let env = ProcessInfo.processInfo.environment
        let fastHybridAllowed = env["COREAI_FAST_HYBRID_ENGINE"] != nil
        let fastCompatible = bundle.minKVCapacity == 0 || fastHybridAllowed
        if env["COREAI_LEGACY_ENGINE"] == nil,
           fastCompatible,
           let fast = try await PipelinedLLM.loadPersistent(bundlePath: bundlePath, verbose: verbose)
        {
            return PersistentModel(bundle: bundle, loadSeconds: fast.loadSeconds, generate: fast.generate)
        }
        if env["COREAI_LEGACY_ENGINE"] == nil {
            return PersistentModel(
                bundle: bundle,
                loadSeconds: 0,
                generate: { messages, options, tools, onToken in
                    if fastCompatible, let fast = try await PipelinedLLM.runIfLanguageMessages(
                        modelPath: bundlePath,
                        messages: messages,
                        tools: tools,
                        options: options,
                        onToken: onToken)
                    {
                        return fast
                    }
                    let engine = try await LLMEngine.load(bundle: bundle, verbose: options.verbose)
                    let promptTokens = try engine.encodePrompt(
                        messages: messages, tools: tools, applyChatTemplate: options.applyChatTemplate)
                    return try await engine.generate(
                        promptTokens: promptTokens, options: options, onToken: onToken)
                })
        }
        let engine = try await LLMEngine.load(bundle: bundle, verbose: verbose)
        return PersistentModel(
            bundle: bundle,
            loadSeconds: engine.loadSeconds,
            generate: { messages, options, tools, onToken in
                let promptTokens = try engine.encodePrompt(
                    messages: messages, tools: tools, applyChatTemplate: options.applyChatTemplate)
                return try await engine.generate(
                    promptTokens: promptTokens, options: options, onToken: onToken)
            })
        #else
        _ = (bundlePath, verbose)
        throw CoreAIPipeline.RuntimeError.runtimeUnavailable
        #endif
    }

    /// Apply the bundle's chat template to `messages` (or tokenize raw), then run prefill +
    /// decode, streaming decoded text deltas to `onToken`. When `tools` is non-nil it is passed
    /// to the tokenizer's chat template (function-calling prompt section), so the model is told
    /// about the callable functions in its own base format.
    @discardableResult
    public func generate(
        messages: [[String: String]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        #if COREAI_RUNTIME
        return try await generateImpl(messages, options, tools, onToken)
        #else
        _ = (messages, options, tools, onToken)
        throw CoreAIPipeline.RuntimeError.runtimeUnavailable
        #endif
    }

    #if COREAI_RUNTIME
    /// Recursive byte size of a bundle directory (best-effort; off the hot path).
    private static func directorySize(_ url: URL) -> UInt64 {
        let fm = FileManager.default
        guard
            let e = fm.enumerator(
                at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        else { return 0 }
        var total: UInt64 = 0
        for case let f as URL in e {
            let vals = try? f.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if vals?.isRegularFile == true, let s = vals?.fileSize { total += UInt64(s) }
        }
        return total
    }
    #endif
}
