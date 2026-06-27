import Foundation

/// A persistently-loaded classic speculative pair: a target LLM bundle plus a smaller draft
/// bundle under the package's `draft/` directory.
///
/// This is for standard-attention model pairs such as Qwen3-4B + Qwen3-0.6B. EAGLE/Gemma MTP uses
/// a different target/draft contract and remains handled by ``EagleEngine``.
public final class PersistentSpeculativeModel {
    public let name: String
    public let bundlePath: String
    public let draftBundlePath: String
    public let vocabSize: Int
    public let maxContextLength: Int
    public let bundleByteSize: UInt64
    public let loadSeconds: Double
    public let draftTokens: Int

    #if COREAI_RUNTIME
    private let engine: SpeculativeEngine

    private init(
        engine: SpeculativeEngine,
        targetBundle: ResolvedBundle,
        draftBundle: ResolvedBundle,
        draftTokens: Int
    ) {
        self.engine = engine
        self.name = targetBundle.name
        self.bundlePath = targetBundle.root.path
        self.draftBundlePath = draftBundle.root.path
        self.vocabSize = targetBundle.vocabSize
        self.maxContextLength = targetBundle.maxContextLength
        self.bundleByteSize =
            Self.directorySize(targetBundle.aimodelURL) + Self.directorySize(draftBundle.aimodelURL)
        self.loadSeconds = max(engine.target.loadSeconds, engine.draft.loadSeconds)
        self.draftTokens = max(1, draftTokens)
    }
    #else
    private init() {
        self.name = ""
        self.bundlePath = ""
        self.draftBundlePath = ""
        self.vocabSize = 0
        self.maxContextLength = 0
        self.bundleByteSize = 0
        self.loadSeconds = 0
        self.draftTokens = 0
    }
    #endif

    public static func load(
        bundlePath: String,
        draftSubdirectory: String = "draft",
        draftTokens: Int = 4,
        verbose: Bool = false
    ) async throws -> PersistentSpeculativeModel {
        #if COREAI_RUNTIME
        let root = URL(fileURLWithPath: bundlePath, isDirectory: true)
        let draftRoot = root.appendingPathComponent(draftSubdirectory, isDirectory: true)
        let targetBundle = try ResolvedBundle.load(at: root.path)
        let draftBundle = try ResolvedBundle.load(at: draftRoot.path)
        let engine = try await SpeculativeEngine.load(
            targetPath: root.path,
            draftPath: draftRoot.path,
            draftTokens: draftTokens,
            verbose: verbose)
        return PersistentSpeculativeModel(
            engine: engine,
            targetBundle: targetBundle,
            draftBundle: draftBundle,
            draftTokens: draftTokens)
        #else
        _ = (bundlePath, draftSubdirectory, draftTokens, verbose)
        throw CoreAIPipeline.RuntimeError.runtimeUnavailable
        #endif
    }

    @discardableResult
    public func generate(
        messages: [[String: String]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.SpeculativeResult {
        #if COREAI_RUNTIME
        let promptTokens = try engine.encodePrompt(
            messages: messages,
            tools: tools,
            applyChatTemplate: options.applyChatTemplate)
        return try await engine.generate(
            promptTokens: promptTokens,
            options: options,
            onToken: onToken)
        #else
        _ = (messages, options, tools, onToken)
        throw CoreAIPipeline.RuntimeError.runtimeUnavailable
        #endif
    }

    #if COREAI_RUNTIME
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
