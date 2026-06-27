import Foundation
import PipelineRuntime

// MARK: - Dashboard / API DTOs

/// One row of `GET /api/models` (`[{name, params, status, bundle}]`). `memoryBytes` is an
/// additive field carrying the resident footprint of loaded models (ignored by the dashboard).
public struct ModelEntry: Codable, Sendable {
    public var name: String
    public var params: String
    public var status: String  // "loaded" | "available"
    public var bundle: Bool
    public var memoryBytes: UInt64?
    public var mode: String?
}

// MARK: - Per-model hot handle

/// A loaded model plus the gate that serialises generation against it.
///
/// `@unchecked Sendable`: it wraps a non-`Sendable` `PersistentModel`, but every access to the
/// engine goes through ``generate(messages:options:onToken:)``, which holds the single-permit
/// `AsyncSemaphore` for the duration — so the engine is only ever driven by one task at a time.
final class ModelHandle: @unchecked Sendable {
    enum Backend {
        case persistent(PersistentModel)
        case speculative(PersistentSpeculativeModel)
        #if COREAI_RUNTIME
        case eagle(EagleEngine)  // fastest gemma MTP (EAGLE speculative decoding)
        #endif
    }
    let backend: Backend
    private let displayName: String
    private let bytes: UInt64
    private let gate = AsyncSemaphore(permits: 1)

    init(model: PersistentModel) {
        self.backend = .persistent(model)
        self.displayName = model.name
        self.bytes = model.bundleByteSize
    }

    init(speculative: PersistentSpeculativeModel, name: String) {
        self.backend = .speculative(speculative)
        self.displayName = name
        self.bytes = speculative.bundleByteSize
    }

    #if COREAI_RUNTIME
    init(eagle: EagleEngine, name: String, bytes: UInt64) {
        self.backend = .eagle(eagle)
        self.displayName = name
        self.bytes = bytes
    }
    #endif

    var name: String { displayName }
    var memoryBytes: UInt64 { bytes }
    var eagleBackbone: Int? {
        #if COREAI_RUNTIME
        if case .eagle(let engine) = backend { return engine.backbone }
        #endif
        return nil
    }

    /// Serialised generation. The engine call runs in the caller's context; the gate only
    /// guarantees mutual exclusion per model. `tools` (when present) is threaded into the chat
    /// template so the model sees the callable functions (EAGLE path ignores tools).
    func generate(
        messages: [[String: String]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        await gate.acquire()
        do {
            let result: CoreAIPipeline.Result
            switch backend {
            case .persistent(let model):
                result = try await model.generate(
                    messages: messages, options: options, tools: tools, onToken: onToken)
            case .speculative(let model):
                let r = try await model.generate(
                    messages: messages, options: options, tools: tools, onToken: onToken)
                LiveStats.record(SpeculativeStats(
                    model: displayName, tokensPerSecond: r.decodeTokensPerSecond,
                    acceptanceRate: r.acceptanceRate, tokensPerPass: r.tokensPerTargetForward,
                    draftTokens: r.draftTokens, generatedTokens: r.generatedTokenCount,
                    promptTokens: r.promptTokenCount, decodeSeconds: r.decodeSeconds,
                    prefillSeconds: r.prefillSeconds, at: Date().timeIntervalSince1970))
                result = CoreAIPipeline.Result(
                    text: r.text, promptTokenCount: r.promptTokenCount,
                    generatedTokenCount: r.generatedTokenCount, stopReason: r.stopReason,
                    modelLoadSeconds: r.modelLoadSeconds, prefillSeconds: r.prefillSeconds,
                    decodeSeconds: r.decodeSeconds)
            #if COREAI_RUNTIME
            case .eagle(let engine):
                let r = try await engine.generate(
                    messages: messages, options: options, tools: tools, onToken: onToken)
                // Publish live speculative metrics for the dashboard, tagged with the served name.
                LiveStats.record(SpeculativeStats(
                    model: displayName, tokensPerSecond: r.decodeTokensPerSecond,
                    acceptanceRate: r.acceptanceRate, tokensPerPass: r.tokensPerTargetForward,
                    draftTokens: r.draftTokens, generatedTokens: r.generatedTokenCount,
                    promptTokens: r.promptTokenCount, decodeSeconds: r.decodeSeconds,
                    prefillSeconds: r.prefillSeconds, at: Date().timeIntervalSince1970))
                result = CoreAIPipeline.Result(
                    text: r.text, promptTokenCount: r.promptTokenCount,
                    generatedTokenCount: r.generatedTokenCount, stopReason: r.stopReason,
                    modelLoadSeconds: r.modelLoadSeconds, prefillSeconds: r.prefillSeconds,
                    decodeSeconds: r.decodeSeconds)
            #endif
            }
            await gate.release()
            Usage.record(model: displayName, inputTokens: result.promptTokenCount,
                         outputTokens: result.generatedTokenCount, decodeSeconds: result.decodeSeconds,
                         at: Date().timeIntervalSince1970)
            return result
        } catch {
            await gate.release()
            throw error
        }
    }
}

/// Configures the EAGLE MTP model the server serves (target + draft [+ unrolled draft] bundles).
public struct EagleConfig: Sendable {
    public let name: String
    public let targetPath: String
    public let draftPath: String
    public let unrolledPath: String?
    public let tokenizerDir: String
    public let vocab: Int
    public let backbone: Int
    public let slidingWindow: Int
    public let maxContext: Int

    public init(
        name: String, targetPath: String, draftPath: String, unrolledPath: String?,
        tokenizerDir: String, vocab: Int = 262144, backbone: Int = 2816,
        slidingWindow: Int = 1024, maxContext: Int = 4096
    ) {
        self.name = name
        self.targetPath = targetPath
        self.draftPath = draftPath
        self.unrolledPath = unrolledPath
        self.tokenizerDir = tokenizerDir
        self.vocab = vocab
        self.backbone = backbone
        self.slidingWindow = slidingWindow
        self.maxContext = maxContext
    }

    var bundleBytes: UInt64 {
        let fm = FileManager.default
        func dirSize(_ p: String) -> UInt64 {
            guard let en = fm.enumerator(at: URL(fileURLWithPath: p), includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
            var total: UInt64 = 0
            for case let u as URL in en {
                total += UInt64((try? u.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
            }
            return total
        }
        return dirSize(targetPath) + dirSize(draftPath) + (unrolledPath.map(dirSize) ?? 0)
    }
}

// MARK: - ModelManager

/// Owns the set of resident models. Discovers exportable bundles under `exportsDir` and
/// convertible entries from `registry.json`, loads/offloads `.aimodel` bundles by name, hot-
/// swaps them in a concurrent-safe registry, and tracks aggregate resident memory.
///
/// An `actor`, so its mutable registry is race-free; heavy `load`s run during an `await`
/// (actor reentrancy keeps the manager responsive to `listModels`/`isLoaded` meanwhile), and
/// concurrent loads of the same name are de-duplicated to a single in-flight `Task`.
public actor ModelManager {
    private let exportsDir: URL
    private let registryPath: URL
    private let verbose: Bool
    private let eagleConfig: EagleConfig?

    private var handles: [String: ModelHandle] = [:]
    private var loadTasks: [String: Task<ModelHandle, Error>] = [:]
    /// Memoized per-model output formats (detected from the bundle tokenizer/chat_template).
    private var formats: [String: OutputFormat] = [:]

    public init(exportsDir: URL, registryPath: URL, verbose: Bool = false,
                eagleConfig: EagleConfig? = nil) {
        self.exportsDir = exportsDir
        self.registryPath = registryPath
        self.verbose = verbose
        self.eagleConfig = eagleConfig
    }

    // MARK: Discovery

    /// Bundle directories under `exportsDir` (a dir is a bundle iff it has a `metadata.json`
    /// whose `kind` is `llm`). Keyed by directory name — the identifier `/api/load` expects.
    private func bundleNames() -> [String] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: exportsDir, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        var names: [String] = []
        for url in entries {
            let meta = url.appendingPathComponent("metadata.json")
            guard fm.fileExists(atPath: meta.path),
                let data = try? Data(contentsOf: meta),
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                (obj["kind"] as? String) == "llm"
            else { continue }
            names.append(url.lastPathComponent)
        }
        return names.sorted()
    }

    /// Registry models (`models/registry.json`) → (key, params string), best-effort.
    private func registryModels() -> [(name: String, params: String)] {
        guard let data = try? Data(contentsOf: registryPath),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let models = obj["models"] as? [String: Any]
        else { return [] }
        var out: [(String, String)] = []
        for (key, value) in models {
            var params = Self.inferParams(from: key)
            if let dict = value as? [String: Any], let pb = dict["params_b"] as? Double {
                params = Self.formatBillions(pb)
            }
            out.append((key, params))
        }
        return out.sorted { $0.0 < $1.0 }
    }

    /// Snapshot for `GET /api/models`: every export bundle, then registry models that don't
    /// already have a bundle, each tagged loaded/available.
    public func listModels() -> [ModelEntry] {
        var entries: [ModelEntry] = []
        var seen = Set<String>()

        // The EAGLE MTP model leads the list so it is the server's default (resolveModelName
        // falls back to bundles.first).
        if let cfg = eagleConfig {
            seen.insert(Self.normalize(cfg.name))
            let loaded = handles[cfg.name] != nil
            entries.append(
                ModelEntry(
                    name: cfg.name, params: Self.inferParams(from: cfg.name),
                    status: loaded ? "loaded" : "available", bundle: true,
                    memoryBytes: handles[cfg.name]?.memoryBytes, mode: "eagle"))
        }

        for name in bundleNames() {
            if seen.contains(Self.normalize(name)) { continue }
            seen.insert(Self.normalize(name))
            let loaded = handles[name] != nil
            let mode = isClassicSpeculativeBundle(name) ? "speculative" : "standard"
            entries.append(
                ModelEntry(
                    name: name,
                    params: Self.inferParams(from: name),
                    status: loaded ? "loaded" : "available",
                    bundle: true,
                    memoryBytes: handles[name]?.memoryBytes,
                    mode: mode))
        }

        for (key, params) in registryModels() {
            // Skip registry entries already represented by an exported bundle.
            if seen.contains(Self.normalize(key)) || seen.contains(Self.normalize(key + "-coreai")) {
                continue
            }
            entries.append(
                ModelEntry(
                    name: key, params: params, status: "available", bundle: false,
                    memoryBytes: nil, mode: "registry"))
        }
        return entries
    }

    // MARK: Load / offload / lookup

    public func isLoaded(_ name: String) -> Bool { handles[name] != nil }

    public func loadedNames() -> [String] { Array(handles.keys).sorted() }

    /// Aggregate resident footprint across loaded models.
    public func loadedMemoryBytes() -> UInt64 {
        handles.values.reduce(0) { $0 + $1.memoryBytes }
    }

    func eagleSummary() -> ServerInfo.Eagle {
        guard let cfg = eagleConfig else {
            return ServerInfo.Eagle(
                enabled: false, name: nil, targetPath: nil, draftPath: nil,
                unrolledPath: nil, tokenizerDir: nil, vocab: nil, backbone: nil,
                slidingWindow: nil, maxContext: nil)
        }
        return ServerInfo.Eagle(
            enabled: true, name: cfg.name, targetPath: cfg.targetPath,
            draftPath: cfg.draftPath, unrolledPath: cfg.unrolledPath,
            tokenizerDir: cfg.tokenizerDir, vocab: cfg.vocab,
            backbone: handles[cfg.name]?.eagleBackbone ?? cfg.backbone,
            slidingWindow: cfg.slidingWindow, maxContext: cfg.maxContext)
    }

    /// Resolve a bundle directory name to its path under `exportsDir`.
    private func bundlePath(for name: String) -> String {
        exportsDir.appendingPathComponent(name).path
    }

    /// Return the hot handle for `name`, loading the bundle if necessary. Concurrent calls for
    /// the same name share a single in-flight load.
    func handle(for name: String) async throws -> ModelHandle {
        if let h = handles[name] { return h }
        return try await load(name)
    }

    /// The normalized output format for `name`, detected (once) from its bundle's
    /// `tokenizer/` directory (chat_template + special tokens) and memoized. Models with no
    /// recognised reasoning/tool markers resolve to ``OutputFormat/passthrough``.
    func outputFormat(for name: String) -> OutputFormat {
        if let f = formats[name] { return f }
        let tokenizerDir: URL
        if let cfg = eagleConfig, cfg.name == name {
            tokenizerDir = URL(fileURLWithPath: cfg.tokenizerDir)
        } else {
            tokenizerDir = exportsDir.appendingPathComponent(name).appendingPathComponent("tokenizer")
        }
        let format = OutputFormat.detect(modelName: name, tokenizerDir: tokenizerDir)
        formats[name] = format
        return format
    }

    /// Load (or hot-swap to) the bundle `name`. Idempotent; de-duplicates concurrent loads.
    @discardableResult
    func load(_ name: String) async throws -> ModelHandle {
        if let h = handles[name] { return h }
        if let task = loadTasks[name] { return try await task.value }

        let path = bundlePath(for: name)
        let verbose = self.verbose
        let eagle = eagleConfig
        if eagle?.name != name {
            var isDir = ObjCBool(false)
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
                  isDir.boolValue
            else {
                throw CoreAIPipeline.RuntimeError.bundleNotFound(path)
            }
        }
        let task = Task<ModelHandle, Error> {
            // EAGLE MTP model: build the speculative engine from its target+draft[+unrolled] bundles.
            if let cfg = eagle, cfg.name == name {
                #if COREAI_RUNTIME
                let engine = try await EagleEngine.load(
                    targetURL: URL(fileURLWithPath: cfg.targetPath),
                    draftURL: URL(fileURLWithPath: cfg.draftPath),
                    tokenizerDir: URL(fileURLWithPath: cfg.tokenizerDir),
                    draftTokens: 4, vocabSize: cfg.vocab, backbone: cfg.backbone,
                    slidingWindow: cfg.slidingWindow, maxContext: cfg.maxContext, verbose: verbose,
                    unrolledURL: cfg.unrolledPath.map { URL(fileURLWithPath: $0) })
                return ModelHandle(eagle: engine, name: cfg.name, bytes: cfg.bundleBytes)
                #else
                throw CoreAIPipeline.RuntimeError.runtimeUnavailable
                #endif
            }
            if Self.isClassicSpeculativeBundle(at: URL(fileURLWithPath: path, isDirectory: true)) {
                let model = try await PersistentSpeculativeModel.load(
                    bundlePath: path, draftTokens: 4, verbose: verbose)
                return ModelHandle(speculative: model, name: name)
            } else {
                let model = try await PersistentModel.load(bundlePath: path, verbose: verbose)
                return ModelHandle(model: model)
            }
        }
        loadTasks[name] = task
        defer { loadTasks[name] = nil }
        do {
            let handle = try await task.value
            handles[name] = handle
            return handle
        } catch {
            throw error
        }
    }

    /// Offload a resident model. Returns `true` if it was loaded.
    @discardableResult
    public func offload(_ name: String) -> Bool {
        handles.removeValue(forKey: name) != nil
    }

    /// Offload every resident model. Returns the model names that were unloaded.
    public func offloadAll() -> [String] {
        let names = handles.keys.sorted()
        handles.removeAll(keepingCapacity: true)
        return names
    }

    /// Permanently delete a converted bundle from disk (offloading it first). Refuses to delete the
    /// configured EAGLE/MTP model's bundle. Returns nil on success or an error string.
    public func deleteBundle(_ name: String) -> String? {
        if let cfg = eagleConfig, cfg.name == name {
            return "refusing to delete the built-in MTP model"
        }
        let dir = exportsDir.appendingPathComponent(name)
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue,
              bundleNames().contains(name) else {
            return "no bundle named '\(name)' under exports"
        }
        handles.removeValue(forKey: name)
        formats.removeValue(forKey: name)
        do { try fm.removeItem(at: dir) } catch { return "delete failed: \(error.localizedDescription)" }
        return nil
    }

    // MARK: Helpers

    /// "qwen3-0.6b-coreai" → "0.6B", "gemma4-31b-assistant-coreai" → "31B".
    static func inferParams(from name: String) -> String {
        let lower = name.lowercased()
        let scalars = Array(lower)
        var i = 0
        while i < scalars.count {
            if scalars[i].isNumber {
                var j = i
                while j < scalars.count, scalars[j].isNumber || scalars[j] == "." { j += 1 }
                if j < scalars.count, scalars[j] == "b" {
                    let num = String(scalars[i..<j])
                    // Avoid matching version-y tokens like "0.6" only when followed by 'b'.
                    return num.uppercased() + "B"
                }
                i = j
            } else {
                i += 1
            }
        }
        return "—"
    }

    static func formatBillions(_ b: Double) -> String {
        if b < 1 { return String(format: "%.2gB", b) }
        return (b.rounded() == b ? String(Int(b)) : String(format: "%.1f", b)) + "B"
    }

    static func normalize(_ s: String) -> String {
        String(s.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) })
    }

    func isClassicSpeculativeBundle(_ name: String) -> Bool {
        Self.isClassicSpeculativeBundle(at: exportsDir.appendingPathComponent(name, isDirectory: true))
    }

    static func isClassicSpeculativeBundle(at root: URL) -> Bool {
        let fm = FileManager.default
        let draftMeta = root.appendingPathComponent("draft", isDirectory: true)
            .appendingPathComponent("metadata.json")
        guard fm.fileExists(atPath: draftMeta.path) else { return false }
        guard
            let data = try? Data(contentsOf: draftMeta),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return (obj["kind"] as? String ?? "llm") == "llm"
    }
}
