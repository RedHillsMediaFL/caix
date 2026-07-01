import Foundation
import PipelineRuntime
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

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

public enum ModelSuitability: Sendable {
    public static func chatWarning(for name: String) -> String? {
        let lower = name.lowercased()
        if isComponent(lower) {
            return "component bundle; benchmark or pair it with its target, do not use as a standalone chat model"
        }
        if isChatTuned(lower) { return nil }
        if lower.contains("gemma") {
            return "this looks like a base Gemma target, not an -it/-instruct chat model"
        }
        return "this model name does not look chat-tuned; answers may be incoherent"
    }

    public static func isChatTuned(_ name: String) -> Bool {
        let lower = name.lowercased()
        return lower.contains("instruct")
            || lower.contains("-it-")
            || lower.hasSuffix("-it")
            || lower.contains("-it-coreai")
            || lower.contains("chat")
            || lower.contains("qwythos")
    }

    public static func inferredBillions(from name: String) -> Double? {
        let lower = name.lowercased()
        let scalars = Array(lower)
        var i = 0
        while i < scalars.count {
            if scalars[i].isNumber {
                var j = i
                while j < scalars.count, scalars[j].isNumber || scalars[j] == "." { j += 1 }
                if j < scalars.count, scalars[j] == "b" {
                    return Double(String(scalars[i..<j]))
                }
                i = j
            } else {
                i += 1
            }
        }
        return nil
    }

    public static func score(_ name: String, mode: String? = nil) -> Int {
        let lower = name.lowercased()
        if isComponent(lower) { return 900 }
        var score = 500
        if isChatTuned(lower) { score -= 300 }
        if lower.contains("qwen") { score -= 50 }
        if lower.contains("gemma") && !isChatTuned(lower) { score += 120 }
        if lower.contains("mtp") || mode == "eagle" { score += 80 }
        return score
    }

    private static func isComponent(_ lower: String) -> Bool {
        lower.contains("draft")
            || lower.contains("eagle-target")
            || lower.contains("eagle_draft")
            || (lower.contains("assistant") && !lower.contains("instruct") && !lower.contains("-it-"))
    }
}

public enum ModelNameRepair: Sendable {
    public static func preferredServedName(
        directoryName: String,
        metadataName: String?,
        sourceModelID: String?,
        tokenizer: String?
    ) -> String {
        let preferred = safeName(metadataName) ?? directoryName
        return preservingInstructionTuningMarker(
            in: preferred,
            fallback: directoryName,
            provenance: [sourceModelID, tokenizer])
    }

    public static func preferredInstallName(
        metadataName: String?,
        repoDerivedName: String?,
        fallbackName: String,
        sourceModelID: String?,
        tokenizer: String?
    ) -> String {
        let preferred = safeName(metadataName) ?? safeName(repoDerivedName) ?? fallbackName
        return preservingInstructionTuningMarker(
            in: preferred,
            fallback: safeName(repoDerivedName) ?? fallbackName,
            provenance: [sourceModelID, tokenizer])
    }

    private static func preservingInstructionTuningMarker(
        in name: String,
        fallback: String,
        provenance: [String?]
    ) -> String {
        guard !ModelSuitability.isChatTuned(name),
              provenance.contains(where: { hasInstructionTunedToken($0) })
        else { return name }
        if let repaired = insertingITMarker(in: name), safeName(repaired) != nil {
            return repaired
        }
        if !ModelSuitability.isChatTuned(fallback),
           let repaired = insertingITMarker(in: fallback),
           safeName(repaired) != nil {
            return repaired
        }
        return name
    }

    private static func hasInstructionTunedToken(_ raw: String?) -> Bool {
        guard let raw else { return false }
        return raw.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .contains("it")
    }

    private static func insertingITMarker(in raw: String) -> String? {
        let lower = raw.lowercased()
        if lower.hasSuffix("-coreai") {
            return String(raw.dropLast("-coreai".count)) + "-it-coreai"
        }
        if lower.hasSuffix("-caix") {
            return String(raw.dropLast("-caix".count)) + "-it-caix"
        }
        return raw + "-it"
    }

    private static func safeName(_ raw: String?) -> String? {
        guard let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        guard value.count <= 160 else { return nil }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        return value.unicodeScalars.allSatisfy { allowed.contains($0) } ? value : nil
    }
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
        case eagle(EagleEngine)  // EAGLE speculative decoding
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

    init(model: PersistentModel, name: String) {
        self.backend = .persistent(model)
        self.displayName = name
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
    private struct DiscoveredBundle: Sendable {
        var name: String
        var directoryName: String
        var mode: String
    }

    private let exportsDir: URL
    private let registryPath: URL
    private let verbose: Bool
    private let eagleConfig: EagleConfig?
    private let heavyTaskLockPath: URL

    private var handles: [String: ModelHandle] = [:]
    private var loadTasks: [String: Task<ModelHandle, Error>] = [:]
    private var bundleCache: (updatedAt: Date, entries: [DiscoveredBundle])?
    private let bundleDiscoveryCacheSeconds: TimeInterval = 5
    /// Memoized per-model output formats (detected from the bundle tokenizer/chat_template).
    private var formats: [String: OutputFormat] = [:]

    public init(exportsDir: URL, registryPath: URL, verbose: Bool = false,
                eagleConfig: EagleConfig? = nil, heavyTaskLockPath: URL? = nil) {
        self.exportsDir = exportsDir
        self.registryPath = registryPath
        self.verbose = verbose
        self.eagleConfig = eagleConfig
        self.heavyTaskLockPath = heavyTaskLockPath ?? Self.defaultHeavyTaskLockPath(exportsDir: exportsDir)
    }

    // MARK: Discovery

    /// Bundle directories under `exportsDir`. A directory is loadable if it is either a direct
    /// `metadata.json` LLM bundle or an EAGLE target+draft package.
    private func bundleEntries() -> [DiscoveredBundle] {
        let now = Date()
        if let cache = bundleCache,
           now.timeIntervalSince(cache.updatedAt) < bundleDiscoveryCacheSeconds {
            return cache.entries
        }
        let entries = Self.discoverBundleEntries(in: exportsDir)
        bundleCache = (updatedAt: now, entries: entries)
        return entries
    }

    private static func discoverBundleEntries(in exportsDir: URL) -> [DiscoveredBundle] {
        if let indexed = indexedBundleEntries(for: exportsDir) {
            return indexed
        }
        var entries: [DiscoveredBundle] = []
        for name in childNames(in: exportsDir) where !name.hasPrefix(".") {
            let url = exportsDir.appendingPathComponent(name, isDirectory: true)
            guard isDirectory(url), let mode = bundleMode(at: url) else {
                continue
            }
            let identity: (metadataName: String?, sourceModelID: String?, tokenizer: String?) =
                mode == "standard" ? bundleIdentity(at: url) : (nil, nil, nil)
            let servedName = mode == "standard"
                ? ModelNameRepair.preferredServedName(
                    directoryName: name,
                    metadataName: identity.metadataName,
                    sourceModelID: identity.sourceModelID,
                    tokenizer: identity.tokenizer)
                : name
            entries.append(
                DiscoveredBundle(
                    name: servedName,
                    directoryName: name,
                    mode: mode))
        }
        return entries.sorted { $0.name < $1.name }
    }

    /// Registry models (`models/registry.json`) → (key, params string), best-effort.
    private func registryModels() -> [(name: String, params: String)] {
        guard let data = Self.readSmallFile(registryPath, timeoutSeconds: 2),
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

        for bundle in bundleEntries() {
            let name = bundle.name
            if seen.contains(Self.normalize(name)) { continue }
            seen.insert(Self.normalize(name))
            let loaded = handles[name] != nil
            entries.append(
                ModelEntry(
                    name: name,
                    params: Self.inferParams(from: name),
                    status: loaded ? "loaded" : "available",
                    bundle: true,
                    memoryBytes: handles[name]?.memoryBytes,
                    mode: bundle.mode))
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

    public func servedModelsPreferredForChat() -> [ModelEntry] {
        listModels()
            .filter { $0.bundle }
            .sorted {
                let lhs = ModelSuitability.score($0.name, mode: $0.mode)
                let rhs = ModelSuitability.score($1.name, mode: $1.mode)
                if lhs == rhs { return $0.name < $1.name }
                return lhs < rhs
            }
    }

    // MARK: Load / offload / lookup

    public func isLoaded(_ name: String) -> Bool { handles[canonicalName(for: name)] != nil }

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
        let directoryName = resolvedBundle(for: name)?.directoryName ?? name
        return exportsDir.appendingPathComponent(directoryName).path
    }

    public func resolveServedModelName(_ requested: String) -> String? {
        resolvedBundle(for: requested)?.name
    }

    /// Return the hot handle for `name`, loading the bundle if necessary. Concurrent calls for
    /// the same name share a single in-flight load.
    func handle(for name: String) async throws -> ModelHandle {
        let name = canonicalName(for: name)
        if let h = handles[name] { return h }
        return try await load(name)
    }

    /// The normalized output format for `name`, detected (once) from its bundle's
    /// `tokenizer/` directory (chat_template + special tokens) and memoized. Models with no
    /// recognised reasoning/tool markers resolve to ``OutputFormat/passthrough``.
    func outputFormat(for name: String) -> OutputFormat {
        let name = canonicalName(for: name)
        if let f = formats[name] { return f }
        let tokenizerDir: URL
        if let cfg = eagleConfig, cfg.name == name {
            tokenizerDir = URL(fileURLWithPath: cfg.tokenizerDir)
        } else {
            let directoryName = resolvedBundle(for: name)?.directoryName ?? name
            tokenizerDir = exportsDir.appendingPathComponent(directoryName).appendingPathComponent("tokenizer")
        }
        let format = OutputFormat.detect(modelName: name, tokenizerDir: tokenizerDir)
        formats[name] = format
        return format
    }

    /// Load (or hot-swap to) the bundle `name`. Idempotent; de-duplicates concurrent loads.
    @discardableResult
    func load(_ name: String) async throws -> ModelHandle {
        let name = canonicalName(for: name)
        let verbose = self.verbose
        func log(_ message: @autoclosure () -> String) {
            if verbose {
                FileHandle.standardError.write(Data("[server] \(message())\n".utf8))
            }
        }
        if let h = handles[name] { return h }
        if let task = loadTasks[name] {
            log("joining in-flight load for \(name)")
            return try await task.value
        }

        let path = bundlePath(for: name)
        let eagle = eagleConfig
        log("load requested for \(name) at \(path)")
        if eagle?.name != name {
            var isDir = ObjCBool(false)
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
                  isDir.boolValue
            else {
                throw CoreAIPipeline.RuntimeError.bundleNotFound(path)
            }
        }
        let task = Task.detached(priority: .userInitiated) {
            if verbose {
                FileHandle.standardError.write(Data("[server] load task started for \(name)\n".utf8))
            }
            // EAGLE MTP model: build the speculative engine from its target+draft[+unrolled] bundles.
            if let cfg = eagle, cfg.name == name {
                #if COREAI_RUNTIME
                if verbose {
                    FileHandle.standardError.write(Data("[server] loading EAGLE bundle \(name)\n".utf8))
                }
                let engine = try await EagleEngine.load(
                    targetURL: URL(fileURLWithPath: cfg.targetPath),
                    draftURL: URL(fileURLWithPath: cfg.draftPath),
                    tokenizerDir: URL(fileURLWithPath: cfg.tokenizerDir),
                    draftTokens: 7, vocabSize: cfg.vocab, backbone: cfg.backbone,
                    slidingWindow: cfg.slidingWindow, maxContext: cfg.maxContext, verbose: verbose,
                    unrolledURL: cfg.unrolledPath.map { URL(fileURLWithPath: $0) })
                return ModelHandle(eagle: engine, name: cfg.name, bytes: cfg.bundleBytes)
                #else
                throw CoreAIPipeline.RuntimeError.runtimeUnavailable
                #endif
            }
            if Self.isEagleBundle(at: URL(fileURLWithPath: path, isDirectory: true)) {
                #if COREAI_RUNTIME
                if verbose {
                    FileHandle.standardError.write(Data("[server] loading EAGLE package \(name)\n".utf8))
                }
                let root = URL(fileURLWithPath: path, isDirectory: true)
                let engine = try await EagleEngine.load(
                    targetURL: root.appendingPathComponent("eagle_target.aimodel", isDirectory: true),
                    draftURL: root.appendingPathComponent("eagle_draft.aimodel", isDirectory: true),
                    tokenizerDir: root.appendingPathComponent("tokenizer", isDirectory: true),
                    draftTokens: 7, vocabSize: 262144, backbone: 2816,
                    slidingWindow: 1024, maxContext: 4096, verbose: verbose,
                    unrolledURL: Self.eagleUnrolledURL(in: root))
                return ModelHandle(eagle: engine, name: name, bytes: Self.dirSize(root))
                #else
                throw CoreAIPipeline.RuntimeError.runtimeUnavailable
                #endif
            }
            if Self.isClassicSpeculativeBundle(at: URL(fileURLWithPath: path, isDirectory: true)) {
                if verbose {
                    FileHandle.standardError.write(Data("[server] loading speculative bundle \(name)\n".utf8))
                }
                let model = try await PersistentSpeculativeModel.load(
                    bundlePath: path, draftTokens: 4, verbose: verbose)
                return ModelHandle(speculative: model, name: name)
            } else {
                if verbose {
                    FileHandle.standardError.write(Data("[server] loading persistent bundle \(name)\n".utf8))
                }
                let model = try await PersistentModel.load(bundlePath: path, verbose: verbose)
                return ModelHandle(model: model, name: name)
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
        handles.removeValue(forKey: canonicalName(for: name)) != nil
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
        if FileManager.default.fileExists(atPath: heavyTaskLockPath.path) {
            return "refusing to delete bundle while heavy-task lock exists: \(heavyTaskLockPath.path)"
        }
        let bundle = resolvedBundle(for: name)
        let servedName = bundle?.name ?? name
        let directoryName = bundle?.directoryName ?? name
        let dir = exportsDir.appendingPathComponent(directoryName)
        guard Self.isDirectory(dir),
              Self.isLoadableBundle(at: dir) else {
            return "no bundle named '\(name)' under exports"
        }
        handles.removeValue(forKey: servedName)
        formats.removeValue(forKey: servedName)
        do {
            try FileManager.default.removeItem(at: dir)
        } catch {
            return "delete failed: \(error.localizedDescription)"
        }
        bundleCache = nil
        return nil
    }

    // MARK: Helpers

    private func resolvedBundle(for name: String) -> DiscoveredBundle? {
        bundleEntries().first { $0.name == name || $0.directoryName == name }
    }

    private func canonicalName(for name: String) -> String {
        resolvedBundle(for: name)?.name ?? name
    }

    private static func defaultHeavyTaskLockPath(exportsDir: URL) -> URL {
        let env = ProcessInfo.processInfo.environment
        if let raw = caixEnv(env, "caix_heavy_task_lock", legacy: "HEAVY_TASK_LOCK"),
           !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: raw)
        }
        let normalized = exportsDir.standardizedFileURL
        if normalized.lastPathComponent == "exports",
           normalized.deletingLastPathComponent().lastPathComponent == "models" {
            return normalized
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(".agent-heavy-task.lock")
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(".agent-heavy-task.lock")
            .standardizedFileURL
    }

    private static func caixEnv(_ env: [String: String], _ name: String, legacy suffix: String) -> String? {
        env[name] ?? env["C" + "AIX_" + suffix]
    }

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

    static func isDirectLLMBundle(at root: URL) -> Bool {
        let meta = root.appendingPathComponent("metadata.json")
        guard
            fileExists(meta),
            let data = try? Data(contentsOf: meta, options: [.mappedIfSafe]),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return (obj["kind"] as? String) == "llm"
    }

    static func isLoadableBundle(at root: URL) -> Bool {
        isDirectLLMBundle(at: root) || isEagleBundle(at: root)
    }

    static func bundleMode(at root: URL) -> String? {
        if isEagleBundle(at: root) { return "eagle" }
        if isClassicSpeculativeBundle(at: root) { return "speculative" }
        if isDirectLLMBundle(at: root) { return "standard" }
        return nil
    }

    private static func indexedBundleEntries(for exportsDir: URL) -> [DiscoveredBundle]? {
        let env = ProcessInfo.processInfo.environment
        let path = caixEnv(env, "caix_export_index", legacy: "EXPORT_INDEX")?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let path, !path.isEmpty else { return nil }
        guard let data = readSmallFile(URL(fileURLWithPath: path), timeoutSeconds: 0.5),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        if let indexedExports = object["exportsDir"] as? String, indexedExports != exportsDir.path {
            return nil
        }
        guard let bundles = object["bundles"] as? [[String: Any]] else { return nil }
        let entries = bundles.compactMap { item -> DiscoveredBundle? in
            guard let name = item["name"] as? String, !name.isEmpty else { return nil }
            let directoryName = item["directoryName"] as? String
                ?? item["directory_name"] as? String
                ?? name
            let mode = item["mode"] as? String ?? "standard"
            return DiscoveredBundle(name: name, directoryName: directoryName, mode: mode)
        }
        return entries.sorted { $0.name < $1.name }
    }

    private static func bundleIdentity(at root: URL) -> (
        metadataName: String?, sourceModelID: String?, tokenizer: String?
    ) {
        let meta = root.appendingPathComponent("metadata.json")
        guard
            let data = readSmallFile(meta, timeoutSeconds: 0.5),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return (nil, nil, nil) }
        let source = object["source"] as? [String: Any]
        let language = object["language"] as? [String: Any]
        return (
            object["name"] as? String,
            source?["hf_model_id"] as? String,
            language?["tokenizer"] as? String
        )
    }

    private static func childNames(in directory: URL) -> [String] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/find")
        process.arguments = [
            directory.path, "-maxdepth", "1", "-mindepth", "1", "-type", "d", "-print",
        ]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return []
        }
        let deadline = Date().addingTimeInterval(2)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.02)
        }
        guard !process.isRunning else {
            process.terminate()
            process.waitUntilExit()
            return []
        }
        guard process.terminationStatus == 0 else { return [] }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").map { path in
            URL(fileURLWithPath: String(path), isDirectory: true).lastPathComponent
        }
    }

    private static func readSmallFile(_ url: URL, timeoutSeconds: TimeInterval) -> Data? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/cat")
        process.arguments = [url.path]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return nil
        }
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.02)
        }
        guard !process.isRunning else {
            process.terminate()
            process.waitUntilExit()
            return nil
        }
        guard process.terminationStatus == 0 else { return nil }
        return output.fileHandleForReading.readDataToEndOfFile()
    }

    private static func fileExists(_ url: URL) -> Bool {
        var st = stat()
        return lstat(url.path, &st) == 0
    }

    private static func isDirectory(_ url: URL) -> Bool {
        var st = stat()
        guard lstat(url.path, &st) == 0 else { return false }
        return (st.st_mode & S_IFMT) == S_IFDIR
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

    func isEagleBundle(_ name: String) -> Bool {
        Self.isEagleBundle(at: exportsDir.appendingPathComponent(name, isDirectory: true))
    }

    static func isEagleBundle(at root: URL) -> Bool {
        func dirExists(_ url: URL) -> Bool {
            isDirectory(url)
        }
        let target = root.appendingPathComponent("eagle_target.aimodel", isDirectory: true)
        let draft = root.appendingPathComponent("eagle_draft.aimodel", isDirectory: true)
        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        return dirExists(target) && dirExists(draft) && dirExists(tokenizer)
    }

    static func eagleUnrolledURL(in root: URL) -> URL? {
        let fm = FileManager.default
        for name in [
            "eagle_draft_unrolled_k7.aimodel",
            "eagle_draft_unrolled_k6.aimodel",
            "eagle_draft_unrolled_k5.aimodel",
            "eagle_draft_unrolled_k4.aimodel",
            "eagle_draft_unrolled.aimodel",
        ] {
            let url = root.appendingPathComponent(name, isDirectory: true)
            var isDir = ObjCBool(false)
            if fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                return url
            }
        }
        return nil
    }

    static func dirSize(_ root: URL) -> UInt64 {
        let fm = FileManager.default
        guard let en = fm.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }
        var total: UInt64 = 0
        for case let url as URL in en {
            total += UInt64((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        return total
    }
}
