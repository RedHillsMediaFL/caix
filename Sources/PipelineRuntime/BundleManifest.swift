import Foundation

/// `metadata.json` (schema `metadata_version` 0.2, `kind` "llm") at the root of an exported
/// Core AI model bundle. Only the fields the runtime needs are decoded.
public struct BundleManifest: Codable, Sendable {
    /// `assets` maps a component name to its `.aimodel` package path. LLM bundles expose a
    /// single `main`; diffusion bundles key the asset by component name
    /// (e.g. `denoiser` → `BidirectionalDenoiser.aimodel`). Decoded as an open dictionary so
    /// the runtime tolerates either layout.
    public struct Assets: Codable, Sendable {
        public let byName: [String: String]

        public init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            self.byName = try c.decode([String: String].self)
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            try c.encode(byName)
        }

        /// The primary model asset path: `main` (LLM) if present, else a known diffusion
        /// component, else the sole/first entry. `nil` only when `assets` is empty.
        public var primary: String? {
            if let m = byName["main"] { return m }
            for k in ["denoiser", "transformer", "model"] where byName[k] != nil { return byName[k] }
            if byName.count == 1 { return byName.values.first }
            return byName.keys.sorted().first.flatMap { byName[$0] }
        }
    }

    public struct Language: Codable, Sendable {
        public let tokenizer: String
        public let vocabSize: Int
        public let maxContextLength: Int
        public let embeddedTokenizer: Bool
        /// Optional per-model KV-cache floor baked into the bundle (overrides registry lookup).
        public let minKVCapacity: Int?

        enum CodingKeys: String, CodingKey {
            case tokenizer
            case vocabSize = "vocab_size"
            case maxContextLength = "max_context_length"
            case embeddedTokenizer = "embedded_tokenizer"
            case minKVCapacity = "min_kv_capacity"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.tokenizer = try c.decode(String.self, forKey: .tokenizer)
            self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
            self.maxContextLength = try c.decode(Int.self, forKey: .maxContextLength)
            self.embeddedTokenizer =
                try c.decodeIfPresent(Bool.self, forKey: .embeddedTokenizer) ?? true
            self.minKVCapacity = try c.decodeIfPresent(Int.self, forKey: .minKVCapacity)
        }
    }

    /// `source` block of `metadata.json` (provenance). Used to map an exported bundle back to its
    /// `models/registry.json` entry (by `hf_model_id` == registry `hf_repo`).
    public struct Source: Codable, Sendable {
        public let hfModelId: String?

        enum CodingKeys: String, CodingKey {
            case hfModelId = "hf_model_id"
        }
    }

    public let metadataVersion: String
    public let kind: String
    public let name: String
    public let assets: Assets
    /// Tokenizer/vocab block. Always present for `llm` bundles; diffusion bundles may omit it
    /// (vocab is then read from the `diffusion` block), so it's optional.
    public let language: Language?
    public let source: Source?

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.metadataVersion = try c.decode(String.self, forKey: .metadataVersion)
        self.kind = try c.decode(String.self, forKey: .kind)
        self.name = try c.decode(String.self, forKey: .name)
        self.assets = try c.decode(Assets.self, forKey: .assets)
        self.language = try c.decodeIfPresent(Language.self, forKey: .language)
        self.source = try c.decodeIfPresent(Source.self, forKey: .source)
    }

    enum CodingKeys: String, CodingKey {
        case metadataVersion = "metadata_version"
        case kind
        case name
        case assets
        case language
        case source
    }
}

/// A model bundle resolved to concrete on-disk locations.
public struct ResolvedBundle: Sendable {
    public let root: URL
    /// The `.aimodel` package directory (the `AIProgram` Apple loads).
    public let aimodelURL: URL
    /// The HuggingFace tokenizer directory (`tokenizer/` containing `tokenizer.json`).
    public let tokenizerDir: URL
    public let manifest: BundleManifest
    /// Minimum KV-cache capacity (tokens) this model requires, 0 when unconstrained. Hybrid
    /// `qwen3_5` models need >= `ssm_pos` positions to hold their packed recurrent state.
    /// Resolved from the bundle metadata (`language.min_kv_capacity`) or, failing that, the
    /// `models/registry.json` entry matching the bundle's `source.hf_model_id`.
    public let minKVCapacity: Int
    /// Block-diffusion schedule, non-nil when this is a diffusion bundle (`kind == "diffusion"`
    /// or a `diffusion` metadata block is present). Drives ``DiffusionEngine`` instead of
    /// ``LLMEngine``.
    public let diffusion: DiffusionSchedule?

    /// Resolved at load from `language` (LLM bundles) or the `diffusion` block (diffusion
    /// bundles, which may omit `language`).
    public let vocabSize: Int
    public let maxContextLength: Int
    public var name: String { manifest.name }
    /// True when this bundle should route to the host-side diffusion denoise loop.
    public var isDiffusion: Bool { diffusion != nil }

    /// Parse `metadata.json` under `path` and resolve the `.aimodel` + `tokenizer/` paths.
    public static func load(at path: String) throws -> ResolvedBundle {
        let expanded = (path as NSString).expandingTildeInPath
        let root = URL(fileURLWithPath: expanded, isDirectory: true)
        let fm = FileManager.default

        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: root.path, isDirectory: &isDir), isDir.boolValue else {
            throw CoreAIPipeline.RuntimeError.bundleNotFound(root.path)
        }

        let metaURL = root.appendingPathComponent("metadata.json")
        guard fm.fileExists(atPath: metaURL.path) else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "missing metadata.json in \(root.path)")
        }

        let manifest: BundleManifest
        do {
            let data = try Data(contentsOf: metaURL)
            manifest = try JSONDecoder().decode(BundleManifest.self, from: data)
        } catch {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "could not parse metadata.json: \(error)")
        }

        // Diffusion bundles carry `kind == "diffusion"` and/or a `diffusion` metadata block;
        // they still expose `language` (tokenizer/vocab) and an `assets.main` package.
        let diffusion = Self.parseDiffusionSchedule(metaURL: metaURL, manifest: manifest)
        guard manifest.kind == "llm" || diffusion != nil else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "unsupported bundle kind '\(manifest.kind)' (expected 'llm' or 'diffusion')")
        }

        guard let assetRel = manifest.assets.primary else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("metadata.json `assets` is empty")
        }
        let aimodelURL = root.appendingPathComponent(assetRel)
        guard fm.fileExists(atPath: aimodelURL.path) else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("missing model asset \(assetRel)")
        }

        let tokenizerDir = root.appendingPathComponent("tokenizer")
        guard fm.fileExists(atPath: tokenizerDir.appendingPathComponent("tokenizer.json").path)
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "missing tokenizer/tokenizer.json")
        }

        let (vocab, context) = Self.resolveVocabContext(metaURL: metaURL, manifest: manifest)
        return ResolvedBundle(
            root: root,
            aimodelURL: aimodelURL,
            tokenizerDir: tokenizerDir,
            manifest: manifest,
            minKVCapacity: Self.resolveMinKVCapacity(root: root, manifest: manifest),
            diffusion: diffusion,
            vocabSize: vocab,
            maxContextLength: context)
    }

    /// Resolve `(vocabSize, maxContextLength)` from the `language` block when present (LLM
    /// bundles), otherwise from the `diffusion` block (which carries `vocab_size`; context falls
    /// back to a generous default since the diffusion runtime bounds generation by `maxTokens`).
    static func resolveVocabContext(metaURL: URL, manifest: BundleManifest) -> (Int, Int) {
        if let lang = manifest.language {
            return (lang.vocabSize, lang.maxContextLength)
        }
        guard let data = try? Data(contentsOf: metaURL),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let block = obj["diffusion"] as? [String: Any]
        else { return (0, 4096) }
        func intVal(_ key: String) -> Int? {
            if let v = block[key] as? Int { return v }
            if let v = block[key] as? Double { return Int(v) }
            return nil
        }
        let vocab = intVal("vocab_size") ?? 0
        let context = intVal("max_context_length") ?? intVal("prompt_length").map { $0 * 16 } ?? 4096
        return (vocab, context)
    }

    // MARK: - Diffusion schedule resolution

    /// True when `metadata.json` at `path` describes a diffusion bundle (cheap, no-throw probe
    /// used by the CLI to route `run` to ``DiffusionEngine``). Detects `kind == "diffusion"` or
    /// the presence of a `diffusion` block.
    public static func isDiffusionBundle(at path: String) -> Bool {
        let expanded = (path as NSString).expandingTildeInPath
        let metaURL = URL(fileURLWithPath: expanded, isDirectory: true)
            .appendingPathComponent("metadata.json")
        guard let data = try? Data(contentsOf: metaURL),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        if (obj["kind"] as? String) == "diffusion" { return true }
        return obj["diffusion"] is [String: Any]
    }

    /// Parse the block-diffusion schedule from `metadata.json`, merging the `diffusion` block,
    /// top-level fields, and the documented `diffusiongemma` defaults. Returns `nil` for a
    /// non-diffusion (`kind == "llm"`, no `diffusion` block) bundle.
    static func parseDiffusionSchedule(metaURL: URL, manifest: BundleManifest) -> DiffusionSchedule? {
        guard let data = try? Data(contentsOf: metaURL),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return manifest.kind == "diffusion" ? DiffusionSchedule() : nil }

        let block = obj["diffusion"] as? [String: Any]
        let isDiffusion = (obj["kind"] as? String) == "diffusion" || block != nil
        guard isDiffusion else { return nil }

        let d = block ?? [:]
        func num(_ keys: [String]) -> Double? {
            for k in keys {
                if let v = d[k] as? Double { return v }
                if let v = d[k] as? Int { return Double(v) }
                if let v = obj[k] as? Double { return v }
                if let v = obj[k] as? Int { return Double(v) }
            }
            return nil
        }
        let defaults = DiffusionSchedule()
        return DiffusionSchedule(
            maxDenoisingSteps: num(["max_denoising_steps", "num_steps", "steps"]).map(Int.init)
                ?? defaults.maxDenoisingSteps,
            tMax: num(["t_max", "tmax"]) ?? defaults.tMax,
            tMin: num(["t_min", "tmin"]) ?? defaults.tMin,
            entropyBound: num(["entropy_bound"]) ?? defaults.entropyBound,
            confidenceThreshold: num(["confidence_threshold"]) ?? defaults.confidenceThreshold,
            stabilityThreshold: num(["stability_threshold"]).map(Int.init)
                ?? defaults.stabilityThreshold,
            canvasLength: num(["canvas_length", "canvas_len"]).map(Int.init) ?? defaults.canvasLength,
            promptLength: num(["prompt_length", "prompt_len"]).map(Int.init) ?? defaults.promptLength)
    }

    // MARK: - KV-cache floor resolution

    /// Determine the per-model KV-cache floor for `manifest`:
    /// 1. an explicit `language.min_kv_capacity` baked into the bundle, else
    /// 2. the `min_kv_capacity` of the `models/registry.json` entry whose `hf_repo` matches the
    ///    bundle's `source.hf_model_id`, else
    /// 3. a built-in default for hybrid `qwen3_5` registry entries that predate the field,
    ///    else 0 (no floor — standard attention models).
    static func resolveMinKVCapacity(root: URL, manifest: BundleManifest) -> Int {
        if let explicit = manifest.language?.minKVCapacity, explicit > 0 { return explicit }
        if let hfId = manifest.source?.hfModelId,
            let entry = registryEntry(forHFRepo: hfId, near: root)
        {
            if let m = entry["min_kv_capacity"] as? Int, m > 0 { return m }
            if let m = entry["min_kv_capacity"] as? Double, m > 0 { return Int(m) }
            // qwen3_5 hybrids pack a recurrent state into a fixed KV prefix.
            if (entry["model_type"] as? String) == "qwen3_5" { return 512 }
            if (entry["model_type"] as? String) == "qwen3_5_moe" { return 1024 }
        }
        if let inferred = inferHybridMinKVCapacity(manifest: manifest), inferred > 0 {
            return inferred
        }
        return 0
    }

    /// Best-effort floor for qwen3_5 hybrid bundles converted outside the registry path.
    /// The converter should eventually bake `language.min_kv_capacity` into metadata, but
    /// dashboard/HF conversions may produce bundles before that metadata exists. A too-small
    /// cache corrupts or rejects the SSM prefix; a conservative floor only allocates extra cache.
    private static func inferHybridMinKVCapacity(manifest: BundleManifest) -> Int? {
        let haystack = [
            manifest.name,
            manifest.source?.hfModelId,
            manifest.language?.tokenizer,
        ].compactMap { $0?.lowercased() }.joined(separator: " ")

        if haystack.contains("ornith") { return 1024 }
        if haystack.contains("qwen3.6-27b") { return 768 }
        if haystack.contains("qwen3_5") || haystack.contains("qwen3.5")
            || haystack.contains("qwythos") || haystack.contains("ornith")
        {
            return 512
        }
        return nil
    }

    /// Find the `models/registry.json` entry whose `hf_repo` equals `hfRepo`, searching upward
    /// from the bundle directory for the registry (standard layout: `exports/<bundle>` →
    /// `models/registry.json` two levels up). Best-effort and offline; returns `nil` if absent.
    private static func registryEntry(forHFRepo hfRepo: String, near root: URL) -> [String: Any]? {
        let fm = FileManager.default
        var dir = root.standardizedFileURL
        for _ in 0..<6 {
            let candidate = dir.appendingPathComponent("models/registry.json")
            if fm.fileExists(atPath: candidate.path),
                let data = try? Data(contentsOf: candidate),
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let models = obj["models"] as? [String: Any]
            {
                for (_, value) in models {
                    if let dict = value as? [String: Any],
                        (dict["hf_repo"] as? String) == hfRepo
                    {
                        return dict
                    }
                }
                return nil  // registry found but no matching entry
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            dir = parent
        }
        return nil
    }
}
