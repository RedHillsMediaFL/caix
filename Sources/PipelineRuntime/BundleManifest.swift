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

        public init(byName: [String: String]) {
            self.byName = byName
        }

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

        public func path(for name: String) -> String? {
            byName[name]
        }
    }

    public struct Language: Codable, Sendable {
        public struct FunctionMap: Codable, Sendable {
            public let byRole: [String: [String]]

            public init(from decoder: Decoder) throws {
                let c = try decoder.singleValueContainer()
                self.byRole = try c.decode([String: [String]].self)
            }

            public func name(for role: String) -> String? {
                byRole[role]?.first
            }
        }

        public let tokenizer: String
        public let vocabSize: Int
        public let maxContextLength: Int
        public let embeddedTokenizer: Bool
        /// Optional logical role -> exported function names map (`main`, `decode`, etc.).
        public let functionMap: FunctionMap?
        /// Optional per-model KV-cache floor baked into the bundle (overrides registry lookup).
        public let minKVCapacity: Int?

        enum CodingKeys: String, CodingKey {
            case tokenizer
            case vocabSize = "vocab_size"
            case maxContextLength = "max_context_length"
            case embeddedTokenizer = "embedded_tokenizer"
            case functionMap = "function_map"
            case minKVCapacity = "min_kv_capacity"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.tokenizer = try c.decode(String.self, forKey: .tokenizer)
            self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
            self.maxContextLength = try c.decode(Int.self, forKey: .maxContextLength)
            self.embeddedTokenizer =
                try c.decodeIfPresent(Bool.self, forKey: .embeddedTokenizer) ?? true
            self.functionMap = try c.decodeIfPresent(FunctionMap.self, forKey: .functionMap)
            self.minKVCapacity = try c.decodeIfPresent(Int.self, forKey: .minKVCapacity)
        }
    }

    /// Optional image/audio/video capability block. For Gemma 4 monolithic image-text bundles this
    /// records the non-staged multimodal ABI so the server can route the bundle explicitly instead
    /// of treating it as a plain text LLM.
    public struct Multimodal: Codable, Sendable {
        public let kind: String
        public let modalities: [String]?
        public let maxImages: Int?
        public let softTokensPerImage: Int?
        public let prefillFunction: String?
        public let visionFunction: String?
        public let embedderAsset: String?
        public let embedderAssetPath: String?
        public let blockIDsRequired: Bool?

        enum CodingKeys: String, CodingKey {
            case kind
            case modalities
            case maxImages = "max_images"
            case softTokensPerImage = "soft_tokens_per_image"
            case prefillFunction = "prefill_function"
            case visionFunction = "vision_function"
            case embedderAsset = "embedder_asset"
            case embedderAssetPath = "embedder_asset_path"
            case blockIDsRequired = "block_ids_required"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.kind = try c.decode(String.self, forKey: .kind)
            self.modalities = try c.decodeIfPresent([String].self, forKey: .modalities)
            self.maxImages = try c.decodeIfPresent(Int.self, forKey: .maxImages)
            self.softTokensPerImage = try c.decodeIfPresent(Int.self, forKey: .softTokensPerImage)
            self.prefillFunction = try c.decodeIfPresent(String.self, forKey: .prefillFunction)
            self.visionFunction = try c.decodeIfPresent(String.self, forKey: .visionFunction)
            self.embedderAsset = try c.decodeIfPresent(String.self, forKey: .embedderAsset)
            self.embedderAssetPath = try c.decodeIfPresent(String.self, forKey: .embedderAssetPath)
            self.blockIDsRequired = try c.decodeIfPresent(Bool.self, forKey: .blockIDsRequired)
        }

        public var normalizedKind: String {
            kind.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }

        public var isGemma4Monolithic: Bool {
            normalizedKind == "gemma4_monolithic"
                || normalizedKind == "gemma4_monolithic_multimodal"
                || normalizedKind == "gemma4_image_text_monolithic"
        }
    }

    /// Native Core AI execution contract for Qwen3.8-27B. This remains deliberately model
    /// specific: it describes the hybrid architecture's four persistent GPU states rather than
    /// weakening the generic two-state LLM engine ABI.
    public struct Qwen38: Codable, Sendable {
        public struct StateLayout: Codable, Sendable {
            public static let nativeStateNames = [
                "keyCache", "valueCache", "convState", "recurrentState",
            ]

            public let names: [String]
            public let fullAttentionLayers: Int
            public let kvHeads: Int
            public let headDimension: Int
            public let convDType: String
            public let recurrentDType: String

            enum CodingKeys: String, CodingKey {
                case names
                case fullAttentionLayers = "full_attention_layers"
                case kvHeads = "kv_heads"
                case headDimension = "head_dimension"
                case convDType = "conv_dtype"
                case recurrentDType = "recurrent_dtype"
            }

            /// True only for the compact K/V plus fixed convolution/recurrent state layout
            /// exported for the 64-layer Qwen3.8 hybrid. Order is part of the Core AI ABI.
            public var isNativeQwen38Layout: Bool {
                names == Self.nativeStateNames
                    && fullAttentionLayers == 16
                    && kvHeads == 4
                    && headDimension == 256
                    && convDType.lowercased() == "float16"
                    && recurrentDType.lowercased() == "float32"
            }
        }

        public struct MTP: Codable, Sendable {
            /// Key in `assets` naming the native Qwen MTP sidecar `.aimodel`.
            public let asset: String
            public let maxDraftTokens: Int
            public let requiresGreedy: Bool
            /// Same-machine evidence written only after deterministic parity and throughput
            /// validation. Absence means the sidecar is never selected automatically.
            public let proof: Qwen38MTPProof?

            enum CodingKeys: String, CodingKey {
                case asset
                case maxDraftTokens = "max_draft_tokens"
                case requiresGreedy = "requires_greedy"
                case proof
            }
        }

        public let stateLayout: StateLayout
        /// Optional until a sidecar has passed exact-greedy parity and the speed gate.
        /// The four-state target remains a native Qwen3.8 bundle without speculative decoding.
        public let mtp: MTP?
        public let thinkingDefault: Bool

        enum CodingKeys: String, CodingKey {
            case stateLayout = "state_layout"
            case mtp
            case thinkingDefault = "thinking_default"
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
    public let multimodal: Multimodal?
    /// Optional Qwen3.8-27B hybrid/Core AI capability block.
    public let qwen38: Qwen38?
    public let source: Source?

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.metadataVersion = try c.decodeIfPresent(String.self, forKey: .metadataVersion) ?? "legacy-coreai-asset"
        self.kind = try c.decodeIfPresent(String.self, forKey: .kind) ?? "coreai_asset"
        self.name = try c.decodeIfPresent(String.self, forKey: .name) ?? "legacy-coreai-asset"
        self.assets = try c.decodeIfPresent(Assets.self, forKey: .assets) ?? Assets(byName: ["main": "."])
        self.language = try c.decodeIfPresent(Language.self, forKey: .language)
        self.multimodal = try c.decodeIfPresent(Multimodal.self, forKey: .multimodal)
        self.qwen38 = try c.decodeIfPresent(Qwen38.self, forKey: .qwen38)
        self.source = try c.decodeIfPresent(Source.self, forKey: .source)
    }

    enum CodingKeys: String, CodingKey {
        case metadataVersion = "metadata_version"
        case kind
        case name
        case assets
        case language
        case multimodal
        case qwen38 = "qwen3_8"
        case source
    }
}

/// A model bundle resolved to concrete on-disk locations.
public struct ResolvedBundle: Sendable {
    public let root: URL
    /// The `.aimodel` package directory (the `AIProgram` Apple loads).
    public let aimodelURL: URL
    /// Optional dedicated one-token decode `.aimodel` package directory.
    public let decodeAimodelURL: URL?
    /// Dedicated native Qwen MTP sidecar. Present only for a validated Qwen3.8 bundle.
    public let mtpAimodelURL: URL?
    /// The HuggingFace tokenizer directory (`tokenizer/` containing `tokenizer.json`).
    public let tokenizerDir: URL
    public let manifest: BundleManifest
    /// Typed native Qwen3.8 execution contract. `nil` preserves legacy routing unchanged.
    public let qwen38: BundleManifest.Qwen38?
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
    /// True only when the bundle passed the strict Qwen3.8 hybrid/MTP validation below.
    public var isQwen38Native: Bool { qwen38 != nil }

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
        let decodeAimodelURL: URL?
        if let decodeRel = manifest.assets.path(for: "decode"), decodeRel != assetRel {
            let url = root.appendingPathComponent(decodeRel)
            guard fm.fileExists(atPath: url.path) else {
                throw CoreAIPipeline.RuntimeError.invalidBundle("missing decode asset \(decodeRel)")
            }
            decodeAimodelURL = url
        } else {
            decodeAimodelURL = nil
        }

        let mtpAimodelURL: URL?
        if let qwen38 = manifest.qwen38 {
            try Self.validateQwen38Contract(manifest: manifest, qwen38: qwen38)
            if let mtp = qwen38.mtp {
                guard let mtpRel = manifest.assets.path(for: mtp.asset) else {
                    throw CoreAIPipeline.RuntimeError.invalidBundle(
                        "Qwen3.8 metadata names missing MTP asset '\(mtp.asset)'")
                }
                let url = root.appendingPathComponent(mtpRel)
                guard fm.fileExists(atPath: url.path) else {
                    throw CoreAIPipeline.RuntimeError.invalidBundle("missing Qwen3.8 MTP asset \(mtpRel)")
                }
                mtpAimodelURL = url
            } else {
                mtpAimodelURL = nil
            }
        } else {
            mtpAimodelURL = nil
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
            decodeAimodelURL: decodeAimodelURL,
            mtpAimodelURL: mtpAimodelURL,
            tokenizerDir: tokenizerDir,
            manifest: manifest,
            qwen38: manifest.qwen38,
            minKVCapacity: Self.resolveMinKVCapacity(root: root, manifest: manifest),
            diffusion: diffusion,
            vocabSize: vocab,
            maxContextLength: context)
    }

    /// Keeps malformed or legacy packed-state Qwen3.5 artifacts out of the native Qwen3.8
    /// route. A failure is preferable to silently allocating a 64 GiB all-layer cache or
    /// serving a speculative sidecar under sampling semantics it cannot preserve.
    private static func validateQwen38Contract(
        manifest: BundleManifest,
        qwen38: BundleManifest.Qwen38
    ) throws {
        guard manifest.kind == "llm" else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("Qwen3.8 bundle kind must be 'llm'")
        }
        guard qwen38.stateLayout.isNativeQwen38Layout else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Qwen3.8 requires compact K/V plus fixed conv/recurrent four-state layout")
        }
        guard manifest.language?.maxContextLength == 262_144,
            manifest.language?.vocabSize == 248_320
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Qwen3.8 metadata must declare the 248320-token vocabulary and 262144-token context")
        }
        let main = manifest.language?.functionMap?.name(for: "prefill")
            ?? manifest.language?.functionMap?.name(for: "main")
        guard main != nil else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Qwen3.8 metadata is missing a main or prefill function")
        }
        if let mtp = qwen38.mtp {
            guard mtp.maxDraftTokens == 3, mtp.requiresGreedy else {
                throw CoreAIPipeline.RuntimeError.invalidBundle(
                    "Qwen3.8 native MTP requires exactly three greedy draft positions")
            }
            let requiredFunctions = ["decode", "verify_1", "verify_2", "verify_3"]
            guard requiredFunctions.allSatisfy({
                manifest.language?.functionMap?.name(for: $0) != nil
            }) else {
                throw CoreAIPipeline.RuntimeError.invalidBundle(
                    "Qwen3.8 MTP metadata is missing decode or verify_1/2/3 functions")
            }
        }
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
