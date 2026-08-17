import Foundation
import MachineStats
import PipelineRuntime
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// MARK: - Dashboard / API DTOs

/// One row of `GET /api/models` (`[{name, params, status, bundle}]`). `memoryBytes` is an
/// additive field carrying the resident footprint of loaded models (ignored by the dashboard).
/// `reasoningSupported` is derived from bundle tokenizer markers and does not require model load.
/// `multimodalSupported` is true for bundles that can accept the currently supported image route.
/// `multimodalCapabilities` may also describe detected-but-blocked multimodal bundle contracts.
public struct MultimodalCapabilities: Codable, Sendable, Equatable {
    public var family: String
    public var backend: String
    public var routeAvailable: Bool
    public var supportedModalities: [String]
    public var maxImages: Int
    public var imageSourceTypes: [String]
    public var maxSoftTokensPerImage: Int?
    public var supportedDecoding: [String]
    public var unsupportedFeatures: [String]

    public init(
        family: String,
        backend: String = "staged",
        routeAvailable: Bool = true,
        supportedModalities: [String],
        maxImages: Int,
        imageSourceTypes: [String],
        maxSoftTokensPerImage: Int?,
        supportedDecoding: [String],
        unsupportedFeatures: [String]
    ) {
        self.family = family
        self.backend = backend
        self.routeAvailable = routeAvailable
        self.supportedModalities = supportedModalities
        self.maxImages = maxImages
        self.imageSourceTypes = imageSourceTypes
        self.maxSoftTokensPerImage = maxSoftTokensPerImage
        self.supportedDecoding = supportedDecoding
        self.unsupportedFeatures = unsupportedFeatures
    }

    public static func gemma4ImageText(
        maxSoftTokensPerImage: Int? = nil,
        backend: String = "staged",
        routeAvailable: Bool = true
    ) -> MultimodalCapabilities {
        var unsupportedFeatures = [
            "audio",
            "video",
            "files",
            "remote_image_urls",
            "multiple_images",
            "tools",
            "response_format",
            "non_greedy_decoding",
        ]
        if !routeAvailable {
            unsupportedFeatures.append("monolithic_prefill_runtime")
        }
        return MultimodalCapabilities(
            family: "gemma4",
            backend: backend,
            routeAvailable: routeAvailable,
            supportedModalities: ["image", "text"],
            maxImages: 1,
            imageSourceTypes: ["base64", "data_url"],
            maxSoftTokensPerImage: maxSoftTokensPerImage,
            supportedDecoding: ["greedy"],
            unsupportedFeatures: unsupportedFeatures)
    }
}

public struct ModelEntry: Codable, Sendable {
    public var name: String
    public var params: String
    public var status: String  // "loaded" | "available"
    public var bundle: Bool
    public var memoryBytes: UInt64?
    public var mode: String?
    public var reasoningSupported: Bool?
    public var multimodalSupported: Bool?
    public var multimodalCapabilities: MultimodalCapabilities?
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
        case multimodalMonolithicGemma(PersistentModel)
        case speculative(PersistentSpeculativeModel)
        #if COREAI_RUNTIME
        case qwen38MTP(PersistentQwen38MTPModel)
        case eagle(EagleEngine)  // EAGLE speculative decoding
        case textStaged(TextStagedModel)
        case multimodalStaged(MultimodalStagedModel, Gemma4VisionEmbedder)
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

    init(monolithicMultimodalGemma model: PersistentModel, name: String) {
        self.backend = .multimodalMonolithicGemma(model)
        self.displayName = name
        self.bytes = model.bundleByteSize
    }

    init(speculative: PersistentSpeculativeModel, name: String) {
        self.backend = .speculative(speculative)
        self.displayName = name
        self.bytes = speculative.bundleByteSize
    }

    #if COREAI_RUNTIME
    init(qwen38MTP model: PersistentQwen38MTPModel, name: String) {
        self.backend = .qwen38MTP(model)
        self.displayName = name
        self.bytes = model.bundleByteSize
    }

    init(eagle: EagleEngine, name: String, bytes: UInt64) {
        self.backend = .eagle(eagle)
        self.displayName = name
        self.bytes = bytes
    }

    init(textStaged: TextStagedModel, name: String, bytes: UInt64) {
        self.backend = .textStaged(textStaged)
        self.displayName = name
        self.bytes = bytes
    }

    init(multimodalStaged: MultimodalStagedModel, embedder: Gemma4VisionEmbedder, name: String, bytes: UInt64) {
        self.backend = .multimodalStaged(multimodalStaged, embedder)
        self.displayName = name
        self.bytes = bytes
    }
    #endif

    var name: String { displayName }
    var memoryBytes: UInt64 { bytes }
    var multimodalCapabilities: MultimodalCapabilities? {
        #if COREAI_RUNTIME
        if case .multimodalStaged = backend {
            return .gemma4ImageText(maxSoftTokensPerImage: Gemma4MultimodalProcessor.maxSoftTokens)
        }
        #endif
        if case .multimodalMonolithicGemma = backend {
            return .gemma4ImageText(
                maxSoftTokensPerImage: 280,
                backend: "monolithic",
                routeAvailable: false)
        }
        return nil
    }
    var supportsMultimodalInput: Bool {
        multimodalCapabilities?.routeAvailable == true
    }
    var supportsConstrainedDecoding: Bool {
        switch backend {
        case .persistent(let model):
            return model.supportsConstrainedDecoding
        case .multimodalMonolithicGemma(let model):
            return model.supportsConstrainedDecoding
        case .speculative:
            return false
        #if COREAI_RUNTIME
        case .qwen38MTP:
            return false
        case .eagle:
            return false
        case .textStaged:
            return false
        case .multimodalStaged:
            return false
        #endif
        }
    }
    var defaultsToGreedyWhenTemperatureOmitted: Bool {
        #if COREAI_RUNTIME
        if case .textStaged = backend { return true }
        #endif
        return false
    }
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
        messages: [[String: any Sendable]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        await gate.acquire()
        do {
            let result: CoreAIPipeline.Result
            switch backend {
            case .persistent(let model):
                result = try await model.generate(
                    messages: try Self.stringMessages(messages),
                    options: options,
                    tools: tools,
                    onToken: onToken)
            case .multimodalMonolithicGemma(let model):
                result = try await model.generate(
                    messages: try Self.stringMessages(messages),
                    options: options,
                    tools: tools,
                    onToken: onToken)
            case .speculative(let model):
                let r = try await model.generate(
                    messages: try Self.stringMessages(messages),
                    options: options,
                    tools: tools,
                    onToken: onToken)
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
            case .qwen38MTP(let model):
                result = try await model.generate(
                    messages: try Self.stringMessages(messages),
                    options: options,
                    tools: tools,
                    onToken: onToken)
            case .eagle(let engine):
                let r = try await engine.generate(
                    messages: messages,
                    options: options,
                    tools: tools,
                    additionalContext: additionalContext,
                    onToken: onToken)
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
            case .textStaged(let model):
                result = try await model.generate(
                    messages: messages,
                    options: options,
                    tools: tools,
                    additionalContext: additionalContext,
                    onToken: onToken)
            case .multimodalStaged(let model, _):
                if let tools, !tools.isEmpty {
                    throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                        "multimodal staged Gemma does not support tool prompting yet")
                }
                result = try await model.generate(
                    messages: try Self.stringMessages(messages),
                    options: options,
                    onToken: onToken)
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

    func stagedMTPStartupProof() async throws -> StagedMTPStartupProof {
        #if !COREAI_RUNTIME
        throw CoreAIPipeline.RuntimeError.runtimeUnavailable
        #else
        await gate.acquire()
        do {
            guard case .textStaged(let model) = backend else {
                throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                    "configured staged MTP primary did not load as a text staged model")
            }
            let telemetry = try await model.prewarmMTPProof()
            guard telemetry.strategy == "sequential_no_rollback" else {
                throw StagedMTPStartupConfiguration.ConfigurationError.proofUnavailable(
                    "runtime reported unexpected MTP strategy: \(telemetry.strategy)")
            }
            await gate.release()
            return StagedMTPStartupProof(
                draftedTokens: telemetry.draftedTokens,
                executionMode: .sequentialNoRollback,
                fast: telemetry.fastMTP)
        } catch {
            await gate.release()
            throw error
        }
        #endif
    }

    private static func stringMessages(
        _ messages: [[String: any Sendable]]
    ) throws -> [[String: String]] {
        try messages.map { message in
            guard let role = message["role"] as? String,
                  let content = message["content"] as? String
            else {
                throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                    "this backend requires string role/content messages")
            }
            return ["role": role, "content": content]
        }
    }

    func generateMultimodal(
        request: GenerationRequest,
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        #if !COREAI_RUNTIME
        _ = (request, options, tools, onToken)
        throw CoreAIPipeline.RuntimeError.runtimeUnavailable
        #else
        await gate.acquire()
        do {
            let result: CoreAIPipeline.Result
            switch backend {
            case .multimodalStaged(let model, let embedder):
                if let tools, !tools.isEmpty {
                    throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                        "multimodal staged Gemma does not support tool prompting yet")
                }
                let imageData = try MultimodalRequestSupport.singleImageData(in: request)
                let messages = try MultimodalRequestSupport.messagesWithImagePlaceholder(
                    from: request.messages,
                    imageTextSeparator: model.imageTextSeparator)
                let generated = try await model.generateSingleImage(
                    messages: messages,
                    imageData: imageData,
                    embedder: embedder,
                    options: options,
                    onToken: onToken)
                if options.verbose {
                    FileHandle.standardError.write(Data(
                        String(
                            format: "[server-mm] model=%@ embedder=%@ compute=%@ preprocess=%.3fs embedder=%.3fs output_shape=%@ rows=%d\n",
                            displayName,
                            embedder.assetURL.path,
                            embedder.computeMode,
                            generated.embedding.preprocessSeconds,
                            generated.embedding.embedderSeconds,
                            String(describing: generated.embedding.outputShape),
                            generated.embedding.rows).utf8))
                }
                result = generated.result
            case .persistent, .multimodalMonolithicGemma, .speculative, .qwen38MTP,
                .eagle, .textStaged:
                throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                    "resolved backend '\(displayName)' does not support routed multimodal input")
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
        #endif
    }
}

enum MultimodalRequestSupport {
    enum RequestError: Error, CustomStringConvertible {
        case unsupported(String)

        var description: String {
            switch self {
            case .unsupported(let message): return message
            }
        }
    }

    static func validateMinimalSingleImageRequest(
        _ request: GenerationRequest,
        capabilities: MultimodalCapabilities = .gemma4ImageText()
    ) -> RequestError? {
        guard request.hasMultimodalContent else { return nil }
        guard capabilities.routeAvailable else {
            if capabilities.backend == "monolithic" {
                return .unsupported(
                    "monolithic Gemma image-text bundles are discovered, but native prefill_multimodal serving is not wired yet")
            }
            return .unsupported("resolved multimodal backend is not available for generation")
        }
        guard capabilities.supportedModalities.contains("image"), capabilities.maxImages == 1 else {
            return .unsupported("resolved multimodal backend does not support single-image generation")
        }
        if let tools = request.tools, !tools.isEmpty {
            return .unsupported("tools are not supported on the multimodal route yet")
        }
        if let responseFormat = request.responseFormat, responseFormat.requiresConstrainedDecoding {
            return .unsupported("response_format is not supported on the multimodal route yet")
        }
        let media = request.media
        let imageCount = media.filter { $0.modality == "image" }.count
        guard media.count == 1, imageCount == 1 else {
            return .unsupported(
                "multimodal route supports exactly one JSON base64 image; got modalities \(request.modalities.joined(separator: ","))")
        }
        do {
            _ = try singleImageData(in: request)
            _ = try messagesWithImagePlaceholder(from: request.messages)
            return nil
        } catch let error as RequestError {
            return error
        } catch {
            return .unsupported("\(error)")
        }
    }

    static func singleImageData(in request: GenerationRequest) throws -> Data {
        let media = request.media
        guard media.count == 1, let image = media.first, image.modality == "image" else {
            throw RequestError.unsupported("multimodal route supports exactly one image")
        }
        return try imageData(from: image.payload)
    }

    static func messagesWithImagePlaceholder(
        from messages: [ChatMessage],
        imageTextSeparator: String = "\n"
    ) throws -> [[String: String]] {
        var inserted = false
        return try messages.map { message in
            guard !message.media.isEmpty else {
                return ["role": message.role, "content": message.content]
            }
            guard message.media.count == 1,
                  message.media.first?.modality == "image",
                  !inserted
            else {
                throw RequestError.unsupported("multimodal route supports exactly one image message")
            }
            inserted = true
            let text = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            let content = text.isEmpty ? "<|image|>" : "<|image|>\(imageTextSeparator)\(text)"
            return ["role": message.role, "content": content]
        }
    }

    private static func imageData(from payload: JSONAny) throws -> Data {
        guard case .object(let object) = payload else {
            throw RequestError.unsupported("image payload must be a JSON object")
        }
        if let imageURL = object["image_url"] {
            if let raw = stringValue(imageURL) {
                return try dataURLOrBase64(raw)
            }
            if case .object(let nested) = imageURL,
               let raw = stringValue(nested["url"])
            {
                return try dataURLOrBase64(raw)
            }
        }
        if let raw = stringValue(object["url"]) ?? stringValue(object["data"]) {
            return try dataURLOrBase64(raw)
        }
        if case .object(let source) = object["source"],
           let raw = stringValue(source["data"])
        {
            return try dataURLOrBase64(raw)
        }
        throw RequestError.unsupported("image payload must contain a base64 data URL")
    }

    private static func dataURLOrBase64(_ raw: String) throws -> Data {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let base64: String
        if trimmed.lowercased().hasPrefix("data:") {
            guard let comma = trimmed.firstIndex(of: ",") else {
                throw RequestError.unsupported("image data URL is missing a comma separator")
            }
            let header = String(trimmed[..<comma]).lowercased()
            guard header.contains(";base64") else {
                throw RequestError.unsupported("image data URL must be base64 encoded")
            }
            base64 = String(trimmed[trimmed.index(after: comma)...])
        } else if trimmed.lowercased().hasPrefix("http://") || trimmed.lowercased().hasPrefix("https://") {
            throw RequestError.unsupported("remote image URLs are not supported; send a base64 data URL")
        } else {
            base64 = trimmed
        }
        guard let data = Data(base64Encoded: base64, options: [.ignoreUnknownCharacters]),
              !data.isEmpty
        else {
            throw RequestError.unsupported("image base64 payload could not be decoded")
        }
        return data
    }

    private static func stringValue(_ value: JSONAny?) -> String? {
        guard case .string(let text) = value else { return nil }
        return text
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
    struct NativeStagedMemorySnapshot: Sendable {
        let totalPhysicalMemoryBytes: UInt64
        let workerResidentBytes: UInt64
        let availableBytes: UInt64
        let allocationCapacityBytes: UInt64
        let pressure: ResidentMemoryPressure?
        let swapUsedBytes: UInt64?
    }

    private struct DiscoveredBundle: Sendable {
        var name: String
        var directoryName: String
        var mode: String
        var rootURL: URL
        var aliases: [String] = []

        var identifiers: [String] {
            Array(Set(([name, directoryName] + aliases).filter { !$0.isEmpty }))
        }
    }

    private let exportsDir: URL
    private let registryPath: URL
    private let verbose: Bool
    private let eagleConfig: EagleConfig?
    private let primaryStagedBundle: PrimaryStagedBundleConfiguration?
    private let stagedMTPConfiguration: StagedMTPStartupConfiguration?
    private let heavyTaskLockPath: URL

    private var handles: [String: ModelHandle] = [:]
    private var loadTasks: [String: Task<ModelHandle, Error>] = [:]
    private var bundleCache: (updatedAt: Date, entries: [DiscoveredBundle])?
    private let bundleDiscoveryCacheSeconds: TimeInterval = 5
    /// Memoized per-model output formats (detected from the bundle tokenizer/chat_template).
    private var formats: [String: OutputFormat] = [:]

    public init(exportsDir: URL, registryPath: URL, verbose: Bool = false,
                eagleConfig: EagleConfig? = nil,
                primaryStagedBundle: PrimaryStagedBundleConfiguration? = nil,
                stagedMTPConfiguration: StagedMTPStartupConfiguration? = nil,
                heavyTaskLockPath: URL? = nil) throws {
        self.exportsDir = exportsDir
        self.registryPath = registryPath
        self.verbose = verbose
        self.eagleConfig = eagleConfig
        self.primaryStagedBundle = primaryStagedBundle
        self.stagedMTPConfiguration = stagedMTPConfiguration
        self.heavyTaskLockPath = heavyTaskLockPath ?? Self.defaultHeavyTaskLockPath(exportsDir: exportsDir)
        if let stagedMTPConfiguration {
            guard primaryStagedBundle?.bundleURL.standardizedFileURL
                == stagedMTPConfiguration.primaryBundleURL.standardizedFileURL
            else {
                throw StagedMTPStartupConfiguration.ConfigurationError.missingPrimaryStagedBundle
            }
        }
        if let primaryStagedBundle {
            try Self.validatePrimaryStagedBundle(primaryStagedBundle, against: exportsDir)
            if let eagleConfig {
                let primary = Self.primaryStagedBundleEntry(primaryStagedBundle)
                if let alias = primary.identifiers.first(where: {
                    Self.normalize($0) == Self.normalize(eagleConfig.name)
                }) {
                    throw PrimaryStagedBundleConfiguration.ConfigurationError.aliasCollision(
                        alias: alias,
                        conflictingModel: eagleConfig.name)
                }
            }
        }
    }

    static func makeStagedMemorySnapshotProvider(
        readSnapshot: @escaping @Sendable () -> NativeStagedMemorySnapshot = {
            let snapshot = MachineStats.memorySafetySnapshot()
            let pressure: ResidentMemoryPressure?
            switch snapshot.pressure {
            case .green: pressure = .green
            case .yellow: pressure = .yellow
            case .red: pressure = .red
            case .unknown: pressure = nil
            }
            return NativeStagedMemorySnapshot(
                totalPhysicalMemoryBytes: snapshot.totalRAMBytes,
                workerResidentBytes: snapshot.processPhysicalFootprintBytes,
                availableBytes: snapshot.availableRAMBytes,
                allocationCapacityBytes: snapshot.allocationCapacityBytes,
                pressure: pressure,
                swapUsedBytes: snapshot.swapUsedBytes)
        }
    ) -> @Sendable () throws -> DistributedStagedMemorySnapshot {
        let baseline = readSnapshot()
        return {
            let current = readSnapshot()
            guard let pressure = current.pressure,
                  let baselineSwap = baseline.swapUsedBytes,
                  let currentSwap = current.swapUsedBytes
            else {
                throw DistributedStagedMemoryAdmissionError.telemetryUnavailable
            }
            return DistributedStagedMemorySnapshot(
                totalPhysicalMemoryBytes: current.totalPhysicalMemoryBytes,
                workerResidentBytes: current.workerResidentBytes,
                availableBytes: current.availableBytes,
                allocationCapacityBytes: current.allocationCapacityBytes,
                pressure: pressure,
                swapGrowthBytes: currentSwap >= baselineSwap
                    ? currentSwap - baselineSwap
                    : 0)
        }
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
        var entries = Self.discoverBundleEntries(in: exportsDir)
        if let primaryStagedBundle {
            let root = primaryStagedBundle.bundleURL.standardizedFileURL
            entries.removeAll { $0.rootURL.standardizedFileURL == root }
            entries.append(Self.primaryStagedBundleEntry(primaryStagedBundle))
        }
        entries.sort { $0.name < $1.name }
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
            let isMetadataBacked = mode == "standard" || mode == "staged" || mode == "multimodal_monolithic"
            let identity: (metadataName: String?, sourceModelID: String?, tokenizer: String?) =
                isMetadataBacked ? bundleIdentity(at: url) : (nil, nil, nil)
            let servedName = isMetadataBacked
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
                    mode: mode,
                    rootURL: url))
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
                    memoryBytes: handles[cfg.name]?.memoryBytes, mode: "eagle",
                    reasoningSupported: outputFormat(for: cfg.name).supportsReasoning,
                    multimodalSupported: false,
                    multimodalCapabilities: nil))
        }

        for bundle in bundleEntries() {
            let name = bundle.name
            if seen.contains(Self.normalize(name)) { continue }
            seen.insert(Self.normalize(name))
            let loaded = handles[name] != nil
            let capabilities = Self.multimodalCapabilities(at: bundle.rootURL)
            entries.append(
                ModelEntry(
                    name: name,
                    params: Self.inferParams(from: name),
                    status: loaded ? "loaded" : "available",
                    bundle: true,
                    memoryBytes: handles[name]?.memoryBytes,
                    mode: bundle.mode,
                    reasoningSupported: outputFormat(for: name).supportsReasoning,
                    multimodalSupported: capabilities?.routeAvailable ?? false,
                    multimodalCapabilities: capabilities))
        }

        for (key, params) in registryModels() {
            // Skip registry entries already represented by an exported bundle.
            if seen.contains(Self.normalize(key)) || seen.contains(Self.normalize(key + "-coreai")) {
                continue
            }
            entries.append(
                ModelEntry(
                    name: key, params: params, status: "available", bundle: false,
                    memoryBytes: nil, mode: "registry", reasoningSupported: nil,
                    multimodalSupported: nil,
                    multimodalCapabilities: nil))
        }
        return entries
    }

    public func servedModelsPreferredForChat() -> [ModelEntry] {
        listModels()
            .filter { $0.bundle }
            .sorted {
                let lhsIsPrimary = $0.name == primaryStagedBundle?.modelID
                let rhsIsPrimary = $1.name == primaryStagedBundle?.modelID
                if lhsIsPrimary != rhsIsPrimary { return lhsIsPrimary }
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

    func stagedMTPConfiguration(for bundleURL: URL) -> StagedMTPStartupConfiguration? {
        guard let stagedMTPConfiguration,
              stagedMTPConfiguration.primaryBundleURL.standardizedFileURL
                == bundleURL.standardizedFileURL
        else {
            return nil
        }
        return stagedMTPConfiguration
    }

    func requireStagedMTPProof() async throws -> StagedMTPStartupProof {
        guard let stagedMTPConfiguration, stagedMTPConfiguration.requireMTP,
              let primaryStagedBundle
        else {
            throw StagedMTPStartupConfiguration.ConfigurationError.proofUnavailable(
                "no required staged MTP primary is configured")
        }
        let handle = try await handle(for: primaryStagedBundle.modelID)
        return try await handle.stagedMTPStartupProof()
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
        resolvedBundle(for: name)?.rootURL.path
            ?? exportsDir.appendingPathComponent(name).path
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
            let root = resolvedBundle(for: name)?.rootURL
                ?? exportsDir.appendingPathComponent(name, isDirectory: true)
            tokenizerDir = root.appendingPathComponent("tokenizer")
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
        let stagedMTP = stagedMTPConfiguration(for: URL(fileURLWithPath: path, isDirectory: true))
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
                    draftTokens: 4, vocabSize: cfg.vocab, backbone: cfg.backbone,
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
                    draftTokens: 4, vocabSize: 262144, backbone: 2816,
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
            } else if let bundle = try? ResolvedBundle.load(at: path),
                bundle.qwen38?.mtp != nil,
                bundle.mtpAimodelURL != nil
            {
                #if COREAI_RUNTIME
                if verbose {
                    FileHandle.standardError.write(
                        Data("[server] loading native Qwen3.8 MTP bundle \(name)\n".utf8))
                }
                let model = try await PersistentQwen38MTPModel.load(
                    bundlePath: path, verbose: verbose)
                return ModelHandle(qwen38MTP: model, name: name)
                #else
                throw CoreAIPipeline.RuntimeError.runtimeUnavailable
                #endif
            } else if Self.isMultimodalStagedBundle(at: URL(fileURLWithPath: path, isDirectory: true)) {
                #if COREAI_RUNTIME
                let root = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
                guard let embedderURL = Self.multimodalEmbedderURL(for: root, exportsDir: URL(fileURLWithPath: path).deletingLastPathComponent())
                else {
                    throw CoreAIPipeline.RuntimeError.invalidBundle(
                        "multimodal staged bundle requires gemma4-mm-embedder_float32.aimodel (set CAIX_MM_EMBEDDER_ASSET or place it under exports)")
                }
                if verbose {
                    FileHandle.standardError.write(
                        Data("[server] loading multimodal staged bundle \(name) embedder=\(embedderURL.path)\n".utf8))
                }
                let model = try await MultimodalStagedModel.load(
                    manifestURL: root.appendingPathComponent("stage-manifest.json"),
                    verbose: verbose)
                let embedder = try await Gemma4VisionEmbedder.load(assetURL: embedderURL)
                return ModelHandle(
                    multimodalStaged: model,
                    embedder: embedder,
                    name: name,
                    bytes: Self.dirSize(root) + Self.dirSize(embedderURL))
                #else
                throw CoreAIPipeline.RuntimeError.runtimeUnavailable
                #endif
            } else if Self.isMultimodalMonolithicBundle(at: URL(fileURLWithPath: path, isDirectory: true)) {
                if verbose {
                    FileHandle.standardError.write(
                        Data("[server] loading monolithic multimodal Gemma bundle \(name) for text generation\n".utf8))
                }
                let model = try await PersistentModel.load(bundlePath: path, verbose: verbose)
                return ModelHandle(monolithicMultimodalGemma: model, name: name)
            } else if Self.isTextStagedBundle(at: URL(fileURLWithPath: path, isDirectory: true)) {
                #if COREAI_RUNTIME
                let root = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
                let stagedMemorySnapshotProvider =
                    Self.makeStagedMemorySnapshotProvider()
                if verbose {
                    FileHandle.standardError.write(
                        Data("[server] loading text staged bundle \(name)\n".utf8))
                }
                let model = try await TextStagedModel.load(
                    manifestURL: root.appendingPathComponent("stage-manifest.json"),
                    verbose: verbose,
                    stagedMemorySnapshotProvider: stagedMemorySnapshotProvider,
                    mtpAssistantURL: stagedMTP?.assistantURL,
                    mtpDraftTokens: stagedMTP?.draftTokens
                        ?? Gemma4MTPDecodeConfiguration.defaultDraftTokens)
                return ModelHandle(
                    textStaged: model,
                    name: name,
                    bytes: Self.dirSize(root)
                        + (stagedMTP.map { Self.dirSize($0.assistantURL) } ?? 0))
                #else
                throw CoreAIPipeline.RuntimeError.runtimeUnavailable
                #endif
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
        if let primaryStagedBundle,
           resolvedBundle(for: name)?.rootURL.standardizedFileURL
                == primaryStagedBundle.bundleURL.standardizedFileURL {
            return "refusing to delete the configured primary staged bundle"
        }
        if FileManager.default.fileExists(atPath: heavyTaskLockPath.path) {
            return "refusing to delete bundle while heavy-task lock exists: \(heavyTaskLockPath.path)"
        }
        let bundle = resolvedBundle(for: name)
        let servedName = bundle?.name ?? name
        let dir = bundle?.rootURL ?? exportsDir.appendingPathComponent(name)
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
        bundleEntries().first { $0.identifiers.contains(name) }
    }

    private func canonicalName(for name: String) -> String {
        resolvedBundle(for: name)?.name ?? name
    }

    private static func primaryStagedBundleEntry(
        _ configuration: PrimaryStagedBundleConfiguration
    ) -> DiscoveredBundle {
        let identity = bundleIdentity(at: configuration.bundleURL)
        let aliases = [
            configuration.bundleURL.lastPathComponent,
            identity.metadataName,
            identity.sourceModelID,
        ].compactMap { $0 }.filter { $0 != configuration.modelID }
        return DiscoveredBundle(
            name: configuration.modelID,
            directoryName: configuration.bundleURL.lastPathComponent,
            mode: "staged",
            rootURL: configuration.bundleURL,
            aliases: aliases)
    }

    private static func validatePrimaryStagedBundle(
        _ configuration: PrimaryStagedBundleConfiguration,
        against exportsDir: URL
    ) throws {
        let primary = primaryStagedBundleEntry(configuration)
        let root = configuration.bundleURL.standardizedFileURL
        let automatic = discoverBundleEntries(in: exportsDir).filter {
            $0.rootURL.standardizedFileURL != root
        }
        for alias in primary.identifiers {
            let normalizedAlias = normalize(alias)
            if let collision = automatic.first(where: {
                $0.identifiers.contains { normalize($0) == normalizedAlias }
            }) {
                throw PrimaryStagedBundleConfiguration.ConfigurationError.aliasCollision(
                    alias: alias,
                    conflictingModel: collision.name)
            }
        }
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
            || isMultimodalStagedBundle(at: root) || isMultimodalMonolithicBundle(at: root)
            || isTextStagedBundle(at: root)
    }

    static func bundleMode(at root: URL) -> String? {
        if isEagleBundle(at: root) { return "eagle" }
        if isClassicSpeculativeBundle(at: root) { return "speculative" }
        if isMultimodalStagedBundle(at: root) { return "multimodal_staged" }
        if isMultimodalMonolithicBundle(at: root) { return "multimodal_monolithic" }
        if isTextStagedBundle(at: root) { return "staged" }
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
            return DiscoveredBundle(
                name: name,
                directoryName: directoryName,
                mode: mode,
                rootURL: exportsDir.appendingPathComponent(directoryName, isDirectory: true))
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

    static func isMultimodalStagedBundle(at root: URL) -> Bool {
        stagedMultimodalMetadata(at: root) != nil
    }

    static func multimodalCapabilities(at root: URL) -> MultimodalCapabilities? {
        if let multimodal = stagedMultimodalMetadata(at: root) {
            let softTokens = intValue(multimodal["soft_tokens_per_image"])
            return .gemma4ImageText(maxSoftTokensPerImage: softTokens)
        }
        if let multimodal = monolithicMultimodalMetadata(at: root) {
            let softTokens = intValue(multimodal["soft_tokens_per_image"]) ?? 280
            return .gemma4ImageText(
                maxSoftTokensPerImage: softTokens,
                backend: "monolithic",
                routeAvailable: false)
        }
        return nil
    }

    private static func stagedMultimodalMetadata(at root: URL) -> [String: Any]? {
        let manifest = root.appendingPathComponent("stage-manifest.json")
        guard fileExists(manifest),
              let data = readSmallFile(manifest, timeoutSeconds: 0.5),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let multimodal = object["multimodal"] as? [String: Any],
              let kind = multimodal["kind"] as? String,
              kind == "gemma4_unified" || kind == "gemma4"
        else { return nil }
        return multimodal
    }

    static func isMultimodalMonolithicBundle(at root: URL) -> Bool {
        monolithicMultimodalMetadata(at: root) != nil
    }

    private static func monolithicMultimodalMetadata(at root: URL) -> [String: Any]? {
        let metadata = root.appendingPathComponent("metadata.json")
        guard fileExists(metadata),
              let data = readSmallFile(metadata, timeoutSeconds: 0.5),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (object["kind"] as? String) == "llm",
              let multimodal = object["multimodal"] as? [String: Any],
              let rawKind = multimodal["kind"] as? String
        else { return nil }
        let kind = rawKind.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard kind == "gemma4_monolithic"
                || kind == "gemma4_monolithic_multimodal"
                || kind == "gemma4_image_text_monolithic"
        else { return nil }
        guard declaresMonolithicMultimodalPrefill(object: object, multimodal: multimodal) else {
            return nil
        }
        return multimodal
    }

    private static func declaresMonolithicMultimodalPrefill(
        object: [String: Any],
        multimodal: [String: Any]
    ) -> Bool {
        if let prefill = multimodal["prefill_function"] as? String,
           prefill.trimmingCharacters(in: .whitespacesAndNewlines) == "prefill_multimodal" {
            return true
        }
        guard let language = object["language"] as? [String: Any],
              let functionMap = language["function_map"] as? [String: Any]
        else { return false }
        for value in functionMap.values {
            if let names = value as? [String],
               names.contains("prefill_multimodal") {
                return true
            }
            if let name = value as? String, name == "prefill_multimodal" {
                return true
            }
        }
        return false
    }

    static func isTextStagedBundle(at root: URL) -> Bool {
        let manifest = root.appendingPathComponent("stage-manifest.json")
        guard fileExists(manifest), !isMultimodalStagedBundle(at: root) else {
            return false
        }
        return true
    }

    private static func intValue(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        if let value = value as? Double { return Int(value) }
        return nil
    }

    static func multimodalEmbedderURL(for root: URL, exportsDir: URL) -> URL? {
        let env = ProcessInfo.processInfo.environment
        if let raw = caixEnv(env, "caix_mm_embedder_asset", legacy: "MM_EMBEDDER_ASSET")?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty
        {
            let url = URL(fileURLWithPath: raw).standardizedFileURL
            return isDirectory(url) ? url : nil
        }

        let manifest = root.appendingPathComponent("stage-manifest.json")
        let declared: (name: String?, path: String?) = {
            guard let data = readSmallFile(manifest, timeoutSeconds: 0.5),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let multimodal = object["multimodal"] as? [String: Any]
            else { return (nil, nil) }
            return (
                multimodal["embedder_asset"] as? String,
                multimodal["embedder_asset_path"] as? String)
        }()
        if let declaredPath = declared.path?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty {
            let url = URL(fileURLWithPath: declaredPath).standardizedFileURL
            if isDirectory(url) { return url }
        }
        let declaredName = declared.name?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
            ?? "gemma4-mm-embedder_float16.aimodel"
        let fp32Name = declaredName
            .replacingOccurrences(of: "_float16.aimodel", with: "_float32.aimodel")
            .replacingOccurrences(of: "float16.aimodel", with: "float32.aimodel")
        let candidates = [
            root.appendingPathComponent(fp32Name, isDirectory: true),
            exportsDir.appendingPathComponent(fp32Name, isDirectory: true),
        ]
        for candidate in candidates where isDirectory(candidate) {
            return candidate.standardizedFileURL
        }
        for child in childNames(in: exportsDir) {
            let candidate = exportsDir
                .appendingPathComponent(child, isDirectory: true)
                .appendingPathComponent(fp32Name, isDirectory: true)
            if isDirectory(candidate) {
                return candidate.standardizedFileURL
            }
        }
        return nil
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
