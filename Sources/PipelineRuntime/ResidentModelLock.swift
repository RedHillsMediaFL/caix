import Foundation

/// Immutable source identities and conversion-critical contracts for the two resident models.
///
/// This lock describes Hugging Face source artifacts, not downloaded or converted output. In
/// particular, the Gemma QAT checkpoints contain BF16 source weights whose conversion target is
/// Q4_0; calling the source weights "4-bit" would be incorrect.
public struct ResidentModelLock: Codable, Sendable, Equatable {
    public struct WeightArtifact: Codable, Sendable, Equatable {
        public let path: String
        public let sizeBytes: UInt64
        public let sha256: String

        enum CodingKeys: String, CodingKey {
            case path, sha256
            case sizeBytes = "size_bytes"
        }
    }

    public struct GemmaMetadata: Codable, Sendable, Equatable {
        public let configSHA256: String
        public let generationConfigSHA256: String
        public let chatTemplateSHA256: String
        public let tokenizerJSONSHA256: String
        public let tokenizerConfigSHA256: String
        public let weightsIndexSHA256: String?

        enum CodingKeys: String, CodingKey {
            case configSHA256 = "config_sha256"
            case generationConfigSHA256 = "generation_config_sha256"
            case chatTemplateSHA256 = "chat_template_sha256"
            case tokenizerJSONSHA256 = "tokenizer_json_sha256"
            case tokenizerConfigSHA256 = "tokenizer_config_sha256"
            case weightsIndexSHA256 = "weights_index_sha256"
        }
    }

    public struct GemmaSource: Codable, Sendable, Equatable {
        public let repository: String
        public let revision: String
        public let sourcePrecision: String
        public let qatRecipe: String
        public let runtimePrecision: String
        public let weights: [WeightArtifact]
        public let metadata: GemmaMetadata

        enum CodingKeys: String, CodingKey {
            case repository, revision, weights, metadata
            case sourcePrecision = "source_precision"
            case qatRecipe = "qat_recipe"
            case runtimePrecision = "runtime_precision"
        }
    }

    public struct GemmaGeometry: Codable, Sendable, Equatable {
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let layers: Int
        public let maxContextLength: Int
        public let slidingWindow: Int
        public let attentionHeads: Int
        public let slidingKVHeads: Int
        public let globalKVHeads: Int
        public let headDimension: Int
        public let globalHeadDimension: Int
        public let slidingLayers: Int
        public let fullLayers: Int

        enum CodingKeys: String, CodingKey {
            case layers
            case vocabularySize = "vocabulary_size"
            case hiddenSize = "hidden_size"
            case maxContextLength = "max_context_length"
            case slidingWindow = "sliding_window"
            case attentionHeads = "attention_heads"
            case slidingKVHeads = "sliding_kv_heads"
            case globalKVHeads = "global_kv_heads"
            case headDimension = "head_dimension"
            case globalHeadDimension = "global_head_dimension"
            case slidingLayers = "sliding_layers"
            case fullLayers = "full_layers"
        }
    }

    public struct AssistantGeometry: Codable, Sendable, Equatable {
        public let vocabularySize: Int
        public let maxContextLength: Int
        public let backboneHiddenSize: Int
        public let hiddenSize: Int
        public let layers: Int
        public let slidingLayers: Int
        public let fullLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let globalKVHeads: Int
        public let sharedKVLayers: Int

        enum CodingKeys: String, CodingKey {
            case layers
            case vocabularySize = "vocabulary_size"
            case maxContextLength = "max_context_length"
            case backboneHiddenSize = "backbone_hidden_size"
            case hiddenSize = "hidden_size"
            case slidingLayers = "sliding_layers"
            case fullLayers = "full_layers"
            case attentionHeads = "attention_heads"
            case kvHeads = "kv_heads"
            case globalKVHeads = "global_kv_heads"
            case sharedKVLayers = "shared_kv_layers"
        }
    }

    public struct SamplingDefaults: Codable, Sendable, Equatable {
        public let temperature: Double
        public let topK: Int
        public let topP: Double
        public let eosTokenIDs: [Int]
        public let padTokenID: Int

        enum CodingKeys: String, CodingKey {
            case temperature
            case topK = "top_k"
            case topP = "top_p"
            case eosTokenIDs = "eos_token_ids"
            case padTokenID = "pad_token_id"
        }
    }

    public struct LLMLock: Codable, Sendable, Equatable {
        public let publicModelID: String
        public let target: GemmaSource
        public let assistant: GemmaSource
        public let chatTemplateSHA256: String
        public let geometry: GemmaGeometry
        public let assistantGeometry: AssistantGeometry
        public let sampling: SamplingDefaults
        public let contextCandidates: [Int]

        enum CodingKeys: String, CodingKey {
            case target, assistant, geometry, sampling
            case publicModelID = "public_model_id"
            case chatTemplateSHA256 = "chat_template_sha256"
            case assistantGeometry = "assistant_geometry"
            case contextCandidates = "context_candidates"
        }
    }

    public struct SpeechMetadata: Codable, Sendable, Equatable {
        public let configSHA256: String
        public let generationConfigSHA256: String
        public let preprocessorConfigSHA256: String
        public let tokenizerJSONSHA256: String
        public let tokenizerConfigSHA256: String
        public let addedTokensSHA256: String
        public let mergesSHA256: String
        public let vocabularySHA256: String
        public let normalizerSHA256: String

        enum CodingKeys: String, CodingKey {
            case configSHA256 = "config_sha256"
            case generationConfigSHA256 = "generation_config_sha256"
            case preprocessorConfigSHA256 = "preprocessor_config_sha256"
            case tokenizerJSONSHA256 = "tokenizer_json_sha256"
            case tokenizerConfigSHA256 = "tokenizer_config_sha256"
            case addedTokensSHA256 = "added_tokens_sha256"
            case mergesSHA256 = "merges_sha256"
            case vocabularySHA256 = "vocab_sha256"
            case normalizerSHA256 = "normalizer_sha256"
        }
    }

    public struct WhisperGeometry: Codable, Sendable, Equatable {
        public let sampleRateHz: Int
        public let melBins: Int
        public let windowSeconds: Int
        public let windowSamples: Int
        public let fftSize: Int
        public let hopLength: Int
        public let maxSourcePositions: Int
        public let maxDecoderPositions: Int
        public let vocabularySize: Int
        public let modelDimension: Int
        public let encoderLayers: Int
        public let decoderLayers: Int
        public let encoderAttentionHeads: Int
        public let decoderAttentionHeads: Int

        enum CodingKeys: String, CodingKey {
            case sampleRateHz = "sample_rate_hz"
            case melBins = "mel_bins"
            case windowSeconds = "window_seconds"
            case windowSamples = "window_samples"
            case fftSize = "fft_size"
            case hopLength = "hop_length"
            case maxSourcePositions = "max_source_positions"
            case maxDecoderPositions = "max_decoder_positions"
            case vocabularySize = "vocabulary_size"
            case modelDimension = "model_dimension"
            case encoderLayers = "encoder_layers"
            case decoderLayers = "decoder_layers"
            case encoderAttentionHeads = "encoder_attention_heads"
            case decoderAttentionHeads = "decoder_attention_heads"
        }
    }

    public struct SpeechLock: Codable, Sendable, Equatable {
        public let publicModelID: String
        public let repository: String
        public let revision: String
        public let sourcePrecision: String
        public let runtimePrecision: String
        public let weights: [WeightArtifact]
        public let metadata: SpeechMetadata
        public let geometry: WhisperGeometry

        enum CodingKeys: String, CodingKey {
            case repository, revision, weights, metadata, geometry
            case publicModelID = "public_model_id"
            case sourcePrecision = "source_precision"
            case runtimePrecision = "runtime_precision"
        }
    }

    public let schema: String
    public let deployment: String
    public let llm: LLMLock
    public let speech: SpeechLock

    public static let filename = "gemma4-whisper-lock.json"
    public static let maximumLockBytes = 256 * 1024

    /// Reads only a bounded regular file and validates every conversion-critical field against
    /// the approved deployment contract. The descriptor is opened without following symlinks,
    /// then its type and size are validated before any content is allocated or decoded.
    public static func load(from url: URL) throws -> ResidentModelLock {
        let data: Data
        do {
            data = try BoundedRegularFileReader.read(url, maximumBytes: maximumLockBytes)
        } catch let error as BoundedRegularFileReaderError {
            switch error {
            case .notRegularFile:
                throw ResidentModelLockError.notRegularFile
            case .tooLarge:
                throw ResidentModelLockError.fileTooLarge
            case .emptyFile:
                throw ResidentModelLockError.malformedJSON
            default:
                throw ResidentModelLockError.unreadable
            }
        }

        let lock: ResidentModelLock
        do {
            lock = try JSONDecoder().decode(Self.self, from: data)
        } catch {
            throw ResidentModelLockError.malformedJSON
        }
        try lock.validate()
        return lock
    }

    /// Validation is intentionally exact for this deployment rather than accepting a model that
    /// merely looks compatible. A converter must fail before downloading or compiling stale or
    /// substituted source artifacts.
    public func validate() throws {
        guard schema == Approved.schema else {
            throw ResidentModelLockError.unsupportedSchema(schema)
        }
        try Self.require(deployment == Approved.deployment, "deployment")

        try Self.require(llm.publicModelID == Approved.llmPublicID, "llm.public_model_id")
        try Self.validateGemmaSource(
            llm.target,
            approvedRepository: Approved.targetRepository,
            approvedRevision: Approved.targetRevision,
            approvedWeights: Approved.targetWeights,
            approvedMetadata: Approved.targetMetadata,
            field: "llm.target")
        try Self.validateGemmaSource(
            llm.assistant,
            approvedRepository: Approved.assistantRepository,
            approvedRevision: Approved.assistantRevision,
            approvedWeights: Approved.assistantWeights,
            approvedMetadata: Approved.assistantMetadata,
            field: "llm.assistant")
        try Self.require(
            Self.isLowercaseHex(llm.chatTemplateSHA256, count: 64)
                && llm.chatTemplateSHA256 == Approved.chatTemplateSHA256,
            "llm.chat_template_sha256")
        try Self.require(llm.geometry == Approved.gemmaGeometry, "llm.geometry")
        try Self.require(
            llm.assistantGeometry == Approved.assistantGeometry,
            "llm.assistant_geometry")
        try Self.require(llm.sampling == Approved.sampling, "llm.sampling")
        try Self.require(
            llm.contextCandidates == Approved.contextCandidates,
            "llm.context_candidates")

        try Self.require(
            speech.publicModelID == Approved.whisperRepository,
            "speech.public_model_id")
        try Self.require(
            speech.repository == Approved.whisperRepository,
            "speech.repository")
        try Self.require(
            Self.isLowercaseHex(speech.revision, count: 40)
                && speech.revision == Approved.whisperRevision,
            "speech.revision")
        try Self.require(speech.sourcePrecision == "fp32", "speech.source_precision")
        try Self.require(speech.runtimePrecision == "fp16", "speech.runtime_precision")
        try Self.validateWeights(
            speech.weights,
            approved: Approved.whisperWeights,
            field: "speech.weights")
        try Self.validateSpeechMetadata(speech.metadata)
        try Self.require(speech.geometry == Approved.whisperGeometry, "speech.geometry")
    }

    private static func validateGemmaSource(
        _ source: GemmaSource,
        approvedRepository: String,
        approvedRevision: String,
        approvedWeights: [WeightArtifact],
        approvedMetadata: GemmaMetadata,
        field: String
    ) throws {
        try require(source.repository == approvedRepository, "\(field).repository")
        try require(
            isLowercaseHex(source.revision, count: 40) && source.revision == approvedRevision,
            "\(field).revision")
        try require(source.sourcePrecision == "bf16", "\(field).source_precision")
        try require(source.qatRecipe == "q4_0", "\(field).qat_recipe")
        try require(source.runtimePrecision == "q4_0", "\(field).runtime_precision")
        try validateWeights(source.weights, approved: approvedWeights, field: "\(field).weights")
        try validateGemmaMetadata(source.metadata, approved: approvedMetadata, field: field)
    }

    private static func validateWeights(
        _ weights: [WeightArtifact],
        approved: [WeightArtifact],
        field: String
    ) throws {
        try require(
            weights.allSatisfy {
                !$0.path.isEmpty && $0.sizeBytes > 0 && isLowercaseHex($0.sha256, count: 64)
            } && weights == approved,
            field)
    }

    private static func validateGemmaMetadata(
        _ metadata: GemmaMetadata,
        approved: GemmaMetadata,
        field: String
    ) throws {
        try requireDigest(
            metadata.configSHA256, approved.configSHA256,
            "\(field).metadata.config_sha256")
        try requireDigest(
            metadata.generationConfigSHA256, approved.generationConfigSHA256,
            "\(field).metadata.generation_config_sha256")
        try requireDigest(
            metadata.chatTemplateSHA256, approved.chatTemplateSHA256,
            "\(field).metadata.chat_template_sha256")
        try requireDigest(
            metadata.tokenizerJSONSHA256, approved.tokenizerJSONSHA256,
            "\(field).metadata.tokenizer_json_sha256")
        try requireDigest(
            metadata.tokenizerConfigSHA256, approved.tokenizerConfigSHA256,
            "\(field).metadata.tokenizer_config_sha256")
        if let expected = approved.weightsIndexSHA256 {
            try requireDigest(
                metadata.weightsIndexSHA256, expected,
                "\(field).metadata.weights_index_sha256")
        } else {
            try require(metadata.weightsIndexSHA256 == nil, "\(field).metadata.weights_index_sha256")
        }
    }

    private static func validateSpeechMetadata(_ metadata: SpeechMetadata) throws {
        let approved = Approved.whisperMetadata
        try requireDigest(
            metadata.configSHA256, approved.configSHA256,
            "speech.metadata.config_sha256")
        try requireDigest(
            metadata.generationConfigSHA256, approved.generationConfigSHA256,
            "speech.metadata.generation_config_sha256")
        try requireDigest(
            metadata.preprocessorConfigSHA256, approved.preprocessorConfigSHA256,
            "speech.metadata.preprocessor_config_sha256")
        try requireDigest(
            metadata.tokenizerJSONSHA256, approved.tokenizerJSONSHA256,
            "speech.metadata.tokenizer_json_sha256")
        try requireDigest(
            metadata.tokenizerConfigSHA256, approved.tokenizerConfigSHA256,
            "speech.metadata.tokenizer_config_sha256")
        try requireDigest(
            metadata.addedTokensSHA256, approved.addedTokensSHA256,
            "speech.metadata.added_tokens_sha256")
        try requireDigest(
            metadata.mergesSHA256, approved.mergesSHA256,
            "speech.metadata.merges_sha256")
        try requireDigest(
            metadata.vocabularySHA256, approved.vocabularySHA256,
            "speech.metadata.vocab_sha256")
        try requireDigest(
            metadata.normalizerSHA256, approved.normalizerSHA256,
            "speech.metadata.normalizer_sha256")
    }

    private static func requireDigest(
        _ actual: String?,
        _ approved: String,
        _ field: String
    ) throws {
        try require(
            actual.map { isLowercaseHex($0, count: 64) && $0 == approved } ?? false,
            field)
    }

    private static func require(_ condition: @autoclosure () -> Bool, _ field: String) throws {
        guard condition() else { throw ResidentModelLockError.contractViolation(field) }
    }

    private static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        guard value.utf8.count == count else { return false }
        return value.utf8.allSatisfy { byte in
            (byte >= Character("0").asciiValue! && byte <= Character("9").asciiValue!)
                || (byte >= Character("a").asciiValue! && byte <= Character("f").asciiValue!)
        }
    }
}

public enum ResidentModelLockError: Error, Sendable, Equatable {
    case notRegularFile
    case fileTooLarge
    case unreadable
    case malformedJSON
    case unsupportedSchema(String)
    case contractViolation(String)
}

extension ResidentModelLockError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .notRegularFile:
            "resident model lock must be a regular, non-symlink file"
        case .fileTooLarge:
            "resident model lock exceeds the bounded file size"
        case .unreadable:
            "resident model lock could not be read"
        case .malformedJSON:
            "resident model lock is not valid schema JSON"
        case .unsupportedSchema(let schema):
            "unsupported resident model lock schema: \(schema)"
        case .contractViolation(let field):
            "resident model lock violates the approved contract at \(field)"
        }
    }
}

private extension ResidentModelLock {
    enum Approved {
        static let schema = "caix.resident-model-lock.v1"
        static let deployment = "gemma4-31b-whisper-large-v2"
        static let llmPublicID = "google/gemma-4-31B-it"
        static let chatTemplateSHA256 =
            "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"

        static let targetRepository = "google/gemma-4-31B-it-qat-q4_0-unquantized"
        static let targetRevision = "1e4d8beecacb8b7590c1d8bedd7335f687bf311f"
        static let targetWeights = [
            WeightArtifact(
                path: "model-00001-of-00002.safetensors",
                sizeBytes: 49_784_788_364,
                sha256: "8ad3c67895dca6184c70d88a31f042eca42971728782dfb2c18edb736f3060a0"),
            WeightArtifact(
                path: "model-00002-of-00002.safetensors",
                sizeBytes: 12_761_549_884,
                sha256: "a373e71426e369a2498a7a69793ce9ccdb07d2c96aa807c6baf675520f9add87"),
        ]
        static let targetMetadata = GemmaMetadata(
            configSHA256: "95b105ad23b315067985415e721ab9c19cfcf90918b34ea0fa479a08489d86b7",
            generationConfigSHA256:
                "b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9",
            chatTemplateSHA256: chatTemplateSHA256,
            tokenizerJSONSHA256:
                "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            tokenizerConfigSHA256:
                "3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999",
            weightsIndexSHA256:
                "d4aff3b976d69c123a29d1c085d7ba4de1ac3f4ca1726a7f81e1b11462a64ea2")

        static let assistantRepository =
            "google/gemma-4-31B-it-qat-q4_0-unquantized-assistant"
        static let assistantRevision = "96d4c8ca3cb38c107a8478587878124895d1e844"
        static let assistantWeights = [
            WeightArtifact(
                path: "model.safetensors",
                sizeBytes: 939_042_560,
                sha256: "50008e854554a1a9c26317216cd99ae5a3567d4942c9e061398b995cc48c34b9")
        ]
        static let assistantMetadata = GemmaMetadata(
            configSHA256: "ed05a080c2284a5c50847289ac7d280b98c695257be890bae7561b7ed440c654",
            generationConfigSHA256:
                "fb53f4c64e58896a63472e8eb304397db4a39453e1da0f5d57625ec5a8c1050e",
            chatTemplateSHA256: chatTemplateSHA256,
            tokenizerJSONSHA256:
                "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c",
            tokenizerConfigSHA256:
                "01f2ff1c21ef2e722891380323edcaecd9c86a776aeb9b40148e2f35e3cee4d3",
            weightsIndexSHA256: nil)

        static let gemmaGeometry = GemmaGeometry(
            vocabularySize: 262_144,
            hiddenSize: 5_376,
            layers: 60,
            maxContextLength: 262_144,
            slidingWindow: 1_024,
            attentionHeads: 32,
            slidingKVHeads: 16,
            globalKVHeads: 4,
            headDimension: 256,
            globalHeadDimension: 512,
            slidingLayers: 50,
            fullLayers: 10)
        static let assistantGeometry = AssistantGeometry(
            vocabularySize: 262_144,
            maxContextLength: 262_144,
            backboneHiddenSize: 5_376,
            hiddenSize: 1_024,
            layers: 4,
            slidingLayers: 3,
            fullLayers: 1,
            attentionHeads: 32,
            kvHeads: 16,
            globalKVHeads: 4,
            sharedKVLayers: 4)
        static let sampling = SamplingDefaults(
            temperature: 1,
            topK: 64,
            topP: 0.95,
            eosTokenIDs: [1, 106, 50],
            padTokenID: 0)
        static let contextCandidates = [262_144, 131_072, 65_536, 32_768, 16_384]

        static let whisperRepository = "openai/whisper-large-v2"
        static let whisperRevision = "ae4642769ce2ad8fc292556ccea8e901f1530655"
        static let whisperWeights = [
            WeightArtifact(
                path: "model.safetensors",
                sizeBytes: 6_173_370_152,
                sha256: "57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b")
        ]
        static let whisperMetadata = SpeechMetadata(
            configSHA256: "5f1573015838f8d679678b09354b537061561c55fd22eecd129ef4cf8588a470",
            generationConfigSHA256:
                "031721643aab5be7250eb668c6b9b5c67d2549420522ac1291bfd346bfff6297",
            preprocessorConfigSHA256:
                "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666",
            tokenizerJSONSHA256:
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
            tokenizerConfigSHA256:
                "2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce",
            addedTokensSHA256:
                "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1",
            mergesSHA256:
                "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6",
            vocabularySHA256:
                "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9",
            normalizerSHA256:
                "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd")
        static let whisperGeometry = WhisperGeometry(
            sampleRateHz: 16_000,
            melBins: 80,
            windowSeconds: 30,
            windowSamples: 480_000,
            fftSize: 400,
            hopLength: 160,
            maxSourcePositions: 1_500,
            maxDecoderPositions: 448,
            vocabularySize: 51_865,
            modelDimension: 1_280,
            encoderLayers: 32,
            decoderLayers: 32,
            encoderAttentionHeads: 20,
            decoderAttentionHeads: 20)
    }
}
