import Foundation
import PipelineRuntime

/// In-memory bridge from one validated OpenAI audio request to the resident native Whisper model.
///
/// The service deliberately has no HTTP concerns. It authenticates the subset of request
/// semantics the native decoder can honor, decodes the upload once, extracts one feature tensor,
/// narrows it to Float16 once, and then enters the serialized resident engine.
struct OpenAIAudioTranscriptionService: Sendable {
    typealias DecodeAudio = @Sendable (Data) async throws -> WhisperPCM
    typealias ExtractFeatures = @Sendable ([Float]) async throws -> [Float]

    enum ServiceError: Error, Sendable, Equatable, CustomStringConvertible {
        case promptUnsupported
        case nonzeroTemperatureUnsupported
        case timestampsUnsupported
        case responseFormatUnsupported(OpenAIAudioResponseFormat)
        case audioExceedsNativeWindow

        var description: String {
            switch self {
            case .promptUnsupported:
                return "prompt is not supported by the native Whisper decoder"
            case .nonzeroTemperatureUnsupported:
                return "temperature must be 0 for deterministic native Whisper decoding"
            case .timestampsUnsupported:
                return "timestamp granularities are not yet supported by the native Whisper decoder"
            case .responseFormatUnsupported(let format):
                return "response_format '\(format.rawValue)' requires timestamp alignment and is not yet supported"
            case .audioExceedsNativeWindow:
                return "audio exceeds the native Whisper 30-second request window"
            }
        }
    }

    struct Output: Sendable, Equatable {
        var text: String
        var language: String
        var durationSeconds: Double
        var native: WhisperResidentEngine.Result
    }

    private let transcriber: any WhisperTranscribing
    private let decodeAudio: DecodeAudio
    private let extractFeatures: ExtractFeatures

    init(
        transcriber: any WhisperTranscribing,
        decodeAudio: @escaping DecodeAudio = { data in
            try InMemoryAudioDecoder.decodeToWhisperPCM(data)
        },
        extractFeatures: @escaping ExtractFeatures = { samples in
            try WhisperLogMelExtractor.extract(samples: samples)
        }
    ) {
        self.transcriber = transcriber
        self.decodeAudio = decodeAudio
        self.extractFeatures = extractFeatures
    }

    func transcribe(
        _ request: OpenAIAudioTranscriptionRequest,
        onTextToken: (@Sendable (Int32) -> Void)? = nil
    ) async throws -> Output {
        try validateSemantics(request)
        try Task.checkCancellation()

        let pcm = try await decodeAudio(request.audio.bytes)
        guard !pcm.wasTruncated else { throw ServiceError.audioExceedsNativeWindow }
        try Task.checkCancellation()

        let floatFeatures = try await extractFeatures(pcm.samples)
        let inputFeatures = floatFeatures.map { Float16($0) }
        try Task.checkCancellation()

        let native = try await transcriber.transcribe(
            inputFeatures: inputFeatures,
            requestedLanguage: request.language,
            includeTimestamps: false,
            onTextToken: onTextToken)
        return Output(
            text: native.text,
            language: native.language,
            durationSeconds: Double(pcm.samples.count) / Double(pcm.sampleRate),
            native: native)
    }

    private func validateSemantics(_ request: OpenAIAudioTranscriptionRequest) throws {
        guard request.prompt == nil else { throw ServiceError.promptUnsupported }
        guard request.temperature == 0 else {
            throw ServiceError.nonzeroTemperatureUnsupported
        }
        guard request.timestampGranularities.isEmpty else {
            throw ServiceError.timestampsUnsupported
        }
        switch request.responseFormat {
        case .json, .text, .verboseJSON:
            break
        case .srt, .vtt:
            throw ServiceError.responseFormatUnsupported(request.responseFormat)
        }
    }
}
