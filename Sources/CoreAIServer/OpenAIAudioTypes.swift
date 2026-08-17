import Foundation

enum OpenAIAudioAPI {
    static let modelID = "openai/whisper-large-v2"
    static let acceptedModelIDs = Set([modelID, "whisper-large-v2"])
    static let maximumUploadBytes = 25 * 1024 * 1024
    /// Leaves one bounded MiB for multipart framing and small text fields around a 25 MiB file.
    static let maximumRequestBytes = maximumUploadBytes + 1024 * 1024
}

enum OpenAIAudioResponseFormat: String, Sendable, CaseIterable {
    case json
    case text
    case srt
    case verboseJSON = "verbose_json"
    case vtt
}

enum OpenAIAudioTimestampGranularity: String, Sendable, CaseIterable {
    case word
    case segment
}

struct OpenAIAudioInput: Sendable, Equatable {
    var bytes: Data
    var filename: String
    var contentType: String?
}

struct OpenAIAudioTranscriptionRequest: Sendable, Equatable {
    enum ValidationError: Error, Sendable, Equatable, CustomStringConvertible {
        case missingField(String)
        case duplicateField(String)
        case invalidUTF8(String)
        case unsupportedModel(String)
        case emptyAudio
        case audioTooLarge
        case invalidResponseFormat(String)
        case invalidTemperature(String)
        case invalidBoolean(String)
        case invalidTimestampGranularity(String)
        case fieldTooLarge(String)

        var description: String {
            switch self {
            case .missingField(let field): return "missing required form field '\(field)'"
            case .duplicateField(let field): return "form field '\(field)' must occur exactly once"
            case .invalidUTF8(let field): return "form field '\(field)' must be UTF-8 text"
            case .unsupportedModel(let model): return "unsupported transcription model '\(model)'"
            case .emptyAudio: return "uploaded audio file is empty"
            case .audioTooLarge: return "uploaded audio exceeds the 25 MiB request limit"
            case .invalidResponseFormat(let value): return "unsupported response_format '\(value)'"
            case .invalidTemperature(let value): return "temperature must be a number from 0 through 1 (got '\(value)')"
            case .invalidBoolean(let value): return "stream must be true or false (got '\(value)')"
            case .invalidTimestampGranularity(let value): return "unsupported timestamp granularity '\(value)'"
            case .fieldTooLarge(let field): return "form field '\(field)' is too large"
            }
        }
    }

    var model: String
    var audio: OpenAIAudioInput
    var language: String?
    var prompt: String?
    var responseFormat: OpenAIAudioResponseFormat
    var temperature: Double
    var stream: Bool
    var timestampGranularities: [OpenAIAudioTimestampGranularity]

    init(form: MultipartFormData) throws {
        func uniquePart(_ name: String, required: Bool = false) throws -> MultipartFormData.Part? {
            let matches = form.parts(named: name)
            if matches.count > 1 { throw ValidationError.duplicateField(name) }
            if required && matches.isEmpty { throw ValidationError.missingField(name) }
            return matches.first
        }

        func text(_ name: String, required: Bool = false, maximumBytes: Int = 16 * 1024) throws -> String? {
            guard let part = try uniquePart(name, required: required) else { return nil }
            guard part.filename == nil else { throw ValidationError.duplicateField(name) }
            guard part.body.count <= maximumBytes else { throw ValidationError.fieldTooLarge(name) }
            guard let string = String(data: part.body, encoding: .utf8), !string.contains("\0") else {
                throw ValidationError.invalidUTF8(name)
            }
            return string.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let requestedModel = try text("model", required: true, maximumBytes: 256) ?? ""
        guard OpenAIAudioAPI.acceptedModelIDs.contains(requestedModel) else {
            throw ValidationError.unsupportedModel(requestedModel)
        }
        model = OpenAIAudioAPI.modelID

        guard let file = try uniquePart("file", required: true) else {
            throw ValidationError.missingField("file")
        }
        guard !file.body.isEmpty else { throw ValidationError.emptyAudio }
        guard file.body.count <= OpenAIAudioAPI.maximumUploadBytes else {
            throw ValidationError.audioTooLarge
        }
        audio = OpenAIAudioInput(
            bytes: file.body,
            filename: file.filename ?? "audio",
            contentType: file.contentType)

        language = try text("language", maximumBytes: 64)?.nonEmpty
        prompt = try text("prompt")?.nonEmpty

        let rawFormat = try text("response_format", maximumBytes: 64) ?? "json"
        guard let parsedFormat = OpenAIAudioResponseFormat(rawValue: rawFormat) else {
            throw ValidationError.invalidResponseFormat(rawFormat)
        }
        responseFormat = parsedFormat

        let rawTemperature = try text("temperature", maximumBytes: 64) ?? "0"
        guard let parsedTemperature = Double(rawTemperature), parsedTemperature.isFinite,
              (0...1).contains(parsedTemperature)
        else { throw ValidationError.invalidTemperature(rawTemperature) }
        temperature = parsedTemperature

        let rawStream = try text("stream", maximumBytes: 16) ?? "false"
        switch rawStream.lowercased() {
        case "true": stream = true
        case "false": stream = false
        default: throw ValidationError.invalidBoolean(rawStream)
        }

        var granularities: [OpenAIAudioTimestampGranularity] = []
        for part in form.parts(named: "timestamp_granularities[]") {
            guard part.body.count <= 64,
                  let value = String(data: part.body, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  let granularity = OpenAIAudioTimestampGranularity(rawValue: value)
            else {
                let value = String(data: part.body, encoding: .utf8) ?? "<non-UTF8>"
                throw ValidationError.invalidTimestampGranularity(value)
            }
            if !granularities.contains(granularity) { granularities.append(granularity) }
        }
        timestampGranularities = granularities
    }
}

struct OpenAIAudioTranscriptionJSONResponse: Encodable, Sendable, Equatable {
    var text: String
}

struct OpenAIAudioVerboseTranscriptionResponse: Encodable, Sendable, Equatable {
    var task = "transcribe"
    var language: String
    var duration: Double
    var text: String
}
