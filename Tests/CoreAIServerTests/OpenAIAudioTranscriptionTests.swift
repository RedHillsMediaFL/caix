import XCTest
import PipelineRuntime

@testable import CoreAIServer

final class OpenAIAudioTranscriptionTests: XCTestCase {
    func testServiceConvertsAudioOnceAndForwardsNativeInputs() async throws {
        let recorder = AudioPipelineRecorder()
        let transcriber = RecordingWhisperTranscriber()
        let request = try transcriptionRequest(
            fields: [("language", "en"), ("response_format", "verbose_json")])
        let service = OpenAIAudioTranscriptionService(
            transcriber: transcriber,
            decodeAudio: { data in
                await recorder.append(.decode)
                XCTAssertEqual(data, Data([1, 2, 3]))
                return WhisperPCM(
                    samples: [0.25, -0.5, 1], sampleRate: 16_000, wasTruncated: false)
            },
            extractFeatures: { samples in
                await recorder.append(.extract)
                XCTAssertEqual(samples, [0.25, -0.5, 1])
                return [0.5, -1.25]
            })

        let result = try await service.transcribe(request)

        XCTAssertEqual(result.text, "native transcript")
        XCTAssertEqual(result.language, "en")
        XCTAssertEqual(result.durationSeconds, 3.0 / 16_000.0, accuracy: 0.000_001)
        let operations = await recorder.snapshot()
        let recordedCall = await transcriber.lastCall()
        XCTAssertEqual(operations, [.decode, .extract])
        let call = try XCTUnwrap(recordedCall)
        XCTAssertEqual(call.inputFeatures, [Float16(0.5), Float16(-1.25)])
        XCTAssertEqual(call.requestedLanguage, "en")
        XCTAssertFalse(call.includeTimestamps)
    }

    func testServiceRejectsUnsupportedSemanticsBeforeAudioWork() async throws {
        let recorder = AudioPipelineRecorder()
        let transcriber = RecordingWhisperTranscriber()
        let service = OpenAIAudioTranscriptionService(
            transcriber: transcriber,
            decodeAudio: { _ in
                await recorder.append(.decode)
                return WhisperPCM(samples: [0], sampleRate: 16_000, wasTruncated: false)
            },
            extractFeatures: { _ in
                await recorder.append(.extract)
                return [0]
            })
        let requests = [
            try transcriptionRequest(fields: [("prompt", "proper noun")]),
            try transcriptionRequest(fields: [("temperature", "0.1")]),
            try transcriptionRequest(fields: [("timestamp_granularities[]", "word")]),
            try transcriptionRequest(fields: [("response_format", "srt")]),
            try transcriptionRequest(fields: [("response_format", "vtt")]),
        ]

        for request in requests {
            do {
                _ = try await service.transcribe(request)
                XCTFail("unsupported request unexpectedly reached transcription")
            } catch is OpenAIAudioTranscriptionService.ServiceError {
                // Expected.
            }
        }

        let operations = await recorder.snapshot()
        let recordedCall = await transcriber.lastCall()
        XCTAssertEqual(operations, [])
        XCTAssertNil(recordedCall)
    }

    func testServiceRejectsAudioBeyondOneNativeWindowBeforeFeatureExtraction() async throws {
        let recorder = AudioPipelineRecorder()
        let transcriber = RecordingWhisperTranscriber()
        let request = try transcriptionRequest()
        let service = OpenAIAudioTranscriptionService(
            transcriber: transcriber,
            decodeAudio: { _ in
                await recorder.append(.decode)
                return WhisperPCM(
                    samples: [0], sampleRate: 16_000, wasTruncated: true)
            },
            extractFeatures: { _ in
                await recorder.append(.extract)
                return [0]
            })

        do {
            _ = try await service.transcribe(request)
            XCTFail("truncated audio unexpectedly reached inference")
        } catch let error as OpenAIAudioTranscriptionService.ServiceError {
            XCTAssertEqual(error, .audioExceedsNativeWindow)
        }
        let operations = await recorder.snapshot()
        let recordedCall = await transcriber.lastCall()
        XCTAssertEqual(operations, [.decode])
        XCTAssertNil(recordedCall)
    }

    func testMultipartParserPreservesBinaryAudioAndRepeatedTimestampFields() throws {
        let boundary = "caix-boundary-7f31"
        let audio = Data([0x52, 0x49, 0x46, 0x46, 0x00, 0xff, 0x0d, 0x0a, 0x01, 0x02])
        let body = multipartBody(
            boundary: boundary,
            parts: [
                ("model", nil, nil, Data("openai/whisper-large-v2".utf8)),
                ("language", nil, nil, Data("en".utf8)),
                ("timestamp_granularities[]", nil, nil, Data("word".utf8)),
                ("timestamp_granularities[]", nil, nil, Data("segment".utf8)),
                ("file", "call.wav", "audio/wav", audio),
            ])

        let form = try MultipartFormData.parse(
            body,
            contentType: "multipart/form-data; charset=utf-8; boundary=\"\(boundary)\"")

        XCTAssertEqual(form.parts(named: "timestamp_granularities[]").count, 2)
        XCTAssertEqual(form.firstPart(named: "file")?.filename, "call.wav")
        XCTAssertEqual(form.firstPart(named: "file")?.contentType, "audio/wav")
        XCTAssertEqual(form.firstPart(named: "file")?.body, audio)
    }

    func testTranscriptionRequestNormalizesStandardFields() throws {
        let boundary = "caix-request"
        let audio = Data([0x00, 0x01, 0x02, 0x03])
        let body = multipartBody(
            boundary: boundary,
            parts: [
                ("file", "sample.m4a", "audio/mp4", audio),
                ("model", nil, nil, Data("openai/whisper-large-v2".utf8)),
                ("language", nil, nil, Data("en".utf8)),
                ("prompt", nil, nil, Data("Kyle, CAIX".utf8)),
                ("response_format", nil, nil, Data("verbose_json".utf8)),
                ("temperature", nil, nil, Data("0.2".utf8)),
                ("stream", nil, nil, Data("true".utf8)),
                ("timestamp_granularities[]", nil, nil, Data("word".utf8)),
            ])
        let form = try MultipartFormData.parse(
            body,
            contentType: "multipart/form-data; boundary=\(boundary)")

        let request = try OpenAIAudioTranscriptionRequest(form: form)

        XCTAssertEqual(request.model, OpenAIAudioAPI.modelID)
        XCTAssertEqual(request.audio.bytes, audio)
        XCTAssertEqual(request.audio.filename, "sample.m4a")
        XCTAssertEqual(request.audio.contentType, "audio/mp4")
        XCTAssertEqual(request.language, "en")
        XCTAssertEqual(request.prompt, "Kyle, CAIX")
        XCTAssertEqual(request.responseFormat, .verboseJSON)
        XCTAssertEqual(request.temperature, 0.2)
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.timestampGranularities, [.word])
    }

    func testTranscriptionRequestAcceptsShortModelAliasButCanonicalizesIdentity() throws {
        let form = MultipartFormData(parts: [
            .init(name: "model", body: Data("whisper-large-v2".utf8)),
            .init(name: "file", filename: "a.wav", contentType: "audio/wav", body: Data([1])),
        ])

        let request = try OpenAIAudioTranscriptionRequest(form: form)
        XCTAssertEqual(request.model, OpenAIAudioAPI.modelID)
    }

    func testTranscriptionRequestRejectsMissingOrDuplicateFile() {
        let model = MultipartFormData.Part(
            name: "model", body: Data(OpenAIAudioAPI.modelID.utf8))
        XCTAssertThrowsError(
            try OpenAIAudioTranscriptionRequest(form: MultipartFormData(parts: [model])))

        let file = MultipartFormData.Part(
            name: "file", filename: "a.wav", contentType: "audio/wav", body: Data([1]))
        XCTAssertThrowsError(
            try OpenAIAudioTranscriptionRequest(
                form: MultipartFormData(parts: [model, file, file])))
    }

    func testTranscriptionRequestRejectsUnsupportedModelAndInvalidOptions() {
        func form(_ fields: [(String, String)]) -> MultipartFormData {
            MultipartFormData(parts:
                fields.map { .init(name: $0.0, body: Data($0.1.utf8)) }
                    + [.init(name: "file", filename: "a.wav", body: Data([1]))])
        }

        XCTAssertThrowsError(
            try OpenAIAudioTranscriptionRequest(form: form([("model", "whisper-1")])))
        XCTAssertThrowsError(
            try OpenAIAudioTranscriptionRequest(form: form([
                ("model", OpenAIAudioAPI.modelID), ("response_format", "xml"),
            ])))
        XCTAssertThrowsError(
            try OpenAIAudioTranscriptionRequest(form: form([
                ("model", OpenAIAudioAPI.modelID), ("temperature", "1.1"),
            ])))
        XCTAssertThrowsError(
            try OpenAIAudioTranscriptionRequest(form: form([
                ("model", OpenAIAudioAPI.modelID), ("stream", "sometimes"),
            ])))
    }

    func testMultipartParserRejectsMalformedBodiesAndUnsafeHeaders() {
        XCTAssertThrowsError(
            try MultipartFormData.parse(Data(), contentType: "application/json"))
        XCTAssertThrowsError(
            try MultipartFormData.parse(
                Data("--missing\r\n".utf8),
                contentType: "multipart/form-data; boundary=missing"))

        let boundary = "bad-header"
        let body = Data(
            "--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"\r\nX-Evil: yes\r\n\r\na\r\n--\(boundary)--\r\n".utf8)
        XCTAssertThrowsError(
            try MultipartFormData.parse(
                body, contentType: "multipart/form-data; boundary=\(boundary)"))
    }

    private func multipartBody(
        boundary: String,
        parts: [(name: String, filename: String?, contentType: String?, body: Data)]
    ) -> Data {
        var data = Data()
        for part in parts {
            data.append(Data("--\(boundary)\r\n".utf8))
            var disposition = "Content-Disposition: form-data; name=\"\(part.name)\""
            if let filename = part.filename { disposition += "; filename=\"\(filename)\"" }
            data.append(Data("\(disposition)\r\n".utf8))
            if let contentType = part.contentType {
                data.append(Data("Content-Type: \(contentType)\r\n".utf8))
            }
            data.append(Data("\r\n".utf8))
            data.append(part.body)
            data.append(Data("\r\n".utf8))
        }
        data.append(Data("--\(boundary)--\r\n".utf8))
        return data
    }

    private func transcriptionRequest(
        fields: [(String, String)] = []
    ) throws -> OpenAIAudioTranscriptionRequest {
        let parts = [
            MultipartFormData.Part(
                name: "model", body: Data(OpenAIAudioAPI.modelID.utf8)),
            MultipartFormData.Part(
                name: "file", filename: "call.wav", contentType: "audio/wav",
                body: Data([1, 2, 3])),
        ] + fields.map { MultipartFormData.Part(name: $0.0, body: Data($0.1.utf8)) }
        return try OpenAIAudioTranscriptionRequest(form: MultipartFormData(parts: parts))
    }
}

private actor AudioPipelineRecorder {
    enum Operation: Equatable { case decode, extract }
    private var operations: [Operation] = []

    func append(_ operation: Operation) { operations.append(operation) }
    func snapshot() -> [Operation] { operations }
}

private actor RecordingWhisperTranscriber: WhisperTranscribing {
    struct Call: Sendable {
        var inputFeatures: [Float16]
        var requestedLanguage: String?
        var includeTimestamps: Bool
    }

    private var call: Call?

    func transcribe(
        inputFeatures: [Float16],
        requestedLanguage: String?,
        includeTimestamps: Bool,
        onTextToken: (@Sendable (Int32) -> Void)?
    ) async throws -> WhisperResidentEngine.Result {
        call = Call(
            inputFeatures: inputFeatures,
            requestedLanguage: requestedLanguage,
            includeTimestamps: includeTimestamps)
        return WhisperResidentEngine.Result(
            text: "native transcript",
            textTokenIDs: [100],
            language: requestedLanguage ?? "en",
            languageTokenID: 50_259,
            reachedEndToken: true,
            wasTruncated: false,
            timings: .init(
                queueSeconds: 0,
                encodeSeconds: 0.1,
                crossKVLoadSeconds: 0.01,
                decodeSeconds: 0.2,
                totalSeconds: 0.31))
    }

    func lastCall() -> Call? { call }
}
