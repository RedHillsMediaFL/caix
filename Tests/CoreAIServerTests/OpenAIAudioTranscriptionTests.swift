import XCTest

@testable import CoreAIServer

final class OpenAIAudioTranscriptionTests: XCTestCase {
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
}
