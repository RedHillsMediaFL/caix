import Foundation

/// Small, strict multipart/form-data reader for the OpenAI audio endpoint.
///
/// The HTTP handler bounds the request body before calling this parser. Parts stay in memory so
/// request audio is never copied to a temporary file. The parser accepts only the two headers a
/// browser or OpenAI client needs for form-data uploads and fails closed on ambiguous framing.
struct MultipartFormData: Sendable, Equatable {
    struct Part: Sendable, Equatable {
        var name: String
        var filename: String?
        var contentType: String?
        var body: Data

        init(name: String, filename: String? = nil, contentType: String? = nil, body: Data) {
            self.name = name
            self.filename = filename
            self.contentType = contentType
            self.body = body
        }
    }

    enum ParseError: Error, Sendable, Equatable, CustomStringConvertible {
        case invalidContentType
        case invalidBoundary
        case malformedBody
        case tooManyParts
        case headersTooLarge
        case invalidHeader
        case unsupportedHeader(String)
        case missingDisposition
        case missingName

        var description: String {
            switch self {
            case .invalidContentType: return "content type must be multipart/form-data"
            case .invalidBoundary: return "multipart boundary is missing or invalid"
            case .malformedBody: return "multipart body framing is malformed"
            case .tooManyParts: return "multipart body has too many parts"
            case .headersTooLarge: return "multipart part headers are too large"
            case .invalidHeader: return "multipart part contains an invalid header"
            case .unsupportedHeader(let name): return "unsupported multipart header '\(name)'"
            case .missingDisposition: return "multipart part is missing Content-Disposition"
            case .missingName: return "multipart part is missing a form field name"
            }
        }
    }

    var parts: [Part]

    func parts(named name: String) -> [Part] { parts.filter { $0.name == name } }
    func firstPart(named name: String) -> Part? { parts.first { $0.name == name } }

    static func parse(
        _ data: Data,
        contentType: String,
        maximumParts: Int = 32,
        maximumHeaderBytes: Int = 16 * 1024
    ) throws -> Self {
        let boundary = try parseBoundary(from: contentType)
        let delimiter = Data("--\(boundary)".utf8)
        let nextDelimiterPrefix = Data("\r\n--\(boundary)".utf8)
        let headerTerminator = Data("\r\n\r\n".utf8)
        let crlf = Data("\r\n".utf8)
        let closing = Data("--".utf8)

        guard data.count >= delimiter.count + closing.count,
              data.starts(with: delimiter)
        else { throw ParseError.malformedBody }

        var cursor = delimiter.count
        var parts: [Part] = []
        while true {
            if data.hasBytes(closing, at: cursor) {
                cursor += closing.count
                if data.hasBytes(crlf, at: cursor) { cursor += crlf.count }
                guard cursor == data.count else { throw ParseError.malformedBody }
                return Self(parts: parts)
            }
            guard data.hasBytes(crlf, at: cursor) else { throw ParseError.malformedBody }
            cursor += crlf.count

            guard let headerRange = data.range(
                of: headerTerminator,
                options: [],
                in: cursor..<min(data.count, cursor + maximumHeaderBytes + headerTerminator.count))
            else {
                if data.count - cursor >= maximumHeaderBytes { throw ParseError.headersTooLarge }
                throw ParseError.malformedBody
            }
            let headerData = data[cursor..<headerRange.lowerBound]
            let fields = try parseHeaders(Data(headerData))
            cursor = headerRange.upperBound

            guard let markerRange = data.range(
                of: nextDelimiterPrefix, options: [], in: cursor..<data.count)
            else { throw ParseError.malformedBody }
            let body = Data(data[cursor..<markerRange.lowerBound])
            cursor = markerRange.lowerBound + crlf.count

            guard data.hasBytes(delimiter, at: cursor) else { throw ParseError.malformedBody }
            cursor += delimiter.count
            parts.append(try makePart(fields: fields, body: body))
            guard parts.count <= maximumParts else { throw ParseError.tooManyParts }
        }
    }

    private static func parseBoundary(from contentType: String) throws -> String {
        let pieces = splitParameters(contentType)
        guard pieces.first?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                == "multipart/form-data"
        else { throw ParseError.invalidContentType }
        var boundary: String?
        for parameter in pieces.dropFirst() {
            let pair = parameter.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2 else { continue }
            if pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "boundary" {
                guard boundary == nil else { throw ParseError.invalidBoundary }
                boundary = unquote(String(pair[1]).trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        guard let boundary,
              (1...70).contains(boundary.utf8.count),
              !boundary.contains("\r"), !boundary.contains("\n"),
              boundary.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value < 0x7f })
        else { throw ParseError.invalidBoundary }
        return boundary
    }

    private static func parseHeaders(_ data: Data) throws -> [String: String] {
        guard let text = String(data: data, encoding: .utf8) else { throw ParseError.invalidHeader }
        var fields: [String: String] = [:]
        for line in text.components(separatedBy: "\r\n") where !line.isEmpty {
            guard !line.first.map({ $0 == " " || $0 == "\t" })!,
                  let colon = line.firstIndex(of: ":")
            else { throw ParseError.invalidHeader }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            guard name == "content-disposition" || name == "content-type" else {
                throw ParseError.unsupportedHeader(name)
            }
            guard fields[name] == nil, !value.isEmpty else { throw ParseError.invalidHeader }
            fields[name] = value
        }
        return fields
    }

    private static func makePart(fields: [String: String], body: Data) throws -> Part {
        guard let disposition = fields["content-disposition"] else {
            throw ParseError.missingDisposition
        }
        let pieces = splitParameters(disposition)
        guard pieces.first?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                == "form-data"
        else { throw ParseError.invalidHeader }
        var parameters: [String: String] = [:]
        for parameter in pieces.dropFirst() {
            let pair = parameter.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2 else { throw ParseError.invalidHeader }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard key == "name" || key == "filename", parameters[key] == nil else {
                throw ParseError.invalidHeader
            }
            parameters[key] = unquote(String(pair[1]).trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard let name = parameters["name"], !name.isEmpty,
              name.utf8.count <= 128, !name.contains("\0")
        else { throw ParseError.missingName }
        let filename = parameters["filename"]
        guard filename == nil || (filename!.utf8.count <= 1024 && !filename!.contains("\0")) else {
            throw ParseError.invalidHeader
        }
        return Part(
            name: name,
            filename: filename,
            contentType: fields["content-type"],
            body: body)
    }

    private static func splitParameters(_ value: String) -> [String] {
        var pieces: [String] = []
        var current = ""
        var quoted = false
        var escaped = false
        for character in value {
            if escaped {
                current.append(character)
                escaped = false
            } else if quoted && character == "\\" {
                current.append(character)
                escaped = true
            } else if character == "\"" {
                current.append(character)
                quoted.toggle()
            } else if character == ";" && !quoted {
                pieces.append(current)
                current = ""
            } else {
                current.append(character)
            }
        }
        pieces.append(current)
        return pieces
    }

    private static func unquote(_ value: String) -> String {
        guard value.count >= 2, value.first == "\"", value.last == "\"" else { return value }
        var result = ""
        var escaped = false
        for character in value.dropFirst().dropLast() {
            if escaped {
                result.append(character)
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else {
                result.append(character)
            }
        }
        if escaped { result.append("\\") }
        return result
    }
}

private extension Data {
    func hasBytes(_ bytes: Data, at offset: Int) -> Bool {
        guard offset >= 0, offset + bytes.count <= count else { return false }
        return self[offset..<(offset + bytes.count)].elementsEqual(bytes)
    }
}
