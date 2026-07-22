import CryptoKit
import Foundation

/// Authenticated Whisper-large-v2 generation rules projected into a small deterministic decoder.
/// The surrounding source lock authenticates `generation_config.json`; this type additionally
/// fails closed unless its contents match the fixed native cache/vocabulary ABI.
public struct WhisperDecodingPolicy: Sendable {
    public static let vocabularySize = 51_865

    public enum PolicyError: Error, Sendable, Equatable, CustomStringConvertible {
        case invalidConfiguration(String)
        case generationConfigurationDigestMismatch
        case timestampsUnsupported
        case unsupportedLanguage(String)
        case invalidLanguageToken(Int32)
        case invalidLogits

        public var description: String {
            switch self {
            case .invalidConfiguration(let reason): return "invalid Whisper generation config: \(reason)"
            case .generationConfigurationDigestMismatch:
                return "Whisper generation config does not match the resident model lock"
            case .timestampsUnsupported:
                return "Whisper timestamp decoding is not supported"
            case .unsupportedLanguage(let language): return "unsupported Whisper language '\(language)'"
            case .invalidLanguageToken(let token): return "invalid Whisper language token \(token)"
            case .invalidLogits: return "Whisper decoder returned invalid logits"
            }
        }
    }

    private struct Source: Decodable {
        var decoderStartTokenID: Int
        var eosTokenID: Int
        var padTokenID: Int
        var forcedDecoderIDs: [[Int?]]
        var suppressTokens: [Int]
        var beginSuppressTokens: [Int]
        var noTimestampsTokenID: Int
        var maximumLength: Int
        var languageToID: [String: Int]
        var taskToID: [String: Int]

        enum CodingKeys: String, CodingKey {
            case decoderStartTokenID = "decoder_start_token_id"
            case eosTokenID = "eos_token_id"
            case padTokenID = "pad_token_id"
            case forcedDecoderIDs = "forced_decoder_ids"
            case suppressTokens = "suppress_tokens"
            case beginSuppressTokens = "begin_suppress_tokens"
            case noTimestampsTokenID = "no_timestamps_token_id"
            case maximumLength = "max_length"
            case languageToID = "lang_to_id"
            case taskToID = "task_to_id"
        }
    }

    public let decoderStartTokenID: Int32
    public let endTokenID: Int32
    public let noTimestampsTokenID: Int32
    public let transcribeTokenID: Int32
    public let maximumSequenceLength: Int
    private let languageToID: [String: Int32]
    private let idToLanguage: [Int32: String]
    private let languageIDs: Set<Int32>
    private let suppressed: Set<Int32>
    private let beginSuppressed: Set<Int32>

    public static let canonicalGenerationConfigSHA256 =
        ResidentModelLock.approvedWhisperGenerationConfigSHA256

    public init(authenticatedContentsOf url: URL) throws {
        let data: Data
        do {
            data = try BoundedRegularFileReader.read(
                url,
                maximumBytes: 256 * 1024)
        } catch {
            throw PolicyError.invalidConfiguration("cannot read a bounded regular file: \(error)")
        }
        try self.init(
            authenticatedData: data,
            expectedSHA256: Self.canonicalGenerationConfigSHA256)
    }

    init(authenticatedData data: Data, expectedSHA256: String) throws {
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == expectedSHA256 else {
            throw PolicyError.generationConfigurationDigestMismatch
        }
        let source: Source
        do {
            source = try JSONDecoder().decode(Source.self, from: data)
        } catch {
            throw PolicyError.invalidConfiguration("cannot decode JSON: \(error)")
        }
        guard source.decoderStartTokenID == 50_258 else {
            throw PolicyError.invalidConfiguration("decoder_start_token_id must be 50258")
        }
        guard source.eosTokenID == 50_257, source.padTokenID == source.eosTokenID else {
            throw PolicyError.invalidConfiguration("EOS and padding IDs must both be 50257")
        }
        guard source.noTimestampsTokenID == 50_363 else {
            throw PolicyError.invalidConfiguration("no_timestamps_token_id must be 50363")
        }
        guard source.maximumLength == 448 else {
            throw PolicyError.invalidConfiguration("max_length must match the 448-slot state cache")
        }
        guard source.taskToID["transcribe"] == 50_359 else {
            throw PolicyError.invalidConfiguration("transcribe task token must be 50359")
        }
        guard source.forcedDecoderIDs.contains(where: { pair in
            pair.count == 2 && pair[0] == 2 && pair[1] == 50_359
        }) else {
            throw PolicyError.invalidConfiguration("forced decoder IDs must include transcribe at position 2")
        }
        guard !source.languageToID.isEmpty else {
            throw PolicyError.invalidConfiguration("language token map is empty")
        }
        let allIDs = source.languageToID.values
            + source.suppressTokens
            + source.beginSuppressTokens
            + Array(source.taskToID.values)
            + [source.decoderStartTokenID, source.eosTokenID, source.noTimestampsTokenID]
        guard allIDs.allSatisfy({ (0..<Self.vocabularySize).contains($0) }) else {
            throw PolicyError.invalidConfiguration("one or more token IDs are outside the vocabulary")
        }

        decoderStartTokenID = Int32(source.decoderStartTokenID)
        endTokenID = Int32(source.eosTokenID)
        noTimestampsTokenID = Int32(source.noTimestampsTokenID)
        transcribeTokenID = Int32(source.taskToID["transcribe"]!)
        maximumSequenceLength = source.maximumLength
        languageToID = source.languageToID.reduce(into: [:]) { result, item in
            result[item.key.lowercased()] = Int32(item.value)
        }
        idToLanguage = source.languageToID.reduce(into: [:]) { result, item in
            let key = item.key.lowercased()
            let code = key.hasPrefix("<|") && key.hasSuffix("|>")
                ? String(key.dropFirst(2).dropLast(2)) : key
            result[Int32(item.value)] = code
        }
        languageIDs = Set(source.languageToID.values.map(Int32.init))
        suppressed = Set(source.suppressTokens.map(Int32.init))
        beginSuppressed = Set(source.beginSuppressTokens.map(Int32.init))
    }

    public func languageTokenID(for language: String) -> Int32? {
        let normalized = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.hasPrefix("<|"), normalized.hasSuffix("|>") {
            return languageToID[normalized]
        }
        return languageToID["<|\(normalized)|>"]
    }

    public func requireLanguageTokenID(for language: String) throws -> Int32 {
        guard let token = languageTokenID(for: language) else {
            throw PolicyError.unsupportedLanguage(language)
        }
        return token
    }

    public func languageCode(for tokenID: Int32) throws -> String {
        guard let language = idToLanguage[tokenID] else {
            throw PolicyError.invalidLanguageToken(tokenID)
        }
        return language
    }

    public func detectLanguageToken(in logits: [Float]) throws -> Int32 {
        guard logits.count == Self.vocabularySize else { throw PolicyError.invalidLogits }
        var selection: (id: Int32, value: Float)?
        for token in languageIDs {
            let value = logits[Int(token)]
            guard value.isFinite else { continue }
            if selection == nil || value > selection!.value
                || (value == selection!.value && token < selection!.id)
            {
                selection = (token, value)
            }
        }
        guard let selection else { throw PolicyError.invalidLogits }
        return selection.id
    }

    public func forcedPrefix(
        languageTokenID: Int32,
        includeTimestamps: Bool
    ) throws -> [Int32] {
        guard !includeTimestamps else { throw PolicyError.timestampsUnsupported }
        guard languageIDs.contains(languageTokenID) else {
            throw PolicyError.invalidLanguageToken(languageTokenID)
        }
        var prefix = [decoderStartTokenID, languageTokenID, transcribeTokenID]
        if !includeTimestamps { prefix.append(noTimestampsTokenID) }
        return prefix
    }

    public func maximumTextTokenCount(includeTimestamps: Bool) -> Int {
        maximumSequenceLength - (includeTimestamps ? 3 : 4)
    }

    public func greedyTextToken(
        logits: [Float],
        textPosition: Int,
        includeTimestamps: Bool
    ) throws -> Int32 {
        guard logits.count == Self.vocabularySize, textPosition >= 0 else {
            throw PolicyError.invalidLogits
        }
        guard !includeTimestamps else { throw PolicyError.timestampsUnsupported }
        var selection: (id: Int32, value: Float)?
        for rawID in 0..<Self.vocabularySize {
            let token = Int32(rawID)
            if suppressed.contains(token) { continue }
            if textPosition == 0 && beginSuppressed.contains(token) { continue }
            if !includeTimestamps && token >= noTimestampsTokenID { continue }
            let value = logits[rawID]
            guard value.isFinite else { continue }
            if selection == nil || value > selection!.value
                || (value == selection!.value && token < selection!.id)
            {
                selection = (token, value)
            }
        }
        guard let selection else { throw PolicyError.invalidLogits }
        return selection.id
    }
}
