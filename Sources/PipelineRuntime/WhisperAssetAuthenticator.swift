import CryptoKit
import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

extension WhisperResidentEngine {
    /// Authentication failures raised before resident Whisper specialization or tokenizer parsing.
    public enum LoadError: Error, Sendable, Equatable, CustomStringConvertible {
        case invalidAssetDirectory
        case assetEntriesMismatch
        case assetFileInvalid(filename: String)
        case assetFileTooLarge(filename: String)
        case mainHashMismatch
        case manifestMismatch
        case invalidMetadataDirectory
        case metadataFileInvalid(filename: String)
        case metadataFileTooLarge(filename: String)
        case metadataDigestMismatch(filename: String)
        case metadataJSONInvalid(filename: String)

        public var description: String {
            switch self {
            case .invalidAssetDirectory:
                return "Whisper asset must be a non-symlink .aimodel directory"
            case .assetEntriesMismatch:
                return "Whisper asset entries do not match the exact CAIX manifest contract"
            case .assetFileInvalid(let filename):
                return "Whisper asset file is missing, unreadable, or non-regular: \(filename)"
            case .assetFileTooLarge(let filename):
                return "Whisper asset control file exceeds its size bound: \(filename)"
            case .mainHashMismatch:
                return "Whisper main.hash does not match main.mlirb"
            case .manifestMismatch:
                return "Whisper caix-manifest.json is noncanonical or differs from the pinned contract"
            case .invalidMetadataDirectory:
                return "Whisper tokenizer metadata must be a non-symlink directory"
            case .metadataFileInvalid(let filename):
                return "Whisper metadata file is missing, unreadable, or non-regular: \(filename)"
            case .metadataFileTooLarge(let filename):
                return "Whisper metadata file exceeds its size bound: \(filename)"
            case .metadataDigestMismatch(let filename):
                return "Whisper metadata digest differs from ResidentModelLock: \(filename)"
            case .metadataJSONInvalid(let filename):
                return "Whisper authenticated metadata is not a JSON object: \(filename)"
            }
        }
    }
}

struct WhisperAuthenticatedAsset: Sendable, Equatable {
    var assetURL: URL
    var assetSizeBytes: UInt64
    var mainSizeBytes: UInt64
    var mainSHA256: String
}

struct WhisperAuthenticatedMetadata: Sendable, Equatable {
    var generationConfiguration: Data
    var tokenizer: Data
    var tokenizerConfiguration: Data
}

/// Native, descriptor-first authentication for the pinned Whisper large-v2 runtime bundle.
///
/// The Core AI asset is accepted only when all four entries match the exact Task 4 contract from
/// commit `08e95d5`. Tokenizer inputs are read into bounded snapshots and authenticated against a
/// validated `ResidentModelLock`, so downstream parsers never reopen mutable paths.
enum WhisperAssetAuthenticator {
    private static let assetEntries = Set([
        "caix-manifest.json", "main.hash", "main.mlirb", "metadata.json",
    ])
    private static let maximumManifestBytes = 128 * 1_024
    private static let maximumAssetMetadataBytes = 1_024 * 1_024
    private static let defaultMaximumMetadataBytes = 16 * 1_024 * 1_024
    private static let maximumTokenizerBytes = 64 * 1_024 * 1_024
    private static let streamChunkBytes = 1_024 * 1_024

    static func authenticateAsset(at assetURL: URL) throws -> WhisperAuthenticatedAsset {
        guard assetURL.pathExtension == "aimodel", isNonSymlinkDirectory(assetURL) else {
            throw WhisperResidentEngine.LoadError.invalidAssetDirectory
        }
        let entries: Set<String>
        do {
            entries = Set(try FileManager.default.contentsOfDirectory(atPath: assetURL.path))
        } catch {
            throw WhisperResidentEngine.LoadError.invalidAssetDirectory
        }
        guard entries == assetEntries else {
            throw WhisperResidentEngine.LoadError.assetEntriesMismatch
        }

        let metadata = try readAssetControlFile(
            assetURL.appendingPathComponent("metadata.json"),
            filename: "metadata.json",
            maximumBytes: maximumAssetMetadataBytes,
            minimumBytes: 0)
        let main = try streamMainIdentity(
            assetURL.appendingPathComponent("main.mlirb"))
        let rawHash = try readAssetControlFile(
            assetURL.appendingPathComponent("main.hash"),
            filename: "main.hash",
            maximumBytes: 32,
            exactBytes: 32)
        guard rawHash == data(fromLowercaseHex: main.sha256) else {
            throw WhisperResidentEngine.LoadError.mainHashMismatch
        }
        let manifest = try readAssetControlFile(
            assetURL.appendingPathComponent("caix-manifest.json"),
            filename: "caix-manifest.json",
            maximumBytes: maximumManifestBytes)
        guard manifest == expectedManifest(main: main) else {
            throw WhisperResidentEngine.LoadError.manifestMismatch
        }

        return WhisperAuthenticatedAsset(
            assetURL: assetURL,
            assetSizeBytes: UInt64(metadata.count)
                + main.sizeBytes
                + UInt64(rawHash.count)
                + UInt64(manifest.count),
            mainSizeBytes: main.sizeBytes,
            mainSHA256: main.sha256)
    }

    static func metadataDigests(for lock: ResidentModelLock) -> [String: String] {
        let metadata = lock.speech.metadata
        return [
            "added_tokens.json": metadata.addedTokensSHA256,
            "config.json": metadata.configSHA256,
            "generation_config.json": metadata.generationConfigSHA256,
            "merges.txt": metadata.mergesSHA256,
            "normalizer.json": metadata.normalizerSHA256,
            "preprocessor_config.json": metadata.preprocessorConfigSHA256,
            "tokenizer.json": metadata.tokenizerJSONSHA256,
            "tokenizer_config.json": metadata.tokenizerConfigSHA256,
            "vocab.json": metadata.vocabularySHA256,
        ]
    }

    static func authenticateMetadata(
        at directory: URL,
        lock: ResidentModelLock
    ) throws -> WhisperAuthenticatedMetadata {
        try authenticateMetadata(
            at: directory,
            expectedDigests: metadataDigests(for: lock))
    }

    static func authenticateMetadata(
        at directory: URL,
        expectedDigests: [String: String]
    ) throws -> WhisperAuthenticatedMetadata {
        guard isNonSymlinkDirectory(directory) else {
            throw WhisperResidentEngine.LoadError.invalidMetadataDirectory
        }
        let retainedNames = Set([
            "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        ])
        guard retainedNames.isSubset(of: Set(expectedDigests.keys)) else {
            let missing = retainedNames.subtracting(expectedDigests.keys).sorted().first!
            throw WhisperResidentEngine.LoadError.metadataFileInvalid(filename: missing)
        }

        var retained: [String: Data] = [:]
        for filename in expectedDigests.keys.sorted() {
            let maximumBytes = filename == "tokenizer.json"
                ? maximumTokenizerBytes : defaultMaximumMetadataBytes
            let data: Data
            do {
                data = try BoundedRegularFileReader.read(
                    directory.appendingPathComponent(filename),
                    maximumBytes: maximumBytes)
            } catch BoundedRegularFileReaderError.tooLarge {
                throw WhisperResidentEngine.LoadError.metadataFileTooLarge(filename: filename)
            } catch {
                throw WhisperResidentEngine.LoadError.metadataFileInvalid(filename: filename)
            }
            guard sha256(data) == expectedDigests[filename] else {
                throw WhisperResidentEngine.LoadError.metadataDigestMismatch(filename: filename)
            }
            if retainedNames.contains(filename) { retained[filename] = data }
        }

        return WhisperAuthenticatedMetadata(
            generationConfiguration: retained["generation_config.json"]!,
            tokenizer: retained["tokenizer.json"]!,
            tokenizerConfiguration: retained["tokenizer_config.json"]!)
    }

    private struct StreamedIdentity {
        var sizeBytes: UInt64
        var sha256: String
    }

    private struct FileSnapshot: Equatable {
        var device: UInt64
        var inode: UInt64
        var mode: UInt32
        var size: UInt64
    }

    private static func streamMainIdentity(_ url: URL) throws -> StreamedIdentity {
        #if canImport(Darwin) || canImport(Glibc)
        let descriptor: Int32
        #if canImport(Darwin)
        descriptor = url.path.withCString {
            Darwin.open($0, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        }
        #else
        descriptor = url.path.withCString {
            Glibc.open($0, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        }
        #endif
        guard descriptor >= 0 else {
            throw WhisperResidentEngine.LoadError.assetFileInvalid(filename: "main.mlirb")
        }
        defer {
            #if canImport(Darwin)
            _ = Darwin.close(descriptor)
            #else
            _ = Glibc.close(descriptor)
            #endif
        }
        let before = try regularFileSnapshot(descriptor, filename: "main.mlirb")
        var hasher = SHA256()
        var consumed: UInt64 = 0
        var buffer = [UInt8](repeating: 0, count: streamChunkBytes)
        while true {
            let count: Int
            #if canImport(Darwin)
            count = buffer.withUnsafeMutableBytes {
                Darwin.read(descriptor, $0.baseAddress, $0.count)
            }
            #else
            count = buffer.withUnsafeMutableBytes {
                Glibc.read(descriptor, $0.baseAddress, $0.count)
            }
            #endif
            if count < 0 {
                #if canImport(Darwin) || canImport(Glibc)
                if errno == EINTR { continue }
                #endif
                throw WhisperResidentEngine.LoadError.assetFileInvalid(filename: "main.mlirb")
            }
            if count == 0 { break }
            let chunk = Data(buffer[0..<count])
            hasher.update(data: chunk)
            consumed += UInt64(count)
        }
        let after = try regularFileSnapshot(descriptor, filename: "main.mlirb")
        guard before == after, consumed == before.size else {
            throw WhisperResidentEngine.LoadError.assetFileInvalid(filename: "main.mlirb")
        }
        return StreamedIdentity(
            sizeBytes: consumed,
            sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined())
        #else
        throw WhisperResidentEngine.LoadError.assetFileInvalid(filename: "main.mlirb")
        #endif
    }

    private static func regularFileSnapshot(
        _ descriptor: Int32,
        filename: String
    ) throws -> FileSnapshot {
        #if canImport(Darwin)
        var info = Darwin.stat()
        guard Darwin.fstat(descriptor, &info) == 0,
            (info.st_mode & mode_t(S_IFMT)) == mode_t(S_IFREG),
            info.st_size >= 0
        else {
            throw WhisperResidentEngine.LoadError.assetFileInvalid(filename: filename)
        }
        #elseif canImport(Glibc)
        var info = Glibc.stat()
        guard Glibc.fstat(descriptor, &info) == 0,
            (info.st_mode & mode_t(S_IFMT)) == mode_t(S_IFREG),
            info.st_size >= 0
        else {
            throw WhisperResidentEngine.LoadError.assetFileInvalid(filename: filename)
        }
        #else
        throw WhisperResidentEngine.LoadError.assetFileInvalid(filename: filename)
        #endif
        return FileSnapshot(
            device: UInt64(info.st_dev),
            inode: UInt64(info.st_ino),
            mode: UInt32(info.st_mode),
            size: UInt64(info.st_size))
    }

    private static func readAssetControlFile(
        _ url: URL,
        filename: String,
        maximumBytes: Int,
        minimumBytes: Int = 1,
        exactBytes: Int? = nil
    ) throws -> Data {
        do {
            return try BoundedRegularFileReader.read(
                url,
                maximumBytes: maximumBytes,
                minimumBytes: minimumBytes,
                exactBytes: exactBytes)
        } catch BoundedRegularFileReaderError.tooLarge {
            throw WhisperResidentEngine.LoadError.assetFileTooLarge(filename: filename)
        } catch {
            throw WhisperResidentEngine.LoadError.assetFileInvalid(filename: filename)
        }
    }

    private static func isNonSymlinkDirectory(_ url: URL) -> Bool {
        #if canImport(Darwin)
        var info = Darwin.stat()
        let result = url.path.withCString { Darwin.lstat($0, &info) }
        #elseif canImport(Glibc)
        var info = Glibc.stat()
        let result = url.path.withCString { Glibc.lstat($0, &info) }
        #else
        return false
        #endif
        return result == 0 && (info.st_mode & mode_t(S_IFMT)) == mode_t(S_IFDIR)
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func data(fromLowercaseHex value: String) -> Data? {
        guard value.utf8.count.isMultiple(of: 2) else { return nil }
        var data = Data(capacity: value.utf8.count / 2)
        var index = value.startIndex
        while index < value.endIndex {
            let end = value.index(index, offsetBy: 2)
            guard let byte = UInt8(value[index..<end], radix: 16) else { return nil }
            data.append(byte)
            index = end
        }
        return data
    }

    private static func expectedManifest(main: StreamedIdentity) -> Data {
        let cross: [String: Any] = [
            "dtype": "float16", "shape": [32, 1, 20, 1_500, 64],
        ]
        let selfCache: [String: Any] = [
            "dtype": "float16", "shape": [32, 1, 20, 448, 64],
        ]
        let intScalar: [String: Any] = ["dtype": "int32", "shape": [1]]
        let payload: [String: Any] = [
            "schema": "caix.whisper-asset.v1",
            "abi": [
                "schema": "caix.whisper-split.v2",
                "strategy": "explicit_cross_kv_bridge",
                "call_order": ["encode", "load_cross_kv", "decode_step*"],
                "entrypoints": [
                    "encoder": "encode",
                    "decoder_load": "load_cross_kv",
                    "decoder_step": "decode_step",
                ],
                "functions": [
                    "encode": [
                        "inputs": ["input_features"],
                        "outputs": ["cross_key_payload", "cross_value_payload"],
                        "states": [],
                    ],
                    "load_cross_kv": [
                        "inputs": ["cross_key_payload", "cross_value_payload"],
                        "outputs": ["load_status"],
                        "states": ["cross_key_cache", "cross_value_cache", "cross_ready"],
                    ],
                    "decode_step": [
                        "inputs": ["token_id"],
                        "outputs": ["logits", "decode_status"],
                        "states": [
                            "cross_key_cache", "cross_value_cache", "self_key_cache",
                            "self_value_cache", "position", "cross_ready",
                        ],
                    ],
                ],
                "statuses": ["success": 1, "invalid_state": 0],
                "tensors": [
                    "input_features": ["dtype": "float16", "shape": [1, 80, 3_000]],
                    "cross_key_cache": cross,
                    "cross_value_cache": cross,
                    "self_key_cache": selfCache,
                    "self_value_cache": selfCache,
                    "token_id": ["dtype": "int32", "shape": [1, 1]],
                    "position": intScalar,
                    "cross_ready": intScalar,
                    "load_status": intScalar,
                    "logits": ["dtype": "float16", "shape": [1, 1, 51_865]],
                    "decode_status": intScalar,
                    "cross_key_payload": cross,
                    "cross_value_payload": cross,
                ],
            ],
            "source": [
                "repository": "openai/whisper-large-v2",
                "revision": "ae4642769ce2ad8fc292556ccea8e901f1530655",
                "weights": [
                    "path": "model.safetensors",
                    "size_bytes": 6_173_370_152,
                    "sha256": "57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b",
                ],
            ],
            "authoring": [
                "stack": [
                    "coreai-core": "1.0.0b2",
                    "coreai-opt": "0.2.0",
                    "coreai-torch": "0.4.1",
                    "torch": "2.9.0",
                    "transformers": "4.57.6",
                ],
                "coreai_models": [
                    "repository": "https://github.com/kylejfrost/coreai-models.git",
                    "revision": "e666cdc9848fd17f41e43504bc574c8964812c9e",
                    "python_root": "python/src",
                    "package_subtree": "python/src/coreai_models",
                    "package_tree": "b2803957eee13084d06924cfc567a770379234ae",
                ],
            ],
            "main": [
                "path": "main.mlirb",
                "size_bytes": Int64(main.sizeBytes),
                "sha256": main.sha256,
            ],
        ]
        return try! JSONSerialization.data(
            withJSONObject: payload,
            options: [.sortedKeys, .withoutEscapingSlashes])
    }
}

#if COREAI_RUNTIME

extension WhisperResidentEngine {
    /// Authenticate and load one resident Whisper-large-v2 engine.
    ///
    /// Every model and tokenizer byte used by the runtime is authenticated before Core AI is
    /// asked to specialize the asset. The returned actor retains one specialized model and creates
    /// bounded, request-exclusive decoder state through its admission gate.
    public static func load(
        assetURL: URL,
        tokenizerDirectory: URL,
        modelLockURL: URL,
        maximumQueuedRequests: Int = defaultMaximumQueuedRequests
    ) async throws -> WhisperResidentEngine {
        let lock = try ResidentModelLock.load(from: modelLockURL)
        let metadata = try WhisperAssetAuthenticator.authenticateMetadata(
            at: tokenizerDirectory,
            lock: lock)
        let policy = try WhisperDecodingPolicy(
            authenticatedData: metadata.generationConfiguration,
            expectedSHA256: lock.speech.metadata.generationConfigSHA256)
        let textDecoder = try WhisperTokenizerDecoder.load(
            authenticatedTokenizerJSON: metadata.tokenizer,
            authenticatedTokenizerConfigurationJSON: metadata.tokenizerConfiguration)
        let authenticatedAsset = try WhisperAssetAuthenticator.authenticateAsset(at: assetURL)
        let factory = try await WhisperCoreAIModelFactory.specialize(
            authenticatedAsset: authenticatedAsset)
        return WhisperResidentEngine(
            factory: factory,
            policy: policy,
            textDecoder: textDecoder,
            maximumQueuedRequests: maximumQueuedRequests)
    }
}

#endif
