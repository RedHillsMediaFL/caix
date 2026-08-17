import Foundation

/// Explicit identity for one text staged bundle served outside normal export discovery.
///
/// Supplying either CLI input requires the other. The bundle path is intentionally exact: CAIX
/// does not search for a staged Gemma artifact on the caller's behalf.
public struct PrimaryStagedBundleConfiguration: Sendable, Equatable {
    public enum ConfigurationError: Error, Sendable, Equatable, CustomStringConvertible {
        case incomplete(missingFlags: [String])
        case emptyValue(flag: String)
        case missingBundle(path: String)
        case notStagedBundle(path: String)
        case aliasCollision(alias: String, conflictingModel: String)

        public var description: String {
            switch self {
            case .incomplete(let missingFlags):
                return "primary staged serving requires both --primary-staged-bundle and --primary-model-id; missing \(missingFlags.joined(separator: ", "))"
            case .emptyValue(let flag):
                return "\(flag) requires a non-empty value"
            case .missingBundle(let path):
                return "primary staged bundle does not exist or is not a directory: \(path)"
            case .notStagedBundle(let path):
                return "primary staged bundle is missing a valid stage-manifest.json: \(path)"
            case .aliasCollision(let alias, let conflictingModel):
                return "primary staged model alias '\(alias)' conflicts with discovered bundle '\(conflictingModel)'"
            }
        }
    }

    public let bundleURL: URL
    public let modelID: String

    private init(bundleURL: URL, modelID: String) {
        self.bundleURL = bundleURL
        self.modelID = modelID
    }

    /// Return `nil` when no primary bundle is requested, otherwise validate a complete staged
    /// bundle selection before the server starts listening.
    public static func resolve(
        bundlePath: String?,
        modelID: String?
    ) throws -> PrimaryStagedBundleConfiguration? {
        guard bundlePath != nil || modelID != nil else { return nil }

        var missingFlags: [String] = []
        if bundlePath == nil { missingFlags.append("--primary-staged-bundle") }
        if modelID == nil { missingFlags.append("--primary-model-id") }
        guard missingFlags.isEmpty else {
            throw ConfigurationError.incomplete(missingFlags: missingFlags)
        }
        guard let bundlePath, let modelID else {
            throw ConfigurationError.incomplete(missingFlags: missingFlags)
        }
        guard !bundlePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ConfigurationError.emptyValue(flag: "--primary-staged-bundle")
        }
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ConfigurationError.emptyValue(flag: "--primary-model-id")
        }

        let bundleURL = URL(fileURLWithPath: bundlePath, isDirectory: true).standardizedFileURL
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: bundleURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw ConfigurationError.missingBundle(path: bundleURL.path)
        }
        let manifestURL = bundleURL.appendingPathComponent("stage-manifest.json")
        guard let data = try? Data(contentsOf: manifestURL, options: [.mappedIfSafe]),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              manifest["schema"] as? String == "caix.cluster.stage_manifest.v0",
              manifest["stages"] is [Any]
        else {
            throw ConfigurationError.notStagedBundle(path: bundleURL.path)
        }

        return PrimaryStagedBundleConfiguration(bundleURL: bundleURL, modelID: modelID)
    }
}
