import Foundation

/// Startup-only configuration for a future staged Gemma MTP loop.
///
/// This type intentionally owns no decode algorithm. It establishes a narrow proof boundary so
/// `--require-mtp` cannot claim speculative serving until the staged target loop has drafted at
/// least one token.
public struct StagedMTPStartupConfiguration: Sendable, Equatable {
    public static let defaultDraftTokens = 4
    public static let maximumDraftTokens = 8

    public enum ConfigurationError: Error, Sendable, Equatable, CustomStringConvertible {
        case missingAssistant
        case missingPrimaryStagedBundle
        case clusterUnsupported
        case prewarmRequired
        case invalidAssistantAsset(String)
        case invalidDraftTokens(Int)
        case unsupportedPrimaryBackend(String)
        case proofUnavailable(String)
        case invalidProof(String)
        case runtimeUnavailable

        public var description: String {
            switch self {
            case .missingAssistant:
                return "staged MTP requires --staged-mtp-assistant <.aimodel>"
            case .missingPrimaryStagedBundle:
                return "staged MTP requires a local --primary-staged-bundle"
            case .clusterUnsupported:
                return "staged MTP cannot be combined with --cluster"
            case .prewarmRequired:
                return "--require-mtp cannot be combined with --no-prewarm"
            case .invalidAssistantAsset(let path):
                return "staged MTP assistant must be an existing .aimodel directory: \(path)"
            case .invalidDraftTokens(let value):
                return "--mtp-draft-tokens must be in 1...\(maximumDraftTokens); got \(value)"
            case .unsupportedPrimaryBackend(let path):
                return "primary staged bundle does not expose an eagle_target backend: \(path)"
            case .proofUnavailable(let reason):
                return "staged MTP could not be proven before startup: \(reason)"
            case .invalidProof(let reason):
                return "staged MTP proof is insufficient: \(reason)"
            case .runtimeUnavailable:
                return "staged MTP requires a Core AI runtime-linked caix build"
            }
        }
    }

    public let assistantURL: URL
    public let draftTokens: Int
    public let requireMTP: Bool
    public let primaryBundleURL: URL

    private init(assistantURL: URL, draftTokens: Int, requireMTP: Bool, primaryBundleURL: URL) {
        self.assistantURL = assistantURL
        self.draftTokens = draftTokens
        self.requireMTP = requireMTP
        self.primaryBundleURL = primaryBundleURL
    }

    /// Resolves absent MTP options to `nil`, preserving standard staged startup unchanged.
    public static func resolve(
        assistantPath: String?,
        draftTokens: Int?,
        requireMTP: Bool,
        primaryBundleURL: URL?,
        clusterMode: Bool,
        prewarm: String
    ) throws -> StagedMTPStartupConfiguration? {
        let requested = assistantPath != nil || draftTokens != nil || requireMTP
        guard requested else { return nil }
        guard let assistantPath, !assistantPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ConfigurationError.missingAssistant
        }
        guard let primaryBundleURL else {
            throw ConfigurationError.missingPrimaryStagedBundle
        }
        if requireMTP, clusterMode {
            throw ConfigurationError.clusterUnsupported
        }
        if requireMTP, prewarm == "off" {
            throw ConfigurationError.prewarmRequired
        }

        let assistantURL = URL(fileURLWithPath: assistantPath, isDirectory: true)
            .standardizedFileURL
        var isDirectory = ObjCBool(false)
        guard assistantURL.pathExtension.lowercased() == "aimodel",
              FileManager.default.fileExists(atPath: assistantURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw ConfigurationError.invalidAssistantAsset(assistantURL.path)
        }

        let selectedDraftTokens = draftTokens ?? defaultDraftTokens
        guard (1...maximumDraftTokens).contains(selectedDraftTokens) else {
            throw ConfigurationError.invalidDraftTokens(selectedDraftTokens)
        }

        let manifestURL = primaryBundleURL.standardizedFileURL
            .appendingPathComponent("stage-manifest.json")
        let manifest: DistributedStageManifest
        do {
            manifest = try DistributedStageManifest.load(from: manifestURL)
        } catch {
            throw ConfigurationError.unsupportedPrimaryBackend(primaryBundleURL.path)
        }
        guard manifest.eagleTarget != nil else {
            throw ConfigurationError.unsupportedPrimaryBackend(primaryBundleURL.path)
        }

        return StagedMTPStartupConfiguration(
            assistantURL: assistantURL,
            draftTokens: selectedDraftTokens,
            requireMTP: requireMTP,
            primaryBundleURL: primaryBundleURL.standardizedFileURL)
    }

    /// Enforces the proof boundary used only by `--require-mtp`.
    public func requireProof(using prover: StagedMTPStartupProver) async throws {
        guard requireMTP else { return }
        let proof = try await prover(self)
        guard proof.executionMode == .sequentialNoRollback else {
            throw ConfigurationError.invalidProof("execution_mode must be sequential_no_rollback")
        }
        guard !proof.fast else {
            throw ConfigurationError.invalidProof("fast must be false")
        }
        guard proof.draftedTokens > 0 else {
            throw ConfigurationError.invalidProof("drafted_tokens must be positive")
        }
    }

    /// Default hook until the staged target-loop owner supplies an actual sequential proof.
    /// It validates and specializes the assistant when Core AI is linked, then fails closed rather
    /// than treating a loaded assistant as evidence that MTP drafted tokens.
    public static func nativeProver(
        _ configuration: StagedMTPStartupConfiguration
    ) async throws -> StagedMTPStartupProof {
        #if COREAI_RUNTIME
        _ = try await Gemma4MTPNativeRunner.load(aimodelURL: configuration.assistantURL)
        throw ConfigurationError.proofUnavailable(
            "the staged target sequential loop has not produced drafted tokens")
        #else
        throw ConfigurationError.runtimeUnavailable
        #endif
    }
}

public enum StagedMTPExecutionMode: String, Sendable, Equatable {
    case sequentialNoRollback = "sequential_no_rollback"
}

/// Evidence returned by the staged target-loop owner after an actual startup draft.
public struct StagedMTPStartupProof: Sendable, Equatable {
    public let draftedTokens: Int
    public let executionMode: StagedMTPExecutionMode
    public let fast: Bool

    public init(draftedTokens: Int, executionMode: StagedMTPExecutionMode, fast: Bool) {
        self.draftedTokens = draftedTokens
        self.executionMode = executionMode
        self.fast = fast
    }
}

public typealias StagedMTPStartupProver = @Sendable (
    StagedMTPStartupConfiguration
) async throws -> StagedMTPStartupProof
