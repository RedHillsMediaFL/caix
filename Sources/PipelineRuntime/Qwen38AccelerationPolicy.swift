import Foundation

/// Caller intent for a Qwen3.8 generation. `auto` is deterministic-only acceleration: it never
/// changes a sampling distribution to make native MTP appear faster.
public enum Qwen38AccelerationRequest: String, Sendable, Codable, Equatable {
    case auto
    case autoregressive
    case mtp
}

/// The execution path selected for one request.
public enum Qwen38AccelerationDecision: String, Sendable, Codable, Equatable {
    case autoregressive
    case nativeMTP
}

/// Measured same-machine evidence for native Qwen MTP. It is intentionally a small value type so
/// the server can persist it with an artifact and the runtime can fail closed without inventing a
/// speed estimate from proposal acceptance alone.
public struct Qwen38MTPProof: Sendable, Codable, Equatable {
    public let exactGreedy: Bool
    public let autoregressiveTokensPerSecond: Double
    public let mtpTokensPerSecond: Double

    public init(
        exactGreedy: Bool,
        autoregressiveTokensPerSecond: Double,
        mtpTokensPerSecond: Double
    ) {
        self.exactGreedy = exactGreedy
        self.autoregressiveTokensPerSecond = autoregressiveTokensPerSecond
        self.mtpTokensPerSecond = mtpTokensPerSecond
    }

    /// The acceptance gate agreed for this artifact: native MTP must be at least 15% faster than
    /// its own Qwen AR path on the same suite and machine.
    public var meetsSpeedGate: Bool {
        autoregressiveTokensPerSecond > 0
            && mtpTokensPerSecond >= autoregressiveTokensPerSecond * 1.15
    }

    enum CodingKeys: String, CodingKey {
        case exactGreedy = "exact_greedy"
        case autoregressiveTokensPerSecond = "autoregressive_tokens_per_second"
        case mtpTokensPerSecond = "mtp_tokens_per_second"
    }
}

/// Resolves Qwen3.8 native MTP without permitting output-quality regressions.
public enum Qwen38AccelerationPolicy {
    public enum Error: Swift.Error, Sendable, Equatable {
        case samplingRequiresAutoregressive
        case greedyParityNotProven
        case speedGateNotMet
    }

    public static func resolve(
        requested: Qwen38AccelerationRequest,
        temperature: Double,
        proof: Qwen38MTPProof?
    ) throws -> Qwen38AccelerationDecision {
        if temperature > 0 {
            guard requested != .mtp else { throw Error.samplingRequiresAutoregressive }
            return .autoregressive
        }
        guard requested != .autoregressive else { return .autoregressive }

        let hasParity = proof?.exactGreedy == true
        let meetsSpeed = proof?.meetsSpeedGate == true
        if requested == .auto {
            return hasParity && meetsSpeed ? .nativeMTP : .autoregressive
        }
        guard hasParity else { throw Error.greedyParityNotProven }
        guard meetsSpeed else { throw Error.speedGateNotMet }
        return .nativeMTP
    }
}

/// Binds the artifact proof gate to the currently compiled native runner. This deliberately
/// distinguishes “the artifact has valid benchmark evidence” from “this caix build can execute
/// its MTP sidecar.” Until the latter exists, `auto` remains AR and an explicit MTP request fails.
public enum Qwen38ExecutionPolicy {
    public enum Error: Swift.Error, Sendable, Equatable {
        case nativeMTPRunnerUnavailable
    }

    public static func resolve(
        requested: Qwen38AccelerationRequest,
        temperature: Double,
        proof: Qwen38MTPProof?,
        nativeMTPAvailable: Bool
    ) throws -> Qwen38AccelerationDecision {
        let requestedDecision = try Qwen38AccelerationPolicy.resolve(
            requested: requested,
            temperature: temperature,
            proof: proof)
        guard requestedDecision == .nativeMTP else { return .autoregressive }
        guard nativeMTPAvailable else {
            if requested == .mtp { throw Error.nativeMTPRunnerUnavailable }
            return .autoregressive
        }
        return .nativeMTP
    }
}
