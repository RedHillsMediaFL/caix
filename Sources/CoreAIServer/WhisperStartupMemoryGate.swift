import Foundation
import MachineStats

/// Fail-closed admission check before specializing the multi-gigabyte resident Whisper asset.
///
/// The real Whisper large-v2 verifier peaked at 11.73 GiB. Requiring 16 GiB available preserves
/// roughly 4 GiB of startup headroom before the server's ongoing resident-memory supervisor takes
/// over request admission.
///
/// `CAIX_WHISPER_MEMORY_FLOOR_GIB` overrides the floor for constrained hosts (e.g. a 32 GB
/// laptop serving Whisper alone, where the measured 11.73 GiB peak fits with less headroom).
/// The override never goes below the measured peak rounded up (12 GiB); the default stays 16.
public enum WhisperStartupMemoryGate {
    public static var requiredAvailableBytes: UInt64 {
        if let raw = ProcessInfo.processInfo.environment["CAIX_WHISPER_MEMORY_FLOOR_GIB"],
           let gib = UInt64(raw), gib >= 12 {
            return gib * 1_073_741_824
        }
        return 16 * 1_073_741_824
    }

    public enum Rejection: Error, Sendable, Equatable, CustomStringConvertible {
        case memoryPressure(MachineMemoryPressure)
        case insufficientAvailableMemory(requiredBytes: UInt64, actualBytes: UInt64)

        public var description: String {
            switch self {
            case .memoryPressure(let pressure):
                return "resident Whisper specialization requires green memory pressure (found \(pressure.rawValue))"
            case .insufficientAvailableMemory(let requiredBytes, let actualBytes):
                return "resident Whisper specialization requires at least \(requiredBytes / 1_073_741_824) GiB available (found \(actualBytes) bytes)"
            }
        }
    }

    public static func validate(_ snapshot: MachineMemorySafetySnapshot) throws {
        guard snapshot.pressure == .green else {
            throw Rejection.memoryPressure(snapshot.pressure)
        }
        guard snapshot.availableRAMBytes >= requiredAvailableBytes else {
            throw Rejection.insufficientAvailableMemory(
                requiredBytes: requiredAvailableBytes,
                actualBytes: snapshot.availableRAMBytes)
        }
    }
}
