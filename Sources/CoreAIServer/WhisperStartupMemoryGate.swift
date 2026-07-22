import MachineStats

/// Fail-closed admission check before specializing the multi-gigabyte resident Whisper asset.
///
/// The real Whisper large-v2 verifier peaked at 11.73 GiB. Requiring 16 GiB available preserves
/// roughly 4 GiB of startup headroom before the server's ongoing resident-memory supervisor takes
/// over request admission.
public enum WhisperStartupMemoryGate {
    public static let requiredAvailableBytes: UInt64 = 16 * 1_073_741_824

    public enum Rejection: Error, Sendable, Equatable, CustomStringConvertible {
        case memoryPressure(MachineMemoryPressure)
        case insufficientAvailableMemory(requiredBytes: UInt64, actualBytes: UInt64)

        public var description: String {
            switch self {
            case .memoryPressure(let pressure):
                return "resident Whisper specialization requires green memory pressure (found \(pressure.rawValue))"
            case .insufficientAvailableMemory(_, let actualBytes):
                return "resident Whisper specialization requires at least 16 GiB available (found \(actualBytes) bytes)"
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
