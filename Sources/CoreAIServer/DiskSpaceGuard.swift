import Foundation

struct DiskSpaceGuard {
    enum Decision: Equatable {
        case allow
        case reject(String)
    }

    private static let bytesPerGiB: Int64 = 1_073_741_824
    static let defaultReserveGiB: Int64 = 500

    static func reserveBytesFromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment) -> Int64 {
        guard let raw = env["CAIX_STOP_FLOOR_GIB"],
              let gib = Int64(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              gib >= 0
        else {
            return defaultReserveGiB * bytesPerGiB
        }
        let (bytes, overflow) = gib.multipliedReportingOverflow(by: bytesPerGiB)
        return overflow ? Int64.max : bytes
    }

    static func preflightInstall(
        destinationRoot: URL,
        incomingBytes: Int64?,
        reserveBytes: Int64 = reserveBytesFromEnvironment()
    ) -> String? {
        guard let available = availableBytes(for: destinationRoot) else {
            return "disk preflight failed: could not inspect free space for \(destinationRoot.path)"
        }
        switch evaluate(availableBytes: available, incomingBytes: incomingBytes, reserveBytes: reserveBytes) {
        case .allow:
            return nil
        case .reject(let message):
            return message
        }
    }

    static func evaluate(availableBytes: Int64, incomingBytes: Int64?, reserveBytes: Int64) -> Decision {
        let payloadBytes = max(0, incomingBytes ?? 0)
        let floorBytes = max(0, reserveBytes)
        let (requiredBytes, overflow) = floorBytes.addingReportingOverflow(payloadBytes)
        if overflow || availableBytes < requiredBytes {
            let payload = incomingBytes.map { formatBytes(max(0, $0)) } ?? "unknown"
            let required = overflow ? "overflow" : formatBytes(requiredBytes)
            return .reject(
                "insufficient disk for model install: free \(formatBytes(availableBytes)), required \(required) (payload \(payload) + reserve \(formatBytes(floorBytes)))")
        }
        return .allow
    }

    private static func availableBytes(for url: URL) -> Int64? {
        let probe = existingAncestor(for: url)
        guard let attrs = try? FileManager.default.attributesOfFileSystem(forPath: probe.path),
              let free = attrs[.systemFreeSize] as? NSNumber
        else { return nil }
        return free.int64Value
    }

    private static func existingAncestor(for url: URL) -> URL {
        var current = url
        var isDir: ObjCBool = false
        while !FileManager.default.fileExists(atPath: current.path, isDirectory: &isDir) {
            let parent = current.deletingLastPathComponent()
            if parent.path == current.path { break }
            current = parent
        }
        return current
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        if bytes == Int64.max || bytes == Int64.min { return "overflow" }
        let sign = bytes < 0 ? "-" : ""
        let value = Double(abs(bytes))
        let gib = value / Double(bytesPerGiB)
        if gib >= 10 {
            return "\(sign)\(Int(gib.rounded())) GiB"
        }
        if gib >= 1 {
            return "\(sign)\(String(format: "%.1f", gib)) GiB"
        }
        let mib = value / 1_048_576
        return "\(sign)\(Int(mib.rounded())) MiB"
    }
}
