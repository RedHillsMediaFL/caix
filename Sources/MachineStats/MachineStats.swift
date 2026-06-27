import Foundation
import Darwin

/// A point-in-time snapshot of host hardware utilization, for the dashboard.
public struct MachineSnapshot: Codable, Sendable {
    public var chip: String
    public var logicalCores: Int
    public var totalRAMBytes: UInt64
    public var usedRAMBytes: UInt64
    public var memoryUsedFraction: Double
    public var gpuUtilizationPercent: Double?
    public var gpuInUseMemoryBytes: UInt64?
}

/// Native host telemetry via sysctl / mach host_statistics64 / IOAccelerator.
/// Off the inference hot path, so the GPU read may shell to ioreg for now.
public enum MachineStats {

    public static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }

    public static func sysctlUInt64(_ name: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return value
    }

    /// Used unified memory ≈ (active + wired + compressed) * page size.
    public static func usedRAMBytes() -> UInt64 {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let kr = withUnsafeMutablePointer(to: &stats) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        var pageSize: vm_size_t = 0
        host_page_size(mach_host_self(), &pageSize)
        let page = UInt64(pageSize)
        return (UInt64(stats.active_count) + UInt64(stats.wire_count)
            + UInt64(stats.compressor_page_count)) * page
    }

    /// GPU "Device Utilization %" + in-use memory from IOAccelerator PerformanceStatistics.
    public static func gpuStats() -> (utilizationPercent: Double?, inUseMemoryBytes: UInt64?) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
        proc.arguments = ["-r", "-d", "1", "-c", "IOAccelerator"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do { try proc.run() } catch { return (nil, nil) }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        guard let s = String(data: data, encoding: .utf8) else { return (nil, nil) }
        func number(after key: String) -> Double? {
            guard let r = s.range(of: key) else { return nil }
            let tail = s[r.upperBound...].drop { !($0.isNumber) }
            let digits = tail.prefix { $0.isNumber }
            return Double(digits)
        }
        let util = number(after: "\"Device Utilization %\"=")
        let mem = number(after: "\"In use system memory\"=")
        return (util, mem.map { UInt64($0) })
    }

    public static func snapshot() -> MachineSnapshot {
        let total = sysctlUInt64("hw.memsize") ?? 0
        let used = usedRAMBytes()
        let gpu = gpuStats()
        return MachineSnapshot(
            chip: sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon",
            logicalCores: Int(sysctlUInt64("hw.ncpu") ?? 0),
            totalRAMBytes: total,
            usedRAMBytes: used,
            memoryUsedFraction: total > 0 ? Double(used) / Double(total) : 0,
            gpuUtilizationPercent: gpu.utilizationPercent,
            gpuInUseMemoryBytes: gpu.inUseMemoryBytes
        )
    }
}
