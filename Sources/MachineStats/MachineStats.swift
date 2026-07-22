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

/// Native macOS memory-pressure state. Unknown is deliberately distinct from green so callers
/// never admit a large allocation when the kernel pressure probe fails.
public enum MachineMemoryPressure: String, Codable, Sendable, Equatable {
    case green
    case yellow
    case red
    case unknown

    init(rawDarwinValue: Int32?) {
        switch rawDarwinValue {
        case 1: self = .green
        case 2: self = .yellow
        case 4: self = .red
        default: self = .unknown
        }
    }
}

/// Allocation-free inputs for the resident-model safety policy.
public struct MachineMemorySafetySnapshot: Codable, Sendable, Equatable {
    public var totalRAMBytes: UInt64
    public var usedRAMBytes: UInt64
    public var availableRAMBytes: UInt64
    public var processPhysicalFootprintBytes: UInt64
    public var pressure: MachineMemoryPressure
    public var swapUsedBytes: UInt64?

    public init(
        totalRAMBytes: UInt64,
        usedRAMBytes: UInt64,
        availableRAMBytes: UInt64,
        processPhysicalFootprintBytes: UInt64,
        pressure: MachineMemoryPressure,
        swapUsedBytes: UInt64?
    ) {
        self.totalRAMBytes = totalRAMBytes
        self.usedRAMBytes = usedRAMBytes
        self.availableRAMBytes = availableRAMBytes
        self.processPhysicalFootprintBytes = processPhysicalFootprintBytes
        self.pressure = pressure
        self.swapUsedBytes = swapUsedBytes
    }
}

/// Native host telemetry via sysctl / mach host_statistics64 / IOAccelerator.
/// Off the inference hot path, so the GPU read may shell to ioreg for now.
public enum MachineStats {

    public static func machineName() -> String {
        var buffer = [CChar](repeating: 0, count: 256)
        let status = buffer.withUnsafeMutableBufferPointer { pointer in
            gethostname(pointer.baseAddress, pointer.count)
        }
        if status == 0 {
            let value = MachineStats.string(fromNullTerminated: buffer)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return value }
        }
        return sysctlString("kern.hostname") ?? "unknown"
    }

    public static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return string(fromNullTerminated: buf)
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

    /// Current process physical footprint, including compressed resident pages attributed by the
    /// kernel. This is the value used for the CAIX worker high-water gate.
    public static func processPhysicalFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    $0,
                    &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }

    /// Point-in-time native inputs used to decide whether resident model loading and request
    /// admission are safe. No model or large backing allocation is created by this probe.
    public static func memorySafetySnapshot() -> MachineMemorySafetySnapshot {
        let total = sysctlUInt64("hw.memsize") ?? 0
        let used = min(usedRAMBytes(), total)

        var rawPressure: Int32 = 0
        var pressureSize = MemoryLayout<Int32>.size
        let pressureStatus = sysctlbyname(
            "kern.memorystatus_vm_pressure_level",
            &rawPressure,
            &pressureSize,
            nil,
            0)

        var swap = xsw_usage()
        var swapSize = MemoryLayout<xsw_usage>.size
        let swapStatus = sysctlbyname("vm.swapusage", &swap, &swapSize, nil, 0)

        return MachineMemorySafetySnapshot(
            totalRAMBytes: total,
            usedRAMBytes: used,
            availableRAMBytes: total - used,
            processPhysicalFootprintBytes: processPhysicalFootprintBytes(),
            pressure: MachineMemoryPressure(
                rawDarwinValue: pressureStatus == 0 ? rawPressure : nil),
            swapUsedBytes: swapStatus == 0 ? UInt64(swap.xsu_used) : nil)
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

    private static func string(fromNullTerminated buffer: [CChar]) -> String {
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
}
