import MachineStats
import PipelineRuntime

/// Fail-closed, process-local admission state for the resident Gemma + Whisper service.
///
/// The first valid swap observation establishes a generation baseline; only growth from that
/// point is charged. A restart decision is terminal for the generation so a transient footprint
/// drop cannot reopen admission while an external service manager is preparing a clean restart.
actor ResidentMemorySupervisor {
    enum Disposition: String, Codable, Sendable, Equatable {
        case admit
        case drain
        case restart
    }

    struct Status: Codable, Sendable, Equatable {
        let disposition: Disposition
        let reason: String?
        let workerResidentBytes: UInt64
        let availableBytes: UInt64
        let pressure: MachineMemoryPressure
        let swapBaselineBytes: UInt64?
        let swapGrowthBytes: UInt64

        var permitsInference: Bool { disposition == .admit }
    }

    private let gate: ResidentServiceHealthGate
    private var swapBaselineBytes: UInt64?
    private var terminalRestart: Status?
    private var latest: Status?

    init(limits: ResidentServiceHealthGate.Limits = .studio64GiB) {
        self.gate = ResidentServiceHealthGate(limits: limits)
    }

    @discardableResult
    func refresh() -> Status {
        observe(MachineStats.memorySafetySnapshot())
    }

    func currentStatus() -> Status? { latest }

    @discardableResult
    func observe(_ sample: MachineMemorySafetySnapshot) -> Status {
        if let terminalRestart { return terminalRestart }

        guard sample.totalRAMBytes > 0,
              sample.processPhysicalFootprintBytes > 0,
              sample.pressure != .unknown,
              let swapUsedBytes = sample.swapUsedBytes
        else {
            let status = Status(
                disposition: .drain,
                reason: "telemetryUnavailable",
                workerResidentBytes: sample.processPhysicalFootprintBytes,
                availableBytes: sample.availableRAMBytes,
                pressure: sample.pressure,
                swapBaselineBytes: swapBaselineBytes,
                swapGrowthBytes: 0)
            latest = status
            return status
        }

        if swapBaselineBytes == nil { swapBaselineBytes = swapUsedBytes }
        let baseline = swapBaselineBytes ?? swapUsedBytes
        let swapGrowth = swapUsedBytes >= baseline ? swapUsedBytes - baseline : 0
        let runtimePressure: ResidentMemoryPressure
        switch sample.pressure {
        case .green: runtimePressure = .green
        case .yellow: runtimePressure = .yellow
        case .red, .unknown: runtimePressure = .red
        }
        let action = gate.action(for: .init(
            workerResidentBytes: sample.processPhysicalFootprintBytes,
            availableBytes: sample.availableRAMBytes,
            pressure: runtimePressure,
            swapGrowthBytes: swapGrowth))

        let disposition: Disposition
        let reason: String?
        switch action {
        case .admit:
            disposition = .admit
            reason = nil
        case .drain(let drainReason):
            disposition = .drain
            reason = drainReason.rawValue
        case .restart(let restartReason):
            disposition = .restart
            reason = restartReason.rawValue
        }
        let status = Status(
            disposition: disposition,
            reason: reason,
            workerResidentBytes: sample.processPhysicalFootprintBytes,
            availableBytes: sample.availableRAMBytes,
            pressure: sample.pressure,
            swapBaselineBytes: baseline,
            swapGrowthBytes: swapGrowth)
        latest = status
        if disposition == .restart { terminalRestart = status }
        return status
    }
}
