import XCTest

@testable import CoreAIServer
import MachineStats

final class ResidentMemorySupervisorTests: XCTestCase {
    private let gib = UInt64(1_073_741_824)
    private let mib = UInt64(1_048_576)

    func testFirstObservationTreatsExistingSwapAsBaseline() async {
        let supervisor = ResidentMemorySupervisor()

        let status = await supervisor.observe(sample(
            resident: 39 * gib,
            available: 12 * gib,
            pressure: .green,
            swap: 5 * gib))

        XCTAssertEqual(status.disposition, .admit)
        XCTAssertEqual(status.swapBaselineBytes, 5 * gib)
        XCTAssertEqual(status.swapGrowthBytes, 0)
    }

    func testSwapGrowthAndPressureDrainNewInference() async {
        let supervisor = ResidentMemorySupervisor()
        _ = await supervisor.observe(sample(
            resident: 39 * gib,
            available: 12 * gib,
            pressure: .green,
            swap: 5 * gib))

        let swap = await supervisor.observe(sample(
            resident: 39 * gib,
            available: 12 * gib,
            pressure: .green,
            swap: 5 * gib + 256 * mib + 1))
        XCTAssertEqual(swap.disposition, .drain)
        XCTAssertEqual(swap.reason, "swapGrowth")

        let pressure = await supervisor.observe(sample(
            resident: 39 * gib,
            available: 12 * gib,
            pressure: .yellow,
            swap: 5 * gib))
        XCTAssertEqual(pressure.disposition, .drain)
        XCTAssertEqual(pressure.reason, "memoryPressure")
    }

    func testResidentThresholdDrainsThenRestartSticks() async {
        let supervisor = ResidentMemorySupervisor()

        let drain = await supervisor.observe(sample(
            resident: 42 * gib,
            available: 12 * gib,
            pressure: .green,
            swap: 0))
        XCTAssertEqual(drain.disposition, .drain)
        XCTAssertEqual(drain.reason, "residentLimit")

        let restart = await supervisor.observe(sample(
            resident: 44 * gib,
            available: 12 * gib,
            pressure: .green,
            swap: 0))
        XCTAssertEqual(restart.disposition, .restart)
        XCTAssertEqual(restart.reason, "residentKillLimit")

        let afterRecovery = await supervisor.observe(sample(
            resident: 20 * gib,
            available: 30 * gib,
            pressure: .green,
            swap: 0))
        XCTAssertEqual(afterRecovery.disposition, .restart)
    }

    func testUnavailableNativeTelemetryFailsClosed() async {
        let supervisor = ResidentMemorySupervisor()

        let missingFootprint = await supervisor.observe(sample(
            resident: 0,
            available: 30 * gib,
            pressure: .green,
            swap: 0))
        XCTAssertEqual(missingFootprint.disposition, .drain)
        XCTAssertEqual(missingFootprint.reason, "telemetryUnavailable")

        let missingSwap = await ResidentMemorySupervisor().observe(sample(
            resident: 20 * gib,
            available: 30 * gib,
            pressure: .green,
            swap: nil))
        XCTAssertEqual(missingSwap.disposition, .drain)
        XCTAssertEqual(missingSwap.reason, "telemetryUnavailable")
    }

    private func sample(
        resident: UInt64,
        available: UInt64,
        pressure: MachineMemoryPressure,
        swap: UInt64?
    ) -> MachineMemorySafetySnapshot {
        MachineMemorySafetySnapshot(
            totalRAMBytes: 64 * gib,
            usedRAMBytes: 64 * gib - available,
            availableRAMBytes: available,
            processPhysicalFootprintBytes: resident,
            pressure: pressure,
            swapUsedBytes: swap)
    }
}
