import XCTest
@testable import MachineStats

final class MachineStatsTests: XCTestCase {
    func testMachineNameIsAvailable() {
        let name = MachineStats.machineName()
        XCTAssertFalse(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertNotEqual(name, "unknown")
    }

    func testSnapshotHasSaneValues() {
        let s = MachineStats.snapshot()
        XCTAssertFalse(s.chip.isEmpty)
        XCTAssertGreaterThan(s.logicalCores, 0)
        XCTAssertGreaterThan(s.totalRAMBytes, 0)
        XCTAssertGreaterThan(s.usedRAMBytes, 0)
        XCTAssertLessThanOrEqual(s.usedRAMBytes, s.totalRAMBytes)
        XCTAssertGreaterThan(s.memoryUsedFraction, 0)
        XCTAssertLessThanOrEqual(s.memoryUsedFraction, 1.0)
    }

    func testNativeMemorySafetySnapshotHasSaneValues() {
        let snapshot = MachineStats.memorySafetySnapshot()

        XCTAssertGreaterThan(snapshot.totalRAMBytes, 0)
        XCTAssertLessThanOrEqual(snapshot.usedRAMBytes, snapshot.totalRAMBytes)
        XCTAssertEqual(
            snapshot.availableRAMBytes,
            snapshot.totalRAMBytes - snapshot.usedRAMBytes)
        XCTAssertGreaterThan(snapshot.processPhysicalFootprintBytes, 0)
        XCTAssertNotEqual(snapshot.pressure, .unknown)
    }

    func testMemoryPressureLevelMapsDarwinBitValues() {
        XCTAssertEqual(MachineMemoryPressure(rawDarwinValue: 1), .green)
        XCTAssertEqual(MachineMemoryPressure(rawDarwinValue: 2), .yellow)
        XCTAssertEqual(MachineMemoryPressure(rawDarwinValue: 4), .red)
        XCTAssertEqual(MachineMemoryPressure(rawDarwinValue: nil), .unknown)
        XCTAssertEqual(MachineMemoryPressure(rawDarwinValue: 99), .unknown)
    }
}
