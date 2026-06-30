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
}
