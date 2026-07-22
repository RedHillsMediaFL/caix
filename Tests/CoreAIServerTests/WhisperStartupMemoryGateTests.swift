import CoreAIServer
import MachineStats
import XCTest

final class WhisperStartupMemoryGateTests: XCTestCase {
    func testAdmitsGreenPressureWithSixteenGiBAvailable() throws {
        let snapshot = memorySnapshot(
            availableBytes: WhisperStartupMemoryGate.requiredAvailableBytes,
            pressure: .green)

        XCTAssertNoThrow(try WhisperStartupMemoryGate.validate(snapshot))
    }

    func testRejectsNonGreenOrUnknownPressure() throws {
        for pressure in [
            MachineMemoryPressure.yellow,
            MachineMemoryPressure.red,
            MachineMemoryPressure.unknown,
        ] {
            XCTAssertThrowsError(try WhisperStartupMemoryGate.validate(memorySnapshot(
                availableBytes: 64 * 1_073_741_824,
                pressure: pressure))) { error in
                    XCTAssertEqual(
                        error as? WhisperStartupMemoryGate.Rejection,
                        .memoryPressure(pressure))
                    XCTAssertTrue(String(describing: error).contains("requires green memory pressure"))
                }
        }
    }

    func testRejectsLessThanSixteenGiBAvailable() throws {
        let actual = WhisperStartupMemoryGate.requiredAvailableBytes - 1

        XCTAssertThrowsError(try WhisperStartupMemoryGate.validate(memorySnapshot(
            availableBytes: actual,
            pressure: .green))) { error in
                XCTAssertEqual(
                    error as? WhisperStartupMemoryGate.Rejection,
                    .insufficientAvailableMemory(
                        requiredBytes: WhisperStartupMemoryGate.requiredAvailableBytes,
                        actualBytes: actual))
                XCTAssertEqual(
                    String(describing: error),
                    "resident Whisper specialization requires at least 16 GiB available (found \(actual) bytes)")
            }
    }

    private func memorySnapshot(
        availableBytes: UInt64,
        pressure: MachineMemoryPressure
    ) -> MachineMemorySafetySnapshot {
        MachineMemorySafetySnapshot(
            totalRAMBytes: 64 * 1_073_741_824,
            usedRAMBytes: 64 * 1_073_741_824 - min(
                availableBytes,
                64 * 1_073_741_824),
            availableRAMBytes: availableBytes,
            allocationCapacityBytes: availableBytes,
            processPhysicalFootprintBytes: 128 * 1_048_576,
            pressure: pressure,
            swapUsedBytes: 0)
    }
}
