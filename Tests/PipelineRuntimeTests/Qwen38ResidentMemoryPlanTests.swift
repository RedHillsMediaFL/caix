import XCTest

@testable import PipelineRuntime

final class Qwen38ResidentMemoryPlanTests: XCTestCase {
    private let gib = UInt64(1_073_741_824)

    func testM1UltraPlanReservesFullContextWithoutFallbackTiers() throws {
        let plan = Qwen38ResidentMemoryPlan.m1Ultra64GiB
        let admission = try plan.admit(
            availableBytes: plan.requiredAvailableBytes,
            pressure: .green,
            swapGrowthBytes: 0)

        XCTAssertEqual(plan.contextLength, 262_144)
        XCTAssertEqual(plan.keyValueCacheBytes, 16 * gib)
        XCTAssertEqual(plan.contextCandidates, [262_144])
        XCTAssertEqual(admission, .admit)
    }

    func testLowMemoryDeniesInsteadOfSilentlyReducingContext() throws {
        let plan = Qwen38ResidentMemoryPlan.m1Ultra64GiB

        let admission = try plan.admit(
            availableBytes: plan.requiredAvailableBytes - 1,
            pressure: .green,
            swapGrowthBytes: 0)

        XCTAssertEqual(admission, .deny(.insufficientAvailableMemory))
    }

    func testNonGreenPressureAndSwapGrowthDenyBeforeAllocation() throws {
        let plan = Qwen38ResidentMemoryPlan.m1Ultra64GiB

        XCTAssertEqual(
            try plan.admit(
                availableBytes: plan.requiredAvailableBytes,
                pressure: .yellow,
                swapGrowthBytes: 0),
            .deny(.memoryPressureNotGreen))
        XCTAssertEqual(
            try plan.admit(
                availableBytes: plan.requiredAvailableBytes,
                pressure: .green,
                swapGrowthBytes: plan.maximumSwapGrowthBytes + 1),
            .deny(.swapGrowthExceeded))
    }
}
