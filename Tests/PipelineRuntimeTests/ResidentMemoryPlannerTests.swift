import XCTest

@testable import PipelineRuntime

final class ResidentMemoryPlannerTests: XCTestCase {
    private let gib = UInt64(1_073_741_824)

    func testStudioPlanSelects64KWithoutAllocatingRejectedTiers() throws {
        let planner = ResidentMemoryPlanner.studio64GiB

        let selection = try planner.selectLargestComfortable(
            availableBytes: 24 * gib,
            pressure: .green,
            swapGrowthBytes: 0)

        XCTAssertEqual(selection.selectedContextLength, 65_536)
        XCTAssertEqual(selection.evaluations.map(\.contextLength), [262_144, 131_072, 65_536])
        XCTAssertEqual(selection.evaluations[0].rejection, .plannedWorkerLimit)
        XCTAssertEqual(selection.evaluations[1].rejection, .plannedWorkerLimit)
        XCTAssertNil(selection.evaluations[2].rejection)
        XCTAssertTrue(selection.evaluations.allSatisfy { !$0.allocated })
    }

    func testSplitKVEstimateMatchesGemma4Geometry() throws {
        let planner = ResidentMemoryPlanner.studio64GiB
        let estimate = try planner.estimate(contextLength: 65_536)

        XCTAssertEqual(estimate.targetGlobalKVBytes, 5 * gib)
        XCTAssertEqual(estimate.targetSlidingKVBytes, 1_258_291_200)
        XCTAssertEqual(estimate.representativeGlobalKVBytes, gib / 2)
        XCTAssertEqual(estimate.representativeSlidingKVBytes, 16_777_216)
        XCTAssertEqual(estimate.fixedResidentBytes, 31_943_819_264)
        XCTAssertEqual(estimate.plannedResidentBytes, 39_124_467_712)
        XCTAssertLessThanOrEqual(
            estimate.plannedResidentBytes,
            planner.budget.plannedWorkerLimitBytes)
    }

    func testTighterMeasuredBudgetFallsBackTo32K() throws {
        var budget = ResidentMemoryPlanner.Budget.studio64GiB
        budget.plannedWorkerLimitBytes = 36 * gib
        let planner = ResidentMemoryPlanner(budget: budget)

        let selection = try planner.selectLargestComfortable(
            availableBytes: 20 * gib,
            pressure: .green,
            swapGrowthBytes: 0)

        XCTAssertEqual(selection.selectedContextLength, 32_768)
        XCTAssertEqual(selection.evaluations.last?.rejection, nil)
    }

    func testOperationalPreconditionsRejectBeforeContextEvaluation() throws {
        let planner = ResidentMemoryPlanner.studio64GiB

        let lowAvailability = try planner.selectLargestComfortable(
            availableBytes: planner.budget.requiredAvailableBytes - 1,
            pressure: .green,
            swapGrowthBytes: 0)
        XCTAssertNil(lowAvailability.selectedContextLength)
        XCTAssertEqual(lowAvailability.preconditionFailure, .insufficientAvailableMemory)
        XCTAssertTrue(lowAvailability.evaluations.isEmpty)

        let pressure = try planner.selectLargestComfortable(
            availableBytes: 24 * gib,
            pressure: .yellow,
            swapGrowthBytes: 0)
        XCTAssertEqual(pressure.preconditionFailure, .memoryPressureNotGreen)
        XCTAssertTrue(pressure.evaluations.isEmpty)

        let swap = try planner.selectLargestComfortable(
            availableBytes: 24 * gib,
            pressure: .green,
            swapGrowthBytes: planner.budget.maximumSwapGrowthBytes + 1)
        XCTAssertEqual(swap.preconditionFailure, .swapGrowthExceeded)
        XCTAssertTrue(swap.evaluations.isEmpty)
    }

    func testInvalidContextFailsWithoutIntegerOverflowOrAllocation() {
        let planner = ResidentMemoryPlanner.studio64GiB

        XCTAssertThrowsError(try planner.estimate(contextLength: 0)) { error in
            XCTAssertEqual(error as? ResidentMemoryPlanner.Error, .invalidContextLength)
        }
        XCTAssertThrowsError(try planner.estimate(contextLength: Int.max)) { error in
            XCTAssertEqual(error as? ResidentMemoryPlanner.Error, .arithmeticOverflow)
        }
    }

    func testRuntimeHealthGateAdmitsDrainsAndRestartsAtExactThresholds() {
        let gate = ResidentServiceHealthGate(limits: .studio64GiB)

        XCTAssertEqual(
            gate.action(for: .init(
                workerResidentBytes: 41 * gib,
                availableBytes: 9 * gib,
                pressure: .green,
                swapGrowthBytes: 256 * 1_048_576)),
            .admit)
        XCTAssertEqual(
            gate.action(for: .init(
                workerResidentBytes: 42 * gib,
                availableBytes: 9 * gib,
                pressure: .green,
                swapGrowthBytes: 0)),
            .drain(.residentLimit))
        XCTAssertEqual(
            gate.action(for: .init(
                workerResidentBytes: 41 * gib,
                availableBytes: 7 * gib,
                pressure: .green,
                swapGrowthBytes: 0)),
            .drain(.insufficientAvailableMemory))
        XCTAssertEqual(
            gate.action(for: .init(
                workerResidentBytes: 41 * gib,
                availableBytes: 9 * gib,
                pressure: .red,
                swapGrowthBytes: 0)),
            .drain(.memoryPressure))
        XCTAssertEqual(
            gate.action(for: .init(
                workerResidentBytes: 44 * gib,
                availableBytes: 9 * gib,
                pressure: .green,
                swapGrowthBytes: 0)),
            .restart(.residentKillLimit))
    }

    func testRuntimeHealthGateTreatsSwapGrowthAsDeltaNotLifetimeUsage() {
        let gate = ResidentServiceHealthGate(limits: .studio64GiB)

        XCTAssertEqual(
            gate.action(for: .init(
                workerResidentBytes: 38 * gib,
                availableBytes: 10 * gib,
                pressure: .green,
                swapGrowthBytes: 256 * 1_048_576 + 1)),
            .drain(.swapGrowth))
    }
}
