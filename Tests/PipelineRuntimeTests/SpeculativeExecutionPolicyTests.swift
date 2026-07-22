import XCTest

@testable import PipelineRuntime

final class SpeculativeExecutionPolicyTests: XCTestCase {
    func testGreedyRequestsUseMTP() {
        XCTAssertEqual(
            SpeculativeExecutionPolicy.route(temperature: 0),
            .mtpGreedy)
        XCTAssertEqual(
            SpeculativeExecutionPolicy.route(temperature: -1),
            .mtpGreedy)
    }

    func testSampledRequestsFailOverToTargetOnlyForDistributionCorrectness() {
        XCTAssertEqual(
            SpeculativeExecutionPolicy.route(temperature: 1),
            .targetOnlySampled)
        XCTAssertEqual(
            SpeculativeExecutionPolicy.route(temperature: 0.000_001),
            .targetOnlySampled)
    }

    func testPublishedGemmaDefaultsNeverEnterGreedyMTP() {
        let options = CoreAIPipeline.Options(
            temperature: 1,
            topK: 64,
            topP: 0.95,
            seed: 7)

        XCTAssertEqual(
            SpeculativeExecutionPolicy.route(options: options),
            .targetOnlySampled)
    }
}
