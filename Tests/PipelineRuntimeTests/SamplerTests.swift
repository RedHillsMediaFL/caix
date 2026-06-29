import XCTest

@testable import PipelineRuntime

final class SamplerTests: XCTestCase {
    private struct TestRNG: RandomNumberGenerator {
        var state: UInt64

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    func testGreedyReturnsFirstMaximum() {
        XCTAssertEqual(Sampler.argmax([1, 9, 9, 3]), 1)
    }

    func testGreedyFindsLateMaximumInLargeBuffer() {
        var logits = [Float](repeating: -3, count: 1024)
        logits[777] = 42

        XCTAssertEqual(Sampler.argmax(logits), 777)
    }

    func testEmptyInputReturnsZero() {
        var rng = TestRNG(state: 1)
        let sampler = Sampler(temperature: 1)

        XCTAssertEqual(sampler.sample([], using: &rng), 0)
    }

    func testTopKOneReturnsArgmaxWithoutFullVocabularySort() {
        var rng = TestRNG(state: 2)
        let sampler = Sampler(temperature: 1, topK: 1)

        XCTAssertEqual(sampler.sample([0, 2, 7, 3, 4], using: &rng), 2)
    }

    func testSmallTopPKeepsHighestCandidate() {
        var rng = TestRNG(state: 3)
        let sampler = Sampler(temperature: 1, topP: 0.01)

        XCTAssertEqual(sampler.sample([0, 2, 7, 3, 4], using: &rng), 2)
    }

    func testTopKAndTopPTogetherKeepHighestCandidate() {
        var rng = TestRNG(state: 4)
        let sampler = Sampler(temperature: 1, topK: 3, topP: 0.01)

        XCTAssertEqual(sampler.sample([0, 2, 7, 3, 4], using: &rng), 2)
    }
}
