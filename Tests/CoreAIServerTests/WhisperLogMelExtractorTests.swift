import XCTest

@testable import CoreAIServer

final class WhisperLogMelExtractorTests: XCTestCase {
    func testMatchesTransformersFiveSixLargeV2Golden() throws {
        let samples = (0..<16_000).map { index in
            Float(sin(2 * Double.pi * 440 * Double(index) / 16_000) * 0.5)
        }

        let features = try WhisperLogMelExtractor.extract(samples: samples)

        XCTAssertEqual(features.count, 80 * 3_000)
        let expected: [(mel: Int, frame: Int, value: Float)] = [
            (0, 0, 0.98327881),
            (0, 1, 0.47127187),
            (1, 0, 0.98662072),
            (10, 10, 1.34873796),
            (20, 100, 0.66739225),
            (70, 100, -0.17510831),
            (79, 0, -0.08169329),
            (0, 100, 0.83275610),
            (0, 101, 0.32075399),
            (79, 2_999, -0.56179583),
        ]
        for item in expected {
            XCTAssertEqual(
                features[item.mel * 3_000 + item.frame],
                item.value,
                accuracy: 3e-4,
                "mel=\(item.mel), frame=\(item.frame)")
        }
        XCTAssertEqual(features.min() ?? 0, -0.56179583, accuracy: 3e-4)
        XCTAssertEqual(features.max() ?? 0, 1.43820417, accuracy: 3e-4)
    }

    func testSilenceHasCanonicalWhisperFloor() throws {
        let features = try WhisperLogMelExtractor.extract(samples: [])
        XCTAssertEqual(features.count, 80 * 3_000)
        XCTAssertTrue(features.allSatisfy { abs($0 + 1.5) < 1e-6 })
    }

    func testRejectsAudioLongerThanOneWhisperWindow() {
        XCTAssertThrowsError(
            try WhisperLogMelExtractor.extract(
                samples: [Float](
                    repeating: 0,
                    count: WhisperLogMelExtractor.maximumSamples + 1)))
    }
}
