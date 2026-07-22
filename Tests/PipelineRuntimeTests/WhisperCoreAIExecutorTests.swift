#if COREAI_RUNTIME

import CoreAI
import XCTest

@testable import PipelineRuntime

final class WhisperCoreAIExecutorTests: XCTestCase {
    func testSessionResourcesReleaseEncoderPayloadsAfterSuccessfulLoad() {
        var resources = makeResources()
        resources.installEncoderPayloads(
            key: makeArray(),
            value: makeArray())

        XCTAssertEqual(resources.retainedEncoderPayloadCount, 2)
        XCTAssertEqual(resources.retainedStateCount, 6)

        resources.releaseEncoderPayloads()

        XCTAssertEqual(resources.retainedEncoderPayloadCount, 0)
        XCTAssertEqual(resources.retainedStateCount, 6)
    }

    func testSessionResourcesDisposeEveryArrayBeforeAdmissionHandoff() {
        var resources = makeResources()
        resources.installEncoderPayloads(
            key: makeArray(),
            value: makeArray())

        resources.dispose()

        XCTAssertEqual(resources.retainedEncoderPayloadCount, 0)
        XCTAssertEqual(resources.retainedStateCount, 0)
    }

    private func makeResources() -> WhisperCoreAISessionResources {
        WhisperCoreAISessionResources(
            crossKeyCache: makeArray(),
            crossValueCache: makeArray(),
            selfKeyCache: makeArray(),
            selfValueCache: makeArray(),
            position: makeIntArray(),
            crossReady: makeIntArray())
    }

    private func makeArray() -> NDArray {
        NDArray(shape: [1], scalarType: .float16)
    }

    private func makeIntArray() -> NDArray {
        NDArray(shape: [1], scalarType: .int32)
    }
}

#endif
