import XCTest

@testable import PipelineRuntime

final class WhisperNativeContractTests: XCTestCase {
    private struct LayoutProbe: Equatable {
        var preferredStrides: [Int]
        var isInterleaved: Bool
    }

    func testAcceptsExactV2ThreeEntrypointContract() throws {
        XCTAssertNoThrow(try WhisperNativeContract.validate(Self.validFunctions))
    }

    func testRejectsMissingEntrypointBeforeModelUse() {
        var functions = Self.validFunctions
        functions.removeValue(forKey: "load_cross_kv")
        assertRejected(functions, contains: "entrypoints")
    }

    func testRejectsUnexpectedTensorName() {
        var functions = Self.validFunctions
        let encode = functions["encode"]!
        functions["encode"] = .init(
            inputs: ["features": encode.inputs["input_features"]!],
            outputs: encode.outputs,
            states: encode.states)
        assertRejected(functions, contains: "encode inputs")
    }

    func testRejectsWrongStateShape() {
        var functions = Self.validFunctions
        var decode = functions["decode_step"]!
        decode.states["self_key_cache"] = .init(
            scalarType: .float16,
            shape: [32, 1, 20, 447, 64])
        functions["decode_step"] = decode
        assertRejected(functions, contains: "self_key_cache")
    }

    func testRejectsWrongStatusTypeAndMissingV2Output() {
        var functions = Self.validFunctions
        var load = functions["load_cross_kv"]!
        load.outputs["load_status"] = .init(scalarType: .float16, shape: [1])
        functions["load_cross_kv"] = load
        assertRejected(functions, contains: "load_status")

        functions = Self.validFunctions
        functions["decode_step"]!.outputs.removeValue(forKey: "decode_status")
        assertRejected(functions, contains: "decode_step outputs")
    }

    func testRejectsMismatchedBridgeDescriptorLayouts() {
        let packed = LayoutProbe(
            preferredStrides: [96_000_000, 96_000_000, 4_800_000, 3_200, 1],
            isInterleaved: false)
        let padded = LayoutProbe(
            preferredStrides: [96_001_024, 96_001_024, 4_800_000, 3_200, 1],
            isInterleaved: false)

        XCTAssertNoThrow(
            try WhisperNativeContract.validateExactRelationship(
                packed,
                matches: packed,
                relationship: "encode cross_key_payload -> load_cross_kv input"))
        XCTAssertThrowsError(
            try WhisperNativeContract.validateExactRelationship(
                packed,
                matches: padded,
                relationship: "encode cross_key_payload -> load_cross_kv input")
        ) { error in
            XCTAssertTrue(String(describing: error).contains("descriptor relationship"))
        }
    }

    func testRejectsNoncontiguousStateStorageBeforeZeroing() {
        XCTAssertNoThrow(
            try WhisperNativeContract.requirePackedStateStorage(
                isContiguous: true,
                stateName: "self_key_cache"))
        XCTAssertThrowsError(
            try WhisperNativeContract.requirePackedStateStorage(
                isContiguous: false,
                stateName: "self_key_cache")
        ) { error in
            XCTAssertTrue(String(describing: error).contains("self_key_cache"))
        }
    }

    private func assertRejected(
        _ functions: [String: WhisperNativeFunctionDescriptor],
        contains expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try WhisperNativeContract.validate(functions),
            file: file,
            line: line
        ) { error in
            XCTAssertTrue(
                String(describing: error).contains(expected),
                "\(error) does not contain \(expected)",
                file: file,
                line: line)
        }
    }

    private static let cross = WhisperNativeTensorDescriptor(
        scalarType: .float16,
        shape: [32, 1, 20, 1_500, 64])
    private static let selfCache = WhisperNativeTensorDescriptor(
        scalarType: .float16,
        shape: [32, 1, 20, 448, 64])
    private static let intScalar = WhisperNativeTensorDescriptor(
        scalarType: .int32,
        shape: [1])

    private static let validFunctions: [String: WhisperNativeFunctionDescriptor] = [
        "encode": .init(
            inputs: [
                "input_features": .init(
                    scalarType: .float16,
                    shape: [1, 80, 3_000])
            ],
            outputs: [
                "cross_key_payload": cross,
                "cross_value_payload": cross,
            ],
            states: [:]),
        "load_cross_kv": .init(
            inputs: [
                "cross_key_payload": cross,
                "cross_value_payload": cross,
            ],
            outputs: ["load_status": intScalar],
            states: [
                "cross_key_cache": cross,
                "cross_value_cache": cross,
                "cross_ready": intScalar,
            ]),
        "decode_step": .init(
            inputs: [
                "token_id": .init(scalarType: .int32, shape: [1, 1])
            ],
            outputs: [
                "logits": .init(
                    scalarType: .float16,
                    shape: [1, 1, 51_865]),
                "decode_status": intScalar,
            ],
            states: [
                "cross_key_cache": cross,
                "cross_value_cache": cross,
                "self_key_cache": selfCache,
                "self_value_cache": selfCache,
                "position": intScalar,
                "cross_ready": intScalar,
            ])
    ]
}
