import Foundation

enum WhisperNativeScalarType: String, Sendable, Equatable {
    case float16
    case int32
}

struct WhisperNativeTensorDescriptor: Sendable, Equatable {
    var scalarType: WhisperNativeScalarType
    var shape: [Int]
}

struct WhisperNativeFunctionDescriptor: Sendable, Equatable {
    var inputs: [String: WhisperNativeTensorDescriptor]
    var outputs: [String: WhisperNativeTensorDescriptor]
    var states: [String: WhisperNativeTensorDescriptor]
}

enum WhisperNativeContract {
    enum ContractError: Error, Sendable, Equatable, CustomStringConvertible {
        case invalid(String)

        var description: String {
            switch self {
            case .invalid(let reason): return "invalid Whisper native v2 contract: \(reason)"
            }
        }
    }

    static let schema = "caix.whisper-split.v2"

    static func validate(_ functions: [String: WhisperNativeFunctionDescriptor]) throws {
        let expected = expectedFunctions
        guard Set(functions.keys) == Set(expected.keys) else {
            throw ContractError.invalid(
                "entrypoints must be exactly \(expected.keys.sorted()); got \(functions.keys.sorted())")
        }

        for functionName in expected.keys.sorted() {
            let actualFunction = functions[functionName]!
            let expectedFunction = expected[functionName]!
            try validate(
                actualFunction.inputs,
                expected: expectedFunction.inputs,
                function: functionName,
                category: "inputs")
            try validate(
                actualFunction.outputs,
                expected: expectedFunction.outputs,
                function: functionName,
                category: "outputs")
            try validate(
                actualFunction.states,
                expected: expectedFunction.states,
                function: functionName,
                category: "states")
        }
    }

    private static func validate(
        _ actual: [String: WhisperNativeTensorDescriptor],
        expected: [String: WhisperNativeTensorDescriptor],
        function: String,
        category: String
    ) throws {
        guard Set(actual.keys) == Set(expected.keys) else {
            throw ContractError.invalid(
                "\(function) \(category) must be exactly \(expected.keys.sorted()); "
                    + "got \(actual.keys.sorted())")
        }
        for name in expected.keys.sorted() {
            guard actual[name] == expected[name] else {
                throw ContractError.invalid(
                    "\(function) \(category) tensor \(name) must be "
                        + "\(String(describing: expected[name]!)); got "
                        + "\(String(describing: actual[name]!))")
            }
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

    private static let expectedFunctions: [String: WhisperNativeFunctionDescriptor] = [
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
