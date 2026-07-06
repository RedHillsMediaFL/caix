import Foundation

#if COREAI_RUNTIME
import CoreAI
import CoreGraphics
import ImageIO
import Tokenizers

public struct MultimodalStagedGenerationInput: Sendable {
    public let inputIDs: [Int32]
    public let blockIDsQ: [Int32]?
    public let blockIDsKV: [Int32]?
    public let softTokenSplice: DistributedSoftTokenSplice?

    public init(
        inputIDs: [Int32],
        blockIDsQ: [Int32]? = nil,
        blockIDsKV: [Int32]? = nil,
        softTokenSplice: DistributedSoftTokenSplice?
    ) {
        self.inputIDs = inputIDs
        self.blockIDsQ = blockIDsQ
        self.blockIDsKV = blockIDsKV
        self.softTokenSplice = softTokenSplice
    }
}

public struct MultimodalStagedStepObservation: Sendable, Equatable {
    public let tokenID: Int32
    public let topLogits: [DistributedLogitScore]

    public init(tokenID: Int32, topLogits: [DistributedLogitScore]) {
        self.tokenID = tokenID
        self.topLogits = topLogits
    }
}

private struct MultimodalStagedPrefillObservation: Sendable {
    let observation: MultimodalStagedStepObservation
    let nextStepIndex: Int
}

public struct Gemma4VisionEmbedding: Sendable {
    public let values: [Float16]
    public let rows: Int
    public let hiddenSize: Int
    public let outputShape: [Int]
    public let preprocessSeconds: Double
    public let embedderSeconds: Double

    public init(
        values: [Float16],
        rows: Int,
        hiddenSize: Int,
        outputShape: [Int],
        preprocessSeconds: Double,
        embedderSeconds: Double
    ) {
        self.values = values
        self.rows = rows
        self.hiddenSize = hiddenSize
        self.outputShape = outputShape
        self.preprocessSeconds = preprocessSeconds
        self.embedderSeconds = embedderSeconds
    }
}

public struct MultimodalStagedImageGenerationResult: Sendable {
    public let result: CoreAIPipeline.Result
    public let embedding: Gemma4VisionEmbedding

    public init(result: CoreAIPipeline.Result, embedding: Gemma4VisionEmbedding) {
        self.result = result
        self.embedding = embedding
    }
}

public struct Gemma4MultimodalConfiguration: Sendable, Equatable {
    public let kind: String
    public let blockIDsRequired: Bool

    public init(kind: String, blockIDsRequired: Bool) {
        self.kind = kind
        self.blockIDsRequired = blockIDsRequired
    }
}

public final class Gemma4VisionEmbedder {
    public let assetURL: URL
    public let loadSeconds: Double
    public let computeMode: String

    private let function: InferenceFunction
    private let pixelDescriptor: NDArrayDescriptor
    private let positionDescriptor: NDArrayDescriptor
    private let outputDescriptor: NDArrayDescriptor

    private init(
        assetURL: URL,
        function: InferenceFunction,
        pixelDescriptor: NDArrayDescriptor,
        positionDescriptor: NDArrayDescriptor,
        outputDescriptor: NDArrayDescriptor,
        loadSeconds: Double,
        computeMode: String
    ) {
        self.assetURL = assetURL
        self.function = function
        self.pixelDescriptor = pixelDescriptor
        self.positionDescriptor = positionDescriptor
        self.outputDescriptor = outputDescriptor
        self.loadSeconds = loadSeconds
        self.computeMode = computeMode
    }

    public static func load(
        assetURL: URL,
        computeMode: String = ProcessInfo.processInfo.environment["CAIX_MM_EMBEDDER_COMPUTE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            ?? "gpu"
    ) async throws -> Gemma4VisionEmbedder {
        let started = Date()
        var specialization = specializationOptions(computeMode: computeMode)
        specialization.expectFrequentReshapes = true
        let model = try await AIModel.specialize(
            contentsOf: assetURL.standardizedFileURL,
            options: specialization,
            cache: .default,
            cachePolicy: .persistent)
        guard let descriptor = model.functionDescriptor(for: "vision") else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Gemma 4 vision embedder function 'vision' not found in \(assetURL.lastPathComponent); have \(model.functionNames)")
        }
        guard case .ndArray(let pixelDescriptor) = descriptor.inputDescriptor(of: "pixel_values"),
              case .ndArray(let positionDescriptor) = descriptor.inputDescriptor(of: "image_position_ids"),
              case .ndArray(let outputDescriptor) = descriptor.outputDescriptor(of: "mm_embeds")
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Gemma 4 vision embedder must expose pixel_values, image_position_ids, and mm_embeds NDArrays")
        }
        guard let function = try model.loadFunction(named: "vision") else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "could not load Gemma 4 vision embedder function 'vision'")
        }
        return Gemma4VisionEmbedder(
            assetURL: assetURL.standardizedFileURL,
            function: function,
            pixelDescriptor: pixelDescriptor,
            positionDescriptor: positionDescriptor,
            outputDescriptor: outputDescriptor,
            loadSeconds: Date().timeIntervalSince(started),
            computeMode: computeMode)
    }

    public func embed(imageData: Data) async throws -> Gemma4VisionEmbedding {
        let preprocessStarted = Date()
        let processed = try Gemma4MultimodalProcessor.preprocessImage(
            data: imageData,
            pixelValuesShape: pixelDescriptor.shape,
            imagePositionIDsShape: positionDescriptor.shape)
        let preprocessSeconds = Date().timeIntervalSince(preprocessStarted)
        let embedStarted = Date()
        let rows = try await embed(processed)
        return Gemma4VisionEmbedding(
            values: rows.values,
            rows: processed.validRows,
            hiddenSize: rows.shape[2],
            outputShape: rows.shape,
            preprocessSeconds: preprocessSeconds,
            embedderSeconds: Date().timeIntervalSince(embedStarted))
    }

    public func embed(_ processed: Gemma4ImageTensors) async throws -> (shape: [Int], values: [Float16]) {
        guard processed.pixelValuesShape == pixelDescriptor.shape else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "pixel_values shape \(processed.pixelValuesShape) does not match embedder \(pixelDescriptor.shape)")
        }
        guard processed.imagePositionIDs.count == positionDescriptor.shape.reduce(1, *) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "image_position_ids count \(processed.imagePositionIDs.count) does not match embedder \(positionDescriptor.shape)")
        }
        var inputs: [String: NDArray] = [:]
        inputs["pixel_values"] = try Self.floatNDArray(
            shape: processed.pixelValuesShape,
            scalarType: pixelDescriptor.scalarType,
            values: processed.pixelValues,
            tensorName: "pixel_values")
        inputs["image_position_ids"] = try Self.int32NDArray(
            shape: positionDescriptor.shape,
            scalarType: positionDescriptor.scalarType,
            values: processed.imagePositionIDs,
            tensorName: "image_position_ids")
        var outputs = try await function.run(inputs: inputs)
        guard let output = outputs.remove("mm_embeds")?.ndArray else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Gemma 4 vision embedder did not produce mm_embeds")
        }
        let rows = try Self.readRank3AsFloat16(output, tensorName: "mm_embeds")
        guard rows.shape.count == 3,
              rows.shape[0] == 1,
              rows.shape[1] >= processed.validRows,
              rows.shape[2] > 0
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Gemma 4 vision embedder output shape \(rows.shape) is invalid")
        }
        guard outputDescriptor.shape.count == 3 else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Gemma 4 vision embedder mm_embeds descriptor shape \(outputDescriptor.shape) is invalid")
        }
        return rows
    }

    private static func specializationOptions(computeMode: String) -> SpecializationOptions {
        switch computeMode {
        case "cpu", "cpuonly", "cpu_only":
            return .cpuOnly
        case "default", "all", "cpu_gpu", "cpugpu", "cpuandgpu", "cpu_and_gpu":
            return .default
        case "ane", "neuralengine", "neural_engine":
            return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        default:
            return SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
    }

    private static func floatNDArray(
        shape: [Int],
        scalarType: NDArray.ScalarType,
        values: [Float],
        tensorName: String
    ) throws -> NDArray {
        guard values.count == shape.reduce(1, *) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) value count \(values.count) does not match shape \(shape)")
        }
        var array = NDArray(shape: shape, scalarType: scalarType)
        switch scalarType {
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { pointer, _, _ in
                for index in values.indices { pointer[index] = Float16(values[index]) }
            }
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { pointer, _, _ in
                pointer.initialize(from: values, count: values.count)
            }
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(scalarType) is not supported for Gemma 4 vision embedding")
        }
        return array
    }

    private static func int32NDArray(
        shape: [Int],
        scalarType: NDArray.ScalarType,
        values: [Int32],
        tensorName: String
    ) throws -> NDArray {
        guard values.count == shape.reduce(1, *) else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) value count \(values.count) does not match shape \(shape)")
        }
        var array = NDArray(shape: shape, scalarType: scalarType)
        switch scalarType {
        case .int32:
            var view = array.mutableView(as: Int32.self)
            view.withUnsafeMutablePointer { pointer, _, _ in
                pointer.initialize(from: values, count: values.count)
            }
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(scalarType) is not supported for Gemma 4 vision embedding")
        }
        return array
    }

    private static func readRank3AsFloat16(
        _ array: NDArray,
        tensorName: String
    ) throws -> (shape: [Int], values: [Float16]) {
        switch array.scalarType {
        case .float16:
            return try array.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
                guard shape.count == 3 else {
                    throw CoreAIPipeline.RuntimeError.modelContract(
                        "\(tensorName) rank \(shape.count) does not match expected rank 3")
                }
                let outputShape = [shape[0], shape[1], shape[2]]
                var values: [Float16] = []
                values.reserveCapacity(outputShape.reduce(1, *))
                for i in 0..<shape[0] {
                    for j in 0..<shape[1] {
                        for k in 0..<shape[2] {
                            values.append(pointer[i * strides[0] + j * strides[1] + k * strides[2]])
                        }
                    }
                }
                return (outputShape, values)
            }
        case .float32:
            return try array.view(as: Float.self).withUnsafePointer { pointer, shape, strides in
                guard shape.count == 3 else {
                    throw CoreAIPipeline.RuntimeError.modelContract(
                        "\(tensorName) rank \(shape.count) does not match expected rank 3")
                }
                let outputShape = [shape[0], shape[1], shape[2]]
                var values: [Float16] = []
                values.reserveCapacity(outputShape.reduce(1, *))
                for i in 0..<shape[0] {
                    for j in 0..<shape[1] {
                        for k in 0..<shape[2] {
                            values.append(Float16(pointer[i * strides[0] + j * strides[1] + k * strides[2]]))
                        }
                    }
                }
                return (outputShape, values)
            }
        default:
            throw CoreAIPipeline.RuntimeError.modelContract(
                "\(tensorName) scalar type \(array.scalarType) is not supported for Gemma 4 vision embedding")
        }
    }
}

public final class MultimodalStagedModel {
    public let manifestURL: URL
    public let manifest: DistributedStageManifest
    public let maxContextLength: Int
    public let loadSeconds: Double

    private let pipeline: DistributedSameMachinePipeline
    private let tokenizer: any Tokenizer
    private let tokenizerDir: URL
    private let stopTokenIDs: Set<Int32>
    private let multimodalConfig: Gemma4MultimodalConfiguration
    private let padTokenID: Int32

    public var imageTextSeparator: String {
        multimodalConfig.kind == "gemma4" ? "" : "\n"
    }

    private init(
        manifestURL: URL,
        manifest: DistributedStageManifest,
        pipeline: DistributedSameMachinePipeline,
        tokenizer: any Tokenizer,
        tokenizerDir: URL,
        multimodalConfig: Gemma4MultimodalConfiguration,
        maxContextLength: Int,
        padTokenID: Int32,
        loadSeconds: Double
    ) {
        self.manifestURL = manifestURL
        self.manifest = manifest
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.tokenizerDir = tokenizerDir
        self.multimodalConfig = multimodalConfig
        self.maxContextLength = maxContextLength
        self.padTokenID = padTokenID
        self.loadSeconds = loadSeconds
        self.stopTokenIDs = Set(
            LLMEngine.stopTokenIds(tokenizer: tokenizer, tokenizerDir: tokenizerDir).map(Int32.init))
    }

    public static func load(
        manifestURL: URL,
        verbose: Bool = false
    ) async throws -> MultimodalStagedModel {
        let started = Date()
        let resolvedManifestURL = manifestURL.standardizedFileURL
        let manifest = try DistributedStageManifest.load(from: resolvedManifestURL)
        let multimodalConfig = try readMultimodalConfiguration(manifestURL: resolvedManifestURL)
        try validateMultimodalManifest(manifest, config: multimodalConfig)
        let root = resolvedManifestURL.deletingLastPathComponent()
        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        let maxContextLength = try readMaxContextLength(bundleRoot: root)
        let padTokenID = readPadTokenID(tokenizerDir: tokenizerDir)
        if verbose {
            FileHandle.standardError.write(
                Data("[mm-staged] loading \(resolvedManifestURL.path)\n".utf8))
        }
        async let pipelineTask = DistributedSameMachinePipeline.make(
            manifest: manifest,
            handleFactory: DistributedCoreAIStageHandleFactory())
        async let tokenizerTask = AutoTokenizer.from(modelFolder: tokenizerDir)
        let pipeline = try await pipelineTask
        let tokenizer = try await tokenizerTask
        return MultimodalStagedModel(
            manifestURL: resolvedManifestURL,
            manifest: manifest,
            pipeline: pipeline,
            tokenizer: tokenizer,
            tokenizerDir: tokenizerDir,
            multimodalConfig: multimodalConfig,
            maxContextLength: maxContextLength,
            padTokenID: padTokenID,
            loadSeconds: Date().timeIntervalSince(started))
    }

    public func generate(
        inputIDs: [Int32],
        blockIDsQ: [Int32]? = nil,
        blockIDsKV: [Int32]? = nil,
        softTokenSplice: DistributedSoftTokenSplice?,
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        let input = MultimodalStagedGenerationInput(
            inputIDs: inputIDs,
            blockIDsQ: blockIDsQ,
            blockIDsKV: blockIDsKV,
            softTokenSplice: softTokenSplice)
        return try await generate(input: input, options: options, onToken: onToken)
    }

    public func makeTextInput(
        messages: [[String: String]],
        applyChatTemplate: Bool = true
    ) throws -> MultimodalStagedGenerationInput {
        let tokens: [Int]
        if applyChatTemplate {
            tokens = try tokenizer.applyChatTemplate(messages: messages)
        } else {
            tokens = tokenizer.encode(text: messages.map { $0["content"] ?? "" }.joined())
        }
        guard !tokens.isEmpty else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("prompt tokenized to 0 tokens")
        }
        let inputIDs = tokens.map(Int32.init)
        let blockIDs = multimodalConfig.blockIDsRequired
            ? Array(repeating: Int32(-1), count: inputIDs.count)
            : nil
        return MultimodalStagedGenerationInput(
            inputIDs: inputIDs,
            blockIDsQ: blockIDs,
            blockIDsKV: blockIDs,
            softTokenSplice: nil)
    }

    public func makeSingleImageInput(
        messages: [[String: String]],
        embedding: Gemma4VisionEmbedding
    ) throws -> MultimodalStagedGenerationInput {
        guard let hiddenSize = manifest.boundaryTensor?.shape.last,
              hiddenSize == embedding.hiddenSize
        else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Gemma 4 image embedding hidden size \(embedding.hiddenSize) does not match staged boundary \(manifest.boundaryTensor?.shape.last ?? -1)")
        }
        guard embedding.values.count >= embedding.rows * embedding.hiddenSize else {
            throw CoreAIPipeline.RuntimeError.modelContract(
                "Gemma 4 image embedding has \(embedding.values.count) values for \(embedding.rows)x\(embedding.hiddenSize)")
        }
        let prompt = try Gemma4MultimodalProcessor.buildSingleImagePrompt(
            messages: messages,
            tokenizer: tokenizer,
            imageRows: embedding.rows,
            blockIDsRequired: multimodalConfig.blockIDsRequired)
        let rowValues = Array(embedding.values.prefix(embedding.rows * embedding.hiddenSize))
        let splice = try DistributedSoftTokenSplice(
            positionStart: prompt.imageStart,
            rows: prompt.imageRows,
            hiddenSize: embedding.hiddenSize,
            float16Values: rowValues)
        return MultimodalStagedGenerationInput(
            inputIDs: prompt.inputIDs,
            blockIDsQ: multimodalConfig.blockIDsRequired ? prompt.blockIDsQ : nil,
            blockIDsKV: multimodalConfig.blockIDsRequired ? prompt.blockIDsKV : nil,
            softTokenSplice: splice)
    }

    public func generate(
        messages: [[String: String]],
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        let input = try makeTextInput(
            messages: messages,
            applyChatTemplate: options.applyChatTemplate)
        return try await generate(input: input, options: options, onToken: onToken)
    }

    public func generateSingleImage(
        messages: [[String: String]],
        imageData: Data,
        embedder: Gemma4VisionEmbedder,
        options: CoreAIPipeline.Options,
        onToken: ((String) -> Void)? = nil
    ) async throws -> MultimodalStagedImageGenerationResult {
        guard options.applyChatTemplate else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "multimodal Gemma serving requires the bundled chat template")
        }
        let embedding = try await embedder.embed(imageData: imageData)
        let input = try makeSingleImageInput(messages: messages, embedding: embedding)
        if options.verbose {
            logSingleImageInput(input)
        }
        let result = try await generate(input: input, options: options, onToken: onToken)
        return MultimodalStagedImageGenerationResult(result: result, embedding: embedding)
    }

    private func logSingleImageInput(_ input: MultimodalStagedGenerationInput) {
        let imageCount = input.inputIDs.filter { $0 == Gemma4MultimodalProcessor.imageTokenID }.count
        let boiPositions = input.inputIDs.enumerated()
            .filter { $0.element == Gemma4MultimodalProcessor.boiTokenID }
            .map(\.offset)
        let eoiPositions = input.inputIDs.enumerated()
            .filter { $0.element == Gemma4MultimodalProcessor.eoiTokenID }
            .map(\.offset)
        let splice = input.softTokenSplice
        let spliceStart = splice?.positionStart ?? -1
        let spliceEnd = splice?.positionEnd ?? -1
        let spliceRows = max(0, spliceEnd - spliceStart)
        let blockIDStatus = input.blockIDsQ == nil ? "none" : "present"
        let line = [
            "[mm-staged]",
            "prompt_tokens=\(input.inputIDs.count)",
            "image_token_count=\(imageCount)",
            "boi_positions=\(boiPositions)",
            "eoi_positions=\(eoiPositions)",
            "splice=\(spliceStart)..<\(spliceEnd)",
            "splice_rows=\(spliceRows)",
            "block_ids=\(blockIDStatus)",
        ].joined(separator: " ") + "\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    public func generate(
        input: MultimodalStagedGenerationInput,
        options: CoreAIPipeline.Options,
        requestID: String = UUID().uuidString,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        try validate(input)
        guard options.maxTokens >= 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "max_tokens must be non-negative")
        }
        guard options.temperature <= 0 else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "multimodal staged serving currently supports greedy decoding only")
        }
        guard options.constrainedJSONSchema == nil else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "multimodal staged serving does not support constrained decoding yet")
        }
        guard options.maxTokens > 0 else {
            return CoreAIPipeline.Result(
                text: "",
                promptTokenCount: input.inputIDs.count,
                generatedTokenCount: 0,
                stopReason: .maxTokens,
                modelLoadSeconds: loadSeconds,
                prefillSeconds: 0,
                decodeSeconds: 0,
                generatedTokenIDs: [])
        }

        let capacity = try resolvedKVCapacity(
            promptCount: input.inputIDs.count,
            maxTokens: options.maxTokens,
            explicitKVCapacity: options.kvCapacity)
        let cacheCapacities = resolvedCacheCapacities(kvCapacity: capacity)
        let started = Date()
        try await pipeline.allocate(
            requestID: requestID,
            kvCapacity: capacity,
            cacheCapacities: cacheCapacities)
        do {
            let prefillStart = Date()
            let prefill = try await prefillObservation(
                input: input,
                requestID: requestID)
            var next = prefill.observation
            let prefillSeconds = Date().timeIntervalSince(prefillStart)

            var generated: [Int32] = []
            var streamedText = ""
            var finalTextOverride: String?
            var stopReason = CoreAIPipeline.StopReason.maxTokens
            let decodeStart = Date()

            func emitIfNeeded() {
                guard let onToken else { return }
                let text = tokenizer.decode(tokens: generated.map(Int.init))
                if text.hasPrefix(streamedText) {
                    let delta = String(text.dropFirst(streamedText.count))
                    if !delta.isEmpty { onToken(delta) }
                }
                streamedText = text
            }

            while generated.count < options.maxTokens {
                if stopTokenIDs.contains(next.tokenID) {
                    stopReason = .eos
                    break
                }
                generated.append(next.tokenID)
                emitIfNeeded()

                let decoded = tokenizer.decode(tokens: generated.map(Int.init))
                if let stopRange = CoreAIPipeline.firstStopRange(
                    in: decoded,
                    stopSequences: options.stopSequences)
                {
                    stopReason = .stopSequence
                    finalTextOverride = String(decoded[..<stopRange.lowerBound])
                    break
                }

                guard generated.count < options.maxTokens else { break }
                if input.inputIDs.count + generated.count >= maxContextLength {
                    stopReason = .contextLimit
                    break
                }

                let decodePosition = input.inputIDs.count + generated.count - 1
                let decodeStepIndex = prefill.nextStepIndex + generated.count - 1
                next = try await nextTokenObservation(
                    requestID: requestID,
                    stepIndex: decodeStepIndex,
                    positionRange: DistributedSequenceRange(
                        lowerBound: decodePosition,
                        upperBound: decodePosition + 1),
                    tokenIDs: [next.tokenID],
                    blockIDsQ: nil,
                    blockIDsKV: nil,
                    softTokenSplice: nil)
            }

            await pipeline.free(requestID: requestID)
            let decodeSeconds = Date().timeIntervalSince(decodeStart)
            let finalText = finalTextOverride ?? tokenizer.decode(tokens: generated.map(Int.init))
            _ = started
            return CoreAIPipeline.Result(
                text: finalText,
                promptTokenCount: input.inputIDs.count,
                generatedTokenCount: generated.count,
                stopReason: stopReason,
                modelLoadSeconds: loadSeconds,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeSeconds,
                generatedTokenIDs: generated)
        } catch {
            await pipeline.free(requestID: requestID)
            throw error
        }
    }

    public func teacherForcedTop1Tokens(
        inputIDs: [Int32],
        blockIDsQ: [Int32]? = nil,
        blockIDsKV: [Int32]? = nil,
        softTokenSplice: DistributedSoftTokenSplice?,
        referenceTokenIDs: [Int32],
        requestID: String = UUID().uuidString
    ) async throws -> [MultimodalStagedStepObservation] {
        let input = MultimodalStagedGenerationInput(
            inputIDs: inputIDs,
            blockIDsQ: blockIDsQ,
            blockIDsKV: blockIDsKV,
            softTokenSplice: softTokenSplice)
        return try await teacherForcedTop1Tokens(
            input: input,
            referenceTokenIDs: referenceTokenIDs,
            requestID: requestID)
    }

    public func teacherForcedTop1Tokens(
        input: MultimodalStagedGenerationInput,
        referenceTokenIDs: [Int32],
        requestID: String = UUID().uuidString
    ) async throws -> [MultimodalStagedStepObservation] {
        try validate(input)
        guard !referenceTokenIDs.isEmpty else { return [] }
        let capacity = try resolvedKVCapacity(
            promptCount: input.inputIDs.count,
            maxTokens: referenceTokenIDs.count,
            explicitKVCapacity: nil)
        let cacheCapacities = resolvedCacheCapacities(kvCapacity: capacity)
        try await pipeline.allocate(
            requestID: requestID,
            kvCapacity: capacity,
            cacheCapacities: cacheCapacities)
        do {
            var observations: [MultimodalStagedStepObservation] = []
            observations.reserveCapacity(referenceTokenIDs.count)
            let prefill = try await prefillObservation(
                input: input,
                requestID: requestID)
            observations.append(prefill.observation)

            for step in 1..<referenceTokenIDs.count {
                let decodePosition = input.inputIDs.count + step - 1
                observations.append(try await nextTokenObservation(
                    requestID: requestID,
                    stepIndex: prefill.nextStepIndex + step - 1,
                    positionRange: DistributedSequenceRange(
                        lowerBound: decodePosition,
                        upperBound: decodePosition + 1),
                    tokenIDs: [referenceTokenIDs[step - 1]],
                    blockIDsQ: nil,
                    blockIDsKV: nil,
                    softTokenSplice: nil))
            }
            await pipeline.free(requestID: requestID)
            return observations
        } catch {
            await pipeline.free(requestID: requestID)
            throw error
        }
    }

    public func decode(tokens: [Int32], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: tokens.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }

    private func nextTokenObservation(
        input: MultimodalStagedGenerationInput,
        requestID: String,
        stepIndex: Int
    ) async throws -> MultimodalStagedStepObservation {
        try await nextTokenObservation(
            requestID: requestID,
            stepIndex: stepIndex,
            positionRange: DistributedSequenceRange(lowerBound: 0, upperBound: input.inputIDs.count),
            tokenIDs: input.inputIDs,
            transformerTokenIDs: try transformerTokenIDs(for: input),
            blockIDsQ: input.blockIDsQ,
            blockIDsKV: input.blockIDsKV,
            softTokenSplice: input.softTokenSplice)
    }

    private func prefillObservation(
        input: MultimodalStagedGenerationInput,
        requestID: String
    ) async throws -> MultimodalStagedPrefillObservation {
        guard let chunkSize = resolvedPrefillChunkSize(promptCount: input.inputIDs.count),
            chunkSize < input.inputIDs.count
        else {
            return MultimodalStagedPrefillObservation(
                observation: try await nextTokenObservation(
                    input: input,
                    requestID: requestID,
                    stepIndex: 0),
                nextStepIndex: 1)
        }

        var lowerBound = 0
        var stepIndex = 0
        var observation: MultimodalStagedStepObservation?
        let maskedTransformerTokenIDs = try transformerTokenIDs(for: input)
        while lowerBound < input.inputIDs.count {
            let upperBound = min(input.inputIDs.count, lowerBound + chunkSize)
            let range = lowerBound..<upperBound
            let positionRange = DistributedSequenceRange(
                lowerBound: lowerBound,
                upperBound: upperBound)
            observation = try await nextTokenObservation(
                requestID: requestID,
                stepIndex: stepIndex,
                positionRange: positionRange,
                tokenIDs: Array(input.inputIDs[range]),
                transformerTokenIDs: maskedTransformerTokenIDs.map { Array($0[range]) },
                blockIDsQ: input.blockIDsQ.map { Array($0[range]) },
                blockIDsKV: input.blockIDsKV.map { Array($0[..<upperBound]) },
                softTokenSplice: try softTokenSplice(
                    input.softTokenSplice,
                    for: positionRange,
                    chunkSize: chunkSize))
            lowerBound = upperBound
            stepIndex += 1
        }
        guard let observation else {
            throw DistributedStageExecutionError.invalidForwardInput("chunked prefill produced no observations")
        }
        return MultimodalStagedPrefillObservation(
            observation: observation,
            nextStepIndex: stepIndex)
    }

    private func resolvedPrefillChunkSize(promptCount: Int) -> Int? {
        guard promptCount > 1 else {
            return nil
        }
        if let raw = ProcessInfo.processInfo.environment["COREAI_MM_PREFILL_CHUNK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty
        {
            guard let value = Int(raw), value > 0 else { return nil }
            return max(1, value)
        }
        return manifest.cacheGroups?.prefillChunk
    }

    private func softTokenSplice(
        _ splice: DistributedSoftTokenSplice?,
        for positionRange: DistributedSequenceRange,
        chunkSize: Int
    ) throws -> DistributedSoftTokenSplice? {
        guard let splice else { return nil }
        if splice.positionStart >= positionRange.lowerBound,
            splice.positionEnd <= positionRange.upperBound
        {
            return splice
        }
        if splice.positionEnd <= positionRange.lowerBound
            || splice.positionStart >= positionRange.upperBound
        {
            return nil
        }
        throw DistributedStageExecutionError.invalidForwardInput(
            "soft_token_splice crosses prefill chunk boundary; increase COREAI_MM_PREFILL_CHUNK above \(chunkSize)")
    }

    private func transformerTokenIDs(
        for input: MultimodalStagedGenerationInput
    ) throws -> [Int32]? {
        guard multimodalConfig.kind == "gemma4",
              let softTokenSplice = input.softTokenSplice
        else {
            return nil
        }
        guard softTokenSplice.positionStart >= 0,
              softTokenSplice.positionEnd <= input.inputIDs.count
        else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "soft_token_splice position range must be inside input_ids")
        }

        var masked = input.inputIDs
        var replaced = 0
        for index in softTokenSplice.positionStart..<softTokenSplice.positionEnd
            where masked[index] == Gemma4MultimodalProcessor.imageTokenID
        {
            masked[index] = padTokenID
            replaced += 1
        }
        guard replaced == softTokenSplice.rowCount else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "gemma4 image splice expected \(softTokenSplice.rowCount) image-token ids in the splice span but found \(replaced)")
        }
        return masked
    }

    private func nextTokenObservation(
        requestID: String,
        stepIndex: Int,
        positionRange: DistributedSequenceRange,
        tokenIDs: [Int32],
        transformerTokenIDs: [Int32]? = nil,
        blockIDsQ: [Int32]?,
        blockIDsKV: [Int32]?,
        softTokenSplice: DistributedSoftTokenSplice?
    ) async throws -> MultimodalStagedStepObservation {
        let output = try await pipeline.forward(
            requestID: requestID,
            stepIndex: stepIndex,
            positionRange: positionRange,
            tokenIDs: tokenIDs,
            transformerTokenIDs: transformerTokenIDs,
            blockIDsQ: blockIDsQ,
            blockIDsKV: blockIDsKV,
            softTokenSplice: softTokenSplice)
        guard let tokenID = output.tokenID else {
            throw DistributedStageExecutionError.invalidStageOutput(
                "multimodal staged pipeline did not return a token id")
        }
        return MultimodalStagedStepObservation(tokenID: tokenID, topLogits: output.topLogits)
    }

    private func validate(_ input: MultimodalStagedGenerationInput) throws {
        guard !input.inputIDs.isEmpty else {
            throw DistributedStageExecutionError.invalidForwardInput("input_ids must be non-empty")
        }
        if multimodalConfig.blockIDsRequired {
            guard let blockIDsQ = input.blockIDsQ else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "block_ids_q is required by this multimodal staged bundle")
            }
            guard let blockIDsKV = input.blockIDsKV else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "block_ids_kv is required by this multimodal staged bundle")
            }
            guard blockIDsQ.count == input.inputIDs.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "block_ids_q count must match input_ids")
            }
            guard blockIDsKV.count == input.inputIDs.count else {
                throw DistributedStageExecutionError.invalidForwardInput(
                    "block_ids_kv count must match input_ids")
            }
        } else if input.blockIDsQ != nil || input.blockIDsKV != nil {
            throw DistributedStageExecutionError.invalidForwardInput(
                "block_ids are not accepted by this causal multimodal staged bundle")
        }
        if let softTokenSplice = input.softTokenSplice,
            let hiddenSize = manifest.boundaryTensor?.shape.last,
            hiddenSize > 0,
            softTokenSplice.hiddenSize != hiddenSize
        {
            throw DistributedStageExecutionError.invalidForwardInput(
                "soft_token_splice hidden size must match staged boundary")
        }
    }

    private func resolvedKVCapacity(
        promptCount: Int,
        maxTokens: Int,
        explicitKVCapacity: Int?
    ) throws -> Int {
        let requested = explicitKVCapacity ?? (promptCount + maxTokens + 8)
        guard requested > 0 else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "kv_capacity must be positive")
        }
        let capacity = min(requested, maxContextLength)
        guard capacity >= promptCount else {
            throw DistributedStageExecutionError.invalidForwardInput(
                "kv_capacity is smaller than prompt")
        }
        return capacity
    }

    private func resolvedCacheCapacities(kvCapacity: Int) -> [String: Int]? {
        manifest.cacheGroups?.capacities(forKVCapacity: kvCapacity)
    }

    private static func validateMultimodalManifest(
        _ manifest: DistributedStageManifest,
        config: Gemma4MultimodalConfiguration
    ) throws {
        guard manifest.positionMode == .fullPrefix else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "multimodal staged serving expects full_prefix position mode")
        }
        guard manifest.runtimePlan.stages.contains(where: { $0.role == .embeddings }),
              manifest.runtimePlan.stages.contains(where: { $0.role == .finalNormHead }),
              manifest.runtimePlan.stages.contains(where: { $0.role == .transformerLayers })
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "multimodal staged serving expects embeddings, transformer_layers, and final_norm_head stages")
        }
        for stage in manifest.runtimePlan.stages where stage.role == .transformerLayers {
            if config.blockIDsRequired {
                guard stage.prefillExtraInputs == ["block_ids_q", "block_ids_kv"] else {
                    throw CoreAIPipeline.RuntimeError.invalidBundle(
                        "multimodal transformer stage \(stage.id) must declare block_ids_q/block_ids_kv")
                }
            } else {
                guard stage.prefillExtraInputs.isEmpty else {
                    throw CoreAIPipeline.RuntimeError.invalidBundle(
                        "causal multimodal transformer stage \(stage.id) must not declare block_ids inputs")
                }
            }
        }
    }

    private static func readMultimodalConfiguration(
        manifestURL: URL
    ) throws -> Gemma4MultimodalConfiguration {
        guard let data = try? Data(contentsOf: manifestURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let multimodal = object["multimodal"] as? [String: Any],
              let kind = (multimodal["kind"] as? String)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !kind.isEmpty
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "multimodal staged bundle is missing stage-manifest.json multimodal.kind")
        }
        guard kind == "gemma4_unified" || kind == "gemma4" else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "unsupported multimodal staged kind \(kind)")
        }
        let defaultBlockIDsRequired = kind == "gemma4_unified"
        let blockIDsRequired = multimodal["block_ids_required"] as? Bool
            ?? defaultBlockIDsRequired
        return Gemma4MultimodalConfiguration(
            kind: kind,
            blockIDsRequired: blockIDsRequired)
    }

    private static func readMultimodalBlockIDsRequired(
        manifestURL: URL
    ) throws -> Bool {
        try readMultimodalConfiguration(manifestURL: manifestURL).blockIDsRequired
    }

    private static func readMultimodalBlockIDsRequired(
        bundleRoot: URL
    ) throws -> Bool {
        try readMultimodalBlockIDsRequired(
            manifestURL: bundleRoot.appendingPathComponent("stage-manifest.json"))
    }

    public static func blockIDsRequired(manifestURL: URL) throws -> Bool {
        try readMultimodalBlockIDsRequired(manifestURL: manifestURL)
    }

    public static func blockIDsRequired(bundleRoot: URL) throws -> Bool {
        try readMultimodalBlockIDsRequired(bundleRoot: bundleRoot)
    }

    private static func readMultimodalKind(bundleRoot: URL) throws -> String {
        try readMultimodalConfiguration(
            manifestURL: bundleRoot.appendingPathComponent("stage-manifest.json")).kind
    }

    public static func multimodalKind(bundleRoot: URL) throws -> String {
        try readMultimodalKind(bundleRoot: bundleRoot)
    }

    private static func assertSupportedMultimodalKind(_ kind: String) throws {
        guard kind == "gemma4_unified" || kind == "gemma4" else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "unsupported multimodal staged kind \(kind)")
        }
    }

    public static func validateKind(_ kind: String) throws {
        try assertSupportedMultimodalKind(kind)
    }

    private static func requireBlockIDs(_ config: Gemma4MultimodalConfiguration) throws {
        if config.kind == "gemma4_unified" && !config.blockIDsRequired {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "gemma4_unified multimodal bundles must declare block_ids_required=true")
        }
    }

    public static func validateConfiguration(_ config: Gemma4MultimodalConfiguration) throws {
        try assertSupportedMultimodalKind(config.kind)
        try requireBlockIDs(config)
    }

    public static func configuration(manifestURL: URL) throws -> Gemma4MultimodalConfiguration {
        let config = try readMultimodalConfiguration(manifestURL: manifestURL)
        try validateConfiguration(config)
        return config
    }

    public static func configuration(bundleRoot: URL) throws -> Gemma4MultimodalConfiguration {
        try configuration(manifestURL: bundleRoot.appendingPathComponent("stage-manifest.json"))
    }

    private static func readMaxContextLength(bundleRoot: URL) throws -> Int {
        let metadataURL = bundleRoot.appendingPathComponent("metadata.json")
        guard let data = try? Data(contentsOf: metadataURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let language = object["language"] as? [String: Any]
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "multimodal staged bundle is missing metadata.json language block")
        }
        if let value = language["max_context_length"] as? Int {
            return value
        }
        if let value = language["max_context_length"] as? Double {
            return Int(value)
        }
        throw CoreAIPipeline.RuntimeError.invalidBundle(
            "multimodal staged bundle is missing language.max_context_length")
    }

    private static func readPadTokenID(tokenizerDir: URL) -> Int32 {
        let configURL = tokenizerDir.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return 0
        }
        if let value = object["pad_token_id"] as? Int {
            return Int32(value)
        }
        let padToken = (object["pad_token"] as? String) ?? "<pad>"
        if let decoder = object["added_tokens_decoder"] as? [String: Any] {
            for (key, value) in decoder {
                guard let id = Int32(key),
                      let entry = value as? [String: Any],
                      entry["content"] as? String == padToken
                else {
                    continue
                }
                return id
            }
        }
        return 0
    }
}

public struct Gemma4MultimodalPrompt: Sendable, Equatable {
    public let inputIDs: [Int32]
    public let imageStart: Int
    public let imageRows: Int
    public let blockIDsQ: [Int32]
    public let blockIDsKV: [Int32]

    public init(
        inputIDs: [Int32],
        imageStart: Int,
        imageRows: Int,
        blockIDsQ: [Int32],
        blockIDsKV: [Int32]
    ) {
        self.inputIDs = inputIDs
        self.imageStart = imageStart
        self.imageRows = imageRows
        self.blockIDsQ = blockIDsQ
        self.blockIDsKV = blockIDsKV
    }
}

public struct Gemma4ImageTensors: Sendable, Equatable {
    public let pixelValuesShape: [Int]
    public let pixelValues: [Float]
    public let imagePositionIDsShape: [Int]
    public let imagePositionIDs: [Int32]
    public let validRows: Int

    public init(
        pixelValuesShape: [Int],
        pixelValues: [Float],
        imagePositionIDsShape: [Int],
        imagePositionIDs: [Int32],
        validRows: Int
    ) {
        self.pixelValuesShape = pixelValuesShape
        self.pixelValues = pixelValues
        self.imagePositionIDsShape = imagePositionIDsShape
        self.imagePositionIDs = imagePositionIDs
        self.validRows = validRows
    }
}

public enum Gemma4MultimodalProcessor {
    public static let imageTokenID: Int32 = 258_880
    public static let boiTokenID: Int32 = 255_999
    public static let eoiTokenID: Int32 = 258_882
    public static let patchSize = 16
    public static let poolingKernelSize = 3
    public static let maxSoftTokens = 280
    public static let channels = 3

    public static func buildSingleImagePrompt(
        messages: [[String: String]],
        tokenizer: any Tokenizer,
        imageRows: Int,
        blockIDsRequired: Bool = true
    ) throws -> Gemma4MultimodalPrompt {
        guard imageRows > 0 else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("imageRows must be positive")
        }
        var raw = try tokenizer.applyChatTemplate(messages: messages)
        // swift-transformers 1.3.3 preserves one leading template newline after <bos> for this
        // external Gemma 4 chat_template.jinja. HF strips it; normalize before media expansion.
        if raw.count >= 3, raw[0] == 2, raw[1] == 107, raw[2] == 105 {
            raw.remove(at: 1)
        }
        let imagePositions = raw.indices.filter { raw[$0] == Int(imageTokenID) }
        guard imagePositions.count == 1, let imageTokenIndex = imagePositions.first else {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "multimodal serving currently supports exactly one image placeholder")
        }
        var inputIDs: [Int32] = []
        inputIDs.reserveCapacity(raw.count + imageRows + 1)
        for (index, token) in raw.enumerated() {
            if index == imageTokenIndex {
                inputIDs.append(boiTokenID)
                inputIDs.append(contentsOf: Array(repeating: imageTokenID, count: imageRows))
                inputIDs.append(eoiTokenID)
            } else {
                inputIDs.append(Int32(token))
            }
        }
        let imageStart = imageTokenIndex + 1
        var blockIDs = Array(repeating: Int32(-1), count: inputIDs.count)
        if blockIDsRequired {
            for index in imageStart..<(imageStart + imageRows) {
                blockIDs[index] = 0
            }
        }
        return Gemma4MultimodalPrompt(
            inputIDs: inputIDs,
            imageStart: imageStart,
            imageRows: imageRows,
            blockIDsQ: blockIDs,
            blockIDsKV: blockIDs)
    }

    public static func preprocessImage(data: Data) throws -> Gemma4ImageTensors {
        let image = try RGBImage(data: data)
        return try preprocessImage(image)
    }

    public static func preprocessImage(
        data: Data,
        pixelValuesShape: [Int],
        imagePositionIDsShape: [Int]
    ) throws -> Gemma4ImageTensors {
        let image = try RGBImage(data: data)
        return try preprocessImage(
            image,
            pixelValuesShape: pixelValuesShape,
            imagePositionIDsShape: imagePositionIDsShape)
    }

    static func preprocessImage(_ image: RGBImage) throws -> Gemma4ImageTensors {
        try preprocessImage(
            image,
            pixelValuesShape: [1, maxSoftTokens, patchSize * poolingKernelSize * patchSize * poolingKernelSize * channels],
            imagePositionIDsShape: [1, maxSoftTokens, 2])
    }

    static func preprocessImage(
        _ image: RGBImage,
        pixelValuesShape: [Int],
        imagePositionIDsShape: [Int]
    ) throws -> Gemma4ImageTensors {
        guard pixelValuesShape.count == 3, pixelValuesShape[0] == 1,
              imagePositionIDsShape.count == 3, imagePositionIDsShape[0] == 1,
              imagePositionIDsShape[2] == 2
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "unsupported Gemma 4 image tensor shapes pixel_values=\(pixelValuesShape) image_position_ids=\(imagePositionIDsShape)")
        }
        let unifiedPatchDim = patchSize * poolingKernelSize * patchSize * poolingKernelSize * channels
        if pixelValuesShape[1] == maxSoftTokens,
           pixelValuesShape[2] == unifiedPatchDim,
           imagePositionIDsShape[1] == maxSoftTokens
        {
            return try preprocessUnifiedImage(
                image,
                pixelValuesShape: pixelValuesShape,
                imagePositionIDsShape: imagePositionIDsShape)
        }
        let patchDim = patchSize * patchSize * channels
        if pixelValuesShape[2] == patchDim,
           imagePositionIDsShape[1] == pixelValuesShape[1]
        {
            return try preprocessVisionTowerPatchImage(
                image,
                pixelValuesShape: pixelValuesShape,
                imagePositionIDsShape: imagePositionIDsShape)
        }
        throw CoreAIPipeline.RuntimeError.invalidBundle(
            "unsupported Gemma 4 image tensor shapes pixel_values=\(pixelValuesShape) image_position_ids=\(imagePositionIDsShape)")
    }

    private static func preprocessUnifiedImage(
        _ image: RGBImage,
        pixelValuesShape: [Int],
        imagePositionIDsShape: [Int]
    ) throws -> Gemma4ImageTensors {
        let maxTeacherPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize
        let (targetHeight, targetWidth) = try aspectRatioPreservingSize(
            height: image.height,
            width: image.width,
            patchSize: patchSize,
            maxPatches: maxTeacherPatches,
            poolingKernelSize: poolingKernelSize)
        let resized = resizeBicubicUInt8(
            image: image,
            targetWidth: targetWidth,
            targetHeight: targetHeight)
        let modelPatchSize = patchSize * poolingKernelSize
        let modelPatchWidth = targetWidth / modelPatchSize
        let modelPatchHeight = targetHeight / modelPatchSize
        let validRows = modelPatchWidth * modelPatchHeight
        guard validRows <= maxSoftTokens else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "image preprocessing produced \(validRows) soft tokens, above \(maxSoftTokens)")
        }

        let patchDim = modelPatchSize * modelPatchSize * channels
        guard pixelValuesShape == [1, maxSoftTokens, patchDim],
              imagePositionIDsShape == [1, maxSoftTokens, 2]
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Gemma 4 unified preprocessing expected shapes [1,\(maxSoftTokens),\(patchDim)]/[1,\(maxSoftTokens),2], got \(pixelValuesShape)/\(imagePositionIDsShape)")
        }
        var pixelValues = Array(repeating: Float(0), count: pixelValuesShape.reduce(1, *))
        var positionIDs = Array(repeating: Int32(-1), count: imagePositionIDsShape.reduce(1, *))
        for my in 0..<modelPatchHeight {
            for mx in 0..<modelPatchWidth {
                let row = my * modelPatchWidth + mx
                positionIDs[row * 2] = Int32(mx)
                positionIDs[row * 2 + 1] = Int32(my)
                var offset = row * patchDim
                for py in 0..<modelPatchSize {
                    let y = my * modelPatchSize + py
                    for px in 0..<modelPatchSize {
                        let x = mx * modelPatchSize + px
                        let source = (y * targetWidth + x) * channels
                        pixelValues[offset] = Float(resized[source]) / 255
                        pixelValues[offset + 1] = Float(resized[source + 1]) / 255
                        pixelValues[offset + 2] = Float(resized[source + 2]) / 255
                        offset += channels
                    }
                }
            }
        }
        return Gemma4ImageTensors(
            pixelValuesShape: pixelValuesShape,
            pixelValues: pixelValues,
            imagePositionIDsShape: imagePositionIDsShape,
            imagePositionIDs: positionIDs,
            validRows: validRows)
    }

    private static func preprocessVisionTowerPatchImage(
        _ image: RGBImage,
        pixelValuesShape: [Int],
        imagePositionIDsShape: [Int]
    ) throws -> Gemma4ImageTensors {
        let maxPatches = pixelValuesShape[1]
        let patchDim = patchSize * patchSize * channels
        guard pixelValuesShape[2] == patchDim,
              imagePositionIDsShape == [1, maxPatches, 2]
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "Gemma 4 vision tower preprocessing expected patch stream shapes [1,N,\(patchDim)]/[1,N,2], got \(pixelValuesShape)/\(imagePositionIDsShape)")
        }
        let (targetHeight, targetWidth) = try aspectRatioPreservingSize(
            height: image.height,
            width: image.width,
            patchSize: patchSize,
            maxPatches: maxPatches,
            poolingKernelSize: poolingKernelSize)
        let resized = resizeBicubicUInt8(
            image: image,
            targetWidth: targetWidth,
            targetHeight: targetHeight)
        let patchWidth = targetWidth / patchSize
        let patchHeight = targetHeight / patchSize
        let patchCount = patchWidth * patchHeight
        guard patchCount <= maxPatches else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "image preprocessing produced \(patchCount) vision patches, above \(maxPatches)")
        }
        guard patchWidth.isMultiple(of: poolingKernelSize),
              patchHeight.isMultiple(of: poolingKernelSize),
              patchCount.isMultiple(of: poolingKernelSize * poolingKernelSize)
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "vision tower patch geometry \(patchWidth)x\(patchHeight) is not compatible with pooling \(poolingKernelSize)")
        }
        let validRows = patchCount / (poolingKernelSize * poolingKernelSize)
        var pixelValues = Array(repeating: Float(0), count: pixelValuesShape.reduce(1, *))
        var positionIDs = Array(repeating: Int32(-1), count: imagePositionIDsShape.reduce(1, *))
        for pyBlock in 0..<patchHeight {
            for pxBlock in 0..<patchWidth {
                let patchIndex = pyBlock * patchWidth + pxBlock
                positionIDs[patchIndex * 2] = Int32(pxBlock)
                positionIDs[patchIndex * 2 + 1] = Int32(pyBlock)
                var offset = patchIndex * patchDim
                for py in 0..<patchSize {
                    let y = pyBlock * patchSize + py
                    for px in 0..<patchSize {
                        let x = pxBlock * patchSize + px
                        let source = (y * targetWidth + x) * channels
                        pixelValues[offset] = Float(resized[source]) / 255
                        pixelValues[offset + 1] = Float(resized[source + 1]) / 255
                        pixelValues[offset + 2] = Float(resized[source + 2]) / 255
                        offset += channels
                    }
                }
            }
        }
        return Gemma4ImageTensors(
            pixelValuesShape: pixelValuesShape,
            pixelValues: pixelValues,
            imagePositionIDsShape: imagePositionIDsShape,
            imagePositionIDs: positionIDs,
            validRows: validRows)
    }

    static func aspectRatioPreservingSize(
        height: Int,
        width: Int,
        patchSize: Int,
        maxPatches: Int,
        poolingKernelSize: Int
    ) throws -> (height: Int, width: Int) {
        let totalPixels = Double(height * width)
        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = sqrt(targetPixels / totalPixels)
        let sideMultiple = poolingKernelSize * patchSize
        var targetHeight = Int(floor((factor * Double(height)) / Double(sideMultiple))) * sideMultiple
        var targetWidth = Int(floor((factor * Double(width)) / Double(sideMultiple))) * sideMultiple
        if targetHeight == 0 && targetWidth == 0 {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "image resize target rounded to 0x0")
        }
        let maxSideLength = (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMultiple
        if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = min(Int(floor(Double(width) / Double(height))) * sideMultiple, maxSideLength)
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = min(Int(floor(Double(height) / Double(width))) * sideMultiple, maxSideLength)
        }
        guard targetHeight * targetWidth <= maxPatches * patchSize * patchSize else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "image resize target exceeds patch budget")
        }
        return (targetHeight, targetWidth)
    }

    private static func resizeBicubicUInt8(
        image: RGBImage,
        targetWidth: Int,
        targetHeight: Int
    ) -> [UInt8] {
        if targetWidth == image.width && targetHeight == image.height {
            return image.pixels
        }
        let xContribs = (0..<targetWidth).map {
            cubicContribs(outputIndex: $0, inputSize: image.width, outputSize: targetWidth)
        }
        let yContribs = (0..<targetHeight).map {
            cubicContribs(outputIndex: $0, inputSize: image.height, outputSize: targetHeight)
        }
        var output = Array(repeating: UInt8(0), count: targetWidth * targetHeight * channels)
        for y in 0..<targetHeight {
            let yc = yContribs[y]
            for x in 0..<targetWidth {
                let xc = xContribs[x]
                let outBase = (y * targetWidth + x) * channels
                for channel in 0..<channels {
                    var value = 0.0
                    for yy in 0..<4 {
                        let srcY = yc.indices[yy]
                        let wy = yc.weights[yy]
                        for xx in 0..<4 {
                            let srcX = xc.indices[xx]
                            let wx = xc.weights[xx]
                            let src = (srcY * image.width + srcX) * channels + channel
                            value += Double(image.pixels[src]) * wy * wx
                        }
                    }
                    output[outBase + channel] = UInt8(
                        max(0, min(255, Int(value.rounded()))))
                }
            }
        }
        return output
    }

    private static func resizeCoreGraphics(
        image: RGBImage,
        targetWidth: Int,
        targetHeight: Int
    ) -> [UInt8] {
        guard targetWidth != image.width || targetHeight != image.height else {
            return image.pixels
        }
        let sourceBytesPerPixel = 4
        let sourceBytesPerRow = image.width * sourceBytesPerPixel
        var sourceRGBA = Array(repeating: UInt8(255), count: image.height * sourceBytesPerRow)
        for index in 0..<(image.width * image.height) {
            sourceRGBA[index * 4] = image.pixels[index * 3]
            sourceRGBA[index * 4 + 1] = image.pixels[index * 3 + 1]
            sourceRGBA[index * 4 + 2] = image.pixels[index * 3 + 2]
            sourceRGBA[index * 4 + 3] = 255
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let sourceContext = CGContext(
            data: &sourceRGBA,
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: sourceBytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
            let sourceImage = sourceContext.makeImage()
        else {
            return resizeBicubicUInt8(image: image, targetWidth: targetWidth, targetHeight: targetHeight)
        }
        let targetBytesPerPixel = 4
        let targetBytesPerRow = targetWidth * targetBytesPerPixel
        var targetRGBA = Array(repeating: UInt8(0), count: targetHeight * targetBytesPerRow)
        guard let targetContext = CGContext(
            data: &targetRGBA,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetBytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        else {
            return resizeBicubicUInt8(image: image, targetWidth: targetWidth, targetHeight: targetHeight)
        }
        targetContext.interpolationQuality = .high
        targetContext.draw(sourceImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        var rgb = Array(repeating: UInt8(0), count: targetWidth * targetHeight * channels)
        for index in 0..<(targetWidth * targetHeight) {
            rgb[index * 3] = targetRGBA[index * 4]
            rgb[index * 3 + 1] = targetRGBA[index * 4 + 1]
            rgb[index * 3 + 2] = targetRGBA[index * 4 + 2]
        }
        return rgb
    }

    private static func cubicContribs(
        outputIndex: Int,
        inputSize: Int,
        outputSize: Int
    ) -> (indices: [Int], weights: [Double]) {
        let scale = Double(inputSize) / Double(outputSize)
        let inPosition = (Double(outputIndex) + 0.5) * scale - 0.5
        let base = Int(floor(inPosition))
        var indices: [Int] = []
        var weights: [Double] = []
        indices.reserveCapacity(4)
        weights.reserveCapacity(4)
        for offset in -1...2 {
            let source = base + offset
            indices.append(max(0, min(inputSize - 1, source)))
            weights.append(cubicWeight(inPosition - Double(source)))
        }
        return (indices, weights)
    }

    private static func cubicWeight(_ x: Double) -> Double {
        let a = -0.5
        let ax = abs(x)
        if ax <= 1 {
            return (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1
        }
        if ax < 2 {
            return a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a
        }
        return 0
    }
}

struct RGBImage: Sendable, Equatable {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init(data: Data) throws {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("could not decode image")
        }
        self.width = cgImage.width
        self.height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var rgba = Array(repeating: UInt8(0), count: height * bytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        else {
            throw CoreAIPipeline.RuntimeError.invalidBundle("could not allocate image context")
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        var rgb = Array(repeating: UInt8(0), count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        self.pixels = rgb
    }
}

#endif
