import Foundation
import PipelineRuntime

struct ClusterRuntimeOptions {
    var remoteStageIDs: [String] = []
    var promptTokens: [Int32]?
    var maxTokens = 1
    var kvCapacity: Int?
    var maxContextLength = 2048
    var joinTimeoutSeconds: Double?
    var once = false
    var verbose = false
}

struct ClusterRuntimeError: Error, CustomStringConvertible {
    var description: String

    init(_ description: String) {
        self.description = description
    }
}

func runClusterJoinRuntime(
    coordinator: String,
    manifestPath: String,
    stagePath: String,
    decodeStagePath: String?,
    stageID requestedStageID: String?,
    listen: String,
    connectTimeoutSeconds: Double
) async throws {
    let (host, port) = try parseClusterHostPort(coordinator)
    let manifest = try DistributedStageManifest.load(
        from: URL(fileURLWithPath: manifestPath),
        defaultModelName: URL(fileURLWithPath: manifestPath).deletingLastPathComponent().lastPathComponent)
    let originalStage = try resolveJoinStage(
        manifest: manifest,
        stagePath: stagePath,
        requestedStageID: requestedStageID)
    let descriptor = try requireDescriptor(for: originalStage.id, in: manifest)
    let localStage = localStageOverride(
        originalStage,
        stagePath: stagePath,
        decodeStagePath: decodeStagePath)
    let context = DistributedStageHandleFactoryContext(
        stage: localStage,
        manifest: manifest,
        descriptor: descriptor)
    let handle = try await makeCoreAIStageHandle(for: context)
    let executor = try DistributedWorkerFrameExecutor(plan: manifest.runtimePlan, handle: handle)
    let connection = try connectToClusterCoordinator(
        host: host,
        port: port,
        timeoutSeconds: connectTimeoutSeconds)
    defer { connection.close() }

    let hello = try executor.makeHello(
        cacheContract: "stateful",
        computeUnit: "all",
        labels: [
            "listen": listen,
            "stage_path": URL(fileURLWithPath: stagePath).lastPathComponent,
        ])
    try connection.send(hello)
    let response = try connection.receive()
    guard case .helloAck(let ack) = response.message else {
        throw ClusterRuntimeError("coordinator did not return hello_ack")
    }
    try ack.validate(against: manifest.runtimePlan)
    guard ack.accepted else {
        throw ClusterRuntimeError("coordinator rejected stage \(ack.stageID): \(ack.reason ?? "unknown")")
    }
    FileHandle.standardError.write(Data("joined cluster stage \(ack.stageID)\n".utf8))

    while true {
        do {
            let frame = try connection.receive()
            if let response = try await executor.processForTransport(frame) {
                try connection.send(response)
            }
        } catch DistributedSocketTransportError.connectionClosed {
            return
        }
    }
}

func runClusterServeRuntime(
    manifestPath: String,
    host: String,
    port: Int,
    options: ClusterRuntimeOptions
) async throws {
    let manifestURL = URL(fileURLWithPath: manifestPath)
    let manifest = try DistributedStageManifest.load(
        from: manifestURL,
        defaultModelName: manifestURL.deletingLastPathComponent().lastPathComponent)
    let remoteStageIDs = try resolveRemoteStageIDs(
        requested: options.remoteStageIDs,
        manifest: manifest)
    let localStageIDs = manifest.runtimePlan.stages.map(\.id).filter {
        !remoteStageIDs.contains($0)
    }
    let listener = try DistributedSocketWorkerListener.bind(host: host, port: port)
    defer { listener.close() }

    FileHandle.standardError.write(
        Data(
            "cluster coordinator listening on \(host):\(listener.boundPort); waiting for \(remoteStageIDs.joined(separator: ", "))\n"
                .utf8))

    var handlesByStageID: [String: DistributedStageHandle] = [:]
    for stageID in localStageIDs {
        let stage = try requireManifestStage(stageID, in: manifest)
        let descriptor = try requireDescriptor(for: stageID, in: manifest)
        let context = DistributedStageHandleFactoryContext(
            stage: stage,
            manifest: manifest,
            descriptor: descriptor)
        handlesByStageID[stageID] = try await makeCoreAIStageHandle(for: context)
        if options.verbose {
            FileHandle.standardError.write(Data("loaded local stage \(stageID)\n".utf8))
        }
    }

    var handshake = try DistributedWorkerHandshakeCoordinator(plan: manifest.runtimePlan)
    var remoteConnections: [String: DistributedSocketWorkerConnection] = [:]
    let joinDeadline = options.joinTimeoutSeconds.map {
        Date().addingTimeInterval($0)
    }
    while Set(remoteConnections.keys) != remoteStageIDs {
        guard let connection = try listener.accept(timeoutSeconds: remainingSeconds(until: joinDeadline)) else {
            let missing = remoteStageIDs.subtracting(Set(remoteConnections.keys)).sorted()
            throw ClusterRuntimeError(
                "timed out waiting for remote stages: \(missing.joined(separator: ", "))")
        }
        do {
            let helloFrame = try connection.receive()
            guard case .hello(let hello) = helloFrame.message else {
                try connection.send(DistributedWorkerWireFrame(message: .error(
                    DistributedWorkerErrorFrame(
                        code: "invalid_handshake",
                        detail: "first frame must be hello"))))
                connection.close()
                continue
            }
            let stageID = hello.stage.id
            guard remoteStageIDs.contains(stageID) else {
                try connection.send(DistributedWorkerWireFrame(message: .helloAck(
                    DistributedWorkerHelloAck(
                        accepted: false,
                        stageID: stageID,
                        reason: "stage not requested by this coordinator"))))
                connection.close()
                continue
            }
            guard remoteConnections[stageID] == nil else {
                try connection.send(DistributedWorkerWireFrame(message: .helloAck(
                    DistributedWorkerHelloAck(
                        accepted: false,
                        stageID: stageID,
                        reason: "stage already connected"))))
                connection.close()
                continue
            }

            let ackFrame = try handshake.processHello(helloFrame)
            try connection.send(ackFrame)
            guard case .helloAck(let ack) = ackFrame.message, ack.accepted else {
                connection.close()
                continue
            }
            let descriptor = try requireDescriptor(for: stageID, in: manifest)
            handlesByStageID[stageID] = try DistributedRemoteStageHandle(
                plan: manifest.runtimePlan,
                descriptor: descriptor
            ) { request in
                try connection.roundTrip(request)
            }
            remoteConnections[stageID] = connection
            FileHandle.standardError.write(Data("worker joined stage \(stageID)\n".utf8))
        } catch {
            connection.close()
            throw error
        }
    }

    let pipeline = try DistributedSameMachinePipeline(
        manifest: manifest,
        handlesByStageID: handlesByStageID)
    let engine = try DistributedStagedEngine(
        pipeline: pipeline,
        maxContextLength: options.maxContextLength)

    guard let promptTokens = options.promptTokens else {
        FileHandle.standardError.write(Data("cluster ready; no prompt_tokens supplied\n".utf8))
        if options.once {
            printClusterReadyJSON(manifest: manifest, remoteStageIDs: remoteStageIDs, localStageIDs: localStageIDs)
            return
        }
        while true {
            try await Task.sleep(nanoseconds: 60_000_000_000)
        }
    }

    let result = try await engine.generate(
        promptTokens: promptTokens,
        options: DistributedStagedGenerationOptions(
            maxTokens: options.maxTokens,
            kvCapacity: options.kvCapacity),
        requestID: "cluster-\(UUID().uuidString)")
    printClusterGenerationJSON(
        manifest: manifest,
        result: result,
        remoteStageIDs: remoteStageIDs,
        localStageIDs: localStageIDs)
    if !options.once {
        while true {
            try await Task.sleep(nanoseconds: 60_000_000_000)
        }
    }
}

func parseClusterHostPort(_ value: String) throws -> (String, Int) {
    let parts = value.split(separator: ":", maxSplits: 1).map(String.init)
    guard parts.count == 2, !parts[0].isEmpty, let port = Int(parts[1]),
        (1...65_535).contains(port)
    else {
        throw ClusterRuntimeError("expected host:port, got \(value)")
    }
    return (parts[0], port)
}

func parseClusterPositiveDouble(_ value: String, flag: String) throws -> Double {
    guard let parsed = Double(value), parsed > 0 else {
        throw ClusterRuntimeError("\(flag) needs a positive number")
    }
    return parsed
}

private func connectToClusterCoordinator(
    host: String,
    port: Int,
    timeoutSeconds: Double
) throws -> DistributedSocketWorkerConnection {
    let deadline = Date().addingTimeInterval(timeoutSeconds)
    var lastError: Error?
    repeat {
        do {
            return try DistributedSocketWorkerConnection.connect(host: host, port: port)
        } catch {
            lastError = error
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { break }
            Thread.sleep(forTimeInterval: min(0.5, remaining))
        }
    } while Date() < deadline

    throw ClusterRuntimeError(
        "could not connect to coordinator \(host):\(port) within \(String(format: "%.1f", timeoutSeconds))s: \(lastError.map { "\($0)" } ?? "unknown error")")
}

private func remainingSeconds(until deadline: Date?) -> Double? {
    guard let deadline else { return nil }
    return max(0, deadline.timeIntervalSinceNow)
}

func parseClusterTokenList(_ value: String) throws -> [Int32] {
    let tokens = value.split(separator: ",").map { raw -> Int32? in
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let intValue = Int32(trimmed), intValue >= 0 else { return nil }
        return intValue
    }
    guard !tokens.isEmpty, tokens.allSatisfy({ $0 != nil }) else {
        throw ClusterRuntimeError("prompt tokens must be a comma-separated list of non-negative integers")
    }
    return tokens.compactMap { $0 }
}

private func resolveRemoteStageIDs(
    requested: [String],
    manifest: DistributedStageManifest
) throws -> Set<String> {
    let stageIDs = Set(manifest.runtimePlan.stages.map(\.id))
    let values = requested.isEmpty
        ? manifest.runtimePlan.stages.filter { $0.role == .transformerLayers }.map(\.id)
        : requested
    var resolved: Set<String> = []
    for stageID in values {
        guard stageIDs.contains(stageID) else {
            throw ClusterRuntimeError("unknown remote stage: \(stageID)")
        }
        resolved.insert(stageID)
    }
    guard !resolved.isEmpty else {
        throw ClusterRuntimeError("cluster coordinator needs at least one remote stage")
    }
    return resolved
}

private func resolveJoinStage(
    manifest: DistributedStageManifest,
    stagePath: String,
    requestedStageID: String?
) throws -> DistributedStageManifestStage {
    if let requestedStageID {
        return try requireManifestStage(requestedStageID, in: manifest)
    }

    let standardized = URL(fileURLWithPath: stagePath).standardizedFileURL.path
    let basename = URL(fileURLWithPath: stagePath).lastPathComponent
    let matches = manifest.stages.filter { stage in
        stage.resolvedAssetPath == standardized
            || URL(fileURLWithPath: stage.assetName).lastPathComponent == basename
    }
    guard matches.count == 1, let match = matches.first else {
        throw ClusterRuntimeError(
            "could not infer stage id for \(basename); pass --stage-id <id>")
    }
    return match
}

private func requireManifestStage(
    _ stageID: String,
    in manifest: DistributedStageManifest
) throws -> DistributedStageManifestStage {
    guard let stage = manifest.stages.first(where: { $0.id == stageID }) else {
        throw ClusterRuntimeError("unknown stage: \(stageID)")
    }
    return stage
}

private func requireDescriptor(
    for stageID: String,
    in manifest: DistributedStageManifest
) throws -> DistributedStageDescriptor {
    guard let descriptor = manifest.runtimePlan.stage(id: stageID) else {
        throw ClusterRuntimeError("unknown stage in runtime plan: \(stageID)")
    }
    return descriptor
}

private func localStageOverride(
    _ stage: DistributedStageManifestStage,
    stagePath: String,
    decodeStagePath: String?
) -> DistributedStageManifestStage {
    let assetURL = URL(fileURLWithPath: stagePath).standardizedFileURL
    let decodeURL = resolvedDecodeStageURL(
        stagePath: assetURL.path,
        explicitDecodePath: decodeStagePath,
        manifestDecodePath: stage.resolvedDecodeAssetPath)
    return DistributedStageManifestStage(
        id: stage.id,
        role: stage.role,
        layerSpec: stage.layerSpec,
        assetName: assetURL.path,
        resolvedAssetPath: assetURL.path,
        decodeAssetName: decodeURL?.path,
        resolvedDecodeAssetPath: decodeURL?.path,
        functionMap: stage.functionMap,
        vocabSize: stage.vocabSize,
        memoryGB: stage.memoryGB,
        rope: stage.rope)
}

private func resolvedDecodeStageURL(
    stagePath: String,
    explicitDecodePath: String?,
    manifestDecodePath: String?
) -> URL? {
    let fileManager = FileManager.default
    if let explicitDecodePath {
        return URL(fileURLWithPath: explicitDecodePath).standardizedFileURL
    }
    if let manifestDecodePath, fileManager.fileExists(atPath: manifestDecodePath) {
        return URL(fileURLWithPath: manifestDecodePath).standardizedFileURL
    }
    let stageURL = URL(fileURLWithPath: stagePath)
    let inferred = stageURL
        .deletingLastPathComponent()
        .appendingPathComponent(stageURL.deletingPathExtension().lastPathComponent + "-decode")
        .appendingPathExtension(stageURL.pathExtension)
        .standardizedFileURL
    if fileManager.fileExists(atPath: inferred.path) {
        return inferred
    }
    return manifestDecodePath.map { URL(fileURLWithPath: $0).standardizedFileURL }
}

private func makeCoreAIStageHandle(
    for context: DistributedStageHandleFactoryContext
) async throws -> DistributedStageHandle {
    #if COREAI_RUNTIME
    let factory = DistributedCoreAIStageHandleFactory()
    return try await factory.makeStageHandle(for: context)
    #else
    throw ClusterRuntimeError(
        "Core AI runtime not linked; rebuild with COREAI_RUNTIME=1 to run cluster stages")
    #endif
}

private func printClusterReadyJSON(
    manifest: DistributedStageManifest,
    remoteStageIDs: Set<String>,
    localStageIDs: [String]
) {
    let payload: [String: Any] = [
        "ready": true,
        "model": manifest.modelName,
        "remote_stage_ids": Array(remoteStageIDs).sorted(),
        "local_stage_ids": localStageIDs,
    ]
    printJSONObject(payload)
}

private func printClusterGenerationJSON(
    manifest: DistributedStageManifest,
    result: DistributedStagedGenerationResult,
    remoteStageIDs: Set<String>,
    localStageIDs: [String]
) {
    let payload: [String: Any] = [
        "model": manifest.modelName,
        "prompt_token_count": result.promptTokenCount,
        "generated_token_ids": result.generatedTokenIDs.map(Int.init),
        "generated_token_count": result.generatedTokenCount,
        "stop_reason": result.stopReason.rawValue,
        "kv_capacity": result.kvCapacity,
        "remote_stage_ids": Array(remoteStageIDs).sorted(),
        "local_stage_ids": localStageIDs,
    ]
    printJSONObject(payload)
}

private func printJSONObject(_ payload: [String: Any]) {
    guard JSONSerialization.isValidJSONObject(payload),
        let data = try? JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys]),
        let string = String(data: data, encoding: .utf8)
    else {
        print("{}")
        return
    }
    print(string)
}
