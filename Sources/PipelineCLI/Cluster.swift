import Foundation
import PipelineRuntime

struct ClusterPlanStage: Codable {
    var name: String
    var path: String?
    var resolvedPath: String?
    var pathExists: Bool?
    var role: String?
    var layers: String?
    var memoryGB: Double?

    enum CodingKeys: String, CodingKey {
        case name
        case path
        case resolvedPath = "resolved_path"
        case pathExists = "path_exists"
        case role
        case layers
        case memoryGB = "memory_gb"
    }
}

struct ClusterWorkerBudget: Codable {
    var name: String
    var memoryGB: Double

    enum CodingKeys: String, CodingKey {
        case name
        case memoryGB = "memory_gb"
    }
}

private struct ClusterPlanSource {
    var modelName: String
    var totalLayerCount: Int
    var totalLayerCountDerived: Bool
    var stages: [ClusterPlanStage]
}

struct ClusterAssignment: Codable {
    var stage: String
    var worker: String?
    var reason: String?
}

struct ClusterPlanOutput: Codable {
    var dryRun: Bool
    var source: String
    var modelName: String
    var totalLayerCount: Int
    var stages: [ClusterPlanStage]
    var workers: [ClusterWorkerBudget]
    var assignments: [ClusterAssignment]
    var runtimePlan: DistributedStagePlan
    var warnings: [String]
    var notes: [String]

    enum CodingKeys: String, CodingKey {
        case dryRun = "dry_run"
        case source
        case modelName = "model_name"
        case totalLayerCount = "total_layer_count"
        case stages
        case workers
        case assignments
        case runtimePlan = "runtime_plan"
        case warnings
        case notes
    }
}

func clusterCommand(_ argv: [String]) {
    guard let subcommand = argv.first else {
        clusterUsage()
        exit(2)
    }
    switch subcommand {
    case "plan":
        clusterPlanCommand(Array(argv.dropFirst()))
    case "join":
        clusterJoinCommand(Array(argv.dropFirst()))
    case "-h", "--help", "help":
        clusterUsage()
        exit(0)
    default:
        FileHandle.standardError.write(Data("unknown cluster command: \(subcommand)\n".utf8))
        clusterUsage()
        exit(2)
    }
}

private func clusterUsage() {
    print(
        """
        USAGE:
          caix cluster plan --manifest <stage-manifest.json> [options]
          caix cluster plan --model <bundle-dir> [options]
          caix cluster join --coordinator <host:port> --stage <stage-dir> [options]

        plan OPTIONS:
          --worker <name=GB>     Worker memory budget; repeatable
          --workers <list>       Comma-separated worker memory budgets
          --dry-run              Accepted for clarity; planning is dry-run only
          --json                 Emit machine-readable JSON

        join OPTIONS:
          --coordinator <host:port>
                                  Coordinator address for this worker
          --stage <dir>           Local staged .aimodel bundle directory
          --listen <host:port>    Worker listen address (default: 127.0.0.1:0)
        """)
}

private func clusterJoinCommand(_ argv: [String]) {
    var coordinator: String?
    var stagePath: String?
    var listen = "127.0.0.1:0"

    func usage() {
        print(
            """
            USAGE:
              caix cluster join --coordinator <host:port> --stage <stage-dir> [options]

            OPTIONS:
              --coordinator <host:port>
                                      Coordinator address for this worker
              --stage <dir>           Local staged .aimodel bundle directory
              --listen <host:port>    Worker listen address (default: 127.0.0.1:0)

            This command is reserved for distributed worker runtime. It does not run workers yet.
            """)
    }

    func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
        exit(2)
    }

    var i = 0
    func value(_ flag: String) -> String {
        i += 1
        guard i < argv.count else { fail("\(flag) needs a value") }
        return argv[i]
    }

    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--coordinator":
            coordinator = value(arg)
        case "--stage":
            stagePath = value(arg)
        case "--listen":
            listen = value(arg)
        case "-h", "--help":
            usage()
            exit(0)
        default:
            fail("unknown cluster join option: \(arg)")
        }
        i += 1
    }

    guard let coordinator, !coordinator.isEmpty else {
        fail("cluster join requires --coordinator <host:port>")
    }
    guard let stagePath, !stagePath.isEmpty else {
        fail("cluster join requires --stage <stage-dir>")
    }
    guard listen.contains(":") else {
        fail("--listen must be host:port")
    }

    FileHandle.standardError.write(
        Data(
            "error: caix cluster join is not implemented yet; use caix cluster plan to validate staged manifests\n"
                .utf8))
    exit(1)
}

private func clusterPlanCommand(_ argv: [String]) {
    var manifestPath: String?
    var modelPath: String?
    var workerSpecs: [String] = []
    var emitJSON = false

    func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
        exit(2)
    }

    var i = 0
    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--manifest":
            i += 1
            guard i < argv.count else { fail("--manifest needs a path") }
            manifestPath = argv[i]
        case "--model":
            i += 1
            guard i < argv.count else { fail("--model needs a bundle directory") }
            modelPath = argv[i]
        case "--worker":
            i += 1
            guard i < argv.count else { fail("--worker needs name=GB") }
            workerSpecs.append(argv[i])
        case "--workers":
            i += 1
            guard i < argv.count else { fail("--workers needs a comma-separated list") }
            workerSpecs += argv[i].split(separator: ",").map(String.init)
        case "--dry-run":
            break
        case "--json":
            emitJSON = true
        case "-h", "--help":
            clusterUsage()
            exit(0)
        default:
            fail("unknown cluster plan option: \(arg)")
        }
        i += 1
    }

    guard (manifestPath == nil) != (modelPath == nil) else {
        fail("use exactly one of --manifest or --model")
    }

    do {
        let source: String
        let baseURL: URL
        let planSource: ClusterPlanSource
        if let manifestPath {
            let manifestURL = URL(fileURLWithPath: expandPath(manifestPath)).standardizedFileURL
            source = manifestURL.path
            baseURL = manifestURL.deletingLastPathComponent()
            planSource = try loadClusterPlanSource(
                from: manifestURL, baseURL: baseURL, metadataMode: false)
        } else {
            let root = URL(fileURLWithPath: expandPath(modelPath!), isDirectory: true)
                .standardizedFileURL
            source = root.appendingPathComponent("metadata.json").path
            baseURL = root
            planSource = try loadClusterPlanSource(
                from: root.appendingPathComponent("metadata.json"), baseURL: baseURL,
                metadataMode: true)
        }

        let workers = try workerSpecs.map(parseWorkerBudget)
        let stages = planSource.stages
        let assignments = assignClusterStages(stages, workers: workers)
        let runtimePlan = try makeDistributedStagePlan(
            modelName: planSource.modelName,
            totalLayerCount: planSource.totalLayerCount,
            stages: stages)
        var warnings = clusterWarnings(stages: stages, workers: workers, assignments: assignments)
        if planSource.totalLayerCountDerived {
            warnings.append(
                "Manifest has no total_layer_count. Derived \(planSource.totalLayerCount) from layer ranges.")
        }
        let output = ClusterPlanOutput(
            dryRun: true,
            source: source,
            modelName: planSource.modelName,
            totalLayerCount: planSource.totalLayerCount,
            stages: stages,
            workers: workers,
            assignments: assignments,
            runtimePlan: runtimePlan,
            warnings: warnings,
            notes: clusterNotes())

        if emitJSON {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(output))
            FileHandle.standardOutput.write(Data("\n".utf8))
        } else {
            print(renderClusterPlan(output))
        }
    } catch let todo as ClusterPlanTodo {
        FileHandle.standardError.write(Data("TODO: \(todo.description)\n".utf8))
        exit(1)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

private func loadClusterPlanSource(
    from url: URL,
    baseURL: URL,
    metadataMode: Bool
) throws -> ClusterPlanSource {
    guard FileManager.default.fileExists(atPath: url.path) else {
        if metadataMode {
            throw ClusterPlanError("missing metadata.json in \(baseURL.path)")
        }
        throw ClusterPlanError("manifest not found: \(url.path)")
    }

    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw ClusterPlanError("expected JSON object in \(url.path)")
    }
    let container: [String: Any]
    if let cluster = root["cluster"] as? [String: Any] {
        container = cluster
    } else if metadataMode {
        throw ClusterPlanTodo(
            "\(url.path) does not include cluster.stages; export staged Core AI bundles first "
                + "or pass --manifest <stage-manifest.json>.")
    } else {
        container = root
    }
    let modelName =
        stringValue(container, keys: ["model_name", "model"])
        ?? stringValue(root, keys: ["name", "model_name", "model"])
        ?? baseURL.lastPathComponent
    guard let rows = container["stages"] as? [[String: Any]], !rows.isEmpty else {
        throw ClusterPlanTodo(
            "\(url.path) has no stage metadata; expected a non-empty stages array with id, "
                + "bundle, role, layers, and memory_gb.")
    }
    let stages = try rows.enumerated().map { index, row in
        let name = stringValue(row, keys: ["name", "id"]) ?? "stage-\(index + 1)"
        guard let path = stringValue(row, keys: ["path", "bundle", "bundle_path", "aimodel"]) else {
            throw ClusterPlanTodo("stage \(name) is missing bundle/path metadata")
        }
        guard let role = try stageRoleValue(row, stageName: name) else {
            throw ClusterPlanTodo("stage \(name) is missing role metadata")
        }
        guard
            let layers = try stageLayerDescription(
                row["layers"] ?? row["layer_range"], role: role, stageName: name)
        else {
            throw ClusterPlanTodo("stage \(name) is missing layers metadata")
        }
        guard let memoryGB = doubleValue(row, keys: [
            "memory_gb", "memoryGB", "required_gb", "requiredMemoryGB", "required_memory_gb",
            "estimated_memory_gb",
        ]) else {
            throw ClusterPlanTodo("stage \(name) is missing memory_gb metadata")
        }
        guard memoryGB > 0 else {
            throw ClusterPlanError("stage \(name) memory_gb must be positive")
        }
        let resolved = resolveStagePath(path, baseURL: baseURL)
        return ClusterPlanStage(
            name: name,
            path: path,
            resolvedPath: resolved.path,
            pathExists: FileManager.default.fileExists(atPath: resolved.path),
            role: role.rawValue,
            layers: layers,
            memoryGB: memoryGB)
    }
    let explicitTotalLayerCount =
        intValue(container, keys: ["total_layer_count", "total_layers"])
        ?? intValue(root, keys: ["total_layer_count", "total_layers"])
    if let explicitTotalLayerCount, explicitTotalLayerCount <= 0 {
        throw ClusterPlanError("total_layer_count must be positive")
    }
    let totalLayerCount: Int
    if let explicitTotalLayerCount {
        totalLayerCount = explicitTotalLayerCount
    } else {
        totalLayerCount = try deriveTotalLayerCount(from: stages)
    }
    try validateClusterStageOrder(stages, totalLayerCount: totalLayerCount)
    return ClusterPlanSource(
        modelName: modelName,
        totalLayerCount: totalLayerCount,
        totalLayerCountDerived: explicitTotalLayerCount == nil,
        stages: stages)
}

private func parseWorkerBudget(_ spec: String) throws -> ClusterWorkerBudget {
    let parts = spec.split(separator: "=", maxSplits: 1).map(String.init)
    guard parts.count == 2, !parts[0].isEmpty, let memory = Double(parts[1]), memory > 0 else {
        throw ClusterPlanError("bad worker budget '\(spec)'; use name=GB")
    }
    return ClusterWorkerBudget(name: parts[0], memoryGB: memory)
}

private func assignClusterStages(
    _ stages: [ClusterPlanStage],
    workers: [ClusterWorkerBudget]
) -> [ClusterAssignment] {
    guard !workers.isEmpty else {
        return stages.map {
            ClusterAssignment(stage: $0.name, worker: nil, reason: "no workers specified")
        }
    }

    var used = Dictionary(uniqueKeysWithValues: workers.map { ($0.name, 0.0) })
    var assignments: [ClusterAssignment] = []
    for (index, stage) in stages.enumerated() {
        guard let need = stage.memoryGB else {
            let worker = workers[index % workers.count].name
            assignments.append(
                ClusterAssignment(stage: stage.name, worker: worker, reason: "stage memory not specified"))
            continue
        }
        if let worker = workers.first(where: { (used[$0.name] ?? 0) + need <= $0.memoryGB }) {
            used[worker.name, default: 0] += need
            assignments.append(ClusterAssignment(stage: stage.name, worker: worker.name, reason: nil))
        } else {
            assignments.append(
                ClusterAssignment(
                    stage: stage.name, worker: nil,
                    reason: String(format: "no worker budget fits %.1f GB", need)))
        }
    }
    return assignments
}

private func clusterWarnings(
    stages: [ClusterPlanStage],
    workers: [ClusterWorkerBudget],
    assignments: [ClusterAssignment]
) -> [String] {
    var warnings: [String] = []
    if workers.isEmpty {
        warnings.append("No workers specified. This validates stage metadata only.")
    }
    let missing = stages.filter { $0.pathExists == false }.map(\.name)
    if !missing.isEmpty {
        warnings.append("Missing local stage bundle paths: \(missing.joined(separator: ", ")).")
    }
    if !workers.isEmpty && assignments.contains(where: { $0.worker == nil }) {
        warnings.append("One or more stages could not be placed with the supplied worker budgets.")
    }
    return warnings
}

private func clusterNotes() -> [String] {
    [
        "dry-run only; caix cluster join and caix serve --cluster do not run workers yet",
        "runtime_plan is validated with the same DistributedStagePlan contract used by PipelineRuntime",
        "KV cache ownership stays with the worker assigned to each stage",
        "hidden-state activations flow between adjacent stages in manifest order",
    ]
}

private func renderClusterPlan(_ output: ClusterPlanOutput) -> String {
    var lines = [
        "cluster plan (dry-run): \(output.source)",
        "model: \(output.modelName)",
        "total layers: \(output.totalLayerCount)",
        "stages: \(output.stages.count)",
    ]
    if output.workers.isEmpty {
        lines.append("workers: none")
    } else {
        lines.append(
            "workers: "
                + output.workers
                .map { "\($0.name)=\(formatGB($0.memoryGB))" }
                .joined(separator: ", "))
    }
    lines.append("")
    for stage in output.stages {
        let parts = [
            stage.role.map { "role=\($0)" },
            stage.layers.map { "layers=\($0)" },
            stage.memoryGB.map { "memory=\(formatGB($0))" },
            stage.resolvedPath.map { "path=\($0)" },
            stage.pathExists.map { $0 ? "path_status=ok" : "path_status=missing" },
        ].compactMap { $0 }
        lines.append("- \(stage.name)" + (parts.isEmpty ? "" : "  " + parts.joined(separator: " ")))
    }
    lines.append("")
    for assignment in output.assignments {
        let target = assignment.worker ?? "unassigned"
        let reason = assignment.reason.map { " (\($0))" } ?? ""
        lines.append("\(assignment.stage) -> \(target)\(reason)")
    }
    if !output.warnings.isEmpty {
        lines.append("")
        lines += output.warnings.map { "warning: \($0)" }
    }
    if !output.notes.isEmpty {
        lines.append("")
        lines += output.notes.map { "note: \($0)" }
    }
    return lines.joined(separator: "\n")
}

private func makeDistributedStagePlan(
    modelName: String,
    totalLayerCount: Int,
    stages: [ClusterPlanStage]
) throws -> DistributedStagePlan {
    let descriptors = try stages.map { stage in
        guard let rawRole = stage.role, let role = DistributedStageRole(rawValue: rawRole) else {
            throw ClusterPlanError("stage \(stage.name) role is missing or invalid")
        }
        let layerRange = try distributedLayerRange(
            stage.layers, role: role, stageName: stage.name)
        return DistributedStageDescriptor(
            id: stage.name,
            role: role,
            layerRange: layerRange,
            assetName: stage.path ?? stage.name,
            workerID: nil)
    }
    let plan = DistributedStagePlan(
        modelName: modelName,
        totalLayerCount: totalLayerCount,
        stages: descriptors,
        workers: [])
    try plan.validate()
    return plan
}

private func stringValue(_ dict: [String: Any], keys: [String]) -> String? {
    for key in keys {
        if let value = dict[key] as? String, !value.isEmpty { return value }
    }
    return nil
}

private func doubleValue(_ dict: [String: Any], keys: [String]) -> Double? {
    for key in keys {
        if let value = dict[key] as? NSNumber { return value.doubleValue }
        if let value = dict[key] as? Double { return value }
        if let value = dict[key] as? String {
            let cleaned = value
                .replacingOccurrences(of: "GB", with: "", options: .caseInsensitive)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let parsed = Double(cleaned) { return parsed }
        }
    }
    return nil
}

private func resolveStagePath(_ path: String, baseURL: URL) -> URL {
    let expanded = expandPath(path)
    if expanded.hasPrefix("/") {
        return URL(fileURLWithPath: expanded).standardizedFileURL
    }
    return baseURL.appendingPathComponent(expanded).standardizedFileURL
}

private func expandPath(_ path: String) -> String {
    (path as NSString).expandingTildeInPath
}

private func stageRoleValue(
    _ row: [String: Any],
    stageName: String
) throws -> DistributedStageRole? {
    guard let value = stringValue(row, keys: ["role", "kind"]) else { return nil }
    guard let role = DistributedStageRole(rawValue: value) else {
        throw ClusterPlanError(
            "stage \(stageName) role must be one of embeddings, transformer_layers, final_norm_head")
    }
    return role
}

private func stageLayerDescription(
    _ raw: Any?,
    role: DistributedStageRole,
    stageName: String
) throws -> String? {
    if role == .transformerLayers {
        guard let range = try layerRangeValue(raw, stageName: stageName) else { return nil }
        return "\(range.lower)..<\(range.upper)"
    }
    if let value = raw as? String, !value.isEmpty { return value }
    if raw != nil {
        throw ClusterPlanError("stage \(stageName) layers must be a label for role \(role.rawValue)")
    }
    return nil
}

private func layerRangeValue(
    _ raw: Any?,
    stageName: String
) throws -> (lower: Int, upper: Int)? {
    if let values = raw as? [Int], !values.isEmpty {
        return try validateLayerRange(values, stageName: stageName)
    }
    if let values = raw as? [NSNumber], !values.isEmpty {
        return try validateLayerRange(values.map(\.intValue), stageName: stageName)
    }
    if let object = raw as? [String: Any] {
        let lower =
            intValue(object, keys: ["lower_bound", "lower", "start"])
            ?? intValue(object, keys: ["from"])
        let upper =
            intValue(object, keys: ["upper_bound", "upper", "end"])
            ?? intValue(object, keys: ["to"])
        guard let lower, let upper else {
            throw ClusterPlanError("stage \(stageName) layer_range needs lower_bound and upper_bound")
        }
        return try validateLayerRange([lower, upper], stageName: stageName)
    }
    if let value = raw as? String, !value.isEmpty {
        let normalized = value
            .replacingOccurrences(of: "..<", with: ",")
            .replacingOccurrences(of: "-", with: ",")
        let parts = normalized
            .split(separator: ",", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        if parts.count == 2, let lower = Int(parts[0]), let upper = Int(parts[1]) {
            return try validateLayerRange([lower, upper], stageName: stageName)
        }
        throw ClusterPlanError("stage \(stageName) transformer_layers needs layers [lower, upper]")
    }
    return nil
}

private func intValue(_ dict: [String: Any], keys: [String]) -> Int? {
    for key in keys {
        if let value = dict[key] as? NSNumber { return value.intValue }
        if let value = dict[key] as? Int { return value }
        if let value = dict[key] as? String, let parsed = Int(value) { return parsed }
    }
    return nil
}

private func validateLayerRange(_ values: [Int], stageName: String) throws -> (lower: Int, upper: Int) {
    guard values.count == 2 else {
        throw ClusterPlanError("stage \(stageName) transformer_layers needs layers [lower, upper]")
    }
    let lower = values[0]
    let upper = values[1]
    guard lower >= 0, upper > lower else {
        throw ClusterPlanError("stage \(stageName) layer range must be non-empty and non-negative")
    }
    return (lower, upper)
}

private func deriveTotalLayerCount(from stages: [ClusterPlanStage]) throws -> Int {
    let ranges = try stages.compactMap { stage -> (lower: Int, upper: Int)? in
        guard stage.role == DistributedStageRole.transformerLayers.rawValue else { return nil }
        guard let range = parseNormalizedLayerRange(stage.layers) else {
            throw ClusterPlanError("stage \(stage.name) transformer_layers needs layers [lower, upper]")
        }
        return range
    }
    guard let last = ranges.last else {
        throw ClusterPlanError("cluster plan needs at least one transformer_layers stage")
    }
    return last.upper
}

private func distributedLayerRange(
    _ value: String?,
    role: DistributedStageRole,
    stageName: String
) throws -> DistributedLayerRange? {
    guard role == .transformerLayers else { return nil }
    guard let range = parseNormalizedLayerRange(value) else {
        throw ClusterPlanError("stage \(stageName) transformer_layers needs layers [lower, upper]")
    }
    return DistributedLayerRange(lowerBound: range.lower, upperBound: range.upper)
}

private func validateClusterStageOrder(_ stages: [ClusterPlanStage], totalLayerCount: Int) throws {
    guard stages.first?.role == DistributedStageRole.embeddings.rawValue else {
        throw ClusterPlanError("first stage role must be embeddings")
    }
    guard stages.last?.role == DistributedStageRole.finalNormHead.rawValue else {
        throw ClusterPlanError("last stage role must be final_norm_head")
    }
    let middle = stages.dropFirst().dropLast()
    guard !middle.isEmpty else {
        throw ClusterPlanError("cluster plan needs at least one transformer_layers stage")
    }
    if let stage = middle.first(where: { $0.role != DistributedStageRole.transformerLayers.rawValue }) {
        throw ClusterPlanError("middle stage \(stage.name) role must be transformer_layers")
    }

    var expectedStart = 0
    for stage in middle {
        guard let range = parseNormalizedLayerRange(stage.layers) else {
            throw ClusterPlanError("stage \(stage.name) transformer_layers needs layers [lower, upper]")
        }
        guard range.lower == expectedStart else {
            throw ClusterPlanError(
                "stage \(stage.name) layer range starts at \(range.lower); expected \(expectedStart)")
        }
        expectedStart = range.upper
    }
    guard expectedStart == totalLayerCount else {
        throw ClusterPlanError("layer coverage ends at \(expectedStart); expected \(totalLayerCount)")
    }
}

private func parseNormalizedLayerRange(_ value: String?) -> (lower: Int, upper: Int)? {
    guard let value else { return nil }
    let parts = value
        .replacingOccurrences(of: "..<", with: ",")
        .split(separator: ",", omittingEmptySubsequences: true)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    guard parts.count == 2, let lower = Int(parts[0]), let upper = Int(parts[1]) else {
        return nil
    }
    return (lower, upper)
}

private func formatGB(_ value: Double) -> String {
    value.rounded() == value ? "\(Int(value))GB" : String(format: "%.1fGB", value)
}

private struct ClusterPlanError: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

private struct ClusterPlanTodo: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}
