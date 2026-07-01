import Dispatch
import Darwin
import Foundation

private final class CatalogExitCode: @unchecked Sendable {
    var value: Int32 = 0
}

func catalogCommand(_ argv: [String]) {
    if argv.first == "install" {
        catalogInstallCommand(Array(argv.dropFirst()))
        return
    }

    var target: String?
    var limit = 25
    var emitJSON = false

    func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
        exit(2)
    }

    var i = 0
    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--limit":
            i += 1
            guard i < argv.count, let parsed = Int(argv[i]), parsed > 0 else {
                fail("--limit needs a positive integer")
            }
            limit = parsed
        case "--json":
            emitJSON = true
        case "-h", "--help":
            print(
                """
                USAGE:
                  caix catalog <owner/search|collection-slug> [--limit N] [--json]
                  caix catalog install [repo|single-result-search] [--exports DIR] [--name NAME]

                Examples:
                  caix catalog redhillsmediafl/qwen
                  caix catalog redhillsmediafl/qwen-caix-6a3fd5f6d272c154dbfcda67
                  caix catalog install
                  caix catalog install redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix
                """
            )
            exit(0)
        default:
            if arg.hasPrefix("-") { fail("unknown catalog option: \(arg)") }
            if target != nil { fail("catalog accepts one target") }
            target = arg
        }
        i += 1
    }

    let query = target ?? "redhillsmediafl/caix"
    let semaphore = DispatchSemaphore(value: 0)
    let exitCode = CatalogExitCode()
    Task {
        defer { semaphore.signal() }
        do {
            let result = try await Catalog.fetch(target: query, limit: limit)
            if emitJSON {
                let enc = JSONEncoder()
                enc.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try enc.encode(result)
                FileHandle.standardOutput.write(data)
                FileHandle.standardOutput.write(Data("\n".utf8))
            } else {
                print(Catalog.render(result))
            }
        } catch {
            FileHandle.standardError.write(Data("catalog error: \(error)\n".utf8))
            exitCode.value = 1
        }
    }
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode.value)
}

func catalogInstallCommand(_ argv: [String]) {
    var target: String?
    var exportsDir = caixDefaultExportsDisplayPath
    var name: String?
    var revision: String?
    var limit = 10
    var dryRun = false

    func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
        exit(2)
    }

    func value(_ option: String, _ index: inout Int) -> String {
        index += 1
        guard index < argv.count else { fail("\(option) needs a value") }
        return argv[index]
    }

    var i = 0
    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--exports":
            exportsDir = value(arg, &i)
        case "--name":
            name = value(arg, &i)
        case "--revision":
            revision = value(arg, &i)
        case "--limit":
            let raw = value(arg, &i)
            guard let parsed = Int(raw), parsed > 0 else { fail("--limit needs a positive integer") }
            limit = parsed
        case "--dry-run":
            dryRun = true
        case "-h", "--help":
            print(
                """
                USAGE:
                  caix catalog install [repo|single-result-search] [options]

                OPTIONS:
                  --exports <dir>        Root directory for installed bundles (default: \(caixDefaultExportsDisplayPath))
                  --name <bundle-name>   Override destination bundle directory name
                  --revision <rev>       Override catalog revision
                  --limit <N>            Search result cap when target is not exact (default: 10)
                  --dry-run              Resolve and preflight only

                Example:
                  caix catalog install
                  caix catalog install redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix
                """
            )
            exit(0)
        default:
            if arg.hasPrefix("-") { fail("unknown catalog install option: \(arg)") }
            if target != nil { fail("catalog install accepts one target") }
            target = arg
        }
        i += 1
    }

    let query = target?.trimmingCharacters(in: .whitespacesAndNewlines)
    if query?.isEmpty == true {
        fail("catalog install target cannot be empty")
    }

    let semaphore = DispatchSemaphore(value: 0)
    let exitCode = CatalogExitCode()
    Task {
        defer { semaphore.signal() }
        do {
            if let query, !query.isEmpty {
                let result = try await Catalog.fetch(target: query, limit: limit)
                let entry = try Catalog.installableEntry(from: result, requested: query)
                try Catalog.install(
                    entry: entry, exportsDir: exportsDir, name: name, revision: revision,
                    dryRun: dryRun)
            } else {
                try await Catalog.interactiveInstall(
                    exportsDir: exportsDir, limit: max(limit, 50), dryRun: dryRun)
            }
        } catch {
            FileHandle.standardError.write(Data("catalog install error: \(error)\n".utf8))
            exitCode.value = 1
        }
    }
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode.value)
}

struct CatalogResult: Codable {
    var target: String
    var source: String
    var count: Int
    var entries: [CatalogEntry]
}

struct CatalogEntry: Codable {
    var repo: String
    var revision: String?
    var bundleName: String?
    var localDir: String?
    var base: String
    var size: String
    var sizeBytes: Int64?
    var license: String
    var verification: String
    var package: String
    var install: String
    var updated: String?
}

enum Catalog {
    private struct Family {
        var name: String
        var target: String
        var note: String
    }

    private static let families: [Family] = [
        Family(
            name: "qwen",
            target: "redhillsmediafl/qwen-caix-6a3fd5f6d272c154dbfcda67",
            note: "small, fast, tool-friendly"),
        Family(
            name: "gemma",
            target: "redhillsmediafl/gemma-caix-6a3fd5f57b67589b85e6eac6",
            note: "regular, unified, assistant/MTP"),
        Family(
            name: "glm",
            target: "redhillsmediafl/glm-caix-6a40604dfb87315cc99a558e",
            note: "glm authored exports"),
        Family(
            name: "mistral",
            target: "redhillsmediafl/mistral-caix-6a404e43a972c2f2621a000e",
            note: "mistral-family exports"),
        Family(
            name: "gpt-oss",
            target: "redhillsmediafl/gpt-oss-caix-6a404e428791003d8e6e79fc",
            note: "gpt-oss exports"),
        Family(
            name: "ornith",
            target: "redhillsmediafl/ornith-caix-6a3ff0de0d269f65f53ef064",
            note: "ornith exports"),
        Family(
            name: "qwythos",
            target: "redhillsmediafl/qwythos-caix-6a409e86fb87315cc9a2d69f",
            note: "qwythos exports"),
        Family(
            name: "all caix",
            target: "redhillsmediafl/caix",
            note: "broad catalog search"),
    ]

    static func fetch(target: String, limit: Int) async throws -> CatalogResult {
        let trimmed = target.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw CatalogError("empty catalog target") }

        let refs: [ModelRef]
        let source: String
        if let collection = try? await fetchCollection(slug: trimmed), !collection.isEmpty {
            refs = Array(collection.prefix(limit))
            source = "collection"
        } else {
            refs = try await searchModels(target: trimmed, limit: limit)
            source = "model-search"
        }

        var entries: [CatalogEntry] = []
        entries.reserveCapacity(refs.count)
        for ref in refs {
            guard let detail = try? await fetchModel(repo: ref.repo) else { continue }
            let readme = try? await fetchREADME(repo: detail.repo, revision: detail.revision)
            entries.append(entry(from: detail, readme: readme))
        }
        entries.sort { $0.repo < $1.repo }
        return CatalogResult(target: trimmed, source: source, count: entries.count, entries: entries)
    }

    static func render(_ result: CatalogResult) -> String {
        var lines: [String] = []
        lines.append("catalog: \(result.target) (\(result.source), \(result.count) repos)")
        if result.entries.isEmpty {
            lines.append("no caix repos found")
            return lines.joined(separator: "\n")
        }
        for entry in result.entries {
            lines.append("")
            lines.append(entry.repo)
            lines.append("  revision: \(entry.revision ?? "unknown")")
            if let bundleName = entry.bundleName {
                lines.append("  bundle: \(bundleName)")
            }
            if let localDir = entry.localDir {
                lines.append("  local-dir: models/exports/\(localDir)")
            }
            lines.append("  base: \(entry.base)")
            lines.append("  size: \(entry.size)")
            lines.append("  license: \(entry.license)")
            lines.append("  verification: \(entry.verification)")
            lines.append("  package: \(entry.package)")
            lines.append("  install: \(entry.install)")
        }
        lines.append("")
        lines.append("download one: caix catalog install <repo>")
        return lines.joined(separator: "\n")
    }

    static func install(
        entry: CatalogEntry,
        exportsDir: String,
        name: String?,
        revision: String?,
        dryRun: Bool
    ) throws {
        let localName = try safeInstallName(name ?? entry.localDir ?? entry.bundleName)
        let root = expandedFileURL(exportsDir, isDirectory: true)
        let destination = root.appendingPathComponent(localName, isDirectory: true)
        try preflightInstall(destinationRoot: root, incomingBytes: entry.sizeBytes)

        let trimmedRevision = revision?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedRevision = (trimmedRevision?.isEmpty == false ? trimmedRevision : nil)
            ?? entry.revision
        let command = hfDownloadCommand(
            repo: entry.repo,
            revision: resolvedRevision,
            destination: destination.path)

        print("installing \(entry.repo)")
        print("  revision: \(resolvedRevision ?? "default")")
        print("  destination: \(destination.path)")
        print("  size: \(entry.size)")
        print("  auth/cache: local Hugging Face settings, if configured")
        if dryRun {
            print("  dry-run: hf \(command.joined(separator: " "))")
            return
        }

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let lock = try acquireHeavyTaskLock(exportsRoot: root, label: "catalog download \(entry.repo)")
        defer { releaseHeavyTaskLock(lock) }
        try runHFDownload(command)
        print("installed: \(destination.path)")
        if root.path == caixDefaultExportsPath() {
            print("serve with: caix serve")
        } else {
            print("serve with: caix serve --exports \(root.path)")
        }
    }

    static func installInteractiveBlocking(
        exportsDir: String = caixDefaultExportsDisplayPath,
        limit: Int = 100,
        dryRun: Bool = false
    ) throws {
        let semaphore = DispatchSemaphore(value: 0)
        final class Box: @unchecked Sendable {
            var result: Result<Void, Error>?
        }
        let box = Box()
        Task {
            defer { semaphore.signal() }
            do {
                try await interactiveInstall(exportsDir: exportsDir, limit: limit, dryRun: dryRun)
                box.result = .success(())
            } catch {
                box.result = .failure(error)
            }
        }
        while semaphore.wait(timeout: .now()) == .timedOut {
            RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
        }
        try box.result?.get()
    }

    static func interactiveInstall(exportsDir: String, limit: Int, dryRun: Bool) async throws {
        print("caix catalog install")
        print("")
        print("families")
        for (index, family) in families.enumerated() {
            let padded = family.name.padding(toLength: 10, withPad: " ", startingAt: 0)
            print(String(format: "%2d. %@ %@", index + 1, padded, family.note))
        }
        let familyIndex = try promptIndex(
            prompt: "family",
            count: families.count,
            defaultIndex: 0)
        let family = families[familyIndex]
        print("")
        print("fetching \(family.name)...")
        let result = try await fetch(target: family.target, limit: limit)
        let installable = result.entries
            .filter { $0.localDir != nil && $0.package != "manual" }
            .sorted { lhs, rhs in
                if lhs.base == rhs.base { return lhs.repo < rhs.repo }
                return lhs.base < rhs.base
            }
        guard !installable.isEmpty else {
            throw CatalogError("no installable bundles found for \(family.name)")
        }
        print("")
        print("models")
        for (index, entry) in installable.enumerated() {
            let name = entry.localDir ?? entry.bundleName ?? entry.repo
            print(String(format: "%2d. %@", index + 1, name))
            print("    \(entry.repo)")
            print("    \(entry.size) / \(entry.package) / \(entry.verification)")
        }
        let modelIndex = try promptIndex(
            prompt: "model",
            count: installable.count,
            defaultIndex: 0)
        let root = prompt(
            "exports root [\(exportsDir)]",
            defaultValue: exportsDir)
        print("")
        try install(
            entry: installable[modelIndex],
            exportsDir: root,
            name: nil,
            revision: nil,
            dryRun: dryRun)
    }

    static func installableEntry(from result: CatalogResult, requested: String) throws -> CatalogEntry {
        let installable = result.entries.filter { $0.localDir != nil && $0.package != "manual" }
        if let exact = installable.first(where: { $0.repo.caseInsensitiveCompare(requested) == .orderedSame }) {
            return exact
        }
        guard installable.count == 1, let only = installable.first else {
            let choices = installable.prefix(8).map(\.repo).joined(separator: "\n  ")
            let suffix = choices.isEmpty ? "no installable caix bundles matched" : "choose one exact repo:\n  \(choices)"
            throw CatalogError("ambiguous install target '\(requested)': \(suffix)")
        }
        return only
    }

    static func safeInstallName(_ raw: String?) throws -> String {
        guard let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            throw CatalogError("catalog entry has no safe local bundle name")
        }
        guard safeLocalName(raw) != nil else {
            throw CatalogError("unsafe bundle name '\(raw)'")
        }
        return raw
    }

    static func expandedFileURL(_ path: String, isDirectory: Bool) -> URL {
        let expanded = caixExpandPath(path)
        return URL(fileURLWithPath: expanded, isDirectory: isDirectory)
    }

    static func preflightInstall(destinationRoot: URL, incomingBytes: Int64?) throws {
        guard let available = availableBytes(for: destinationRoot) else {
            throw CatalogError("disk preflight failed: could not inspect free space for \(destinationRoot.path)")
        }
        let reserve = installReserveBytes()
        let payload = max(0, incomingBytes ?? 0)
        let (required, overflow) = reserve.addingReportingOverflow(payload)
        if overflow || available < required {
            throw CatalogError(
                "insufficient disk: free \(humanBytes(available)), required \(overflow ? "overflow" : humanBytes(required)) (payload \(incomingBytes.map(humanBytes) ?? "unknown") + reserve \(humanBytes(reserve)))")
        }
    }

    static func hfDownloadCommand(repo: String, revision: String?, destination: String) -> [String] {
        var command = ["download", repo]
        if let revision, !revision.isEmpty {
            command += ["--revision", revision]
        }
        command += ["--local-dir", destination]
        return command
    }

    static func runHFDownload(_ arguments: [String]) throws {
        guard let hf = findExecutable("hf") else {
            throw CatalogError("hf CLI not found; install it with: curl -LsSf https://hf.co/cli/install.sh | bash -s")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: hf)
        process.arguments = arguments
        var env = ProcessInfo.processInfo.environment
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        if env["HF_HUB_DISABLE_XET", default: ""].isEmpty {
            env["HF_HUB_DISABLE_XET"] = "1"
        }
        process.environment = env
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw CatalogError("hf download failed with exit code \(process.terminationStatus)")
        }
    }

    private static func searchModels(target: String, limit: Int) async throws -> [ModelRef] {
        let parts = target.split(separator: "/", maxSplits: 1).map(String.init)
        let author: String
        let search: String
        if parts.count == 2 {
            author = parts[0]
            search = parts[1]
        } else {
            author = "redhillsmediafl"
            search = parts[0]
        }

        var components = URLComponents(url: apiURL(["api", "models"]), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "author", value: author),
            URLQueryItem(name: "search", value: search),
            URLQueryItem(name: "full", value: "true"),
        ]
        let rows = try await fetchJSONArray(components.url!)
        return rows.compactMap { row -> ModelRef? in
            guard let repo = row["id"] as? String else { return nil }
            let tags = row["tags"] as? [String] ?? []
            let library = row["library_name"] as? String
            let caixTagged = tags.contains("caix") || tags.contains("core-ai") || tags.contains("coreai")
            guard library == "caix" || caixTagged || repo.lowercased().hasSuffix("-caix") else {
                return nil
            }
            return ModelRef(repo: repo)
        }
        .prefix(limit)
        .map { $0 }
    }

    private static func fetchCollection(slug: String) async throws -> [ModelRef] {
        let url = apiURL(["api", "collections"] + slug.split(separator: "/").map(String.init))
        let obj = try await fetchJSONObject(url)
        guard let items = obj["items"] as? [[String: Any]] else { return [] }
        return items.compactMap { item -> ModelRef? in
            let type = (item["repoType"] as? String) ?? (item["type"] as? String) ?? (item["item_type"] as? String)
            guard type == "model", let repo = (item["id"] as? String) ?? (item["item_id"] as? String) else {
                return nil
            }
            return ModelRef(repo: repo)
        }
    }

    private static func fetchModel(repo: String) async throws -> ModelDetail {
        let url = apiURL(["api", "models"] + repo.split(separator: "/").map(String.init))
        let obj = try await fetchJSONObject(url)
        let tags = obj["tags"] as? [String] ?? []
        let siblings = (obj["siblings"] as? [[String: Any]] ?? []).compactMap { $0["rfilename"] as? String }
        let card = obj["cardData"] as? [String: Any]
        let resolvedRepo = (obj["id"] as? String) ?? repo
        let revision = obj["sha"] as? String
        let metadata = siblings.contains("metadata.json")
            ? try? await fetchBundleMetadata(repo: resolvedRepo, revision: revision)
            : nil
        return ModelDetail(
            repo: resolvedRepo,
            revision: revision,
            bundleName: safeLocalName(metadata?.name),
            tags: tags,
            siblings: siblings,
            usedStorage: int64Value(obj["usedStorage"]),
            lastModified: (obj["lastModified"] as? String) ?? (obj["last_modified"] as? String),
            cardData: card)
    }

    private static func fetchREADME(repo: String, revision: String?) async throws -> String {
        let ref = revision?.isEmpty == false ? revision! : "main"
        let url = apiURL(repo.split(separator: "/").map(String.init) + ["resolve", ref, "README.md"])
        let data = try await fetchData(url)
        return String(data: data, encoding: .utf8) ?? ""
    }

    private static func entry(from detail: ModelDetail, readme: String?) -> CatalogEntry {
        let name = detail.repo.split(separator: "/").last.map(String.init) ?? detail.repo
        let package = packageKind(siblings: detail.siblings)
        let localName = package == "manual" ? nil : (detail.bundleName ?? name)
        let install = localName.map {
            installCommand(repo: detail.repo, revision: detail.revision, localDir: $0)
        } ?? "manual; inspect the model card"
        return CatalogEntry(
            repo: detail.repo,
            revision: detail.revision,
            bundleName: detail.bundleName,
            localDir: localName,
            base: baseModel(tags: detail.tags),
            size: detail.usedStorage.map(humanBytes) ?? "unknown",
            sizeBytes: detail.usedStorage,
            license: license(tags: detail.tags, cardData: detail.cardData),
            verification: verificationState(readme: readme ?? ""),
            package: package,
            install: install,
            updated: detail.lastModified)
    }

    private static func installCommand(repo: String, revision: String?, localDir: String) -> String {
        if let revision, !revision.isEmpty {
            return "caix catalog install \(repo) --revision \(revision)"
        }
        return "caix catalog install \(repo)"
    }

    private static func packageKind(siblings: [String]) -> String {
        if siblings.contains(where: { $0.hasPrefix("eagle_target.aimodel/") })
            && siblings.contains(where: { $0.hasPrefix("eagle_draft.aimodel/") })
            && siblings.contains(where: { $0.hasPrefix("tokenizer/") })
        {
            return "eagle"
        }
        if siblings.contains("metadata.json") && siblings.contains("draft/metadata.json") {
            return "speculative"
        }
        if siblings.contains("metadata.json") {
            return "standard"
        }
        return "manual"
    }

    private static func baseModel(tags: [String]) -> String {
        tags.first { $0.hasPrefix("base_model:") && !$0.hasPrefix("base_model:finetune:") }
            .map { String($0.dropFirst("base_model:".count)) } ?? "unknown"
    }

    private static func license(tags: [String], cardData: [String: Any]?) -> String {
        if let tag = tags.first(where: { $0.hasPrefix("license:") }) {
            return String(tag.dropFirst("license:".count))
        }
        if let name = cardData?["license_name"] as? String, !name.isEmpty { return name }
        if let value = cardData?["license"] as? String, !value.isEmpty { return value }
        return "unknown"
    }

    private static func verificationState(readme: String) -> String {
        let text = readme.lowercased()
        if text.contains("not yet verified") || text.contains("not verified") || text.contains("unverified") {
            return "unverified"
        }
        if text.contains("verified") || text.contains("smoke passed") || text.contains("readback ok") {
            return "verified"
        }
        if text.contains("tester") || text.contains("testing wanted") {
            return "needs-test"
        }
        return "not-stated"
    }

    private static func humanBytes(_ bytes: Int64) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var value = Double(bytes)
        var unit = 0
        while value >= 1024, unit < units.count - 1 {
            value /= 1024
            unit += 1
        }
        if unit == 0 { return "\(bytes) B" }
        return String(format: "%.1f %@", value, units[unit])
    }

    private static func installReserveBytes() -> Int64 {
        let env = ProcessInfo.processInfo.environment
        let raw = env["CAIX_INSTALL_RESERVE_GIB"]
            ?? env["CAIX_STOP_FLOOR_GIB"]
            ?? env["STOP_FLOOR_GIB"]
        let gib = raw.flatMap { Int64($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 10
        let clamped = max(0, gib)
        let (bytes, overflow) = clamped.multipliedReportingOverflow(by: 1_073_741_824)
        return overflow ? Int64.max : bytes
    }

    private static func availableBytes(for url: URL) -> Int64? {
        let probe = existingAncestor(for: url)
        guard let attrs = try? FileManager.default.attributesOfFileSystem(forPath: probe.path),
              let free = attrs[.systemFreeSize] as? NSNumber
        else { return nil }
        return free.int64Value
    }

    private static func existingAncestor(for url: URL) -> URL {
        var current = url
        var isDir: ObjCBool = false
        while !FileManager.default.fileExists(atPath: current.path, isDirectory: &isDir) {
            let parent = current.deletingLastPathComponent()
            if parent.path == current.path { break }
            current = parent
        }
        _ = isDir
        return current
    }

    private static func heavyTaskLockPath(exportsRoot: URL) -> URL {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["caix_heavy_task_lock"] ?? env["CAIX_HEAVY_TASK_LOCK"],
           !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: caixExpandPath(raw))
        }
        let normalized = exportsRoot.standardizedFileURL
        if normalized.lastPathComponent == "exports",
           normalized.deletingLastPathComponent().lastPathComponent == "models" {
            return normalized
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(".agent-heavy-task.lock")
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(".agent-heavy-task.lock")
            .standardizedFileURL
    }

    private static func acquireHeavyTaskLock(exportsRoot: URL, label: String) throws -> URL {
        let lock = heavyTaskLockPath(exportsRoot: exportsRoot)
        try FileManager.default.createDirectory(
            at: lock.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let fd = open(lock.path, O_CREAT | O_EXCL | O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
        guard fd >= 0 else {
            if errno == EEXIST {
                throw CatalogError("heavy-task lock exists: \(lock.path)")
            }
            throw CatalogError("could not create heavy-task lock \(lock.path): errno \(errno)")
        }
        let text = "pid=\(ProcessInfo.processInfo.processIdentifier)\nlabel=\(label)\nstarted=\(ISO8601DateFormatter().string(from: Date()))\n"
        let data = Array(text.utf8)
        _ = data.withUnsafeBytes { raw in
            Darwin.write(fd, raw.baseAddress, raw.count)
        }
        close(fd)
        return lock
    }

    private static func releaseHeavyTaskLock(_ lock: URL) {
        try? FileManager.default.removeItem(at: lock)
    }

    private static func int64Value(_ value: Any?) -> Int64? {
        if let n = value as? NSNumber { return n.int64Value }
        if let i = value as? Int { return Int64(i) }
        if let s = value as? String { return Int64(s) }
        return nil
    }

    private static func fetchBundleMetadata(repo: String, revision: String?) async throws -> BundleMetadata {
        let ref = revision?.isEmpty == false ? revision! : "main"
        let url = apiURL(repo.split(separator: "/").map(String.init) + ["resolve", ref, "metadata.json"])
        let data = try await fetchData(url)
        return try JSONDecoder().decode(BundleMetadata.self, from: data)
    }

    private static func safeLocalName(_ raw: String?) -> String? {
        guard let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        guard value.count <= 160 else { return nil }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        return value.unicodeScalars.allSatisfy { allowed.contains($0) } ? value : nil
    }

    private static func fetchJSONObject(_ url: URL) async throws -> [String: Any] {
        let data = try await fetchData(url)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CatalogError("invalid JSON object from \(url.absoluteString)")
        }
        return obj
    }

    private static func fetchJSONArray(_ url: URL) async throws -> [[String: Any]] {
        let data = try await fetchData(url)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw CatalogError("invalid JSON array from \(url.absoluteString)")
        }
        return obj
    }

    private static func fetchData(_ url: URL) async throws -> Data {
        var request = URLRequest(url: url, timeoutInterval: 20)
        request.setValue("caix/\(CaixBuildInfo.version)", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode < 200 || http.statusCode >= 300 {
            throw CatalogError("HTTP \(http.statusCode) from \(url.absoluteString)")
        }
        return data
    }

    private static func endpoint() -> URL {
        let raw = ProcessInfo.processInfo.environment["HF_ENDPOINT"] ?? "https://huggingface.co"
        return URL(string: raw.trimmingCharacters(in: CharacterSet(charactersIn: "/")))!
    }

    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.urlCache = nil
        return URLSession(configuration: config)
    }()

    private static func apiURL(_ components: [String]) -> URL {
        components.reduce(endpoint()) { url, component in
            url.appendingPathComponent(component)
        }
    }

    private struct ModelRef {
        var repo: String
    }

    private struct ModelDetail {
        var repo: String
        var revision: String?
        var bundleName: String?
        var tags: [String]
        var siblings: [String]
        var usedStorage: Int64?
        var lastModified: String?
        var cardData: [String: Any]?
    }

    private struct BundleMetadata: Decodable {
        var name: String?
    }
}

struct CatalogError: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

private func prompt(_ prompt: String, defaultValue: String? = nil) -> String {
    if let defaultValue {
        print("\(prompt): ", terminator: "")
        fflush(stdout)
        let value = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return value.isEmpty ? defaultValue : value
    }
    print("\(prompt): ", terminator: "")
    fflush(stdout)
    return readLine()?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
}

private func promptIndex(prompt: String, count: Int, defaultIndex: Int) throws -> Int {
    while true {
        let raw = CatalogErrorPrompt.read(prompt: "\(prompt) [\(defaultIndex + 1)]")
        if raw.isEmpty { return defaultIndex }
        if let value = Int(raw), value >= 1, value <= count {
            return value - 1
        }
        print("enter a number from 1 to \(count)")
    }
}

private enum CatalogErrorPrompt {
    static func read(prompt: String) -> String {
        print("\(prompt): ", terminator: "")
        fflush(stdout)
        return readLine()?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }
}
