import Dispatch
import Foundation

func catalogCommand(_ argv: [String]) {
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

                Examples:
                  caix catalog redhillsmediafl/qwen
                  caix catalog redhillsmediafl/qwen-caix-6a3fd5f6d272c154dbfcda67
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
    nonisolated(unsafe) var exitCode: Int32 = 0
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
            exitCode = 1
        }
    }
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode)
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
        return lines.joined(separator: "\n")
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
            return "hf download \(repo) --revision \(revision) --local-dir models/exports/\(localDir)"
        }
        return "hf download \(repo) --local-dir models/exports/\(localDir)"
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
        request.setValue("caix/1.0", forHTTPHeaderField: "User-Agent")
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
