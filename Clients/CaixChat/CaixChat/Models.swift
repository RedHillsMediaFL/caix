import Foundation

enum ServerSource: String, Codable, Sendable {
    case manual
    case recent
    case bonjour
    case network

    var label: String {
        switch self {
        case .manual: return "Manual"
        case .recent: return "Recent"
        case .bonjour: return "Bonjour"
        case .network: return "Network"
        }
    }
}

struct ServerEndpoint: Identifiable, Hashable, Codable, Sendable {
    var id: String { baseURL.absoluteString }

    var name: String
    var baseURL: URL
    var source: ServerSource
    var modelCount: Int
    var isReachable: Bool
    var lastSeen: Date
    var detail: String?

    init(
        baseURL: URL,
        source: ServerSource,
        name: String? = nil,
        modelCount: Int = 0,
        isReachable: Bool = false,
        lastSeen: Date = Date(),
        detail: String? = nil
    ) {
        self.baseURL = Self.normalizedBaseURL(baseURL) ?? baseURL
        self.source = source
        self.name = name ?? Self.displayName(for: self.baseURL)
        self.modelCount = modelCount
        self.isReachable = isReachable
        self.lastSeen = lastSeen
        self.detail = detail
    }

    static func normalizedURL(from rawValue: String) -> URL? {
        var raw = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return nil }
        if !raw.contains("://") {
            raw = "http://\(raw)"
        }
        guard var components = URLComponents(string: raw),
              let scheme = components.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              components.host?.isEmpty == false
        else {
            return nil
        }

        if components.port == nil, scheme == "http" {
            components.port = 1237
        }
        let trimmedPath = components.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if trimmedPath.isEmpty || trimmedPath == "v1" {
            components.path = ""
        }
        components.query = nil
        components.fragment = nil
        return components.url.flatMap(normalizedBaseURL)
    }

    static func normalizedBaseURL(_ url: URL) -> URL? {
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return nil
        }
        let trimmedPath = components.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if trimmedPath.isEmpty || trimmedPath == "v1" {
            components.path = ""
        }
        components.query = nil
        components.fragment = nil
        return components.url
    }

    static func displayName(for url: URL) -> String {
        let host = url.host ?? url.absoluteString
        if let port = url.port {
            return "\(host):\(port)"
        }
        return host
    }
}

struct CaixModel: Identifiable, Hashable, Sendable {
    var id: String
    var status: String?
    var mode: String?
    var supportsReasoning: Bool?
    var supportsImages: Bool?

    var subtitle: String {
        var parts: [String] = []
        if let mode, !mode.isEmpty {
            parts.append(mode.replacingOccurrences(of: "_", with: " "))
        }
        if supportsImages == true {
            parts.append("image")
        }
        if supportsReasoning == true {
            parts.append("reasoning")
        }
        if let status, !status.isEmpty {
            parts.append(status)
        }
        return parts.joined(separator: " · ")
    }

    static func inferredImageSupport(multimodalSupported: Bool?, mode: String?, name: String) -> Bool? {
        if let multimodalSupported {
            return multimodalSupported
        }
        if let mode {
            let lower = mode.lowercased()
            if lower.contains("multimodal") || lower.contains("vision") {
                return true
            }
            if lower == "eagle" || lower == "registry" || lower == "monolithic" || lower == "staged" {
                return false
            }
        }
        let lowerName = name.lowercased()
        if lowerName.contains("vision") || lowerName.contains("multimodal") || lowerName.contains("gemma4") {
            return true
        }
        return nil
    }
}

enum ChatRole: String, Codable, Sendable {
    case user
    case assistant
    case system
}

struct ChatTurn: Identifiable, Equatable {
    var id = UUID()
    var role: ChatRole
    var text: String
    var reasoning: String = ""
    var attachment: ImageAttachment?
    var isStreaming = false
    var errorText: String?

    var isUser: Bool { role == .user }
}

enum ChatDelta: Sendable {
    case content(String)
    case reasoning(String)
    case usage(ChatUsage)
    case finished(String?)
}

struct ChatUsage: Decodable, Sendable, Equatable {
    var prompt_tokens: Int?
    var completion_tokens: Int?
    var total_tokens: Int?
}
