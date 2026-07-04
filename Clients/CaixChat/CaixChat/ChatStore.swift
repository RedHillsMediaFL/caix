import Foundation

@MainActor
final class ChatStore: ObservableObject {
    @Published var endpoints: [ServerEndpoint] = []
    @Published var selectedEndpoint: ServerEndpoint?
    @Published var models: [CaixModel] = []
    @Published var selectedModelID: String?
    @Published var messages: [ChatTurn] = []
    @Published var inputText = ""
    @Published var attachedImage: ImageAttachment?
    @Published var status = "Offline"
    @Published var isDiscovering = false
    @Published var isSending = false
    @Published var usage: ChatUsage?

    private let discovery = DiscoveryService()
    private var sendTask: Task<Void, Never>?
    private let recentKey = "caix.chat.recentServers"

    var selectedModel: CaixModel? {
        guard let selectedModelID else { return nil }
        return models.first { $0.id == selectedModelID }
    }

    var canAttachImages: Bool {
        selectedModel?.supportsImages != false
    }

    func start() async {
        guard !ProcessInfo.processInfo.caixIsRunningTests else { return }
        endpoints = loadRecentEndpoints()
        if selectedEndpoint == nil, let first = endpoints.first {
            await connect(to: first)
        }
        await discover()
    }

    func discover() async {
        isDiscovering = true
        status = "Discovering"
        let discovered = await discovery.discover()
        endpoints = mergeEndpoints(discovered + endpoints)
        isDiscovering = false
        if selectedEndpoint == nil, let first = endpoints.first {
            await connect(to: first)
        } else if let selectedEndpoint {
            status = "Connected to \(selectedEndpoint.name)"
        } else {
            status = "No server"
        }
    }

    func connect(rawEndpoint: String) async {
        guard let url = ServerEndpoint.normalizedURL(from: rawEndpoint) else {
            status = "Invalid server"
            return
        }
        await connect(to: ServerEndpoint(baseURL: url, source: .manual, isReachable: true))
    }

    func connect(to endpoint: ServerEndpoint) async {
        selectedEndpoint = endpoint
        status = "Connecting"
        do {
            let fetchedModels = try await CaixClient(baseURL: endpoint.baseURL).fetchModels()
            models = fetchedModels
            if let selectedModelID, fetchedModels.contains(where: { $0.id == selectedModelID }) {
                self.selectedModelID = selectedModelID
            } else {
                selectedModelID = fetchedModels.first?.id
            }
            var updated = endpoint
            updated.modelCount = fetchedModels.count
            updated.isReachable = true
            updated.lastSeen = Date()
            selectedEndpoint = updated
            endpoints = mergeEndpoints([updated] + endpoints)
            saveRecentEndpoints()
            status = fetchedModels.isEmpty ? "No models" : "Connected"
        } catch {
            models = []
            selectedModelID = nil
            status = error.localizedDescription
        }
    }

    func selectModel(_ model: CaixModel) {
        selectedModelID = model.id
        attachedImage = nil
    }

    func attachImage(data: Data) {
        do {
            attachedImage = try ImageAttachment(data: data)
        } catch {
            status = error.localizedDescription
        }
    }

    func clearAttachment() {
        attachedImage = nil
    }

    func clearChat() {
        sendTask?.cancel()
        sendTask = nil
        isSending = false
        messages = []
        usage = nil
    }

    func stopGeneration() {
        sendTask?.cancel()
        sendTask = nil
        isSending = false
        if let index = messages.lastIndex(where: { $0.isStreaming }) {
            messages[index].isStreaming = false
        }
    }

    func send() {
        guard !isSending else { return }
        guard let selectedEndpoint else {
            status = "No server"
            return
        }
        guard let model = selectedModelID else {
            status = "No model"
            return
        }

        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || attachedImage != nil else { return }

        let attachment = attachedImage
        inputText = ""
        attachedImage = nil
        usage = nil

        let userTurn = ChatTurn(role: .user, text: text, attachment: attachment)
        messages.append(userTurn)
        let assistantID = UUID()
        messages.append(ChatTurn(id: assistantID, role: .assistant, text: "", isStreaming: true))
        isSending = true
        status = "Streaming"

        let requestMessages = outboundMessages()
        let client = CaixClient(baseURL: selectedEndpoint.baseURL)

        sendTask = Task {
            do {
                for try await delta in client.streamChat(messages: requestMessages, model: model) {
                    if Task.isCancelled { break }
                    apply(delta, to: assistantID)
                }
                finishAssistant(assistantID)
                status = "Connected"
            } catch {
                finishAssistant(assistantID, error: error.localizedDescription)
                status = error.localizedDescription
            }
            isSending = false
            sendTask = nil
        }
    }

    private func outboundMessages() -> [OutboundChatMessage] {
        messages.compactMap { turn in
            guard turn.role == .user || turn.role == .assistant else { return nil }
            guard !turn.isStreaming else { return nil }
            if let attachment = turn.attachment, turn.role == .user {
                return OutboundChatMessage(
                    role: turn.role.rawValue,
                    content: .image(text: turn.text, image: attachment)
                )
            }
            return OutboundChatMessage(role: turn.role.rawValue, content: .text(turn.text))
        }
    }

    private func apply(_ delta: ChatDelta, to id: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        switch delta {
        case .content(let text):
            messages[index].text += text
        case .reasoning(let text):
            messages[index].reasoning += text
        case .usage(let usage):
            self.usage = usage
        case .finished:
            messages[index].isStreaming = false
        }
    }

    private func finishAssistant(_ id: UUID, error: String? = nil) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].isStreaming = false
        if let error {
            messages[index].errorText = error
            if messages[index].text.isEmpty {
                messages[index].text = error
            }
        }
    }

    private func mergeEndpoints(_ endpoints: [ServerEndpoint]) -> [ServerEndpoint] {
        var byID: [String: ServerEndpoint] = [:]
        for endpoint in endpoints {
            if let existing = byID[endpoint.id] {
                var merged = existing
                merged.source = existing.source == .manual ? .manual : endpoint.source
                merged.modelCount = max(existing.modelCount, endpoint.modelCount)
                merged.isReachable = existing.isReachable || endpoint.isReachable
                merged.lastSeen = max(existing.lastSeen, endpoint.lastSeen)
                byID[endpoint.id] = merged
            } else {
                byID[endpoint.id] = endpoint
            }
        }
        return byID.values.sorted {
            if $0.isReachable != $1.isReachable { return $0.isReachable && !$1.isReachable }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
    }

    private func loadRecentEndpoints() -> [ServerEndpoint] {
        guard let data = UserDefaults.standard.data(forKey: recentKey),
              let endpoints = try? JSONDecoder().decode([ServerEndpoint].self, from: data)
        else {
            return []
        }
        return endpoints.map {
            ServerEndpoint(
                baseURL: $0.baseURL,
                source: .recent,
                name: $0.name,
                modelCount: $0.modelCount,
                isReachable: false,
                lastSeen: $0.lastSeen,
                detail: $0.detail
            )
        }
    }

    private func saveRecentEndpoints() {
        let recent = endpoints
            .filter(\.isReachable)
            .prefix(12)
            .map {
                ServerEndpoint(
                    baseURL: $0.baseURL,
                    source: .recent,
                    name: $0.name,
                    modelCount: $0.modelCount,
                    isReachable: false,
                    lastSeen: $0.lastSeen,
                    detail: $0.detail
                )
            }
        guard let data = try? JSONEncoder().encode(Array(recent)) else { return }
        UserDefaults.standard.set(data, forKey: recentKey)
    }
}

private extension ProcessInfo {
    var caixIsRunningTests: Bool {
        environment["XCTestConfigurationFilePath"] != nil
    }
}
