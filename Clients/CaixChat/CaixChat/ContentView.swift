import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject private var store = ChatStore()
    @State private var showsServers = false
    @State private var selectedPhoto: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ConnectionBar(store: store) {
                    showsServers = true
                }
                Divider()
                MessageList(messages: store.messages)
                Divider()
                ComposerBar(store: store, selectedPhoto: $selectedPhoto)
            }
            .navigationTitle("CAIX")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItemGroup(placement: .topBarLeading) {
                    Button {
                        showsServers = true
                    } label: {
                        Image(systemName: "server.rack")
                    }
                    .accessibilityLabel("Servers")
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        Task { await store.discover() }
                    } label: {
                        if store.isDiscovering {
                            ProgressView()
                        } else {
                            Image(systemName: "dot.radiowaves.left.and.right")
                        }
                    }
                    .accessibilityLabel("Discover")

                    Button(role: .destructive) {
                        store.clearChat()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .disabled(store.messages.isEmpty)
                    .accessibilityLabel("Clear")
                }
            }
            .sheet(isPresented: $showsServers) {
                ServerPickerView(store: store)
                    .presentationDetents([.medium, .large])
            }
            .task {
                await store.start()
            }
            .onChange(of: selectedPhoto) { _, item in
                guard let item else { return }
                Task {
                    defer { selectedPhoto = nil }
                    if let data = try? await item.loadTransferable(type: Data.self) {
                        store.attachImage(data: data)
                    }
                }
            }
        }
    }
}

private struct ConnectionBar: View {
    @ObservedObject var store: ChatStore
    var showServers: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Button(action: showServers) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(store.selectedEndpoint?.isReachable == true ? Color.green : Color.orange)
                        .frame(width: 9, height: 9)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(store.selectedEndpoint?.name ?? "No server")
                            .font(.subheadline.weight(.semibold))
                            .lineLimit(1)
                        Text(store.status)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .buttonStyle(.plain)

            Spacer(minLength: 8)

            Menu {
                ForEach(store.models) { model in
                    Button {
                        store.selectModel(model)
                    } label: {
                        Label(model.id, systemImage: model.supportsImages == true ? "photo" : "cpu")
                    }
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: store.selectedModel?.supportsImages == true ? "photo" : "cpu")
                    Text(store.selectedModelID ?? "No model")
                        .lineLimit(1)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .font(.subheadline)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.thinMaterial, in: Capsule())
            }
            .disabled(store.models.isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(uiColor: .systemBackground))
    }
}

private struct MessageList: View {
    var messages: [ChatTurn]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    if messages.isEmpty {
                        ContentUnavailableView("No messages", systemImage: "bubble.left.and.bubble.right")
                            .padding(.top, 80)
                    }
                    ForEach(messages) { turn in
                        MessageBubble(turn: turn)
                            .id(turn.id)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 16)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .onChange(of: messages) { _, newValue in
                guard let id = newValue.last?.id else { return }
                withAnimation(.snappy(duration: 0.2)) {
                    proxy.scrollTo(id, anchor: .bottom)
                }
            }
        }
    }
}

private struct MessageBubble: View {
    var turn: ChatTurn

    var body: some View {
        HStack {
            if turn.isUser { Spacer(minLength: 42) }

            VStack(alignment: turn.isUser ? .trailing : .leading, spacing: 7) {
                if let image = turn.attachment?.previewImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 150, height: 112)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }

                if !turn.reasoning.isEmpty {
                    Text(turn.reasoning)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(10)
                        .background(Color(uiColor: .tertiarySystemFill), in: RoundedRectangle(cornerRadius: 8))
                }

                if !turn.text.isEmpty || turn.isStreaming {
                    HStack(alignment: .bottom, spacing: 8) {
                        Text(turn.text.isEmpty ? " " : turn.text)
                            .font(.body)
                            .textSelection(.enabled)
                        if turn.isStreaming {
                            ProgressView()
                                .controlSize(.mini)
                        }
                    }
                    .foregroundStyle(turn.isUser ? .white : .primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(bubbleBackground, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                }

                if let error = turn.errorText {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: 620, alignment: turn.isUser ? .trailing : .leading)

            if !turn.isUser { Spacer(minLength: 42) }
        }
    }

    private var bubbleBackground: Color {
        if turn.isUser { return Color.accentColor }
        if turn.errorText != nil { return Color(uiColor: .systemRed).opacity(0.12) }
        return Color(uiColor: .secondarySystemGroupedBackground)
    }
}

private struct ComposerBar: View {
    @ObservedObject var store: ChatStore
    @Binding var selectedPhoto: PhotosPickerItem?

    var body: some View {
        VStack(spacing: 8) {
            if let attachment = store.attachedImage, let image = attachment.previewImage {
                HStack {
                    ZStack(alignment: .topTrailing) {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 82, height: 64)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        Button {
                            store.clearAttachment()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .symbolRenderingMode(.hierarchical)
                                .font(.title3)
                                .foregroundStyle(.white)
                        }
                        .padding(4)
                        .accessibilityLabel("Remove image")
                    }
                    Spacer()
                }
            }

            HStack(alignment: .bottom, spacing: 9) {
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Image(systemName: "photo.badge.plus")
                        .font(.title3)
                        .frame(width: 36, height: 36)
                }
                .disabled(!store.canAttachImages || store.isSending)
                .opacity(store.canAttachImages ? 1 : 0.35)
                .accessibilityLabel("Attach image")

                TextField("Message", text: $store.inputText, axis: .vertical)
                    .lineLimit(1...6)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
                    .disabled(store.isSending)
                    .submitLabel(.send)
                    .onSubmit {
                        store.send()
                    }

                Button {
                    store.isSending ? store.stopGeneration() : store.send()
                } label: {
                    Image(systemName: store.isSending ? "stop.fill" : "arrow.up")
                        .font(.headline)
                        .foregroundStyle(.white)
                        .frame(width: 36, height: 36)
                        .background(store.isSending ? Color.red : Color.accentColor, in: Circle())
                }
                .disabled(!store.isSending && store.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && store.attachedImage == nil)
                .accessibilityLabel(store.isSending ? "Stop" : "Send")
            }

            if let usage = store.usage {
                HStack {
                    Label("\(usage.prompt_tokens ?? 0) in", systemImage: "arrow.down.left")
                    Label("\(usage.completion_tokens ?? 0) out", systemImage: "arrow.up.right")
                    Spacer()
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color(uiColor: .systemBackground))
    }
}

private struct ServerPickerView: View {
    @ObservedObject var store: ChatStore
    @Environment(\.dismiss) private var dismiss
    @State private var manualEndpoint = ""

    var body: some View {
        NavigationStack {
            List {
                Section("Manual") {
                    HStack {
                        TextField("http://host:1237", text: $manualEndpoint)
                            .textInputAutocapitalization(.never)
                            .keyboardType(.URL)
                            .autocorrectionDisabled()
                        Button {
                            Task {
                                await store.connect(rawEndpoint: manualEndpoint)
                                dismiss()
                            }
                        } label: {
                            Image(systemName: "link")
                        }
                        .disabled(manualEndpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .accessibilityLabel("Connect")
                    }
                }

                Section("Servers") {
                    if store.endpoints.isEmpty {
                        ContentUnavailableView("No servers", systemImage: "server.rack")
                    }
                    ForEach(store.endpoints) { endpoint in
                        Button {
                            Task {
                                await store.connect(to: endpoint)
                                dismiss()
                            }
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: endpoint.isReachable ? "checkmark.circle.fill" : "clock")
                                    .foregroundStyle(endpoint.isReachable ? .green : .secondary)
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(endpoint.name)
                                        .font(.body.weight(.medium))
                                        .lineLimit(1)
                                    Text(endpoint.baseURL.absoluteString)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                                Spacer()
                                VStack(alignment: .trailing, spacing: 3) {
                                    Text(endpoint.source.label)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    Text("\(endpoint.modelCount)")
                                        .font(.caption.monospacedDigit())
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                if !store.models.isEmpty {
                    Section("Models") {
                        ForEach(store.models) { model in
                            Button {
                                store.selectModel(model)
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(model.id)
                                            .lineLimit(2)
                                        if !model.subtitle.isEmpty {
                                            Text(model.subtitle)
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    Spacer()
                                    if model.id == store.selectedModelID {
                                        Image(systemName: "checkmark")
                                            .foregroundStyle(.tint)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Servers")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await store.discover() }
                    } label: {
                        if store.isDiscovering {
                            ProgressView()
                        } else {
                            Image(systemName: "arrow.clockwise")
                        }
                    }
                    .accessibilityLabel("Refresh")
                }
            }
        }
    }
}
