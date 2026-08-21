import SwiftUI

struct KyutaiConversationView: View {
    @ObservedObject var session: KyutaiSession
    @ObservedObject var engine: KyutaiEngine
    var showsEngineDetails = true

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            controls
            WaveformView(micLevel: session.micLevel, speakerLevel: session.speakerLevel)
            conversation
            if showsEngineDetails { engineSummary }
            engineEvents
        }
    }

    private var controls: some View {
        HStack {
            Label(session.status, systemImage: session.isListening ? "waveform" : "mic.slash")
                .foregroundStyle(session.isListening ? .green : .orange)
            Spacer()
            Button(session.isListening ? "Stop" : "Start") {
                session.isListening ? session.stop() : session.start()
            }
            Button("Interrupt") { session.interrupt() }
                .disabled(!session.isSpeaking)
        }
        .font(.headline)
    }

    private var conversation: some View {
        GroupBox {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 14) {
                        if session.messages.isEmpty {
                            ContentUnavailableView(
                                "No conversation yet",
                                systemImage: "bubble.left.and.bubble.right",
                                description: Text("Speak naturally. Each detected turn will appear here with its timestamp.")
                            )
                            .frame(maxWidth: .infinity, minHeight: 220)
                        } else {
                            ForEach(session.messages) { message in
                                ChatMessageRow(message: message)
                                    .id(message.id)
                            }
                        }
                    }
                    .padding(14)
                }
                .frame(minHeight: 300, maxHeight: 440)
                .background(.background.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
                .onChange(of: session.messages) { _, messages in
                    guard let id = messages.last?.id else { return }
                    withAnimation(.easeOut(duration: 0.16)) {
                        proxy.scrollTo(id, anchor: .bottom)
                    }
                }
            }
        } label: {
            Label("Conversation", systemImage: "bubble.left.and.bubble.right.fill")
        }
    }

    private var engineSummary: some View {
        GroupBox("Current engine") {
            VStack(alignment: .leading, spacing: 6) {
                row("VAD", "Silero acoustic gate with Kyutai semantic endpointing")
                row("STT", "Built-in Kyutai streaming speech recognition")
                row("LLM", "OpenAI-compatible endpoint")
                row("TTS", "Built-in Kyutai streaming speech synthesis")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var engineEvents: some View {
        GroupBox {
            ScrollViewReader { proxy in
                ScrollView {
                    Text(engine.logs)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .id("logBottom")
                        .padding(10)
                }
                .frame(minHeight: 150, maxHeight: 220, alignment: .topLeading)
                .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 8))
                .onChange(of: engine.logs) { _, _ in
                    proxy.scrollTo("logBottom", anchor: .bottom)
                }
            }
        } label: {
            HStack {
                Label("Engine events", systemImage: "list.bullet.rectangle.portrait")
                Spacer()
                Text("Turn ownership · endpoint reason · cancellation source")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Timestamped engine events")
    }

    private func row(_ name: String, _ detail: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(name).frame(width: 45, alignment: .leading).foregroundStyle(.cyan)
            Text(detail).foregroundStyle(.secondary)
        }
        .font(.caption)
    }
}

private struct ChatMessageRow: View {
    let message: VoiceChatMessage

    var body: some View {
        HStack(alignment: .bottom, spacing: 12) {
            if message.role == .user { Spacer(minLength: 90) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 5) {
                HStack(spacing: 6) {
                    Text(message.role.rawValue).fontWeight(.semibold)
                    if let turnID = message.turnID {
                        Text("Turn \(turnID)").foregroundStyle(.tertiary)
                    }
                    Text(message.timestamp.formatted(date: .omitted, time: .standard))
                        .foregroundStyle(.secondary)
                }
                .font(.caption2)

                VStack(alignment: .leading, spacing: 7) {
                    if message.text.isEmpty, message.state == .streaming {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("Preparing a reply…").foregroundStyle(.secondary)
                        }
                    } else {
                        Text(message.text)
                            .textSelection(.enabled)
                    }
                    stateLabel
                }
                .padding(.horizontal, 13)
                .padding(.vertical, 10)
                .background(bubbleColor, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 15, style: .continuous)
                        .stroke(borderColor, lineWidth: 0.7)
                )
            }
            .frame(maxWidth: 640, alignment: message.role == .user ? .trailing : .leading)
            if message.role == .assistant { Spacer(minLength: 90) }
        }
    }

    @ViewBuilder private var stateLabel: some View {
        switch message.state {
        case .streaming:
            if !message.text.isEmpty {
                Label("Streaming", systemImage: "ellipsis")
                    .foregroundStyle(.secondary)
                    .font(.caption2)
            }
        case .completed:
            EmptyView()
        case .cancelled(let reason):
            Label("Cancelled · \(displayReason(reason))", systemImage: "xmark.circle.fill")
                .foregroundStyle(.orange)
                .font(.caption2)
        case .failed(let reason):
            Label(reason, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
                .font(.caption2)
        }
    }

    private var bubbleColor: Color {
        message.role == .user ? Color.cyan.opacity(0.13) : Color.indigo.opacity(0.16)
    }

    private var borderColor: Color {
        message.role == .user ? Color.cyan.opacity(0.28) : Color.indigo.opacity(0.34)
    }

    private func displayReason(_ reason: String) -> String {
        reason.replacingOccurrences(of: "_", with: " ").capitalized
    }
}
