import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var engine: KyutaiEngine
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var session: KyutaiSession
    @EnvironmentObject private var metrics: SystemMetrics

    var body: some View {
        AssistantView(session: session, engine: engine)
        .task {
            engine.startIfNeeded(configuration: settings.engineEnvironment)
            metrics.start(engine: engine)
        }
        .task(id: engine.isReady) {
            if engine.isReady { session.start() } else { session.stop() }
        }
        .onDisappear {
            session.stop()
            metrics.stop()
        }
    }
}

private struct AssistantView: View {
    @ObservedObject var session: KyutaiSession
    @ObservedObject var engine: KyutaiEngine
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(spacing: 10) {
            VoiceOrb(
                micLevel: session.micLevel,
                speakerLevel: session.speakerLevel,
                active: session.isListening || session.isSpeaking
            )
            .frame(width: 124, height: 124)
            controls
            if !session.transcript.isEmpty {
                Text(session.transcript)
                    .font(.system(size: 14, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.94))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .frame(maxWidth: 250, minHeight: 36, alignment: .top)
                    .shadow(color: .black.opacity(0.9), radius: 4, y: 1)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(.easeOut(duration: 0.18), value: session.transcript)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.clear)
        .contentShape(Rectangle())
        .contextMenu {
            Button {
                openWindow(id: "debug")
            } label: {
                Label("Open Debug Mode", systemImage: "wrench.and.screwdriver")
            }
            Divider()
            Button {
                settings.persistCredential()
                engine.restart(configuration: settings.engineEnvironment)
            } label: {
                Label("Restart Voice Engine", systemImage: "arrow.clockwise")
            }
        }
        .accessibilityLabel("\(settings.agentName). \(primaryStatus). Right-click for Debug Mode.")
    }

    private var controls: some View {
        HStack(spacing: 12) {
            compactButton(icon: session.isListening ? "mic.fill" : "mic.slash.fill", help: session.isListening ? "Mute microphone" : "Unmute microphone") {
                session.isListening ? session.stop() : session.start()
            }
            compactButton(icon: "xmark", help: "End conversation") { session.stop() }
            compactButton(icon: session.isOutputMuted ? "speaker.slash.fill" : "speaker.wave.2.fill", help: session.isOutputMuted ? "Unmute speaker" : "Mute speaker") {
                session.toggleOutputMuted()
            }
        }
    }

    private func compactButton(icon: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 13, weight: .medium))
                .frame(width: 30, height: 30)
                .background(Color.black.opacity(0.46), in: Circle())
                .overlay(Circle().stroke(.white.opacity(0.24), lineWidth: 0.7))
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white.opacity(0.92))
        .help(help)
    }

    private var primaryStatus: String {
        if !engine.isReady { return "Getting ready" }
        if session.isSpeaking { return "\(settings.agentName) is speaking" }
        if session.isListening { return "Listening" }
        return "Conversation paused"
    }

}

private struct VoiceOrb: View {
    let micLevel: Float
    let speakerLevel: Float
    let active: Bool

    var body: some View {
        GeometryReader { geometry in
            TimelineView(.animation(minimumInterval: 1 / 30)) { timeline in
                let time = timeline.date.timeIntervalSinceReferenceDate
                let level = CGFloat(min(max(micLevel * 7 + speakerLevel * 3, 0), 1))
                let size = min(geometry.size.width, geometry.size.height)
                ZStack {
                    Circle()
                        .fill(Color.indigo.opacity(0.30))
                        .frame(width: size * 0.78, height: size * 0.78)
                        .blur(radius: size * 0.13)
                        .scaleEffect(active ? 1.0 + level * 0.12 : 0.92)
                    ForEach(0..<4, id: \.self) { index in
                        let phase = time * (0.72 + Double(index) * 0.08) + Double(index)
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [
                                        Color(red: 0.92, green: 0.95, blue: 1.0).opacity(0.92),
                                        Color(red: 0.48, green: 0.57, blue: 1.0).opacity(0.72),
                                        Color(red: 0.70, green: 0.88, blue: 1.0).opacity(0.76),
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: size * (0.55 + CGFloat(index) * 0.025), height: size * (0.55 + CGFloat(index) * 0.018))
                            .blur(radius: size * (0.025 + CGFloat(index) * 0.012))
                            .offset(
                                x: CGFloat(sin(phase)) * size * 0.025 * (1 + level),
                                y: CGFloat(cos(phase * 0.83)) * size * 0.022 * (1 + level)
                            )
                            .scaleEffect(1 + level * (0.04 + CGFloat(index) * 0.012))
                            .blendMode(.screen)
                    }
                    Circle()
                        .fill(.white.opacity(0.12))
                        .frame(width: size * 0.51, height: size * 0.51)
                        .overlay(Circle().stroke(.white.opacity(0.22), lineWidth: 0.7))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(active ? "Voice activity" : "Assistant idle")
    }
}

struct DebugDashboardView: View {
    @ObservedObject var session: KyutaiSession
    @ObservedObject var engine: KyutaiEngine
    @ObservedObject var metrics: SystemMetrics
    @EnvironmentObject private var settings: AppSettings
    @State private var editingStage: PipelineStage?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 18) {
                Label(engine.status, systemImage: engine.isReady ? "checkmark.circle.fill" : "arrow.triangle.2.circlepath")
                    .foregroundStyle(engine.isReady ? .green : .orange)
                Spacer()
                metric("CPU", metrics.cpu)
                metric("GPU", metrics.gpu)
                metric("RAM", metrics.ram)
                Button(engine.isRunning ? "Restart" : "Start") {
                    settings.persistCredential()
                    engine.restart(configuration: settings.engineEnvironment)
                }
                .disabled(engine.isStarting || !settings.validationIssues.isEmpty)
            }
            .padding(14)
            .background(.bar)

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    pipeline
                    if engine.isReady {
                        KyutaiConversationView(session: session, engine: engine, showsEngineDetails: false)
                    } else {
                        ContentUnavailableView("Preparing voice engine", systemImage: "waveform", description: Text(engine.status))
                            .frame(maxWidth: .infinity, minHeight: 260)
                    }
                }
                .padding(20)
            }
        }
        .sheet(item: $editingStage) { stage in
            PipelineStageEditor(stage: stage, engine: engine)
                .environmentObject(settings)
        }
    }

    private var pipeline: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Voice pipeline").font(.title3.bold())
                    Text("Every stage is configurable. Changes apply after an engine restart.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text("Agent: \(settings.agentName)").font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 10) {
                stageCard(.vad, settings.vadRepo)
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                stageCard(.stt, settings.sttRepo)
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                stageCard(.llm, settings.llmModel)
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                stageCard(.tts, settings.ttsRepo)
            }
        }
        .padding(16)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 14))
    }

    private func stageCard(_ stage: PipelineStage, _ value: String) -> some View {
        Button { editingStage = stage } label: {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Label(stage.rawValue, systemImage: stage.icon).font(.headline)
                    Spacer()
                    Image(systemName: "slider.horizontal.3").foregroundStyle(.secondary)
                }
                Text(value).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            .frame(maxWidth: .infinity, minHeight: 58, alignment: .leading)
            .padding(12)
            .background(.background, in: RoundedRectangle(cornerRadius: 11))
            .overlay(RoundedRectangle(cornerRadius: 11).stroke(.separator.opacity(0.45)))
        }
        .buttonStyle(.plain)
    }

    private func metric(_ name: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 1) {
            Text(name).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.system(.body, design: .monospaced))
        }
    }
}
