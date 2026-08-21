import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var engine: KyutaiEngine
    @EnvironmentObject private var settings: AppSettings
    @StateObject private var session = KyutaiSession()
    @StateObject private var metrics = SystemMetrics()

    var body: some View {
        ZStack {
            if settings.mode == .assistant {
                AssistantView(session: session, engine: engine)
                    .transition(.opacity.combined(with: .scale(scale: 0.985)))
            } else {
                DebugDashboardView(session: session, engine: engine, metrics: metrics)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.22), value: settings.mode)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("Mode", selection: $settings.mode) {
                    ForEach(AppMode.allCases) { mode in
                        Label(mode.title, systemImage: mode.icon).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
            }
        }
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
    @State private var showSettings = false

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.055, green: 0.06, blue: 0.075), Color(red: 0.025, green: 0.028, blue: 0.038)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                header
                Spacer(minLength: 20)
                VoiceOrb(
                    micLevel: session.micLevel,
                    speakerLevel: session.speakerLevel,
                    active: session.isListening || session.isSpeaking
                )
                .frame(width: 270, height: 270)

                VStack(spacing: 8) {
                    Text(primaryStatus)
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                    Text(secondaryStatus)
                        .font(.callout)
                        .foregroundStyle(.white.opacity(0.55))
                }
                .padding(.top, 24)

                transcriptPanel
                    .frame(maxWidth: 720)
                    .padding(.top, 22)

                Spacer(minLength: 24)
                controls
                    .padding(.bottom, 34)
            }
            .padding(.horizontal, 34)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(engine: engine)
                .environmentObject(settings)
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(settings.agentName)
                    .font(.headline)
                    .foregroundStyle(.white)
                HStack(spacing: 6) {
                    Circle().fill(engine.isReady ? .green : .orange).frame(width: 7, height: 7)
                    Text(engine.isReady ? "On-device voice ready" : engine.status)
                        .lineLimit(1)
                }
                .font(.caption)
                .foregroundStyle(.white.opacity(0.5))
            }
            Spacer()
            Button { showSettings = true } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 34, height: 34)
                    .background(.white.opacity(0.09), in: Circle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white.opacity(0.82))
            .help("Assistant settings")
        }
        .padding(.top, 18)
    }

    @ViewBuilder private var transcriptPanel: some View {
        if !session.transcript.isEmpty || !session.reply.isEmpty {
            VStack(spacing: 10) {
                if !session.transcript.isEmpty {
                    Text(session.transcript)
                        .font(.body)
                        .foregroundStyle(.white.opacity(0.65))
                        .lineLimit(2)
                }
                if !session.reply.isEmpty {
                    Text(session.reply)
                        .font(.system(size: 17, weight: .medium, design: .rounded))
                        .foregroundStyle(.white.opacity(0.94))
                        .multilineTextAlignment(.center)
                        .lineLimit(4)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
            .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(.white.opacity(0.07), lineWidth: 1))
        } else {
            Text("Start speaking whenever you're ready")
                .font(.callout)
                .foregroundStyle(.white.opacity(0.42))
        }
    }

    private var controls: some View {
        HStack(spacing: 16) {
            roundButton(icon: session.isListening ? "mic.fill" : "mic.slash.fill", label: session.isListening ? "Mute" : "Unmute") {
                session.isListening ? session.stop() : session.start()
            }
            roundButton(icon: "hand.raised.fill", label: "Interrupt", destructive: session.isSpeaking) {
                session.interrupt()
            }
            .disabled(!session.isSpeaking)
            roundButton(icon: "xmark", label: "End") { session.stop() }
        }
    }

    private func roundButton(icon: String, label: String, destructive: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 7) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .semibold))
                    .frame(width: 52, height: 52)
                    .background(destructive ? Color.orange.opacity(0.8) : Color.white.opacity(0.1), in: Circle())
                    .overlay(Circle().stroke(.white.opacity(0.08)))
                Text(label).font(.caption2).foregroundStyle(.white.opacity(0.58))
            }
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
    }

    private var primaryStatus: String {
        if !engine.isReady { return "Getting ready" }
        if session.isSpeaking { return "\(settings.agentName) is speaking" }
        if session.isListening { return "Listening" }
        return "Conversation paused"
    }

    private var secondaryStatus: String {
        if !engine.isReady { return "Models load locally and stay on your Mac" }
        if session.isSpeaking { return "You can interrupt at any time" }
        if session.isListening { return "Speak naturally — no button press needed" }
        return "Unmute when you want to continue"
    }
}

private struct VoiceOrb: View {
    let micLevel: Float
    let speakerLevel: Float
    let active: Bool

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let level = CGFloat(min(max(micLevel * 7 + speakerLevel * 3, 0), 1))
            ZStack {
                Circle()
                    .fill(Color.cyan.opacity(0.13))
                    .blur(radius: 30)
                    .scaleEffect(active ? 1.04 + level * 0.12 : 0.92)
                ForEach(0..<5, id: \.self) { index in
                    let phase = time * (0.75 + Double(index) * 0.07) + Double(index)
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [
                                    Color(red: 0.18, green: 0.72, blue: 0.96).opacity(0.68),
                                    Color(red: 0.50, green: 0.32, blue: 0.95).opacity(0.56),
                                    Color(red: 0.10, green: 0.92, blue: 0.76).opacity(0.45),
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 185 + CGFloat(index * 8), height: 185 + CGFloat(index * 5))
                        .blur(radius: CGFloat(10 + index * 3))
                        .offset(
                            x: CGFloat(sin(phase)) * CGFloat(8 + index * 2) * (1 + level),
                            y: CGFloat(cos(phase * 0.83)) * CGFloat(7 + index) * (1 + level)
                        )
                        .scaleEffect(1 + level * (0.08 + CGFloat(index) * 0.018))
                        .blendMode(.screen)
                }
                Circle()
                    .fill(.ultraThinMaterial.opacity(0.22))
                    .frame(width: 154, height: 154)
                    .overlay(Circle().stroke(.white.opacity(0.22), lineWidth: 1))
                Image(systemName: active ? "waveform" : "sparkles")
                    .font(.system(size: 42, weight: .light))
                    .foregroundStyle(.white.opacity(0.92))
                    .symbolEffect(.variableColor.iterative, isActive: active)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(active ? "Voice activity" : "Assistant idle")
    }
}

private struct DebugDashboardView: View {
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
