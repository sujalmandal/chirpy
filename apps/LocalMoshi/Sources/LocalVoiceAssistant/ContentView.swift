import AppKit
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
    @State private var displayedTranscript = ""
    @State private var displayedReply = ""
    @State private var transcriptVisible = false
    @State private var replyVisible = false
    @State private var transcriptFadeTask: Task<Void, Never>?
    @State private var replyFadeTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 10) {
            VoiceOrb(
                speakerLevel: session.speakerLevel,
                isReady: engine.isReady,
                isListening: session.isListening,
                isSpeaking: session.isSpeaking
            )
            .frame(width: 132, height: 132)
            controls
            VStack(spacing: 7) {
                if !displayedTranscript.isEmpty {
                    floatingCaption(
                        displayedTranscript,
                        icon: "waveform",
                        color: Color(red: 0.55, green: 0.95, blue: 1.0)
                    )
                    .opacity(transcriptVisible ? 1 : 0)
                }
                if !displayedReply.isEmpty {
                    floatingCaption(
                        displayedReply,
                        icon: "sparkles",
                        color: Color(red: 0.78, green: 0.72, blue: 1.0)
                    )
                    .opacity(replyVisible ? 1 : 0)
                }
            }
            .frame(maxWidth: 270, minHeight: 72, alignment: .top)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.clear)
        .contentShape(Rectangle())
        .onChange(of: session.transcript) { _, text in
            showTemporary(text, isReply: false)
        }
        .onChange(of: session.reply) { _, text in
            showTemporary(text, isReply: true)
        }
        .onDisappear {
            transcriptFadeTask?.cancel()
            replyFadeTask?.cancel()
        }
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
            compactButton(icon: "xmark", help: "Quit Local Voice Assistant") {
                NSApplication.shared.terminate(nil)
            }
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

    private func floatingCaption(_ text: String, icon: String, color: Color) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 7) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(color)
            Text(text)
                .font(.system(size: 14, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.96))
                .multilineTextAlignment(.leading)
                .lineLimit(2)
        }
        .frame(maxWidth: 260, alignment: .center)
        .shadow(color: .black.opacity(0.92), radius: 4, y: 1)
        .transition(.opacity.combined(with: .scale(scale: 0.97)))
    }

    private func showTemporary(_ text: String, isReply: Bool) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if isReply {
            displayedReply = trimmed
            replyFadeTask?.cancel()
            withAnimation(.easeOut(duration: 0.16)) { replyVisible = true }
            replyFadeTask = fadeTask {
                withAnimation(.easeInOut(duration: 0.55)) { replyVisible = false }
                try? await Task.sleep(for: .milliseconds(600))
                if !Task.isCancelled { displayedReply = "" }
            }
        } else {
            displayedTranscript = trimmed
            transcriptFadeTask?.cancel()
            withAnimation(.easeOut(duration: 0.16)) { transcriptVisible = true }
            transcriptFadeTask = fadeTask {
                withAnimation(.easeInOut(duration: 0.55)) { transcriptVisible = false }
                try? await Task.sleep(for: .milliseconds(600))
                if !Task.isCancelled { displayedTranscript = "" }
            }
        }
    }

    private func fadeTask(_ completion: @escaping @MainActor () async -> Void) -> Task<Void, Never> {
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(6))
            guard !Task.isCancelled else { return }
            await completion()
        }
    }

    private var primaryStatus: String {
        if !engine.isReady { return "Getting ready" }
        if session.isSpeaking { return "\(settings.agentName) is speaking" }
        if session.isListening { return "Listening" }
        return "Conversation paused"
    }

}

private struct VoiceOrb: View {
    let speakerLevel: Float
    let isReady: Bool
    let isListening: Bool
    let isSpeaking: Bool

    var body: some View {
        GeometryReader { geometry in
            TimelineView(.animation(minimumInterval: 1 / 30)) { timeline in
                let time = timeline.date.timeIntervalSinceReferenceDate
                let output = CGFloat(min(max(speakerLevel * 4, 0), 1))
                let size = min(geometry.size.width, geometry.size.height)
                let restingBeat = CGFloat((sin(time * 7.2) + 1) * 0.008)
                let beat = isSpeaking ? 1 + max(restingBeat, output * 0.09) : 1
                ZStack {
                    if isReady {
                        ZStack {
                            Circle()
                                .fill(orbGradient)
                                .shadow(color: glowColor.opacity(isSpeaking ? 0.44 : 0.26), radius: isSpeaking ? 8 : 5)
                            if isListening && !isSpeaking {
                                ListeningGasCloud(time: time, size: size)
                                    .transition(.opacity)
                            }
                            Circle()
                                .fill(
                                    RadialGradient(
                                        colors: [.white.opacity(0.58), .white.opacity(0.10), .clear],
                                        center: UnitPoint(x: 0.34, y: 0.25),
                                        startRadius: 0,
                                        endRadius: size * 0.46
                                    )
                                )
                            Circle()
                                .stroke(.white.opacity(0.52), lineWidth: 1.15)
                        }
                        .scaleEffect(beat)
                        .transition(.scale(scale: 0.58).combined(with: .opacity))
                    } else {
                        let breathe = 0.88 + CGFloat((sin(time * 2.7) + 1) * 0.04)
                        ZStack {
                            Circle()
                                .trim(from: 0.04, to: 0.60)
                                .stroke(
                                    AngularGradient(
                                        colors: [
                                            Color(red: 0.34, green: 0.94, blue: 1.0).opacity(0.22),
                                            Color(red: 0.46, green: 0.58, blue: 1.0),
                                            Color(red: 0.86, green: 0.66, blue: 1.0).opacity(0.26),
                                        ],
                                        center: .center
                                    ),
                                    style: StrokeStyle(lineWidth: 7, lineCap: .round)
                                )
                                .rotationEffect(.degrees(time * 92))
                            Circle()
                                .trim(from: 0.12, to: 0.43)
                                .stroke(
                                    LinearGradient(
                                        colors: [.white.opacity(0.84), Color.cyan.opacity(0.22)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    style: StrokeStyle(lineWidth: 3.5, lineCap: .round)
                                )
                                .padding(size * 0.13)
                                .rotationEffect(.degrees(-time * 128))
                            Circle()
                                .fill(
                                    RadialGradient(
                                        colors: [
                                            Color.white.opacity(0.34),
                                            Color(red: 0.38, green: 0.52, blue: 1.0).opacity(0.16),
                                            .clear,
                                        ],
                                        center: .center,
                                        startRadius: 0,
                                        endRadius: size * 0.30
                                    )
                                )
                                .padding(size * 0.21)
                                .scaleEffect(breathe)
                        }
                        .scaleEffect(breathe)
                        .transition(.scale(scale: 0.78).combined(with: .opacity))
                    }
                }
                .padding(size * 0.10)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .animation(.spring(response: 0.44, dampingFraction: 0.62), value: isReady)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            !isReady ? "Assistant connecting" : (isSpeaking ? "Assistant speaking" : (isListening ? "Listening" : "Assistant idle"))
        )
    }

    private var glowColor: Color {
        if isSpeaking { return Color(red: 0.54, green: 0.30, blue: 1.0) }
        if isListening { return Color(red: 0.10, green: 0.86, blue: 0.92) }
        return Color(red: 0.32, green: 0.42, blue: 0.95)
    }

    private var orbGradient: LinearGradient {
        let colors: [Color]
        if isSpeaking {
            colors = [
                Color(red: 0.92, green: 0.48, blue: 1.0),
                Color(red: 0.42, green: 0.30, blue: 0.98),
                Color(red: 0.18, green: 0.48, blue: 0.98),
            ]
        } else if isListening {
            colors = [
                Color(red: 0.42, green: 1.0, blue: 0.92),
                Color(red: 0.08, green: 0.72, blue: 0.92),
                Color(red: 0.16, green: 0.38, blue: 0.94),
            ]
        } else {
            colors = [
                Color(red: 0.90, green: 0.94, blue: 1.0),
                Color(red: 0.52, green: 0.60, blue: 0.96),
                Color(red: 0.34, green: 0.38, blue: 0.78),
            ]
        }
        return LinearGradient(colors: colors, startPoint: .topLeading, endPoint: .bottomTrailing)
    }
}

private struct ListeningGasCloud: View {
    let time: Double
    let size: CGFloat

    var body: some View {
        ZStack {
            ForEach(0..<5, id: \.self) { index in
                let phase = time * (0.62 + Double(index) * 0.07) + Double(index) * 1.37
                let diameter = size * (0.34 + CGFloat(index % 3) * 0.07)
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                index.isMultiple(of: 2) ? Color.white.opacity(0.52) : Color.cyan.opacity(0.46),
                                Color(red: 0.52, green: 0.82, blue: 1.0).opacity(0.25),
                                .clear,
                            ],
                            center: .center,
                            startRadius: 0,
                            endRadius: diameter * 0.52
                        )
                    )
                    .frame(width: diameter, height: diameter)
                    .blur(radius: 6 + CGFloat(index) * 0.7)
                    .scaleEffect(0.92 + CGFloat(sin(phase * 0.73)) * 0.10)
                    .offset(
                        x: CGFloat(sin(phase)) * size * 0.13,
                        y: CGFloat(cos(phase * 0.81)) * size * 0.11
                    )
                    .blendMode(.screen)
            }
            Ellipse()
                .fill(
                    LinearGradient(
                        colors: [.clear, Color.white.opacity(0.28), Color.cyan.opacity(0.18), .clear],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .frame(width: size * 0.72, height: size * 0.19)
                .blur(radius: 4)
                .rotationEffect(.degrees(sin(time * 0.68) * 38 - 18))
                .offset(
                    x: CGFloat(cos(time * 0.58)) * size * 0.07,
                    y: CGFloat(sin(time * 0.54)) * size * 0.13
                )
                .blendMode(.screen)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .mask(Circle())
        .opacity(0.96)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

struct DebugDashboardView: View {
    @ObservedObject var session: KyutaiSession
    @ObservedObject var engine: KyutaiEngine
    @ObservedObject var metrics: SystemMetrics
    @EnvironmentObject private var settings: AppSettings
    @State private var isEditingLLM = false

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
        .sheet(isPresented: $isEditingLLM) {
            LLMSettingsEditor(engine: engine)
                .environmentObject(settings)
        }
    }

    private var pipeline: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Voice pipeline").font(.title3.bold())
                    Text("The local speech pipeline is built in. Only the LLM connection is configurable.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text("Agent: \(settings.agentName)").font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 10) {
                fixedStageCard(.vad, "Built-in")
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                fixedStageCard(.stt, "Built-in")
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                llmStageCard
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                fixedStageCard(.tts, "Built-in")
            }
        }
        .padding(16)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 14))
    }

    private var llmStageCard: some View {
        Button { isEditingLLM = true } label: {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Label("LLM", systemImage: PipelineStage.llm.icon).font(.headline)
                    Spacer()
                    Image(systemName: "slider.horizontal.3").foregroundStyle(.secondary)
                }
                Text(settings.llmModel).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            .frame(maxWidth: .infinity, minHeight: 58, alignment: .leading)
            .padding(12)
            .background(.background, in: RoundedRectangle(cornerRadius: 11))
            .overlay(RoundedRectangle(cornerRadius: 11).stroke(.separator.opacity(0.45)))
        }
        .buttonStyle(.plain)
    }

    private func fixedStageCard(_ stage: PipelineStage, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(stage.rawValue, systemImage: stage.icon).font(.headline)
                Spacer()
                Image(systemName: "lock.fill").font(.caption).foregroundStyle(.tertiary)
            }
            Text(value).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 58, alignment: .leading)
        .padding(12)
        .background(.background, in: RoundedRectangle(cornerRadius: 11))
        .overlay(RoundedRectangle(cornerRadius: 11).stroke(.separator.opacity(0.45)))
    }

    private func metric(_ name: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 1) {
            Text(name).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.system(.body, design: .monospaced))
        }
    }
}
