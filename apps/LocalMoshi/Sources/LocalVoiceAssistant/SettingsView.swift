import SwiftUI

struct SettingsView: View {
    @ObservedObject var engine: KyutaiEngine
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @State private var selectedStage: PipelineStage = .llm

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedStage) {
                Section("Voice pipeline") {
                    ForEach(PipelineStage.allCases) { stage in
                        Label(stage == .llm ? "Agent & LLM" : stage.rawValue, systemImage: stage.icon).tag(stage)
                    }
                }
            }
            .navigationSplitViewColumnWidth(min: 170, ideal: 190)
        } detail: {
            PipelineStageForm(stage: selectedStage)
                .padding(24)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .safeAreaInset(edge: .bottom) {
            HStack {
                if let issue = settings.validationIssues.first {
                    Label(issue, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                } else {
                    Text("Credentials are stored in your macOS Keychain.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save & Restart") {
                    settings.persistCredential()
                    engine.restart(configuration: settings.engineEnvironment)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!settings.validationIssues.isEmpty)
            }
            .padding(14)
            .background(.bar)
        }
        .frame(minWidth: 760, minHeight: 560)
    }
}

struct PipelineStageEditor: View {
    let stage: PipelineStage
    @ObservedObject var engine: KyutaiEngine
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Configure \(stage.rawValue)", systemImage: stage.icon).font(.title2.bold())
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save & Restart") {
                    settings.persistCredential()
                    engine.restart(configuration: settings.engineEnvironment)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!settings.validationIssues.isEmpty)
            }
            .padding(20)
            Divider()
            ScrollView {
                PipelineStageForm(stage: stage)
                    .padding(24)
            }
        }
        .frame(minWidth: 680, minHeight: 520)
    }
}

private struct PipelineStageForm: View {
    let stage: PipelineStage
    @EnvironmentObject private var settings: AppSettings
    @State private var browserStage: PipelineStage?

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title).font(.title2.bold())
                Text(subtitle).foregroundStyle(.secondary)
            }

            switch stage {
            case .vad: vadForm
            case .stt: sttForm
            case .llm: llmForm
            case .tts: ttsForm
            }
        }
        .frame(maxWidth: 650, alignment: .leading)
        .sheet(item: $browserStage) { selected in
            HuggingFaceModelBrowser(stage: selected, selection: binding(for: selected))
        }
    }

    private var vadForm: some View {
        VStack(alignment: .leading, spacing: 18) {
            modelPicker(label: "Hugging Face model", value: $settings.vadRepo, stage: .vad)
            callout(
                icon: "info.circle.fill",
                text: "Kyutai's semantic VAD is embedded in its STT checkpoint. Other Hugging Face VAD repositories can be saved here, but need a matching runtime adapter before the engine can load them."
            )
            LabeledContent("Energy threshold") {
                HStack {
                    Slider(value: $settings.vadThreshold, in: 0.002...0.08, step: 0.001)
                    Text(settings.vadThreshold.formatted(.number.precision(.fractionLength(3))))
                        .monospacedDigit().frame(width: 50, alignment: .trailing)
                }.frame(maxWidth: 360)
            }
            LabeledContent("Minimum speech") {
                Stepper("\(settings.minSpeechMS) ms", value: $settings.minSpeechMS, in: 80...1600, step: 80)
            }
            LabeledContent("End-of-turn silence") {
                Stepper("\(settings.minSilenceMS) ms", value: $settings.minSilenceMS, in: 160...2400, step: 80)
            }
        }
    }

    private var sttForm: some View {
        VStack(alignment: .leading, spacing: 18) {
            modelPicker(label: "Hugging Face model", value: $settings.sttRepo, stage: .stt)
            callout(
                icon: "apple.logo",
                text: "The built-in adapter supports Kyutai delayed-stream models through MLX. The browser also shows other repositories so future adapters can reuse this configuration screen."
            )
        }
    }

    private var llmForm: some View {
        VStack(alignment: .leading, spacing: 18) {
            GroupBox("Assistant identity") {
                VStack(alignment: .leading, spacing: 12) {
                    LabeledContent("Agent name") {
                        TextField("Nova", text: $settings.agentName).frame(width: 320)
                    }
                    LabeledContent("Behavior") {
                        TextEditor(text: $settings.systemPrompt)
                            .font(.body)
                            .frame(width: 420, height: 90)
                            .padding(6)
                            .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
                    }
                    Text("The name and behavior are injected into the system context for every conversation.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(8)
            }
            GroupBox("OpenAI-compatible endpoint") {
                VStack(alignment: .leading, spacing: 14) {
                    LabeledContent("API endpoint") {
                        TextField("http://localhost:1234/v1", text: $settings.llmURL)
                            .textFieldStyle(.roundedBorder).frame(width: 420)
                    }
                    LabeledContent("Model") {
                        TextField("provider/model-name", text: $settings.llmModel)
                            .textFieldStyle(.roundedBorder).frame(width: 420)
                    }
                    LabeledContent("API key") {
                        SecureField("Optional for local endpoints", text: $settings.llmAPIKey)
                            .textFieldStyle(.roundedBorder).frame(width: 420)
                    }
                }
                .padding(8)
            }
        }
    }

    private var ttsForm: some View {
        VStack(alignment: .leading, spacing: 18) {
            modelPicker(label: "Hugging Face model", value: $settings.ttsRepo, stage: .tts)
            LabeledContent("Voice repository") {
                TextField("owner/voice-repository", text: $settings.ttsVoiceRepo)
                    .textFieldStyle(.roundedBorder).frame(width: 420)
            }
            LabeledContent("Voice file") {
                TextField("path/to/voice.wav", text: $settings.ttsVoice)
                    .textFieldStyle(.roundedBorder).frame(width: 420)
            }
            LabeledContent("MLX quantization") {
                Picker("MLX quantization", selection: $settings.ttsQuantize) {
                    Text("None").tag(0)
                    Text("4-bit").tag(4)
                    Text("8-bit").tag(8)
                }
                .labelsHidden().frame(width: 160)
            }
            callout(
                icon: "apple.logo",
                text: "The current streaming adapter supports Kyutai TTS checkpoints. Eight-bit quantization is the balanced default for Apple Silicon."
            )
        }
    }

    private func modelPicker(label: String, value: Binding<String>, stage: PipelineStage) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(label).font(.headline)
            HStack {
                TextField("owner/model", text: value).textFieldStyle(.roundedBorder)
                Button("Browse Hugging Face") { browserStage = stage }
            }
        }
    }

    private func callout(icon: String, text: String) -> some View {
        Label {
            Text(text).font(.caption).foregroundStyle(.secondary)
        } icon: {
            Image(systemName: icon).foregroundStyle(.cyan)
        }
        .padding(12)
        .background(Color.cyan.opacity(0.07), in: RoundedRectangle(cornerRadius: 10))
    }

    private func binding(for stage: PipelineStage) -> Binding<String> {
        switch stage {
        case .vad: $settings.vadRepo
        case .stt: $settings.sttRepo
        case .tts: $settings.ttsRepo
        case .llm: .constant("")
        }
    }

    private var title: String {
        switch stage {
        case .vad: "Voice activity detection"
        case .stt: "Speech to text"
        case .llm: "Assistant & language model"
        case .tts: "Text to speech"
        }
    }

    private var subtitle: String {
        switch stage {
        case .vad: "Tune when a turn starts and when the assistant should respond."
        case .stt: "Choose the model that turns microphone audio into text."
        case .llm: "Connect any OpenAI-compatible endpoint and define your agent."
        case .tts: "Choose the local voice model and its speaking voice."
        }
    }
}
