import SwiftUI

struct SettingsView: View {
    @ObservedObject var engine: KyutaiEngine
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                LLMSettingsForm()
                    .padding(24)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            Divider()
            saveBar
        }
        .frame(minWidth: 720, minHeight: 520)
    }

    private var saveBar: some View {
        HStack {
            validationMessage
            Spacer()
            Button("Cancel") { dismiss() }
            Button("Save & Restart") { save() }
                .keyboardShortcut(.defaultAction)
                .disabled(!settings.validationIssues.isEmpty)
        }
        .padding(14)
        .background(.bar)
    }

    @ViewBuilder private var validationMessage: some View {
        if let issue = settings.validationIssues.first {
            Label(issue, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
        } else {
            Text("The API key is stored in your macOS Keychain.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func save() {
        settings.persistCredential()
        engine.restart(configuration: settings.engineEnvironment)
        dismiss()
    }
}

struct LLMSettingsEditor: View {
    @ObservedObject var engine: KyutaiEngine
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Configure Agent & LLM", systemImage: "brain.head.profile")
                    .font(.title2.bold())
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save & Restart") { save() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!settings.validationIssues.isEmpty)
            }
            .padding(20)
            Divider()
            ScrollView {
                LLMSettingsForm()
                    .padding(24)
            }
        }
        .frame(minWidth: 680, minHeight: 500)
    }

    private func save() {
        settings.persistCredential()
        engine.restart(configuration: settings.engineEnvironment)
        dismiss()
    }
}

private struct LLMSettingsForm: View {
    @EnvironmentObject private var settings: AppSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Assistant & language model").font(.title2.bold())
                Text("Connect an OpenAI-compatible endpoint and define your agent.")
                    .foregroundStyle(.secondary)
            }

            GroupBox("Assistant identity") {
                VStack(alignment: .leading, spacing: 12) {
                    LabeledContent("Agent name") {
                        TextField("Nova", text: $settings.agentName).frame(width: 320)
                    }
                    LabeledContent("System prompt") {
                        TextEditor(text: $settings.systemPrompt)
                            .font(.body)
                            .frame(width: 500, height: 180)
                            .padding(6)
                            .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
                    }
                    Text("This is the complete system message sent to the LLM. Use {{agent_name}} wherever the configured agent name should appear.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
            }

            GroupBox("OpenAI-compatible endpoint") {
                VStack(alignment: .leading, spacing: 14) {
                    LabeledContent("API endpoint") {
                        TextField("http://localhost:1234/v1", text: $settings.llmURL)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 420)
                    }
                    LabeledContent("Model") {
                        TextField("provider/model-name", text: $settings.llmModel)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 420)
                    }
                    LabeledContent("API key") {
                        SecureField("Optional for local endpoints", text: $settings.llmAPIKey)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 420)
                    }
                }
                .padding(8)
            }

            Label(
                "Voice detection, speech recognition, and speech synthesis use the app's built-in local pipeline.",
                systemImage: "lock.shield.fill"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(12)
            .background(Color.cyan.opacity(0.07), in: RoundedRectangle(cornerRadius: 10))
        }
        .frame(maxWidth: 720, alignment: .leading)
    }
}
