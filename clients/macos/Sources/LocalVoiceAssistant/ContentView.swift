import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var assistant: VoiceAssistant
    @State private var showingSettings = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Local Voice Assistant").font(.title.bold())
                Spacer()
                Button("Model settings") { showingSettings = true }
            }
            Text(assistant.status).foregroundStyle(.secondary)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if !assistant.transcript.isEmpty {
                        Text("You").font(.caption.bold()).foregroundStyle(.secondary)
                        Text(assistant.transcript)
                    }
                    if !assistant.reply.isEmpty {
                        Text("Assistant").font(.caption.bold()).foregroundStyle(.secondary)
                        Text(assistant.reply)
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack {
                Button(assistant.isRecording ? "Release to send" : "Hold to talk") {
                    assistant.toggleRecording()
                }
                .keyboardShortcut(.space, modifiers: [])
                .buttonStyle(.borderedProminent)
                Button("Stop") { assistant.cancelTurn() }
                    .keyboardShortcut(.escape, modifiers: [])
                    .disabled(!assistant.isBusy)
                Button("New chat") { assistant.clearConversation() }
            }
            Toggle("Hands-free listening", isOn: $assistant.handsFreeEnabled)
                .onChange(of: assistant.handsFreeEnabled) { _, enabled in assistant.setHandsFree(enabled) }
            Text("Hold Space or the button while speaking. Escape interrupts speech and cancels the current turn.")
                .font(.footnote).foregroundStyle(.secondary)
        }
        .padding(24)
        .sheet(isPresented: $showingSettings) {
            EndpointSettingsView(settings: assistant.settings)
        }
    }
}

private struct EndpointSettingsView: View {
    @ObservedObject var settings: EndpointSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Model settings").font(.title2.bold())
            Text("Leave the STT or TTS endpoint blank to use the local Whisper/Piper setup. Enter a base URL ending in /v1 for an OpenAI-compatible endpoint.")
                .font(.footnote).foregroundStyle(.secondary)
            endpointFields(title: "Speech to text", endpoint: $settings.sttEndpoint, model: $settings.sttModel, placeholder: "Local Whisper (default)")
            endpointFields(title: "Language model", endpoint: $settings.llmEndpoint, model: $settings.llmModel, placeholder: "http://127.0.0.1:11434/v1")
            endpointFields(title: "Text to speech", endpoint: $settings.ttsEndpoint, model: $settings.ttsModel, placeholder: "Local Piper (default)")
            HStack { Spacer(); Button("Done") { dismiss() }.keyboardShortcut(.defaultAction) }
        }
        .padding(24).frame(width: 540)
    }

    private func endpointFields(title: String, endpoint: Binding<String>, model: Binding<String>, placeholder: String) -> some View {
        GroupBox(title) {
            VStack(alignment: .leading) {
                TextField(placeholder, text: endpoint).textFieldStyle(.roundedBorder)
                TextField("Model (optional for local provider)", text: model).textFieldStyle(.roundedBorder)
            }.padding(.top, 4)
        }
    }
}
