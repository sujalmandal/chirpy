import SwiftUI

struct KyutaiConversationView: View {
    @ObservedObject var session: KyutaiSession
    @ObservedObject var engine: KyutaiEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label(session.status, systemImage: session.isListening ? "waveform" : "mic.slash")
                    .foregroundStyle(session.isListening ? .green : .orange)
                Spacer()
                Button(session.isListening ? "Stop" : "Start") { session.isListening ? session.stop() : session.start() }
                Button("Interrupt") { session.interrupt() }.disabled(!session.isSpeaking)
            }.font(.headline)

            WaveformView(micLevel: session.micLevel, speakerLevel: session.speakerLevel)

            GroupBox("You") {
                ScrollView {
                    Text(session.transcript.isEmpty ? "Speak naturally. Kyutai STT detects the end of your turn." : session.transcript)
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }
                .frame(minHeight: 120, alignment: .topLeading)
            }
            GroupBox("Assistant") {
                ScrollView {
                    Text(session.reply.isEmpty ? "Your model's streamed answer appears here and is spoken locally." : session.reply)
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }
                .frame(minHeight: 150, alignment: .topLeading)
            }
            GroupBox("Current engine") {
                VStack(alignment: .leading, spacing: 6) {
                    row("VAD", "Kyutai STT 1B · semantic end-of-turn + barge-in")
                    row("STT", "Kyutai STT 1B (MLX) · English + French")
                    row("LLM", "Configured Ollama Cloud OpenAI-compatible endpoint")
                    row("TTS", "Kyutai TTS 1.6B (MLX) · streaming")
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            GroupBox("Engine log") {
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(engine.logs).font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                            .id("logBottom")
                    }
                    .frame(minHeight: 110, maxHeight: 170, alignment: .topLeading)
                    .onChange(of: engine.logs) { _, _ in
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo("logBottom", anchor: .bottom)
                        }
                    }
                }
            }
        }.padding(20)
    }

    private func row(_ name: String, _ detail: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(name).frame(width: 45, alignment: .leading).foregroundStyle(.cyan)
            Text(detail).foregroundStyle(.secondary)
        }.font(.caption)
    }
}
