import AVFoundation
import Foundation

@MainActor
final class VoiceAssistant: ObservableObject {
    @Published var status = "Ready"
    @Published var transcript = ""
    @Published var reply = ""
    @Published var isRecording = false
    @Published var isBusy = false
    @Published var handsFreeEnabled = false

    private let recorder = MicrophoneRecorder()
    private let speaker = AudioQueue()
    let settings = EndpointSettings()
    private var requestTask: Task<Void, Never>?
    private let endpoint = URL(string: ProcessInfo.processInfo.environment["VOICE_AGENT_URL"] ?? "http://127.0.0.1:8787/v1/turn")!
    private let sessionID = UUID().uuidString
    private var heardSpeech = false
    private var silenceTask: Task<Void, Never>?

    init() {
        recorder.onLevel = { [weak self] level in
            Task { @MainActor in self?.handleAudioLevel(level) }
        }
    }

    func toggleRecording() {
        isRecording ? finishRecording() : startRecording()
    }

    func startRecording() {
        cancelTurn()
        transcript = ""
        reply = ""
        do {
            try recorder.start()
            isRecording = true
            status = "Listening…"
        } catch {
            status = "Microphone error: \(error.localizedDescription)"
        }
    }

    func finishRecording() {
        guard isRecording else { return }
        isRecording = false
        do {
            let wav = try recorder.stop()
            status = "Thinking…"
            isBusy = true
            requestTask = Task { [weak self] in await self?.send(wav: wav) }
        } catch {
            status = "Recording error: \(error.localizedDescription)"
        }
    }

    func cancelTurn() {
        recorder.cancel()
        requestTask?.cancel()
        requestTask = nil
        speaker.stop()
        isRecording = false
        silenceTask?.cancel()
        isBusy = false
        status = "Ready"
    }

    func setHandsFree(_ enabled: Bool) {
        if enabled { startHandsFreeListening() } else { cancelTurn() }
    }

    private func startHandsFreeListening() {
        guard handsFreeEnabled, !isBusy, !isRecording else { return }
        heardSpeech = false
        startRecording()
        status = "Listening hands-free…"
    }

    private func handleAudioLevel(_ level: Float) {
        guard handsFreeEnabled, isRecording else { return }
        if level > 0.025 {
            heardSpeech = true
            silenceTask?.cancel()
            silenceTask = nil
        } else if heardSpeech, silenceTask == nil {
            silenceTask = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(700))
                guard !Task.isCancelled else { return }
                self?.finishHandsFreeTurn()
            }
        }
    }

    private func finishHandsFreeTurn() {
        guard handsFreeEnabled, isRecording else { return }
        silenceTask = nil
        finishRecording()
    }

    func clearConversation() {
        cancelTurn()
        transcript = ""
        reply = ""
        var request = URLRequest(url: endpoint.deletingLastPathComponent().appending(path: "sessions/\(sessionID)"))
        request.httpMethod = "DELETE"
        Task { _ = try? await URLSession.shared.data(for: request) }
    }

    private func send(wav: Data) async {
        defer {
            isBusy = false
            if handsFreeEnabled && !Task.isCancelled {
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(1))
                    self.startHandsFreeListening()
                }
            }
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data("--\(boundary)\r\n".utf8)
        body += Data("Content-Disposition: form-data; name=\"audio\"; filename=\"speech.wav\"\r\n".utf8)
        body += Data("Content-Type: audio/wav\r\n\r\n".utf8)
        body += wav
        appendField("session_id", sessionID, to: &body, boundary: boundary)
        appendField("stt_endpoint", settings.sttEndpoint, to: &body, boundary: boundary)
        appendField("stt_model", settings.sttModel, to: &body, boundary: boundary)
        appendField("llm_endpoint", settings.llmEndpoint, to: &body, boundary: boundary)
        appendField("llm_model", settings.llmModel, to: &body, boundary: boundary)
        appendField("tts_endpoint", settings.ttsEndpoint, to: &body, boundary: boundary)
        appendField("tts_model", settings.ttsModel, to: &body, boundary: boundary)
        body += Data("\r\n--\(boundary)--\r\n".utf8)
        request.httpBody = body

        do {
            let (bytes, response) = try await URLSession.shared.bytes(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            for try await line in bytes.lines {
                guard !Task.isCancelled, let data = line.data(using: .utf8), let event = try? JSONDecoder().decode(TurnEvent.self, from: data) else { continue }
                apply(event)
            }
            if !Task.isCancelled { status = "Ready" }
        } catch is CancellationError {
            // User barge-in: no error UI required.
        } catch {
            status = "Service error: \(error.localizedDescription)"
        }
    }

    private func appendField(_ name: String, _ value: String, to body: inout Data, boundary: String) {
        body += Data("\r\n--\(boundary)\r\n".utf8)
        body += Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)".utf8)
    }

    private func apply(_ event: TurnEvent) {
        switch event.type {
        case "transcript": transcript = event.text ?? ""
        case "text": reply += event.delta ?? ""
        case "audio":
            if let encoded = event.wav_base64, let wav = Data(base64Encoded: encoded) { speaker.enqueue(wav) }
        case "error": status = event.message ?? "Unknown service error"
        default: break
        }
    }
}

private struct TurnEvent: Decodable {
    let type: String
    let text: String?
    let delta: String?
    let wav_base64: String?
    let message: String?
}
