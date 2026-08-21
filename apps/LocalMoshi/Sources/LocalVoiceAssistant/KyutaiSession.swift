@preconcurrency import AVFoundation
import Foundation

@MainActor
final class KyutaiSession: NSObject, ObservableObject {
    @Published private(set) var status = "Waiting for microphone permission"
    @Published private(set) var transcript = ""
    @Published private(set) var reply = ""
    @Published private(set) var isListening = false
    @Published private(set) var isSpeaking = false
    @Published private(set) var isOutputMuted = false
    @Published private(set) var micLevel: Float = 0
    @Published private(set) var speakerLevel: Float = 0

    private let audioEngine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let sampleRate = 24_000.0
    private let blockSamples = 1_920  // 80 ms of mono Float32 @ 24 kHz
    private let recordingFormat: AVAudioFormat
    private var converter: AVAudioConverter?
    private var socket: URLSessionWebSocketTask?
    private let wsURL = URL(string: "ws://127.0.0.1:9000")!
    private var reconnectTask: Task<Void, Never>?
    private var reconnectAttempt = 0
    private var tapInstalled = false

    private let pcmLock = NSLock()
    private var pcmBuffer = Data()
    private var sendTask: Task<Void, Never>?
    private var pendingBuffers = 0
    private var turnDone = false

    override init() {
        recordingFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        super.init()
        audioEngine.attach(player)
        audioEngine.connect(player, to: audioEngine.mainMixerNode, format: recordingFormat)
    }

    func start() {
        guard !isListening else { return }
        let permission = AVCaptureDevice.authorizationStatus(for: .audio)
        if permission == .notDetermined {
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                Task { @MainActor in
                    granted ? self?.beginListening() : self?.setDeniedStatus()
                }
            }
        } else if permission == .authorized {
            beginListening()
        } else {
            setDeniedStatus()
        }
    }

    private func setDeniedStatus() {
        status = "Microphone permission denied — allow it in System Settings > Privacy & Security > Microphone, then restart."
    }

    private func beginListening() {
        let input = audioEngine.inputNode
        do {
            // Note: setVoiceProcessingEnabled(true) (echo cancellation) leaves
            // the tap silent on this Mac, so it stays off. The agent discards
            // mic audio while the assistant is speaking, which prevents the
            // echo loop instead.
            let format = input.inputFormat(forBus: 0)
            converter = AVAudioConverter(from: format, to: recordingFormat)
            input.installTap(onBus: 0, bufferSize: AVAudioFrameCount(format.sampleRate / 100), format: format) { [weak self] buffer, _ in
                guard let self else { return }
                var level: Float = 0
                if let samples = buffer.floatChannelData?[0], buffer.frameLength > 0 {
                    var power: Float = 0
                    for i in 0..<Int(buffer.frameLength) { power += samples[i] * samples[i] }
                    level = sqrt(power / Float(buffer.frameLength))
                }
                // Convert on the render thread (realtime-safe) and enqueue.
                if let data = self.convert(buffer) {
                    self.pcmLock.lock()
                    self.pcmBuffer.append(data)
                    self.pcmLock.unlock()
                }
                Task { @MainActor in self.handleLevel(level) }
            }
            tapInstalled = true
            audioEngine.prepare(); try audioEngine.start()
            player.play()
            isListening = true; status = "Listening — speak naturally"
            connect()
            startSending()
        } catch {
            let permission = AVCaptureDevice.authorizationStatus(for: .audio)
            if permission == .denied || permission == .restricted {
                setDeniedStatus()
            } else {
                status = "Could not start microphone: \(error.localizedDescription)"
            }
        }
    }

    func stop() {
        sendTask?.cancel(); sendTask = nil
        reconnectTask?.cancel(); reconnectTask = nil
        disconnect()
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        audioEngine.stop()
        player.stop()
        pcmLock.withLock { pcmBuffer.removeAll(keepingCapacity: true) }
        pendingBuffers = 0
        reconnectAttempt = 0
        micLevel = 0; speakerLevel = 0
        isListening = false; isSpeaking = false; status = "Conversation stopped"
    }

    func interrupt() {
        guard isSpeaking else { return }
        sendJSON(["type": "interrupt"])
        player.stop()
        pendingBuffers = 0
        turnDone = false
        isSpeaking = false; status = "Listening…"
    }

    func toggleOutputMuted() {
        isOutputMuted.toggle()
        if isOutputMuted {
            player.stop()
            pendingBuffers = 0
            speakerLevel = 0
        } else if isListening {
            status = "Listening — speak naturally"
        }
    }

    private func handleLevel(_ level: Float) {
        // Display-only: energy-based barge-in is disabled (without echo
        // cancellation the assistant's own voice would trigger it). This level
        // just drives the mic waveform in the UI.
        micLevel = level
    }

    // -- Audio conversion (render-thread safe) --------------------------------
    private func convert(_ buffer: AVAudioPCMBuffer) -> Data? {
        guard let converter else { return nil }
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * sampleRate / buffer.format.sampleRate + 128)
        guard let output = AVAudioPCMBuffer(pcmFormat: recordingFormat, frameCapacity: capacity) else { return nil }
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in status.pointee = .haveData; return buffer }
        guard error == nil, let samples = output.floatChannelData, output.frameLength > 0 else { return nil }
        let count = Int(output.frameLength) * MemoryLayout<Float>.size
        return Data(bytes: samples[0], count: count)
    }

    // -- Audio send loop (80 ms chunks, proper backpressure) -------------------
    private func startSending() {
        sendTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                var chunk: Data?
                self.pcmLock.withLock {
                    if self.pcmBuffer.count >= self.blockSamples * 4 {
                        chunk = self.pcmBuffer.prefix(self.blockSamples * 4)
                        self.pcmBuffer.removeFirst(self.blockSamples * 4)
                    }
                }
                if let chunk, let socket = self.socket {
                    try? await socket.send(.data(chunk))
                } else {
                    try? await Task.sleep(for: .milliseconds(20))
                }
            }
        }
    }

    // -- WebSocket ----------------------------------------------------------
    private func connect() {
        guard isListening, socket == nil else { return }
        let task = URLSession.shared.webSocketTask(with: wsURL)
        socket = task
        task.resume()
        status = reconnectAttempt == 0 ? "Connecting to local voice engine…" : "Reconnecting…"
        receive()
    }

    private func disconnect() {
        socket?.cancel(with: .goingAway, reason: nil); socket = nil
    }

    private func receive() {
        socket?.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let message):
                    self.reconnectAttempt = 0
                    self.status = self.isSpeaking ? "Speaking…" : "Listening — speak naturally"
                    self.handle(message)
                    self.receive()
                case .failure:
                    self.socket = nil
                    self.scheduleReconnect()
                }
            }
        }
    }

    private func scheduleReconnect() {
        guard isListening, reconnectTask == nil else { return }
        reconnectAttempt += 1
        let delay = min(pow(2.0, Double(reconnectAttempt - 1)) * 0.35, 5.0)
        reconnectTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard let self, !Task.isCancelled else { return }
            self.reconnectTask = nil
            self.connect()
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .data(let data): playPCM(data)
        case .string(let text):
            guard let event = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else { return }
            switch event["type"] as? String {
            case "transcript":
                transcript = event["text"] as? String ?? ""
                reply = ""
            case "partial":
                transcript = event["text"] as? String ?? ""
            case "turn_started":
                isSpeaking = true
                turnDone = false
            case "text":
                if let delta = event["delta"] as? String { reply += delta }
            case "done":
                turnDone = true
                if pendingBuffers <= 0 { isSpeaking = false; sendJSON(["type": "playback_done"]) }
            case "interrupted":
                player.stop()
                pendingBuffers = 0
                turnDone = false
                isSpeaking = false
            case "error": status = event["message"] as? String ?? "Agent error"
            default: break
            }
        default: break
        }
    }

    private func sendJSON(_ object: [String: Any]) {
        guard let socket, let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8) else { return }
        socket.send(.string(text)) { _ in }
    }

    private func playPCM(_ data: Data) {
        guard !isOutputMuted else { return }
        let frames = data.count / MemoryLayout<Float>.size
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: recordingFormat, frameCapacity: AVAudioFrameCount(frames)), let samples = buffer.floatChannelData else { return }
        buffer.frameLength = AVAudioFrameCount(frames)
        _ = data.withUnsafeBytes { source in memcpy(samples[0], source.baseAddress!, data.count) }
        var power: Float = 0
        for i in 0..<frames { power += samples[0][i] * samples[0][i] }
        speakerLevel = sqrt(power / Float(frames))
        pendingBuffers += 1
        player.scheduleBuffer(buffer) { [weak self] in
            Task { @MainActor in self?.bufferFinished() }
        }
        if !player.isPlaying { player.play() }
    }

    private func bufferFinished() {
        pendingBuffers -= 1
        if pendingBuffers <= 0 {
            speakerLevel = 0
            if turnDone {
                isSpeaking = false
                sendJSON(["type": "playback_done"])
            }
        }
    }
}
