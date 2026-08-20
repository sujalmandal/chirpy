import AVFoundation
import Foundation

@MainActor
final class KyutaiSession: NSObject, ObservableObject {
    @Published private(set) var status = "Waiting for microphone permission"
    @Published private(set) var transcript = ""
    @Published private(set) var reply = ""
    @Published private(set) var isListening = false
    @Published private(set) var isSpeaking = false

    private let audioEngine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let sampleRate = 24_000.0
    private let blockSamples = 1_920  // 80 ms of mono Float32 @ 24 kHz
    private let recordingFormat: AVAudioFormat
    private var converter: AVAudioConverter?
    private var socket: URLSessionWebSocketTask?
    private let wsURL = URL(string: "ws://127.0.0.1:9000")!
    private let bargeInThreshold: Float = 0.014
    private var bargeInFrames = 0

    private let pcmLock = NSLock()
    private var pcmBuffer = Data()
    private var sendTask: Task<Void, Never>?

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
        disconnect()
        audioEngine.inputNode.removeTap(onBus: 0); audioEngine.stop()
        player.stop()
        isListening = false; isSpeaking = false; status = "Conversation stopped"
    }

    func interrupt() {
        guard isSpeaking else { return }
        sendJSON(["type": "interrupt"])
        player.stop()
        isSpeaking = false; status = "Listening…"
    }

    private func handleLevel(_ level: Float) {
        if level >= bargeInThreshold {
            bargeInFrames += 1
            if isSpeaking && bargeInFrames >= 4 {
                interrupt()
                bargeInFrames = 0
            }
        } else {
            bargeInFrames = 0
        }
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
                self.pcmLock.lock()
                if self.pcmBuffer.count >= blockSamples * 4 {
                    chunk = self.pcmBuffer.prefix(blockSamples * 4)
                    self.pcmBuffer.removeFirst(blockSamples * 4)
                }
                self.pcmLock.unlock()
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
        let task = URLSession.shared.webSocketTask(with: wsURL)
        socket = task
        task.resume()
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
                    self.handle(message)
                    self.receive()
                case .failure:
                    if self.isListening { self.connect() }
                }
            }
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
            case "turn_started": isSpeaking = true
            case "text":
                if let delta = event["delta"] as? String { reply += delta }
            case "done": isSpeaking = false
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
        let frames = data.count / MemoryLayout<Float>.size
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: recordingFormat, frameCapacity: AVAudioFrameCount(frames)), let samples = buffer.floatChannelData else { return }
        buffer.frameLength = AVAudioFrameCount(frames)
        data.withUnsafeBytes { source in memcpy(samples[0], source.baseAddress!, data.count) }
        player.scheduleBuffer(buffer)
        if !player.isPlaying { player.play() }
    }
}
