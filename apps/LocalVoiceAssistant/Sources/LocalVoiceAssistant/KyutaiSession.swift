@preconcurrency import AVFoundation
@preconcurrency import WebKit
import Foundation

enum VoiceChatRole: String {
    case user = "You"
    case assistant = "Assistant"
}

enum VoiceTurnState: Equatable {
    case streaming
    case completed
    case cancelled(String)
    case failed(String)
}

struct VoiceChatMessage: Identifiable, Equatable {
    let id: UUID
    var turnID: Int?
    let role: VoiceChatRole
    var text: String
    let timestamp: Date
    var state: VoiceTurnState
}

@MainActor
final class KyutaiSession: NSObject, ObservableObject, WKScriptMessageHandler, WKUIDelegate, WKNavigationDelegate {
    @Published private(set) var status = "Waiting for microphone permission"
    @Published private(set) var transcript = ""
    @Published private(set) var reply = ""
    @Published private(set) var isListening = false
    @Published private(set) var isSpeaking = false
    @Published private(set) var isOutputMuted = false
    @Published private(set) var micLevel: Float = 0
    @Published private(set) var speakerLevel: Float = 0
    @Published private(set) var messages: [VoiceChatMessage] = []

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
    private var voiceProcessingEnabled = false
    private var voiceProcessingWatchdog: Task<Void, Never>?
    private var webCaptureWatchdog: Task<Void, Never>?
    private var preferWebCapture = true
    private var webCaptureReady = false
    private var webEchoCancellation = false
    private var webAudioBlocks = 0
    private var webFallbackInProgress = false

    private let pcmLock = NSLock()
    private var pcmBuffer = Data()
    private var inputFrameSeen = false
    private var inputPeak: Float = 0
    private var sendTask: Task<Void, Never>?
    private var pendingBuffers = 0
    private var turnDone = false
    private var pendingUserMessageID: UUID?
    private var activeAssistantMessageID: UUID?

    lazy var webMicrophoneView: WKWebView = {
        let controller = WKUserContentController()
        controller.add(self, name: "microphoneBridge")
        let configuration = WKWebViewConfiguration()
        configuration.userContentController = controller
        configuration.mediaTypesRequiringUserActionForPlayback = []
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.uiDelegate = self
        view.navigationDelegate = self
        view.setValue(false, forKey: "drawsBackground")
        return view
    }()

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
                    granted ? self?.beginPreferredListening() : self?.setDeniedStatus()
                }
            }
        } else if permission == .authorized {
            beginPreferredListening()
        } else {
            setDeniedStatus()
        }
    }

    private func setDeniedStatus() {
        status = "Microphone permission denied — allow it in System Settings > Privacy & Security > Microphone, then restart."
    }

    private func beginPreferredListening() {
        if preferWebCapture { beginWebListening() }
        else { beginNativeListening() }
    }

    private func beginWebListening() {
        do {
            webCaptureReady = false
            webEchoCancellation = false
            webAudioBlocks = 0
            webFallbackInProgress = false
            audioEngine.prepare()
            try audioEngine.start()
            player.play()
            isListening = true
            status = "Starting echo-cancelled microphone…"
            connect()
            startSending()
            let url = URL(string: "http://127.0.0.1:8999/microphone?\(UUID().uuidString)")!
            webMicrophoneView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
            startWebCaptureWatchdog()
        } catch {
            fallbackToNativeCapture(reason: "Web microphone output setup failed")
        }
    }

    private func beginNativeListening(enableVoiceProcessing: Bool = false) {
        let input = audioEngine.inputNode
        do {
            // Voice processing gives the endpoint detector an echo-cancelled
            // microphone stream. It must be configured while the engine is
            // stopped and before the tap is installed.
            if enableVoiceProcessing {
                do {
                    try input.setVoiceProcessingEnabled(true)
                    voiceProcessingEnabled = input.isVoiceProcessingEnabled
                } catch {
                    voiceProcessingEnabled = false
                }
            } else {
                if input.isVoiceProcessingEnabled {
                    try? input.setVoiceProcessingEnabled(false)
                }
                voiceProcessingEnabled = false
            }
            let format = input.inputFormat(forBus: 0)
            converter = AVAudioConverter(from: format, to: recordingFormat)
            pcmLock.withLock {
                inputFrameSeen = false
                inputPeak = 0
            }
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
                    self.inputFrameSeen = true
                    self.inputPeak = max(self.inputPeak, level)
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
            startVoiceProcessingWatchdogIfNeeded()
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
        voiceProcessingWatchdog?.cancel(); voiceProcessingWatchdog = nil
        webCaptureWatchdog?.cancel(); webCaptureWatchdog = nil
        webMicrophoneView.evaluateJavaScript("window.stopCapture?.()")
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

    private func startWebCaptureWatchdog() {
        webCaptureWatchdog?.cancel()
        webCaptureWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(7))
            guard let self, !Task.isCancelled else { return }
            guard self.webCaptureReady, self.webEchoCancellation, self.webAudioBlocks > 0 else {
                self.fallbackToNativeCapture(reason: "WebRTC microphone did not become ready")
                return
            }
        }
    }

    private func fallbackToNativeCapture(reason: String) {
        guard !webFallbackInProgress else { return }
        webFallbackInProgress = true
        reportCapture(stage: "web_fallback", details: reason)
        preferWebCapture = false
        status = "\(reason). Retrying native microphone…"
        webCaptureWatchdog?.cancel(); webCaptureWatchdog = nil
        webMicrophoneView.evaluateJavaScript("window.stopCapture?.()")
        sendTask?.cancel(); sendTask = nil
        reconnectTask?.cancel(); reconnectTask = nil
        disconnect()
        audioEngine.stop()
        player.stop()
        pcmLock.withLock { pcmBuffer.removeAll(keepingCapacity: true) }
        pendingBuffers = 0
        isListening = false
        beginNativeListening()
    }

    private func startVoiceProcessingWatchdogIfNeeded() {
        voiceProcessingWatchdog?.cancel()
        guard voiceProcessingEnabled else { return }
        voiceProcessingWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard let self, !Task.isCancelled else { return }
            let inputHealth = self.pcmLock.withLock { (self.inputFrameSeen, self.inputPeak) }
            // Some macOS/device combinations deliver voice-processing buffers
            // containing near-zero samples. Treat that as a failed capture
            // path even though the audio tap itself is firing.
            guard !inputHealth.0 || inputHealth.1 < 0.0001 else { return }
            self.restartWithoutVoiceProcessing()
        }
    }

    private func restartWithoutVoiceProcessing() {
        status = "Retrying microphone without echo cancellation…"
        voiceProcessingWatchdog?.cancel(); voiceProcessingWatchdog = nil
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
        isListening = false
        beginNativeListening(enableVoiceProcessing: false)
    }

    func interrupt() {
        guard isSpeaking else { return }
        sendJSON(["type": "interrupt"])
        player.stop()
        webMicrophoneView.evaluateJavaScript("window.stopPlayback?.()")
        pendingBuffers = 0
        turnDone = false
        isSpeaking = false; status = "Listening…"
    }

    func toggleOutputMuted() {
        isOutputMuted.toggle()
        if isOutputMuted {
            player.stop()
            webMicrophoneView.evaluateJavaScript("window.stopPlayback?.()")
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

    private func acceptWebPCM(_ data: Data) {
        guard preferWebCapture, webCaptureReady else { return }
        var level: Float = 0
        data.withUnsafeBytes { raw in
            let samples = raw.bindMemory(to: Float.self)
            guard !samples.isEmpty else { return }
            var power: Float = 0
            for sample in samples { power += sample * sample }
            level = sqrt(power / Float(samples.count))
        }
        pcmLock.withLock {
            pcmBuffer.append(data)
            webAudioBlocks += 1
        }
        if webAudioBlocks == 1 {
            reportCapture(stage: "first_audio", details: "bytes=\(data.count) level=\(level)")
        }
        handleLevel(level)
    }

    private func reportCapture(stage: String, details: String) {
        sendJSON([
            "type": "capture_diagnostic",
            "stage": stage,
            "details": details,
        ])
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
        sendJSON([
            "type": "audio_processing",
            "voice_processing": webCaptureReady ? webEchoCancellation : voiceProcessingEnabled,
        ])
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
                finishUserMessage(
                    text: transcript,
                    turnID: event["turn_id"] as? Int,
                    timestamp: eventDate(event)
                )
            case "partial":
                transcript = event["text"] as? String ?? ""
                updateUserPartial(transcript, timestamp: eventDate(event))
            case "turn_started":
                isSpeaking = true
                turnDone = false
                beginAssistantMessage(
                    turnID: event["turn_id"] as? Int,
                    timestamp: eventDate(event)
                )
            case "text":
                if let delta = event["delta"] as? String {
                    reply += delta
                    appendAssistantText(delta)
                }
            case "done":
                turnDone = true
                if pendingBuffers <= 0 {
                    finishAssistantMessage()
                    isSpeaking = false
                    sendJSON(["type": "playback_done"])
                }
            case "interrupted":
                player.stop()
                webMicrophoneView.evaluateJavaScript("window.stopPlayback?.()")
                pendingBuffers = 0
                turnDone = false
                isSpeaking = false
                cancelAssistantMessage(reason: event["reason"] as? String ?? "interrupted")
            case "error":
                let message = event["message"] as? String ?? "Agent error"
                status = message
                failAssistantMessage(reason: message)
            default: break
            }
        default: break
        }
    }

    private func eventDate(_ event: [String: Any]) -> Date {
        guard let timestamp = event["timestamp"] as? Double else { return Date() }
        return Date(timeIntervalSince1970: timestamp)
    }

    private func updateUserPartial(_ text: String, timestamp: Date) {
        if let id = pendingUserMessageID, let index = messages.firstIndex(where: { $0.id == id }) {
            messages[index].text = text
        } else {
            let message = VoiceChatMessage(
                id: UUID(), turnID: nil, role: .user, text: text,
                timestamp: timestamp, state: .streaming
            )
            pendingUserMessageID = message.id
            appendMessage(message)
        }
    }

    private func finishUserMessage(text: String, turnID: Int?, timestamp: Date) {
        if let id = pendingUserMessageID, let index = messages.firstIndex(where: { $0.id == id }) {
            messages[index].text = text
            messages[index].turnID = turnID
            messages[index].state = .completed
        } else {
            appendMessage(
                VoiceChatMessage(
                    id: UUID(), turnID: turnID, role: .user, text: text,
                    timestamp: timestamp, state: .completed
                )
            )
        }
        pendingUserMessageID = nil
    }

    private func beginAssistantMessage(turnID: Int?, timestamp: Date) {
        let message = VoiceChatMessage(
            id: UUID(), turnID: turnID, role: .assistant, text: "",
            timestamp: timestamp, state: .streaming
        )
        activeAssistantMessageID = message.id
        appendMessage(message)
    }

    private func appendAssistantText(_ delta: String) {
        guard let id = activeAssistantMessageID,
              let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].text += delta
    }

    private func finishAssistantMessage() {
        guard let id = activeAssistantMessageID,
              let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].state = .completed
        activeAssistantMessageID = nil
    }

    private func cancelAssistantMessage(reason: String) {
        guard let id = activeAssistantMessageID,
              let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].state = .cancelled(reason)
        activeAssistantMessageID = nil
    }

    private func failAssistantMessage(reason: String) {
        guard let id = activeAssistantMessageID,
              let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].state = .failed(reason)
        activeAssistantMessageID = nil
    }

    private func appendMessage(_ message: VoiceChatMessage) {
        messages.append(message)
        if messages.count > 100 { messages.removeFirst(messages.count - 100) }
    }

    private func sendJSON(_ object: [String: Any]) {
        guard let socket, let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8) else { return }
        socket.send(.string(text)) { _ in }
    }

    private func playPCM(_ data: Data) {
        guard !isOutputMuted else { return }
        let frames = data.count / MemoryLayout<Float>.size
        guard frames > 0 else { return }
        var power: Float = 0
        data.withUnsafeBytes { raw in
            let samples = raw.bindMemory(to: Float.self)
            for sample in samples { power += sample * sample }
        }
        speakerLevel = sqrt(power / Float(frames))
        pendingBuffers += 1
        if webCaptureReady && webEchoCancellation {
            let encoded = data.base64EncodedString()
            webMicrophoneView.evaluateJavaScript("window.playPCM?.('\(encoded)')") { [weak self] result, error in
                Task { @MainActor in
                    guard let self else { return }
                    if error != nil || (result as? Bool) == false {
                        self.pendingBuffers -= 1
                        self.fallbackToNativeCapture(reason: "WebRTC playback failed")
                    }
                }
            }
            return
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: recordingFormat,
            frameCapacity: AVAudioFrameCount(frames)
        ), let samples = buffer.floatChannelData else {
            pendingBuffers -= 1
            return
        }
        buffer.frameLength = AVAudioFrameCount(frames)
        _ = data.withUnsafeBytes { source in memcpy(samples[0], source.baseAddress!, data.count) }
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
                finishAssistantMessage()
                isSpeaking = false
                sendJSON(["type": "playback_done"])
            }
        }
    }

    // -- Hidden WebRTC microphone bridge -------------------------------------
    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard message.name == "microphoneBridge",
              let body = message.body as? [String: Any],
              let type = body["type"] as? String else { return }
        switch type {
        case "page_ready":
            reportCapture(stage: "page_ready", details: "starting getUserMedia")
            webMicrophoneView.evaluateJavaScript("window.startCapture?.()")
        case "ready":
            let echoCancellation = body["echoCancellation"] as? Bool ?? false
            reportCapture(
                stage: "webrtc_ready",
                details: "echo=\(echoCancellation) noise=\(body["noiseSuppression"] ?? false) " +
                    "agc=\(body["autoGainControl"] ?? false) rate=\(body["sampleRate"] ?? "unknown")"
            )
            guard echoCancellation else {
                fallbackToNativeCapture(reason: "WebRTC echo cancellation is unavailable")
                return
            }
            webCaptureReady = true
            webEchoCancellation = true
            status = "Listening — WebRTC echo cancellation active"
            sendJSON(["type": "audio_processing", "voice_processing": true])
        case "audio":
            guard let encoded = body["pcm"] as? String,
                  let data = Data(base64Encoded: encoded),
                  data.count == blockSamples * MemoryLayout<Float>.size else { return }
            acceptWebPCM(data)
        case "playback_block_done":
            bufferFinished()
        case "playback_reference":
            guard let rms = body["rms"] as? Double else { return }
            sendJSON(["type": "playback_reference", "rms": max(0, rms)])
        case "error":
            let message = body["message"] as? String ?? "unknown WebRTC error"
            reportCapture(stage: "webrtc_error", details: message)
            fallbackToNativeCapture(reason: message)
        default:
            break
        }
    }

    func webView(
        _ webView: WKWebView,
        requestMediaCapturePermissionFor origin: WKSecurityOrigin,
        initiatedByFrame frame: WKFrameInfo,
        type: WKMediaCaptureType,
        decisionHandler: @escaping (WKPermissionDecision) -> Void
    ) {
        decisionHandler(type == .microphone ? .grant : .deny)
    }

    func webView(
        _ webView: WKWebView,
        didFail navigation: WKNavigation!,
        withError error: Error
    ) {
        fallbackToNativeCapture(reason: "Web microphone page failed: \(error.localizedDescription)")
    }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
    ) {
        fallbackToNativeCapture(reason: "Web microphone page failed: \(error.localizedDescription)")
    }
}
