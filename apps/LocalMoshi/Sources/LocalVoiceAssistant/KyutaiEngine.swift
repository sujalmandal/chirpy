import Foundation

@MainActor
final class KyutaiEngine: ObservableObject {
    static weak var shared: KyutaiEngine?
    @Published private(set) var status = "Preparing local voice engine…"
    @Published private(set) var isReady = false
    @Published private(set) var isRunning = false
    @Published private(set) var isStarting = false
    @Published private(set) var logs = "Waiting for the local voice agent…"
    @Published private(set) var sttReady = false
    @Published private(set) var ttsReady = false

    let baseURL = URL(string: "http://127.0.0.1:8999")!
    private var process: Process?
    private var pipe: Pipe?
    private var launched = false
    private var healthTask: Task<Void, Never>?
    private var logHandle: FileHandle?
    private let readyTimeoutSeconds: Double = 120

    init() { Self.shared = self }

    func startIfNeeded() { guard !launched else { return }; launched = true; start() }
    func restart() { stop(); launched = true; start() }

    func start() {
        isStarting = true; isReady = false; sttReady = false; ttsReady = false
        let root = projectRoot()
        let venv = root.appending(path: "engine/kyutai/.venv/bin/python")
        let script = root.appending(path: "engine/kyutai/agent.py")
        guard FileManager.default.isExecutableFile(atPath: venv.path),
              FileManager.default.fileExists(atPath: script.path) else {
            status = "Voice agent is not set up. Run scripts/setup-kyutai.sh first."
            isStarting = false
            return
        }
        stopStaleAgent()
        beginLog(in: root)
        let task = Process()
        task.executableURL = venv
        task.arguments = [script.path]
        task.currentDirectoryURL = root
        let output = Pipe()
        task.standardOutput = output; task.standardError = output
        pipe = output
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.appendLog(text) }
        }
        task.terminationHandler = { [weak self] _ in Task { @MainActor in
            self?.isRunning = false; self?.isReady = false; self?.isStarting = false
            self?.sttReady = false; self?.ttsReady = false
            self?.healthTask?.cancel(); self?.healthTask = nil
            if self?.status == "Local voice engine ready" { self?.status = "Local voice engine stopped" }
        }}
        do {
            status = "Starting local voice agent…"
            try task.run(); process = task; isRunning = true
            pollHealth()
        } catch {
            status = "Could not start voice agent: \(error.localizedDescription)"
            isStarting = false
        }
    }

    func stop() {
        healthTask?.cancel(); healthTask = nil
        pipe?.fileHandleForReading.readabilityHandler = nil
        process?.terminate(); process = nil; pipe = nil
        isReady = false; isRunning = false; isStarting = false
        sttReady = false; ttsReady = false
        closeLog()
    }

    func appendLog(_ text: String) {
        logs = String((logs + text).suffix(20_000))
        if let data = text.data(using: .utf8) { try? logHandle?.write(contentsOf: data) }
        updateStatus(from: text)
    }

    private func beginLog(in root: URL) {
        let directory = root.appending(path: "logs", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appending(path: "local-voice-assistant.log")
        FileManager.default.createFile(atPath: file.path, contents: nil)
        logHandle?.closeFile()
        logHandle = try? FileHandle(forWritingTo: file)
        try? logHandle?.seekToEnd()
    }

    private func closeLog() {
        try? logHandle?.close()
        logHandle = nil
    }

    private func updateStatus(from text: String) {
        if text.contains("STT: loading LM weights") {
            status = "Loading speech recognition model…"
        } else if text.contains("STT ready") {
            sttReady = true
            status = ttsReady ? "Local voice engine ready" : "Loading speech synthesis model…"
        } else if text.contains("TTS: loading LM weights") || text.contains("TTS: quantizing") {
            status = "Loading speech synthesis model…"
        } else if text.contains("TTS ready") {
            ttsReady = true
            status = "Local voice engine ready"
        }
    }

    private func pollHealth() {
        healthTask = Task { [weak self] in
            guard let self else { return }
            let deadline = Date().addingTimeInterval(readyTimeoutSeconds)
            while !Task.isCancelled {
                let healthy = await self.health()
                if self.isReady != healthy.stt || self.isReady != healthy.tts {
                    self.isReady = healthy.stt && healthy.tts
                    self.sttReady = healthy.stt
                    self.ttsReady = healthy.tts
                    self.status = self.isReady ? "Local voice engine ready" : self.status
                    if self.isReady { self.isStarting = false; return }
                }
                if Date() >= deadline {
                    self.isStarting = false
                    self.status = "Voice engine did not start — see logs/kyutai-agent.log"
                    return
                }
                try? await Task.sleep(for: .milliseconds(600))
            }
        }
    }

    private func health() async -> (stt: Bool, tts: Bool) {
        var request = URLRequest(url: baseURL.appending(path: "health"))
        request.timeoutInterval = 1; request.cachePolicy = .reloadIgnoringLocalCacheData
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (false, false)
        }
        return (json["stt_ready"] as? Bool ?? false, json["tts_ready"] as? Bool ?? false)
    }

    private func stopStaleAgent() {
        let pids = runCommand("/usr/sbin/lsof", ["-ti", "tcp:8999"])
            .split(whereSeparator: \.isNewline).compactMap { Int32($0) }
        for pid in pids {
            appendLog("Stopping prior voice agent process (\(pid))\n")
            _ = runCommand("/bin/kill", ["-TERM", String(pid)])
        }
        if !pids.isEmpty { Thread.sleep(forTimeInterval: 0.25) }
    }

    private func runCommand(_ executable: String, _ arguments: [String]) -> String {
        let task = Process(), output = Pipe()
        task.executableURL = URL(fileURLWithPath: executable)
        task.arguments = arguments; task.standardOutput = output; task.standardError = Pipe()
        guard (try? task.run()) != nil else { return "" }
        task.waitUntilExit()
        return String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    private func projectRoot() -> URL {
        let bundleURL = Bundle.main.bundleURL
        if bundleURL.pathExtension == "app" { return bundleURL.deletingLastPathComponent() }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .deletingLastPathComponent().deletingLastPathComponent()
    }
}
