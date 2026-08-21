import Foundation
import Security

enum AppMode: String, CaseIterable, Identifiable {
    case assistant
    case debug

    var id: String { rawValue }
    var title: String { self == .assistant ? "Assistant" : "Debug" }
    var icon: String { self == .assistant ? "sparkles" : "wrench.and.screwdriver" }
}

enum PipelineStage: String, CaseIterable, Identifiable {
    case vad = "VAD"
    case stt = "STT"
    case llm = "LLM"
    case tts = "TTS"

    var id: String { rawValue }
    var icon: String {
        switch self {
        case .vad: "waveform.badge.magnifyingglass"
        case .stt: "text.bubble"
        case .llm: "brain.head.profile"
        case .tts: "speaker.wave.2"
        }
    }
}

@MainActor
final class AppSettings: ObservableObject {
    private enum Key {
        static let mode = "ui.mode"
        static let agentName = "agent.name"
        static let vadRepo = "pipeline.vad.repo"
        static let vadThreshold = "pipeline.vad.threshold"
        static let minSpeechMS = "pipeline.vad.minSpeechMS"
        static let minSilenceMS = "pipeline.vad.minSilenceMS"
        static let sttRepo = "pipeline.stt.repo"
        static let llmURL = "pipeline.llm.url"
        static let llmModel = "pipeline.llm.model"
        static let ttsRepo = "pipeline.tts.repo"
        static let ttsVoiceRepo = "pipeline.tts.voiceRepo"
        static let ttsVoice = "pipeline.tts.voice"
        static let ttsQuantize = "pipeline.tts.quantize"
        static let systemPrompt = "agent.systemPrompt"
    }

    private let defaults = UserDefaults.standard

    @Published var mode: AppMode { didSet { defaults.set(mode.rawValue, forKey: Key.mode) } }
    @Published var agentName: String { didSet { defaults.set(agentName, forKey: Key.agentName) } }
    @Published var vadRepo: String { didSet { defaults.set(vadRepo, forKey: Key.vadRepo) } }
    @Published var vadThreshold: Double { didSet { defaults.set(vadThreshold, forKey: Key.vadThreshold) } }
    @Published var minSpeechMS: Int { didSet { defaults.set(minSpeechMS, forKey: Key.minSpeechMS) } }
    @Published var minSilenceMS: Int { didSet { defaults.set(minSilenceMS, forKey: Key.minSilenceMS) } }
    @Published var sttRepo: String { didSet { defaults.set(sttRepo, forKey: Key.sttRepo) } }
    @Published var llmURL: String { didSet { defaults.set(llmURL, forKey: Key.llmURL) } }
    @Published var llmModel: String { didSet { defaults.set(llmModel, forKey: Key.llmModel) } }
    @Published var llmAPIKey: String
    @Published var ttsRepo: String { didSet { defaults.set(ttsRepo, forKey: Key.ttsRepo) } }
    @Published var ttsVoiceRepo: String { didSet { defaults.set(ttsVoiceRepo, forKey: Key.ttsVoiceRepo) } }
    @Published var ttsVoice: String { didSet { defaults.set(ttsVoice, forKey: Key.ttsVoice) } }
    @Published var ttsQuantize: Int { didSet { defaults.set(ttsQuantize, forKey: Key.ttsQuantize) } }
    @Published var systemPrompt: String { didSet { defaults.set(systemPrompt, forKey: Key.systemPrompt) } }

    init() {
        mode = AppMode(rawValue: defaults.string(forKey: Key.mode) ?? "") ?? .assistant
        agentName = defaults.string(forKey: Key.agentName) ?? "Nova"
        vadRepo = defaults.string(forKey: Key.vadRepo) ?? "kyutai/stt-1b-en_fr-candle"
        vadThreshold = defaults.object(forKey: Key.vadThreshold) as? Double ?? 0.01
        minSpeechMS = defaults.object(forKey: Key.minSpeechMS) as? Int ?? 320
        minSilenceMS = defaults.object(forKey: Key.minSilenceMS) as? Int ?? 320
        sttRepo = defaults.string(forKey: Key.sttRepo) ?? "kyutai/stt-1b-en_fr-candle"
        llmURL = defaults.string(forKey: Key.llmURL) ?? "http://localhost:1234/v1"
        llmModel = defaults.string(forKey: Key.llmModel) ?? "liquid/lfm2.5-1.2b"
        llmAPIKey = KeychainStore.read(account: "llm-api-key") ?? ""
        ttsRepo = defaults.string(forKey: Key.ttsRepo) ?? "kyutai/tts-1.6b-en_fr"
        ttsVoiceRepo = defaults.string(forKey: Key.ttsVoiceRepo) ?? "kyutai/tts-voices"
        ttsVoice = defaults.string(forKey: Key.ttsVoice) ?? "expresso/ex03-ex01_happy_001_channel1_334s.wav"
        ttsQuantize = defaults.object(forKey: Key.ttsQuantize) as? Int ?? 8
        systemPrompt = defaults.string(forKey: Key.systemPrompt) ?? "Be concise, warm, and natural. Prefer short spoken answers unless the user asks for detail."
    }

    var validationIssues: [String] {
        var issues: [String] = []
        if agentName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { issues.append("Agent name is required") }
        if vadRepo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { issues.append("Choose a VAD model") }
        if sttRepo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { issues.append("Choose an STT model") }
        if URL(string: llmURL)?.scheme == nil { issues.append("Enter a valid LLM endpoint") }
        if llmModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { issues.append("Enter an LLM model name") }
        if ttsRepo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { issues.append("Choose a TTS model") }
        return issues
    }

    var engineEnvironment: [String: String] {
        let name = agentName.trimmingCharacters(in: .whitespacesAndNewlines)
        return [
            "AGENT_NAME": name,
            "ASSISTANT_SYSTEM": "Your name is \(name). \(systemPrompt)",
            "VAD_REPO": vadRepo.trimmingCharacters(in: .whitespacesAndNewlines),
            "VAD_THRESHOLD": String(vadThreshold),
            "VAD_MIN_SPEECH_MS": String(minSpeechMS),
            "VAD_MIN_SILENCE_MS": String(minSilenceMS),
            "STT_REPO": sttRepo.trimmingCharacters(in: .whitespacesAndNewlines),
            "LLM_BASE_URL": llmURL.trimmingCharacters(in: .whitespacesAndNewlines),
            "LLM_MODEL_NAME": llmModel.trimmingCharacters(in: .whitespacesAndNewlines),
            "LLM_API_KEY": llmAPIKey,
            "TTS_REPO": ttsRepo.trimmingCharacters(in: .whitespacesAndNewlines),
            "TTS_VOICE_REPO": ttsVoiceRepo.trimmingCharacters(in: .whitespacesAndNewlines),
            "TTS_VOICE": ttsVoice.trimmingCharacters(in: .whitespacesAndNewlines),
            "TTS_QUANTIZE": String(ttsQuantize),
        ]
    }

    func persistCredential() {
        if llmAPIKey.isEmpty { KeychainStore.delete(account: "llm-api-key") }
        else { KeychainStore.write(llmAPIKey, account: "llm-api-key") }
    }
}

private enum KeychainStore {
    private static let service = "dev.localvoiceassistant.settings"

    static func read(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func write(_ value: String, account: String) {
        delete(account: account)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: Data(value.utf8),
        ]
        SecItemAdd(query as CFDictionary, nil)
    }

    static func delete(account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
