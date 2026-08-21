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
        static let llmURL = "pipeline.llm.url"
        static let llmModel = "pipeline.llm.model"
        static let systemPrompt = "agent.systemPrompt"
        static let systemPromptVersion = "agent.systemPrompt.version"
    }

    private enum FixedVoicePipeline {
        static let vadRepo = "kyutai/stt-1b-en_fr-candle"
        static let vadThreshold = 0.01
        static let minSpeechMS = 320
        static let minSilenceMS = 800
        static let sttRepo = "kyutai/stt-1b-en_fr-candle"
        static let ttsRepo = "kyutai/tts-1.6b-en_fr"
        static let ttsVoiceRepo = "kyutai/tts-voices"
        static let ttsVoice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
        static let ttsQuantize = 8
    }

    private let defaults = UserDefaults.standard
    private static let promptVersion = 3
    private static let defaultBehavior = "Be concise, warm, and natural. Prefer short spoken answers unless the user asks for detail."
    private static let identityPrompt = "You are {{agent_name}}, the voice assistant. {{agent_name}} is your name, never the user's name. The user's name is unknown unless they explicitly provide it. Never greet or address the user as {{agent_name}}."
    private static let previousDefaultSystemPrompt = "\(identityPrompt)\n\n\(defaultBehavior)"
    static let defaultSystemPrompt = """
    You are {{agent_name}}, an AI voice assistant.

    Identity rules:

    - Your name is {{agent_name}}.
    - {{agent_name}} always refers to you, the assistant, not the user.
    - The user's name is unknown unless they explicitly tell you their name.
    - Never call, greet, or address the user as {{agent_name}}.
    - Do not assume the user's name from the conversation, system instructions, metadata, or examples.
    - If the user provides their name, remember it for the conversation and use it only when natural.
    - If you are uncertain who a name refers to, ask for clarification instead of guessing.

    Conversation style:

    - Speak in a warm, natural, and confident manner.
    - Keep responses concise and suitable for spoken conversation.
    - Prefer one or two short paragraphs unless the user requests detail.
    - Answer directly without unnecessary introductions, repeated greetings, or restating the question.
    - Do not introduce yourself repeatedly. Mention your name only when asked or when contextually useful.
    - Avoid overly formal language, filler phrases, and repetitive acknowledgements.
    - Use clear sentences that sound natural when spoken aloud.
    - Ask only one clarification question at a time when more information is required.
    - If the user interrupts or changes the subject, follow their latest request naturally.

    Accuracy and behavior:

    - Do not invent facts, personal details, or conversation history.
    - Clearly acknowledge uncertainty when you do not know something.
    - Correct misunderstandings briefly and respectfully.
    - Follow the user's requested language and communication style when possible.
    - Provide longer explanations only when requested or when additional detail is necessary for safety or correctness.
    """

    @Published var mode: AppMode { didSet { defaults.set(mode.rawValue, forKey: Key.mode) } }
    @Published var agentName: String { didSet { defaults.set(agentName, forKey: Key.agentName) } }
    @Published var llmURL: String { didSet { defaults.set(llmURL, forKey: Key.llmURL) } }
    @Published var llmModel: String { didSet { defaults.set(llmModel, forKey: Key.llmModel) } }
    @Published var llmAPIKey: String
    @Published var systemPrompt: String { didSet { defaults.set(systemPrompt, forKey: Key.systemPrompt) } }

    init() {
        mode = AppMode(rawValue: defaults.string(forKey: Key.mode) ?? "") ?? .assistant
        agentName = defaults.string(forKey: Key.agentName) ?? "Nova"
        llmURL = defaults.string(forKey: Key.llmURL) ?? "http://localhost:1234/v1"
        llmModel = defaults.string(forKey: Key.llmModel) ?? "liquid/lfm2.5-1.2b"
        llmAPIKey = KeychainStore.read(account: "llm-api-key") ?? ""
        let storedPrompt = defaults.string(forKey: Key.systemPrompt)
        let storedVersion = defaults.integer(forKey: Key.systemPromptVersion)
        if storedVersion < 2 {
            let previousBehavior = storedPrompt ?? Self.defaultBehavior
            systemPrompt = "\(Self.identityPrompt)\n\n\(previousBehavior)"
        } else if storedVersion < Self.promptVersion,
                  storedPrompt == nil || storedPrompt == Self.previousDefaultSystemPrompt {
            systemPrompt = Self.defaultSystemPrompt
        } else {
            systemPrompt = storedPrompt ?? Self.defaultSystemPrompt
        }
        defaults.set(systemPrompt, forKey: Key.systemPrompt)
        defaults.set(Self.promptVersion, forKey: Key.systemPromptVersion)
    }

    var validationIssues: [String] {
        var issues: [String] = []
        if agentName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { issues.append("Agent name is required") }
        if URL(string: llmURL)?.scheme == nil { issues.append("Enter a valid LLM endpoint") }
        if llmModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { issues.append("Enter an LLM model name") }
        return issues
    }

    var engineEnvironment: [String: String] {
        let name = agentName.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedSystemPrompt = systemPrompt.replacingOccurrences(
            of: "{{agent_name}}",
            with: name
        )
        return [
            "AGENT_NAME": name,
            "ASSISTANT_SYSTEM": resolvedSystemPrompt,
            "VAD_REPO": FixedVoicePipeline.vadRepo,
            "VAD_THRESHOLD": String(FixedVoicePipeline.vadThreshold),
            "VAD_MIN_SPEECH_MS": String(FixedVoicePipeline.minSpeechMS),
            "VAD_MIN_SILENCE_MS": String(FixedVoicePipeline.minSilenceMS),
            "STT_REPO": FixedVoicePipeline.sttRepo,
            "LLM_BASE_URL": llmURL.trimmingCharacters(in: .whitespacesAndNewlines),
            "LLM_MODEL_NAME": llmModel.trimmingCharacters(in: .whitespacesAndNewlines),
            "LLM_API_KEY": llmAPIKey,
            "TTS_REPO": FixedVoicePipeline.ttsRepo,
            "TTS_VOICE_REPO": FixedVoicePipeline.ttsVoiceRepo,
            "TTS_VOICE": FixedVoicePipeline.ttsVoice,
            "TTS_QUANTIZE": String(FixedVoicePipeline.ttsQuantize),
        ]
    }

    func persistCredential() {
        if llmAPIKey.isEmpty { KeychainStore.delete(account: "llm-api-key") }
        else { KeychainStore.write(llmAPIKey, account: "llm-api-key") }
    }

    func resetSystemPromptToDefault() {
        systemPrompt = Self.defaultSystemPrompt
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
