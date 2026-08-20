import Foundation

@MainActor
final class EndpointSettings: ObservableObject {
    @Published var sttEndpoint: String { didSet { save("sttEndpoint", sttEndpoint) } }
    @Published var sttModel: String { didSet { save("sttModel", sttModel) } }
    @Published var llmEndpoint: String { didSet { save("llmEndpoint", llmEndpoint) } }
    @Published var llmModel: String { didSet { save("llmModel", llmModel) } }
    @Published var ttsEndpoint: String { didSet { save("ttsEndpoint", ttsEndpoint) } }
    @Published var ttsModel: String { didSet { save("ttsModel", ttsModel) } }

    init() {
        let defaults = UserDefaults.standard
        sttEndpoint = defaults.string(forKey: "sttEndpoint") ?? ""
        sttModel = defaults.string(forKey: "sttModel") ?? ""
        llmEndpoint = defaults.string(forKey: "llmEndpoint") ?? "http://127.0.0.1:11434/v1"
        llmModel = defaults.string(forKey: "llmModel") ?? "qwen3:8b"
        ttsEndpoint = defaults.string(forKey: "ttsEndpoint") ?? ""
        ttsModel = defaults.string(forKey: "ttsModel") ?? ""
    }

    private func save(_ key: String, _ value: String) { UserDefaults.standard.set(value, forKey: key) }
}
