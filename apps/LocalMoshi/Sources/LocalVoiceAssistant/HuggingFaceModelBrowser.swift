import SwiftUI

struct HuggingFaceModel: Codable, Identifiable, Hashable {
    let id: String
    let downloads: Int?
    let likes: Int?
    let pipelineTag: String?

    enum CodingKeys: String, CodingKey {
        case id, downloads, likes
        case pipelineTag = "pipeline_tag"
    }
}

@MainActor
final class HuggingFaceModelSearch: ObservableObject {
    @Published var models: [HuggingFaceModel] = []
    @Published var isLoading = false
    @Published var error: String?
    private var task: Task<Void, Never>?

    func search(_ text: String, stage: PipelineStage) {
        task?.cancel()
        task = Task {
            isLoading = true
            error = nil
            defer { isLoading = false }
            var components = URLComponents(string: "https://huggingface.co/api/models")!
            components.queryItems = [
                URLQueryItem(name: "pipeline_tag", value: pipelineTag(for: stage)),
                URLQueryItem(name: "search", value: searchTerm(text, stage: stage)),
                URLQueryItem(name: "sort", value: "downloads"),
                URLQueryItem(name: "direction", value: "-1"),
                URLQueryItem(name: "limit", value: "40"),
            ]
            do {
                var request = URLRequest(url: components.url!)
                request.timeoutInterval = 15
                let (data, response) = try await URLSession.shared.data(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                    throw URLError(.badServerResponse)
                }
                models = try JSONDecoder().decode([HuggingFaceModel].self, from: data)
            } catch is CancellationError {
                return
            } catch {
                self.error = "Hugging Face could not be reached. You can still enter a repository ID manually."
                models = []
            }
        }
    }

    private func pipelineTag(for stage: PipelineStage) -> String {
        switch stage {
        case .vad: "audio-classification"
        case .stt: "automatic-speech-recognition"
        case .tts: "text-to-speech"
        case .llm: "text-generation"
        }
    }

    private func searchTerm(_ text: String, stage: PipelineStage) -> String {
        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return text }
        return switch stage {
        case .vad: "voice activity detection"
        case .stt: "kyutai stt"
        case .tts: "kyutai tts"
        case .llm: ""
        }
    }
}

struct HuggingFaceModelBrowser: View {
    let stage: PipelineStage
    @Binding var selection: String
    @Environment(\.dismiss) private var dismiss
    @StateObject private var search = HuggingFaceModelSearch()
    @State private var query = ""

    private var recommendations: [String] {
        switch stage {
        case .vad: ["kyutai/stt-1b-en_fr-candle", "snakers4/silero-vad"]
        case .stt: ["kyutai/stt-1b-en_fr-candle", "openai/whisper-large-v3-turbo", "mlx-community/whisper-large-v3-turbo"]
        case .tts: ["kyutai/tts-1.6b-en_fr", "hexgrad/Kokoro-82M", "kyutai/pocket-tts"]
        case .llm: []
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Image(systemName: stage.icon).font(.title2).foregroundStyle(.cyan)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Choose \(stage.rawValue) model").font(.title2.bold())
                    Text("Browse public repositories on Hugging Face").foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            .padding(20)

            Divider()

            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search models or paste owner/repository", text: $query)
                    .textFieldStyle(.plain)
                    .onSubmit { search.search(query, stage: stage) }
                if search.isLoading { ProgressView().controlSize(.small) }
                Button("Search") { search.search(query, stage: stage) }
            }
            .padding(12)
            .background(.quaternary.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
            .padding(20)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if !recommendations.isEmpty {
                        Text("Recommended").font(.headline).padding(.bottom, 2)
                        ForEach(recommendations, id: \.self) { model in
                            modelRow(model, detail: compatibility(for: model))
                        }
                        Divider().padding(.vertical, 8)
                    }
                    if let error = search.error {
                        Label(error, systemImage: "wifi.exclamationmark")
                            .foregroundStyle(.orange)
                            .padding(.vertical, 8)
                    }
                    if !search.models.isEmpty {
                        Text("Hugging Face results").font(.headline).padding(.bottom, 2)
                        ForEach(search.models) { model in
                            modelRow(model.id, detail: metadata(for: model))
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
            }
        }
        .frame(minWidth: 680, minHeight: 560)
        .task { search.search("", stage: stage) }
    }

    private func modelRow(_ model: String, detail: String) -> some View {
        Button {
            selection = model
            dismiss()
        } label: {
            HStack(spacing: 12) {
                Image(systemName: selection == model ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(selection == model ? .cyan : .secondary)
                VStack(alignment: .leading, spacing: 3) {
                    Text(model).font(.body.weight(.medium)).foregroundStyle(.primary)
                    Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
                Spacer()
                Image(systemName: "arrow.down.to.line.compact").foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
            .padding(12)
            .background(selection == model ? Color.cyan.opacity(0.09) : Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(.plain)
    }

    private func metadata(for model: HuggingFaceModel) -> String {
        let count = model.downloads.map { $0.formatted(.number.notation(.compactName)) + " downloads" } ?? "Public model"
        return "\(count) · verify compatibility with the selected local adapter"
    }

    private func compatibility(for model: String) -> String {
        if model.hasPrefix("kyutai/") { return "Optimized for the app's current MLX adapter on Apple Silicon" }
        return "Available on Hugging Face · may require an additional runtime adapter"
    }
}
