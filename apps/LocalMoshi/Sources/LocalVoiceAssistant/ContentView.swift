import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var engine: KyutaiEngine
    @StateObject private var session = KyutaiSession()
    @StateObject private var metrics = SystemMetrics()

    var body: some View {
        VStack(spacing: 0) {
            dashboard
            Divider()
            if engine.isReady {
                KyutaiConversationView(session: session, engine: engine)
            } else {
                ContentUnavailableView("Preparing voice engine", systemImage: "waveform", description: Text(engine.status))
            }
        }
        .task { engine.startIfNeeded(); metrics.start(engine: engine) }
        .task(id: engine.isReady) { if engine.isReady { session.start() } else { session.stop() } }
        .onDisappear { session.stop(); metrics.stop() }
    }

    private var dashboard: some View {
        HStack(spacing: 18) {
            Label(engine.status, systemImage: engine.isReady ? "checkmark.circle.fill" : "arrow.triangle.2.circlepath")
                .foregroundStyle(engine.isReady ? .green : .orange)
            Spacer()
            metric("CPU", metrics.cpu)
            metric("GPU", metrics.gpu)
            metric("RAM", metrics.ram)
            Button(engine.isRunning ? "Restart" : "Start") { engine.restart() }.disabled(engine.isStarting)
        }
        .padding(14)
        .background(.bar)
    }

    private func metric(_ name: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 1) { Text(name).font(.caption2).foregroundStyle(.secondary); Text(value).font(.system(.body, design: .monospaced)) }
    }
}
