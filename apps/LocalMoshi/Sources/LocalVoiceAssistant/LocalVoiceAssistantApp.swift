import SwiftUI

@main
struct LocalMoshiApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var engine = KyutaiEngine()
    @StateObject private var settings = AppSettings()
    @StateObject private var session = KyutaiSession()
    @StateObject private var metrics = SystemMetrics()

    var body: some Scene {
        Window("Local Voice Assistant", id: "assistant") {
            ContentView()
                .environmentObject(engine)
                .environmentObject(settings)
                .environmentObject(session)
                .environmentObject(metrics)
                .frame(width: 320, height: 360)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: 320, height: 360)

        Window("Voice Assistant Debug", id: "debug") {
            DebugDashboardView(session: session, engine: engine, metrics: metrics)
                .environmentObject(settings)
                .frame(minWidth: 980, minHeight: 720)
        }
        .defaultSize(width: 1080, height: 780)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        DispatchQueue.main.async {
            NSApp.activate(ignoringOtherApps: true)
            guard let window = NSApp.windows.first(where: { $0.title == "Local Voice Assistant" }) else { return }
            window.standardWindowButton(.closeButton)?.isHidden = true
            window.standardWindowButton(.miniaturizeButton)?.isHidden = true
            window.standardWindowButton(.zoomButton)?.isHidden = true
            window.isMovableByWindowBackground = true
            window.level = .floating
            window.collectionBehavior.insert(.fullScreenAuxiliary)
            window.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        KyutaiEngine.shared?.stop()
    }
}
