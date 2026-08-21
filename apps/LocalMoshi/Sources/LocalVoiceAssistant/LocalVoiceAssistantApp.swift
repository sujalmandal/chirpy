import SwiftUI

@main
struct LocalMoshiApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var engine = KyutaiEngine()
    @StateObject private var settings = AppSettings()

    var body: some Scene {
        WindowGroup("Local Moshi") {
            ContentView()
                .environmentObject(engine)
                .environmentObject(settings)
                .frame(minWidth: 980, minHeight: 720)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1080, height: 780)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        DispatchQueue.main.async {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        KyutaiEngine.shared?.stop()
    }
}
