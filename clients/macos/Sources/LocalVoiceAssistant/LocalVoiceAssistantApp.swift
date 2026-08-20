import AppKit
import SwiftUI

@main
struct LocalVoiceAssistantApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var assistant = VoiceAssistant()

    var body: some Scene {
        WindowGroup("Local Voice Assistant") {
            ContentView()
                .environmentObject(assistant)
                .frame(minWidth: 520, minHeight: 400)
        }
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
}
