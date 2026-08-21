import SwiftUI
import WebKit

/// Keeps WebKit's WebRTC capture graph attached to a live view hierarchy while
/// remaining visually absent from the native floating-orb interface.
struct WebMicrophoneHost: NSViewRepresentable {
    @ObservedObject var session: KyutaiSession

    func makeNSView(context: Context) -> WKWebView {
        session.webMicrophoneView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {}
}
