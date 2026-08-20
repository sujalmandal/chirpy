// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "LocalMoshi",
    platforms: [.macOS(.v14)],
    targets: [.executableTarget(name: "LocalMoshi", path: "Sources/LocalVoiceAssistant")]
)
