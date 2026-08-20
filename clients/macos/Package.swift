// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "LocalVoiceAssistant",
    platforms: [.macOS(.v14)],
    targets: [.executableTarget(name: "LocalVoiceAssistant")]
)
