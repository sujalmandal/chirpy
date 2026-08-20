import AVFoundation
import Foundation

final class MicrophoneRecorder {
    private let engine = AVAudioEngine()
    private var recorded = Data()
    private var sampleRate = 16_000
    var onLevel: ((Float) -> Void)?

    func start() throws {
        recorded.removeAll(keepingCapacity: true)
        let input = engine.inputNode
        let sourceFormat = input.inputFormat(forBus: 0)
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false) else {
            throw NSError(domain: "Voice", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not create recording format"])
        }
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: sourceFormat) { [weak self] buffer, _ in
            guard let self, let converted = self.convert(buffer, to: format) else { return }
            let frames = Int(converted.frameLength)
            guard let channel = converted.floatChannelData?[0] else { return }
            let rms = sqrt((0..<frames).reduce(Float.zero) { $0 + channel[$1] * channel[$1] } / Float(max(frames, 1)))
            self.onLevel?(rms)
            for index in 0..<frames {
                var sample = Int16(max(-1, min(1, channel[index])) * Float(Int16.max)).littleEndian
                withUnsafeBytes(of: &sample) { self.recorded.append(contentsOf: $0) }
            }
        }
        engine.prepare()
        try engine.start()
    }

    func stop() throws -> Data {
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        guard !recorded.isEmpty else { throw NSError(domain: "Voice", code: 2, userInfo: [NSLocalizedDescriptionKey: "No audio captured"]) }
        return wavHeader(dataBytes: recorded.count) + recorded
    }

    func cancel() {
        if engine.isRunning { engine.stop() }
        engine.inputNode.removeTap(onBus: 0)
        recorded.removeAll()
    }

    private func convert(_ input: AVAudioPCMBuffer, to format: AVAudioFormat) -> AVAudioPCMBuffer? {
        guard let converter = AVAudioConverter(from: input.format, to: format),
              let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(Double(input.frameLength) * format.sampleRate / input.format.sampleRate + 1)) else { return nil }
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in status.pointee = .haveData; return input }
        return error == nil ? output : nil
    }

    private func wavHeader(dataBytes: Int) -> Data {
        var header = Data("RIFF".utf8)
        func append(_ value: UInt32) { var v = value.littleEndian; withUnsafeBytes(of: &v) { header.append(contentsOf: $0) } }
        func append16(_ value: UInt16) { var v = value.littleEndian; withUnsafeBytes(of: &v) { header.append(contentsOf: $0) } }
        append(UInt32(36 + dataBytes)); header += Data("WAVEfmt ".utf8); append(16); append16(1); append16(1)
        append(UInt32(sampleRate)); append(UInt32(sampleRate * 2)); append16(2); append16(16)
        header += Data("data".utf8); append(UInt32(dataBytes))
        return header
    }
}
