import SwiftUI

struct WaveformView: View {
    let micLevel: Float
    let speakerLevel: Float

    @State private var micHistory: [Float] = []
    @State private var speakerHistory: [Float] = []
    private let maxSamples = 200

    var body: some View {
        VStack(spacing: 6) {
            strip(label: "Mic", color: .green, history: micHistory, level: micLevel)
            strip(label: "Speaker", color: .cyan, history: speakerHistory, level: speakerLevel)
        }
        .onChange(of: micLevel) { _, new in append(new, to: &micHistory) }
        .onChange(of: speakerLevel) { _, new in append(new, to: &speakerHistory) }
    }

    private func append(_ value: Float, to history: inout [Float]) {
        history.append(value)
        if history.count > maxSamples { history.removeFirst(history.count - maxSamples) }
    }

    private func strip(label: String, color: Color, history: [Float], level: Float) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 52, alignment: .leading)
            Canvas { context, size in
                let midY = size.height / 2
                let amp = size.height / 2 - 2
                // Baseline
                var baseline = Path()
                baseline.move(to: CGPoint(x: 0, y: midY))
                baseline.addLine(to: CGPoint(x: size.width, y: midY))
                context.stroke(baseline, with: .color(.secondary.opacity(0.3)), lineWidth: 1)

                guard history.count > 1 else { return }
                let step = size.width / CGFloat(maxSamples - 1)
                var path = Path()
                for (i, v) in history.enumerated() {
                    let x = size.width - CGFloat(history.count - 1 - i) * step
                    let y = midY - CGFloat(min(max(v, 0), 1)) * amp
                    if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
                    else { path.addLine(to: CGPoint(x: x, y: y)) }
                }
                // Filled area under the line
                var fill = path
                fill.addLine(to: CGPoint(x: size.width, y: midY))
                fill.addLine(to: CGPoint(x: size.width - CGFloat(history.count - 1) * step, y: midY))
                fill.closeSubpath()
                context.fill(fill, with: .color(color.opacity(0.15)))
                context.stroke(path, with: .color(color), lineWidth: 1.5)

                // Live dot at the right edge
                let dotY = midY - CGFloat(min(max(level, 0), 1)) * amp
                let dot = CGRect(x: size.width - 4, y: dotY - 2, width: 4, height: 4)
                context.fill(Path(ellipseIn: dot), with: .color(color))
            }
            .frame(height: 34)
        }
    }
}
