import Foundation

@MainActor
final class SystemMetrics: ObservableObject {
    @Published private(set) var cpu = "—"
    @Published private(set) var gpu = "—"
    @Published private(set) var ram = "—"
    private var task: Task<Void, Never>?

    func start(engine: KyutaiEngine) {
        guard task == nil else { return }
        task = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let values = await Self.read()
                self.cpu = values.cpu; self.gpu = values.gpu; self.ram = values.ram
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }
    func stop() { task?.cancel(); task = nil }

    private static func read() async -> (cpu: String, gpu: String, ram: String) {
        let script = #"""
        cores=$(sysctl -n hw.ncpu)
        cpu=$(ps -A -o %cpu= | awk -v n="$cores" '{s+=$1} END {printf "%.0f", s/n}')
        gpu=$(ioreg -l -w0 -r -c IOAccelerator 2>/dev/null | sed -n 's/.*"Device Utilization %"=\([0-9]*\).*/\1/p' | head -1)
        pagesize=$(vm_stat | sed -n 's/.*page size of \([0-9]*\).*/\1/p')
        used=$(vm_stat | awk -F: '/Pages active|Pages inactive|Pages wired down|Pages occupied by compressor/ {gsub("[^0-9]", "", $2); s+=$2} END {print s}')
        total=$(sysctl -n hw.memsize)
        awk -v c="$cpu" -v g="$gpu" -v u="$used" -v p="$pagesize" -v t="$total" 'BEGIN {printf "CPU %s%%|GPU %s%%|RAM %.1f / %.0f GB", c, (g==""?"—":g), u*p/1073741824, t/1073741824}'
        """#
        let output = await shell(script)
        let parts = output.split(separator: "|").map(String.init)
        return (parts.indices.contains(0) ? parts[0].replacingOccurrences(of: "CPU ", with: "") : "—", parts.indices.contains(1) ? parts[1].replacingOccurrences(of: "GPU ", with: "") : "—", parts.indices.contains(2) ? parts[2].replacingOccurrences(of: "RAM ", with: "") : "—")
    }

    private static func shell(_ command: String) async -> String {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                let process = Process(); let pipe = Pipe()
                process.executableURL = URL(fileURLWithPath: "/bin/zsh")
                process.arguments = ["-lc", command]; process.standardOutput = pipe
                try? process.run(); process.waitUntilExit()
                continuation.resume(returning: String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "")
            }
        }
    }
}
