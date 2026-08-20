import AVFoundation

final class AudioQueue: NSObject, AVAudioPlayerDelegate {
    private var players: [AVAudioPlayer] = []

    func enqueue(_ wav: Data) {
        DispatchQueue.main.async {
            guard let player = try? AVAudioPlayer(data: wav) else { return }
            player.delegate = self
            self.players.append(player)
            if self.players.count == 1 { player.play() }
        }
    }

    func stop() {
        DispatchQueue.main.async {
            self.players.forEach { $0.stop() }
            self.players.removeAll()
        }
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async {
            self.players.removeAll { $0 === player }
            self.players.first?.play()
        }
    }
}
