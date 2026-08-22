"""Server-side acoustic echo cancellation (AEC) for Chirpy.

The client plays the agent's TTS through the Web Audio graph, which is smooth
(no stutter) but is *not* included in WebKit's browser AEC reference. To stop
the agent from hearing its own voice and triggering bogus barge-in, the engine
cancels the echo from the microphone using its own TTS output as the reference.

WebRTC's ``AudioProcessingModule`` requires 10 ms frames, so we split/combine
the 50 ms audio frames to and from the module. The TTS plugin feeds the reverse
(reference) stream via :func:`feed_reference`.
"""

from __future__ import annotations

import numpy as np

from livekit import rtc

# WebRTC APM works on exactly 10 ms frames.
_AEC_FRAME_MS = 10


def _samples_per_frame(sample_rate: int) -> int:
    return sample_rate * _AEC_FRAME_MS // 1000


def _to_f32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _to_pcm16(f32: np.ndarray) -> np.ndarray:
    return np.clip(f32, -1, 1) * 32767


class AECFilter(rtc.FrameProcessor[rtc.AudioFrame]):
    """Mic FrameProcessor that applies WebRTC echo cancellation in place."""

    def __init__(self, apm: rtc.AudioProcessingModule):
        self._apm = apm
        self._enabled = True
        self._carry: np.ndarray | None = None  # <10 ms leftover from last frame

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled:
            return frame
        ch = frame.num_channels
        spf = _samples_per_frame(frame.sample_rate)
        pcm = _to_f32(frame.data)
        if self._carry is not None:
            pcm = np.concatenate([self._carry, pcm])
        n_blocks = pcm.size // (spf * ch)
        if n_blocks == 0:
            self._carry = pcm
            return frame
        keep = n_blocks * spf * ch
        self._carry = pcm[keep:] if pcm.size > keep else None
        blocks = pcm[:keep].reshape(n_blocks, spf, ch)
        out = np.zeros(0, dtype=np.int16)
        for block in blocks:
            mono = block.reshape(-1) if ch == 1 else block
            sub = rtc.AudioFrame(
                data=_to_pcm16(mono).astype(np.int16).tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=1 if ch == 1 else ch,
                samples_per_channel=spf,
            )
            self._apm.process_stream(sub)  # modifies sub.data in place
            out = np.concatenate([out, np.frombuffer(sub.data, dtype=np.int16)])
        return rtc.AudioFrame(
            data=out.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1 if ch == 1 else ch,
            samples_per_channel=out.size // (1 if ch == 1 else ch),
        )

    def _close(self) -> None:
        pass


def feed_reference(apm: rtc.AudioProcessingModule, pcm: np.ndarray, sample_rate: int) -> None:
    """Feed the agent's TTS audio to the AEC as the reverse (reference) stream."""
    pcm = np.asarray(pcm, dtype=np.float32)
    spf = _samples_per_frame(sample_rate)
    n = (pcm.size // spf) * spf
    for block in pcm[:n].reshape(-1, spf):
        sub = rtc.AudioFrame(
            data=_to_pcm16(block).astype(np.int16).tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=spf,
        )
        apm.process_reverse_stream(sub)
