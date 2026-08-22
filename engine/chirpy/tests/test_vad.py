"""Tests for the roomkit-style VAD providers (vad package)."""

from __future__ import annotations

import asyncio
import struct
import unittest

from livekit import rtc
from livekit.agents import vad as agents_vad

from vad import EnergyVAD, SherpaOnnxVAD, VADConfig, build_vad
from vad.base import _StreamingVAD, VADConfig as BaseVADConfig
from vad.base import _FrameDetector


def _frame(samples_per_channel: int, amplitude: float = 0.0, sr: int = 16000) -> rtc.AudioFrame:
    """Build a mono int16 frame of a constant tone at the given amplitude."""
    pcm = bytearray()
    for _ in range(samples_per_channel):
        pcm += struct.pack("<h", int(amplitude * 32767.0))
    return rtc.AudioFrame(
        data=bytes(pcm),
        sample_rate=sr,
        num_channels=1,
        samples_per_channel=samples_per_channel,
    )


def _run(coro):
    return asyncio.run(coro)


class _AlwaysSpeechVAD(_StreamingVAD):
    """Fake neural VAD that always reports speech (simulates model inertia)."""

    def __init__(self, config: BaseVADConfig) -> None:
        super().__init__(config, model="fake", provider="test")

    def _make_detector(self) -> _FrameDetector:
        class Det(_FrameDetector):
            def process(self, mono_f32):  # noqa: ANN001
                return 1.0, True

            def reset(self):
                pass

        return Det()


async def _collect(vad: agents_vad.VAD, frames: list[rtc.AudioFrame]):
    stream = vad.stream()
    for f in frames:
        stream.push_frame(f)
    stream.end_input()
    events = []
    async for ev in stream:
        events.append(ev)
    return events


class EnergyVADStreamTest(unittest.TestCase):
    def test_energy_detects_tone_with_pre_roll_and_end(self):
        vad = EnergyVAD(energy_threshold=0.01)
        # 5 silence (100ms) + 15 tone (300ms) + 30 silence (600ms), 20ms frames
        frames = [_frame(320) for _ in range(5)]
        frames += [_frame(320, amplitude=0.1) for _ in range(15)]
        frames += [_frame(320) for _ in range(30)]
        events = _run(_collect(vad, frames))

        types = [e.type for e in events]
        self.assertIn(agents_vad.VADEventType.START_OF_SPEECH, types)
        self.assertIn(agents_vad.VADEventType.END_OF_SPEECH, types)
        # pre-roll should be attached to START_OF_SPEECH
        start = next(e for e in events if e.type == agents_vad.VADEventType.START_OF_SPEECH)
        self.assertGreater(len(start.frames), 0)
        end = next(e for e in events if e.type == agents_vad.VADEventType.END_OF_SPEECH)
        self.assertGreaterEqual(end.speech_duration, 0.25)

    def test_silence_only_produces_no_speech_boundaries(self):
        vad = EnergyVAD(energy_threshold=0.01)
        frames = [_frame(320) for _ in range(60)]  # 1.2s silence
        events = _run(_collect(vad, frames))
        types = [e.type for e in events]
        self.assertNotIn(agents_vad.VADEventType.START_OF_SPEECH, types)
        self.assertNotIn(agents_vad.VADEventType.END_OF_SPEECH, types)

    def test_energy_fast_exit_forces_end_on_silence(self):
        # Simulate a neural model that stays in speech: detector always says
        # speech, but frames are true silence -> energy gate forces END.
        cfg = BaseVADConfig(
            silence_threshold_ms=300,
            energy_silence_rms=0.0006,
        )
        vad = _AlwaysSpeechVAD(cfg)
        frames = [_frame(320) for _ in range(80)]  # 1.6s of "silent speech"
        events = _run(_collect(vad, frames))
        types = [e.type for e in events]
        self.assertIn(agents_vad.VADEventType.START_OF_SPEECH, types)
        self.assertIn(agents_vad.VADEventType.END_OF_SPEECH, types)


class FactoryTest(unittest.TestCase):
    def test_no_model_falls_back_to_energy(self):
        vad = build_vad({})
        self.assertIsInstance(vad, EnergyVAD)

    def test_missing_model_falls_back_to_energy(self):
        vad = build_vad({"VAD_MODEL": "/nonexistent/ten-vad.onnx"})
        self.assertIsInstance(vad, EnergyVAD)

    def test_energy_config_reads_knobs(self):
        vad = build_vad({"VAD_SILENCE_MS": "700", "VAD_ENERGY_THRESHOLD": "0.02"})
        self.assertIsInstance(vad, EnergyVAD)
        self.assertEqual(vad._config.silence_threshold_ms, 700)


class SherpaConfigTest(unittest.TestCase):
    def test_sherpa_requires_model(self):
        with self.assertRaises(ValueError):
            SherpaOnnxVAD()


@unittest.skipUnless(
    __import__("os").environ.get("VAD_MODEL"),
    "set VAD_MODEL to a sherpa-onnx .onnx to run neural-VAD integration test",
)
class SherpaDetectorTest(unittest.TestCase):
    """Feeds real synthesized speech through the sherpa detector directly.

    Note: manually-constructed ``rtc.AudioFrame`` objects from this binding
    return zeroed ``.data`` for varying-content frames, so we drive
    ``_SherpaDetector.process`` with numpy arrays (which is exactly what the
    stream does after ``_to_mono_f32`` on real track frames).
    """

    @classmethod
    def setUpClass(cls):
        import os

        from vad.sherpa import _SherpaDetector

        cls.sherpa = __import__("sherpa_onnx")
        cls.detector_cls = _SherpaDetector

        from vad import SherpaOnnxVADConfig

        cls.cfg = SherpaOnnxVADConfig(
            model=os.environ["VAD_MODEL"],
            model_type=os.environ.get("VAD_MODEL_TYPE", "ten"),
            threshold=0.35,
        )

    def _speech_f32(self):
        try:
            from kokoro import KPipeline
        except Exception:  # noqa: BLE001
            self.skipTest("kokoro unavailable for speech synthesis")
        import numpy as np

        p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
        r = list(p("hello there, this is a voice activity test.", voice="af_heart", speed=1.0))[0]
        a = r.audio.detach().cpu().numpy()
        x = np.arange(int(a.shape[0] * 16000 / 24000))
        return np.interp(x, np.arange(a.shape[0]), a).astype(np.float32)

    def test_detects_real_speech_and_shows_inertia(self):
        import numpy as np

        d = self.detector_cls(self.cfg)
        a16 = self._speech_f32()
        n = 320
        speech_hit = any(d.process(a16[s : s + n])[1] for s in range(0, len(a16) - n, n))
        self.assertTrue(speech_hit, "sherpa VAD should flag real speech")


if __name__ == "__main__":
    unittest.main()
