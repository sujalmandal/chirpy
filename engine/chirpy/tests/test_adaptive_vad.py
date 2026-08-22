"""Tests for the adaptive VAD reference implementations in :mod:`adaptive_vad`.

Synthesized silence / noise / tone fixtures only — no LiveKit, no models.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[1]  # engine/chirpy
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from adaptive_vad import EnergyVADCore, SileroThresholdResolver, NoiseFloorTracker  # noqa: E402


def _tone(amplitude: float, n: int) -> list[float]:
    return [amplitude] * n


def _silence(n: int) -> list[float]:
    return [0.0] * n


class EnergyVADTests(unittest.TestCase):
    def test_silence_is_not_speech_and_floor_stays_low(self):
        v = EnergyVADCore(samples_per_frame=64)
        active, rms = v.classify(_silence(64))
        self.assertFalse(active)
        self.assertAlmostEqual(rms, 0.0)
        self.assertLess(v.noise_floor, 1e-3)

    def test_loud_tone_is_speech(self):
        v = EnergyVADCore(samples_per_frame=64)
        active, _ = v.classify(_tone(0.3, 64))
        self.assertTrue(active)

    def test_hangover_keeps_active_after_tone(self):
        v = EnergyVADCore(samples_per_frame=64, hangover_frames=5)
        v.classify(_tone(0.3, 64))  # speech
        active_after = [v.classify(_silence(64))[0] for _ in range(6)]
        # hangover 5 -> the first 5 silence frames remain active, the 6th drops.
        self.assertTrue(all(active_after[:5]))
        self.assertFalse(active_after[5])

    def test_noise_floor_raises_threshold(self):
        # A 0.08 tone is speech when the room is quiet (floor 0.02, thresh 0.06)...
        quiet = EnergyVADCore(
            samples_per_frame=64, threshold_ratio=3.0, initial_noise_floor=0.02
        )
        self.assertTrue(quiet.classify(_tone(0.08, 64))[0])

        # ...but after moderate noise (0.04, below the threshold, so non-speech)
        # raises the adaptive floor, the SAME tone is rejected as non-speech.
        noisy = EnergyVADCore(
            samples_per_frame=64,
            threshold_ratio=3.0,
            noise_floor_alpha=0.5,
            initial_noise_floor=0.02,
        )
        for _ in range(60):
            noisy.classify(_tone(0.04, 64))
        self.assertGreater(noisy.noise_floor, 0.03)
        self.assertFalse(noisy.classify(_tone(0.08, 64))[0])

    def test_speech_frame_does_not_pollute_noise_floor(self):
        v = EnergyVADCore(samples_per_frame=64, initial_noise_floor=0.02)
        before = v.noise_floor
        v.classify(_tone(0.5, 64))  # clearly speech
        self.assertAlmostEqual(v.noise_floor, before)  # floor unchanged


class NoiseFloorTrackerTests(unittest.TestCase):
    def test_ema_tracks_toward_silence(self):
        t = NoiseFloorTracker(alpha=0.2, initial=0.5)
        for _ in range(200):
            t.update(0.01)
        self.assertLess(t.value, 0.05)

    def test_percentile_returns_sorted_quantile(self):
        t = NoiseFloorTracker()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            t.update(v)
        self.assertAlmostEqual(t.percentile(0.0), 0.1)
        self.assertAlmostEqual(t.percentile(1.0), 0.5)


class SileroThresholdResolverTests(unittest.TestCase):
    def test_clean_speech_classified(self):
        r = SileroThresholdResolver()
        self.assertFalse(r.classify(0.05))  # silence seed
        self.assertTrue(r.classify(0.9))    # confident speech

    def test_threshold_rises_in_noisy_environment(self):
        r = SileroThresholdResolver(margin=2.0, min_threshold=0.25, max_threshold=0.6)
        # Environment with elevated noise: the model lingers at moderate
        # probabilities (0.45, below the initial 0.5 threshold), treated as
        # non-speech, raising the adaptive floor and thus the threshold.
        for _ in range(200):
            r.classify(0.45)
        self.assertAlmostEqual(r.activation_threshold, 0.6)  # clamped high
        # A mid-level 0.55 is now below threshold -> not speech.
        self.assertFalse(r.classify(0.55))

    def test_threshold_drops_to_floor_in_quiet_room(self):
        r = SileroThresholdResolver(margin=2.0, min_threshold=0.25, max_threshold=0.6)
        for _ in range(200):
            r.classify(0.05)  # very quiet
        self.assertAlmostEqual(r.activation_threshold, 0.25)  # clamped low
        # Soft but real speech clears it.
        self.assertTrue(r.classify(0.6))

    def test_hysteresis_pair_valid(self):
        r = SileroThresholdResolver()
        for _ in range(50):
            r.classify(0.05)
        self.assertLessEqual(r.deactivation_threshold, r.activation_threshold)
        self.assertGreaterEqual(r.deactivation_threshold, r.min_threshold)


if __name__ == "__main__":
    unittest.main()
