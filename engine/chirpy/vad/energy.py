"""Energy-based VAD provider (roomkit-style fallback).

Detects speech by RMS amplitude with no model dependency, mirroring roomkit's
``EnergyVADProvider``. Uses a configurable RMS threshold; the per-frame speech
probability is a soft mapping of RMS onto [0, 1] around the threshold.
"""

from __future__ import annotations

import numpy as np

from .base import VADConfig, _FrameDetector, _StreamingVAD


class _EnergyDetector(_FrameDetector):
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def process(self, mono_f32: np.ndarray) -> tuple[float, bool]:
        rms = float(np.sqrt(np.mean(mono_f32.astype(np.float64) ** 2))) if mono_f32.size else 0.0
        is_speech = rms >= self._threshold
        # Soft probability: 0.5 at threshold, ->1 as rms reaches ~2x threshold.
        prob = min(1.0, rms / (self._threshold * 2.0))
        return prob, is_speech


class EnergyVAD(_StreamingVAD):
    """RMS-threshold VAD. Suitable as a zero-dependency fallback."""

    def __init__(
        self,
        *,
        energy_threshold: float = 0.01,
        config: VADConfig | None = None,
    ) -> None:
        # energy_threshold is on the float32 [-1, 1] scale (roomkit uses int16
        # scale; 0.01 float ~ 328 int16, a quiet-room default).
        self._energy_threshold = energy_threshold
        super().__init__(
            config or VADConfig(),
            model="energy",
            provider="roomkit-style",
        )

    def _make_detector(self) -> _FrameDetector:
        return _EnergyDetector(self._energy_threshold)
