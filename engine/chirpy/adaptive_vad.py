"""Reference implementations for adaptive voice-activity detection (VAD).

This is the *research reference* backing ``docs/adaptive-vad.md``. It contains
the core adaptive algorithms as pure, unit-testable functions (no LiveKit
dependency) so the ideas can be validated before wiring them into
``AgentSession`` as a custom ``agents.vad.VAD``.

Three pieces:

1. ``EnergyVAD`` — the canonical adaptive noise-floor + hangover design
   (adapted from AssemblyAI's EnergyVad, itself a WebRTC-style energy VAD).
2. ``SileroThresholdResolver`` — adapt a Silero-style speech-probability
   threshold from the observed non-speech probability distribution (hysteresis
   activation/deactivation pair).
3. ``noise_floor_percentile`` — a small adaptive floor estimate used by both.

All functions are deterministic and tested with synthesized silence/noise/tone.
"""

from __future__ import annotations

import math
from collections import deque

# --------------------------------------------------------------------------- #
# 1. Adaptive energy / noise-floor VAD (canonical)
# --------------------------------------------------------------------------- #
class EnergyVADCore:
    """Frame-level adaptive energy VAD: RMS > noiseFloor * ratio, with hangover.

    noiseFloor is updated (EMA) *only while not speaking*, so a sustained loud
    voice cannot slowly raise the floor and then be misclassified as noise.

    Args:
        threshold_ratio: threshold = noise_floor * threshold_ratio. ~3 ≈ +9.5 dB
            above the floor. < 2 is too sensitive, > 6 misses quiet onsets.
        noise_floor_alpha: EMA weight on non-speech frames. > 0.1 tracks fast
            (good for non-stationary noise) but risks drifting up on quiet speech.
        hangover_frames: keep "active" this many frames after the last speech frame.
        initial_noise_floor: seed estimate; adapts after the first non-speech frame.
    """

    def __init__(
        self,
        *,
        samples_per_frame: int = 512,
        threshold_ratio: float = 3.0,
        noise_floor_alpha: float = 0.05,
        hangover_frames: int = 10,
        initial_noise_floor: float = 1e-4,
    ) -> None:
        self.samples_per_frame = samples_per_frame
        self.threshold_ratio = threshold_ratio
        self.noise_floor_alpha = noise_floor_alpha
        self.hangover_frames = hangover_frames
        self._initial_noise_floor = initial_noise_floor
        self.noise_floor = initial_noise_floor
        self._hangover_remaining = 0

    def _frame_rms(self, samples) -> float:
        if len(samples) == 0:
            return 0.0
        return math.sqrt(sum(x * x for x in samples) / len(samples))

    def classify(self, samples) -> tuple[bool, float]:
        """Process one frame; returns (active, rms_energy)."""
        rms = self._frame_rms(samples)
        threshold = self.noise_floor * self.threshold_ratio
        active = rms > threshold
        if active:
            self._hangover_remaining = self.hangover_frames
        elif self._hangover_remaining > 0:
            self._hangover_remaining -= 1
            active = True  # still in hangover
        else:
            # Non-speech: adapt the floor on this frame's energy.
            self.noise_floor = (
                self.noise_floor * (1 - self.noise_floor_alpha)
                + rms * self.noise_floor_alpha
            )
        return active, rms

    def reset(self) -> None:
        self.noise_floor = self._initial_noise_floor
        self._hangover_remaining = 0


# --------------------------------------------------------------------------- #
# 2. Adaptive threshold on a model speech probability (Silero-style)
# --------------------------------------------------------------------------- #
class NoiseFloorTracker:
    """Adaptive estimate of the silence level via EMA/percentile on samples.

    Feed it only non-speech samples (probabilities or energies) so a sustained
    loud voice can't drag the floor up.
    """

    def __init__(self, *, alpha: float = 0.05, buffer: int = 128, initial: float = 1e-4):
        self.alpha = alpha
        self.buffer = buffer
        self.value = initial
        self._history: deque[float] = deque(maxlen=buffer)

    def update(self, sample: float) -> float:
        """Adapt the floor from a non-speech sample; returns the new floor."""
        self._history.append(sample)
        self.value = self.value * (1 - self.alpha) + sample * self.alpha
        return self.value

    def percentile(self, pct: float = 0.5) -> float:
        if not self._history:
            return self.value
        ordered = sorted(self._history)
        idx = min(len(ordered) - 1, max(0, int(pct * len(ordered))))
        return ordered[idx]


class SileroThresholdResolver:
    """Adapt a speech-probability threshold from the silence distribution.

    Model-based flavour: maintain the noise floor on *silence* probabilities and
    set ``activation_threshold = floor * margin`` (clamped), with a hysteresis
    ``deactivation_threshold = activation_threshold - gap`` to avoid jitter.

    Args:
        margin: how far above the silence floor speech must sit (analogous to
            threshold_ratio; ~2-3).
        hysteresis_gap: gap between activation and deactivation thresholds.
        min_threshold / max_threshold: clamp band for the activation threshold.
        floor_alpha: EMA weight for the silence floor (<= 0.1 recommended).
        debounce_step: min change before the threshold is moved (anti-flutter).
    """

    def __init__(
        self,
        *,
        margin: float = 2.0,
        hysteresis_gap: float = 0.15,
        min_threshold: float = 0.25,
        max_threshold: float = 0.6,
        floor_alpha: float = 0.05,
        debounce_step: float = 0.01,
    ) -> None:
        self.margin = margin
        self.hysteresis_gap = hysteresis_gap
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.debounce_step = debounce_step
        self._floor = NoiseFloorTracker(alpha=floor_alpha)
        self.activation_threshold = 0.5
        self.deactivation_threshold = 0.35

    def _resolve(self) -> float:
        # Use the percentile of the observed silence distribution as the floor
        # estimate (more robust than a single EMA: it reflects the actual
        # ambient level rather than decaying from a tiny seed), then place the
        # threshold at `floor * margin` within the clamp band.
        floor = self._floor.percentile(0.5) or self._floor.value
        target = max(self.min_threshold, min(self.max_threshold, floor * self.margin))
        return target

    def classify(self, probability: float) -> bool:
        """Return True if this frame is speech, adapting the threshold on silence."""
        is_speech = probability >= self.activation_threshold
        if is_speech:
            # Don't let speech pollute the silence floor.
            return True
        # Non-speech frame: adapt the floor and maybe move the threshold.
        self._floor.update(probability)
        target = self._resolve()
        if abs(target - self.activation_threshold) >= self.debounce_step:
            self.activation_threshold = target
            self.deactivation_threshold = max(
                self.min_threshold, target - self.hysteresis_gap
            )
        return False
