"""Turn endpointing primitives with no ML/runtime dependencies.

The Kyutai semantic head predicts a conversational pause, while microphone
energy is only a fallback. Keeping this state machine separate makes the edge
cases reproducible without loading several gigabytes of model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class EndpointState(str, Enum):
    WARMING_UP = "warming_up"
    WAITING = "waiting"
    SPEAKING = "user_speaking"


@dataclass(frozen=True)
class EndpointDecision:
    reason: str
    speech_ms: int
    silence_ms: int
    semantic_probability: float
    smoothed_probability: float
    energy_threshold: float


class EndpointDetector:
    """Fuse semantic pause prediction, recognized text, and adaptive energy."""

    block_ms = 80

    def __init__(
        self,
        *,
        base_energy_threshold: float = 0.01,
        min_speech_ms: int = 320,
        min_silence_ms: int = 320,
        semantic_end_threshold: float = 0.6,
        semantic_speech_threshold: float = 0.4,
        warmup_blocks: int = 12,
        semantic_hold_blocks: int = 2,
        noise_multiplier: float = 3.0,
        noise_alpha: float = 0.04,
        semantic_time_constant: float = 0.01,
    ):
        self.base_energy_threshold = max(0.0001, base_energy_threshold)
        self.min_speech_blocks = max(1, round(min_speech_ms / self.block_ms))
        self.min_silence_blocks = max(1, round(min_silence_ms / self.block_ms))
        self.semantic_end_threshold = min(1.0, max(0.0, semantic_end_threshold))
        self.semantic_speech_threshold = min(
            self.semantic_end_threshold, max(0.0, semantic_speech_threshold)
        )
        self.warmup_blocks = max(0, warmup_blocks)
        self.semantic_hold_blocks = max(1, semantic_hold_blocks)
        self.noise_multiplier = max(1.0, noise_multiplier)
        self.noise_alpha = min(1.0, max(0.001, noise_alpha))
        self.semantic_alpha = 1.0 - math.exp(
            -(self.block_ms / 1000.0) / max(0.001, semantic_time_constant)
        )
        self.reset()

    def reset(self) -> None:
        self.state = EndpointState.WARMING_UP if self.warmup_blocks else EndpointState.WAITING
        self.blocks_seen = 0
        self.noise_floor = self.base_energy_threshold / self.noise_multiplier
        self.smoothed_probability = 0.0
        self.pause_run = 0
        self.speech_blocks = 0
        self.silence_run = 0
        self.energy_run = 0
        self.candidate_speech_peak = 0
        self.candidate_silence_run = 0

    @property
    def energy_threshold(self) -> float:
        return max(self.base_energy_threshold, self.noise_floor * self.noise_multiplier)

    @property
    def armed(self) -> bool:
        return self.state == EndpointState.SPEAKING

    def observe(
        self, *, rms: float, semantic_probability: float, has_recognized_text: bool
    ) -> EndpointDecision | None:
        rms = max(0.0, rms)
        semantic_probability = min(1.0, max(0.0, semantic_probability))
        self.blocks_seen += 1
        self.smoothed_probability += self.semantic_alpha * (
            semantic_probability - self.smoothed_probability
        )

        if not self.armed:
            # Follow a sustained room-noise increase without letting a single
            # click or the first syllable of speech poison the estimate.
            capped_rms = min(rms, max(self.base_energy_threshold, self.noise_floor * 1.5))
            self.noise_floor += self.noise_alpha * (capped_rms - self.noise_floor)

        if self.blocks_seen <= self.warmup_blocks:
            self.state = EndpointState.WARMING_UP
        elif not self.armed:
            self.state = EndpointState.WAITING

        above_energy = rms >= self.energy_threshold
        if not self.armed:
            if above_energy:
                self.energy_run += 1
                self.candidate_speech_peak = max(self.candidate_speech_peak, self.energy_run)
                self.candidate_silence_run = 0
            else:
                self.energy_run = 0
                self.candidate_silence_run += 1
                # Preserve recent speech across Kyutai's ~0.5 s text delay,
                # but expire old noise bursts before they can arm a turn.
                if self.candidate_silence_run > 8:
                    self.candidate_speech_peak = 0

        # A real decoder token is the strongest protection against fan noise,
        # keyboard clicks, and separated noise bursts starting a turn.
        just_armed = False
        if has_recognized_text:
            if not self.armed:
                self.state = EndpointState.SPEAKING
                self.pause_run = 0
                self.smoothed_probability = 0.0
                self.speech_blocks = self.candidate_speech_peak
                self.silence_run = self.candidate_silence_run
                just_armed = True

        if self.armed and not just_armed:
            if above_energy:
                self.energy_run += 1
                self.silence_run = 0
                self.speech_blocks += 1
            else:
                # This reset fixes the old cumulative-noise-burst behavior.
                self.energy_run = 0
                self.silence_run += 1

        if not self.armed or self.blocks_seen <= self.warmup_blocks:
            self.pause_run = 0
            return None

        if self.smoothed_probability >= self.semantic_end_threshold:
            self.pause_run += 1
        elif self.smoothed_probability <= self.semantic_speech_threshold:
            self.pause_run = 0

        semantic_end = self.pause_run >= self.semantic_hold_blocks and self.silence_run >= 1
        energy_end = (
            self.speech_blocks >= self.min_speech_blocks
            and self.silence_run >= self.min_silence_blocks
        )
        if not semantic_end and not energy_end:
            return None

        return EndpointDecision(
            reason="semantic_vad" if semantic_end else "adaptive_silence_timeout",
            speech_ms=self.speech_blocks * self.block_ms,
            silence_ms=self.silence_run * self.block_ms,
            semantic_probability=semantic_probability,
            smoothed_probability=self.smoothed_probability,
            energy_threshold=self.energy_threshold,
        )
