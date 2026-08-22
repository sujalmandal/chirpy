"""Shared base for Chirpy's roomkit-style VAD providers.

Mirrors roomkit's clean ``VADProvider`` abstraction but adapted to LiveKit
Agents: each provider is a drop-in ``agents.vad.VAD`` whose ``stream()`` emits
LiveKit ``VADEvent``s. The per-frame speech decision is delegated to a small
``_FrameDetector`` (energy or sherpa-onnx), and a shared state machine produces
START_OF_SPEECH / END_OF_SPEECH / INFERENCE_DONE events with pre-roll and
silence debounce — the same shape roomkit's ``energy.py`` / ``sherpa_onnx.py``
implement, minus the external dependency on roomkit itself.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from livekit import rtc
from livekit.agents import vad as agents_vad


@dataclass
class VADConfig:
    """Configuration shared by all VAD providers (mirrors roomkit's VADConfig)."""

    silence_threshold_ms: int = 500
    """Consecutive silence (ms) before SPEECH_END."""
    speech_pad_ms: int = 300
    """Pre-roll padding (ms) kept before detected speech so onsets aren't clipped."""
    min_speech_duration_ms: int = 250
    """Minimum speech duration (ms) to emit SPEECH_END; shorter segments dropped."""
    max_speech_duration_ms: int = 60_000
    """Safety cap: force SPEECH_END after this many ms of speech."""
    energy_silence_rms: float = 0.0
    """Energy-based fast exit (float32 scale). When the frame RMS drops below
    this for ``silence_threshold_ms``, force SPEECH_END even if the model still
    reports speech (fixes neural VAD inertia on silence). 0 disables. Roomkit's
    int16 default 20.0 ≈ 0.0006 on the float32 scale."""
    sample_rate: int = 16000
    """Sample rate the detector operates at (frames are resampled to this)."""
    update_interval: float = 0.1
    """LiveKit INFERENCE_DONE interval (used for metrics aggregation)."""


class _FrameDetector(ABC):
    """One stream's frame-level speech classifier."""

    @abstractmethod
    def process(self, mono_f32: np.ndarray) -> tuple[float, bool]:
        """Return (probability, is_speech) for a mono float32 frame in [-1, 1]."""
        ...

    def reset(self) -> None:
        """Drop accumulated state (hard segment boundary)."""

    def close(self) -> None:
        """Release resources."""


def _to_mono_f32(frame: rtc.AudioFrame) -> np.ndarray:
    """Convert an AudioFrame to mono float32 in [-1, 1] (mean of channels)."""
    pcm = np.frombuffer(frame.data, dtype="<i2").astype(np.float32) / 32768.0
    if frame.num_channels > 1:
        pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1)
    return pcm


def _resample_f32(x: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or x.size == 0:
        return x
    n = round(x.size * to_rate / from_rate)
    src = np.arange(n, dtype=np.float32) * (from_rate / to_rate)
    return np.interp(src, np.arange(x.size, dtype=np.float32), x).astype(np.float32)


class _VADStream(agents_vad.VADStream):
    """Shared streaming state machine for all providers."""

    def __init__(self, vad: "_StreamingVAD") -> None:
        super().__init__(vad)
        self._vad = vad
        self._cfg = vad._config
        self._detector = vad._make_detector()
        self._sr = self._cfg.sample_rate

        self._speaking = False
        self._speech_dur = 0.0
        self._silence_dur = 0.0
        self._energy_silence_dur = 0.0
        self._samples = 0
        self._pre_roll: list[rtc.AudioFrame] = []
        self._pre_roll_dur = 0.0
        self._speech_frames: list[rtc.AudioFrame] = []
        self._start_samples = 0

    def _frame_dur(self, frame: rtc.AudioFrame) -> float:
        return frame.samples_per_channel / frame.sample_rate

    # -- state helpers -------------------------------------------------------
    def _push_pre_roll(self, frame: rtc.AudioFrame) -> None:
        self._pre_roll.append(frame)
        self._pre_roll_dur += self._frame_dur(frame)
        while self._pre_roll_dur > self._cfg.speech_pad_ms / 1000 and len(self._pre_roll) > 1:
            self._pre_roll_dur -= self._frame_dur(self._pre_roll.pop(0))

    def _start_speech(self, frame: rtc.AudioFrame, prob: float) -> None:
        self._speaking = True
        self._speech_dur = self._pre_roll_dur + self._frame_dur(frame)
        self._speech_frames = list(self._pre_roll) + [frame]
        self._start_samples = self._samples
        self._pre_roll = []
        self._pre_roll_dur = 0.0
        self._silence_dur = 0.0

    def _end_speech(self) -> None:
        self._speaking = False
        self._speech_dur = 0.0
        self._silence_dur = 0.0
        self._energy_silence_dur = 0.0
        self._speech_frames = []

    async def _main_task(self) -> None:
        while True:
            item = await self._input_ch.__anext__()
            if isinstance(item, self._FlushSentinel):
                self._end_speech()
                self._detector.reset()
                continue
            await self._process_frame(item)

    async def _process_frame(self, frame: rtc.AudioFrame) -> None:
        mono = _to_mono_f32(frame)
        mono = _resample_f32(mono, frame.sample_rate, self._sr)
        prob, is_speech = self._detector.process(mono)
        self._samples += mono.size

        if not self._speaking:
            self._push_pre_roll(frame)
            if is_speech:
                self._start_speech(frame, prob)
                self._emit(
                    agents_vad.VADEventType.START_OF_SPEECH,
                    prob, frames=self._speech_frames,
                )
        else:
            dur = self._frame_dur(frame)
            self._speech_frames.append(frame)
            self._speech_dur += dur
            if is_speech:
                self._silence_dur = 0.0
            else:
                self._silence_dur += dur

            # Energy-based fast exit (roomkit anti-inertia): if the model stays
            # in speech while the audio is clearly silent, force SPEECH_END.
            energy_gate = self._cfg.energy_silence_rms
            if energy_gate > 0:
                rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
                if rms < energy_gate:
                    self._energy_silence_dur += dur
                else:
                    self._energy_silence_dur = 0.0

            force = self._speech_dur >= self._cfg.max_speech_duration_ms / 1000
            energy_end = self._energy_silence_dur >= self._cfg.silence_threshold_ms / 1000
            if self._silence_dur >= self._cfg.silence_threshold_ms / 1000 or force or energy_end:
                if force or energy_end:
                    # Reset the model's stuck internal state so it can't
                    # immediately re-trigger a false SPEECH_START.
                    self._detector.reset()
                frames = self._speech_frames
                dur_ms = self._speech_dur * 1000
                if dur_ms >= self._cfg.min_speech_duration_ms:
                    self._emit(
                        agents_vad.VADEventType.END_OF_SPEECH,
                        prob, frames=frames,
                        speech_duration=self._speech_dur,
                    )
                self._end_speech()

        # Always emit an inference result for turn detection.
        self._emit(
            agents_vad.VADEventType.INFERENCE_DONE,
            prob,
            frames=[frame],
            speech_duration=self._speech_dur,
            silence_duration=self._silence_dur,
            speaking=self._speaking,
        )

    def _emit(
        self,
        etype,
        prob: float,
        *,
        frames=None,
        speech_duration=0.0,
        silence_duration=0.0,
        speaking=False,
    ) -> None:
        self._event_ch.send_nowait(
            agents_vad.VADEvent(
                type=etype,
                samples_index=self._samples,
                timestamp=time.time(),
                speech_duration=speech_duration,
                silence_duration=silence_duration,
                frames=frames or [],
                probability=prob,
                inference_duration=0.0,
                speaking=speaking,
            )
        )

    async def aclose(self) -> None:
        self._detector.close()
        await super().aclose()


class _StreamingVAD(agents_vad.VAD):
    """Base for roomkit-style LiveKit VAD providers."""

    def __init__(
        self,
        config: VADConfig,
        *,
        model: str,
        provider: str,
    ) -> None:
        super().__init__(
            capabilities=agents_vad.VADCapabilities(update_interval=config.update_interval)
        )
        self._config = config
        self._model = model
        self._provider = provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    @abstractmethod
    def _make_detector(self) -> _FrameDetector:
        ...

    def stream(self) -> agents_vad.VADStream:
        return _VADStream(self)
