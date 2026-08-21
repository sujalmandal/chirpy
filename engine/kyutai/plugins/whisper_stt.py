"""LiveKit Agents STT plugin backed by faster-whisper (local, small)."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from livekit import rtc
from livekit.agents import APIConnectOptions, stt

from faster_whisper import WhisperModel

logger = logging.getLogger("chirpy.whisper_stt")

SAMPLE_RATE = 16_000


class WhisperSTT(stt.STT):
    """LiveKit Agents STT plugin using a local faster-whisper model."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                aligned_transcript=False,
                offline_recognize=True,
            )
        )
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model: WhisperModel | None = None

    def prewarm(self):
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    async def _recognize_impl(
        self,
        buffer,
        *,
        language=None,
        conn_options: APIConnectOptions = None,
    ) -> stt.SpeechEvent:
        model = self._ensure_model()
        frames = buffer if isinstance(buffer, list) else [buffer]
        pcm = np.concatenate(
            [np.frombuffer(f.data, dtype=np.int16) for f in frames]
        ).astype(np.float32) / 32768.0
        # faster-whisper expects 16 kHz mono float32.
        if frames and frames[0].sample_rate != SAMPLE_RATE:
            pcm = _resample(pcm, frames[0].sample_rate, SAMPLE_RATE)
        segments, _info = model.transcribe(
            pcm,
            language=language or self._language,
            beam_size=1,
            vad_filter=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or self._language)],
        )


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or samples.size == 0:
        return samples
    output_count = round(samples.size * to_rate / from_rate)
    src = np.arange(output_count, dtype=np.float32) * (from_rate / to_rate)
    return np.interp(src, np.arange(samples.size, dtype=np.float32), samples).astype(
        np.float32
    )
