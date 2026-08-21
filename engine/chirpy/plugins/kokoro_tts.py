"""LiveKit Agents TTS plugin backed by Kokoro (local, small, 82M)."""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np
import torch

from livekit.agents import APIConnectOptions, tts

from kokoro import KPipeline

logger = logging.getLogger("chirpy.kokoro_tts")

SAMPLE_RATE = 24_000


class KokoroTTS(tts.TTS):
    """LiveKit Agents TTS plugin using the local Kokoro model."""

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        device: str = "cpu",
        speed: float = 1.0,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._lang_code = lang_code
        self._voice = voice
        self._device = device
        self._speed = speed
        self._pipeline: KPipeline | None = None

    def prewarm(self):
        self._pipeline = KPipeline(
            lang_code=self._lang_code,
            device=self._device,
        )

    def _ensure_pipeline(self) -> KPipeline:
        if self._pipeline is None:
            self._pipeline = KPipeline(
                lang_code=self._lang_code,
                device=self._device,
            )
        return self._pipeline

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = None):
        opts = conn_options or APIConnectOptions(max_retry=0)
        return KokoroChunkedStream(self, text, opts)


class KokoroChunkedStream(tts.ChunkedStream):
    """ChunkedStream pushing Kokoro PCM to the LiveKit output emitter."""

    def __init__(self, kokoro_tts: KokoroTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=kokoro_tts, input_text=input_text, conn_options=conn_options)
        self._tts: KokoroTTS = kokoro_tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="kokoro",
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        loop = asyncio.get_running_loop()

        def synthesize_blocking() -> list[bytes]:
            pipeline = self._tts._ensure_pipeline()
            chunks: list[bytes] = []
            for result in pipeline(
                self._input_text,
                voice=self._tts._voice,
                speed=self._tts._speed,
            ):
                if result.audio is None:
                    continue
                audio = result.audio.detach().cpu().numpy()
                pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                chunks.append(pcm16.tobytes())
            return chunks

        chunks = await loop.run_in_executor(None, synthesize_blocking)
        for chunk in chunks:
            output_emitter.push(chunk)
        output_emitter.flush()
