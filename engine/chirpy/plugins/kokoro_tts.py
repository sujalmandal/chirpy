"""LiveKit Agents TTS plugin backed by Kokoro (local, small, 82M)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

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
        self._prev_pipeline: KPipeline | None = None
        self._prev_lang: str | None = None
        self._lock = threading.Lock()

    def prewarm(self):
        self._ensure_pipeline()

    def reload(
        self,
        lang_code: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
    ) -> None:
        """Hot-reload voice/speed/language in place, without replacing the plugin.

        The language is baked into the Kokoro pipeline at construction, so a
        language change recreates the pipeline (on the next synthesis) with the
        new ``lang_code``. Voice and speed apply immediately. If the new language
        can't be loaded (e.g. a missing tokenizer dependency), the previous
        pipeline/language is kept and the failure is logged rather than crashing.
        """
        with self._lock:
            if voice is not None:
                self._voice = voice
            if speed is not None:
                self._speed = speed
            if lang_code is not None and lang_code != self._lang_code:
                self._prev_pipeline = self._pipeline
                self._prev_lang = self._lang_code
                self._lang_code = lang_code
                # Recreate on next synthesize so the new language takes effect.
                self._pipeline = None

    def _ensure_pipeline(self) -> KPipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    try:
                        self._pipeline = KPipeline(
                            lang_code=self._lang_code,
                            device=self._device,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "failed to load Kokoro pipeline for lang %r: %s",
                            self._lang_code, exc,
                        )
                        if self._prev_pipeline is not None:
                            # Revert to the last working language so we don't crash.
                            self._pipeline = self._prev_pipeline
                            self._lang_code = self._prev_lang or self._lang_code
                            logger.warning("reverted TTS language to %r", self._lang_code)
                    self._prev_pipeline = None
                    self._prev_lang = None
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
        tts_obj = self._tts

        def synthesize_blocking() -> list[bytes]:
            pipeline = tts_obj._ensure_pipeline()
            chunks: list[bytes] = []
            for result in pipeline(
                self._input_text,
                voice=tts_obj._voice,
                speed=tts_obj._speed,
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
