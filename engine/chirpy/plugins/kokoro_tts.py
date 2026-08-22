"""LiveKit Agents TTS plugin backed by Kokoro (local, small, 82M)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np
import torch

from livekit import rtc
from livekit.agents import APIConnectOptions, tts

from kokoro import KPipeline

logger = logging.getLogger("chirpy.kokoro_tts")

SAMPLE_RATE = 24_000


def _feed_reference(apm, pcm: np.ndarray, sample_rate: int) -> None:
    """Feed TTS audio to the AEC module as the echo-cancellation reference."""
    spf = sample_rate * 10 // 1000  # 10 ms frames (WebRTC AEC requirement)
    n = (pcm.size // spf) * spf
    for block in pcm[:n].reshape(-1, spf):
        sub = rtc.AudioFrame(
            data=(np.clip(block, -1, 1) * 32767).astype(np.int16).tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=spf,
        )
        apm.process_reverse_stream(sub)


class KokoroTTS(tts.TTS):
    """LiveKit Agents TTS plugin using the local Kokoro model."""

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        device: str = "cpu",
        speed: float = 1.0,
        latency_cb=None,
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
        # Optional callback(stage, event, text="") wired to the latency tracker
        # by the agent so TTS timing reaches the debug UI.
        self.latency_cb = latency_cb or (lambda stage, event, text="": None)
        # Optional AEC module wired by the agent; the synthesized audio is fed
        # to it as the echo-cancellation reference (reverse stream).
        self.apm = None

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

        def synthesize_blocking() -> bytes:
            pipeline = tts_obj._ensure_pipeline()
            parts: list[np.ndarray] = []
            for result in pipeline(
                self._input_text,
                voice=tts_obj._voice,
                speed=tts_obj._speed,
            ):
                if result.audio is None:
                    continue
                audio = result.audio.detach().cpu().numpy()
                pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                parts.append(pcm16)
            if not parts:
                return b""
            full = np.concatenate(parts)
            trimmed = _trim_and_cap_silence(full, SAMPLE_RATE)
            return trimmed.tobytes()

        # Synthesize the whole reply, trim dead air, then stream it to the
        # emitter in small real-time-paced chunks so the client's audio buffer
        # stays fed and doesn't underrun/stutter on the live WebRTC stream.
        tts_obj.latency_cb("tts", "start")
        data = await loop.run_in_executor(None, synthesize_blocking)
        tts_obj.latency_cb("tts", "done")
        if data:
            chunk_bytes = SAMPLE_RATE * 2 * 50 // 1000  # 50 ms of int16 mono
            # Deliver slightly faster than real-time so the client builds a small
            # jitter buffer instead of underrunning at every packet boundary.
            interval = 50 / 1000 * 0.8
            for i in range(0, len(data), chunk_bytes):
                chunk = data[i : i + chunk_bytes]
                # Feed this audio to the AEC as the reference (reverse stream) so
                # the server can cancel the agent's own voice from the mic.
                if tts_obj.apm is not None:
                    chunk_pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    _feed_reference(tts_obj.apm, chunk_pcm, SAMPLE_RATE)
                output_emitter.push(chunk)
                await asyncio.sleep(interval)
        output_emitter.flush()


def _trim_and_cap_silence(
    pcm: np.ndarray,
    sample_rate: int,
    frame_ms: int = 10,
    threshold: float = 150.0,
    max_silence_ms: int = 120,
) -> np.ndarray:
    """Trim leading/trailing near-silence and cap internal silence runs.

    Kokoro adds ~800ms of padding on each end and ~200ms+ between sentences,
    which plays back as an audible stop/start. This removes the dead air at the
    start/end and trims internal silent runs down to ``max_silence_ms`` so the
    reply flows continuously.
    """
    if pcm.size == 0:
        return pcm
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n = pcm.size
    nframes = (n + frame_len - 1) // frame_len
    padded = np.zeros(nframes * frame_len, dtype=pcm.dtype)
    padded[:n] = pcm
    frames = padded.reshape(nframes, frame_len).astype(np.float32)
    rms = np.sqrt((frames * frames).mean(axis=1))
    active = rms >= threshold
    if not active.any():
        return pcm

    start_frame = int(np.argmax(active))
    end_frame = int(nframes - np.argmax(active[::-1]))
    max_frames = max(1, int(max_silence_ms / frame_ms))

    # Mark frames to keep: everything in [start,end) except the excess of any
    # internal silence run longer than max_frames.
    keep = np.ones(end_frame, dtype=bool)
    i = start_frame
    while i < end_frame:
        if not active[i]:
            j = i
            while j < end_frame and not active[j]:
                j += 1
            if j - i > max_frames:
                keep[i + max_frames : j] = False
            i = j
        else:
            i += 1

    kept = [padded[f * frame_len : (f + 1) * frame_len] for f in range(start_frame, end_frame) if keep[f]]
    if not kept:
        return pcm
    out = np.concatenate(kept)
    # Drop any trailing padding zeros we introduced by frame-alignment.
    return out.astype(pcm.dtype)
