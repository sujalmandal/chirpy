"""LiveKit Agents TTS plugin backed by Kokoro (local, small, 82M)."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time

import numpy as np
import torch

from livekit import rtc
from livekit.agents import APIConnectOptions, tts

from kokoro import KPipeline

logger = logging.getLogger("chirpy.kokoro_tts")

SAMPLE_RATE = 24_000


def _send_native(data: bytes) -> None:
    """Stream TTS PCM to the app's native (non-WebRTC) audio output."""
    port = os.environ.get("CHIRPY_NATIVE_AUDIO_PORT")
    if not port:
        return
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0) as s:
            s.sendall(data)
    except Exception:  # native audio unavailable; fall back to silence/room
        pass


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
            trimmed = _trim_edges(full, SAMPLE_RATE)
            return trimmed.tobytes()

        # Synthesize the whole reply, trim dead air, then stream it to the
        # emitter in small real-time-paced chunks so the client's audio buffer
        # stays fed and doesn't underrun/stutter on the live WebRTC stream.
        tts_obj.latency_cb("tts", "start")
        data = await loop.run_in_executor(None, synthesize_blocking)
        tts_obj.latency_cb("tts", "done")
        if data:
            # Feed the audio to the AEC as the echo-cancellation reference.
            if tts_obj.apm is not None:
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                _feed_reference(tts_obj.apm, pcm, SAMPLE_RATE)
            # Play natively (bypassing the stutter-prone webview WebRTC path).
            _send_native(data)
            # Also publish to the room so the session's turn handling stays in
            # sync (the webview no longer plays this track).
            output_emitter.push(data)
        output_emitter.flush()


def _trim_edges(
    pcm: np.ndarray,
    sample_rate: int,
    frame_ms: int = 10,
    threshold: float = 150.0,
) -> np.ndarray:
    """Trim only the leading/trailing near-silence from int16 PCM.

    Kokoro pads ~800ms of dead air on each end. We cut it from the edges only —
    never from the middle — because removing interior samples creates abrupt
    waveform discontinuities (clicks) that play back as stutter/chopping. Natural
    inter-sentence pauses are kept intact.
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
    start = int(np.argmax(active)) * frame_len
    end = min(int(nframes - np.argmax(active[::-1])) * frame_len, n)
    return pcm[start:end]
