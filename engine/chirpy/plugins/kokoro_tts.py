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

from livekit.agents import APIConnectOptions, tts

from kokoro import KPipeline

from aec import feed_reference

logger = logging.getLogger("chirpy.kokoro_tts")

SAMPLE_RATE = 24_000


class _NativeStream:
    """A single persistent TCP connection to the app's native audio output.

    Reconnecting per chunk is far too slow for real-time streaming, so one
    connection is opened for the whole utterance and kept until it is done.
    """

    def __init__(self, port: str | None):
        self._s: socket.socket | None = None
        self._port = port

    def open(self) -> None:
        if not self._port:
            return
        try:
            self._s = socket.create_connection(("127.0.0.1", int(self._port)), timeout=2.0)
        except Exception:  # native audio unavailable; fall back to the room only
            self._s = None

    def send(self, data: bytes) -> None:
        if self._s is None:
            return
        try:
            self._s.sendall(data)
        except Exception:
            self.close()

    def close(self) -> None:
        if self._s is not None:
            try:
                self._s.close()
            except Exception:
                pass
            self._s = None


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
        if not data:
            output_emitter.flush()
            return

        # ---- Server-side AEC reference, paced to playback --------------------
        # The reply is sent to the native speaker, fed to the AEC as its
        # far-end (echo) reference, and pushed to the LiveKit room all at the
        # same real-time cadence. This is the load-bearing detail:
        #
        #   WebRTC's echo-cancellation delay estimator only searches a small,
        #   bounded delay between the far-end (TTS reference) and the near-end
        #   (mic). If the whole reply were fed to the AEC up front, the
        #   reference would lead the mic by the entire utterance duration --
        #   far beyond the estimator's range -- so it could never converge and
        #   the agent would keep hearing its own voice. Streaming each 10 ms
        #   block at the moment it is actually played keeps that lead bounded
        #   to just the OS/speaker/acoustic latency, which the AEC can absorb.
        #
        # Pushing to the room at the same pace also keeps the session's turn
        # state in sync: the "speaking" turn stays open for the full spoken
        # duration, so the agent's own late-arriving echo can't be re-parsed
        # as a fresh user utterance after the turn has ended.
        native = _NativeStream(os.environ.get("CHIRPY_NATIVE_AUDIO_PORT"))
        native.open()
        frame_len = SAMPLE_RATE * 10 // 1000  # 160 samples == 10 ms
        pcm = np.frombuffer(data, dtype=np.int16)
        total = pcm.size
        start = time.perf_counter()
        cursor = 0
        try:
            while cursor < total:
                block = pcm[cursor : cursor + frame_len]
                if block.size == 0:
                    break
                b16 = block.astype(np.int16)
                # Far-end reference for the AEC, exactly the audio being played.
                if tts_obj.apm is not None:
                    feed_reference(tts_obj.apm, b16.astype(np.float32) / 32768.0, SAMPLE_RATE)
                # Play natively (bypassing the stutter-prone webview WebRTC path).
                native.send(b16.tobytes())
                # Keep the live room track in sync with real-time playback.
                output_emitter.push(b16.tobytes())
                cursor += frame_len
                # Pace to wall-clock: if we catch up (CPU kept up with real
                # time), sleep until this block would have been played.
                target = start + cursor / SAMPLE_RATE
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            native.close()
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
