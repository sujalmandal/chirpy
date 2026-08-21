"""LiveKit Agents STT plugin backed by faster-whisper (local, small).

Supports *streaming* recognition so the client can see the user's words in
real time instead of only after they stop talking. faster-whisper is an offline
model, so we pseudo-stream: audio is segmented with Silero VAD and the current
speech segment is re-transcribed on a short interval, emitting interim
(``INTERIM_TRANSCRIPT``) events as the user speaks. A final transcript is
emitted when the segment ends.
"""

from __future__ import annotations

import asyncio

import numpy as np

from livekit import rtc
from livekit.agents import APIConnectOptions, stt, utils, vad as agent_vad
from livekit.agents.stt import (
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.plugins import silero

from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000
# Re-transcribe the in-progress segment roughly this often to refresh the
# interim caption. Keep it modest so CPU (faster-whisper on Apple Silicon)
# doesn't lag the conversation.
DEFAULT_INTERIM_INTERVAL = 0.6


class WhisperSTT(stt.STT):
    """LiveKit Agents STT plugin using a local faster-whisper model.

    The plugin declares itself streaming-capable. Audio is segmented with an
    internal Silero VAD and the in-progress segment is transcribed on an
    interval, emitting interim transcripts so the UI can show the user's speech
    as it happens. Set ``interim_interval`` to 0 to disable interim emission.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        interim_interval: float = DEFAULT_INTERIM_INTERVAL,
    ):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=interim_interval > 0,
                aligned_transcript=False,
                offline_recognize=True,
            )
        )
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self.interim_interval = interim_interval
        self._model: WhisperModel | None = None
        self._vad: agent_vad.VAD | None = None
        self._vad_lock = asyncio.Lock()

    def prewarm(self):
        self._ensure_model()

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    async def _ensure_vad(self) -> agent_vad.VAD:
        if self._vad is None:
            async with self._vad_lock:
                if self._vad is None:
                    self._vad = silero.VAD.load()
        return self._vad

    def stream(
        self,
        *,
        language=None,
        conn_options: APIConnectOptions | None = None,
    ) -> RecognizeStream:
        return _WhisperStream(
            stt=self,
            language=language,
            conn_options=conn_options or APIConnectOptions(),
        )

    async def _recognize_impl(
        self,
        buffer,
        *,
        language=None,
        conn_options: APIConnectOptions | None = None,
    ) -> SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        text = await self._transcribe(frames, language=language)
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text=text, language=language or self._language)],
        )

    async def _transcribe(self, frames, *, language=None) -> str:
        """Transcribe a list of frames to text, off the event loop."""
        if not frames:
            return ""
        model = self._ensure_model()
        pcm = _frames_to_pcm(frames)
        loop = asyncio.get_running_loop()

        def run() -> str:
            segments, _info = model.transcribe(
                pcm,
                language=language or self._language,
                beam_size=1,
                vad_filter=True,
                without_timestamps=True,
            )
            return "".join(s.text for s in segments).strip()

        return await loop.run_in_executor(None, run)


class _WhisperStream(RecognizeStream):
    """Segments mic audio with VAD and emits interim + final transcripts."""

    def __init__(
        self,
        *,
        stt: WhisperSTT,
        language,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options)
        self._stt: WhisperSTT = stt
        self._language = language

    async def _run(self) -> None:
        stt_obj = self._stt
        vad = await stt_obj._ensure_vad()
        vad_stream = vad.stream()

        segment: list[rtc.AudioFrame] = []
        in_speech = False
        lock = asyncio.Lock()

        async def _forward_input() -> None:
            """Push mic audio into the VAD and accumulate the current segment."""
            nonlocal segment, in_speech
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    vad_stream.flush()
                    continue
                vad_stream.push_frame(item)
                if in_speech:
                    segment.append(item)

        async def _emit_interim() -> None:
            """Periodically re-transcribe the live segment as a partial caption."""
            interval = stt_obj.interim_interval
            if interval <= 0:
                return
            while True:
                await asyncio.sleep(interval)
                if lock.locked():
                    # A transcription is still running; skip and transcribe the
                    # latest audio on the next tick instead of queueing up.
                    continue
                if not in_speech or not segment:
                    continue
                buf = list(segment)
                async with lock:
                    text = await stt_obj._transcribe(buf, language=self._language)
                if text:
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[SpeechData(text=text, language=self._language)],
                        )
                    )

        async def _recognize() -> None:
            """Turn VAD boundaries into start/end + a final transcript."""
            nonlocal segment, in_speech
            async for event in vad_stream:
                if event.type == agent_vad.VADEventType.START_OF_SPEECH:
                    self._event_ch.send_nowait(
                        SpeechEvent(SpeechEventType.START_OF_SPEECH)
                    )
                    in_speech = True
                    segment = []
                elif event.type == agent_vad.VADEventType.END_OF_SPEECH:
                    self._event_ch.send_nowait(
                        SpeechEvent(SpeechEventType.END_OF_SPEECH)
                    )
                    in_speech = False
                    buf = list(segment)
                    async with lock:
                        text = await stt_obj._transcribe(buf, language=self._language)
                    if text:
                        self._event_ch.send_nowait(
                            SpeechEvent(
                                type=SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[SpeechData(text=text, language=self._language)],
                            )
                        )

        forward_task = asyncio.create_task(_forward_input(), name="forward_input")
        interim_task = asyncio.create_task(_emit_interim(), name="emit_interim")
        recognize_task = asyncio.create_task(_recognize(), name="recognize")
        try:
            # Run all three loops for the life of the stream. This returns only
            # when the stream is closed: `stream.aclose()` cancels our task,
            # which propagates through `gather` and reaches the `finally`.
            await asyncio.gather(forward_task, interim_task, recognize_task)
        finally:
            for task in (forward_task, interim_task, recognize_task):
                await utils.aio.cancel_and_wait(task)
            await vad_stream.aclose()


def _frames_to_pcm(frames: list[rtc.AudioFrame]) -> np.ndarray:
    pcm = np.concatenate(
        [np.frombuffer(f.data, dtype=np.int16) for f in frames]
    ).astype(np.float32) / 32768.0
    if frames and frames[0].sample_rate != SAMPLE_RATE:
        pcm = _resample(pcm, frames[0].sample_rate, SAMPLE_RATE)
    return pcm


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or samples.size == 0:
        return samples
    output_count = round(samples.size * to_rate / from_rate)
    src = np.arange(output_count, dtype=np.float32) * (from_rate / to_rate)
    return np.interp(src, np.arange(samples.size, dtype=np.float32), samples).astype(
        np.float32
    )
