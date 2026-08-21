#!/usr/bin/env python3
"""Smoke test for the Chirpy engine: load STT + TTS, synthesize a phrase to a
WAV file, transcribe it back, and print both. Run after setup with:

    engine/chirpy/.venv/bin/python engine/chirpy/validate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from agent import ROOT, load_config
from plugins.kokoro_tts import KokoroTTS
from plugins.whisper_stt import WhisperSTT

OUT = ROOT / "engine" / "chirpy" / "smoke.wav"
PHRASE = "Welcome to Chirpy."


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int):
    pcm16 = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    import wave
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())


def main() -> int:
    config = load_config()
    t0 = time.time()
    print("Loading STT…", flush=True)
    stt = WhisperSTT(
        model_size=config.get("STT_MODEL", "base"),
        device=config.get("STT_DEVICE", "cpu"),
        compute_type=config.get("STT_COMPUTE_TYPE", "int8"),
        language=config.get("STT_LANGUAGE", "en"),
    )
    stt.prewarm()
    print(f"STT loaded in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    print("Loading TTS…", flush=True)
    tts = KokoroTTS(
        lang_code=config.get("TTS_LANG", "a"),
        voice=config.get("TTS_VOICE", "af_heart"),
        device=config.get("TTS_DEVICE", "cpu"),
        speed=float(config.get("TTS_SPEED", "1.0")),
    )
    tts.prewarm()
    print(f"TTS loaded in {time.time() - t0:.1f}s", flush=True)

    import asyncio

    async def run():
        frames: list[bytes] = []
        stream = tts.synthesize(PHRASE)
        async for chunk in stream:
            frames.append(chunk.frame.data)
        pcm = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        seconds = len(pcm) / 24_000
        print(f"Synthesized {PHRASE!r} -> {seconds:.2f}s audio", flush=True)
        write_wav(OUT, pcm, 24_000)
        print(f"Wrote {OUT}", flush=True)

        begin = time.time()
        pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
        from livekit import rtc
        frame = rtc.AudioFrame(
            data=pcm16.tobytes(),
            sample_rate=24_000,
            num_channels=1,
            samples_per_channel=len(pcm16),
        )
        transcript = await stt.recognize(frame)
        text = transcript.alternatives[0].text if transcript.alternatives else ""
        print(f"Transcribed in {time.time() - begin:.2f}s: {text!r}", flush=True)
        print(f"Expected:  {PHRASE!r}", flush=True)
        ok = text.strip().lower() == PHRASE.lower()
        print("OK" if ok else "MISMATCH", flush=True)
        return 0 if ok else 1

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
