#!/usr/bin/env python3
"""Smoke test for the Kyutai engine: load STT + TTS, synthesize a phrase to a
WAV file, transcribe it back, and print both. Run after setup with:

    engine/kyutai/.venv/bin/python engine/kyutai/validate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from agent import ROOT, BLOCK_SAMPLES, TextToSpeech, SpeechToText, load_config

OUT = ROOT / "engine" / "kyutai" / "smoke.wav"
PHRASE = "Welcome to your local voice assistant."


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
    stt = SpeechToText(config)
    print(f"STT loaded in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    print("Loading TTS…", flush=True)
    tts = TextToSpeech(config)
    print(f"TTS loaded in {time.time() - t0:.1f}s", flush=True)

    frames: list[bytes] = []
    begin = time.time()
    tts.synthesize(PHRASE, frames.append)
    elapsed = time.time() - begin
    pcm = np.frombuffer(b"".join(frames), dtype=np.float32)
    seconds = len(pcm) / 24_000
    print(f"Synthesized {PHRASE!r} -> {len(frames)} frames ({seconds:.2f}s audio) "
          f"in {elapsed:.2f}s ({seconds / elapsed:.2f}x realtime)", flush=True)
    write_wav(OUT, pcm, 24_000)
    print(f"Wrote {OUT}", flush=True)

    begin = time.time()
    transcript = transcribe_pcm(stt, pcm)
    print(f"Transcribed in {time.time() - begin:.2f}s: {transcript!r}", flush=True)
    print(f"Expected:  {PHRASE!r}", flush=True)
    ok = transcript.strip().lower() == PHRASE.lower()
    print("OK" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


def transcribe_pcm(stt: SpeechToText, pcm: np.ndarray) -> str:
    """Run the STT stream over a full PCM buffer, chunked into 80 ms blocks.

    The STT model delays its output by 0.5 s, so feed trailing silence blocks
    after the audio to let the final words flush out.
    """
    out: list[str] = []
    stt.reset()
    silence = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
    blocks = list(range(0, len(pcm) - 1919, 1_920))
    trailing = [silence] * 7  # ~0.5 s of silence for the model delay
    for i in blocks:
        chunk = pcm[i:i + 1_920]
        if chunk.size < 1_920:
            break
        fragment, _ = stt.step(chunk.tobytes())
        if fragment:
            out.append(fragment)
    for chunk in trailing:
        fragment, _ = stt.step(chunk.tobytes())
        if fragment:
            out.append(fragment)
    return "".join(out)


if __name__ == "__main__":
    sys.exit(main())
