from __future__ import annotations

import asyncio
import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Sequence

import httpx

from .conversation import Message


class SpeechToText(ABC):
    @abstractmethod
    async def transcribe(self, wav_path: Path) -> str: ...


class ChatModel(ABC):
    @abstractmethod
    async def stream_reply(self, messages: Sequence[Message]) -> AsyncIterator[str]: ...


class TextToSpeech(ABC):
    @abstractmethod
    async def synthesize_wav(self, text: str) -> bytes: ...


class WhisperCppSTT(SpeechToText):
    """Adapter for whisper.cpp CLI; replace this class for MLX Whisper, etc."""

    def __init__(self, model: str | None = None) -> None:
        self.binary = os.environ.get("WHISPER_CPP_BIN", "whisper-cli")
        self.model = model or os.environ.get("WHISPER_MODEL")

    async def transcribe(self, wav_path: Path) -> str:
        if not self.model:
            raise RuntimeError("WHISPER_MODEL is not configured")
        output_base = Path(tempfile.mkstemp(prefix="voice-stt-")[1])
        output_base.unlink(missing_ok=True)
        process = await asyncio.create_subprocess_exec(
            self.binary, "-m", self.model, "-f", str(wav_path), "-otxt", "-of", str(output_base),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        text_file = output_base.with_suffix(".txt")
        try:
            if process.returncode != 0:
                raise RuntimeError(f"whisper.cpp failed: {stderr.decode(errors='replace')[-500:]}")
            return text_file.read_text().strip()
        finally:
            text_file.unlink(missing_ok=True)


class OpenAICompatibleChat(ChatModel):
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "qwen3:14b")
        self.system = os.environ.get(
            "ASSISTANT_SYSTEM", "You are a concise, helpful local voice assistant. Reply naturally in short spoken sentences."
        )

    async def stream_reply(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        payload = {"model": self.model, "stream": True, "messages": [
            {"role": "system", "content": self.system},
            *({"role": message.role, "content": message.content} for message in messages),
        ]}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    token = json.loads(data).get("choices", [{}])[0].get("delta", {}).get("content")
                    if token:
                        yield token


class PiperTTS(TextToSpeech):
    """Adapter for local Piper. It returns a self-contained WAV for simple client playback."""

    def __init__(self, model: str | None = None) -> None:
        self.binary = os.environ.get("PIPER_BIN", "piper")
        self.model = model or os.environ.get("PIPER_MODEL")

    async def synthesize_wav(self, text: str) -> bytes:
        if not self.model:
            raise RuntimeError("PIPER_MODEL is not configured")
        fd, output_name = tempfile.mkstemp(prefix="voice-tts-", suffix=".wav")
        os.close(fd)
        output = Path(output_name)
        process = await asyncio.create_subprocess_exec(
            self.binary, "--model", self.model, "--output_file", str(output),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate(text.encode())
        try:
            if process.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr.decode(errors='replace')[-500:]}")
            return output.read_bytes()
        finally:
            output.unlink(missing_ok=True)


class OpenAICompatibleSTT(SpeechToText):
    """Adapter for POST /audio/transcriptions-compatible local servers."""

    def __init__(self, base_url: str, model: str) -> None:
        self.url = _endpoint(base_url, "audio/transcriptions")
        self.model = model

    async def transcribe(self, wav_path: Path) -> str:
        async with httpx.AsyncClient(timeout=None) as client:
            with wav_path.open("rb") as audio:
                response = await client.post(self.url, data={"model": self.model}, files={"file": ("speech.wav", audio, "audio/wav")})
            response.raise_for_status()
            return response.json().get("text", "").strip()


class OpenAICompatibleTTS(TextToSpeech):
    """Adapter for POST /audio/speech-compatible local servers."""

    def __init__(self, base_url: str, model: str, voice: str = "alloy") -> None:
        self.url = _endpoint(base_url, "audio/speech")
        self.model = model
        self.voice = voice

    async def synthesize_wav(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(self.url, json={"model": self.model, "input": text, "voice": self.voice, "response_format": "wav"})
            response.raise_for_status()
            return response.content


def _endpoint(base_url: str, suffix: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith(suffix) else f"{base_url}/{suffix}"
