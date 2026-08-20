from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .conversation import ConversationStore, Message
from .providers import OpenAICompatibleChat, OpenAICompatibleSTT, OpenAICompatibleTTS, PiperTTS, SpeechToText, TextToSpeech, WhisperCppSTT
from .text import complete_sentences

app = FastAPI(title="Local Voice Agent", version="0.1.0")
conversations = ConversationStore()


def event(kind: str, **payload: object) -> bytes:
    return (json.dumps({"type": kind, **payload}) + "\n").encode()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "local-voice-agent"}


@app.post("/v1/turn")
async def turn(
    audio: UploadFile = File(...),
    session_id: str = Form(default="default"),
    stt_endpoint: str = Form(default=""), stt_model: str = Form(default=""),
    llm_endpoint: str = Form(default=""), llm_model: str = Form(default=""),
    tts_endpoint: str = Form(default=""), tts_model: str = Form(default=""),
) -> StreamingResponse:
    if audio.content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
        raise HTTPException(415, "Upload 16-bit WAV audio")
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    fd, filename = tempfile.mkstemp(prefix="voice-input-", suffix=suffix)
    os.close(fd)
    input_path = Path(filename)
    input_path.write_bytes(await audio.read())
    selected_stt: SpeechToText = OpenAICompatibleSTT(stt_endpoint, stt_model) if stt_endpoint else WhisperCppSTT(stt_model or None)
    selected_chat = OpenAICompatibleChat(llm_endpoint or None, llm_model or None)
    selected_tts: TextToSpeech = OpenAICompatibleTTS(tts_endpoint, tts_model) if tts_endpoint else PiperTTS(tts_model or None)

    async def stream() -> AsyncIterator[bytes]:
        try:
            transcript = await selected_stt.transcribe(input_path)
            yield event("transcript", text=transcript)
            if not transcript:
                yield event("done")
                return
            buffer = ""
            full_reply = ""
            messages = [*conversations.context(session_id), Message("user", transcript)]
            async for token in selected_chat.stream_reply(messages):
                yield event("text", delta=token)
                full_reply += token
                buffer += token
                sentences, buffer = complete_sentences(buffer)
                for sentence in sentences:
                    wav = await selected_tts.synthesize_wav(sentence)
                    yield event("audio", text=sentence, wav_base64=base64.b64encode(wav).decode())
            sentences, _ = complete_sentences(buffer, final=True)
            for sentence in sentences:
                wav = await selected_tts.synthesize_wav(sentence)
                yield event("audio", text=sentence, wav_base64=base64.b64encode(wav).decode())
            conversations.add_turn(session_id, transcript, full_reply)
            yield event("done")
        except Exception as exc:
            yield event("error", message=str(exc))
        finally:
            input_path.unlink(missing_ok=True)

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


@app.delete("/v1/sessions/{session_id}", status_code=204)
async def clear_session(session_id: str) -> None:
    conversations.clear(session_id)
