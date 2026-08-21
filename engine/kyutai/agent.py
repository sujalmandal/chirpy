#!/usr/bin/env python3
"""Chirpy agent worker built on LiveKit Agents.

The worker runs as a LiveKit agent: it connects to a self-hosted LiveKit server,
joins the "chirpy" room, and runs a voice AgentSession. Speech recognition and
synthesis stay on-device via small local models (faster-whisper STT + Kokoro TTS,
wrapped as LiveKit STT/TTS plugins). The LLM is any OpenAI-compatible endpoint
(LM Studio or hosted). VAD and turn detection are LiveKit-native (Silero).

Run with:
    engine/kyutai/.venv/bin/python engine/kyutai/agent.py start
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, room_io
from livekit.agents.inference import TurnDetector
from livekit.plugins import dtln, openai, silero

from plugins import KokoroTTS, WhisperSTT

logger = logging.getLogger("chirpy.agent")

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "local.env"


def load_config() -> dict[str, str]:
    values: dict[str, str] = {}
    if CONFIG.exists():
        for raw in CONFIG.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    for key, value in os.environ.items():
        if value:
            values[key] = value
    return values


async def entrypoint(ctx: JobContext):
    config = load_config()
    await ctx.connect()

    system_prompt = config.get(
        "ASSISTANT_SYSTEM",
        "You are Chirpy, a concise, warm, and natural voice assistant. "
        "Use short, natural spoken sentences unless the user asks for detail.",
    )

    session = AgentSession(
        stt=WhisperSTT(
            model_size=config.get("STT_MODEL", "base"),
            device=config.get("STT_DEVICE", "cpu"),
            compute_type=config.get("STT_COMPUTE_TYPE", "int8"),
            language=config.get("STT_LANGUAGE", "en"),
        ),
        tts=KokoroTTS(
            lang_code=config.get("TTS_LANG", "a"),
            voice=config.get("TTS_VOICE", "af_heart"),
            device=config.get("TTS_DEVICE", "cpu"),
            speed=float(config.get("TTS_SPEED", "1.0")),
        ),
        vad=silero.VAD.load(),
        llm=openai.LLM(
            base_url=config.get("LLM_BASE_URL"),
            model=config.get("LLM_MODEL_NAME"),
            api_key=config.get("LLM_API_KEY") or "local",
        ),
        turn_handling={
            "turn_detection": TurnDetector(),
            # Barge-in is disabled: the client plays the agent's TTS through the
            # speakers, and without reliable echo cancellation the mic picks that
            # audio back up, causing the agent to interrupt its own speech.
            "interruption": {"enabled": False},
        },
        # DTLN noise suppression on the inbound mic audio (self-hosted, in-process).
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(),
            ),
        ),
    )

    agent = Agent(instructions=system_prompt)
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Greet the user briefly.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="chirpy-agent",
        )
    )
