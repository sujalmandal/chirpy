#!/usr/bin/env python3
"""Chirpy agent worker built on LiveKit Agents.

The worker runs as a LiveKit agent: it connects to a self-hosted LiveKit server,
joins the "chirpy" room, and runs a voice AgentSession. Speech recognition and
synthesis stay on-device via small local models (faster-whisper STT + Kokoro TTS,
wrapped as LiveKit STT/TTS plugins). The LLM is any OpenAI-compatible endpoint
(LM Studio or hosted). VAD and turn detection are LiveKit-native (Silero).

Barge-in (interruption) is configured through :mod:`bargein` — a validated,
data-driven policy with no hardcoded magic numbers — and guarded against speaker
echo by :mod:`echoguard`.

Run with:
    engine/chirpy/.venv/bin/python engine/chirpy/agent.py start
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, room_io
from livekit.plugins import dtln, openai

from bargein import (
    ConfigWatcher,
    _as_bool,
    _as_float,
    _as_int,
    build_turn_handling,
    load_barge_in_policy,
)
from echoguard import EchoGuard
from latency import LatencyTracker
from plugins import KokoroTTS, WhisperSTT
from vad import build_vad

logger = logging.getLogger("chirpy.agent")

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "local.env"
# Runtime barge-in overrides; the worker watches this file and applies changes
# without a restart (endpointing live, interruption on the next room/restart).
RUNTIME_BARGEIN = ROOT / "config" / "barge-in.json"

# Live TTS overrides; the worker watches this file and hot-swaps the voice
# without a restart (the debug UI writes it via the `set_tts` command).
RUNTIME_TTS = ROOT / "config" / "tts.json"
# Live STT overrides (model size / language), applied without a restart.
RUNTIME_STT = ROOT / "config" / "stt.json"

# Default assistant system prompt tuned for natural, spoken conversation.
DEFAULT_SYSTEM_PROMPT = (
    "You are Chirpy, a warm and natural voice assistant. Sound like a friendly "
    "person speaking aloud rather than a written answer: keep replies short and "
    "spoken, use complete sentences, avoid lists and markdown, and don't "
    "over-explain unless the user asks for detail. Welcome the user steering or "
    "interrupting you at any time."
)

# Re-exported for tests / convenience.
__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "_as_bool",
    "_as_float",
    "_as_int",
    "resolve_system_prompt",
    "resolve_turn_handling",
    "build_session",
    "load_config",
]


def resolve_system_prompt(config: dict[str, str]) -> str:
    """Resolve the assistant system prompt from config, defaulting to natural."""
    return config.get("ASSISTANT_SYSTEM") or DEFAULT_SYSTEM_PROMPT


def resolve_turn_handling(config: dict[str, str]) -> dict:
    """Build the AgentSession ``turn_handling`` from the validated barge-in policy."""
    return build_turn_handling(load_barge_in_policy(config))


def build_session(config: dict[str, str]) -> AgentSession:
    """Construct the AgentSession with the configured speech pipeline.

    Separated from the worker entrypoint so tests can exercise the exact
    session configuration (turn handling, barge-in, models) without connecting
    to LiveKit or loading models.
    """
    policy = load_barge_in_policy(config)
    return AgentSession(
        stt=WhisperSTT(
            model_size=config.get("STT_MODEL", "base"),
            device=config.get("STT_DEVICE", "cpu"),
            compute_type=config.get("STT_COMPUTE_TYPE", "int8"),
            language=config.get("STT_LANGUAGE", "en"),
            interim_interval=float(config.get("STT_INTERIM_INTERVAL", "0.6")),
        ),
        tts=KokoroTTS(
            lang_code=config.get("TTS_LANG", "a"),
            voice=config.get("TTS_VOICE", "af_heart"),
            device=config.get("TTS_DEVICE", "cpu"),
            speed=float(config.get("TTS_SPEED", "1.0")),
        ),
        vad=build_vad(config),
        llm=openai.LLM(
            base_url=config.get("LLM_BASE_URL"),
            model=config.get("LLM_MODEL_NAME"),
            api_key=config.get("LLM_API_KEY") or "local",
        ),
        turn_handling=build_turn_handling(policy),
        # Don't allow interruptions until acoustic echo cancellation converges,
        # so the agent's own startup audio can't trigger a bogus barge-in.
        aec_warmup_duration=policy.aec_warmup_duration,
    )


async def _publish(ctx: JobContext, payload: dict) -> None:
    """Publish a JSON data message to the room so the client can render it."""
    if ctx.room.isconnected():
        await ctx.room.local_participant.publish_data(
            json.dumps(payload).encode("utf-8"), topic="chirpy"
        )


def _setup_transcript_bridge(
    ctx: JobContext, session: AgentSession, guard: EchoGuard, tracker: LatencyTracker | None = None
) -> None:
    """Forward the voice session's transcript to the client as data events.

    The client renders a full user/assistant conversation from these events:
      - {"type": "partial", "text": ...}   live user caption
      - {"type": "user", "text": ...}      final user transcript
      - {"type": "assistant_delta", "text": ...}  assistant text (committed)
      - {"type": "assistant_end"}          assistant finished speaking

    The EchoGuard feeds the assistant's speech and withholds transcripts that
    look like the agent's own echo so the client never shows a phantom "you".
    """

    def on_user_transcribed(ev) -> None:
        if not getattr(ev, "transcript", None):
            return
        # Echo of the agent's own last words can be transcribed right after the
        # TTS ends, so we check content overlap against recent agent speech
        # unconditionally rather than gating on the momentary speaking state.
        if guard.is_echo(ev.transcript):
            # Likely the agent's own echo on the mic; don't forward as a user turn.
            logger.info(
                "echo-classified user transcript withheld final=%s", ev.is_final
            )
            return
        logger.info("bridge user_transcribed final=%s text=%r", ev.is_final, ev.transcript)
        asyncio.create_task(
            _publish(
                ctx,
                {
                    "type": "partial" if not ev.is_final else "user",
                    "text": ev.transcript,
                },
            )
        )

    def on_conversation_item_added(ev) -> None:
        item = getattr(ev, "item", None)
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            text = getattr(item, "text_content", None)
            if text:
                guard.note_assistant_text(text)
                if tracker is not None:
                    tracker.handle("llm", "text", text)
                logger.info(
                    "bridge assistant item role=%s text=%r",
                    getattr(item, "role", None),
                    text,
                )
                asyncio.create_task(_publish(ctx, {"type": "assistant_delta", "text": text}))
                asyncio.create_task(_publish(ctx, {"type": "assistant_end"}))

    session.on("user_input_transcribed", on_user_transcribed)
    session.on("conversation_item_added", on_conversation_item_added)


def _apply_policy_live(session: AgentSession, guard: EchoGuard) -> None:
    """Apply a reloaded policy to the running session/guard (best-effort)."""

    def on_change(policy) -> None:
        # Endpointing can be hot-swapped via the SDK.
        session.update_options(endpointing_opts=policy.endpointing_options())
        # Echo guard threshold is live.
        guard.policy = policy
        logger.info(
            "barge-in policy reloaded: enabled=%s mode=%s endpointing=%s echo_th=%.2f",
            policy.enabled,
            policy.mode,
            policy.endpointing_options(),
            policy.echo_overlap_threshold,
        )

    return on_change


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
    policy = load_barge_in_policy(config)
    await ctx.connect()

    system_prompt = resolve_system_prompt(config)
    session = build_session(config)
    guard = EchoGuard(policy)

    # Per-turn latency tracker: the STT/TTS plugins stamp VAD/STT/TTS timing,
    # the bridge stamps LLM timing, and when a turn completes the tracker
    # publishes a `latency` event the debug UI renders as a waterfall graph.
    def _publish_latency(payload: dict) -> None:
        try:
            asyncio.get_running_loop().create_task(_publish(ctx, payload))
        except RuntimeError:
            pass
        logger.info(
            "latency turn=%s vad=%sms stt=%sms llm=%sms tts=%sms total=%sms speech_to_transcript=%sms",
            payload.get("id"),
            payload.get("vad_ms"),
            payload.get("stt_ms"),
            payload.get("llm_ms"),
            payload.get("tts_ms"),
            payload.get("total_ms"),
            payload.get("speech_to_transcript_ms"),
        )

    tracker = LatencyTracker(publish=_publish_latency)
    session._stt.latency_cb = lambda stage, event, text="": tracker.handle(stage, event, text)
    session._tts.latency_cb = lambda stage, event, text="": tracker.handle(stage, event, text)

    agent = Agent(instructions=system_prompt)
    _setup_transcript_bridge(ctx, session, guard, tracker)
    await session.start(
        agent=agent,
        room=ctx.room,
        # DTLN noise suppression on the inbound mic audio (self-hosted, in-process).
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(),
            ),
        ),
    )

    # Dynamic, live barge-in tuning: watch config/barge-in.json.
    watcher = ConfigWatcher(
        RUNTIME_BARGEIN,
        _apply_policy_live(session, guard),
        base_config=config,
    )
    asyncio.create_task(watcher.run())

    await session.generate_reply(instructions="Greet the user briefly.")

    # Hot TTS/STT reload: watch config/tts.json and config/stt.json and apply
    # changes live without a restart.
    import tts_watch

    tts_watch.ConfigWatcher(
        RUNTIME_TTS, lambda data: tts_watch.apply_tts_live(session._tts, data)
    ).start()
    tts_watch.ConfigWatcher(
        RUNTIME_STT, lambda data: tts_watch.apply_stt_live(session._stt, data)
    ).start()


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
