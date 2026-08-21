#!/usr/bin/env python3
"""Local voice agent built on Kyutai STT 1B (semantic VAD) + Ollama Cloud LLM
+ Kyutai TTS 1.6B (MLX).

Audio recognition (STT) and speech synthesis (TTS) run locally on MLX. The LLM
is an OpenAI-compatible cloud endpoint configured in config/local.env. The Swift
client streams microphone audio over WebSocket and receives streamed text events
plus decoded 24 kHz Float32 mono PCM frames for playback.

Wire protocol:
  client -> agent : binary Float32 LE mono 24 kHz PCM, 1920-sample (80 ms) blocks
  client -> agent : {"type":"interrupt"}  (barge-in)
  agent -> client : {"type":"transcript"|"turn_started"|"text"|"done"|"error", ...}
  agent -> client : binary Float32 LE mono 24 kHz PCM frames
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
import websockets

from moshi_mlx import models
from moshi_mlx.models import Lm, LmConfig
from moshi_mlx.models.mimi import Mimi
from moshi_mlx.models.tts import TTSModel, DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO
from moshi_mlx.utils import Sampler
from moshi_mlx.utils.loaders import hf_get

from endpointing import EndpointDetector

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "local.env"

SAMPLE_RATE = 24_000
BLOCK_SAMPLES = 1_920  # 80 ms
BLOCK_BYTES = BLOCK_SAMPLES * 4  # Float32
FLUSH_BLOCKS = 8  # ~0.64 s of silence to drain the STT's 0.5 s output delay
_SILENCE = np.zeros(BLOCK_SAMPLES, dtype=np.float32).tobytes()

HISTORY: dict[str, list[dict[str, str]]] = {}


class TurnCancelled(Exception):
    """Raised inside TTS generation when a barge-in cancels the turn."""


# MLX/Metal is not thread-safe: STT and TTS must never run concurrently on
# different threads. All inference goes through this single-worker executor so
# there is a total order over Metal operations.
MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

LOG_PATH = ROOT / "logs" / "kyutai-agent.log"
LOG_BACKUP_PATH = ROOT / "logs" / "kyutai-agent.log.1"
LOG_MAX_BYTES = 2_000_000
LOG_LOCK = threading.Lock()

DEBUG = False


def log(message: str) -> None:
    """Write a line to the agent log file and best-effort stdout.

    Never raises: a broken stdout pipe (e.g. the parent app quit) must not
    crash request handling or model loading.
    """
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n"
    try:
        with LOG_LOCK:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if LOG_PATH.exists() and LOG_PATH.stat().st_size >= LOG_MAX_BYTES:
                if LOG_BACKUP_PATH.exists():
                    LOG_BACKUP_PATH.unlink()
                LOG_PATH.replace(LOG_BACKUP_PATH)
            with open(LOG_PATH, "a") as fobj:
                fobj.write(line)
    except OSError:
        pass
    try:
        print(line.rstrip(), flush=True)
    except (BrokenPipeError, OSError):
        pass


def debug(message: str) -> None:
    """Log a line only when DEBUG is enabled (set DEBUG=1 in config/local.env)."""
    if DEBUG:
        log(f"[debug] {message}")


def load_config() -> dict[str, str]:
    values: dict[str, str] = {}
    if CONFIG.exists():
        for raw in CONFIG.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    # Explicit process environment always wins. The macOS app supplies its
    # persisted UI settings here, while local.env remains a CLI-friendly fallback.
    values.update(os.environ)
    return values


# ---------------------------------------------------------------------------
# LLM streaming (unchanged OpenAI-compatible logic from the previous agent)
# ---------------------------------------------------------------------------

def complete_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def stream_chat(text: str, session_id: str, config: dict[str, str]):
    endpoint = config.get("LLM_BASE_URL", "").strip()
    model = config.get("LLM_MODEL_NAME", "").strip()
    if not endpoint or not model:
        raise RuntimeError("Configure LLM_BASE_URL and LLM_MODEL_NAME in config/local.env")
    messages = [
        {"role": "system", "content": config.get(
            "ASSISTANT_SYSTEM",
            "You are a concise helpful voice assistant. Use short, natural spoken sentences.",
        )},
        *HISTORY.get(session_id, [])[-12:],
        {"role": "user", "content": text},
    ]
    payload = json.dumps({"model": model, "stream": True, "messages": messages}).encode()
    headers = {"Content-Type": "application/json"}
    if config.get("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {config['LLM_API_KEY']}"
    request = urllib.request.Request(complete_url(endpoint), data=payload, method="POST", headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=90)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Cloud LLM rejected the request ({error.code}): {error.read().decode(errors='replace')[-500:]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the cloud LLM: {error.reason}") from error
    with response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            event = line[6:]
            if event == "[DONE]":
                return
            try:
                delta = json.loads(event)["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                delta = None
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# HTTP /health endpoint (readiness polling from the Swift app)
# ---------------------------------------------------------------------------

_READY = {"stt_ready": False, "tts_ready": False, "config": {}}
_READY.update({"stt_error": None, "stt_error_count": 0, "stt_last_recovery": None})


class HealthHandler(BaseHTTPRequestHandler):
    server_version = "LocalVoiceAgent/0.2"

    def log_message(self, format: str, *args: object) -> None:
        # Health is polled continuously by the macOS app. Logging every request
        # would add thousands of low-value lines per hour.
        return

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        cfg = _READY["config"]
        payload = {
            "status": "ok",
            "service": "kyutai-voice-agent",
            "stt_ready": _READY["stt_ready"],
            "tts_ready": _READY["tts_ready"],
            "llm_configured": bool(cfg.get("LLM_BASE_URL") and cfg.get("LLM_MODEL_NAME")),
            "agent_name": cfg.get("AGENT_NAME", "Nova"),
            "vad_model": cfg.get("VAD_REPO", cfg.get("STT_REPO", "")),
            "stt_model": cfg.get("STT_REPO", ""),
            "tts_model": cfg.get("TTS_REPO", ""),
            "stt_error": _READY["stt_error"],
            "stt_error_count": _READY["stt_error_count"],
            "stt_last_recovery": _READY["stt_last_recovery"],
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# STT pipeline (streaming, semantic VAD)
# ---------------------------------------------------------------------------

class SpeechToText:
    def __init__(self, config: dict[str, str]):
        repo = config.get("STT_REPO", "kyutai/stt-1b-en_fr-candle").strip()
        log("STT: downloading/loading config")
        cfg_path = hf_get("config.json", repo)
        with open(cfg_path) as fobj:
            raw_config = json.load(fobj)
        mimi_weights = hf_get(raw_config["mimi_name"], repo)
        moshi_weights = hf_get(raw_config.get("moshi_name", "model.safetensors"), repo)
        tokenizer_name = hf_get(raw_config["tokenizer_name"], repo)

        lm_config = LmConfig.from_config_dict(raw_config)
        self.lm = Lm(lm_config)
        self.lm.set_dtype(mx.bfloat16)
        log("STT: loading LM weights")
        self.lm.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        self.tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_name))
        generated_codebooks = lm_config.generated_codebooks
        self.other_codebooks = lm_config.other_codebooks
        num_codebooks = max(generated_codebooks, self.other_codebooks)
        self.mimi_weights = str(mimi_weights)
        self.num_codebooks = num_codebooks
        self.audio_tokenizer = self._new_audio_tokenizer()
        # rustymimi advances more than one internal position for some audio
        # blocks. Rotate well before its fixed 8192-position RoPE table fills.
        self.rotate_blocks = max(500, int(config.get("STT_ROTATE_BLOCKS", "3000")))
        self.blocks_since_hard_reset = 0
        log("STT: warming up")
        self.lm.warmup()
        self.buffer = bytearray()
        self.gen = self._new_gen()

    def _new_gen(self):
        return models.LmGen(
            model=self.lm,
            max_steps=8192,
            text_sampler=Sampler(top_k=50, temp=0),
            audio_sampler=Sampler(top_k=250, temp=0.8),
            check=False,
        )

    def _new_audio_tokenizer(self):
        return rustymimi.Tokenizer(self.mimi_weights, num_codebooks=self.num_codebooks)

    def reset(self, hard: bool = False):
        # Clear model caches before constructing the next generator. Reversing
        # this order can leave the new LmGen attached to state from the old turn.
        for c in self.lm.transformer_cache:
            c.reset()
        for c in self.lm.depformer_cache:
            c.reset()
        self.buffer = bytearray()
        if hard:
            # rustymimi.Tokenizer.reset() does not reliably recover after its
            # internal positional context has overflowed. Recreate the tokenizer
            # from the already-cached weights for a true streaming-state reset.
            self.audio_tokenizer = self._new_audio_tokenizer()
            self.blocks_since_hard_reset = 0
        else:
            self.audio_tokenizer.reset()
        self.gen = self._new_gen()

    def should_rotate(self) -> bool:
        return self.blocks_since_hard_reset >= self.rotate_blocks

    def step(self, pcm_float32: bytes) -> tuple[str | None, float]:
        """Feed arbitrary-size PCM bytes; returns (new text fragment, VAD prob).

        Buffers input and processes exactly one 80 ms block per call once enough
        bytes are available. Returns (None, 0.0) when a full block is not yet
        buffered.
        """
        self.buffer += pcm_float32
        if len(self.buffer) < BLOCK_BYTES:
            return None, 0.0
        block = bytes(self.buffer[:BLOCK_BYTES])
        del self.buffer[:BLOCK_BYTES]
        samples = np.frombuffer(block, dtype=np.float32).reshape(1, 1, BLOCK_SAMPLES)
        audio_tokens = self.audio_tokenizer.encode_step(samples)
        audio_tokens = mx.array(audio_tokens).transpose(0, 2, 1)[:, :, : self.other_codebooks]
        text_token, extra_heads = self.gen.step_with_extra_heads(audio_tokens[0])
        fragment = None
        token = text_token[0].item()
        if token not in (0, 3):
            piece = self.tokenizer.id_to_piece(token)
            fragment = piece.replace("▁", " ")
        vad_prob = 0.0
        if extra_heads and len(extra_heads) > 2:
            vad_prob = extra_heads[2][0, 0, 0].item()
        self.blocks_since_hard_reset += 1
        return fragment, vad_prob


# ---------------------------------------------------------------------------
# TTS pipeline (streaming generation, frame-by-frame output)
# ---------------------------------------------------------------------------

class TextToSpeech:
    def __init__(self, config: dict[str, str]):
        repo = config.get("TTS_REPO", DEFAULT_DSM_TTS_REPO).strip()
        voice_repo = config.get("TTS_VOICE_REPO", DEFAULT_DSM_TTS_VOICE_REPO).strip()
        self.voice_name = config.get("TTS_VOICE", "expresso/ex03-ex01_happy_001_channel1_334s.wav").strip()
        quantize = int(config.get("TTS_QUANTIZE", "8") or 0)

        log("TTS: downloading/loading config")
        raw_config_path = hf_get("config.json", repo)
        with open(raw_config_path) as fobj:
            raw_config = json.load(fobj)
        mimi_weights = hf_get(raw_config["mimi_name"], repo)
        moshi_weights = hf_get(raw_config.get("moshi_name", "model.safetensors"), repo)
        tokenizer_name = hf_get(raw_config["tokenizer_name"], repo)

        lm_config = LmConfig.from_config_dict(raw_config)
        lm_config.transformer.max_seq_len = lm_config.transformer.context
        self.lm = Lm(lm_config)
        self.lm.set_dtype(mx.bfloat16)
        log("TTS: loading LM weights")
        self.lm.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        if quantize:
            log(f"TTS: quantizing to {quantize} bits")
            nn.quantize(self.lm.depformer, bits=quantize)
            for layer in self.lm.transformer.layers:
                nn.quantize(layer.self_attn, bits=quantize)
                nn.quantize(layer.gating, bits=quantize)

        self.tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_name))
        log("TTS: loading audio decoder")
        self.mimi = Mimi(models.mimi_202407(lm_config.generated_codebooks))
        self.mimi.load_pytorch_weights(str(mimi_weights), strict=True)

        self.model = TTSModel(
            self.lm,
            self.mimi,
            self.tokenizer,
            voice_repo=voice_repo,
            temp=0.6,
            cfg_coef=1.0,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            padding_bonus=0,
            raw_config=raw_config,
        )
        if self.model.valid_cfg_conditionings:
            self.cfg_coef_conditioning = self.model.cfg_coef
            self.model.cfg_coef = 1.0
            self.cfg_is_no_text = False
            self.cfg_is_no_prefix = False
        else:
            self.cfg_coef_conditioning = None
            self.cfg_is_no_text = True
            self.cfg_is_no_prefix = True

        self.voice_path = self.model.get_voice_path(self.voice_name)

    def synthesize(self, text: str, on_pcm, should_cancel=None):
        entries = self.model.prepare_script([text])
        voices = [self.voice_path] if self.model.multi_speaker else []
        attributes = self.model.make_condition_attributes(voices, self.cfg_coef_conditioning)

        def on_frame(frame):
            if should_cancel is not None and should_cancel():
                raise TurnCancelled()
            if (frame == -1).any():
                return
            pcm = self.model.mimi.decode_step(frame[:, :, None])
            pcm = np.array(mx.clip(pcm[0, 0], -1, 1))
            on_pcm(pcm.tobytes())

        self.model.generate(
            [entries],
            [attributes],
            cfg_is_no_prefix=self.cfg_is_no_prefix,
            cfg_is_no_text=self.cfg_is_no_text,
            on_frame=on_frame,
        )


# ---------------------------------------------------------------------------
# Agent: WebSocket orchestration (all async, MLX calls offloaded to threads)
# ---------------------------------------------------------------------------

class Agent:
    active_session_id: str | None = None
    next_turn_id = 0
    def __init__(self, config: dict[str, str], websocket):
        self.config = config
        self.ws = websocket
        self.session_id = str(uuid.uuid4())
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.current_turn: asyncio.Task | None = None
        self.current_turn_id: int | None = None
        self.epoch = 0
        self.loop: asyncio.AbstractEventLoop | None = None
        self.playback_active = False
        self.synthesizing = False
        # Barge-in state: played-PCM RMS history and estimated echo coupling.
        self.played_rms: deque[tuple[float, float]] = deque(maxlen=64)
        self.played_now = 0.0
        self.coupling_k: float | None = None
        self.coupling_samples: list[float] = []
        self.barge_run = 0
        self.cancel_flag = False
        self.playback_started = 0.0
        self.last_k_update = 0.0
        self.last_rms_log = 0.0
        self.last_vad_state = None
        self.aec_enabled = False
        # Audio/text accounting (per turn + per session).
        self.turn_audio_bytes = 0
        self.turn_audio_seconds = 0.0
        self.session_audio_seconds = 0.0
        self.session_text_chars = 0
        self.endpoint = EndpointDetector(
            base_energy_threshold=float(config.get("VAD_THRESHOLD", "0.01")),
            min_speech_ms=int(config.get("VAD_MIN_SPEECH_MS", "320")),
            min_silence_ms=int(config.get("VAD_MIN_SILENCE_MS", "320")),
            semantic_end_threshold=float(config.get("VAD_SEMANTIC_THRESHOLD", "0.6")),
            semantic_speech_threshold=float(config.get("VAD_SEMANTIC_SPEECH_THRESHOLD", "0.4")),
            warmup_blocks=int(config.get("VAD_WARMUP_BLOCKS", "12")),
            semantic_hold_blocks=int(config.get("VAD_SEMANTIC_HOLD_BLOCKS", "2")),
            noise_multiplier=float(config.get("VAD_NOISE_MULTIPLIER", "3.0")),
        )
        self.stt_error_streak = 0
        self.stt_suppressed_errors = 0
        self.stt_last_error_log = 0.0

    async def send_event(self, **payload):
        payload.setdefault("timestamp", time.time())
        await self.ws.send(json.dumps(payload))

    async def _mlx(self, fn, *args):
        """Run an MLX/Metal call on the single shared worker thread.

        MLX is not thread-safe, so every inference call must be serialized
        through one thread to avoid concurrent Metal command-encoder state.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(MLX_EXECUTOR, fn, *args)

    async def out_writer(self):
        while True:
            epoch, pcm = await self.tts_queue.get()
            if epoch != self.epoch:
                continue
            samples = np.frombuffer(pcm, dtype=np.float32)
            if samples.size:
                rms = float(np.sqrt(np.mean(samples * samples)))
                self.played_rms.append((time.time(), rms))
                # Decaying played-energy: covers echo tails during generation
                # pauses (moshi-mlx emits frames in bursts). Slow decay so the
                # echo tail (longer than one 80 ms block) is still covered.
                self.played_now = max(rms, self.played_now * 0.7)
                self.turn_audio_bytes += len(pcm)
                self.turn_audio_seconds += samples.size / SAMPLE_RATE
            await self.ws.send(pcm)

    def _played_energy(self) -> float:
        """Current decayed played-RMS (echo tail aware)."""
        return self.played_now

    def _played_rms_delayed(self, delay: float = 0.08) -> float:
        """Played RMS from ~`delay` seconds ago (echo arrives at the mic late)."""
        target = time.time() - delay
        best = 0.0
        for t, r in self.played_rms:
            if t <= target:
                best = r
            else:
                break
        return best

    async def _cancel_current_turn(self, reason: str, cancelled_by: str, reset_stt: bool = False):
        """Cancel assistant generation/playback with an explicit audit reason."""
        turn_id = self.current_turn_id
        had_active_turn = self.playback_active or (self.current_turn is not None and not self.current_turn.done())
        if not had_active_turn:
            return
        self.cancel_flag = True
        self.epoch += 1
        self.playback_active = False
        self.synthesizing = False
        if self.current_turn and not self.current_turn.done():
            self.current_turn.cancel()
        await self.send_event(
            type="interrupted",
            turn_id=turn_id,
            owner="assistant",
            cancelled_by=cancelled_by,
            reason=reason,
        )
        log(
            f"turn={turn_id or '-'} owner=assistant state=cancelled "
            f"by={cancelled_by} reason={reason} "
            f"audio_seconds={self.turn_audio_seconds:.1f} audio_bytes={self.turn_audio_bytes}"
        )
        if reset_stt:
            await self._mlx(Agent.stt.reset)
        self.current_turn_id = None

    async def _barge_in(self):
        """Cancel the assistant because VAD detected user speech over playback."""
        await self._cancel_current_turn("vad_barge_in", "user", reset_stt=True)
        debug("barge-in: assistant turn cancelled, STT reset")

    async def stt_loop(self):
        """Consume mic PCM, detect end-of-turn, and handle barge-in.

        During synthesis, mic energy is compared with played audio because STT
        and TTS share one Metal lane. During the playback tail, native AEC makes
        it safe to resume STT and recognized words can interrupt immediately.
        """
        stt = Agent.stt
        fragments: list[str] = []
        grace_until = 0.0
        while True:
            if self.session_id != Agent.active_session_id:
                return
            try:
                pcm = await asyncio.wait_for(self.mic_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            rms = 0.0
            if pcm:
                samples = np.frombuffer(pcm, dtype=np.float32)
                if samples.size:
                    rms = float(np.sqrt(np.mean(samples * samples)))

            # --- Barge-in detection while the assistant is playing ----------
            # If STT already decoded the start of a new user utterance while
            # the LLM was responding, never let newly-started playback cover it.
            if self.playback_active and fragments:
                await self._cancel_current_turn(
                    "recognized_speech_before_playback", "user", reset_stt=False
                )
            if self.playback_active and (self.synthesizing or not self.aec_enabled):
                played = self._played_energy()
                elapsed = time.time() - self.playback_started
                # Only arm barge-in while TTS is actively synthesizing. Once
                # synthesis finishes, the remaining playback is just the drain
                # of already-buffered audio; its echo tail must not trigger a
                # false barge-in.
                if not self.synthesizing:
                    continue
                # Estimate echo coupling k = mic_rms / played_rms(delayed). The
                # mic hears the echo ~80 ms AFTER the speaker plays it, so we
                # divide by the delayed played RMS; using the instantaneous value
                # drags the estimate down during the attack transient.
                played_delayed = self._played_rms_delayed(0.08)
                if played_delayed > 0.005 and self.coupling_k is None and elapsed > 0.16:
                    self.coupling_samples.append(rms / played_delayed)
                    if len(self.coupling_samples) >= 12:
                        self.coupling_samples.sort()
                        # 90th percentile: echo is variable, worst case matters.
                        idx = int(len(self.coupling_samples) * 0.9)
                        self.coupling_k = max(0.1, self.coupling_samples[idx])
                        self.last_k_update = time.time()
                        debug(f"barge-in: coupling k={self.coupling_k:.3f}")
                # Slow continuous re-estimation: when the user is likely silent
                # (mic below threshold) for a while, nudge k toward the current
                # delayed ratio so it adapts to volume changes.
                elif (
                    self.coupling_k is not None
                    and played_delayed > 0.005
                    and time.time() - self.last_k_update > 1.0
                ):
                    ratio = rms / played_delayed
                    if ratio < self.coupling_k * 1.5:
                        self.coupling_k = max(0.1, self.coupling_k * 0.9 + ratio * 0.1)
                        self.last_k_update = time.time()
                threshold = 0.05
                if self.coupling_k is not None:
                    threshold = max(0.05, self.coupling_k * played * 3.0)
                # Blind period: never trigger barge-in in the first 1.0 s of
                # playback (needed to learn the echo level).
                if elapsed >= 1.0:
                    if rms > threshold:
                        self.barge_run += 1
                        if self.barge_run >= 3:
                            await self._barge_in()
                            self.barge_run = 0
                            grace_until = time.time() + 0.3
                    else:
                        self.barge_run = 0
                if DEBUG and self.barge_run > 0:
                    debug(f"barge-in: mic={rms:.4f} played={played:.4f} k={self.coupling_k} thr={threshold:.4f} run={self.barge_run}")
                continue

            # --- Grace period after barge-in (let echo tail die) ------------
            if time.time() < grace_until:
                continue

            if DEBUG and time.time() - self.last_rms_log >= 1.0:
                self.last_rms_log = time.time()
                debug(
                    f"vad: state={self.endpoint.state.value} rms={rms:.4f} "
                    f"noise_floor={self.endpoint.noise_floor:.4f} "
                    f"energy_threshold={self.endpoint.energy_threshold:.4f} "
                    f"silence_run={self.endpoint.silence_run} fragments={len(fragments)} "
                    f"aec={self.aec_enabled} playback={self.playback_active}"
                )

            try:
                fragment, vad_probability = await self._mlx(stt.step, pcm)
            except Exception as error:
                self.stt_error_streak += 1
                _READY["stt_ready"] = False
                _READY["stt_error"] = str(error)
                _READY["stt_error_count"] += 1
                now = time.time()
                if self.stt_last_error_log == 0.0 or now - self.stt_last_error_log >= 10.0:
                    suffix = f" ({self.stt_suppressed_errors} repeated errors suppressed)" if self.stt_suppressed_errors else ""
                    log(f"STT step error: {error}{suffix}; performing hard recovery")
                    self.stt_last_error_log = now
                    self.stt_suppressed_errors = 0
                else:
                    self.stt_suppressed_errors += 1
                try:
                    await self._mlx(stt.reset, True)
                    _READY["stt_last_recovery"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                except Exception as reset_error:
                    _READY["stt_error"] = f"Recovery failed: {reset_error}"
                    log(f"STT hard recovery failed: {reset_error}")
                    await asyncio.sleep(1.0)
                fragments.clear()
                self.endpoint.reset()
                continue
            if self.stt_error_streak:
                log(f"STT recovered after {self.stt_error_streak} failed block(s)")
                self.stt_error_streak = 0
                self.stt_suppressed_errors = 0
                self.stt_last_error_log = 0.0
                _READY["stt_error"] = None
                _READY["stt_ready"] = True
            if fragment:
                fragments.append(fragment)
                partial = "".join(fragments).strip()
                await self.send_event(type="partial", text=partial)
                if partial and self.playback_active and self.aec_enabled:
                    await self._cancel_current_turn(
                        "recognized_speech_barge_in", "user", reset_stt=False
                    )

            has_text = bool("".join(fragments).strip())
            decision = self.endpoint.observe(
                rms=rms,
                semantic_probability=vad_probability,
                has_recognized_text=has_text,
            )
            if self.endpoint.state != self.last_vad_state:
                debug(
                    f"vad: transition={self.last_vad_state or '-'}->{self.endpoint.state.value} "
                    f"text_armed={self.endpoint.armed} rms={rms:.4f} "
                    f"raw={vad_probability:.3f} smooth={self.endpoint.smoothed_probability:.3f}"
                )
                self.last_vad_state = self.endpoint.state
            # The STT LmGen has a finite step window (~8192 blocks); it steps on
            # idle silence too, so reset it during long silences to avoid the
            # "narrow invalid args" overflow that would otherwise stall listening
            # after ~11 minutes.
            if stt.should_rotate() and not fragments:
                await self._mlx(stt.reset, True)
                log("STT context rotated before positional limit")
                self.endpoint.reset()
            elif self.endpoint.blocks_seen >= 120 and not fragments:
                await self._mlx(stt.reset)
                self.endpoint.reset()
            if decision:
                transcript = await self._finalize(stt, fragments)
                fragments = []
                self.endpoint.reset()
                while not self.mic_queue.empty():
                    self.mic_queue.get_nowait()
                if not transcript:
                    log(
                        "turn=- owner=user state=discarded reason=empty_transcript "
                        f"endpoint={decision.reason} speech_ms={decision.speech_ms} "
                        f"silence_ms={decision.silence_ms} "
                        f"vad_raw={decision.semantic_probability:.3f} "
                        f"vad_smooth={decision.smoothed_probability:.3f} "
                        f"energy_threshold={decision.energy_threshold:.4f}"
                    )
                    continue
                # If a turn is still running (e.g. LLM still streaming), cancel
                # it and start the new one with the fresh transcript.
                if self.current_turn is not None and not self.current_turn.done():
                    await self._cancel_current_turn("new_user_turn_detected", "user")
                Agent.next_turn_id += 1
                turn_id = Agent.next_turn_id
                self.current_turn_id = turn_id
                log(
                    f"turn={turn_id} owner=user state=completed endpoint={decision.reason} "
                    f"speech_ms={decision.speech_ms} silence_ms={decision.silence_ms} "
                    f"vad_raw={decision.semantic_probability:.3f} "
                    f"vad_smooth={decision.smoothed_probability:.3f} "
                    f"energy_threshold={decision.energy_threshold:.4f} transcript={transcript!r}"
                )
                self.current_turn = asyncio.create_task(
                    self.run_turn(transcript, turn_id, decision.reason)
                )

    async def _finalize(self, stt, fragments: list[str]) -> str:
        """Feed trailing silence to flush the delayed STT output, then reset."""
        for _ in range(FLUSH_BLOCKS):
            fragment, _ = await self._mlx(stt.step, _SILENCE)
            if fragment:
                fragments.append(fragment)
        await self._mlx(stt.reset)
        return "".join(fragments).strip()

    async def run_turn(self, transcript: str, turn_id: int, endpoint_reason: str):
        self.epoch += 1
        epoch = self.epoch
        self.cancel_flag = False
        self.coupling_k = None
        self.coupling_samples = []
        self.barge_run = 0
        self.played_now = 0.0
        self.last_k_update = 0.0
        self.turn_audio_bytes = 0
        self.turn_audio_seconds = 0.0
        await self.send_event(
            type="transcript", text=transcript, turn_id=turn_id,
            owner="user", endpoint=endpoint_reason,
        )
        log(f"turn={turn_id} owner=assistant state=started reason=user_turn_completed")
        await self.send_event(type="turn_started", turn_id=turn_id, owner="assistant")
        reply = ""
        try:
            async for delta in self._llm_stream_async(transcript):
                reply += delta
                await self.send_event(type="text", delta=delta)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log(f"turn={turn_id} owner=assistant state=failed reason=llm_error error={error}")
            await self.send_event(type="error", message=str(error), turn_id=turn_id, owner="assistant")
            return
        if not reply.strip():
            log(f"turn={turn_id} owner=assistant state=completed reason=empty_llm_reply")
            await self.send_event(type="done", turn_id=turn_id, owner="assistant")
            return
        log(f"turn={turn_id} owner=assistant state=reply_ready text_chars={len(reply)}")
        self.playback_active = True
        self.synthesizing = True
        self.playback_started = time.time()
        log(f"turn={turn_id} owner=assistant state=synthesizing text_chars={len(reply)}")
        try:
            await self._mlx(self._tts_synthesize, reply, epoch)
        except asyncio.CancelledError:
            raise
        except TurnCancelled:
            debug(f"turn={turn_id}: TTS observed cancellation flag")
            return
        except Exception as error:
            log(f"turn={turn_id} owner=assistant state=failed reason=tts_error error={error}")
            await self.send_event(
                type="error", message=f"TTS: {error}", turn_id=turn_id, owner="assistant"
            )
        self.synthesizing = False
        HISTORY.setdefault(self.session_id, []).extend(
            [{"role": "user", "content": transcript}, {"role": "assistant", "content": reply}]
        )
        HISTORY[self.session_id] = HISTORY[self.session_id][-12:]
        self.session_audio_seconds += self.turn_audio_seconds
        self.session_text_chars += len(reply)
        log(
            f"turn={turn_id} owner=assistant state=audio_ready reply_chars={len(reply)} "
            f"audio_seconds={self.turn_audio_seconds:.1f} audio_bytes={self.turn_audio_bytes} "
            f"session_audio_seconds={self.session_audio_seconds:.1f} "
            f"session_text_chars={self.session_text_chars}"
        )
        await self.send_event(type="done", turn_id=turn_id, owner="assistant")

    async def _llm_stream_async(self, transcript: str):
        """Bridge the synchronous `stream_chat` generator into async deltas."""
        loop = asyncio.get_running_loop()
        llm_queue: asyncio.Queue = asyncio.Queue()

        def pump():
            try:
                for delta in stream_chat(transcript, self.session_id, self.config):
                    loop.call_soon_threadsafe(llm_queue.put_nowait, ("delta", delta))
            except Exception as error:
                loop.call_soon_threadsafe(llm_queue.put_nowait, ("error", str(error)))
            finally:
                loop.call_soon_threadsafe(llm_queue.put_nowait, (None, None))

        asyncio.get_running_loop().run_in_executor(None, pump)
        while True:
            kind, value = await llm_queue.get()
            if kind is None:
                return
            if kind == "error":
                raise RuntimeError(value)
            yield value

    def _tts_synthesize(self, reply: str, epoch: int):
        Agent.tts.synthesize(reply, self._emit_pcm(epoch), should_cancel=lambda: self.cancel_flag)

    def _emit_pcm(self, epoch: int):
        loop = self.loop

        def emit(pcm: bytes):
            loop.call_soon_threadsafe(self.tts_queue.put_nowait, (epoch, pcm))
        return emit

    async def run(self):
        self.loop = asyncio.get_running_loop()
        Agent.active_session_id = self.session_id
        stt_task = asyncio.create_task(self.stt_loop())
        writer_task = asyncio.create_task(self.out_writer())
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    if self.session_id != Agent.active_session_id:
                        continue
                    if self.mic_queue.full():
                        self.mic_queue.get_nowait()
                    self.mic_queue.put_nowait(message)
                else:
                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "interrupt":
                        await self._cancel_current_turn("manual_interrupt", "user", reset_stt=True)
                        self.endpoint.reset()
                    elif event.get("type") == "audio_processing":
                        self.aec_enabled = bool(event.get("voice_processing"))
                        log(
                            "audio input voice_processing="
                            f"{'enabled' if self.aec_enabled else 'unavailable'}"
                        )
                    elif event.get("type") == "playback_done":
                        if self.current_turn_id is not None:
                            log(
                                f"turn={self.current_turn_id} owner=assistant "
                                "state=completed reason=playback_finished"
                            )
                            self.current_turn_id = None
                        self.playback_active = False
                        self.cancel_flag = False
                        await self._mlx(Agent.stt.reset)
                        self.endpoint.reset()
        except websockets.ConnectionClosed:
            pass
        finally:
            stt_task.cancel()
            writer_task.cancel()
            if self.current_turn and not self.current_turn.done():
                self.current_turn.cancel()
                reason = (
                    "session_superseded"
                    if Agent.active_session_id not in (None, self.session_id)
                    else "client_disconnected"
                )
                log(
                    f"turn={self.current_turn_id or '-'} owner=assistant "
                    f"state=cancelled by=system reason={reason}"
                )
            if Agent.active_session_id == self.session_id:
                Agent.active_session_id = None
            HISTORY.pop(self.session_id, None)


def main():
    global DEBUG
    config = load_config()
    DEBUG = config.get("DEBUG", "").strip() in ("1", "true", "yes", "on")
    _READY["config"] = config
    port = int(config.get("AGENT_PORT", "8999"))
    ws_port = port + 1

    log("Loading Kyutai models (first run downloads weights)…")
    if DEBUG:
        log("debug logging enabled")
    vad_repo = config.get("VAD_REPO", config.get("STT_REPO", "")).strip()
    stt_repo = config.get("STT_REPO", "kyutai/stt-1b-en_fr-candle").strip()
    if vad_repo and vad_repo != stt_repo:
        log(f"VAD model {vad_repo!r} selected; using the semantic VAD head bundled with {stt_repo!r} until a matching adapter is installed")
    Agent.stt = SpeechToText(config)
    _READY["stt_ready"] = True
    log("STT ready")
    Agent.tts = TextToSpeech(config)
    _READY["tts_ready"] = True
    log("TTS ready")

    threading.Thread(
        target=lambda: ThreadingHTTPServer(("127.0.0.1", port), HealthHandler).serve_forever(),
        daemon=True,
    ).start()
    log(f"Health on http://127.0.0.1:{port}/health · WebSocket on ws://127.0.0.1:{ws_port}")

    async def serve():
        async with websockets.serve(lambda ws: Agent(config, ws).run(), "127.0.0.1", ws_port):
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
