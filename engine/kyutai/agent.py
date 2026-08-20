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

DEBUG = False


def log(message: str) -> None:
    """Write a line to the agent log file and best-effort stdout.

    Never raises: a broken stdout pipe (e.g. the parent app quit) must not
    crash request handling or model loading.
    """
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n"
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as fobj:
            fobj.write(line)
    except OSError:
        pass
    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError):
        pass


def debug(message: str) -> None:
    """Log a line only when DEBUG is enabled (set DEBUG=1 in config/local.env)."""
    if DEBUG:
        log(f"[debug] {message}")


def load_config() -> dict[str, str]:
    values = dict(os.environ)
    if CONFIG.exists():
        for raw in CONFIG.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


# ---------------------------------------------------------------------------
# LLM streaming (unchanged OpenAI-compatible logic from the previous agent)
# ---------------------------------------------------------------------------

def complete_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


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


class HealthHandler(BaseHTTPRequestHandler):
    server_version = "LocalVoiceAgent/0.2"

    def log_message(self, format: str, *args: object) -> None:
        # File-backed and exception-safe so a broken stdout never aborts a
        # response (send_response() logs before writing the body).
        log(format % args)

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
        self.audio_tokenizer = rustymimi.Tokenizer(str(mimi_weights), num_codebooks=num_codebooks)
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

    def reset(self):
        self.gen = self._new_gen()
        self.buffer = bytearray()

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
    def __init__(self, config: dict[str, str], websocket):
        self.config = config
        self.ws = websocket
        self.session_id = "default"
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.current_turn: asyncio.Task | None = None
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
        # Audio/text accounting (per turn + per session).
        self.turn_audio_bytes = 0
        self.turn_audio_seconds = 0.0
        self.session_audio_seconds = 0.0
        self.session_text_chars = 0

    async def send_event(self, **payload):
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

    async def _barge_in(self):
        """Cancel the current turn and notify the client the user interrupted."""
        self.cancel_flag = True
        self.epoch += 1
        self.playback_active = False
        if self.current_turn and not self.current_turn.done():
            self.current_turn.cancel()
        await self.send_event(type="interrupted")
        await self._mlx(Agent.stt.reset)
        log(f"turn audio: {self.turn_audio_seconds:.1f} s ({self.turn_audio_bytes} bytes) — cancelled")
        debug("barge-in: turn cancelled, STT reset")

    async def stt_loop(self):
        """Consume mic PCM, detect end-of-turn, and handle barge-in.

        While the assistant is playing, mic audio is not transcribed; instead it
        is compared against the played audio to detect the user talking over the
        assistant (barge-in). When idle, energy + hangover detects end-of-turn.
        """
        stt = Agent.stt
        fragments: list[str] = []
        speech_run = 0
        silence_run = 0
        grace_until = 0.0
        while True:
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
            if self.playback_active:
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

            try:
                fragment, _vad = await self._mlx(stt.step, pcm)
            except Exception as error:
                log(f"STT step error: {error}")
                await self._mlx(stt.reset)
                fragments.clear()
                speech_run = 0
                silence_run = 0
                continue
            if fragment:
                fragments.append(fragment)
                partial = "".join(fragments).strip()
                await self.send_event(type="partial", text=partial)
            # Energy gate: track consecutive blocks above/below the speech level.
            if rms >= 0.01:
                speech_run += 1
                silence_run = 0
            else:
                silence_run += 1
            if speech_run >= 4 and silence_run >= 4:
                transcript = await self._finalize(stt, fragments)
                fragments = []
                speech_run = 0
                silence_run = 0
                while not self.mic_queue.empty():
                    self.mic_queue.get_nowait()
                if not transcript:
                    continue
                # If a turn is still running (e.g. LLM still streaming), cancel
                # it and start the new one with the fresh transcript.
                if self.current_turn is not None and not self.current_turn.done():
                    self.cancel_flag = True
                    self.epoch += 1
                    self.current_turn.cancel()
                    await self.send_event(type="interrupted")
                    debug("stt: superseded running turn with new transcript")
                self.current_turn = asyncio.create_task(self.run_turn(transcript))

    async def _finalize(self, stt, fragments: list[str]) -> str:
        """Feed trailing silence to flush the delayed STT output, then reset."""
        for _ in range(FLUSH_BLOCKS):
            fragment, _ = await self._mlx(stt.step, _SILENCE)
            if fragment:
                fragments.append(fragment)
        await self._mlx(stt.reset)
        return "".join(fragments).strip()

    async def run_turn(self, transcript: str):
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
        log(f"turn transcript: {transcript!r}")
        await self.send_event(type="transcript", text=transcript)
        await self.send_event(type="turn_started")
        reply = ""
        try:
            async for delta in self._llm_stream_async(transcript):
                reply += delta
                await self.send_event(type="text", delta=delta)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.send_event(type="error", message=str(error))
            return
        if not reply.strip():
            await self.send_event(type="done")
            return
        log(f"turn text: {len(reply)} chars")
        self.playback_active = True
        self.synthesizing = True
        self.playback_started = time.time()
        log(f"turn flushed: {len(reply)} chars → TTS")
        try:
            await self._mlx(self._tts_synthesize, reply, epoch)
        except asyncio.CancelledError:
            raise
        except TurnCancelled:
            debug("turn cancelled during TTS")
            return
        except Exception as error:
            await self.send_event(type="error", message=f"TTS: {error}")
        self.synthesizing = False
        HISTORY.setdefault(self.session_id, []).extend(
            [{"role": "user", "content": transcript}, {"role": "assistant", "content": reply}]
        )
        self.session_audio_seconds += self.turn_audio_seconds
        self.session_text_chars += len(reply)
        log(
            f"turn complete, reply={len(reply)} chars · "
            f"turn audio: {self.turn_audio_seconds:.1f} s ({self.turn_audio_bytes} bytes) · "
            f"session audio: {self.session_audio_seconds:.1f} s · session text: {self.session_text_chars} chars"
        )
        await self.send_event(type="done")

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
        stt_task = asyncio.create_task(self.stt_loop())
        writer_task = asyncio.create_task(self.out_writer())
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    self.mic_queue.put_nowait(message)
                else:
                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "interrupt":
                        self.epoch += 1
                        self.playback_active = False
                        self.cancel_flag = True
                        if self.current_turn and not self.current_turn.done():
                            self.current_turn.cancel()
                        await self._mlx(Agent.stt.reset)
                    elif event.get("type") == "playback_done":
                        self.playback_active = False
                        self.cancel_flag = False
                        await self._mlx(Agent.stt.reset)
        except websockets.ConnectionClosed:
            pass
        finally:
            stt_task.cancel()
            writer_task.cancel()


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
