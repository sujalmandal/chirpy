#!/usr/bin/env python3
"""Local voice agent built on Kyutai STT 1B (semantic VAD) + an
OpenAI-compatible LLM + Kyutai TTS 1.6B (MLX).

STT and semantic VAD run continuously in a dedicated MLX CPU process while TTS
runs on the GPU in the main process. The Swift client streams microphone audio
over WebSocket and receives streamed text events plus decoded 24 kHz Float32
mono PCM frames for playback.

Wire protocol:
  client -> agent : binary Float32 LE mono 24 kHz PCM, 1920-sample (80 ms) blocks
  client -> agent : {"type":"interrupt"}  (barge-in)
  agent -> client : {"type":"transcript"|"turn_started"|"text"|"done"|"error", ...}
  agent -> client : binary Float32 LE mono 24 kHz PCM frames
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing
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

from endpointing import (
    BargeInGate,
    EndpointDetector,
    is_probable_playback_echo,
    recognized_barge_in_ready,
)

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


# TTS owns the main process's MLX/Metal state. STT calls use a separate executor
# only to wait on IPC from the dedicated CPU worker; they never enter Metal.
MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
STT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-ipc")

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

MICROPHONE_PAGE = b"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<body style="margin:0;background:transparent;overflow:hidden">
<script>
(() => {
  const bridge = window.webkit?.messageHandlers?.microphoneBridge;
  let stream, context, source, processor, mute, pending = [];
  const playbackSources = new Set();
  let playbackEpoch = 0;
  let nextPlaybackTime = 0;
  const targetRate = 24000;
  const targetBlock = 1920;

  function post(message) { bridge?.postMessage(message); }
  function encodeBlock(samples) {
    const floats = new Float32Array(samples);
    const bytes = new Uint8Array(floats.buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
  }
  function resample(input, fromRate) {
    if (fromRate === targetRate) return Array.from(input);
    const ratio = fromRate / targetRate;
    const count = Math.floor(input.length / ratio);
    const output = new Array(count);
    for (let i = 0; i < count; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.max(start + 1, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = start; j < Math.min(end, input.length); j++) sum += input[j];
      output[i] = sum / Math.max(1, Math.min(end, input.length) - start);
    }
    return output;
  }
  async function stopCapture() {
    stopPlayback();
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (mute) mute.disconnect();
    if (stream) stream.getTracks().forEach(track => track.stop());
    if (context && context.state !== 'closed') await context.close();
    stream = context = source = processor = mute = null;
    pending = [];
  }
  function decodeBlock(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  function playPCM(encoded) {
    if (!context || context.state === 'closed') return false;
    const samples = decodeBlock(encoded);
    const buffer = context.createBuffer(1, samples.length, targetRate);
    buffer.copyToChannel(samples, 0);
    const node = context.createBufferSource();
    const epoch = playbackEpoch;
    let power = 0;
    for (const sample of samples) power += sample * sample;
    const rms = Math.sqrt(power / Math.max(1, samples.length));
    node.buffer = buffer;
    node.connect(context.destination);
    nextPlaybackTime = Math.max(nextPlaybackTime, context.currentTime + 0.025);
    const scheduledStart = nextPlaybackTime;
    node.start(scheduledStart);
    nextPlaybackTime += buffer.duration;
    const referenceDelay = Math.max(0, (scheduledStart - context.currentTime) * 1000);
    setTimeout(() => {
      if (epoch === playbackEpoch) post({ type: 'playback_reference', rms });
    }, referenceDelay);
    playbackSources.add(node);
    node.onended = () => {
      playbackSources.delete(node);
      if (epoch === playbackEpoch) post({ type: 'playback_block_done' });
    };
    return true;
  }
  function stopPlayback() {
    playbackEpoch++;
    for (const node of playbackSources) {
      try { node.stop(); } catch (_) {}
    }
    playbackSources.clear();
    nextPlaybackTime = context ? context.currentTime : 0;
  }
  async function startCapture() {
    try {
      await stopCapture();
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: { ideal: true },
          noiseSuppression: { ideal: true },
          autoGainControl: { ideal: true },
          channelCount: { ideal: 1 }
        },
        video: false
      });
      context = new AudioContext({ latencyHint: 'interactive' });
      await context.resume();
      source = context.createMediaStreamSource(stream);
      processor = context.createScriptProcessor(2048, 1, 1);
      mute = context.createGain();
      mute.gain.value = 0;
      processor.onaudioprocess = event => {
        pending.push(...resample(event.inputBuffer.getChannelData(0), context.sampleRate));
        while (pending.length >= targetBlock) {
          const block = pending.splice(0, targetBlock);
          post({ type: 'audio', pcm: encodeBlock(block) });
        }
      };
      source.connect(processor);
      processor.connect(mute);
      mute.connect(context.destination);
      const settings = stream.getAudioTracks()[0]?.getSettings?.() || {};
      post({
        type: 'ready',
        echoCancellation: settings.echoCancellation === true,
        noiseSuppression: settings.noiseSuppression === true,
        autoGainControl: settings.autoGainControl === true,
        sampleRate: settings.sampleRate || context.sampleRate
      });
    } catch (error) {
      post({ type: 'error', message: `${error.name || 'Error'}: ${error.message || error}` });
    }
  }
  window.startCapture = startCapture;
  window.stopCapture = stopCapture;
  window.playPCM = playPCM;
  window.stopPlayback = stopPlayback;
  post({ type: 'page_ready' });
})();
</script>
"""


class HealthHandler(BaseHTTPRequestHandler):
    server_version = "LocalVoiceAgent/0.2"

    def log_message(self, format: str, *args: object) -> None:
        # Health is polled continuously by the macOS app. Logging every request
        # would add thousands of low-value lines per hour.
        return

    def do_GET(self) -> None:
        if self.path.startswith("/microphone"):
            log("web microphone page served")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(MICROPHONE_PAGE)))
            self.end_headers()
            self.wfile.write(MICROPHONE_PAGE)
            return
        if self.path != "/health":
            self.send_error(404)
            return
        cfg = _READY["config"]
        stt = getattr(Agent, "stt", None)
        stt_alive = bool(stt and stt.is_alive())
        payload = {
            "status": "ok",
            "service": "kyutai-voice-agent",
            "stt_ready": _READY["stt_ready"] and stt_alive,
            "tts_ready": _READY["tts_ready"],
            "stt_device": "cpu",
            "stt_continuous": True,
            "stt_last_step_ms": round(getattr(stt, "last_step_ms", 0.0), 1),
            "stt_max_step_ms": round(getattr(stt, "max_step_ms", 0.0), 1),
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
        quantize = int(config.get("STT_QUANTIZE", "0") or 0)
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
        if quantize:
            group_size = 32 if quantize == 4 else 64
            log(f"STT: quantizing CPU model to {quantize} bits")
            nn.quantize(self.lm, bits=quantize, group_size=group_size)

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


def _stt_worker_main(config: dict[str, str], connection) -> None:
    """Own the Kyutai STT model and all of its streaming state on MLX CPU."""
    try:
        mx.set_default_device(mx.cpu)
        started = time.monotonic()
        log("STT worker: loading continuously on MLX CPU")
        worker_config = dict(config)
        worker_config["STT_QUANTIZE"] = config.get("STT_CPU_QUANTIZE", "0")
        stt = SpeechToText(worker_config)
        connection.send(("ready", {"load_seconds": time.monotonic() - started}))
    except BaseException as error:
        try:
            connection.send(("startup_error", repr(error)))
        finally:
            connection.close()
        return

    try:
        while True:
            request = connection.recv()
            operation = request[0]
            if operation == "close":
                return
            try:
                if operation == "step":
                    result = stt.step(request[1])
                elif operation == "reset":
                    result = stt.reset(bool(request[1]))
                else:
                    raise ValueError(f"Unknown STT worker operation: {operation}")
                connection.send(("ok", result))
            except BaseException as error:
                connection.send(("error", repr(error)))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class SpeechToTextWorker:
    """Synchronous IPC facade used from the agent's dedicated STT executor."""

    def __init__(self, config: dict[str, str]):
        self.rotate_blocks = max(500, int(config.get("STT_ROTATE_BLOCKS", "3000")))
        self.blocks_since_hard_reset = 0
        self.lock = threading.Lock()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.last_step_ms = 0.0
        self.max_step_ms = 0.0
        self.process = context.Process(
            target=_stt_worker_main,
            args=(config, child),
            name="kyutai-stt-cpu",
            daemon=True,
        )
        self.process.start()
        child.close()
        timeout = float(config.get("STT_CPU_START_TIMEOUT_SECONDS", "180"))
        if not self.connection.poll(timeout):
            self.close(force=True)
            raise RuntimeError(f"CPU STT worker did not become ready within {timeout:.0f}s")
        try:
            status, payload = self.connection.recv()
        except EOFError as error:
            self.close(force=True)
            raise RuntimeError("CPU STT worker exited during startup") from error
        if status != "ready":
            self.close(force=True)
            raise RuntimeError(f"CPU STT worker failed to start: {payload}")
        log(
            "STT worker ready device=cpu "
            f"load_seconds={float(payload['load_seconds']):.1f} pid={self.process.pid}"
        )

    def _request(self, operation: str, payload=None):
        with self.lock:
            if not self.process.is_alive():
                raise RuntimeError("CPU STT worker stopped unexpectedly")
            self.connection.send((operation, payload))
            if not self.connection.poll(30.0):
                raise TimeoutError(f"CPU STT worker timed out during {operation}")
            status, result = self.connection.recv()
            if status != "ok":
                raise RuntimeError(f"CPU STT worker {operation} failed: {result}")
            return result

    def step(self, pcm_float32: bytes) -> tuple[str | None, float]:
        started = time.monotonic()
        result = self._request("step", pcm_float32)
        self.last_step_ms = (time.monotonic() - started) * 1000.0
        self.max_step_ms = max(self.max_step_ms, self.last_step_ms)
        self.blocks_since_hard_reset += 1
        return result

    def reset(self, hard: bool = False) -> None:
        self._request("reset", hard)
        if hard:
            self.blocks_since_hard_reset = 0

    def should_rotate(self) -> bool:
        return self.blocks_since_hard_reset >= self.rotate_blocks

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def close(self, force: bool = False) -> None:
        try:
            if not force and self.process.is_alive():
                self.connection.send(("close", None))
                self.process.join(timeout=2.0)
        except (BrokenPipeError, EOFError, OSError):
            pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)
        self.connection.close()


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
        # CPU STT can occasionally run slower than the 80 ms capture cadence.
        # Keep only a short live window rather than letting several seconds of
        # stale audio delay barge-in and end-of-turn decisions.
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=max(4, int(config.get("MIC_QUEUE_BLOCKS", "8")))
        )
        self.mic_blocks_dropped = 0
        self.last_mic_drop_log = 0.0
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
        self.last_playback_reference_at = 0.0
        self.playback_reference_logged = False
        self.coupling_k: float | None = None
        self.coupling_samples: list[float] = []
        self.barge_gate = BargeInGate(
            blind_ms=int(config.get("BARGE_IN_BLIND_MS", "1200")),
            min_speech_ms=int(config.get("BARGE_IN_MIN_SPEECH_MS", "640")),
        )
        self.barge_min_rms = float(config.get("BARGE_IN_MIN_RMS", "0.05"))
        self.barge_echo_multiplier = float(
            config.get("BARGE_IN_ECHO_MULTIPLIER", "1.6")
        )
        self.barge_energy_confirmed_until = 0.0
        self.current_reply_text = ""
        self.stt_discard_epoch = 0
        self.continuous_stt_logged = False
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
            min_silence_ms=int(config.get("VAD_MIN_SILENCE_MS", "800")),
            semantic_silence_ms=int(config.get("VAD_SEMANTIC_SILENCE_MS", "320")),
            semantic_end_threshold=float(config.get("VAD_SEMANTIC_THRESHOLD", "0.6")),
            semantic_speech_threshold=float(config.get("VAD_SEMANTIC_SPEECH_THRESHOLD", "0.4")),
            warmup_blocks=int(config.get("VAD_WARMUP_BLOCKS", "12")),
            semantic_hold_blocks=int(config.get("VAD_SEMANTIC_HOLD_BLOCKS", "3")),
            noise_multiplier=float(config.get("VAD_NOISE_MULTIPLIER", "3.0")),
        )
        self.stt_error_streak = 0
        self.stt_suppressed_errors = 0
        self.stt_last_error_log = 0.0

    async def send_event(self, **payload):
        payload.setdefault("timestamp", time.time())
        await self.ws.send(json.dumps(payload))

    async def _mlx(self, fn, *args):
        """Run a TTS MLX/Metal call on the main process's serial executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(MLX_EXECUTOR, fn, *args)

    async def _stt(self, fn, *args):
        """Call the continuous CPU STT process without blocking asyncio."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(STT_EXECUTOR, fn, *args)

    async def out_writer(self):
        while True:
            epoch, pcm = await self.tts_queue.get()
            if epoch != self.epoch:
                continue
            samples = np.frombuffer(pcm, dtype=np.float32)
            if samples.size:
                self.turn_audio_bytes += len(pcm)
                self.turn_audio_seconds += samples.size / SAMPLE_RATE
            await self.ws.send(pcm)

    def _played_energy(self) -> float:
        """RMS of audio the renderer reports as currently playing."""
        age = time.time() - self.last_playback_reference_at
        return self.played_now if age <= 0.20 else 0.0

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
        self.current_reply_text = ""
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
            await self._stt(Agent.stt.reset)
        self.current_turn_id = None

    async def stt_loop(self):
        """Consume mic PCM, detect end-of-turn, and handle barge-in.

        The dedicated CPU worker keeps Kyutai STT and semantic VAD active during
        GPU TTS. Recognized words still pass echo and residual-energy checks.
        """
        stt = Agent.stt
        fragments: list[str] = []
        discard_epoch = self.stt_discard_epoch
        while True:
            if self.session_id != Agent.active_session_id:
                return
            try:
                pcm = await asyncio.wait_for(self.mic_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if discard_epoch != self.stt_discard_epoch:
                fragments.clear()
                self.endpoint.reset()
                discard_epoch = self.stt_discard_epoch
            rms = 0.0
            if pcm:
                samples = np.frombuffer(pcm, dtype=np.float32)
                if samples.size:
                    rms = float(np.sqrt(np.mean(samples * samples)))

            # --- Barge-in detection while the assistant is playing ----------
            if self.playback_active:
                played = self._played_energy()
                elapsed = time.time() - self.playback_started
                # Text already decoded before playback began represents a real
                # user continuation, not playback echo.
                if fragments and elapsed < 0.4:
                    await self._cancel_current_turn(
                        "recognized_speech_before_playback", "user", reset_stt=False
                    )
                    continue
                # Only arm barge-in while TTS is actively synthesizing. Once
                # synthesis finishes, STT can run again, but recognized text
                # must still be backed by independent residual mic energy.
                # Estimate echo coupling k = mic_rms / played_rms(delayed). The
                # mic hears the echo ~80 ms AFTER the speaker plays it, so we
                # divide by the delayed played RMS; using the instantaneous value
                # drags the estimate down during the attack transient.
                played_delayed = self._played_rms_delayed(0.08)
                if played_delayed > 0.005 and self.coupling_k is None and elapsed > 0.16:
                    self.coupling_samples.append(rms / played_delayed)
                    if len(self.coupling_samples) >= 12:
                        self.coupling_samples.sort()
                        # Median avoids user speech during calibration inflating
                        # the echo estimate until real interruptions are ignored.
                        idx = len(self.coupling_samples) // 2
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
                threshold = self.barge_min_rms
                if self.coupling_k is not None:
                    echo_reference = max(played, played_delayed)
                    threshold = max(
                        threshold,
                        self.coupling_k * echo_reference * self.barge_echo_multiplier,
                    )
                residual_confirmed = self.barge_gate.observe(
                    elapsed_ms=elapsed * 1000.0,
                    rms=rms,
                    threshold=threshold,
                    calibrated=self.coupling_k is not None,
                )
                if residual_confirmed:
                    self.barge_energy_confirmed_until = time.time() + 1.2
                if DEBUG and self.barge_gate.run > 0:
                    debug(
                        f"barge-in: mic={rms:.4f} played={played:.4f} "
                        f"k={self.coupling_k} thr={threshold:.4f} "
                        f"run={self.barge_gate.run}/{self.barge_gate.required_blocks}"
                    )
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
                fragment, vad_probability = await self._stt(stt.step, pcm)
                if self.synthesizing and not self.continuous_stt_logged:
                    self.continuous_stt_logged = True
                    log(
                        f"turn={self.current_turn_id or '-'} owner=engine "
                        "state=continuous_stt_during_tts "
                        f"step_ms={stt.last_step_ms:.1f}"
                    )
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
                    await self._stt(stt.reset, True)
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
                if partial and self.playback_active and self.aec_enabled:
                    if is_probable_playback_echo(partial, self.current_reply_text):
                        log(
                            f"turn={self.current_turn_id or '-'} owner=user "
                            f"state=suppressed reason=playback_echo transcript={partial[:120]!r}"
                        )
                        fragments.clear()
                        await self._stt(stt.reset)
                        self.endpoint.reset()
                        continue
                    residual_confirmed = (
                        not self.playback_reference_logged
                        or time.time() <= self.barge_energy_confirmed_until
                    )
                    if recognized_barge_in_ready(
                        partial, residual_confirmed=residual_confirmed
                    ):
                        log(
                            f"turn={self.current_turn_id or '-'} owner=user "
                            "state=barge_in_detected "
                            f"evidence={'recognized_plus_residual' if residual_confirmed else 'recognized_non_echo'} "
                            f"transcript={partial[:120]!r}"
                        )
                        await self.send_event(type="partial", text=partial)
                        await self._cancel_current_turn(
                            "recognized_speech_barge_in", "user", reset_stt=False
                        )
                    else:
                        debug(f"barge-in: awaiting more speech: {partial[:80]!r}")
                        continue
                else:
                    await self.send_event(type="partial", text=partial)

            # While playback is active, retain an unconfirmed transcript
            # candidate for more words instead of endpointing it as a new turn.
            if self.playback_active and fragments:
                continue

            has_text = bool("".join(fragments).strip())
            decision = self.endpoint.observe(
                rms=rms,
                semantic_probability=vad_probability,
                has_recognized_text=has_text,
                new_recognized_text=bool(fragment and has_text),
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
                await self._stt(stt.reset, True)
                log("STT context rotated before positional limit")
                self.endpoint.reset()
            elif self.endpoint.blocks_seen >= 120 and not fragments:
                await self._stt(stt.reset)
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
                        f"energy_threshold={decision.energy_threshold:.4f} "
                        f"token_age_ms={decision.token_age_ms}"
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
                    f"energy_threshold={decision.energy_threshold:.4f} "
                    f"token_age_ms={decision.token_age_ms} transcript={transcript!r}"
                )
                self.current_turn = asyncio.create_task(
                    self.run_turn(transcript, turn_id, decision.reason)
                )

    async def _finalize(self, stt, fragments: list[str]) -> str:
        """Feed trailing silence to flush the delayed STT output, then reset."""
        for _ in range(FLUSH_BLOCKS):
            fragment, _ = await self._stt(stt.step, _SILENCE)
            if fragment:
                fragments.append(fragment)
        await self._stt(stt.reset)
        return "".join(fragments).strip()

    async def run_turn(self, transcript: str, turn_id: int, endpoint_reason: str):
        self.epoch += 1
        epoch = self.epoch
        self.cancel_flag = False
        self.coupling_k = None
        self.coupling_samples = []
        self.barge_gate.reset()
        self.barge_energy_confirmed_until = 0.0
        self.current_reply_text = ""
        self.continuous_stt_logged = False
        self.played_now = 0.0
        self.played_rms.clear()
        self.last_playback_reference_at = 0.0
        self.playback_reference_logged = False
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
        self.current_reply_text = reply
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
        # The STT model is a single stateful stream. Reject a second client
        # instead of silently superseding the live app: superseding used to
        # leave the first WebSocket open with a stopped STT loop, so its UI
        # still showed microphone activity while the engine ignored it.
        if Agent.active_session_id is not None:
            log(
                "session rejected reason=voice_session_already_active "
                f"active_session={Agent.active_session_id}"
            )
            await self.ws.close(code=1013, reason="A voice session is already active")
            return
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
                        self.mic_blocks_dropped += 1
                        now = time.time()
                        if now - self.last_mic_drop_log >= 300.0:
                            log(
                                "owner=engine state=mic_backlog_trimmed "
                                f"dropped_blocks={self.mic_blocks_dropped} "
                                f"queue_blocks={self.mic_queue.maxsize}"
                            )
                            self.last_mic_drop_log = now
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
                    elif event.get("type") == "capture_diagnostic":
                        stage = str(event.get("stage", "unknown"))[:80]
                        details = str(event.get("details", ""))[:500]
                        log(f"audio capture stage={stage} details={details}")
                    elif event.get("type") == "playback_reference":
                        try:
                            rms = max(0.0, float(event.get("rms", 0.0)))
                        except (TypeError, ValueError):
                            rms = 0.0
                        now = time.time()
                        self.played_now = rms
                        self.last_playback_reference_at = now
                        self.played_rms.append((now, rms))
                        if not self.playback_reference_logged:
                            self.playback_reference_logged = True
                            log(
                                f"turn={self.current_turn_id or '-'} owner=assistant "
                                f"state=playback_reference_active rms={rms:.4f}"
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
                        self.current_reply_text = ""
                        self.barge_energy_confirmed_until = 0.0
                        self.stt_discard_epoch += 1
                        await self._stt(Agent.stt.reset)
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
    stt_worker = SpeechToTextWorker(config)
    Agent.stt = stt_worker
    _READY["stt_ready"] = True
    log("STT ready (continuous CPU worker)")
    try:
        mx.set_default_device(mx.gpu)
        Agent.tts = TextToSpeech(config)
        _READY["tts_ready"] = True
        log("TTS ready (MLX GPU)")

        threading.Thread(
            target=lambda: ThreadingHTTPServer(("127.0.0.1", port), HealthHandler).serve_forever(),
            daemon=True,
        ).start()
        log(f"Health on http://127.0.0.1:{port}/health · WebSocket on ws://127.0.0.1:{ws_port}")

        async def serve():
            async with websockets.serve(lambda ws: Agent(config, ws).run(), "127.0.0.1", ws_port):
                await asyncio.Future()

        asyncio.run(serve())
    finally:
        _READY["stt_ready"] = False
        stt_worker.close()


if __name__ == "__main__":
    main()
