"""Per-turn latency tracking for the Chirpy debug UI.

The speech pipeline is VAD -> STT -> LLM -> TTS. This module records a wall-clock
timestamp at each stage for the current turn and publishes a compact ``latency``
event to the client, which renders a Chrome-style waterfall graph above the logs.

Key numbers it exposes:

- ``vad_ms``       how long the user's speech segment lasted (VAD start -> end)
- ``stt_ms``       time from *finishing speech* to the final transcript
  (VAD end -> STT done) -- the headline number the user asked for
- ``llm_ms``       time from the final transcript to the first assistant text
- ``tts_ms``       time from the assistant text to TTS audio ready
- ``tts_gen_ms``   pure synthesis time (TTS start -> TTS done)
- ``speech_to_reply_ms``  finishing speech -> reply audio ready
- ``total_ms``     VAD start -> TTS done (whole turn)

The tracker is event-loop friendly (its hooks are called from LiveKit's async
loops) but keeps a lock so timestamps stay consistent even if a callback lands
on a worker thread.
"""

from __future__ import annotations

import threading
import time

_STAGES = ("vad", "stt", "llm", "tts")


def _ms(a: float | None, b: float | None) -> int | None:
    """Milliseconds between two timestamps (None if either is missing)."""
    if a is None or b is None:
        return None
    return int(round((b - a) * 1000))


class LatencyTracker:
    def __init__(self, publish=None):
        self._publish = publish or (lambda payload: None)
        self._turn: dict | None = None
        self._seq = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------
    def on_vad_start(self):
        """A new speech segment always begins a fresh turn. The assistant reply
        that follows is attributed to this turn."""
        with self._lock:
            self._seq += 1
            self._turn = {
                "id": self._seq,
                "vad_start": time.time(),
                "vad_end": None,
                "stt_done": None,
                "llm_first": None,
                "tts_start": None,
                "tts_done": None,
                "user_text": "",
                "assistant_text": "",
            }

    def on_vad_end(self):
        with self._lock:
            if self._turn is not None:
                self._turn["vad_end"] = time.time()

    def on_stt_done(self, text: str):
        with self._lock:
            if self._turn is not None:
                self._turn["stt_done"] = time.time()
                self._turn["user_text"] = text

    def on_assistant_text(self, text: str):
        with self._lock:
            if self._turn is None:
                # Assistant turn with no preceding tracked user speech
                # (e.g. the startup greeting). Bootstrap a turn.
                self._seq += 1
                self._turn = {
                    "id": self._seq,
                    "vad_start": time.time(),
                    "vad_end": None,
                    "stt_done": None,
                    "llm_first": None,
                    "tts_start": None,
                    "tts_done": None,
                    "user_text": "",
                    "assistant_text": "",
                }
            turn = self._turn
            if turn["llm_first"] is None:
                turn["llm_first"] = time.time()
                turn["tts_start"] = turn["llm_first"]
            turn["assistant_text"] += text

    def on_tts_start(self):
        with self._lock:
            if self._turn is not None and self._turn["tts_start"] is None:
                self._turn["tts_start"] = time.time()

    def on_tts_done(self):
        with self._lock:
            turn = self._turn
            if turn is None:
                return
            turn["tts_done"] = time.time()
            payload = self._finalize(turn)
            self._turn = None
        self._publish(payload)

    # ------------------------------------------------------------------
    # Unified dispatcher used by the plugin callbacks
    # ------------------------------------------------------------------
    def handle(self, stage: str, event: str, text: str = ""):
        if stage == "vad":
            if event == "start":
                self.on_vad_start()
            elif event == "end":
                self.on_vad_end()
        elif stage == "stt":
            if event == "done":
                self.on_stt_done(text)
        elif stage == "llm":
            if event == "text":
                self.on_assistant_text(text)
        elif stage == "tts":
            if event == "start":
                self.on_tts_start()
            elif event == "done":
                self.on_tts_done()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _finalize(self, t: dict) -> dict:
        vad_ms = _ms(t["vad_start"], t["vad_end"]) or 0
        stt_ms = _ms(t["vad_end"], t["stt_done"]) or 0
        llm_ms = _ms(t["stt_done"], t["llm_first"]) or 0
        tts_ms = _ms(t["llm_first"], t["tts_done"]) or 0
        tts_gen_ms = _ms(t["tts_start"], t["tts_done"]) or 0
        total_ms = _ms(t["vad_start"], t["tts_done"]) or 0

        # Chrome-waterfall style segments laid out on the turn timeline. Each
        # segment's start/dur are in ms relative to vad_start.
        starts = {"vad": 0}
        stages = []
        acc = 0
        for idx, stage in enumerate(_STAGES):
            dur = (vad_ms, stt_ms, llm_ms, tts_ms)[idx]
            stages.append(
                {
                    "stage": stage,
                    "label": stage.upper(),
                    "start_ms": acc,
                    "dur_ms": dur,
                    "color": stage,
                }
            )
            acc += dur

        return {
            "type": "latency",
            "id": t["id"],
            "stages": stages,
            "total_ms": total_ms,
            "vad_ms": vad_ms,
            "stt_ms": stt_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "tts_gen_ms": tts_gen_ms,
            "speech_to_transcript_ms": stt_ms,
            "speech_to_reply_ms": _ms(t["vad_end"], t["tts_done"]) or 0,
            "user_text": t["user_text"],
            "assistant_text": t["assistant_text"],
            "t": time.time(),
        }
