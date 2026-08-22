"""Robust, dynamic, config-driven barge-in policy for Chirpy.

Why this exists
---------------
Pure VAD barge-in over-triggers: the mic picks up the assistant's own TTS
(speaker echo), plus backchannels ("mm-hmm", "yeah"), coughs and background
noise, and treats each as a full interruption (industry research: LiveKit
"Adaptive Interruption Handling", Deepgram "Audio Preprocessing & Barge-In").
LiveKit's cloud ``adaptive`` interruption model rejects ~51% of VAD-triggered
interruptions as false, but it is a hosted service and is not available against
a self-hosted LiveKit server. This module provides the local, data-driven
defences: a validated, configurable policy (no hardcoded magic numbers), an
adaptive->vad fallback, a content-based echo-overlap scorer, and a runtime
config watcher so barge-in can be tuned without a rebuild.

Everything here is pure and unit-testable without a LiveKit server, local
speech models, or an LLM endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from livekit.agents.inference import TurnDetector

logger = logging.getLogger("chirpy.bargein")

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
# We adopt LiveKit's own documented interruption defaults (voice/turn.py
# ``_INTERRUPTION_DEFAULTS``) rather than inventing arbitrary numbers.
LK_INTERRUPTION_DEFAULTS: dict = {
    "min_duration": 0.5,                # sustained speech (s) before an interruption counts
    "min_words": 0,                     # min transcribed words before interrupting (0 = any)
    "resume_false_interruption": True,  # resume the reply after a false start
    "false_interruption_timeout": 2.0,  # silence (s) after an interruption = false start
    "backchannel_boundary": (1.0, 1.0),  # suppress interruptions near turn start/end
    "discard_audio_if_uninterruptible": True,
}

DEFAULT_ENDPOINTING_MIN_DELAY = 0.7
DEFAULT_ENDPOINTING_MAX_DELAY = 2.0
DEFAULT_AEC_WARMUP_DURATION = 3.0

# At/above this token-overlap ratio we treat a "user" transcript as the agent's
# own echo and refuse to treat it as a real barge-in. Content-based, not a fixed
# word/time count.
DEFAULT_ECHO_OVERLAP_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
# Value parsing + validation
# --------------------------------------------------------------------------- #
def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() not in {"0", "false", "off", "no", "disabled"}


def _as_float(value, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def has_adaptive_credentials(config: dict[str, str]) -> bool:
    """True when hosted LiveKit inference credentials are available.

    The cloud ``adaptive`` interruption detector requires an API key + secret
    (or ``LIVEKIT_INFERENCE_API_KEY``). Without them it raises ValueError at
    construction, so we fall back to local VAD.
    """
    return bool(
        config.get("LIVEKIT_INFERENCE_API_KEY")
        or (config.get("LIVEKIT_API_KEY") and config.get("LIVEKIT_API_SECRET"))
    )


def _resolve_mode(config: dict[str, str]) -> str:
    """Resolve and validate the interruption mode with adaptive->vad fallback."""
    mode = (config.get("INTERRUPTION_MODE") or "vad").strip().lower()
    if mode not in ("vad", "adaptive"):
        logger.warning("invalid INTERRUPTION_MODE %r, falling back to vad", mode)
        mode = "vad"
    if mode == "adaptive" and not has_adaptive_credentials(config):
        logger.warning(
            "INTERRUPTION_MODE=adaptive requested but no hosted LiveKit inference "
            "credentials are present; falling back to local vad interruption"
        )
        mode = "vad"
    return mode


def _resolve_backchannel(value: str | None) -> float | tuple[float, float] | None:
    """Parse BARGE_IN_BACKCHANNEL: off/0 -> None; auto/true -> default pair; else seconds."""
    if value is None or str(value).strip() == "":
        return LK_INTERRUPTION_DEFAULTS["backchannel_boundary"]
    raw = str(value).strip().lower()
    if raw in {"0", "off", "no", "false", "none", "disabled"}:
        return None
    if raw in {"1", "on", "yes", "true", "auto"}:
        return LK_INTERRUPTION_DEFAULTS["backchannel_boundary"]
    try:
        parts = [float(p) for p in re.split(r"[,;\s]+", raw) if p]
        if len(parts) == 1:
            v = max(0.0, parts[0])
            return (v, v)
        if len(parts) >= 2:
            return (max(0.0, parts[0]), max(0.0, parts[1]))
    except (TypeError, ValueError):
        pass
    logger.warning("invalid BARGE_IN_BACKCHANNEL %r, using default", value)
    return LK_INTERRUPTION_DEFAULTS["backchannel_boundary"]


def _resolve_false_timeout(value: str | None) -> float | None:
    """Parse BARGE_IN_FALSE_TIMEOUT; a false/0/off value disables resume (None)."""
    if value is None or str(value).strip() == "":
        return LK_INTERRUPTION_DEFAULTS["false_interruption_timeout"]
    raw = str(value).strip().lower()
    if raw in {"0", "off", "no", "false", "none", "disabled"}:
        return None
    return max(0.0, _as_float(raw, LK_INTERRUPTION_DEFAULTS["false_interruption_timeout"]))


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
@dataclass
class BargeInPolicy:
    """Validated, runtime-tunable barge-in policy. Defaults mirror LiveKit's."""

    enabled: bool = True
    mode: str = "vad"
    min_duration: float = LK_INTERRUPTION_DEFAULTS["min_duration"]
    min_words: int = LK_INTERRUPTION_DEFAULTS["min_words"]
    resume_false_interruption: bool = LK_INTERRUPTION_DEFAULTS["resume_false_interruption"]
    false_interruption_timeout: float | None = LK_INTERRUPTION_DEFAULTS[
        "false_interruption_timeout"
    ]
    backchannel_boundary: float | tuple[float, float] | None = LK_INTERRUPTION_DEFAULTS[
        "backchannel_boundary"
    ]
    discard_audio_if_uninterruptible: bool = LK_INTERRUPTION_DEFAULTS[
        "discard_audio_if_uninterruptible"
    ]
    aec_warmup_duration: float = DEFAULT_AEC_WARMUP_DURATION
    endpointing_min_delay: float = DEFAULT_ENDPOINTING_MIN_DELAY
    endpointing_max_delay: float = DEFAULT_ENDPOINTING_MAX_DELAY
    echo_overlap_threshold: float = DEFAULT_ECHO_OVERLAP_THRESHOLD

    def interruption_options(self) -> dict:
        """The exact ``interruption`` block LiveKit's turn_handling reads."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "min_duration": max(0.0, self.min_duration),
            "min_words": max(0, self.min_words),
            "resume_false_interruption": self.resume_false_interruption,
            "false_interruption_timeout": self.false_interruption_timeout,
            "backchannel_boundary": self.backchannel_boundary,
            "discard_audio_if_uninterruptible": self.discard_audio_if_uninterruptible,
        }

    def endpointing_options(self) -> dict:
        lo = max(0.0, self.endpointing_min_delay)
        hi = max(0.0, self.endpointing_max_delay)
        if lo > hi:
            lo, hi = hi, lo
        return {"mode": "dynamic", "min_delay": lo, "max_delay": hi}


def load_barge_in_policy(config: dict[str, str]) -> BargeInPolicy:
    """Build a validated BargeInPolicy from config/env values."""
    return BargeInPolicy(
        enabled=_as_bool(config.get("BARGE_IN"), default=True),
        mode=_resolve_mode(config),
        min_duration=max(
            0.0, _as_float(config.get("BARGE_IN_MIN_DURATION"), LK_INTERRUPTION_DEFAULTS["min_duration"])
        ),
        min_words=max(
            0, _as_int(config.get("BARGE_IN_MIN_WORDS"), LK_INTERRUPTION_DEFAULTS["min_words"])
        ),
        resume_false_interruption=_as_bool(config.get("BARGE_IN_RESUME_FALSE"), default=True),
        false_interruption_timeout=_resolve_false_timeout(config.get("BARGE_IN_FALSE_TIMEOUT")),
        backchannel_boundary=_resolve_backchannel(config.get("BARGE_IN_BACKCHANNEL")),
        discard_audio_if_uninterruptible=_as_bool(
            config.get("BARGE_IN_DISCARD_AUDIO"), default=True
        ),
        aec_warmup_duration=max(
            0.0, _as_float(config.get("AEC_WARMUP_DURATION"), DEFAULT_AEC_WARMUP_DURATION)
        ),
        endpointing_min_delay=_as_float(
            config.get("ENDPOINTING_MIN_DELAY"), DEFAULT_ENDPOINTING_MIN_DELAY
        ),
        endpointing_max_delay=_as_float(
            config.get("ENDPOINTING_MAX_DELAY"), DEFAULT_ENDPOINTING_MAX_DELAY
        ),
        echo_overlap_threshold=max(
            0.0,
            min(1.0, _as_float(config.get("ECHO_OVERLAP_THRESHOLD"), DEFAULT_ECHO_OVERLAP_THRESHOLD)),
        ),
    )


def build_turn_handling(policy: BargeInPolicy) -> dict:
    """Return the AgentSession ``turn_handling`` dict from a policy."""
    return {
        "turn_detection": TurnDetector(),
        "endpointing": policy.endpointing_options(),
        "interruption": policy.interruption_options(),
    }


# --------------------------------------------------------------------------- #
# Echo detection (content overlap)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)

# Common function words contribute nothing to "is this the agent's own echo?",
# so they are dropped before scoring to keep the overlap metric content-based.
_STOPWORDS = frozenset(
    "the a an and or but if then so of to in on for with at by from as is are was were "
    "be been being i you he she it we they them me him her us my your his its our their "
    "do does did have has had will would can could should shall may might not no yes "
    "this that these those there here what which who whom whose when where why how "
    "about into over under again further then once all any both each few more most "
    "other some such only own same so than too very just".split()
)


def _tokens(text: str) -> list[str]:
    return [
        t
        for t in _WORD_RE.sub(" ", (text or "").lower()).split()
        if t and t not in _STOPWORDS
    ]


def echo_overlap(user_text: str, agent_text: str) -> float:
    """Fraction of the user's tokens that appear in the agent's recent speech.

    Returns a value in [0, 1]; ~1.0 means the "user" utterance is almost
    entirely the agent's own words (speaker echo). 0 if either input is empty.
    """
    user = _tokens(user_text)
    agent = set(_tokens(agent_text))
    if not user or not agent:
        return 0.0
    return sum(1 for t in user if t in agent) / len(user)


# --------------------------------------------------------------------------- #
# Runtime config watcher (dynamic reload without a restart)
# --------------------------------------------------------------------------- #
class ConfigWatcher:
    """Watches a JSON config file and reports policy changes to a callback.

    Polls the file (cheap, default 1s). On a content change it loads the new
    policy and calls ``on_change(policy)``. Run via ``run()`` as an asyncio
    background task.
    """

    def __init__(
        self,
        path: Path | str,
        on_change,
        *,
        base_config: dict[str, str] | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.on_change = on_change
        self.base_config = base_config or {}
        self.poll_interval = poll_interval
        self._mtime: float | None = None

    def load_policy(self) -> BargeInPolicy:
        """Load the policy, merging the JSON file over the base config."""
        merged = dict(self.base_config)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    for k, v in data.items():
                        merged[str(k)] = str(v)
            except (OSError, ValueError) as exc:
                logger.warning("could not parse %s: %s", self.path, exc)
        return load_barge_in_policy(merged)

    def poll_once(self) -> bool:
        """Reload if the file changed; returns True when a change was applied."""
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else None
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return False
        self._mtime = mtime
        policy = self.load_policy()
        try:
            self.on_change(policy)
        except Exception:  # noqa: BLE001 - a watcher must not die on one bad apply
            logger.exception("on_change handler failed; policy not applied")
            return False
        return True

    async def run(self, *, stop: asyncio.Event | None = None) -> None:
        try:
            self._mtime = self.path.stat().st_mtime if self.path.exists() else None
        except OSError:
            self._mtime = None
        while True:
            if stop is not None and stop.is_set():
                return
            self.poll_once()
            await asyncio.sleep(self.poll_interval)
