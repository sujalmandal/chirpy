"""EchoGuard: content-based defence against the agent interrupting itself.

The agent's own TTS plays through the speakers; residual echo on the mic can be
picked up by VAD and interrupt the agent mid-sentence ("it stops without me
speaking"). LiveKit cannot tell the agent's own voice from the user's purely by
VAD. This guard keeps a rolling window of the assistant's recently-spoken text
and, when a "user" transcript appears while the agent is speaking, measures how
much of it is actually the agent's own words (``bargein.echo_overlap``). Above a
threshold it is classified as echo: it is logged, counted, and withheld from the
client transcript instead of being treated as a genuine barge-in.

Pure policy logic lives in :mod:`bargein`; this class only wires it to a LiveKit
AgentSession's events so it stays unit-testable without a running session.
"""

from __future__ import annotations

import collections
import logging
from typing import Callable

from bargein import BargeInPolicy, echo_overlap

logger = logging.getLogger("chirpy.echoguard")

# Keep the last N assistant utterances to compare against.
RECENT_AGENT_MAX = 8


class EchoGuard:
    """Tracks recent assistant speech and classifies "user" input as echo."""

    def __init__(self, policy: BargeInPolicy) -> None:
        self.policy = policy
        self._recent: collections.deque[str] = collections.deque(maxlen=RECENT_AGENT_MAX)
        self.echo_suppressed: int = 0
        self.checked: int = 0
        self.on_echo: Callable[[float, str], None] | None = None

    # -- feed assistant text -------------------------------------------------
    def note_assistant_text(self, text: str) -> None:
        if text:
            self._recent.append(text)

    def recent_agent_text(self) -> str:
        return " ".join(self._recent)

    def clear(self) -> None:
        self._recent.clear()

    # -- classification ------------------------------------------------------
    def overlap(self, user_text: str) -> float:
        self.checked += 1
        return echo_overlap(user_text, self.recent_agent_text())

    def is_echo(self, user_text: str) -> bool:
        """True when ``user_text`` looks like the agent's own recent speech."""
        if not user_text:
            return False
        ov = self.overlap(user_text)
        if ov >= self.policy.echo_overlap_threshold:
            self.echo_suppressed += 1
            logger.info("suppressing echo barge-in overlap=%.2f text=%r", ov, user_text)
            if self.on_echo is not None:
                try:
                    self.on_echo(ov, user_text)
                except Exception:  # noqa: BLE001
                    logger.exception("on_echo callback failed")
            return True
        return False
