from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class ConversationStore:
    """Small in-memory session history; persistence stays opt-in."""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._messages: dict[str, list[Message]] = defaultdict(list)

    def context(self, session_id: str) -> list[Message]:
        return self._messages[session_id][-self.max_turns * 2 :]

    def add_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        self._messages[session_id].extend([Message("user", user_text), Message("assistant", assistant_text)])

    def clear(self, session_id: str) -> None:
        self._messages.pop(session_id, None)
