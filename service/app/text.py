from __future__ import annotations

import re

SENTENCE_END = re.compile(r"(?<=[.!?])(?:\s+|$)")


def complete_sentences(buffer: str, final: bool = False) -> tuple[list[str], str]:
    """Return only complete, speakable sentences and retain the unfinished tail."""
    parts = SENTENCE_END.split(buffer)
    if len(parts) == 1 and not final:
        return [], buffer
    remainder = "" if final else parts.pop()
    sentences = [part.strip() for part in parts if part.strip()]
    if final and remainder.strip():
        sentences.append(remainder.strip())
    return sentences, remainder
