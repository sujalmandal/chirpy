"""Local STT/TTS plugins for LiveKit Agents.

These wrap small on-device models so the Chirpy agent worker can use them through
LiveKit's provider-agnostic STT/TTS interfaces. Speech never leaves the Mac.
"""

from .kokoro_tts import KokoroTTS
from .whisper_stt import WhisperSTT

__all__ = ["WhisperSTT", "KokoroTTS"]
