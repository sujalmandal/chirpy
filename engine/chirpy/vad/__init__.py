"""Roomkit-style VAD providers for Chirpy.

Drop-in ``agents.vad.VAD`` implementations with a pluggable per-stream frame
detector (energy RMS or sherpa-onnx TEN-VAD/Silero), a shared pre-roll + silence
state machine, and roomkit's energy fast-exit anti-inertia fix.
"""

from .base import VADConfig
from .energy import EnergyVAD
from .factory import build_vad
from .sherpa import SherpaOnnxVAD, SherpaOnnxVADConfig

__all__ = [
    "EnergyVAD",
    "SherpaOnnxVAD",
    "SherpaOnnxVADConfig",
    "VADConfig",
    "build_vad",
]
