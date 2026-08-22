"""sherpa-onnx neural VAD provider (TEN-VAD / Silero), roomkit-style.

Replicates roomkit's ``SherpaOnnxVADProvider``: a sherpa-onnx
``VoiceActivityDetector`` (TEN-VAD or Silero) plus the energy-based fast exit
that forces SPEECH_END when the model stays in speech on true silence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import VADConfig, _FrameDetector, _StreamingVAD

logger = logging.getLogger("chirpy.vad.sherpa")


@dataclass
class SherpaOnnxVADConfig(VADConfig):
    """Configuration for the sherpa-onnx neural VAD.

    Fields not listed here are inherited from :class:`VADConfig`.
    """

    model: str = ""
    """Path to the .onnx model file."""
    model_type: str = "ten"
    """'ten' (TEN-VAD) or 'silero' (Silero VAD)."""
    threshold: float = 0.35
    """Speech probability threshold (0-1). 0.35 suits denoised audio; use 0.5
    without a denoiser."""
    max_speech_duration: float = 20.0
    """Max segment length (s) enforced inside sherpa."""
    sherpa_min_silence_duration: float = 0.05
    """Keep low; debounce is handled by our state machine."""
    sherpa_min_speech_duration: float = 0.1
    """Keep low; debounce is handled by our state machine."""
    num_threads: int = 1
    """CPU threads for inference."""
    provider: str = "cpu"
    """ONNX execution provider."""


class _SherpaDetector(_FrameDetector):
    def __init__(self, config: SherpaOnnxVADConfig) -> None:
        import sherpa_onnx  # noqa: F401

        self._config = config
        self._sherpa = __import__("sherpa_onnx")
        self._detector = None
        self._ensure()

    def _ensure(self) -> None:
        if self._detector is not None:
            return
        cfg = self._config
        sherpa = self._sherpa
        vad_config = sherpa.VadModelConfig()
        if cfg.model_type == "silero":
            vad_config.silero_vad.model = cfg.model
            vad_config.silero_vad.threshold = cfg.threshold
            vad_config.silero_vad.max_speech_duration = cfg.max_speech_duration
            vad_config.silero_vad.min_silence_duration = cfg.sherpa_min_silence_duration
            vad_config.silero_vad.min_speech_duration = cfg.sherpa_min_speech_duration
        else:  # TEN-VAD
            vad_config.ten_vad.model = cfg.model
            vad_config.ten_vad.threshold = cfg.threshold
            vad_config.ten_vad.max_speech_duration = cfg.max_speech_duration
            vad_config.ten_vad.min_silence_duration = cfg.sherpa_min_silence_duration
            vad_config.ten_vad.min_speech_duration = cfg.sherpa_min_speech_duration
        vad_config.sample_rate = cfg.sample_rate
        vad_config.num_threads = cfg.num_threads
        vad_config.provider = cfg.provider
        self._detector = sherpa.VoiceActivityDetector(vad_config)
        logger.info(
            "sherpa-onnx VAD detector created model_type=%s threshold=%.2f model=%s",
            cfg.model_type, cfg.threshold, cfg.model,
        )

    def process(self, mono_f32: np.ndarray) -> tuple[float, bool]:
        self._ensure()
        self._detector.accept_waveform(np.ascontiguousarray(mono_f32))
        while not self._detector.empty():
            self._detector.pop()
        is_speech = bool(self._detector.is_speech_detected())
        # sherpa's detector reports a binary state; surface it as confidence.
        return (1.0 if is_speech else 0.0), is_speech

    def reset(self) -> None:
        if self._detector is not None:
            self._detector.reset()

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.flush()
            except Exception:  # noqa: BLE001
                pass
            self._detector = None


class SherpaOnnxVAD(_StreamingVAD):
    """Neural VAD (TEN-VAD or Silero) backed by sherpa-onnx."""

    def __init__(self, config: SherpaOnnxVADConfig | None = None) -> None:
        self._sherpa_config = config or SherpaOnnxVADConfig()
        if not self._sherpa_config.model:
            raise ValueError("SherpaOnnxVAD requires a model path (SherpaOnnxVADConfig.model)")
        super().__init__(
            self._sherpa_config,
            model=f"sherpa-{self._sherpa_config.model_type}",
            provider="roomkit-style",
        )

    def _make_detector(self) -> _FrameDetector:
        return _SherpaDetector(self._sherpa_config)
