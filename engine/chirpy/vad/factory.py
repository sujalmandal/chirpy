"""Factory: pick the roomkit-style VAD provider from config.

Mirrors roomkit's ``build_vad``: use the neural sherpa-onnx VAD when a model
file is available (and sherpa-onnx is installed), otherwise fall back to the
zero-dependency energy VAD. Threshold/silence knobs are data-driven.
"""

from __future__ import annotations

import logging
from pathlib import Path

from livekit.agents import vad as agents_vad

from bargein import _as_bool, _as_float, _as_int

from .base import VADConfig
from .energy import EnergyVAD
from .sherpa import SherpaOnnxVAD, SherpaOnnxVADConfig

logger = logging.getLogger("chirpy.vad.factory")


def build_vad(config: dict[str, str]) -> agents_vad.VAD:
    """Return a VAD provider chosen from ``config``.

    Neural path (sherpa-onnx TEN-VAD / Silero) is used when ``VAD_MODEL`` points
    at an existing .onnx file and sherpa-onnx is importable; otherwise it falls
    back to the energy (RMS) VAD so the worker always has a VAD.
    """
    model = (config.get("VAD_MODEL") or "").strip()
    model_exists = bool(model) and Path(model).expanduser().exists()

    try:
        import sherpa_onnx  # noqa: F401
        sherpa_ok = True
    except Exception:  # noqa: BLE001
        sherpa_ok = False

    if model_exists and sherpa_ok:
        cfg = SherpaOnnxVADConfig(
            model=str(Path(model).expanduser()),
            model_type=(config.get("VAD_MODEL_TYPE") or "ten").strip().lower() or "ten",
            threshold=_as_float(config.get("VAD_THRESHOLD"), 0.35),
            silence_threshold_ms=_as_int(config.get("VAD_SILENCE_MS"), 500),
            min_speech_duration_ms=_as_int(config.get("VAD_MIN_SPEECH_MS"), 250),
            speech_pad_ms=_as_int(config.get("VAD_SPEECH_PAD_MS"), 300),
            energy_silence_rms=_as_float(
                config.get("VAD_ENERGY_SILENCE_RMS"), 0.0006
            ),
            sample_rate=_as_int(config.get("VAD_SAMPLE_RATE"), 16000),
        )
        logger.info("VAD: sherpa-onnx (%s) threshold=%.2f silence=%dms",
                    cfg.model_type, cfg.threshold, cfg.silence_threshold_ms)
        return SherpaOnnxVAD(cfg)

    if model and not model_exists:
        logger.warning("VAD_MODEL %r not found; falling back to energy VAD", model)
    if model_exists and not sherpa_ok:
        logger.warning("sherpa-onnx not importable; falling back to energy VAD")

    cfg = VADConfig(
        silence_threshold_ms=_as_int(config.get("VAD_SILENCE_MS"), 500),
        min_speech_duration_ms=_as_int(config.get("VAD_MIN_SPEECH_MS"), 250),
        speech_pad_ms=_as_int(config.get("VAD_SPEECH_PAD_MS"), 300),
        sample_rate=_as_int(config.get("VAD_SAMPLE_RATE"), 16000),
    )
    threshold = _as_float(config.get("VAD_ENERGY_THRESHOLD"), 0.01)
    logger.info("VAD: energy (RMS) threshold=%.4f silence=%dms",
                threshold, cfg.silence_threshold_ms)
    return EnergyVAD(energy_threshold=threshold, config=cfg)


__all__ = ["build_vad"]
