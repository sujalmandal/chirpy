"""Hot TTS/STT configuration reload.

Watches ``config/tts.json`` (and ``config/stt.json``) and applies changes to the
live plugins without restarting the agent worker. Voices and STT model
checkpoints are pre-fetched into the Hugging Face cache so the next use is fast.
"""

from __future__ import annotations

import json
import logging
import threading

from huggingface_hub import hf_hub_download

from download import _KOKORO_DEFAULT_REPO

logger = logging.getLogger("chirpy.tts_watch")


def apply_tts_live(tts_plugin, data: dict) -> bool:
    """Update a KokoroTTS plugin's voice/language/speed in place.

    Returns True if anything changed. The voice checkpoint is pre-fetched in a
    background thread so the first synthesis after the change is not blocked on
    a download.
    """
    voice = str(data.get("voice") or "af_heart").strip() or "af_heart"
    lang = str(data.get("lang") or "a").strip() or "a"
    try:
        speed = float(data.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0

    changed = (
        voice != tts_plugin._voice
        or lang != tts_plugin._lang_code
        or abs(speed - tts_plugin._speed) > 1e-6
    )
    if changed:
        # reload() applies voice/speed immediately and recreates the pipeline on
        # a language change so the new language actually takes effect.
        tts_plugin.reload(lang_code=lang, voice=voice, speed=speed)
        logger.info("TTS voice hot-reloaded -> %s (lang=%s speed=%.1f)", voice, lang, speed)
        threading.Thread(target=_prefetch_voice, args=(voice,), daemon=True).start()
    return changed


def apply_stt_live(stt_plugin, data: dict) -> bool:
    """Update a WhisperSTT plugin's model size and/or language in place.

    The language applies immediately; a changed model size reloads the
    faster-whisper model in a background thread (so a large model download or
    load doesn't block the watcher or the running session).
    """
    model = str(data.get("model") or "base").strip() or "base"
    language = str(data.get("language") or "en").strip() or "en"

    changed = model != stt_plugin._model_size or language != stt_plugin._language
    if changed:
        logger.info("STT hot-reloaded -> model=%s language=%s", model, language)
        stt_plugin._language = language
        if model != stt_plugin._model_size:
            threading.Thread(
                target=stt_plugin.reload, kwargs={"model_size": model, "language": language},
                daemon=True,
            ).start()
        else:
            stt_plugin.reload(language=language)
    return changed


def _prefetch_voice(voice: str) -> None:
    try:
        hf_hub_download(repo_id=_KOKORO_DEFAULT_REPO, filename=f"voices/{voice}.pt")
        logger.info("voice %s pre-cached", voice)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not pre-cache voice %s: %s", voice, exc)


class ConfigWatcher:
    """Poll a JSON config file and call ``apply_cb`` on changes."""

    def __init__(self, path, apply_cb, poll_interval: float = 0.5) -> None:
        self._path = path
        self._apply = apply_cb
        self._interval = poll_interval
        self._stop = threading.Event()
        self._last_mtime = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> "ConfigWatcher":
        self._thread = threading.Thread(target=self._run, daemon=True, name="cfg-watch")
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.poll_once()

    def poll_once(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("bad %s: %s", self._path.name, exc)
            return
        if isinstance(data, dict):
            self._apply(data)

    def stop(self) -> None:
        self._stop.set()


# Backward-compatible alias.
TTSWatcher = ConfigWatcher
