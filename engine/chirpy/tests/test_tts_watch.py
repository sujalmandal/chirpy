"""Tests for hot TTS reload (tts_watch.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tts_watch
from tts_watch import ConfigWatcher, TTSWatcher, apply_stt_live, apply_tts_live


class _FakeTTS:
    def __init__(self):
        self._voice = "af_heart"
        self._lang_code = "a"
        self._speed = 1.0


class _FakeSTT:
    def __init__(self, model="base", language="en"):
        self._model_size = model
        self._language = language
        self.reload_calls: list[dict] = []

    def reload(self, **kwargs):
        self.reload_calls.append(kwargs)


class ApplyTtsLiveTest(unittest.TestCase):
    def test_updates_voice_lang_speed(self):
        tts = _FakeTTS()
        changed = apply_tts_live(tts, {"voice": "am_michael", "lang": "a", "speed": 1.2})
        self.assertTrue(changed)
        self.assertEqual(tts._voice, "am_michael")
        self.assertEqual(tts._speed, 1.2)

    def test_returns_false_when_unchanged(self):
        tts = _FakeTTS()
        changed = apply_tts_live(tts, {"voice": "af_heart", "lang": "a", "speed": 1.0})
        self.assertFalse(changed)

    def test_prefetch_backgrounded_no_network_in_test(self):
        calls = []

        def fake_prefetch(voice):
            calls.append(voice)

        orig = tts_watch._prefetch_voice
        tts_watch._prefetch_voice = fake_prefetch
        try:
            apply_tts_live(_FakeTTS(), {"voice": "em_alex"})
        finally:
            tts_watch._prefetch_voice = orig
        # The pre-fetch runs in a daemon thread; give it a moment.
        import time
        for _ in range(20):
            if calls:
                break
            time.sleep(0.01)
        self.assertEqual(calls, ["em_alex"])


class TTSWatcherTest(unittest.TestCase):
    def test_poll_once_reads_file_and_applies(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tts.json"
            applied = []
            w = TTSWatcher(path, applied.append, poll_interval=0.01)
            # no file yet -> no-op
            w.poll_once()
            self.assertEqual(applied, [])

            path.write_text(json.dumps({"voice": "pm_alex", "lang": "a"}))
            w.poll_once()
            self.assertEqual(applied, [{"voice": "pm_alex", "lang": "a"}])

            # unchanged mtime -> no re-apply
            w.poll_once()
            self.assertEqual(len(applied), 1)

    def test_bad_json_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tts.json"
            applied = []
            w = TTSWatcher(path, applied.append, poll_interval=0.01)
            path.write_text("{not json")
            w.poll_once()
            self.assertEqual(applied, [])
            w.stop()


class ApplySttLiveTest(unittest.TestCase):
    def test_language_change_applies_immediately(self):
        stt = _FakeSTT()
        changed = apply_stt_live(stt, {"model": "base", "language": "es"})
        self.assertTrue(changed)
        self.assertEqual(stt._language, "es")
        # no model change -> reload called synchronously with just the language
        self.assertEqual(stt.reload_calls, [{"language": "es"}])

    def test_model_change_reloads_in_background(self):
        stt = _FakeSTT()
        changed = apply_stt_live(stt, {"model": "small", "language": "en"})
        self.assertTrue(changed)
        self.assertEqual(stt._language, "en")
        import time
        for _ in range(50):
            if stt.reload_calls:
                break
            time.sleep(0.01)
        self.assertEqual(stt.reload_calls, [{"model_size": "small", "language": "en"}])

    def test_returns_false_when_unchanged(self):
        stt = _FakeSTT()
        self.assertFalse(apply_stt_live(stt, {"model": "base", "language": "en"}))


class ConfigWatcherTest(unittest.TestCase):
    def test_works_for_stt_and_tts_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            stt_path = Path(d) / "stt.json"
            applied = []
            w = ConfigWatcher(stt_path, applied.append, poll_interval=0.01)
            stt_path.write_text(json.dumps({"model": "small", "language": "es"}))
            w.poll_once()
            self.assertEqual(applied, [{"model": "small", "language": "es"}])
            w.stop()


if __name__ == "__main__":
    unittest.main()
