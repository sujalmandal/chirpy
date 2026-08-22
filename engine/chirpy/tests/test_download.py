"""Tests for the model download helper (download.py)."""

from __future__ import annotations

import unittest

from download import _KOKORO_DEFAULT_REPO, _resolve_stt_repo, download_stt, download_tts


class ResolveSttTest(unittest.TestCase):
    def test_size_maps_to_systran_repo(self):
        self.assertEqual(_resolve_stt_repo("base"), "Systran/faster-whisper-base")
        self.assertEqual(_resolve_stt_repo("small"), "Systran/faster-whisper-small")
        self.assertEqual(_resolve_stt_repo("large-v3"), "Systran/faster-whisper-large-v3")

    def test_full_repo_id_passes_through(self):
        self.assertEqual(_resolve_stt_repo("Systran/faster-whisper-medium"), "Systran/faster-whisper-medium")
        self.assertEqual(_resolve_stt_repo("org/custom"), "org/custom")

    def test_unknown_size_raises(self):
        with self.assertRaises(ValueError):
            _resolve_stt_repo("garbage")


class ResolveTtsTest(unittest.TestCase):
    def test_default_repo(self):
        self.assertEqual(_KOKORO_DEFAULT_REPO, "hexgrad/Kokoro-82M")

    def test_download_tts_resolves_repo(self):
        # The download itself is network-bound; verify it targets the right repo
        # and files by monkeypatching the hub calls (which also receive the
        # tqdm_class progress hook).
        calls: list[tuple] = []

        import download

        def fake(repo_id, **kwargs):
            assert "tqdm_class" in kwargs  # progress reporting is wired up
            calls.append((repo_id, {k: v for k, v in kwargs.items() if k != "tqdm_class"}))
            return repo_id

        download.snapshot_download = fake
        download.hf_hub_download = fake
        try:
            repo = download_tts("af_heart")
            download_stt("base")
        finally:
            download.snapshot_download = original_snapshot
            download.hf_hub_download = original_hf

        self.assertEqual(repo, "hexgrad/Kokoro-82M")
        # model snapshot
        self.assertEqual(calls[0][0], "hexgrad/Kokoro-82M")
        self.assertEqual(calls[0][1]["allow_patterns"], ["*.pt", "*.json", "*.md"])
        # voice file via hf_hub_download
        self.assertEqual(calls[1][1]["filename"], "voices/af_heart.pt")
        # stt maps to the Systran repo
        self.assertEqual(calls[2][0], "Systran/faster-whisper-base")


original_snapshot = __import__("download").snapshot_download
original_hf = __import__("download").hf_hub_download


if __name__ == "__main__":
    unittest.main()
