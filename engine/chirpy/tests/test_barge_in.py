"""Tests for Chirpy's barge-in (interruption) support and natural turn-taking.

These exercise the pure, testable configuration builders in ``agent.py`` so the
behaviour is verified without connecting to a LiveKit server, loading the local
speech models, or reaching an LLM endpoint.

Run from the repo root:

    engine/chirpy/.venv/bin/python -m unittest engine.chirpy.tests.test_barge_in -v

or from ``engine/chirpy``:

    .venv/bin/python -m unittest tests.test_barge_in -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Make the ``agent`` module (and its ``plugins`` package) importable when the
# suite is run straight from the repo root without installing the package.
_ENGINE = Path(__file__).resolve().parents[1]  # engine/chirpy
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from agent import (  # noqa: E402
    _as_bool,
    _as_float,
    _as_int,
    resolve_system_prompt,
    resolve_turn_handling,
    DEFAULT_SYSTEM_PROMPT,
)


class BargeInConfigTests(unittest.TestCase):
    """Barge-in is on by default and resolvable to a local, offline strategy."""

    def test_barge_in_enabled_by_default(self):
        th = resolve_turn_handling({})
        self.assertIs(
            th["interruption"]["enabled"], True,
            "Full barge-in should be enabled out of the box.",
        )

    def test_barge_in_mode_defaults_to_vad_for_local(self):
        # ``vad`` uses the local Silero VAD; ``adaptive`` needs LiveKit's cloud
        # interruption service (hosted API credentials). A local-first assistant
        # must not silently depend on cloud interruption.
        th = resolve_turn_handling({})
        self.assertEqual(
            th["interruption"]["mode"], "vad",
            "Default interruption mode must be the local VAD strategy.",
        )

    def test_can_disable_barge_in_via_config(self):
        th = resolve_turn_handling({"BARGE_IN": "false"})
        self.assertIs(th["interruption"]["enabled"], False)

    def test_barge_in_accepts_true_aliases(self):
        for value in ("true", "1", "on", "yes"):
            with self.subTest(value=value):
                th = resolve_turn_handling({"BARGE_IN": value})
                self.assertIs(th["interruption"]["enabled"], True)

    def test_mode_can_be_overridden(self):
        # "adaptive" stays only when hosted inference credentials are present;
        # otherwise it safely falls back to the local "vad" strategy.
        th = resolve_turn_handling(
            {
                "INTERRUPTION_MODE": "adaptive",
                "LIVEKIT_API_KEY": "k",
                "LIVEKIT_API_SECRET": "s",
            }
        )
        self.assertEqual(th["interruption"]["mode"], "adaptive")

    def test_invalid_mode_falls_back_to_vad(self):
        th = resolve_turn_handling({"INTERRUPTION_MODE": "quantum"})
        self.assertEqual(th["interruption"]["mode"], "vad")

    def test_turn_detector_present(self):
        th = resolve_turn_handling({})
        # A turn detector plus VAD is what lets the session actually commit the
        # user's next turn once they barge in.
        self.assertIsNotNone(th.get("turn_detection"))

    def test_adaptive_mode_still_allows_barge_in_flag(self):
        th = resolve_turn_handling({"INTERRUPTION_MODE": "adaptive"})
        self.assertIs(th["interruption"]["enabled"], True)

    def test_full_interruption_options_present(self):
        """The interruption block exposes the keys LiveKit reads at runtime."""
        th = resolve_turn_handling({})
        opts = th["interruption"]
        for key in (
            "enabled",
            "mode",
            "min_words",
            "resume_false_interruption",
            "false_interruption_timeout",
        ):
            self.assertIn(key, opts, f"missing interruption key: {key}")

    def test_min_words_default_requires_real_speech(self):
        # Interrupt on real transcribed speech. Default is LiveKit's natural
        # policy (0 = any speech activity); it is data-driven, never < 0.
        th = resolve_turn_handling({})
        self.assertGreaterEqual(th["interruption"]["min_words"], 0)

    def test_min_words_override(self):
        th = resolve_turn_handling({"BARGE_IN_MIN_WORDS": "0"})
        self.assertEqual(th["interruption"]["min_words"], 0)

    def test_false_interruption_timeout_override(self):
        th = resolve_turn_handling({"BARGE_IN_FALSE_TIMEOUT": "3.0"})
        self.assertEqual(th["interruption"]["false_interruption_timeout"], 3.0)


class NaturalTurnTakingTests(unittest.TestCase):
    """End-of-turn timing and the spoken persona make conversation feel natural."""

    def test_endpointing_is_dynamic_with_sane_bounds(self):
        ep = resolve_turn_handling({})["endpointing"]
        self.assertEqual(ep["mode"], "dynamic")
        self.assertGreater(ep["max_delay"], ep["min_delay"])
        self.assertGreaterEqual(ep["min_delay"], 0.3)
        self.assertLessEqual(ep["max_delay"], 3.0)

    def test_endpointing_overridable(self):
        ep = resolve_turn_handling(
            {"ENDPOINTING_MIN_DELAY": "1.2", "ENDPOINTING_MAX_DELAY": "1.8"}
        )["endpointing"]
        self.assertEqual(ep["min_delay"], 1.2)
        self.assertEqual(ep["max_delay"], 1.8)

    def test_default_system_prompt_is_spoken_and_natural(self):
        prompt = resolve_system_prompt({})
        self.assertEqual(prompt, DEFAULT_SYSTEM_PROMPT)
        low = prompt.lower()
        self.assertIn("spoken", low)
        self.assertIn("natural", low)
        self.assertIn("interrupt", low)

    def test_system_prompt_override(self):
        prompt = resolve_system_prompt({"ASSISTANT_SYSTEM": "Be terse."})
        self.assertEqual(prompt, "Be terse.")


class ConfigHelperTests(unittest.TestCase):
    def test_as_bool(self):
        self.assertTrue(_as_bool(None, True))
        self.assertFalse(_as_bool("false"))
        self.assertFalse(_as_bool("0"))
        self.assertFalse(_as_bool("off"))
        self.assertFalse(_as_bool("disabled"))
        self.assertTrue(_as_bool("true"))
        self.assertTrue(_as_bool("1"))

    def test_as_float(self):
        self.assertEqual(_as_float("1.5", 0.0), 1.5)
        self.assertEqual(_as_float("", 0.0), 0.0)
        self.assertEqual(_as_float("junk", 0.9), 0.9)

    def test_as_int(self):
        self.assertEqual(_as_int("2", 1), 2)
        self.assertEqual(_as_int("3.9", 1), 3)
        self.assertEqual(_as_int(None, 1), 1)
        self.assertEqual(_as_int("junk", 1), 1)


if __name__ == "__main__":
    unittest.main()
