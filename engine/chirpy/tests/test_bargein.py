"""Tests for the validated, data-driven barge-in policy and echo guard.

Covers :mod:`bargein` (policy loading, clamping, adaptive->vad fallback,
echo-overlap scoring, runtime ConfigWatcher) and :mod:`echoguard`.
Pure unit tests — no LiveKit server, models, or LLM endpoint needed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[1]  # engine/chirpy
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from bargein import (  # noqa: E402
    ConfigWatcher,
    DEFAULT_ECHO_OVERLAP_THRESHOLD,
    LK_INTERRUPTION_DEFAULTS,
    echo_overlap,
    has_adaptive_credentials,
    load_barge_in_policy,
    BargeInPolicy,
)
from echoguard import EchoGuard  # noqa: E402


class PolicyDefaultsTests(unittest.TestCase):
    def test_defaults_mirror_livekit(self):
        p = load_barge_in_policy({})
        self.assertIs(p.enabled, True)
        self.assertEqual(p.mode, "vad")
        self.assertEqual(p.min_duration, LK_INTERRUPTION_DEFAULTS["min_duration"])
        self.assertEqual(p.min_words, LK_INTERRUPTION_DEFAULTS["min_words"])
        self.assertEqual(p.false_interruption_timeout, LK_INTERRUPTION_DEFAULTS["false_interruption_timeout"])
        self.assertEqual(p.backchannel_boundary, LK_INTERRUPTION_DEFAULTS["backchannel_boundary"])
        self.assertAlmostEqual(p.aec_warmup_duration, 3.0)
        self.assertAlmostEqual(p.echo_overlap_threshold, DEFAULT_ECHO_OVERLAP_THRESHOLD)

    def test_interruption_options_shape(self):
        opts = load_barge_in_policy({}).interruption_options()
        for key in (
            "enabled", "mode", "min_duration", "min_words", "resume_false_interruption",
            "false_interruption_timeout", "backchannel_boundary", "discard_audio_if_uninterruptible",
        ):
            self.assertIn(key, opts)

    def test_endpointing_dynamic_sane_bounds(self):
        ep = load_barge_in_policy({}).endpointing_options()
        self.assertEqual(ep["mode"], "dynamic")
        self.assertGreater(ep["max_delay"], ep["min_delay"])
        self.assertGreaterEqual(ep["min_delay"], 0.0)

    def test_overrides_applied(self):
        p = load_barge_in_policy(
            {
                "BARGE_IN": "false",
                "BARGE_IN_MIN_DURATION": "1.2",
                "BARGE_IN_MIN_WORDS": "3",
                "BARGE_IN_FALSE_TIMEOUT": "4.0",
                "AEC_WARMUP_DURATION": "5",
                "ECHO_OVERLAP_THRESHOLD": "0.8",
            }
        )
        self.assertIs(p.enabled, False)
        self.assertEqual(p.min_duration, 1.2)
        self.assertEqual(p.min_words, 3)
        self.assertEqual(p.false_interruption_timeout, 4.0)
        self.assertEqual(p.aec_warmup_duration, 5.0)
        self.assertEqual(p.echo_overlap_threshold, 0.8)


class PolicyClampingTests(unittest.TestCase):
    def test_negative_values_clamp_to_zero(self):
        p = load_barge_in_policy(
            {"BARGE_IN_MIN_DURATION": "-2", "BARGE_IN_MIN_WORDS": "-5"}
        )
        self.assertEqual(p.min_duration, 0.0)
        self.assertEqual(p.min_words, 0)

    def test_echo_threshold_clamped_to_unit_range(self):
        p = load_barge_in_policy({"ECHO_OVERLAP_THRESHOLD": "99"})
        self.assertEqual(p.echo_overlap_threshold, 1.0)
        p = load_barge_in_policy({"ECHO_OVERLAP_THRESHOLD": "-1"})
        self.assertEqual(p.echo_overlap_threshold, 0.0)

    def test_endpointing_bounds_swapped_when_min_gt_max(self):
        p = load_barge_in_policy({"ENDPOINTING_MIN_DELAY": "5", "ENDPOINTING_MAX_DELAY": "1"})
        ep = p.endpointing_options()
        self.assertLessEqual(ep["min_delay"], ep["max_delay"])
        self.assertEqual((ep["min_delay"], ep["max_delay"]), (1.0, 5.0))

    def test_invalid_mode_falls_back_to_vad(self):
        self.assertEqual(load_barge_in_policy({"INTERRUPTION_MODE": "quantum"}).mode, "vad")


class AdaptiveFallbackTests(unittest.TestCase):
    def test_adaptive_without_credentials_falls_back_to_vad(self):
        self.assertFalse(has_adaptive_credentials({}))
        p = load_barge_in_policy({"INTERRUPTION_MODE": "adaptive"})
        self.assertEqual(p.mode, "vad")

    def test_adaptive_with_credentials_stays(self):
        cfg = {"INTERRUPTION_MODE": "adaptive", "LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "s"}
        self.assertTrue(has_adaptive_credentials(cfg))
        self.assertEqual(load_barge_in_policy(cfg).mode, "adaptive")

    def test_inference_api_key_enables_adaptive(self):
        cfg = {"INTERRUPTION_MODE": "adaptive", "LIVEKIT_INFERENCE_API_KEY": "k"}
        self.assertTrue(has_adaptive_credentials(cfg))


class BackchannelParsingTests(unittest.TestCase):
    def test_off_disables(self):
        p = load_barge_in_policy({"BARGE_IN_BACKCHANNEL": "off"})
        self.assertIsNone(p.backchannel_boundary)

    def test_auto_default_pair(self):
        p = load_barge_in_policy({"BARGE_IN_BACKCHANNEL": "auto"})
        self.assertEqual(p.backchannel_boundary, LK_INTERRUPTION_DEFAULTS["backchannel_boundary"])

    def test_pair_parsed(self):
        p = load_barge_in_policy({"BARGE_IN_BACKCHANNEL": "0.5,1.2"})
        self.assertEqual(p.backchannel_boundary, (0.5, 1.2))

    def test_single_parsed(self):
        p = load_barge_in_policy({"BARGE_IN_BACKCHANNEL": "1.5"})
        self.assertEqual(p.backchannel_boundary, (1.5, 1.5))


class FalseTimeoutTests(unittest.TestCase):
    def test_numeric(self):
        self.assertEqual(load_barge_in_policy({"BARGE_IN_FALSE_TIMEOUT": "3.5"}).false_interruption_timeout, 3.5)

    def test_off_disables_resume(self):
        self.assertIsNone(load_barge_in_policy({"BARGE_IN_FALSE_TIMEOUT": "off"}).false_interruption_timeout)


class EchoOverlapTests(unittest.TestCase):
    def test_exact_echo_high(self):
        self.assertEqual(echo_overlap("the quick brown fox jumps", "quick brown fox jumps"), 1.0)

    def test_unrelated_zero(self):
        self.assertEqual(echo_overlap("please play some music now", "the weather is fine today"), 0.0)

    def test_echo_prefix_high(self):
        self.assertGreaterEqual(echo_overlap("the weather", "the weather is fine today"), 0.8)

    def test_empty_inputs_zero(self):
        self.assertEqual(echo_overlap("", "hello world"), 0.0)
        self.assertEqual(echo_overlap("hello", ""), 0.0)


class EchoGuardTests(unittest.TestCase):
    def test_echo_suppressed(self):
        guard = EchoGuard(load_barge_in_policy({}))
        guard.note_assistant_text("the weather is fine today")
        self.assertTrue(guard.is_echo("the weather"))
        self.assertEqual(guard.echo_suppressed, 1)

    def test_genuine_input_not_echo(self):
        guard = EchoGuard(load_barge_in_policy({}))
        guard.note_assistant_text("the weather is fine today")
        self.assertFalse(guard.is_echo("please play some music"))
        self.assertEqual(guard.echo_suppressed, 0)

    def test_no_agent_text_no_echo(self):
        guard = EchoGuard(load_barge_in_policy({}))
        self.assertFalse(guard.is_echo("anything"))
        self.assertEqual(guard.echo_suppressed, 0)


class ConfigWatcherTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "barge-in.json"

    def _write(self, obj):
        self.path.write_text(json.dumps(obj))

    def test_loads_policy_from_file_merged_over_base(self):
        watcher = ConfigWatcher(self.path, on_change=lambda p: None, base_config={"BARGE_IN": "true"})
        self._write({"BARGE_IN_MIN_DURATION": 1.8, "BARGE_IN_MIN_WORDS": 2})
        policy = watcher.load_policy()
        self.assertEqual(policy.min_duration, 1.8)
        self.assertEqual(policy.min_words, 2)
        self.assertIs(policy.enabled, True)  # base preserved

    def test_on_change_fired_on_poll(self):
        seen = {}
        watcher = ConfigWatcher(self.path, on_change=lambda p: seen.update(policy=p))
        self._write({"ECHO_OVERLAP_THRESHOLD": 0.9})
        self.assertTrue(watcher.poll_once())
        self.assertEqual(seen["policy"].echo_overlap_threshold, 0.9)
        # unchanged file -> no callback
        self.assertFalse(watcher.poll_once())

    def test_missing_file_uses_base(self):
        watcher = ConfigWatcher(self.path, on_change=lambda p: None, base_config={"BARGE_IN_MIN_WORDS": "4"})
        self.assertEqual(watcher.load_policy().min_words, 4)

    def test_bad_json_ignored_gracefully(self):
        watcher = ConfigWatcher(self.path, on_change=lambda p: None, base_config={})
        self.path.write_text("not json")
        # Should not raise; falls back to base policy.
        policy = watcher.load_policy()
        self.assertEqual(policy.min_words, LK_INTERRUPTION_DEFAULTS["min_words"])


if __name__ == "__main__":
    unittest.main()
