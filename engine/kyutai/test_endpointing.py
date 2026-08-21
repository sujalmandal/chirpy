import unittest

from endpointing import (
    BargeInGate,
    EndpointDetector,
    EndpointState,
    is_probable_playback_echo,
    recognized_barge_in_ready,
)


class EndpointDetectorTests(unittest.TestCase):
    def detector(self, **overrides):
        values = dict(warmup_blocks=0, min_speech_ms=160, min_silence_ms=160)
        values.update(overrides)
        return EndpointDetector(**values)

    def test_does_not_endpoint_before_recognized_text(self):
        detector = self.detector()
        for _ in range(8):
            decision = detector.observe(
                rms=0.08, semantic_probability=0.95, has_recognized_text=False
            )
        self.assertIsNone(decision)
        self.assertEqual(detector.state, EndpointState.WAITING)

    def test_semantic_endpoint_requires_text_and_hold(self):
        detector = self.detector(semantic_silence_ms=240, semantic_hold_blocks=3)
        detector.observe(
            rms=0.08, semantic_probability=0.1,
            has_recognized_text=True, new_recognized_text=True,
        )
        self.assertIsNone(detector.observe(
            rms=0.001, semantic_probability=0.9, has_recognized_text=True
        ))
        self.assertIsNone(detector.observe(
            rms=0.001, semantic_probability=0.9, has_recognized_text=True
        ))
        decision = detector.observe(
            rms=0.001, semantic_probability=0.9, has_recognized_text=True
        )
        self.assertEqual(decision.reason, "semantic_vad")

    def test_short_mid_sentence_pause_does_not_end_turn(self):
        detector = self.detector(
            min_silence_ms=800, semantic_silence_ms=320, semantic_hold_blocks=3
        )
        detector.observe(
            rms=0.08, semantic_probability=0.1,
            has_recognized_text=True, new_recognized_text=True,
        )
        for _ in range(3):
            decision = detector.observe(
                rms=0.001, semantic_probability=0.95, has_recognized_text=True
            )
        self.assertIsNone(decision)
        detector.observe(
            rms=0.08, semantic_probability=0.95, has_recognized_text=True
        )
        self.assertEqual(detector.pause_run, 0)

    def test_new_token_cancels_pending_semantic_endpoint(self):
        detector = self.detector(semantic_silence_ms=320, semantic_hold_blocks=3)
        detector.observe(
            rms=0.08, semantic_probability=0.1,
            has_recognized_text=True, new_recognized_text=True,
        )
        for _ in range(2):
            detector.observe(
                rms=0.001, semantic_probability=0.95, has_recognized_text=True
            )
        detector.observe(
            rms=0.001, semantic_probability=0.95,
            has_recognized_text=True, new_recognized_text=True,
        )
        self.assertEqual(detector.pause_run, 0)

    def test_warmup_suppresses_unstable_semantic_frames(self):
        detector = self.detector(warmup_blocks=3)
        detector.observe(rms=0.08, semantic_probability=0.95, has_recognized_text=True)
        detector.observe(rms=0.001, semantic_probability=0.95, has_recognized_text=False)
        self.assertIsNone(
            detector.observe(rms=0.001, semantic_probability=0.95, has_recognized_text=False)
        )
        self.assertIsNone(
            detector.observe(rms=0.001, semantic_probability=0.95, has_recognized_text=False)
        )

    def test_adaptive_energy_fallback_ends_a_text_turn(self):
        detector = self.detector(semantic_hold_blocks=10)
        detector.observe(rms=0.08, semantic_probability=0.1, has_recognized_text=True)
        detector.observe(rms=0.08, semantic_probability=0.1, has_recognized_text=False)
        detector.observe(rms=0.001, semantic_probability=0.1, has_recognized_text=False)
        decision = detector.observe(rms=0.001, semantic_probability=0.1, has_recognized_text=False)
        self.assertEqual(decision.reason, "adaptive_silence_timeout")

    def test_energy_fallback_remembers_speech_until_delayed_text_arrives(self):
        detector = self.detector(semantic_hold_blocks=10)
        detector.observe(rms=0.08, semantic_probability=0.1, has_recognized_text=False)
        detector.observe(rms=0.08, semantic_probability=0.1, has_recognized_text=False)
        detector.observe(rms=0.001, semantic_probability=0.1, has_recognized_text=False)
        decision = detector.observe(
            rms=0.001, semantic_probability=0.1, has_recognized_text=True
        )
        self.assertEqual(decision.reason, "adaptive_silence_timeout")

    def test_separated_noise_bursts_do_not_accumulate_consecutive_energy(self):
        detector = self.detector()
        detector.observe(rms=0.08, semantic_probability=0.1, has_recognized_text=True)
        detector.observe(rms=0.001, semantic_probability=0.1, has_recognized_text=False)
        detector.observe(rms=0.08, semantic_probability=0.1, has_recognized_text=False)
        detector.observe(rms=0.001, semantic_probability=0.1, has_recognized_text=False)
        self.assertEqual(detector.energy_run, 0)

    def test_noise_floor_raises_energy_gate_in_a_loud_room(self):
        detector = self.detector(base_energy_threshold=0.005)
        for _ in range(200):
            detector.observe(rms=0.02, semantic_probability=0.1, has_recognized_text=False)
        self.assertGreater(detector.energy_threshold, 0.04)

    def test_delayed_first_token_does_not_train_noise_floor_on_speech(self):
        detector = self.detector(base_energy_threshold=0.01)
        for _ in range(25):
            detector.observe(rms=0.002, semantic_probability=0.1, has_recognized_text=False)
        baseline = detector.energy_threshold
        for _ in range(7):
            detector.observe(rms=0.08, semantic_probability=0.1, has_recognized_text=False)
        self.assertLess(detector.energy_threshold, baseline * 1.25)


class BargeInGateTests(unittest.TestCase):
    def test_short_echo_spike_does_not_interrupt(self):
        gate = BargeInGate(blind_ms=1200, min_speech_ms=640)
        for index in range(5):
            interrupted = gate.observe(
                elapsed_ms=1200 + index * 80, rms=0.09, threshold=0.05
            )
        self.assertFalse(interrupted)

    def test_sustained_residual_signal_interrupts(self):
        gate = BargeInGate(blind_ms=1200, min_speech_ms=640)
        results = [
            gate.observe(elapsed_ms=1200 + index * 80, rms=0.09, threshold=0.05)
            for index in range(8)
        ]
        self.assertEqual(results, [False] * 7 + [True])

    def test_quiet_block_resets_speech_evidence(self):
        gate = BargeInGate(blind_ms=1200, min_speech_ms=640)
        for index in range(6):
            gate.observe(elapsed_ms=1200 + index * 80, rms=0.09, threshold=0.05)
        gate.observe(elapsed_ms=1680, rms=0.02, threshold=0.05)
        for index in range(6):
            interrupted = gate.observe(
                elapsed_ms=1760 + index * 80, rms=0.09, threshold=0.05
            )
        self.assertFalse(interrupted)

    def test_blind_period_does_not_accumulate_evidence(self):
        gate = BargeInGate(blind_ms=1200, min_speech_ms=640)
        for index in range(12):
            gate.observe(elapsed_ms=index * 80, rms=0.09, threshold=0.05)
        self.assertEqual(gate.run, 0)

    def test_uncalibrated_echo_cannot_trigger_energy_barge_in(self):
        gate = BargeInGate(blind_ms=1200, min_speech_ms=640)
        results = [
            gate.observe(
                elapsed_ms=1200 + index * 80,
                rms=0.09,
                threshold=0.05,
                calibrated=False,
            )
            for index in range(12)
        ]
        self.assertFalse(any(results))
        self.assertEqual(gate.run, 0)


class PlaybackEchoTests(unittest.TestCase):
    def test_exact_reply_fragment_is_echo(self):
        self.assertTrue(is_probable_playback_echo(
            "If you're curious about",
            "If you're curious about how it works, I can explain it.",
        ))

    def test_minor_recognition_error_is_echo(self):
        self.assertTrue(is_probable_playback_echo(
            "the quick brown fox jumps",
            "The quick brown box jumps over the fence.",
        ))

    def test_unrelated_user_interruption_is_not_echo(self):
        self.assertFalse(is_probable_playback_echo(
            "Please change the subject now",
            "I can continue explaining how the system works.",
        ))

    def test_short_phrase_is_not_suppressed_by_text_alone(self):
        self.assertFalse(is_probable_playback_echo(
            "hold on",
            "You can hold on from the menu.",
        ))

    def test_residual_energy_allows_short_command(self):
        self.assertTrue(recognized_barge_in_ready(
            "hello", residual_confirmed=True
        ))

    def test_multiword_speech_still_requires_acoustic_confirmation(self):
        self.assertFalse(recognized_barge_in_ready(
            "please change direction", residual_confirmed=False
        ))

    def test_single_unconfirmed_token_is_not_enough(self):
        self.assertFalse(recognized_barge_in_ready(
            "yeah", residual_confirmed=False
        ))


if __name__ == "__main__":
    unittest.main()
