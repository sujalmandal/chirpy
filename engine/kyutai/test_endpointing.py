import unittest

from endpointing import EndpointDetector, EndpointState


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
        detector = self.detector()
        detector.observe(rms=0.08, semantic_probability=0.1, has_recognized_text=True)
        first = detector.observe(rms=0.001, semantic_probability=0.9, has_recognized_text=False)
        second = detector.observe(rms=0.001, semantic_probability=0.9, has_recognized_text=False)
        self.assertIsNone(first)
        self.assertEqual(second.reason, "semantic_vad")

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


if __name__ == "__main__":
    unittest.main()
