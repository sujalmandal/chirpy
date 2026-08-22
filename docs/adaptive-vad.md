# Adaptive VAD — Research & Implementation Design

**Status:** Research / design (no code shipped yet)
**Scope:** How to make voice-activity detection (VAD) adapt to the acoustic
environment — noise floor, non-stationary background, and the agent's own
speaker echo — so barge-in and turn-taking stay natural and robust.

> **Reference code:** a working, unit-tested reference for both adaptive
> flavours lives in `engine/chirpy/adaptive_vad.py` (pure Python, no LiveKit
> dependency), with tests in `tests/test_adaptive_vad.py`. It backs §4 below.

---

## 1. Why "adaptive" VAD?

A fixed-threshold VAD is brittle. The ambient level, noise character, and even
the agent's own speech (echo) change over time. A threshold that is right for a
quiet room will either:

- **miss** quiet or far-from-mic speech in a noisy room (false negatives → the
  assistant doesn't hear you), or
- **over-trigger** on noise/typing/echo (false positives → the assistant stops
  for no reason, which is exactly the bug we've been chasing in `echoguard.py`).

"Adaptive VAD" means the decision boundary is **derived from the live signal**
(an estimated noise floor, ambient level, or a rolling distribution of the
model's per-frame speech probability) rather than being a fixed constant.

---

## 2. Approaches surveyed

### 2.1 Adaptive energy / noise-floor VAD (canonical)

The most widely reused design is energy-based with an **adaptively tracked noise
floor** and a **hangover** (keep "speaking" a few frames after the last speech
frame so you don't chop mid-word).

Canonical implementation — [AssemblyAI EnergyVad](https://assemblyai.github.io/assemblyai-node-sdk/classes/EnergyVad.html):

```js
process(frame) {
  rms = sqrt(mean(frame[i]^2))
  threshold = noiseFloor * thresholdRatio          // ratio ≈ 3 → ~9.5 dB above floor
  active = rms > threshold
  if (active)            hangoverRemaining = hangoverFrames
  else if (hangoverRemaining > 0) { hangoverRemaining--; active = true }   // in hangover
  else noiseFloor = noiseFloor*(1 - noiseFloorAlpha) + rms*noiseFloorAlpha // EMA on silence only
  return { active, energy: rms }
}
```

Key parameters and their tuning notes (from the source):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `thresholdRatio` | 3.0 | below 2 → too sensitive (treats noise as speech); above 6 → misses quiet onsets |
| `noiseFloorAlpha` | 0.05 | above 0.1 → floor tracks fast (good for non-stationary noise) but can drift up to swallow a sustained quiet voice |
| `hangoverFrames` | 10 (~200ms @20ms) | length of the "still speaking" tail after the last speech frame |
| `initialNoiseFloor` | 1e-4 | seed; adaptive after first non-speech frame |

Related energy/WebRTC-style detectors: [aprender `vad/mod.rs`](https://docs.rs/aprender/0.26.3/src/aprender/speech/vad/mod.rs.html)
implements the same WebRTC energy idea (`energy_threshold`, frame/hop sizes,
RMS-per-frame). The core trick that makes it adaptive: **update the noise floor
only while NOT in speech**, so a sustained loud voice doesn't slowly raise the
floor and then get misclassified as noise.

### 2.2 Adaptive threshold by ambient noise level

Instead of continuously tracking a floor, classify the environment and pick a
threshold bucket. [Continuum VAD production config](https://github.com/CambrianTech/continuum/blob/main/docs/live/VAD-PRODUCTION-CONFIG.md)
does exactly this:

```rust
let threshold = match noise_level {
    NoiseLevel::Quiet    => 0.4,
    NoiseLevel::Moderate => 0.3,
    NoiseLevel::Loud     => 0.25,   // lower in noise → catch more speech
};
```

Same goal as 2.1 but coarse-grained: simpler, more stable, but reacts slower to
a changing environment.

### 2.3 Model-based VAD (Silero) with adaptive thresholds

Modern DNN VADs (Silero, Marat) emit a **per-frame speech probability** in
[0,1] rather than a hard on/off. Silero uses an `activation_threshold` (to start
speech) and a lower `deactivation_threshold` (to end it) — a hysteresis pair
that prevents jitter. Our local plugin exposes both and can update them live:

- `activation_threshold` (default `0.5`): probability at/above which speech is
  considered present.
- `deactivation_threshold` (default `max(activation_threshold - 0.15, 0.01)`):
  once in speech, values *below* this are non-speech.
- per-frame probability arrives on every `INFERENCE_DONE` event, and
  `update_options(...)` can change the thresholds at runtime.

Adapting this model-based path means making `activation_threshold` (and the
deactivation gap) **depend on the observed distribution of the probability
during silence** (see §4, flavor A) instead of a constant `0.5`.

Silero-specific tuning discussion: [Silero VAD threshold tuning — balancing
precision/recall](https://adg.csdn.net/696f3cdb437a6b403369bc6f.html).

### 2.4 Two-stage VAD (fast prefilter + DNN confirm)

[Continuum](https://github.com/CambrianTech/continuum/blob/main/docs/live/VAD-PRODUCTION-CONFIG.md)
also recommends a two-stage pipeline: a **cheap WebRTC energy VAD (1–10µs)** as
a pre-filter, and **Silero (54ms/frame)** only to confirm speech candidates.
This slashes CPU on silence while keeping DNN accuracy. It is orthogonal to the
adaptive threshold above and can be combined.

### 2.5 Note: LiveKit's "adaptive interruption" is not the same thing

LiveKit's [Adaptive Interruption Handling](https://livekit.com/blog/adaptive-interruption-handling)
is an ML model that decides whether overlapping speech is a *genuine barge-in*
versus a backchannel/noise — a *speaker-intent* model on top of VAD, and it's a
**cloud service** (not available self-hosted). Adaptive VAD is a separate,
lower-level concern: "was there speech at all?" The two complement each other.

---

## 3. Where this plugs into Chirpy today

The codebase already uses Silero VAD in two places:

1. **AgentSession** (`engine/chirpy/agent.py`): `vad=silero.VAD.load()` powers
   turn detection and barge-in. LiveKit's `AgentSession` accepts any
   `agents.vad.VAD`, so a custom adaptive VAD can be passed here directly.
2. **Streaming STT** (`engine/chirpy/plugins/whisper_stt.py`): an internal
   `silero.VAD.load()` segments the mic stream for pseudo-streaming transcripts.

The LiveKit `agents.vad.VAD`/`VADStream` interface (see
`livekit/agents/vad.py` in the venv):

- `VAD.stream() -> VADStream`; `VADStream` emits `VADEvent`s:
  - `INFERENCE_DONE` — carries `probability` (the model's speech confidence),
  - `START_OF_SPEECH` / `END_OF_SPEECH` — the derived segment boundaries.
- Silero's stream emits `INFERENCE_DONE` every 32ms window with `probability=p`,
  smoothed by an `ExpFilter(alpha=0.35)`.

This is the hook an adaptive VAD needs: **read `probability` per frame, adapt
the threshold, and emit START/END accordingly** — while staying a drop-in
`agents.vad.VAD` for `AgentSession`.

---

## 4. Recommended implementation design for Chirpy

Implement an `AdaptiveVAD(agents.vad.VAD)` subclass that wraps the Silero model
and adapts its threshold. Two compatible flavors; flavor A is recommended.

### Flavor A — Adaptive threshold on the Silero speech probability

Model a rolling distribution of the *non-speech* probability and place the
threshold adaptively. Concretely:

- Keep a short ring buffer of recent `probability` values that were classified
  **non-speech** (like the noise floor update in §2.1 — update only on silence).
- Maintain an adaptive estimate, e.g. a percentile / exponential moving average:
  `activation_threshold = clamp(p50_non_speech * margin, lo, hi)`.
  - `margin` ~ 2–3 (analogous to `thresholdRatio`), so speech must clear the
    observed silence distribution by a comfortable margin.
- `deactivation_threshold = max(activation_threshold - hysteresis_gap, min)`
  to keep the anti-jitter hysteresis.
- Apply via the underlying Silero `update_options(activation_threshold=…,
  deactivation_threshold=…)` (or gate internally if we replicate the state
  machine).
- EMA/percentile parameters (margins, buffer length) come from config and can be
  tuned live through the existing `ConfigWatcher` (see `bargein.py`).

Rationale: this is exactly the energy-approach from §2.1 transplanted onto the
model's confidence axis, so it inherits the well-studied tuning guidance.

### Flavor B — Adaptive energy noise-floor VAD (simplest, no model)

If a lighter-weight option is preferred (or as the WebRTC prefilter of §2.4),
port the [EnergyVad](https://assemblyai.github.io/assemblyai-node-sdk/classes/EnergyVad.html)
algorithm to numpy as a `VAD` subclass: compute per-frame RMS, track the noise
floor with EMA on non-speech frames only, classify `rms > noiseFloor * thresholdRatio`,
with hangover. This is ~40 lines and very cheap, but less accurate than Silero
on background-music/non-stationary noise.

### Integration

- Pass `AdaptiveVAD(...)` as `vad=` in `build_session` (`engine/chirpy/agent.py`).
- Expose its tuning params through the existing validated policy in
  `engine/chirpy/bargein.py` and `config/barge-in.json`, so it stays data-driven
  and live-tunable (no hardcoded numbers).
- Keep `echoguard.py` as a second, content-based layer for the specific
  speaker-echo failure mode (overlap with the agent's own recent speech), which
  pure VAD/energy can't solve.

### Pseudocode (Flavor A)

```python
class AdaptiveSileroVAD(agents.vad.VAD):
    def __init__(self, base: silero.VAD, *, margin, buffer, min_, max_, gap, alpha):
        self._base = base
        self._silence_probs = deque(maxlen=buffer)
        self.threshold = 0.5
        self.deactivation = 0.35

    def _on_inference(self, p: float):
        # p from the wrapped stream's INFERENCE_DONE
        is_speech = p >= self.threshold
        if not is_speech and self._in_speech_state(p):
            self._silence_probs.append(p)                    # update only on silence
        if not is_speech:
            # adapt threshold from observed silence distribution
            noise = percentile(self._silence_probs, 0.5) or 1e-4
            new = clamp(noise * margin, lo, hi)
            if abs(new - self.threshold) > step:            # debounce
                self.threshold = new
                self.deactivation = max(new - gap, min)
                self._base.update_options(
                    activation_threshold=new,
                    deactivation_threshold=self.deactivation)
```

---

## 5. Parameter guidance (from research)

| Param | Analogue in §2 | Guidance |
|-------|----------------|----------|
| threshold margin (`margin`) | `thresholdRatio` (AssemblyAI) | 2–3 ≈ +6–9.5 dB above silence; < 2 too sensitive, > 6 misses onsets |
| adaptation speed (`alpha`) | `noiseFloorAlpha` (AssemblyAI) | > 0.1 tracks fast but can drift up on a quiet sustained voice; start 0.05 |
| hysteresis gap | `deactivation_threshold` (Silero) | keep a gap so we don't flutter at the boundary |
| buffer length | — | seconds of silence to estimate the floor; 1–3s |
| min/max clamp | continuum buckets (0.25–0.4) | keep threshold in a sane band |
| hangover / min_silence | `hangoverFrames`, `min_silence_duration` | keep natural end-of-utterance; avoids chopping |

---

## 6. Testing strategy

- **Unit**: feed synthesized frames (silence, white noise at various levels, a
  sine tone) through the adaptive VAD and assert START/END boundaries and that
  the threshold moves correctly with injected noise levels. Pure functions
  (noise-floor estimator, threshold resolver) are easy to unit test.
  **Done for the reference** — `tests/test_adaptive_vad.py` covers silence vs.
  tone, hangover, noise-floor adaptation raising/lowering the threshold, and
  that a speech frame never pollutes the noise floor.
- **Echo robustness**: reuse `echo_overlap` (in `bargein.py`) to assert that
  the agent's own recent text is never treated as a user barge-in.
- **Live tuning**: extend the `ConfigWatcher` tests to verify threshold changes
  reach the VAD at runtime.

---

## 7. References

- [AssemblyAI EnergyVad — adaptive noise-floor VAD](https://assemblyai.github.io/assemblyai-node-sdk/classes/EnergyVad.html)
- [aprende `vad/mod.rs` — WebRTC-style energy VAD](https://docs.rs/aprender/0.26.3/src/aprender/speech/vad/mod.rs.html)
- [Continuum VAD production config — two-stage + adaptive threshold by noise level](https://github.com/CambrianTech/continuum/blob/main/docs/live/VAD-PRODUCTION-CONFIG.md)
- [LiveKit adaptive interruption handling (cloud, ML-based; complement to VAD)](https://livekit.com/blog/adaptive-interruption-handling)
- [Silero VAD threshold tuning (precision/recall)](https://adg.csdn.net/696f3cdb437a6b403a6bc6f.html)
- [Silero VAD plugin (this repo's venv) — `activation_threshold`, `deactivation_threshold`, `update_options`, per-frame `probability`](https://github.com/livekit/agents)

---

## 8. Next steps / open questions

1. **Choose flavor**: A (adaptive Silero probability threshold — recommended for
   accuracy) or B (adaptive energy — simplest) or A + B (two-stage).
2. Decide if this becomes a drop-in `vad=` for `AgentSession`, a separate stream
   the `whisper_stt` uses, or both.
3. Wire tuning params into `bargein.py` / `config/barge-in.json` for live tuning.
4. Implement + unit test (synthetic silence/noise/speech fixtures) before a live
   mic test against changing background noise and speaker echo.
