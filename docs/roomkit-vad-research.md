# Research: roomkit-live/roomkit — VAD setup

**Status:** Research notes (no code shipped yet)
**Source:** cloned [`roomkit-live/roomkit`](https://github.com/roomkit-live/roomkit)
@ commit `642a649` (README "RoomKit" — local-first realtime voice/AI framework).
Paths below refer to files in that repo.

---

## 1. What RoomKit is (in one paragraph)

RoomKit is a Python realtime voice/conference framework. Audio flows through a
staged `VoicePipeline` on the **inbound** path:

```
Backend -> [Resampler] -> [Recorder] -> [AEC] -> [AGC] -> [Denoiser] -> VAD -> [Diarization] + [DTMF]
```

`VAD` is the **required** stage (everything else is optional/skippable). AEC and
AGC are skipped when the backend declares native capabilities (e.g. native AEC).
The outbound path feeds TTS audio back into AEC as a reference for echo
cancellation. (docs/c7/voice-pipeline.md, src/roomkit/voice/pipeline/engine.py)

---

## 2. VAD architecture

A clean ABC lives in `src/roomkit/voice/pipeline/vad/base.py`:

- **`VADProvider`** — interface with `process(frame, stream) -> VADEvent | None`,
  `reset(stream)`, `close()`. **State is kept per audio `stream`** (a voice
  session and a conference track are separate speakers), so one speaker's
  silence never closes another's utterance.
- **`VADEventType`** — `SPEECH_START`, `SPEECH_END`, `SILENCE`, `AUDIO_LEVEL`.
- **`VADEvent`** — carries `audio_bytes` (pre-roll on START, full accumulated
  segment on END), `confidence`, `duration_ms`, `level_db`.
- **`VADConfig`** — shared knobs: `silence_threshold_ms` (default 500), `speech_pad_ms`
  (default 300), `min_speech_duration_ms` (default 250), plus a per-provider `extra` dict.

Three providers (`src/roomkit/voice/pipeline/vad/`):

| Provider | Detection | Notes |
|----------|-----------|-------|
| `EnergyVADProvider` (`energy.py`) | RMS threshold (`energy_threshold`, int16 scale, default `300.0`) | No deps; local testing / fallback |
| `SherpaOnnxVADProvider` (`sherpa_onnx.py`) | sherpa-onnx neural VAD (**TEN-VAD** or **Silero**) | Production; `pip install roomkit[sherpa-onnx]` |
| `MockVADProvider` (`mock.py`) | preconfigured event sequence | Testing |

---

## 3. The two real VAD providers

### 3.1 `EnergyVADProvider` (energy / RMS)

- `process()` computes per-frame RMS of int16 PCM (`_rms_int16`); `is_speech = rms >= energy_threshold`.
- Maintains a rolling **pre-roll** buffer (`speech_pad_ms`) so utterance onsets
  aren't clipped.
- State machine: idle → (speech) → SPEECH_START (+ pre-roll) → accumulate →
  silence for `silence_threshold_ms` → SPEECH_END; segments shorter than
  `min_speech_duration_ms` are silently discarded. A `max_speech_duration_ms`
  (default 60s) force-ends speech as a safety cap.
- Has rich `DEBUG` summary logging (RMS avg/max, speech/idle frame counts).

### 3.2 `SherpaOnnxVADProvider` (neural — TEN-VAD / Silero)

Key `SherpaOnnxVADConfig` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `model` / `model_type` | — / `"ten"` | `.onnx` path; `ten` (TEN-VAD) or `silero` |
| `threshold` | `0.35` | speech probability threshold. **`0.35` is tuned for denoised audio; raise to `0.5` without a denoiser** |
| `silence_threshold_ms` | 500 | consecutive silence → SPEECH_END |
| `min_speech_duration_ms` / `speech_pad_ms` | 250 / 300 | duration gating / pre-roll |
| `max_speech_duration` | 20.0 s | segment cap |
| `energy_silence_rms` | `20.0` | **energy-based fast exit** (see below); `0` disables |
| `sherpa_min_silence_duration` / `sherpa_min_speech_duration` | 0.05 / 0.1 | keep low; debounce handled by our state machine |
| `sample_rate`, `num_threads`, `provider` | 16000 / 1 / cpu | |

Notable implementation details:

- Each stream gets its **own sherpa `VoiceActivityDetector`** (sherpa holds its own
  probability history, so it can't be shared across speakers).
- `is_speech_detected()` drives the same START/END state machine as energy.
- **Energy-based fast exit (anti-model-inertia):** sherpa's `is_speech_detected()`
  can stay `True` on silence after speech. RoomKit tracks consecutive frames
  below `energy_silence_rms` *independently* and forces SPEECH_END after
  `silence_threshold_ms` of low energy, then calls `detector.reset()` to clear
  the stuck internal state so a false SPEECH_START doesn't re-fire.
- Safety cap forces a segment break and `detector.reset()` at `max_speech_duration`.

---

## 4. Pipeline context that shapes VAD accuracy

RoomKit's VAD doesn't run in isolation — it sits after AEC/AGC/denoiser:

- **Denoiser** (RNNoise, WebRTC NS, SherpaOnnx GTCRN) cleans audio *before* VAD,
  which is why the sherpa `threshold` can stay low (`0.35`). Threshold choice is
  coupled to whether a denoiser is present (`0.5` without).
- **AGC** (`agc/simple.py`, "Adaptive RMS-based gain control") normalizes level
  before VAD so quiet and loud speech reach a predictable RMS.
- **AEC** (`aec/webrtc.py`, `aec/speex.py`) uses an **adaptive echo filter**
  fed the TTS reference; that's the primary defence against the agent
  interrupting itself (speaker echo).

---

## 5. Interruption / barge-in (complements VAD)

`docs/c7/voice-pipeline.md` "Interruption Handling" (`InterruptionConfig`,
`InterruptionStrategy`):

| Strategy | Use |
|----------|-----|
| `IMMEDIATE` | fast, accept false positives |
| `CONFIRMED` | balanced — waits for sustained speech (`min_speech_ms=300`) |
| `SEMANTIC` | ignore backchannel ("uh-huh") via `BackchannelDetector` |
| `DISABLED` | never interrupt |

An `allow_during_first_ms` guard (default e.g. `600` ms) forwards equal-duration
silence to the AI provider right after playback begins, **giving AEC time to
converge** so residual onset echo can't trigger a server-side interruption. Set
it to `0` on a headset / already-clean path when immediate barge-in matters more.

`VADSilenceEvent` (voice/events.py) is documented as enabling "adaptive silence
thresholds" — i.e., the roadmap wants silence-based, adaptive turn boundaries,
but the shipped `VAD` providers use **fixed** thresholds.

---

## 6. Practical setups from the examples

- **`examples/voice_sherpa_onnx_vad.py`** — env-driven: `VAD_MODEL`, `VAD_MODEL_TYPE`
  (`ten`|`silero`), `VAD_THRESHOLD` (default `0.35`); optional `DENOISE_MODEL`
  (GTCRN). 16 kHz mic, 20 ms blocks.
- **`examples/voice_neutts.py`** — `SherpaOnnxVADConfig(threshold=0.35, silence_threshold_ms=600, min_speech_duration_ms=200, speech_pad_ms=300)`.
- **`examples/conference_livekit.py`** — `build_vad()`: use TEN-VAD if the `.onnx`
  model file exists, else fall back to `EnergyVADProvider`; **threshold 0.5**
  "without a denoiser", **silence 700 ms** (conversational pauses run longer
  than the 500 ms default; the "knob to turn" if sentences get cut or drag).
- **`src/roomkit/channels/conference.py`** — defaults to `EnergyVADProvider()`
  when nothing else is set.

---

## 7. Key takeaways vs. Chirpy

**What RoomKit does that we should borrow:**

1. **Stage order matters** — denoiser → AGC → VAD; denoiser is why a low VAD
   threshold is safe. Our stack already has DTLN noise suppression on inbound
   audio.
2. **Per-stream VAD state** — don't let one speaker's silence close another's
   utterance (we have a single user so less critical).
3. **Energy fast-exit** — the neural model's `is_speech_detected()` inertia on
   silence is fixed with an RMS gate + `reset()`; directly relevant to our
   "agent stops without me speaking" echo/tail problem.
4. **Denoiser↔threshold coupling** — document that a denoiser lets you run a
   lower threshold (`0.35`); without it raise to `0.5`.
5. **AEC convergence guard** — `allow_during_first_ms` pauses interruption until
   AEC converges; analogous to our `aec_warmup_duration`.

**What RoomKit does *not* do (gap / our opportunity):**

- Its VAD thresholds are **fixed**, not adaptively derived from the noise floor.
  The only "adaptive" bits are the AEC/AGC filters and the energy fast-exit.
  Our `docs/adaptive-vad.md` + `engine/chirpy/adaptive_vad.py` (adaptive noise-floor /
  adaptive probability-threshold) fills that gap.
- RoomKit's `VADSilenceEvent` roadmap explicitly names "adaptive silence
  thresholds" as a future use-case — consistent with the direction we're taking.

---

## 8. References

- Repo: [roomkit-live/roomkit](https://github.com/roomkit-live/roomkit)
- VAD base: `src/roomkit/voice/pipeline/vad/base.py`
- Energy VAD: `src/roomkit/voice/pipeline/vad/energy.py`
- Sherpa-onnx VAD: `src/roomkit/voice/pipeline/vad/sherpa_onnx.py`
- Pipeline + VAD stage: `src/roomkit/voice/pipeline/engine.py`, `docs/c7/voice-pipeline.md`
- Example wiring: `examples/voice_sherpa_onnx_vad.py`, `examples/voice_neutts.py`,
  `examples/conference_livekit.py`
