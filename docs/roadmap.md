# Build roadmap

## Phase 0 — vertical slice (this repository)

- Push-to-talk recording, local transcription, streamed model text, local speech playback.
- Cancellation is a first-class action: Escape stops playback and cancels the request.
- Provider boundaries are `SpeechToText`, `ChatModel`, and `TextToSpeech`.

## Phase 1 — conversational quality

1. Add voice activity detection and pre-roll, while retaining push-to-talk as a fallback.
2. Maintain short conversation memory and a compact, local summary.
3. Replace per-sentence Piper invocation with a streaming TTS provider if needed.
4. Measure: end-of-speech to first transcript, first token, and first audible sample.

## Phase 2 — screen context

1. Ask for Screen Recording permission and capture only on an explicit hotkey or when sharing is enabled.
2. Downscale and sample frames (for example, one on request), rather than injecting continuous video.
3. Send an image plus a concise task prompt to a local vision-capable model.
4. Render a visible “screen sharing” indicator and keep captured frames in memory only by default.

## Phase 3 — project and coding tools

1. Introduce a tool registry with typed schemas and explicit per-tool permissions.
2. Start read-only: project search, file read, git diff/status, diagnostics.
3. Require approval for writes and shell execution; scope all paths to a selected project.
4. Use a stronger local coding model only for tool-heavy turns.

## Phase 4 — web tools

1. Add a browser/search adapter behind the same tool registry.
2. Show sources in the client and ask before sign-in, purchases, posts, or form submissions.
3. Keep browsing opt-in; it is the one phase that is not fully local.

## Phase 5 — language tutor mode

1. Add a mode prompt and target language profile.
2. Create exercises from the transcript, then speak at an adjustable rate.
3. Use STT confidence and pronunciation-focused feedback; store progress locally.

