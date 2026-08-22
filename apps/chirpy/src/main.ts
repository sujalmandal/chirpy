import { invoke } from "@tauri-apps/api/core";
import { emit, listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { load } from "@tauri-apps/plugin-store";
import { Room, RoomEvent, Track } from "livekit-client";
import "./styles.css";

// Suppress the webview's default right-click context menu (which includes
// Reload/Inspect) — Chirpy has its own controls.
window.addEventListener("contextmenu", (e) => e.preventDefault());

const LIVEKIT_URL = "ws://127.0.0.1:7880";
const ROOM_NAME = "chirpy";

interface Settings {
  agentName: string;
  systemPrompt: string;
  llmURL: string;
  llmModel: string;
  llmAPIKey: string;
  sttModel: string;
  sttLanguage: string;
  ttsVoice: string;
  ttsLang: string;
}

const defaultSettings: Settings = {
  agentName: "Chirpy",
  systemPrompt: `You are {{agent_name}}, an AI voice assistant.

Identity rules:

- Your name is {{agent_name}}.
- {{agent_name}} always refers to you, the assistant, not the user.
- The user's name is unknown unless they explicitly tell you their name.
- Never call, greet, or address the user as {{agent_name}}.
- Do not assume the user's name from the conversation, system instructions, metadata, or examples.
- If the user provides their name, remember it for the conversation and use it only when natural.
- If you are uncertain who a name refers to, ask for clarification instead of guessing.

Conversation style:

- Speak in a warm, natural, and confident manner.
- Keep responses concise and suitable for spoken conversation.
- Prefer one or two short paragraphs unless the user requests detail.
- Answer directly without unnecessary introductions, repeated greetings, or restating the question.
- Do not introduce yourself repeatedly. Mention your name only when asked or when contextually useful.
- Avoid overly formal language, filler phrases, and repetitive acknowledgements.
- Use clear sentences that sound natural when spoken aloud.
- Ask only one clarification question at a time when more information is required.
- If the user interrupts or changes the subject, follow their latest request naturally.

Accuracy and behavior:

- Do not invent facts, personal details, or conversation history.
- Clearly acknowledge uncertainty when you do not know something.
- Correct misunderstandings briefly and respectfully.
- Follow the user's requested language and communication style when possible.
- Provide longer explanations only when requested or when additional detail is necessary for safety or correctness.`,
  llmURL: "http://localhost:1234/v1",
  llmModel: "",
  llmAPIKey: "",
  sttModel: "tiny",
  sttLanguage: "en",
  ttsVoice: "af_heart",
  ttsLang: "a",
};

let settings: Settings = { ...defaultSettings };

async function loadSettings() {
  try {
    const store = await load("settings.json", { autoSave: true });
    const stored = (await store.get<Partial<Settings>>("settings")) ?? {};
    settings = { ...defaultSettings, ...stored };
  } catch {
    settings = { ...defaultSettings };
  }
  // A voice that isn't in the model's voices/ dir would 404 on the worker, so
  // fall back to the default rather than forwarding a bad id.
  if (!TTS_VOICES.includes(settings.ttsVoice)) {
    settings.ttsVoice = defaultSettings.ttsVoice;
  }
}

async function saveSettings() {
  try {
    const store = await load("settings.json", { autoSave: true });
    await store.set("settings", settings);
  } catch {
    /* ignore */
  }
}

function engineEnvironment(): Record<string, string> {
  const name = settings.agentName.trim();
  const resolved = settings.systemPrompt.replaceAll("{{agent_name}}", name);
  const useRemote = settings.llmModel.trim() !== "" && settings.llmURL.trim() !== "";
  return {
    AGENT_NAME: name,
    ASSISTANT_SYSTEM: resolved,
    // A blank model uses the bundled in-process Qwen; setting an endpoint +
    // model switches the engine to that OpenAI-compatible LLM.
    LLM_BACKEND: useRemote ? "remote" : "local",
    LLM_BASE_URL: settings.llmURL.trim(),
    LLM_MODEL_NAME: settings.llmModel.trim(),
    LLM_API_KEY: settings.llmAPIKey,
    STT_MODEL: settings.sttModel.trim(),
    STT_LANGUAGE: settings.sttLanguage.trim(),
    TTS_VOICE: settings.ttsVoice.trim(),
    TTS_LANG: settings.ttsLang.trim(),
  };
}

// ---------------------------------------------------------------------------
// LiveKit voice session
// ---------------------------------------------------------------------------

class VoiceSession {
  private room: Room | null = null;
  private audioCtx: AudioContext | null = null;
  private listening = false;
  private speaking = false;
  private outputMuted = false;
  private playbackEls: HTMLAudioElement[] = [];
  private reply = "";
  private transcript = "";

  onStatus: (s: string) => void = () => {};
  onTranscript: (t: string, final?: boolean) => void = () => {};
  onReply: (t: string, final?: boolean) => void = () => {};
  onSpeaking: (b: boolean) => void = () => {};
  onListening: (b: boolean) => void = () => {};
  onUserMessage: (text: string) => void = () => {};
  onUserPartial: (text: string) => void = () => {};
  onAssistantStart: () => void = () => {};
  onAssistantDelta: (delta: string) => void = () => {};
  onAssistantEnd: () => void = () => {};
  onLatency: (payload: Record<string, unknown>) => void = () => {};

  get isListening() {
    return this.listening;
  }
  get isSpeaking() {
    return this.speaking;
  }
  get isOutputMuted() {
    return this.outputMuted;
  }

  async start() {
    if (this.listening) return;
    this.listening = true;
    this.onListening(true);
    this.onStatus("Connecting to Chirpy…");
    try {
      const token = await invoke<string>("get_token", { room: ROOM_NAME });
      const room = new Room({ adaptiveStream: true, dynacast: true });
      this.room = room;

      room
        .on(RoomEvent.TrackSubscribed, (track) => {
          if (track.kind === Track.Kind.Audio) this.attachPlayback(track);
        })
        .on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
          const speaking = speakers.length > 0;
          this.speaking = speaking;
          this.onSpeaking(speaking);
        })
        .on(RoomEvent.DataReceived, (payload) => {
          this.handleData(payload);
        })
        .on(RoomEvent.Disconnected, () => {
          this.onStatus("Disconnected from Chirpy");
        });

      await room.connect(LIVEKIT_URL, token);
      await room.localParticipant.setMicrophoneEnabled(true);
      this.audioCtx = new AudioContext({ latencyHint: "interactive" });
      await this.audioCtx.resume();
      // Ensure the agent worker is dispatched into this room. The worker may
      // still be starting up, so retry until the agent actually joins the room
      // (the dispatch API can return 200 before the worker is ready). After each
      // dispatch, wait for the agent to join before creating another, so we
      // don't spawn duplicate agents.
      for (let attempt = 0; attempt < 10; attempt++) {
        try {
          await invoke("create_dispatch", { room: ROOM_NAME });
        } catch {
          /* ignore and retry */
        }
        for (let i = 0; i < 5; i++) {
          if (this.agentInRoom(room)) break;
          await new Promise((r) => setTimeout(r, 1000));
        }
        if (this.agentInRoom(room)) break;
      }
      this.onStatus("Listening — speak naturally");
    } catch (e) {
      this.onStatus(`Could not connect: ${(e as Error).message}`);
      this.listening = false;
      this.onListening(false);
    }
  }

  stop() {
    this.listening = false;
    this.onListening(false);
    this.speaking = false;
    this.onSpeaking(false);
    this.room?.disconnect();
    this.room = null;
    this.playbackEls.forEach((el) => el.remove());
    this.playbackEls = [];
    this.audioCtx?.close();
    this.audioCtx = null;
    this.onStatus("Conversation stopped");
  }

  toggleOutputMuted() {
    this.outputMuted = !this.outputMuted;
    this.playbackEls.forEach((el) => (el.muted = this.outputMuted));
    if (this.room) {
      this.room.remoteParticipants.forEach((p) => {
        p.audioTrackPublications.forEach((pub) => pub.setSubscribed(!this.outputMuted));
      });
    }
  }

  private agentInRoom(room: Room): boolean {
    for (const p of room.remoteParticipants.values()) {
      if (p.identity.startsWith("agent-")) return true;
    }
    return false;
  }

  private attachPlayback(track: Track) {
    // Play the agent's audio through an <audio> element (not Web Audio) so the
    // browser's echo canceller includes it as a reference and cancels it from
    // the microphone, preventing the agent from hearing its own speech.
    const el = document.createElement("audio");
    el.srcObject = new MediaStream([track.mediaStreamTrack]);
    el.autoplay = true;
    el.muted = this.outputMuted;
    el.style.display = "none";
    document.body.appendChild(el);
    this.playbackEls.push(el);
  }

  private handleData(payload: Uint8Array) {
    let event: Record<string, unknown>;
    try {
      event = JSON.parse(new TextDecoder().decode(payload));
    } catch {
      return;
    }
    switch (event.type) {
      case "partial":
        this.transcript = (event.text as string) ?? "";
        this.onTranscript(this.transcript, false);
        this.onUserPartial(this.transcript);
        break;
      case "user":
        this.transcript = (event.text as string) ?? "";
        this.reply = "";
        this.onTranscript(this.transcript, true);
        this.onUserMessage(this.transcript);
        break;
      case "assistant_delta":
        this.reply += (event.text as string) ?? "";
        this.onReply(this.reply, false);
        this.onAssistantDelta((event.text as string) ?? "");
        break;
      case "assistant_end":
        this.speaking = false;
        this.onSpeaking(false);
        this.onReply(this.reply, true);
        this.onAssistantEnd();
        break;
      case "latency":
        this.onLatency(event);
        break;
      case "error":
        this.onStatus((event.message as string) ?? "Chirpy error");
        break;
    }
  }
}

const session = new VoiceSession();

// Whether the user explicitly muted the microphone. The session auto-starts
// when the backend becomes ready, so without this the mute button would be
// immediately overridden by pollStatus.
let micMuted = false;

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

const root = document.getElementById("root")!;
const isDebug = new URLSearchParams(window.location.search).get("window") === "debug";

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Inline SVG glyphs (no emoji / image assets).
const ICONS = {
  mic: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>`,
  micOff: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 9v3a3 3 0 0 0 5.12 2.12"/><path d="M15 9.34V4a3 3 0 0 0-5.94-.6"/><path d="M17 16.95A7 7 0 0 1 5 12v-2"/><path d="M19 10v2a7 7 0 0 1-.11 1.23"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="1" y1="1" x2="23" y2="23"/></svg>`,
  speaker: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a9 9 0 0 1 0 14"/></svg>`,
  speakerMuted: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>`,
  debug: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`,
  quit: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
  gear: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`,
  restart: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>`,
  model: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
  download: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`,
  external: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>`,
};

// ---------------------------------------------------------------------------
// Orb window
// ---------------------------------------------------------------------------

function renderOrb() {
  root.innerHTML = `
    <div class="orb-shell" id="orb-shell">
      <div class="orb-wrap">
        <div class="orb" id="orb"></div>
      </div>
      <div class="caption" id="transcript"></div>
      <div class="caption reply" id="reply"></div>
      <div class="controls top-right">
        <button id="debug" title="Open debug mode">${ICONS.debug}</button>
        <button id="quit" title="Quit Chirpy">${ICONS.quit}</button>
      </div>
      <div class="controls under-orb">
        <button id="mic" title="Toggle microphone">${ICONS.mic}</button>
        <button id="speaker" title="Mute speaker">${ICONS.speaker}</button>
      </div>
    </div>
  `;
  document.getElementById("mic")!.onclick = () => {
    if (session.isListening) {
      micMuted = true;
      session.stop();
    } else {
      micMuted = false;
      session.start();
    }
    updateOrbControls();
  };
  document.getElementById("speaker")!.onclick = () => {
    session.toggleOutputMuted();
    updateOrbControls();
  };
  document.getElementById("debug")!.onclick = () => invoke("open_debug");
  document.getElementById("quit")!.onclick = () => getCurrentWindow().close();
  document.getElementById("orb-shell")!.addEventListener("mousedown", (e) => {
    if ((e.target as HTMLElement).closest("button")) return;
    getCurrentWindow().startDragging();
  });
  updateOrbControls();
}

// Reflect the live mic/speaker state in the under-orb buttons.
function updateOrbControls() {
  const mic = document.getElementById("mic");
  const speaker = document.getElementById("speaker");
  if (mic) {
    mic.innerHTML = session.isListening ? ICONS.mic : ICONS.micOff;
    mic.classList.toggle("off", !session.isListening);
  }
  if (speaker) {
    speaker.innerHTML = session.isOutputMuted ? ICONS.speakerMuted : ICONS.speaker;
    speaker.classList.toggle("off", session.isOutputMuted);
  }
}

// ---------------------------------------------------------------------------
// Debug window
// ---------------------------------------------------------------------------

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  time: string;
  state: "streaming" | "completed" | "cancelled";
}

const messages: ChatMessage[] = [];

function renderDebug() {
  root.innerHTML = `
    <div class="debug">
      <header>
        <div class="brand">Chirpy <span class="badge">debug</span></div>
        <span class="status-dot" id="status-dot"></span>
        <span id="status" class="status">Getting ready</span>
        <span class="metrics" id="metrics"></span>
        <button id="settings" title="Configure agent & LLM">${ICONS.gear} Settings</button>
        <button id="models" title="Pick STT & TTS models">${ICONS.model} Models</button>
        <button id="restart" title="Restart the engine">${ICONS.restart} Restart</button>
      </header>
      <div id="llm-warn" class="llm-warn" hidden>
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <span>LLM not configured yet — open <b>Settings</b> to add an endpoint and model, then <b>Save &amp; Restart</b>.</span>
      </div>
      <div class="sys-strip">
        <div class="stages">
          <div class="stage">VAD</div>
          <div class="arrow">→</div>
          <div class="stage">STT</div>
          <div class="arrow">→</div>
          <div class="stage">LLM</div>
          <div class="arrow">→</div>
          <div class="stage">TTS</div>
        </div>
        <div class="sys-info">
          <span>STT <b id="sys-stt">—</b></span>
          <span>TTS <b id="sys-tts">—</b></span>
          <span>VAD <b>sherpa/TEN</b></span>
          <span>Noise <b>DTLN</b></span>
          <span class="sep">·</span>
          <span>LLM <b id="llm-url">—</b></span>
          <span>Model <b id="llm-model">—</b></span>
        </div>
      </div>
      <main>
        <section class="conversation">
          <div class="panel-head">
            <h3>Conversation</h3>
            <button id="clear" title="Clear transcript">Clear</button>
          </div>
          <div id="messages" class="messages"></div>
        </section>
      </main>
      <section class="latency-panel">
        <div class="panel-head">
          <h3>Latency</h3>
          <button id="latency-clear" title="Clear latency graph">Clear</button>
        </div>
        <div id="latency-graph" class="latency-graph"></div>
        <div id="latency-legend" class="latency-legend"></div>
      </section>
      <div class="splitter" id="logs-splitter" title="Drag to resize logs"></div>
      <section class="logs-panel">
        <div class="panel-head">
          <h3>Logs</h3>
        </div>
        <pre id="logs">Waiting for the Chirpy agent…</pre>
      </section>
    </div>
  `;
  document.getElementById("restart")!.onclick = async () => {
    await invoke("restart_backend", { config: engineEnvironment() });
  };
  document.getElementById("settings")!.onclick = () => openSettings();
  document.getElementById("models")!.onclick = () => openModelPicker();
  document.getElementById("clear")!.onclick = () => {
    messages.length = 0;
    const box = document.getElementById("messages");
    if (box) box.innerHTML = "";
  };
  document.getElementById("latency-clear")!.onclick = () => clearLatency();
  updateSystemModels();
  renderLatencyLegend();
  setupLogsSplitter();
  updateLlmWarning();
}

// Make the bottom logs panel height adjustable with a drag handle, and keep the
// conversation / latency areas fluid so everything fits when the window resizes.
function setupLogsSplitter() {
  const splitter = document.getElementById("logs-splitter");
  const logs = document.querySelector<HTMLElement>(".logs-panel");
  if (!splitter || !logs) return;
  let dragging = false;

  splitter.addEventListener("mousedown", (e) => {
    dragging = true;
    e.preventDefault();
    document.body.style.cursor = "ns-resize";
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = document.body.getBoundingClientRect();
    const newHeight = Math.max(80, Math.min(rect.height * 0.7, rect.bottom - e.clientY));
    logs.style.height = `${Math.round(newHeight)}px`;
  });

  window.addEventListener("mouseup", () => {
    dragging = false;
    document.body.style.cursor = "";
  });
}

function updateSystemModels() {
  const stt = document.getElementById("sys-stt");
  if (stt) stt.textContent = settings.sttModel || "—";
  const tts = document.getElementById("sys-tts");
  if (tts) tts.textContent = settings.ttsVoice || "—";
}

// The bundled local LLM works out of the box, so there's nothing to configure.
// Only flag a half-entered remote setup (endpoint without a model, or vice
// versa) so a user who's trying to switch backends isn't left confused.
function updateLlmWarning() {
  const el = document.getElementById("llm-warn");
  if (!el) return;
  const model = settings.llmModel.trim();
  const url = settings.llmURL.trim();
  const incompleteRemote = (model && !url) || (url && !model);
  el.hidden = !incompleteRemote;
}

function appendMessage(m: ChatMessage) {
  const box = document.getElementById("messages");
  if (!box) return;
  const el = document.createElement("div");
  el.className = `log-line ${m.role} ${m.state}`;
  el.innerHTML = `
    <span class="log-time">${m.time}</span>
    <span class="log-role">${m.role === "user" ? "you" : "chirpy"}</span>
    <span class="log-text">${escapeHtml(m.text) || (m.state === "streaming" ? "▍" : "")}</span>
  `;
  box.appendChild(el);
  tailScroll(box);
  return el;
}

// Scroll a container to the bottom only if the user is already near the
// bottom, so reading older content isn't yanked away by new messages/logs.
function tailScroll(el: HTMLElement) {
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  if (nearBottom) el.scrollTop = el.scrollHeight;
}

function updateLastAssistant(delta: string) {
  const box = document.getElementById("messages");
  if (!box) return;
  const last = box.lastElementChild as HTMLElement | null;
  if (last && last.classList.contains("assistant")) {
    const textEl = last.querySelector(".log-text");
    if (textEl) textEl.textContent = (textEl.textContent || "") + delta;
    const m = messages[messages.length - 1];
    if (m && m.role === "assistant") m.text += delta;
    tailScroll(box);
  }
}

function finishLastAssistant() {
  const box = document.getElementById("messages");
  if (!box) return;
  const last = box.lastElementChild as HTMLElement | null;
  if (last && last.classList.contains("assistant")) {
    last.classList.remove("streaming");
    last.classList.add("completed");
    const m = messages[messages.length - 1];
    if (m && m.role === "assistant") m.state = "completed";
  }
}

// Update the in-progress user message as the user speaks (partial), or finalize
// it once the transcript is committed. Creates the user bubble lazily.
function updateLastUser(text: string, final: boolean) {
  const box = document.getElementById("messages");
  if (!box) return;
  let last: HTMLElement | null | undefined = box.lastElementChild as HTMLElement | null;
  let m = messages[messages.length - 1];
  if (!last || !last.classList.contains("user") || m?.state !== "streaming") {
    m = { role: "user", text: "", time: now(), state: "streaming" };
    messages.push(m);
    last = appendMessage(m);
  }
  if (!last) return;
  const textEl = last.querySelector(".log-text");
  if (textEl) textEl.textContent = text || (final ? "" : "▍");
  m.text = text;
  if (final) {
    last.classList.remove("streaming");
    last.classList.add("completed");
    m.state = "completed";
  }
  tailScroll(box);
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string));
}

// ---------------------------------------------------------------------------
// Latency waterfall
// ---------------------------------------------------------------------------

interface LatencyStage {
  stage: string;
  label: string;
  start_ms: number;
  dur_ms: number;
  color: string;
}

interface LatencySample {
  id: number;
  stages: LatencyStage[];
  total_ms: number;
  vad_ms: number;
  stt_ms: number;
  llm_ms: number;
  tts_ms: number;
  tts_gen_ms: number;
  speech_to_transcript_ms: number;
  speech_to_reply_ms: number;
  user_text: string;
  assistant_text: string;
  t: number;
}

const latencySamples: LatencySample[] = [];
const LATENCY_STAGES = ["vad", "stt", "llm", "tts"];

// Human-readable labels for the headline latency numbers.
const LATENCY_LABELS: Record<string, string> = {
  vad_ms: "Speech (VAD)",
  stt_ms: "Speech → transcript",
  llm_ms: "Transcript → reply",
  tts_ms: "Reply → audio",
  total_ms: "Whole turn",
};

function renderLatencyLegend() {
  const el = document.getElementById("latency-legend");
  if (!el) return;
  el.innerHTML = LATENCY_STAGES.map(
    (s) => `<span class="legend-item"><i class="swatch ${s}"></i>${s.toUpperCase()}</span>`,
  ).join("");
}

function addLatencySample(sample: LatencySample) {
  latencySamples.push(sample);
  if (latencySamples.length > 60) latencySamples.shift(); // keep the graph bounded
  renderLatencyGraph();
}

function clearLatency() {
  latencySamples.length = 0;
  renderLatencyGraph();
}

function renderLatencyGraph() {
  const box = document.getElementById("latency-graph");
  if (!box) return;

  // No turns yet: show a friendly placeholder in the graph area, away from the
  // panel title so they don't run together.
  if (!latencySamples.length) {
    box.innerHTML = `<div class="latency-empty">Waiting for a turn…</div>`;
    return;
  }

  // Keep the latest turns visible, oldest at the top like a scrollback log.
  const rows = latencySamples.slice(-20);
  const last = latencySamples[latencySamples.length - 1];
  const summary =
    `latest · speech → transcript ${last.speech_to_transcript_ms}ms · ` +
    `reply → audio ${last.tts_ms}ms · total ${last.total_ms}ms`;

  box.innerHTML =
    `<div class="latency-summary">${summary}</div>` +
    rows
      .map((s) => {
      const time = new Date(s.t * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      const total = Math.max(s.total_ms, 1);
      const segs = s.stages
        .map((st) => {
          const pct = Math.max((st.dur_ms / total) * 100, 0.4);
          return `<div class="seg ${st.color}" style="left:${((st.start_ms / total) * 100).toFixed(2)}%;width:${pct.toFixed(2)}%" title="${st.label}: ${st.dur_ms}ms"></div>`;
        })
        .join("");
      const user = s.user_text ? ` · ${truncate(s.user_text, 40)}` : "";
      const reply = s.assistant_text ? ` · → ${truncate(s.assistant_text, 40)}` : "";
      return `
      <div class="latency-row" title="turn #${s.id}${user}${reply}">
        <span class="latency-time">${time}</span>
        <span class="latency-id">#${s.id}</span>
        <div class="latency-track">${segs}</div>
        <span class="latency-total">${s.total_ms}ms</span>
      </div>`;
    })
    .join("");
  tailScroll(box);
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// ---------------------------------------------------------------------------
// Settings modal
// ---------------------------------------------------------------------------

function openSettings() {
  const modal = document.createElement("div");
  modal.className = "modal";
  modal.innerHTML = `
    <div class="modal-box">
      <h2>Configure Agent &amp; LLM</h2>
      <label>Agent name <input id="s-name" /></label>
      <label>System prompt <textarea id="s-prompt" rows="8"></textarea></label>
      <label>API endpoint <input id="s-url" placeholder="http://localhost:1234/v1" /></label>
      <label>Model <input id="s-model" placeholder="e.g. your-model-id" /></label>
      <label>API key <input id="s-key" type="password" placeholder="optional for local endpoints" /></label>
      <div class="modal-actions">
        <button id="s-reset" title="Restore built-in defaults">Reset to Default</button>
        <button id="s-cancel">Cancel</button>
        <button id="s-save">Save &amp; Restart</button>
      </div>
    </div>
  `;
  (modal.querySelector("#s-name") as HTMLInputElement).value = settings.agentName;
  (modal.querySelector("#s-prompt") as HTMLTextAreaElement).value = settings.systemPrompt;
  (modal.querySelector("#s-url") as HTMLInputElement).value = settings.llmURL;
  (modal.querySelector("#s-model") as HTMLInputElement).value = settings.llmModel;
  (modal.querySelector("#s-key") as HTMLInputElement).value = settings.llmAPIKey;
  (modal.querySelector("#s-cancel") as HTMLButtonElement).onclick = () => modal.remove();
  (modal.querySelector("#s-reset") as HTMLButtonElement).onclick = () => {
    settings.agentName = defaultSettings.agentName;
    settings.systemPrompt = defaultSettings.systemPrompt;
    settings.llmURL = defaultSettings.llmURL;
    settings.llmModel = defaultSettings.llmModel;
    settings.llmAPIKey = defaultSettings.llmAPIKey;
    (modal.querySelector("#s-name") as HTMLInputElement).value = settings.agentName;
    (modal.querySelector("#s-prompt") as HTMLTextAreaElement).value = settings.systemPrompt;
    (modal.querySelector("#s-url") as HTMLInputElement).value = settings.llmURL;
    (modal.querySelector("#s-model") as HTMLInputElement).value = settings.llmModel;
    (modal.querySelector("#s-key") as HTMLInputElement).value = settings.llmAPIKey;
  };
  (modal.querySelector("#s-save") as HTMLButtonElement).onclick = async () => {
    settings.agentName = (modal.querySelector("#s-name") as HTMLInputElement).value;
    settings.systemPrompt = (modal.querySelector("#s-prompt") as HTMLTextAreaElement).value;
    settings.llmURL = (modal.querySelector("#s-url") as HTMLInputElement).value;
    settings.llmModel = (modal.querySelector("#s-model") as HTMLInputElement).value;
    settings.llmAPIKey = (modal.querySelector("#s-key") as HTMLInputElement).value;
    await saveSettings();
    updateLlmWarning();
    await invoke("restart_backend", { config: engineEnvironment() });
    modal.remove();
  };
  document.body.appendChild(modal);
}

// ---------------------------------------------------------------------------
// STT / TTS model picker
// ---------------------------------------------------------------------------

// Faster-whisper sizes (mapped to Systran/faster-whisper-<size> by the engine).
const STT_PRESETS = ["tiny", "base", "small", "medium", "large-v3"];
const STT_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "hi"];
const STT_LANG_NAMES: Record<string, string> = {
  en: "English", es: "Spanish", fr: "French", de: "German", it: "Italian",
  pt: "Portuguese", ru: "Russian", zh: "Chinese", ja: "Japanese", hi: "Hindi",
};
// Common Kokoro voices, validated against hexgrad/Kokoro-82M's `voices/`
// directory so the picker only ever offers voices that exist.
const TTS_VOICES = [
  "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
  "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
  "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
  "am_onyx", "am_puck", "am_santa",
  "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
  "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
  "ef_dora", "em_alex", "em_santa", "ff_siwis",
  "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola",
  "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
  "pf_dora", "pm_alex", "pm_santa",
  "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
  "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
];
const TTS_LANGS = ["a", "b", "e", "f", "h", "i", "j", "p", "z"];
const TTS_LANG_NAMES: Record<string, string> = {
  a: "American English", b: "British English", e: "Spanish", f: "French",
  h: "Hindi", i: "Italian", j: "Japanese", p: "Portuguese", z: "Chinese (Mandarin)",
};

// Kokoro voices are prefixed with their language code, so the first letter of a
// voice id identifies its language (af_*, am_* = English; hf_*, hm_* = Hindi; …).
function voicesForLang(lang: string): string[] {
  return TTS_VOICES.filter((v) => v.startsWith(lang));
}
const HF_STT_URL = "https://huggingface.co/models?search=faster-whisper";
const HF_TTS_URL = "https://huggingface.co/models?search=kokoro";

function openModelPicker() {
  const modal = document.createElement("div");
  modal.className = "modal";
  modal.innerHTML = `
    <div class="modal-box model-picker">
      <div class="mp-head">
        <h2>STT &amp; TTS Models</h2>
        <p class="mp-sub">Download and switch local speech models.</p>
      </div>

      <section class="model-group">
        <h3>Speech-to-text <span class="tag">faster-whisper</span></h3>
        <label>Model <select id="mp-stt"></select></label>
        <label>Language <select id="mp-stt-lang"></select></label>
        <input id="mp-stt-custom" placeholder="Custom repo id, e.g. Systran/faster-whisper-small" />
        <div class="model-actions">
          <button id="mp-stt-browse" class="ghost">${ICONS.external} Browse Hugging Face</button>
          <button id="mp-stt-download" class="ghost">${ICONS.download} Download</button>
          <button id="mp-stt-apply">Apply (live)</button>
        </div>
        <div class="download-row" id="mp-stt-dlrow" hidden>
          <div class="progress"><div class="progress-fill" id="mp-stt-fill"></div></div>
          <span class="progress-pct" id="mp-stt-pct">0%</span>
        </div>
        <p id="mp-stt-status" class="m-status"></p>
      </section>

      <section class="model-group">
        <h3>Text-to-speech <span class="tag">Kokoro</span></h3>
        <label>Voice <select id="mp-tts"></select></label>
        <label>Language <select id="mp-tts-lang"></select></label>
        <input id="mp-tts-custom" placeholder="Custom voice id, e.g. am_michael" />
        <div class="model-actions">
          <button id="mp-tts-browse" class="ghost">${ICONS.external} Browse Hugging Face</button>
          <button id="mp-tts-download" class="ghost">${ICONS.download} Download</button>
          <button id="mp-tts-apply">Apply (live)</button>
        </div>
        <div class="download-row" id="mp-tts-dlrow" hidden>
          <div class="progress"><div class="progress-fill" id="mp-tts-fill"></div></div>
          <span class="progress-pct" id="mp-tts-pct">0%</span>
        </div>
        <p id="mp-tts-status" class="m-status"></p>
      </section>

      <div class="modal-actions">
        <button id="mp-cancel">Done</button>
      </div>
    </div>
  `;

  // --- STT ---
  fillSelect(modal, "#mp-stt", STT_PRESETS, settings.sttModel);
  fillSelect(modal, "#mp-stt-lang", STT_LANGUAGES, settings.sttLanguage, STT_LANG_NAMES);
  const sttCustom = modal.querySelector("#mp-stt-custom") as HTMLInputElement;
  sttCustom.value = STT_PRESETS.includes(settings.sttModel) ? "" : settings.sttModel;
  (modal.querySelector("#mp-stt-browse") as HTMLButtonElement).onclick = () =>
    invoke("open_url", { url: HF_STT_URL }).catch(console.error);
  (modal.querySelector("#mp-stt-download") as HTMLButtonElement).onclick = () =>
    runDownload(modal, "stt", () => sttCustom.value.trim() || sttSelectValue(modal, "#mp-stt"));
  (modal.querySelector("#mp-stt-apply") as HTMLButtonElement).onclick = async () => {
    settings.sttModel = sttCustom.value.trim() || sttSelectValue(modal, "#mp-stt");
    settings.sttLanguage = sttSelectValue(modal, "#mp-stt-lang");
    await saveSettings();
    updateSystemModels();
    const status = modal.querySelector("#mp-stt-status") as HTMLElement;
    status.textContent = "Applying…";
    try {
      await invoke("set_stt", { model: settings.sttModel, language: settings.sttLanguage });
      status.textContent = `STT switched to ${settings.sttModel} — applied live, no restart.`;
    } catch (e) {
      status.textContent = `Apply failed: ${e}`;
    }
  };

  // --- TTS ---
  const ttsLangSel = modal.querySelector("#mp-tts-lang") as HTMLSelectElement;
  const ttsVoiceSel = modal.querySelector("#mp-tts") as HTMLSelectElement;
  const ttsCustom = modal.querySelector("#mp-tts-custom") as HTMLInputElement;

  // Kokoro voices are prefixed with their language code, so the voice list
  // follows the selected language to avoid voice/language mismatches.
  function refreshTTSVoices() {
    const lang = ttsLangSel.value;
    const voices = voicesForLang(lang);
    const current = settings.ttsVoice;
    const pick = voices.includes(current) ? current : (voices[0] || "");
    ttsVoiceSel.innerHTML = voices.map((v) =>
      `<option value="${v}" ${v === pick ? "selected" : ""}>${v}</option>`).join("");
  }

  fillSelect(modal, "#mp-tts-lang", TTS_LANGS, settings.ttsLang, TTS_LANG_NAMES);
  refreshTTSVoices();
  ttsLangSel.onchange = refreshTTSVoices;

  (modal.querySelector("#mp-tts-browse") as HTMLButtonElement).onclick = () =>
    invoke("open_url", { url: HF_TTS_URL }).catch(console.error);
  (modal.querySelector("#mp-tts-download") as HTMLButtonElement).onclick = () =>
    runDownload(modal, "tts", () => ttsCustom.value.trim() || ttsVoiceSel.value);
  (modal.querySelector("#mp-tts-apply") as HTMLButtonElement).onclick = async () => {
    settings.ttsVoice = ttsCustom.value.trim() || ttsVoiceSel.value;
    settings.ttsLang = ttsLangSel.value;
    await saveSettings();
    updateSystemModels();
    const status = modal.querySelector("#mp-tts-status") as HTMLElement;
    status.textContent = "Applying…";
    try {
      await invoke("set_tts", { voice: settings.ttsVoice, lang: settings.ttsLang });
      status.textContent = `Voice switched to ${settings.ttsVoice} (${TTS_LANG_NAMES[settings.ttsLang] || settings.ttsLang}) — applied live, no restart.`;
    } catch (e) {
      status.textContent = `Apply failed: ${e}`;
    }
  };

  (modal.querySelector("#mp-cancel") as HTMLButtonElement).onclick = () => modal.remove();
  document.body.appendChild(modal);
}

interface DownloadProgressEvent {
  kind: string;
  pct: number;
}

async function runDownload(
  modal: HTMLElement,
  kind: string,
  getTarget: () => string,
) {
  const status = modal.querySelector(`#mp-${kind}-status`) as HTMLElement;
  const row = modal.querySelector(`#mp-${kind}-dlrow`) as HTMLElement;
  const fill = modal.querySelector(`#mp-${kind}-fill`) as HTMLElement;
  const pct = modal.querySelector(`#mp-${kind}-pct`) as HTMLElement;
  const btn = modal.querySelector(`#mp-${kind}-download`) as HTMLButtonElement;

  btn.disabled = true;
  fill.classList.remove("error");
  fill.style.width = "0%";
  pct.textContent = "0%";
  row.hidden = false;
  status.textContent = "Downloading…";

  const unlisten = await listen<DownloadProgressEvent>("download-progress", (e) => {
    if (e.payload.kind !== kind) return;
    const p = Math.round(e.payload.pct * 100);
    fill.style.width = `${p}%`;
    pct.textContent = `${p}%`;
  });
  try {
    const msg = await invoke<string>("download_model", { kind, id: getTarget() });
    status.textContent = msg;
    fill.style.width = "100%";
    pct.textContent = "100%";
  } catch (e) {
    status.textContent = `Download failed: ${e}`;
    fill.classList.add("error");
  } finally {
    unlisten();
    btn.disabled = false;
  }
}

function fillSelect(
  modal: HTMLElement,
  sel: string,
  values: string[],
  current: string,
  names?: Record<string, string>,
) {
  const el = modal.querySelector(sel) as HTMLSelectElement;
  el.innerHTML = values
    .map((v) => {
      const label = names ? names[v] || v : v;
      return `<option value="${v}" ${v === current ? "selected" : ""}>${label}</option>`;
    })
    .join("");
}

function sttSelectValue(modal: HTMLElement, sel: string): string {
  return (modal.querySelector(sel) as HTMLSelectElement).value;
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

async function pollStatus() {
  const status = await invoke<{
    livekit_running: boolean;
    agent_running: boolean;
    ready: boolean;
    error: string | null;
  }>("backend_status");
  const el = document.getElementById("status");
  const dot = document.getElementById("status-dot");
  if (el) {
    if (!status.livekit_running) {
      el.textContent = "LiveKit server not running";
      dot?.classList.add("error");
    } else if (!status.agent_running) {
      el.textContent = "Agent worker starting…";
      dot?.classList.add("warn");
    } else if (status.ready) {
      el.textContent = "Chirpy ready";
      dot?.classList.add("ok");
    } else {
      el.textContent = "Loading models…";
      dot?.classList.add("warn");
    }
  }
  // Only the main window drives the voice session; the debug window just
  // observes status and the conversation via events. Don't auto-start if the
  // user explicitly muted the mic.
  if (!isDebug) {
    if (status.ready && !session.isListening && !micMuted) session.start();
    if (!status.ready && session.isListening) session.stop();
  }
}

async function pollMetrics() {
  const m = await invoke<{ cpu: string; gpu: string; ram: string }>("system_metrics");
  const el = document.getElementById("metrics");
  if (el) el.textContent = `CPU ${m.cpu} · GPU ${m.gpu} · RAM ${m.ram}`;
}

async function pollLogs() {
  const logs = await invoke<string>("tail_logs");
  const el = document.getElementById("logs");
  if (el) {
    el.textContent = logs;
    tailScroll(el);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  await loadSettings();
  await invoke("start_backend", { config: engineEnvironment() }).catch((e) => {
    console.error("start_backend failed", e);
  });

  if (isDebug) {
    renderDebug();
    setInterval(pollStatus, 1000);
    setInterval(pollMetrics, 1000);
    setInterval(pollLogs, 1000);
    // The debug window observes the conversation via events broadcast by the
    // main window (which owns the voice session).
    listen("conversation-user", (e) => {
      updateLastUser(e.payload as string, true);
    });
    listen("conversation-user-partial", (e) => {
      updateLastUser(e.payload as string, false);
    });
    listen("conversation-assistant-delta", (e) => {
      // Create the assistant bubble lazily on the first delta.
      const last = messages[messages.length - 1];
      if (!last || last.role !== "assistant" || last.state !== "streaming") {
        messages.push({ role: "assistant", text: "", time: now(), state: "streaming" });
        appendMessage(messages[messages.length - 1]);
      }
      updateLastAssistant(e.payload as string);
    });
    listen("conversation-assistant-end", () => finishLastAssistant());
    listen("conversation-latency", (e) => {
      addLatencySample(e.payload as LatencySample);
    });
    session.onStatus = (s) => {
      const el = document.getElementById("status");
      if (el) el.textContent = s;
    };
    const model = document.getElementById("llm-model");
    if (model) model.textContent = settings.llmModel.trim() || "local · qwen2.5-0.5b";
    const url = document.getElementById("llm-url");
    if (url) url.textContent = settings.llmModel.trim() ? settings.llmURL : "in-process";
  } else {
    renderOrb();
    session.onStatus = (s) => {
      const el = document.getElementById("status");
      if (el) el.textContent = s;
    };
    session.onTranscript = (t, final) => showCaption("transcript", t, final);
    session.onReply = (t, final) => showCaption("reply", t, final);
    session.onSpeaking = (b) => {
      const orb = document.getElementById("orb");
      if (orb) orb.classList.toggle("speaking", b);
    };
    session.onListening = (b) => {
      const orb = document.getElementById("orb");
      if (orb) orb.classList.toggle("listening", b);
      updateOrbControls();
    };
    // Broadcast the conversation to the debug window.
    session.onUserMessage = (text) => emit("conversation-user", text);
    session.onUserPartial = (text) => emit("conversation-user-partial", text);
    session.onAssistantDelta = (delta) => emit("conversation-assistant-delta", delta);
    session.onAssistantEnd = () => emit("conversation-assistant-end", null);
    session.onLatency = (payload) => emit("conversation-latency", payload);
    setInterval(pollStatus, 1000);
  }
}

let captionTimers: Record<string, number> = {};
function showCaption(id: string, text: string, persist = false) {
  const el = document.getElementById(id);
  if (!el) return;
  const trimmed = text.trim();
  if (!trimmed) return;
  el.textContent = trimmed;
  el.classList.add("visible");
  if (captionTimers[id]) window.clearTimeout(captionTimers[id]);
  // Final transcripts/replies stay on the orb until the next turn replaces
  // them; live partials still fade out after a moment so they don't linger.
  if (persist) {
    captionTimers[id] = 0;
    return;
  }
  captionTimers[id] = window.setTimeout(() => {
    el.classList.remove("visible");
  }, 6000);
}

init();
