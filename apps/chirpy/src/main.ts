import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { load } from "@tauri-apps/plugin-store";
import { Room, RoomEvent, Track } from "livekit-client";
import "./styles.css";

const LIVEKIT_URL = "ws://127.0.0.1:7880";
const ROOM_NAME = "chirpy";

interface Settings {
  agentName: string;
  systemPrompt: string;
  llmURL: string;
  llmModel: string;
  llmAPIKey: string;
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
  return {
    AGENT_NAME: name,
    ASSISTANT_SYSTEM: resolved,
    LLM_BASE_URL: settings.llmURL.trim(),
    LLM_MODEL_NAME: settings.llmModel.trim(),
    LLM_API_KEY: settings.llmAPIKey,
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
  onTranscript: (t: string) => void = () => {};
  onReply: (t: string) => void = () => {};
  onSpeaking: (b: boolean) => void = () => {};
  onListening: (b: boolean) => void = () => {};
  onMessage: (m: ChatMessage) => void = () => {};

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
      // Ensure the agent worker is dispatched into this room.
      await invoke("create_dispatch", { room: ROOM_NAME }).catch(() => {});
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

  interrupt() {
    if (!this.speaking) return;
    this.speaking = false;
    this.onSpeaking(false);
    this.onStatus("Listening…");
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
      case "transcript":
        this.transcript = (event.text as string) ?? "";
        this.reply = "";
        this.onTranscript(this.transcript);
        this.onMessage({ role: "user", text: this.transcript, state: "completed" });
        break;
      case "partial":
        this.transcript = (event.text as string) ?? "";
        this.onTranscript(this.transcript);
        break;
      case "turn_started":
        this.speaking = true;
        this.onSpeaking(true);
        this.onMessage({ role: "assistant", text: "", state: "streaming" });
        break;
      case "text":
        this.reply += (event.delta as string) ?? "";
        this.onReply(this.reply);
        break;
      case "done":
        this.speaking = false;
        this.onSpeaking(false);
        break;
      case "interrupted":
        this.speaking = false;
        this.onSpeaking(false);
        break;
      case "error":
        this.onStatus((event.message as string) ?? "Chirpy error");
        break;
    }
  }
}

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  state: "streaming" | "completed" | "cancelled" | "failed";
}

const session = new VoiceSession();

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

const root = document.getElementById("root")!;
const isDebug = new URLSearchParams(window.location.search).get("window") === "debug";

function renderOrb() {
  root.innerHTML = `
    <div class="orb-shell" id="orb-shell">
      <div class="orb" id="orb"></div>
      <div class="caption" id="transcript"></div>
      <div class="caption reply" id="reply"></div>
      <div class="controls">
        <button id="mic" title="Mute microphone">🎤</button>
        <button id="quit" title="Quit Chirpy">✕</button>
        <button id="speaker" title="Mute speaker">🔊</button>
      </div>
    </div>
  `;
  document.getElementById("mic")!.onclick = () => {
    session.isListening ? session.stop() : session.start();
  };
  document.getElementById("quit")!.onclick = () => getCurrentWindow().close();
  document.getElementById("speaker")!.onclick = () => session.toggleOutputMuted();
  document.getElementById("orb-shell")!.addEventListener("mousedown", (e) => {
    if ((e.target as HTMLElement).closest("button")) return;
    getCurrentWindow().startDragging();
  });
}

function renderDebug() {
  root.innerHTML = `
    <div class="debug">
      <header>
        <span id="status">Getting ready</span>
        <span class="metrics" id="metrics"></span>
        <button id="restart">Restart</button>
        <button id="settings">Settings</button>
      </header>
      <main>
        <section class="pipeline">
          <h3>Voice pipeline</h3>
          <div class="stages">
            <div class="stage">VAD · Built-in</div>
            <div class="arrow">→</div>
            <div class="stage">STT · Built-in</div>
            <div class="arrow">→</div>
            <div class="stage">LLM · <span id="llm-model">—</span></div>
            <div class="arrow">→</div>
            <div class="stage">TTS · Built-in</div>
          </div>
        </section>
        <section class="conversation">
          <h3>Conversation</h3>
          <div id="messages"></div>
        </section>
        <section class="events">
          <h3>Engine events</h3>
          <pre id="logs">Waiting for the Chirpy agent…</pre>
        </section>
      </main>
    </div>
  `;
  document.getElementById("restart")!.onclick = async () => {
    await invoke("restart_backend", { config: engineEnvironment() });
  };
  document.getElementById("settings")!.onclick = () => openSettings();
}

function openSettings() {
  const modal = document.createElement("div");
  modal.className = "modal";
  modal.innerHTML = `
    <div class="modal-box">
      <h2>Configure Agent & LLM</h2>
      <label>Agent name <input id="s-name" /></label>
      <label>System prompt <textarea id="s-prompt" rows="8"></textarea></label>
      <label>API endpoint <input id="s-url" /></label>
      <label>Model <input id="s-model" /></label>
      <label>API key <input id="s-key" type="password" /></label>
      <div class="modal-actions">
        <button id="s-cancel">Cancel</button>
        <button id="s-save">Save & Restart</button>
      </div>
    </div>
  `;
  (modal.querySelector("#s-name") as HTMLInputElement).value = settings.agentName;
  (modal.querySelector("#s-prompt") as HTMLTextAreaElement).value = settings.systemPrompt;
  (modal.querySelector("#s-url") as HTMLInputElement).value = settings.llmURL;
  (modal.querySelector("#s-model") as HTMLInputElement).value = settings.llmModel;
  (modal.querySelector("#s-key") as HTMLInputElement).value = settings.llmAPIKey;
  (modal.querySelector("#s-cancel") as HTMLButtonElement).onclick = () => modal.remove();
  (modal.querySelector("#s-save") as HTMLButtonElement).onclick = async () => {
    settings.agentName = (modal.querySelector("#s-name") as HTMLInputElement).value;
    settings.systemPrompt = (modal.querySelector("#s-prompt") as HTMLTextAreaElement).value;
    settings.llmURL = (modal.querySelector("#s-url") as HTMLInputElement).value;
    settings.llmModel = (modal.querySelector("#s-model") as HTMLInputElement).value;
    settings.llmAPIKey = (modal.querySelector("#s-key") as HTMLInputElement).value;
    await saveSettings();
    await invoke("restart_backend", { config: engineEnvironment() });
    modal.remove();
  };
  document.body.appendChild(modal);
}

function appendMessage(m: ChatMessage) {
  const box = document.getElementById("messages");
  if (!box) return;
  const el = document.createElement("div");
  el.className = `msg ${m.role}`;
  el.textContent = m.text || (m.state === "streaming" ? "Preparing a reply…" : "");
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}

async function pollStatus() {
  const status = await invoke<{
    livekit_running: boolean;
    agent_running: boolean;
    ready: boolean;
    error: string | null;
  }>("backend_status");
  const el = document.getElementById("status");
  if (el) {
    if (!status.livekit_running) el.textContent = "LiveKit server not running";
    else if (!status.agent_running) el.textContent = "Agent worker starting…";
    else if (status.ready) el.textContent = "Chirpy ready";
    else el.textContent = "Loading models…";
  }
  if (status.ready && !session.isListening) session.start();
  if (!status.ready && session.isListening) session.stop();
}

async function pollMetrics() {
  const m = await invoke<{ cpu: string; gpu: string; ram: string }>("system_metrics");
  const el = document.getElementById("metrics");
  if (el) el.textContent = `CPU ${m.cpu} · GPU ${m.gpu} · RAM ${m.ram}`;
}

async function pollLogs() {
  const logs = await invoke<string>("tail_logs");
  const el = document.getElementById("logs");
  if (el) el.textContent = logs;
}

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
    session.onMessage = appendMessage;
    session.onStatus = (s) => {
      const el = document.getElementById("status");
      if (el) el.textContent = s;
    };
    const model = document.getElementById("llm-model");
    if (model) model.textContent = settings.llmModel || "—";
  } else {
    renderOrb();
    session.onStatus = (s) => {
      const el = document.getElementById("status");
      if (el) el.textContent = s;
    };
    session.onTranscript = (t) => showCaption("transcript", t);
    session.onReply = (t) => showCaption("reply", t);
    session.onSpeaking = (b) => {
      const orb = document.getElementById("orb");
      if (orb) orb.classList.toggle("speaking", b);
    };
    session.onListening = (b) => {
      const orb = document.getElementById("orb");
      if (orb) orb.classList.toggle("listening", b);
    };
    setInterval(pollStatus, 1000);
  }
}

let captionTimers: Record<string, number> = {};
function showCaption(id: string, text: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const trimmed = text.trim();
  if (!trimmed) return;
  el.textContent = trimmed;
  el.classList.add("visible");
  if (captionTimers[id]) window.clearTimeout(captionTimers[id]);
  captionTimers[id] = window.setTimeout(() => {
    el.classList.remove("visible");
  }, 6000);
}

init();
