#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use livekit_api::access_token::{AccessToken, VideoGrants};
use livekit_api::services::LiveKitApi;
use livekit_protocol as proto;
use serde::Serialize;
use tauri::{Emitter, Manager};

const LIVEKIT_URL: &str = "ws://127.0.0.1:7880";
const LIVEKIT_API_KEY: &str = "devkey";
const LIVEKIT_API_SECRET: &str = "secret";
const ROOM_NAME: &str = "chirpy";

struct BackendState {
    livekit: Option<Child>,
    agent: Option<Child>,
    livekit_log: PathBuf,
    agent_log: PathBuf,
}

impl BackendState {
    fn new() -> Self {
        Self {
            livekit: None,
            agent: None,
            livekit_log: PathBuf::new(),
            agent_log: PathBuf::new(),
        }
    }
}

#[derive(Serialize, Clone)]
struct BackendStatus {
    livekit_running: bool,
    agent_running: bool,
    ready: bool,
    error: Option<String>,
}

fn project_root() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    if exe.to_string_lossy().contains(".app/Contents/MacOS/") {
        let mut root = exe.clone();
        for _ in 0..4 {
            root.pop();
        }
        root
    } else {
        let mut root = std::env::current_dir().unwrap_or_default();
        for _ in 0..3 {
            root.pop();
        }
        root
    }
}

fn engine_paths() -> (PathBuf, PathBuf) {
    let root = project_root();
    let venv = root.join("engine/chirpy/.venv/bin/python");
    let script = root.join("engine/chirpy/agent.py");
    (venv, script)
}

fn is_running(child: &mut Option<Child>) -> bool {
    match child {
        Some(c) => c.try_wait().ok().flatten().is_none(),
        None => false,
    }
}

fn open_log(dir: &PathBuf, name: &str) -> Result<(std::fs::File, PathBuf), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = dir.join(name);
    let file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    Ok((file, path))
}

#[tauri::command]
fn start_backend(state: tauri::State<Mutex<BackendState>>, config: serde_json::Value) -> Result<BackendStatus, String> {
    let mut state = state.lock().map_err(|e| e.to_string())?;
    let root = project_root();
    let log_dir = root.join("logs");

    // 1. Start livekit-server if not already running.
    if !is_running(&mut state.livekit) {
        let (file, path) = open_log(&log_dir, "livekit-server.log")?;
        let mut cmd = Command::new("livekit-server");
        cmd.arg("--dev")
            .env("LIVEKIT_LOG_LEVEL", "info")
            .current_dir(&root)
            .stdout(Stdio::from(file.try_clone().map_err(|e| e.to_string())?))
            .stderr(Stdio::from(file));
        match cmd.spawn() {
            Ok(child) => {
                state.livekit = Some(child);
                state.livekit_log = path;
            }
            Err(e) => {
                return Err(format!(
                    "Could not start livekit-server. Install it with `brew install livekit`: {e}"
                ));
            }
        }
    }

    // 2. Start the agent worker if not already running.
    if !is_running(&mut state.agent) {
        let (venv, script) = engine_paths();
        if !venv.exists() || !script.exists() {
            return Err("Chirpy engine is not set up. Run scripts/setup.sh first.".into());
        }
        let (file, path) = open_log(&log_dir, "chirpy-agent.log")?;
        let mut cmd = Command::new(&venv);
        cmd.arg(&script)
            .arg("start")
            .current_dir(root.join("engine/chirpy"))
            .stdout(Stdio::from(file.try_clone().map_err(|e| e.to_string())?))
            .stderr(Stdio::from(file))
            .env("LIVEKIT_URL", LIVEKIT_URL)
            .env("LIVEKIT_API_KEY", LIVEKIT_API_KEY)
            .env("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET);
        if let Some(obj) = config.as_object() {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    cmd.env(k, s);
                }
            }
        }
        let child = cmd.spawn().map_err(|e| format!("Could not start Chirpy agent: {e}"))?;
        state.agent = Some(child);
        state.agent_log = path;
    }

    Ok(status_of(&mut state))
}

#[tauri::command]
fn stop_backend(state: tauri::State<Mutex<BackendState>>) -> Result<(), String> {
    let mut state = state.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = state.agent.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    if let Some(mut child) = state.livekit.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    Ok(())
}

#[tauri::command]
fn restart_backend(state: tauri::State<Mutex<BackendState>>, config: serde_json::Value) -> Result<BackendStatus, String> {
    {
        let mut s = state.lock().map_err(|e| e.to_string())?;
        if let Some(mut child) = s.agent.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    start_backend(state, config)
}

#[tauri::command]
fn backend_status(state: tauri::State<Mutex<BackendState>>) -> Result<BackendStatus, String> {
    let mut state = state.lock().map_err(|e| e.to_string())?;
    Ok(status_of(&mut state))
}

fn status_of(state: &mut BackendState) -> BackendStatus {
    let livekit_running = is_running(&mut state.livekit);
    let agent_running = is_running(&mut state.agent);
    BackendStatus {
        livekit_running,
        agent_running,
        ready: livekit_running && agent_running,
        error: None,
    }
}

#[tauri::command]
fn get_token(room: String) -> Result<String, String> {
    let room = if room.trim().is_empty() { ROOM_NAME.to_string() } else { room };
    let token = AccessToken::with_api_key(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("chirpy-client")
        .with_grants(VideoGrants {
            room_join: true,
            room: room.clone(),
            can_publish: true,
            can_subscribe: true,
            can_publish_data: true,
            ..Default::default()
        })
        .to_jwt()
        .map_err(|e| e.to_string())?;
    Ok(token)
}

#[tauri::command]
async fn create_dispatch(room: String) -> Result<(), String> {
    let room = if room.trim().is_empty() { ROOM_NAME.to_string() } else { room };
    let api = LiveKitApi::with_api_key("http://127.0.0.1:7880", LIVEKIT_API_KEY, LIVEKIT_API_SECRET);
    let req = proto::CreateAgentDispatchRequest {
        room: room.clone(),
        agent_name: "chirpy-agent".to_string(),
        metadata: "{}".to_string(),
        ..Default::default()
    };
    api.agent_dispatch()
        .create_dispatch(req)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn tail_logs(state: tauri::State<Mutex<BackendState>>) -> Result<String, String> {
    let state = state.lock().map_err(|e| e.to_string())?;
    let path = if state.agent_log.exists() {
        state.agent_log.clone()
    } else {
        project_root().join("logs/chirpy-agent.log")
    };
    if !path.exists() {
        return Ok("Waiting for the Chirpy agent…".into());
    }
    let data = std::fs::read(&path).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&data);
    let tail: String = text.chars().rev().take(20_000).collect::<String>().chars().rev().collect();
    Ok(tail)
}

#[derive(Serialize)]
struct Metrics {
    cpu: String,
    gpu: String,
    ram: String,
}

#[tauri::command]
fn system_metrics() -> Metrics {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let cpu = sys.global_cpu_info().cpu_usage();
    let total = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let used = sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let gpu = gpu_usage();
    Metrics {
        cpu: format!("{cpu:.0}%"),
        gpu,
        ram: format!("{used:.1} / {total:.0} GB"),
    }
}

fn gpu_usage() -> String {
    #[cfg(target_os = "macos")]
    {
        let out = Command::new("/bin/zsh")
            .args(["-lc", "ioreg -l -w0 -r -c IOAccelerator 2>/dev/null | sed -n 's/.*\"Device Utilization %\"=\\([0-9]*\\).*/\\1/p' | head -1"])
            .output();
        if let Ok(out) = out {
            let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !v.is_empty() {
                return format!("{v}%");
            }
        }
        "—".into()
    }
    #[cfg(not(target_os = "macos"))]
    {
        "—".into()
    }
}

#[tauri::command]
fn keychain_get(account: String) -> Result<String, String> {
    let entry = keyring::Entry::new("dev.chirpy.settings", &account).map_err(|e| e.to_string())?;
    entry.get_password().map_err(|_| "not found".into())
}

#[tauri::command]
fn keychain_set(account: String, value: String) -> Result<(), String> {
    let entry = keyring::Entry::new("dev.chirpy.settings", &account).map_err(|e| e.to_string())?;
    entry.set_password(&value).map_err(|e| e.to_string())
}

#[tauri::command]
fn keychain_delete(account: String) -> Result<(), String> {
    let entry = keyring::Entry::new("dev.chirpy.settings", &account).map_err(|e| e.to_string())?;
    entry.delete_credential().map_err(|_| "not found".into())
}

#[tauri::command]
fn open_debug(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(win) = app.get_webview_window("debug") {
        win.show().map_err(|e| e.to_string())?;
        win.set_focus().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn open_logs() -> Result<(), String> {
    let dir = project_root().join("logs");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Command::new("open")
        .arg(&dir)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Hot-apply an STT model/language change by writing config/stt.json; the agent
/// worker watches the file and swaps the model live (no restart).
#[tauri::command]
fn set_stt(model: String, language: String) -> Result<(), String> {
    let path = project_root().join("config/stt.json");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let value = serde_json::json!({ "model": model, "language": language });
    let text = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("could not write stt config: {e}"))
}

/// Hot-apply a TTS voice change by writing config/tts.json; the agent worker
/// watches the file and swaps the voice live (no restart).
#[tauri::command]
fn set_tts(voice: String, lang: String) -> Result<(), String> {
    let path = project_root().join("config/tts.json");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let value = serde_json::json!({ "voice": voice, "lang": lang });
    let text = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("could not write tts config: {e}"))
}

/// Open a URL in the default browser (used by the model picker to browse
/// Hugging Face with the right tag).
#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&url)
            .spawn()
            .map_err(|e| format!("could not open {url}: {e}"))?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map_err(|e| format!("could not open {url}: {e}"))?;
    }
    Ok(())
}

/// Pre-download a local speech model (STT or TTS) via the engine's Python
/// helper, so switching models in the picker is fast on first use. Streams
/// ``progress <frac>`` lines from the helper as ``download-progress`` events.
#[derive(Serialize, Clone)]
struct DownloadProgress {
    kind: String,
    pct: f32,
}

#[tauri::command]
async fn download_model(app: tauri::AppHandle, kind: String, id: String) -> Result<String, String> {
    let (venv, _script) = engine_paths();
    let script = project_root().join("engine/chirpy/download.py");
    if !venv.exists() {
        return Err("Chirpy engine is not set up. Run scripts/setup.sh first.".into());
    }
    if !script.exists() {
        return Err("model download helper not found (engine/chirpy/download.py)".into());
    }
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        use std::io::{BufRead, BufReader, Read};

        let mut child = Command::new(&venv)
            .arg(&script)
            .arg(&kind)
            .arg(&id)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not run download: {e}"))?;

        if let Some(stdout) = child.stdout.take() {
            let app = app.clone();
            let kind = kind.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { continue };
                    if let Some(frac) = line.trim().strip_prefix("progress ") {
                        if let Ok(pct) = frac.parse::<f32>() {
                            let _ = app.emit(
                                "download-progress",
                                DownloadProgress { kind: kind.clone(), pct },
                            );
                        }
                    }
                }
            });
        }

        let status = child.wait().map_err(|e| e.to_string())?;
        let err_msg = match child.stderr.take() {
            Some(mut stderr) => {
                let mut s = String::new();
                let _ = stderr.read_to_string(&mut s);
                s.trim().to_string()
            }
            None => String::new(),
        };
        if status.success() {
            Ok(format!("cached {kind} from {id}"))
        } else {
            Err(if err_msg.is_empty() { "download failed".to_string() } else { err_msg })
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::new().build())
        .manage(Mutex::new(BackendState::new()))
        .invoke_handler(tauri::generate_handler![
            start_backend,
            stop_backend,
            restart_backend,
            backend_status,
            get_token,
            create_dispatch,
            tail_logs,
            system_metrics,
            keychain_get,
            keychain_set,
            keychain_delete,
            open_debug,
            open_logs,
            open_url,
            download_model,
            set_tts,
            set_stt
        ])
        .setup(|app| {
            let _ = app;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Chirpy");
}
