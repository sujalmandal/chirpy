//! Native audio output for Chirpy.
//!
//! The agent's TTS audio is streamed over a local TCP socket (raw int16 mono PCM
//! at 24 kHz) and played through the system audio device via `cpal` (CoreAudio).
//! This bypasses the WebKit webview's WebRTC audio playback, which stutters on
//! live streams.

use std::collections::VecDeque;
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub const SAMPLE_RATE: u32 = 24000;
pub const DEFAULT_PORT: u16 = 8765;

#[derive(Clone, Default)]
struct Sink {
    buf: Arc<Mutex<VecDeque<f32>>>,
}

impl Sink {
    /// Push raw int16 mono PCM bytes into the play buffer.
    fn push(&self, data: &[u8]) {
        let mut buf = self.buf.lock().unwrap();
        for chunk in data.chunks_exact(2) {
            let v = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
            buf.push_back(v);
        }
    }

    /// Fill the output callback buffer from the play queue (silence if empty).
    fn fill(&self, out: &mut [f32]) {
        let mut buf = self.buf.lock().unwrap();
        for o in out.iter_mut() {
            *o = buf.pop_front().unwrap_or(0.0);
        }
    }
}

/// Starts the TCP PCM listener and the native output device.
pub struct NativeAudioServer;

impl NativeAudioServer {
    pub fn start(port: u16) -> Result<(), String> {
        let sink = Sink::default();

        // Start the native output device (kept alive forever).
        let out_sink = sink.clone();
        thread::Builder::new()
            .name("chirpy-native-out".into())
            .spawn(move || start_output(&out_sink))
            .map_err(|e| e.to_string())?;

        // Start the TCP listener that receives the agent's PCM.
        let in_sink = sink.clone();
        thread::Builder::new()
            .name("chirpy-native-in".into())
            .spawn(move || {
                let listener = match TcpListener::bind(("127.0.0.1", port)) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("native audio tcp bind {port}: {e}");
                        return;
                    }
                };
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { continue };
                    let sink = in_sink.clone();
                    thread::spawn(move || read_stream(stream, &sink));
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(())
    }
}

fn read_stream(mut stream: TcpStream, sink: &Sink) {
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => sink.push(&buf[..n]),
            Err(_) => break,
        }
    }
}

fn start_output(sink: &Sink) {
    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => {
            eprintln!("native audio: no output device");
            return;
        }
    };
    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };
    let out_sink = sink.clone();
    let stream = match device.build_output_stream(
        &config,
        move |data: &mut [f32], _| out_sink.fill(data),
        |err| eprintln!("native audio stream error: {err}"),
        None,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("native audio: build stream failed: {e}");
            return;
        }
    };
    if let Err(e) = stream.play() {
        eprintln!("native audio: play failed: {e}");
        return;
    }
    // Keep the stream alive for the life of the app.
    Box::leak(Box::new(stream));
}
