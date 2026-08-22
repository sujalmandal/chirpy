"""Pre-download a local speech model (STT or TTS) into the Hugging Face cache.

Used by the debug UI's model picker so that switching to a model is fast the
first time (the model is already cached when the agent worker loads it).

Usage:
    python download.py stt <size-or-repo-id> [--language en]
    python download.py tts <voice> [--repo hexgrad/Kokoro-82M]
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

# faster-whisper accepts a bare size (mapped to Systran/faster-whisper-<size>)
# or a full Hugging Face repo id.
_FASTER_WHISPER_SIZES = {
    "tiny", "base", "small", "medium",
    "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo",
}

_KOKORO_DEFAULT_REPO = "hexgrad/Kokoro-82M"
_KOKORO_LANGUAGES = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}

# All voices in hexgrad/Kokoro-82M (voice prefix letter = language code).
_KOKORO_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
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
]


class _ReportingTqdm(tqdm):
    """A tqdm subclass that prints an aggregate ``progress <frac>`` line (0..1)
    to stdout so the debug UI can drive a progress bar across all files."""

    _chirpy_instances: list["_ReportingTqdm"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chirpy_instances.append(self)

    def update(self, n: float = 1) -> None:
        super().update(n)
        done = sum(t.n for t in self._chirpy_instances)
        total = sum(t.total or 0 for t in self._chirpy_instances)
        if total:
            print(f"progress {min(1.0, done / total):.4f}", flush=True)


def _with_progress(func, *args, **kwargs):
    """Run a huggingface_hub download, emitting aggregate progress lines."""
    return func(*args, **kwargs, tqdm_class=_ReportingTqdm)


def _resolve_stt_repo(model: str) -> str:
    """Map a faster-whisper size to its HF repo, or pass through a repo id."""
    size = model.strip()
    if size in _FASTER_WHISPER_SIZES:
        return f"Systran/faster-whisper-{size}"
    if "/" in size:
        return size
    raise ValueError(f"Unknown faster-whisper model {model!r}")


def download_stt(model: str, language: str = "en") -> str:
    repo = _resolve_stt_repo(model)
    _with_progress(
        snapshot_download,
        repo_id=repo,
        ignore_patterns=["*.ckpt"],
    )
    return repo


def download_tts(voice: str, repo: str = _KOKORO_DEFAULT_REPO) -> str:
    # Fetch the model weights + config and the specific voice checkpoint so the
    # agent worker doesn't download on first TTS call.
    _with_progress(
        snapshot_download,
        repo_id=repo,
        allow_patterns=["*.pt", "*.json", "*.md"],
    )
    download_voice_only(voice, repo)
    return repo


def download_voice_only(voice: str, repo: str = _KOKORO_DEFAULT_REPO) -> None:
    _with_progress(hf_hub_download, repo_id=repo, filename=f"voices/{voice}.pt")


def download_all_tts_voices(repo: str = _KOKORO_DEFAULT_REPO) -> None:
    """Pre-cache the model snapshot and every Kokoro voice so any language works
    offline on first run."""
    _with_progress(
        snapshot_download,
        repo_id=repo,
        allow_patterns=["*.pt", "*.json", "*.md"],
    )
    for voice in _KOKORO_VOICES:
        download_voice_only(voice, repo)
        print(f"ok: cached tts voice {voice}", flush=True)


def download_vad(model_type: str = "ten") -> str:
    """Download the sherpa-onnx VAD model into the engine's .models dir."""
    from pathlib import Path

    dest_dir = Path(__file__).resolve().parent / ".models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = "ten-vad.onnx" if model_type == "ten" else "silero_vad.onnx"
    dest = dest_dir / filename
    if dest.exists():
        return str(dest)

    import urllib.request

    url = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx"
        if model_type == "ten"
        else "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    )
    urllib.request.urlretrieve(url, str(dest))
    return str(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download a local speech model")
    sub = parser.add_subparsers(dest="kind", required=True)

    stt = sub.add_parser("stt", help="Download a faster-whisper STT model")
    stt.add_argument("model", help="size (e.g. base, small) or HF repo id")
    stt.add_argument("--language", default="en")

    tts = sub.add_parser("tts", help="Download a Kokoro TTS model + voice")
    tts.add_argument("voice", help="voice id, e.g. af_heart, or 'all' for every voice")
    tts.add_argument("--repo", default=_KOKORO_DEFAULT_REPO)

    vad = sub.add_parser("vad", help="Download the sherpa-onnx TEN-VAD model")
    vad.add_argument("--model-type", default="ten", help="ten (TEN-VAD) or silero")

    args = parser.parse_args()

    try:
        if args.kind == "stt":
            repo = download_stt(args.model, args.language)
            print(f"ok: cached stt from {repo}")
        elif args.kind == "tts":
            if args.voice == "all":
                download_all_tts_voices(args.repo)
            else:
                repo = download_tts(args.voice, args.repo)
                print(f"ok: cached tts voice {args.voice} from {repo}")
        else:  # vad
            path = download_vad(args.model_type)
            print(f"ok: cached vad model at {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
