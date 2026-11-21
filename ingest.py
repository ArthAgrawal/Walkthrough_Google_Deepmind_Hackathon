"""Step 1: ffmpeg -> frames at config.FPS (JPEG, 768 px wide) + mono mp3 audio track."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import config


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, raising RuntimeError with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr[-1500:]}")
    return proc


def probe(video_path: Path) -> dict:
    """Return {'duration': seconds, 'has_audio': bool} via ffprobe."""
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                "-of", "json", str(video_path)]).stdout
    info = json.loads(out)
    streams = [s.get("codec_type") for s in info.get("streams", [])]
    duration = float(info.get("format", {}).get("duration") or 0.0)
    return {"duration": duration, "has_audio": "audio" in streams}


def work_dir_for(video_path: Path) -> Path:
    """Per-clip working directory under data/work/<video stem>/."""
    return config.WORK_DIR / video_path.stem


def ingest(video_path: str | Path, work_dir: str | Path | None = None) -> dict:
    """Extract frames + audio from a video.

    Frame files are frames/0000.jpg, 0001.jpg, ... where index/FPS = timestamp in seconds.
    Returns {"work_dir", "duration", "frames": [{"t", "frame_path"}], "audio_path" | None}.
    frame_path is relative to work_dir (POSIX style), matching the FrameCheck contract.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    work = Path(work_dir) if work_dir else work_dir_for(video_path)
    frames_dir = work / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    meta = probe(video_path)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
          "-vf", f"fps={config.FPS},scale={config.FRAME_WIDTH}:-2",
          "-q:v", "3", "-start_number", "0", str(frames_dir / "%04d.jpg")])
    frames = [{"t": round(int(p.stem) / config.FPS, 2), "frame_path": f"frames/{p.name}"}
              for p in sorted(frames_dir.glob("*.jpg"))]

    audio_path = None
    if meta["has_audio"]:
        audio_path = work / "audio.mp3"
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vn",
              "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio_path)])

    return {"work_dir": work, "duration": meta["duration"], "frames": frames, "audio_path": audio_path}
