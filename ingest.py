"""Step 1: ffmpeg -> frames (1 fps, max 40, near-duplicates skipped; JPEG 768 px wide) + mono mp3 audio."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

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
    try:  # browser MediaRecorder webm often reports duration "N/A"
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
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
    fps = sampling_fps(meta["duration"])
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
          "-vf", f"fps={fps:.5f},scale={config.FRAME_WIDTH}:-2",
          "-q:v", "3", "-start_number", "0", str(frames_dir / "%04d.jpg")])
    all_frames = [{"t": round(int(p.stem) / fps, 2), "frame_path": f"frames/{p.name}"}
                  for p in sorted(frames_dir.glob("*.jpg"))]
    if len(all_frames) > config.MAX_FRAMES:   # duration unknown (e.g. webm "N/A"): thin evenly
        step = len(all_frames) / config.MAX_FRAMES
        all_frames = [all_frames[int(i * step)] for i in range(config.MAX_FRAMES)]
    frames = drop_near_duplicates(all_frames, work)

    audio_path = None
    if meta["has_audio"]:
        audio_path = work / "audio.mp3"
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vn",
              "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio_path)])

    duration = meta["duration"] or (all_frames[-1]["t"] + 1 / fps if all_frames else 0.0)
    return {"work_dir": work, "duration": duration, "frames": frames, "audio_path": audio_path,
            "fps": fps, "frames_extracted": len(all_frames), "frames_skipped_dup": len(all_frames) - len(frames)}


def sampling_fps(duration: float) -> float:
    """config.FPS for clips up to MAX_FRAMES seconds; longer clips get MAX_FRAMES frames spread evenly."""
    if duration and duration > config.MAX_FRAMES / config.FPS:
        return config.MAX_FRAMES / duration
    return float(config.FPS)


def _thumb(path: Path) -> list[int]:
    """64x36 grayscale pixels for cheap frame comparison."""
    with Image.open(path) as im:
        return list(im.convert("L").resize((64, 36)).getdata())


def drop_near_duplicates(frames: list[dict], work: Path) -> list[dict]:
    """Skip frames nearly identical to the last kept one (mean abs diff < DEDUP_DIFF),
    but never more than DEDUP_MAX_RUN in a row."""
    kept, last, run = [], None, 0
    for f in frames:
        px = _thumb(work / f["frame_path"])
        if last is not None and run < config.DEDUP_MAX_RUN:
            diff = sum(abs(a - b) for a, b in zip(px, last)) / len(px)
            if diff < config.DEDUP_DIFF:
                run += 1
                continue
        kept.append(f)
        last, run = px, 0
    return kept
