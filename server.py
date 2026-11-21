"""FastAPI server for the Walkthrough page.

  POST /analyze                      upload video (+ intro, video_only, address) -> {job_id}
  GET  /analyze/{job_id}             status, live trace, partial report (findings first), final report
  POST /analyze/{job_id}/video_only  re-run the same video without audio, reusing its frame checks
  POST /analyze/{job_id}/retry       re-run a failed job with the same inputs
Evidence frames are served from /work/<work dir>/frames/NNNN.jpg.

Run:  uvicorn server:app --port 8000     then open http://localhost:8000
One analysis runs at a time (temporary hackathon accounts: keep request volume modest).
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

import config
import pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

UPLOAD_DIR = config.RAW_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.WORK_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED = {".webm", ".mp4", ".mov", ".mkv"}

app = FastAPI(title="Walkthrough")
JOBS: dict[str, dict] = {}
_run_lock = threading.Lock()


def friendly_error(exc: Exception) -> str:
    """Map an exception to a short message a tenant can act on (no stack traces)."""
    text = f"{type(exc).__name__} {exc}"
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status == 429 or "RESOURCE_EXHAUSTED" in text:
        return "The AI service is busy or our usage limit was reached. Wait a minute, then retry."
    if status in (500, 502, 503, 504) or "UNAVAILABLE" in text or "overloaded" in text.lower():
        return "The AI model is overloaded right now. Please retry in a moment."
    if "ffmpeg" in text or "ffprobe" in text:
        return "We couldn't read this video file. Try recording again or upload an mp4/webm."
    if "GEMINI_API_KEY" in text:
        return "The server has no API key configured."
    return "Something went wrong while analyzing this video. Please retry."


def _start(video: Path, intro: str | None, video_only: bool, address: str | None,
           work_key: str | None = None, reuse_frames: bool = False, parent: str | None = None) -> str:
    """Create a job and run the pipeline in a background thread. Returns the job id."""
    job_id = uuid.uuid4().hex[:12]
    work_key = work_key or job_id
    JOBS[job_id] = {"status": "queued", "trace": [], "running": None, "partial": None, "report": None,
                    "error": None, "parent": parent,
                    "inputs": {"video": str(video), "intro": intro, "video_only": video_only,
                               "address": address, "work_key": work_key, "reuse_frames": reuse_frames}}

    def work() -> None:
        """Background thread body."""
        job = JOBS[job_id]

        def on_event(entries: list[dict], running: str | None) -> None:
            """Pipeline progress callback."""
            job.update(trace=entries, running=running)

        def on_partial(rep: dict) -> None:
            """Findings are ready: publish them before writing/verification finishes."""
            rep["frame_base"] = f"/work/{work_key}/"
            rep.pop("work_dir", None)
            job["partial"] = rep

        with _run_lock:
            job["status"] = "running"
            try:
                rep = pipeline.run(video, intro_override=intro, video_only=video_only, on_event=on_event,
                                   work_dir=config.WORK_DIR / work_key, address=address,
                                   on_partial=on_partial, reuse_frames=reuse_frames)
                rep["frame_base"] = f"/work/{work_key}/"
                rep.pop("work_dir", None)
                job.update(status="done", report=rep, running=None, trace=rep["trace"])
            except Exception as exc:
                log.error("job %s failed:\n%s", job_id, traceback.format_exc())
                job.update(status="error", error=friendly_error(exc), running=None)

    threading.Thread(target=work, daemon=True).start()
    return job_id


@app.post("/analyze")
async def analyze(video: UploadFile = File(...), intro: str | None = Form(None),
                  video_only: bool = Form(False), address: str | None = Form(None)) -> dict:
    """Accept a video (+ optional typed intro, video_only flag, address); start analysis."""
    suffix = Path(video.filename or "clip.webm").suffix.lower() or ".webm"
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {suffix}; please use webm, mp4 or mov.")
    upload_id = uuid.uuid4().hex[:12]
    path = UPLOAD_DIR / f"{upload_id}{suffix}"
    path.write_bytes(await video.read())
    intro = None if video_only else ((intro or "").strip() or None)
    return {"job_id": _start(path, intro, video_only, (address or "").strip() or None)}


def _job(job_id: str) -> dict:
    """Look up a job or 404."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (the server may have restarted). Please start a new walkthrough.")
    return job


@app.get("/analyze/{job_id}")
def job_status(job_id: str) -> dict:
    """Return {"status": queued|running|done|error, "trace", "running", "partial", "report", "error"}."""
    job = _job(job_id)
    return {k: v for k, v in job.items() if k != "inputs"}


@app.post("/analyze/{job_id}/video_only")
def rerun_video_only(job_id: str) -> dict:
    """Re-run the same video video-only, reusing its frames + frame checks (only spoken facts change)."""
    inp = _job(job_id)["inputs"]
    return {"job_id": _start(Path(inp["video"]), None, True, inp["address"], work_key=inp["work_key"],
                             reuse_frames=True, parent=job_id)}


@app.post("/analyze/{job_id}/retry")
def retry(job_id: str) -> dict:
    """Re-run a job with the same inputs (after a friendly error)."""
    inp = _job(job_id)["inputs"]
    return {"job_id": _start(Path(inp["video"]), inp["intro"], inp["video_only"], inp["address"],
                             work_key=inp["work_key"] if inp["reuse_frames"] else None,
                             reuse_frames=inp["reuse_frames"], parent=job_id)}


app.mount("/work", StaticFiles(directory=config.WORK_DIR), name="work")
app.mount("/", StaticFiles(directory=config.ROOT / "static", html=True), name="static")
