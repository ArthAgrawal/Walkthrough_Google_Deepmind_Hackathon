"""FastAPI server: POST /analyze (upload video) -> job id; GET /analyze/{job_id} -> live trace + Report.

Run:  uvicorn server:app --port 8000     then open http://localhost:8000
Evidence frames are served from /work/<job_id>/frames/NNNN.jpg.
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


def _run_job(job_id: str, video: Path, intro: str | None) -> None:
    """Background thread: run the pipeline, streaming trace progress into JOBS[job_id]."""
    job = JOBS[job_id]

    def on_event(entries: list[dict], running: str | None) -> None:
        """Pipeline progress callback."""
        job.update(trace=entries, running=running)

    with _run_lock:
        job["status"] = "running"
        try:
            report = pipeline.run(video, intro_override=intro, on_event=on_event,
                                  work_dir=config.WORK_DIR / job_id)
            report["frame_base"] = f"/work/{job_id}/"
            report.pop("work_dir", None)
            job.update(status="done", report=report, running=None, trace=report["trace"])
        except Exception as exc:
            log.error("job %s failed:\n%s", job_id, traceback.format_exc())
            job.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:500]}", running=None)


@app.post("/analyze")
async def analyze(video: UploadFile = File(...), intro: str | None = Form(None)) -> dict:
    """Accept a video (+ optional typed intro that overrides the spoken one); start analysis."""
    suffix = Path(video.filename or "clip.webm").suffix.lower() or ".webm"
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {suffix}; use webm/mp4/mov.")
    job_id = uuid.uuid4().hex[:12]
    path = UPLOAD_DIR / f"{job_id}{suffix}"
    path.write_bytes(await video.read())
    JOBS[job_id] = {"status": "queued", "trace": [], "running": None, "report": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, path, (intro or "").strip() or None), daemon=True).start()
    return {"job_id": job_id}


@app.get("/analyze/{job_id}")
def job_status(job_id: str) -> dict:
    """Return {"status": queued|running|done|error, "trace", "running", "report", "error"}."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


app.mount("/work", StaticFiles(directory=config.WORK_DIR), name="work")
app.mount("/", StaticFiles(directory=config.ROOT / "static", html=True), name="static")
