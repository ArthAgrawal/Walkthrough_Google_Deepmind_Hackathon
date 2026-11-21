"""Glues Section 6 steps 1-11 together. `run(video_path, intro_override=None) -> Report`."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import classify_route
import config
import context as context_mod
import frame_checks as frame_checks_mod
import household as household_mod
import ingest as ingest_mod
import rules
import thermostat as thermostat_mod
import transcribe as transcribe_mod
import write_verify


class Trace:
    """Collects {"step","model","calls","ms"} entries; optional on_event(entries, running_step) callback."""

    def __init__(self, on_event: Callable[[list[dict], str | None], None] | None = None):
        """Start an empty trace."""
        self.entries: list[dict] = []
        self.on_event = on_event

    def _emit(self, running: str | None) -> None:
        """Notify the listener (e.g. the server's job status) of progress."""
        if self.on_event:
            self.on_event(list(self.entries), running)

    def run(self, step: str, model: str | None, calls: int | Callable, fn, *args, **kwargs):
        """Time fn(*args, **kwargs), record a trace entry, and return its result.

        `calls` may be a function of the result (for steps whose call count is known only after).
        """
        self._emit(step)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        n = calls(result) if callable(calls) else calls
        self.entries.append({"step": step, "model": model if n else None, "calls": n,
                             "ms": round((time.perf_counter() - t0) * 1000)})
        self._emit(None)
        return result


def run_steps_1_7(video_path: str | Path, intro_override: str | None = None, video_only: bool = False,
                  local_time: str | None = None, outdoor_f: float | None = None,
                  trace: Trace | None = None, work_dir: str | Path | None = None) -> dict:
    """Run Section 6 steps 1-7 on one clip and return a partial Report.

    intro_override: use this text instead of the spoken transcript for household facts.
    video_only:     drop all spoken facts (Section 10 baseline): no transcription, household=None.
    local_time / outdoor_f: override context (for eval replays).
    """
    tr = trace or Trace()
    ing = tr.run("ingest", None, 0, ingest_mod.ingest, video_path, work_dir)
    work = ing["work_dir"]

    transcript, household = None, None
    if not video_only:
        if ing["audio_path"]:
            transcript = tr.run("transcribe", config.HYDROGEN, 1, transcribe_mod.transcribe, ing["audio_path"])
        else:
            transcript = transcribe_mod.transcribe(None)
        text = intro_override if intro_override is not None else transcribe_mod.format_for_prompt(transcript)
        household = tr.run("household", config.NEON, 1, household_mod.extract, text, transcript["language"])

    checks = tr.run("frame_checks", config.DECISION, len(ing["frames"]),
                    frame_checks_mod.check_frames, ing["frames"], work)
    thermo = tr.run("thermostat", config.NEON, lambda r: int(r is not None),
                    thermostat_mod.read_thermostat, checks, work)
    ctx = tr.run("context", None, 0, context_mod.get_context, local_time, outdoor_f)
    candidates = tr.run("rules", None, 0, rules.evaluate, checks, household, ctx, thermo, transcript)

    report = {
        "video": str(Path(video_path)), "work_dir": str(work), "duration": ing["duration"],
        "video_only": video_only, "transcript": transcript, "household": household, "context": ctx,
        "thermostat": thermo, "frame_checks": checks, "candidates": candidates, "trace": tr.entries,
    }
    (Path(work) / ("steps_1_7_video_only.json" if video_only else "steps_1_7.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run(video_path: str | Path, intro_override: str | None = None, video_only: bool = False,
        local_time: str | None = None, outdoor_f: float | None = None,
        on_event: Callable[[list[dict], str | None], None] | None = None,
        work_dir: str | Path | None = None) -> dict:
    """Run steps 1-11 and return the Report (Section 7), also saved as report.json in the work dir."""
    tr = Trace(on_event)
    r = run_steps_1_7(video_path, intro_override, video_only, local_time, outdoor_f, tr, work_dir)
    work = Path(r["work_dir"])

    routed = tr.run("route", config.CARBON, lambda x: x["carbon_calls"], classify_route.route,
                    r["candidates"], r["frame_checks"], r["transcript"], work)
    tr.run("classify", config.DECISION, len(routed["findings"]),
           lambda: [classify_route.classify(f, r["household"], work) for f in routed["findings"]])
    writing = tr.run("write", config.NEON, 1, write_verify.write, routed, r["household"], r["context"])
    verified = tr.run("verify", config.DECISION, lambda v: len({c["finding_id"] for c in v}),
                      write_verify.verify, writing.get("claims", []), routed["findings"],
                      r["context"], r["transcript"], work)
    complaint = write_verify.assemble_complaint(writing.get("complaint_intro", ""), verified, routed,
                                                r["household"], r["context"])

    report = {
        "household": r["household"], "context": r["context"], "thermostat": r["thermostat"],
        "findings": routed["findings"], "not_violations": routed["not_violations"],
        "refilm": routed["refilm"], "dropped": routed["dropped"],
        "rights_explanation": writing.get("rights_explanation", ""), "complaint_draft_en": complaint,
        "claims": verified, "transcript": r["transcript"], "trace": tr.entries,
        "caseworker_job_id": None, "work_dir": r["work_dir"], "duration": r["duration"],
        "video_only": video_only, "frames_total": len(r["frame_checks"]),
        "frames_usable": sum(f["usable"] for f in r["frame_checks"]),
    }
    (work / ("report_video_only.json" if video_only else "report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
