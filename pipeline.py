"""Glues Section 6 steps 1-11 together. `run(video_path, intro_override=None) -> Report`."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import call_log
import classify_route
import config
import models
import context as context_mod
import frame_checks as frame_checks_mod
import household as household_mod
import ingest as ingest_mod
import reported
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
                  trace: Trace | None = None, work_dir: str | Path | None = None,
                  reuse_frames: bool = False) -> dict:
    """Run Section 6 steps 1-7 on one clip and return a partial Report.

    intro_override: use this text instead of the spoken transcript for household facts.
    video_only:     drop all spoken facts (Section 10 baseline): no transcription, household=None.
    local_time / outdoor_f: override context (for eval replays).
    reuse_frames:   reuse frames, frame checks, thermostat and context from work_dir/steps_1_7.json
                    (the "run again without audio" comparison: only the spoken facts change).
    """
    tr = trace or Trace()
    prev = None
    if reuse_frames and work_dir and (Path(work_dir) / "steps_1_7.json").exists():
        prev = json.loads((Path(work_dir) / "steps_1_7.json").read_text(encoding="utf-8"))
    if prev:
        work, duration, no_audio, audio_path = Path(work_dir), prev["duration"], prev["no_audio"], None
    else:
        ing = tr.run("ingest", None, 0, ingest_mod.ingest, video_path, work_dir)
        work, duration, audio_path = ing["work_dir"], ing["duration"], ing["audio_path"]
        no_audio = audio_path is None

    transcript, household = None, None
    if not video_only:
        if not no_audio:
            transcript = tr.run("transcribe", config.HYDROGEN, 1, transcribe_mod.transcribe, audio_path)
        else:
            transcript = transcribe_mod.transcribe(None)      # no audio track: empty transcript, no call
        if intro_override is not None:
            household = tr.run("household", config.NEON, 1, household_mod.extract, intro_override,
                               transcript["language"])
        elif no_audio:
            household = household_mod.unknown_facts("no audio track")
        else:
            household = tr.run("household", config.NEON, 1, household_mod.extract,
                               transcribe_mod.format_for_prompt(transcript), transcript["language"])

    if prev:
        checks, thermo, ctx = prev["frame_checks"], prev["thermostat"], prev["context"]
        tr.entries.append({"step": "frame_checks", "model": None, "calls": 0, "ms": 0, "reused": True})
    else:
        checks = tr.run("frame_checks", config.DECISION, len(ing["frames"]),
                        frame_checks_mod.check_frames, ing["frames"], work)
        thermo = tr.run("thermostat", config.NEON, lambda r: int(r is not None),
                        thermostat_mod.read_thermostat, checks, work)
        ctx = tr.run("context", None, 0, context_mod.get_context, local_time, outdoor_f)
    candidates = tr.run("rules", None, 0, rules.evaluate, checks, household, ctx, thermo, transcript)

    report = {
        "video": str(Path(video_path)), "work_dir": str(work), "duration": duration,
        "video_only": video_only, "no_audio": no_audio, "transcript": transcript, "household": household,
        "context": ctx, "thermostat": thermo, "frame_checks": checks, "candidates": candidates,
        "trace": tr.entries,
    }
    (Path(work) / ("steps_1_7_video_only.json" if video_only else "steps_1_7.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run(video_path: str | Path, intro_override: str | None = None, video_only: bool = False,
        local_time: str | None = None, outdoor_f: float | None = None,
        on_event: Callable[[list[dict], str | None], None] | None = None,
        work_dir: str | Path | None = None, address: str | None = None,
        on_partial: Callable[[dict], None] | None = None, reuse_frames: bool = False) -> dict:
    """Run steps 1-11 and return the Report (Section 7), also saved as report.json in the work dir.

    on_partial(report) fires once findings are ready, before the (slower) writing + verification.
    Every model call is collected into the per-job log (model_log.json + report["model_log"]).
    """
    with models.collect_calls() as calls:
        report = _run(video_path, intro_override, video_only, local_time, outdoor_f, on_event, work_dir,
                      address, on_partial, reuse_frames)
    work = Path(report["work_dir"])
    report["model_log"] = call_log.build(calls, work, "model_log_video_only.json" if video_only else "model_log.json")
    (work / ("report_video_only.json" if video_only else "report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def attach_evidence_frames(items: list[dict], frame_checks: list[dict], n: int = 3) -> None:
    """Give each finding up to n evidence thumbnails: its evidence frame + usable frames in its time range."""
    for f in items:
        if f.get("t_start") is None or not f.get("evidence_frame"):
            continue
        inside = [c["frame_path"] for c in frame_checks if c["usable"]
                  and f["t_start"] <= c["t"] <= f["t_end"] and c["frame_path"] != f["evidence_frame"]]
        if len(inside) > n - 1:
            step = len(inside) / (n - 1)
            inside = [inside[int(i * step)] for i in range(n - 1)]
        f["evidence_frames"] = [f["evidence_frame"]] + inside


def _run(video_path, intro_override, video_only, local_time, outdoor_f, on_event, work_dir,
         address=None, on_partial=None, reuse_frames=False) -> dict:
    """Steps 1-11 body for run()."""
    tr = Trace(on_event)
    r = run_steps_1_7(video_path, intro_override, video_only, local_time, outdoor_f, tr, work_dir, reuse_frames)
    work = Path(r["work_dir"])

    routed = tr.run("route", config.CARBON, lambda x: x["carbon_calls"], classify_route.route,
                    r["candidates"], r["frame_checks"], r["transcript"], work)
    tr.run("classify", config.DECISION, len(routed["findings"]),
           lambda: [classify_route.classify(f, r["household"], work) for f in routed["findings"]])
    reported_items = tr.run("reported", config.DECISION,
                            lambda x: sum(i.get("p_visible") is not None for i in x),
                            reported.check_reported, r["household"], r["frame_checks"], work)
    for key in ("findings", "not_violations", "refilm"):
        attach_evidence_frames(routed[key], r["frame_checks"])

    report = {
        "household": r["household"], "context": r["context"], "thermostat": r["thermostat"],
        "address": address, "findings": routed["findings"], "reported": reported_items,
        "not_violations": routed["not_violations"], "refilm": routed["refilm"], "dropped": routed["dropped"],
        "rights_explanation": None, "complaint_draft_en": None, "claims": [],
        "transcript": r["transcript"], "trace": tr.entries, "caseworker_job_id": None,
        "work_dir": r["work_dir"], "duration": r["duration"], "video_only": video_only,
        "no_audio": r["no_audio"], "frames_total": len(r["frame_checks"]),
        "frames_usable": sum(f["usable"] for f in r["frame_checks"]), "complete": False,
    }
    if on_partial:
        on_partial(json.loads(json.dumps(report)))

    writing = tr.run("write", config.NEON, 1, write_verify.write, routed, r["household"], r["context"],
                     reported_items)
    verified = tr.run("verify", config.DECISION, lambda v: len({c["finding_id"] for c in v}),
                      write_verify.verify, writing.get("claims", []), routed["findings"],
                      r["context"], r["transcript"], work)
    # Claim check as a false-alarm filter: a finding whose every claim failed goes to "film again".
    for f in list(routed["findings"]):
        fc = [c for c in verified if c["finding_id"] == f["id"]]
        if fc and not any(c["kept"] for c in fc):
            routed["findings"].remove(f)
            f["refilm_reason"] = ("Our evidence check couldn't confirm this from the video. "
                                  "Please film it again, closer and steadier.")
            f["claim_check_failed"] = True
            routed["refilm"].append(f)
    complaint = write_verify.assemble_complaint(writing.get("complaint_intro", ""), verified, routed,
                                                r["household"], r["context"], reported_items, address)
    report.update(findings=routed["findings"], refilm=routed["refilm"],
                  rights_explanation=writing.get("rights_explanation", ""), complaint_draft_en=complaint,
                  claims=verified, trace=tr.entries, complete=True)
    return report
