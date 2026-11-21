"""Glues the pipeline steps together. Currently steps 1-7 (ingest -> rules); 8-11 come next."""
from __future__ import annotations

import json
import time
from pathlib import Path

import config
import context as context_mod
import frame_checks as frame_checks_mod
import household as household_mod
import ingest as ingest_mod
import rules
import thermostat as thermostat_mod
import transcribe as transcribe_mod


class Trace:
    """Collects {"step","model","calls","ms"} entries for the Report's trace panel."""

    def __init__(self):
        """Start an empty trace."""
        self.entries: list[dict] = []

    def run(self, step: str, model: str | None, calls: int, fn, *args, **kwargs):
        """Time fn(*args, **kwargs), record a trace entry, and return its result."""
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        self.entries.append({"step": step, "model": model, "calls": calls,
                             "ms": round((time.perf_counter() - t0) * 1000)})
        return result


def run_steps_1_7(video_path: str | Path, intro_override: str | None = None, video_only: bool = False,
                  local_time: str | None = None, outdoor_f: float | None = None) -> dict:
    """Run Section 6 steps 1-7 on one clip and return a partial Report.

    intro_override: use this text instead of the spoken transcript for household facts.
    video_only:     drop all spoken facts (Section 10 baseline): no transcription, household=None.
    local_time / outdoor_f: override context (for eval replays).
    """
    tr = Trace()
    ing = tr.run("ingest", None, 0, ingest_mod.ingest, video_path)
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
    has_thermo = thermostat_mod.best_thermostat_frame(checks) is not None
    thermo = tr.run("thermostat", config.NEON if has_thermo else None, int(has_thermo),
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
