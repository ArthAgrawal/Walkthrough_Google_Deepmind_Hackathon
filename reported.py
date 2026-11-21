"""Problems the tenant REPORTED (said out loud): is each one visible in the video?

ONE Decisions call per reported problem, on the frames filmed around when they said it
(medium resolution, evidence frames): a yes/no "visible?" + a choice "which frame shows it best?".
Many reports can't be seen on video (noise, appliances not working); those are listed as
"reported, not shown" so the tenant knows to describe them in the complaint.
"""
from __future__ import annotations

from pathlib import Path

import config
import models

MAX_PROBLEMS = 6
MAX_FRAMES = 4
WINDOW_BEFORE_S, WINDOW_AFTER_S = 2.0, 10.0
VISIBLE_P = 0.5


def _frames_for(t: float, frame_checks: list[dict]) -> list[dict]:
    """Usable frames from t-2 s to t+10 s (else the 3 nearest to t), at most MAX_FRAMES, evenly spread."""
    usable = [f for f in frame_checks if f["usable"]]
    near = [f for f in usable if t - WINDOW_BEFORE_S <= f["t"] <= t + WINDOW_AFTER_S]
    if not near:
        near = sorted(sorted(usable, key=lambda f: abs(f["t"] - t))[:3], key=lambda f: f["t"])
    if len(near) > MAX_FRAMES:
        step = len(near) / MAX_FRAMES
        near = [near[int(i * step)] for i in range(MAX_FRAMES)]
    return near


def check_one(i: int, prob: dict, frame_checks: list[dict], work_dir: Path) -> dict:
    """Check one reported problem; returns a card dict (id r1.., visible, evidence frames)."""
    frames = _frames_for(float(prob.get("t") or 0), frame_checks)
    base = {"id": f"r{i + 1}", "text": prob["text"], "quote": prob.get("quote"), "t_said": prob.get("t"),
            "decided_by": "decisions", "evidence_frames": [], "evidence_frame": None,
            "t_start": None, "t_end": None}
    if not frames:
        return {**base, "visible": False, "p_visible": None, "confidence": None}
    labels = [chr(65 + k) for k in range(len(frames))]
    inp = [{"type": "text", "text": f"A NYC tenant filming their apartment said: \"{prob.get('quote') or prob['text']}\" "
                                    f"(meaning: {prob['text']}). Frames from around that moment follow.\n"}]
    for lab, f in zip(labels, frames):
        inp.append({"type": "text", "text": f"Frame {lab} ({f['t']:.0f} s):\n"})
        inp.append(models.media_block(work_dir / f["frame_path"], resolution=config.EVIDENCE_RESOLUTION))
    questions = {"visible": {
        "type": "noul",
        "instructions": f"Is this reported problem visible in at least one of the frames: {prob['text']}?",
        "criteria": {"true_description": "The problem itself, or a clear physical sign of it, is visible.",
                     "false_description": "Not visible, or the item is shown but the problem can't be seen "
                                          "(e.g. a noise, or an appliance that looks normal)."}}}
    if len(frames) >= 2:
        questions["best_frame"] = {"type": "choice", "instructions": "Which frame shows the reported problem best?",
                                   "criteria": {lab: f"Frame {lab} ({f['t']:.0f} s)" for lab, f in zip(labels, frames)}}
    a = models.decide(inp, questions, step="reported",
                      purpose=f"is the reported problem \"{prob['text']}\" visible? ({len(frames)} frames)",
                      key=lambda a: f"P(visible)={a['visible'].noul:.2f}"
                                    + (f", best frame {a['best_frame'].choice}" if "best_frame" in a else ""))
    p = a["visible"].noul
    best = frames[labels.index(a["best_frame"].choice)] if "best_frame" in a else frames[0]
    visible = p >= VISIBLE_P
    return {**base, "visible": visible, "p_visible": round(p, 3), "confidence": round(max(p, 1 - p), 3),
            "evidence_frame": best["frame_path"] if visible else None,
            "evidence_frames": ([best["frame_path"]] + [f["frame_path"] for f in frames if f is not best])[:3]
                               if visible else [],
            "t_start": best["t"] if visible else None, "t_end": best["t"] if visible else None}


def check_reported(household: dict | None, frame_checks: list[dict], work_dir: Path) -> list[dict]:
    """Check every reported problem (max MAX_PROBLEMS). Video-only / unknown household -> []."""
    if not household or household.get("unknown"):
        return []
    out = []
    for i, prob in enumerate((household.get("reported_problems") or [])[:MAX_PROBLEMS]):
        try:
            out.append(check_one(i, prob, frame_checks, Path(work_dir)))
        except Exception as exc:
            out.append({"id": f"r{i + 1}", "text": prob["text"], "quote": prob.get("quote"), "visible": False,
                        "p_visible": None, "evidence_frames": [], "error": type(exc).__name__})
    return out
