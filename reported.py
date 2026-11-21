"""Problems the tenant REPORTED (said out loud): does the VIDEO support, contradict, or not show them?

R1 LOCATE   Neon reads the timestamped transcript and returns, per problem, the object to look for (e.g. "toilet")
            and every time the tenant talks about it. ONE Decisions call over all usable frames asks "is <object>
            visible in frame k?" for every (problem, frame). Moments = said times (snapped to where the object is
            visible) + stretches where the object is on screen.
R2 VIDEO    For each moment, ffmpeg cuts t-3 s .. t+6 s (with audio) and sends it as a VIDEO block to the Decisions
            API: supports / contradicts / not_shown. "contradicts" requires the item to be clearly USED and WORKING.
            If the video block is rejected, 8 frames from the window are sent instead.
R3 ADJUDICATE  If any moment says contradicts (any confidence), or the best answer is < 0.75, CARBON (MEDIUM) watches
            the clip with the tenant's words and decides supports / contradicts / not_shown. Carbon's answer is final.
(R4, the HPD verdict, is in verdict.py.)
"""
from __future__ import annotations

import base64
import json
import logging
import subprocess
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel

import config
import models
import transcribe

log = logging.getLogger("reported")

MAX_PROBLEMS = 6
MAX_MOMENTS = 3
CLIP_BEFORE_S, CLIP_AFTER_S = 3.0, 6.0
SNAP_BEFORE_S, SNAP_AFTER_S = 3.0, 8.0
VISIBLE_P = 0.5
DECISIVE_P = 0.75
CHECK_OPTIONS = {
    "supports": "The reported problem is visible in the clip: the item is used and fails (e.g. the flush button is "
                "pressed and nothing flushes), water is leaking or dripping, or a part is clearly broken.",
    "contradicts": "The item is clearly USED and WORKS normally in the clip: e.g. the flush button is pressed AND the "
                   "water flushes and refills; the burner is turned on AND lights; the window is pushed AND opens. "
                   "Merely seeing the item is NOT enough.",
    "not_shown": "The item is not shown being used, or the problem can't be judged from video (e.g. water "
                 "temperature, noise level, how long a cycle takes).",
}


# ---------------------------------------------------------------- R1 locate

class Located(BaseModel):
    """Where one reported problem is talked about."""
    id: str
    object: str            # what to look for on camera, e.g. "toilet", "stove burners", "washer and dryer"
    said_times: list[float]


class LocateOut(BaseModel):
    """Neon's locate output."""
    problems: list[Located]


LOCATE_PROMPT = """A tenant filmed their apartment and talked about problems. For each problem below, give:
- object: the physical thing a camera would need to show to check it (short noun phrase, e.g. "toilet",
  "stove burners", "kitchen light", "sink faucet", "washer and dryer").
- said_times: the start time in seconds of EVERY transcript line where the tenant talks about this problem
  (including follow-ups like "these two are working").
Use only the transcript. Keep the same ids.

Problems:
{problems}

Transcript (seconds):
{transcript}
"""


def locate(problems: list[dict], transcript: dict) -> dict[str, dict]:
    """NEON: object to look for + every time the problem is mentioned."""
    resp = models.neon(
        LOCATE_PROMPT.format(problems=json.dumps([{"id": p["id"], "problem": p["text"], "quote": p.get("quote")}
                                                  for p in problems], ensure_ascii=False),
                             transcript=transcribe.format_for_prompt(transcript)),
        types.GenerateContentConfig(response_mime_type="application/json", response_schema=LocateOut),
        step="locate", purpose=f"find when each of {len(problems)} reported problem(s) is talked about",
        key=lambda r: "; ".join(f"{x['id']}: {x['object']} @ {x['said_times']}" for x in models.parse_json(r)["problems"]))
    return {x["id"]: x for x in models.parse_json(resp)["problems"]}


def object_visibility(problems: list[dict], located: dict, frame_checks: list[dict], work_dir: Path) -> dict:
    """ONE Decisions call: for every (problem, usable frame): is the problem's object visible? -> {id: {t: p}}."""
    frames = [f for f in frame_checks if f["usable"]]
    if not frames or not problems:
        return {}
    inp = [{"type": "text", "text": "Frames from a tenant's apartment video, in order.\n"}]
    for k, f in enumerate(frames):
        inp.append({"type": "text", "text": f"Frame {k} ({f['t']:.0f} s):\n"})
        inp.append(models.media_block(work_dir / f["frame_path"], resolution=config.FRAME_RESOLUTION))
    questions = {}
    for p in problems:
        obj = located.get(p["id"], {}).get("object") or p["text"]
        for k in range(len(frames)):
            questions[f"{p['id']}_{k}"] = {"type": "noul", "instructions": f"Is the {obj} visible in Frame {k}?"}
    a = models.decide(inp, questions, step="locate_objects",
                      purpose=f"which frames show each reported object ({len(problems)} objects × {len(frames)} frames)",
                      key=lambda a: "; ".join(
                          f"{p['id']}: visible at " + ",".join(f"{frames[k]['t']:.0f}s" for k in range(len(frames))
                                                               if a[f"{p['id']}_{k}"].noul >= VISIBLE_P)
                          for p in problems))
    return {p["id"]: {frames[k]["t"]: a[f"{p['id']}_{k}"].noul for k in range(len(frames))} for p in problems}


def moments_for(said: list[float], visible: dict[float, float]) -> list[dict]:
    """Said times snapped to where the object is visible, plus stretches where it is on screen. Max MAX_MOMENTS."""
    vis_ts = sorted(t for t, p in visible.items() if p >= VISIBLE_P)
    out = []
    for t in sorted(said):
        near = [v for v in vis_ts if t - SNAP_BEFORE_S <= v <= t + SNAP_AFTER_S]
        out.append({"t": near[0] if near else t, "said_at": t, "object_visible": bool(near)})
    runs, cur = [], []
    for v in vis_ts:                                   # stretches of visibility not already covered
        if cur and v - cur[-1] > 3.0:
            runs.append(cur)
            cur = []
        cur.append(v)
    if cur:
        runs.append(cur)
    for run in runs:
        if not any(abs(m["t"] - run[0]) < 6 or run[0] <= m["t"] <= run[-1] for m in out):
            out.append({"t": run[0], "said_at": None, "object_visible": True})
    dedup = []
    for m in sorted(out, key=lambda m: (not m["object_visible"], m["t"])):
        if all(abs(m["t"] - d["t"]) >= 4 for d in dedup):
            dedup.append(m)
    return sorted(dedup[:MAX_MOMENTS], key=lambda m: m["t"])


# ---------------------------------------------------------------- R2 video check

def cut_clip(video: Path, t: float, out: Path) -> Path:
    """ffmpeg: t-3 s .. t+6 s, 480p, 15 fps, with audio -> mp4."""
    start = max(0.0, t - CLIP_BEFORE_S)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{CLIP_BEFORE_S + CLIP_AFTER_S:.2f}",
                    "-i", str(video), "-vf", "scale=-2:480", "-r", "15", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "30", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(out)],
                   check=True, capture_output=True)
    return out


def clip_frames(clip: Path, n: int = 8) -> list[bytes]:
    """n JPEG frames spread over the clip (fallback when a video block is rejected, and for thumbnails)."""
    tmp = clip.with_suffix("")
    tmp.mkdir(exist_ok=True)
    for old in tmp.glob("*.jpg"):
        old.unlink()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-vf", f"fps={n / (CLIP_BEFORE_S + CLIP_AFTER_S):.3f},scale=-2:480",
                    "-frames:v", str(n), str(tmp / "%02d.jpg")], check=True, capture_output=True)
    return [p.read_bytes() for p in sorted(tmp.glob("*.jpg"))]


def check_clip(prob: dict, clip: Path) -> dict:
    """Decisions on the clip as VIDEO (fallback: 8 frames): supports / contradicts / not_shown."""
    quote = prob.get("quote") or prob["text"]
    q = {"check": {"type": "choice",
                   "instructions": f"Compared with what the tenant said ('{quote}'), what does this clip show?",
                   "criteria": CHECK_OPTIONS}}
    text = {"type": "text", "text": f"The tenant said: \"{quote}\" (problem: {prob['text']}). Clip from their video:\n"}
    mode = "video"
    try:
        a = models.decide([text, {"type": "video", "data": base64.b64encode(clip.read_bytes()).decode("ascii"),
                                  "mime_type": "video/mp4", "resolution": config.EVIDENCE_RESOLUTION}], q,
                          step="video_check", purpose=f"video clip check: \"{prob['text']}\" ({clip.name})",
                          key=lambda a: f"{a['check'].choice} ({a['check'].confidence:.2f}) "
                                        + json.dumps({k: round(v, 2) for k, v in a["check"].probabilities.items()}))
    except Exception as exc:
        log.warning("video block failed (%s); falling back to 8 frames", type(exc).__name__)
        mode = "frames"
        blocks = [{"type": "image", "data": base64.b64encode(b).decode("ascii"), "mime_type": "image/jpeg",
                   "resolution": config.EVIDENCE_RESOLUTION} for b in clip_frames(clip)]
        a = models.decide([text, *blocks], q, step="video_check",
                          purpose=f"clip check from 8 frames: \"{prob['text']}\" ({clip.name})",
                          key=lambda a: f"{a['check'].choice} ({a['check'].confidence:.2f})")
    c = a["check"]
    return {"choice": c.choice, "confidence": round(c.confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in c.probabilities.items()}, "mode": mode}


# ---------------------------------------------------------------- R3 adjudicate

class CarbonCheck(BaseModel):
    """Carbon's final word on one clip."""
    verdict: Literal["supports", "contradicts", "not_shown"]
    reason: str


CARBON_PROMPT = """A tenant said: "{quote}" (problem: {problem}).
Watch this clip from their apartment video (with sound) and decide:
- supports: the problem is visible (the item is used and fails, water leaking, a part clearly broken).
- contradicts: the item is clearly USED and WORKS normally (e.g. flush pressed AND water flushes and refills).
  Just seeing the item is not enough.
- not_shown: the item isn't shown being used, or the problem can't be judged from video.
Be strict and literal. Give a one-line reason in plain words."""


def adjudicate(prob: dict, clip: Path) -> dict:
    """CARBON (MEDIUM thinking) watches the clip; its answer is final."""
    resp = models.carbon(
        [types.Part.from_bytes(data=clip.read_bytes(), mime_type="video/mp4"),
         CARBON_PROMPT.format(quote=prob.get("quote") or prob["text"], problem=prob["text"])],
        types.GenerateContentConfig(response_mime_type="application/json", response_schema=CarbonCheck),
        step="video_adjudicate", purpose=f"Carbon re-checks the clip for \"{prob['text']}\" ({clip.name})",
        key=lambda r: (lambda d: f"{d['verdict']}: {d['reason']}")(models.parse_json(r)))
    return models.parse_json(resp)


# ---------------------------------------------------------------- per problem

def check_one(prob: dict, located: dict, visible: dict, video: Path, work_dir: Path,
              frame_checks: list[dict]) -> dict:
    """R1-R3 for one reported problem. Returns the problem with video_check, moments and evidence."""
    loc = located.get(prob["id"], {})
    said = loc.get("said_times") or ([prob["t"]] if prob.get("t") is not None else [])
    moments = moments_for(said, visible.get(prob["id"], {}))
    chain = ["Hydrogen", "Neon"]
    results = []
    for m in moments:
        clip = cut_clip(video, m["t"], work_dir / "clips" / f"{prob['id']}_{int(m['t'])}.mp4")
        results.append({**m, "clip": f"clips/{clip.name}", **check_clip(prob, clip)})
    if results:
        chain.append("Decisions API")
    final, reason, by_carbon, decisive = "not_shown", None, False, None
    sup = [r for r in results if r["choice"] == "supports" and r["confidence"] >= DECISIVE_P]
    con = [r for r in results if r["choice"] == "contradicts"]
    unsure = [r for r in results if r["confidence"] < DECISIVE_P]
    if sup:
        final, decisive = "supports", max(sup, key=lambda r: r["confidence"])
    elif con or unsure:
        decisive = (max(con, key=lambda r: r["probabilities"].get("contradicts", 0)) if con else
                    max(unsure, key=lambda r: 1 - r["probabilities"].get("not_shown", 0)))
        try:
            v = adjudicate(prob, work_dir / decisive["clip"])
            final, reason, by_carbon = v["verdict"], v["reason"], True
            chain.append("Carbon")
        except Exception as exc:
            log.warning("Carbon adjudication failed: %s", exc)
            final = decisive["choice"] if decisive["choice"] != "contradicts" else "not_shown"
            reason = "Re-check unavailable; not counted as contradicting."
    elif results:
        decisive = max(results, key=lambda r: r["confidence"])
        final = decisive["choice"]
    shown_t = decisive["t"] if (decisive and final in ("supports", "contradicts")) else None
    ev = []
    if shown_t is not None:
        ev = [f["frame_path"] for f in frame_checks if f["usable"]
              and shown_t - CLIP_BEFORE_S <= f["t"] <= shown_t + CLIP_AFTER_S][:3]
    return {**prob, "object": loc.get("object"), "said_times": said, "moments": results,
            "video_check": final, "video_reason": reason, "video_by_carbon": by_carbon,
            "visible": final in ("supports", "contradicts"), "t_start": shown_t, "t_end": shown_t,
            "clip": decisive["clip"] if (decisive and shown_t is not None) else None,
            "evidence_frames": ev, "evidence_frame": ev[0] if ev else None, "decided_chain": chain,
            "confidence": decisive["confidence"] if decisive else None}


def check_reported(household: dict | None, transcript: dict | None, frame_checks: list[dict],
                   work_dir: Path, video: Path) -> list[dict]:
    """R1-R3 for every reported problem (max MAX_PROBLEMS). Video-only / unknown household -> []."""
    if not household or household.get("unknown"):
        return []
    probs = [{"id": f"r{i + 1}", **p} for i, p in enumerate((household.get("reported_problems") or [])[:MAX_PROBLEMS])]
    if not probs:
        return []
    work_dir = Path(work_dir)
    try:
        located = locate(probs, transcript or {"segments": []})
    except Exception as exc:
        log.warning("locate failed: %s", exc)
        located = {}
    try:
        visible = object_visibility(probs, located, frame_checks, work_dir)
    except Exception as exc:
        log.warning("object visibility failed: %s", exc)
        visible = {}
    out = []
    for p in probs:
        try:
            out.append(check_one(p, located, visible, Path(video), work_dir, frame_checks))
        except Exception as exc:
            log.warning("video check failed for %s: %s", p["id"], exc)
            out.append({**p, "video_check": "not_shown", "visible": False, "evidence_frames": [],
                        "decided_chain": ["Hydrogen", "Neon"], "error": type(exc).__name__})
    return out
