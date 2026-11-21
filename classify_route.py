"""Steps 8-9: route candidates by confidence (Carbon re-checks the unsure ones), then classify
each included finding with ONE Decisions call (HPD category choice + likely-severity score).

Routing (Section 6 step 9), on rules.py candidates:
  is_violation False             -> not_violations (if confidence >= CONF_LOW)
  is_violation None              -> refilm / confirm
  confidence < CONF_LOW          -> dropped
  CONF_LOW <= confidence < HIGH  -> CARBON re-checks frames ±5 s + transcript:
                                    present -> findings (escalated); absent -> dropped; unsure -> refilm
  confidence >= CONF_HIGH        -> findings
Classification runs after routing so we only spend calls on findings we keep.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from google.genai import types
from pydantic import BaseModel

import config
import models
import transcribe
from rules import RULE_LABELS


class CarbonVerdict(BaseModel):
    """Carbon's re-check of one uncertain finding."""
    verdict: Literal["present", "absent", "unsure"]
    best_frame_t: Optional[float]
    reason: str


CARBON_PROMPT = """You are double-checking an automated housing-conditions detector.
It flagged a possible **{label}** between {t0:.0f}s and {t1:.0f}s of a tenant's apartment video,
but with only moderate confidence ({conf:.2f}). Detector note: {explanation}

Above are video frames from around that time, each preceded by its timestamp.
Tenant's narration (may be in another language):
{narration}

Look carefully. Is the condition actually visible?
- "present": clearly visible in at least one frame (give that frame's timestamp as best_frame_t).
- "absent": the frames show it is NOT there (e.g. a guard is installed; it's a shadow, not a stain).
- "unsure": the frames are not clear enough to tell.
Be strict: rust is not water damage; shadows and decor are not mold. Give a one-sentence reason."""


def _frames_near(frame_checks: list[dict], t0: float, t1: float) -> list[dict]:
    """Usable frames within ±CARBON_WINDOW_S of [t0, t1], evenly subsampled to CARBON_MAX_FRAMES."""
    near = [f for f in frame_checks if f["usable"]
            and t0 - config.CARBON_WINDOW_S <= f["t"] <= t1 + config.CARBON_WINDOW_S]
    if len(near) > config.CARBON_MAX_FRAMES:
        step = len(near) / config.CARBON_MAX_FRAMES
        near = [near[int(i * step)] for i in range(config.CARBON_MAX_FRAMES)]
    return near


def carbon_recheck(finding: dict, frame_checks: list[dict], transcript: dict | None, work_dir: Path) -> dict:
    """Ask CARBON to re-check one finding; returns the CarbonVerdict dict."""
    frames = _frames_near(frame_checks, finding["t_start"], finding["t_end"])
    if not frames:
        return {"verdict": "unsure", "best_frame_t": None, "reason": "No usable frames near this time."}
    parts: list = []
    for f in frames:
        parts.append(f"[{f['t']:.0f}s]")
        parts.append(types.Part.from_bytes(data=(work_dir / f["frame_path"]).read_bytes(), mime_type="image/jpeg"))
    parts.append(CARBON_PROMPT.format(
        label=RULE_LABELS[finding["rule"]], t0=finding["t_start"], t1=finding["t_end"],
        conf=finding["confidence"], explanation=finding["explanation"],
        narration=transcribe.format_for_prompt(transcript or {}) or "(none)"))
    outcome = {"present": "kept", "absent": "dropped", "unsure": "film again"}
    resp = models.carbon(
        parts, types.GenerateContentConfig(response_mime_type="application/json", response_schema=CarbonVerdict),
        step="carbon_recheck",
        purpose=f"re-checked uncertain {RULE_LABELS[finding['rule']]} finding "
                f"(confidence {finding['confidence']:.2f}, {len(frames)} frames)",
        key=lambda r: (lambda d: f"{d['verdict']} → {outcome[d['verdict']]}: {d['reason']}")(models.parse_json(r)),
    )
    return models.parse_json(resp)


def route(candidates: list[dict], frame_checks: list[dict], transcript: dict | None,
          work_dir: Path) -> dict:
    """Split candidates into findings / not_violations / refilm / dropped (see module docstring)."""
    out = {"findings": [], "not_violations": [], "refilm": [], "dropped": []}
    carbon_used = 0
    for c in candidates:
        f = dict(c)
        conf = f["confidence"]
        if conf < config.CONF_LOW:
            f["route_reason"] = "confidence below threshold"
            out["dropped"].append(f)
        elif f["is_violation"] is False:
            out["not_violations"].append(f)
        elif f["is_violation"] is None:
            f["refilm_reason"] = f["explanation"]
            out["refilm"].append(f)
        elif conf >= config.CONF_HIGH:
            out["findings"].append(f)
        elif carbon_used >= config.CARBON_MAX_PER_CLIP:
            f["refilm_reason"] = "Not sure from this footage (re-check budget used). Please film this spot again, closer and steadier."
            out["refilm"].append(f)
        else:
            carbon_used += 1
            try:
                v = carbon_recheck(f, frame_checks, transcript, work_dir)
            except Exception as exc:  # timeout or API failure: don't block the report
                v = {"verdict": "unsure", "best_frame_t": None,
                     "reason": f"re-check unavailable ({type(exc).__name__})"}
            f.update(escalated=True, carbon=v)
            if v["verdict"] == "present":
                f["decided_by"] = "carbon"
                if v.get("best_frame_t") is not None:
                    best = min((x for x in frame_checks if x["usable"]),
                               key=lambda x: abs(x["t"] - v["best_frame_t"]), default=None)
                    if best:
                        f["evidence_frame"] = best["frame_path"]
                out["findings"].append(f)
            elif v["verdict"] == "absent":
                f["route_reason"] = f"Carbon: {v['reason']}"
                out["dropped"].append(f)
            else:
                f["refilm_reason"] = f"Not sure from this footage ({v['reason']}). Please film this spot again, closer and steadier."
                out["refilm"].append(f)
    out["carbon_calls"] = carbon_used
    return out


def _household_text(household: dict | None) -> str:
    """One-line summary of household facts for Decisions context."""
    if household is None or household.get("unknown"):
        return "Household facts: unknown."
    ages = household.get("children_ages") or []
    kids = f"children aged {ages}" if ages else ("a child 10 or younger" if household["child_10_or_younger"] else "no children")
    units = "3+ apartments" if household["building_3plus_units"] else "fewer than 3 apartments"
    return f"Household facts (stated by tenant): {kids}; building has {units}."


def classify(finding: dict, household: dict | None, work_dir: Path) -> dict:
    """ONE Decisions call: HPD category (choice) + likely severity (score). Mutates and returns finding."""
    text = (f"NYC apartment condition found in a tenant's video: {RULE_LABELS[finding['rule']]}.\n"
            f"Details: {finding['explanation']}\n{_household_text(household)}\n")
    answers = models.decide(
        [{"type": "text", "text": text},
         models.media_block(work_dir / finding["evidence_frame"], resolution=config.FRAME_RESOLUTION)],
        {
            "category": {"type": "choice",
                         "instructions": "Which NYC HPD complaint category fits this condition?",
                         "criteria": config.HPD_CATEGORIES},
            "severity": {"type": "score",
                         "instructions": "How severe is this condition for the people living here?",
                         "criteria": config.SEVERITY_LEVELS},
        },
        step="classify", purpose=f"HPD category + likely severity for {RULE_LABELS[finding['rule']]}",
        key=lambda a: (f"category={a['category'].choice} ({a['category'].confidence:.2f}), "
                       f"severity={config.SEVERITY_NAMES[int(max(a['severity'].probabilities, key=a['severity'].probabilities.get))]} "
                       f"(score {a['severity'].score:.2f})"),
    )
    cat, sev = answers["category"], answers["severity"]
    level = int(max(sev.probabilities, key=sev.probabilities.get))
    finding.update(category=cat.choice, category_confidence=round(cat.confidence, 3),
                   likely_severity=config.SEVERITY_NAMES[level], severity_score=round(sev.score, 3),
                   severity_confidence=round(sev.confidence, 3))
    return finding
