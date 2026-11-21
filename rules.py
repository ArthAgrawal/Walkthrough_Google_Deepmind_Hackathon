"""Step 7: plain-Python NYC rules (PLAN.md Section 5). No LLM calls here.

Combines FrameChecks + HouseholdFacts + context + thermostat reading into candidate Findings
(Section 7 contract). Consecutive frames with the same condition merge into one finding with a
time range. `is_violation` is True / False (with the reason in `explanation`) / None (can't
decide: e.g. illegible thermostat; step 9 routes these to "film this spot again").
Category, likely severity and routing are filled in later (steps 8-9).

household=None (or household["unknown"], e.g. no audio) means "video-only" mode (spoken facts removed, the Section 10 baseline): rules that
depend on who lives there assume the condition counts, because the video alone can't rule it out.
"""
from __future__ import annotations

from typing import Callable

import config

SOURCES = {
    "HEAT": "https://portal.311.nyc.gov/article/?kanumber=KA-01036",
    "WINDOW_GUARD": "https://www.nyc.gov/site/hpd/news/014-25/save-lives-install-window-guards",
    "HEAT_SEASON_INSPECTION": "https://www.nyc.gov/site/hpd/news/043-22/hpd-start-heat-season-new-york-city",
}
SOURCE_BY_RULE = {
    "WINDOW_GUARD": SOURCES["WINDOW_GUARD"],
    "HEAT_DAY": SOURCES["HEAT"],
    "HEAT_NIGHT": SOURCES["HEAT"],
    "SMOKE_CO": SOURCES["HEAT_SEASON_INSPECTION"],
    "PEELING_PAINT": SOURCES["HEAT_SEASON_INSPECTION"],
    "LEAK": "https://www.nyc.gov/site/hpd/news/030-25/stay-informed-stay-connected-nyc-city-agencies-target-hidden-asthma-risks-inside-home",
    "MOLD": "https://www.nyc.gov/site/hpd/services-and-information/indoor-allergen-hazards-mold-and-pests.page",
}

# HPD order numbers for each rule (from HPD's data dictionary, via data/hpd_reference.json). The rules above stay the
# source of truth for WHEN something is flagged; these only say which HPD orders cover the condition.
RULE_ORDERS = {
    "WINDOW_GUARD": ["790", "791"],
    "HEAT_DAY": ["664", "666", "964", "966"],
    "HEAT_NIGHT": ["664", "666", "964", "966"],
    "SMOKE_CO": ["701", "702", "1001", "1002", "1502", "1503"],
    "LEAK": ["583", "883", "578", "579", "507"],
    "MOLD": ["550"],
    "PEELING_PAINT": ["616", "617", "626", "508", "808", "557", "857"],
}

# Plain-English names, used by Carbon prompts, the complaint and the UI.
RULE_LABELS = {
    "WINDOW_GUARD": "window without a window guard",
    "HEAT_DAY": "insufficient daytime heat",
    "HEAT_NIGHT": "insufficient nighttime heat",
    "SMOKE_CO": "smoke / carbon monoxide detector not seen",
    "LEAK": "water leak or water damage",
    "MOLD": "mold condition",
    "PEELING_PAINT": "peeling paint",
}

# Per-frame evidence score for each visual rule, from Decisions probabilities.
FRAME_SCORES: dict[str, Callable[[dict], float]] = {
    "WINDOW_GUARD": lambda p: p.get("window_visible", 0) * (1 - p.get("window_guard_installed", 0)),
    "LEAK": lambda p: max(p.get("active_drip", 0), p.get("water_stain", 0) * (1 - p.get("rust_stain", 0))),
    "MOLD": lambda p: p.get("dark_growth_on_surface", 0),
    "PEELING_PAINT": lambda p: p.get("peeling_paint", 0),
}


# ---------------------------------------------------------------- helpers

def _known(household: dict | None) -> bool:
    """False in video-only mode (None) or when nothing was said (household["unknown"])."""
    return household is not None and not household.get("unknown")


def merge_segments(frame_checks: list[dict], score_fn: Callable[[dict], float]) -> list[dict]:
    """Group usable frames with score >= CONF_LOW into time segments (gap <= MERGE_GAP_S).

    Returns [{"t_start","t_end","confidence","evidence_frame"}], confidence = max score in segment.
    """
    segs, cur = [], None
    for fc in frame_checks:
        if not fc["usable"]:
            continue
        s = score_fn(fc["p"])
        if s < config.CONF_LOW:
            continue
        if cur and fc["t"] - cur["t_end"] <= config.MERGE_GAP_S:
            cur["t_end"] = fc["t"]
            if s > cur["confidence"]:
                cur.update(confidence=s, evidence_frame=fc["frame_path"])
        else:
            cur = {"t_start": fc["t"], "t_end": fc["t"], "confidence": s, "evidence_frame": fc["frame_path"]}
            segs.append(cur)
    return segs


def nearby_quote(transcript: dict | None, t0: float, t1: float, pad: float = 2.0) -> str | None:
    """Return narration spoken within [t0-pad, t1+pad], if any."""
    if not transcript:
        return None
    parts = [s["text"].strip() for s in transcript.get("segments", [])
             if s["end"] >= t0 - pad and s["start"] <= t1 + pad]
    return " ".join(parts) or None


def _finding(rule: str, seg: dict, is_violation: bool | None, explanation: str,
             quote: str | None, decided_by: str = "decisions", **extra) -> dict:
    """Build a Finding dict (Section 7) with category/severity left for step 8."""
    return {"id": None, "rule": rule, "t_start": seg["t_start"], "t_end": seg["t_end"],
            "evidence_frame": seg["evidence_frame"], "confidence": round(seg["confidence"], 3),
            "category": None, "likely_severity": None, "decided_by": decided_by, "escalated": False,
            "is_violation": is_violation, "explanation": explanation, "tenant_quote": quote,
            "source_url": SOURCE_BY_RULE[rule], "hpd_orders": RULE_ORDERS[rule], **extra}


# ---------------------------------------------------------------- rules

def window_guard(frame_checks, household, transcript) -> list[dict]:
    """WINDOW_GUARD: window with no guard; violation only if child <=10 AND building has 3+ units."""
    out = []
    for seg in merge_segments(frame_checks, FRAME_SCORES["WINDOW_GUARD"]):
        if not _known(household):
            v, why, q = True, "Window visible with no guard. Household facts unknown (video only).", None
        else:
            child, big = household["child_10_or_younger"], household["building_3plus_units"]
            q = household.get("child_quote")
            if child and big:
                v, why = True, "Window visible with no guard; tenant said a child 10 or younger lives here, in a building with 3+ apartments."
            elif not child:
                v, why = False, "Window has no guard, but no child 10 or younger was reported, so guards are not required."
            else:
                v, why = False, "Window has no guard, but the building has fewer than 3 apartments, so the guard rule does not apply."
        out.append(_finding("WINDOW_GUARD", seg, v, why, q or nearby_quote(transcript, seg["t_start"], seg["t_end"])))
    return out


def simple_condition(rule: str, frame_checks, household, transcript) -> list[dict]:
    """LEAK / MOLD / PEELING_PAINT: the visible condition itself is the issue.

    MOLD also requires a building with 3+ apartments (Local Law 55); video-only mode assumes it.
    """
    out = []
    for seg in merge_segments(frame_checks, FRAME_SCORES[rule]):
        extra, v = {}, True
        if rule == "LEAK":
            why = "Water stain or active leak visible."
        elif rule == "MOLD":
            why = "Dark growth observed on a surface (condition observed; no health claim)."
            if _known(household) and not household["building_3plus_units"]:
                v = False
                why += " The building has fewer than 3 apartments, so Local Law 55 mold rules do not apply."
        else:
            why = "Peeling paint visible."
            if _known(household) and household["child_under_6"]:
                why += " A child under 6 lives here: possible lead-paint concern."
                extra["lead_concern"] = True
        q = nearby_quote(transcript, seg["t_start"], seg["t_end"])
        out.append(_finding(rule, seg, v, why, q, **extra))
    return out


def smoke_co(frame_checks, transcript) -> list[dict]:
    """SMOKE_CO: ceiling clearly filmed in >= MIN_CEILING_FRAMES frames but no detector seen in ANY frame.

    Never a violation: not seeing a detector in one clip proves nothing, so is_violation=None
    and the tenant is asked to confirm (refilm/confirm list).
    """
    usable = [f for f in frame_checks if f["usable"]]
    ceil = [f for f in usable if f["p"].get("ceiling_visible", 0) >= config.CONF_HIGH]
    if len(ceil) < config.MIN_CEILING_FRAMES:
        return []
    smoke_max = max(f["p"].get("smoke_detector_visible", 0) for f in usable)
    co_max = max(f["p"].get("co_detector_visible", 0) for f in usable)
    missing = [name for name, m in (("smoke", smoke_max), ("CO", co_max)) if m < config.CONF_LOW]
    if not missing:
        return []
    worst_seen = max(smoke_max if "smoke" in missing else 0, co_max if "CO" in missing else 0)
    best = max(ceil, key=lambda f: f["p"]["ceiling_visible"])
    mean_ceil = sum(f["p"]["ceiling_visible"] for f in ceil) / len(ceil)
    seg = {"t_start": ceil[0]["t"], "t_end": ceil[-1]["t"], "evidence_frame": best["frame_path"],
           "confidence": mean_ceil * (1 - worst_seen)}
    why = "No smoke/CO detector seen in this clip. It may be in another room; please confirm."
    return [_finding("SMOKE_CO", seg, None, why, nearby_quote(transcript, seg["t_start"], seg["t_end"]),
                     missing=missing)]


def heat(thermo: dict | None, context: dict) -> list[dict]:
    """HEAT_DAY / HEAT_NIGHT from the thermostat reading, local time and outdoor temperature."""
    if thermo is None:
        return []
    hour = int(context["local_time"].split(":")[0])
    night = hour >= 22 or hour < 6
    rule = "HEAT_NIGHT" if night else "HEAT_DAY"
    seg = {"t_start": thermo["t"], "t_end": thermo["t"], "evidence_frame": thermo["frame_path"],
           "confidence": thermo["p_visible"]}
    indoor, outdoor = thermo["indoor_f"], context.get("outdoor_f")
    extra = {"indoor_f": indoor, "outdoor_f": outdoor, "local_time": context["local_time"]}

    if indoor is None:
        v, why = None, "Thermostat visible but the reading is not legible. Film it closer."
    elif not context["heat_season"]:
        v, why = False, f"Thermostat reads {indoor:.0f}°F, but heat rules apply only Oct 1 to May 31."
    elif night:
        v = indoor < 62
        why = (f"Thermostat reads {indoor:.0f}°F at {context['local_time']}; "
               f"between 10 PM and 6 AM it must be at least 62°F.")
    elif outdoor is None:
        v, why = None, f"Thermostat reads {indoor:.0f}°F, but the outdoor temperature is unavailable."
    elif outdoor >= 55:
        v, why = False, (f"Thermostat reads {indoor:.0f}°F; outdoors is {outdoor:.0f}°F (55°F or above), "
                         "so the daytime heat rule does not apply.")
    else:
        v = indoor < 68
        why = (f"Thermostat reads {indoor:.0f}°F at {context['local_time']} with {outdoor:.0f}°F outdoors; "
               "between 6 AM and 10 PM it must be at least 68°F when outdoors is below 55°F.")
    return [_finding(rule, seg, v, why, None, decided_by="neon+rules", **extra)]


def evaluate(frame_checks: list[dict], household: dict | None, context: dict,
             thermo: dict | None, transcript: dict | None = None) -> list[dict]:
    """Run all rules; return candidate Findings (violations and non-violations) sorted by time, ids f1..fN."""
    findings = []
    findings += window_guard(frame_checks, household, transcript)
    for rule in ("LEAK", "MOLD", "PEELING_PAINT"):
        findings += simple_condition(rule, frame_checks, household, transcript)
    findings += smoke_co(frame_checks, transcript)
    findings += heat(thermo, context)
    findings.sort(key=lambda f: (f["t_start"], f["rule"]))
    for i, f in enumerate(findings, 1):
        f["id"] = f"f{i}"
    return findings
