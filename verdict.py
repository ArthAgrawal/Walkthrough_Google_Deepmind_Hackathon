"""Verdicts + the 3 report buckets, using data/hpd_reference.json (built from HPD's own data dictionary).

Buckets (each sorted most serious first):
  1 reported_violation      "You reported it, and it's likely a violation"
  2 reported_not_violation  "You reported it, but it's likely not a violation"
  3 found_violation         "We found it on your video, and it's likely a violation"
  + confirm                 "Please confirm or film again" (unclear / film again)
  + other_checks            things we checked on video that you didn't mention and that aren't violations

- rules.py findings keep their rules.py verdict; they are linked to their reference entry (plain_rule, HPD orders).
- Each tenant-reported problem: ONE Decisions call: closest reference entry (choice over condition ids +
  not_covered ids + "none") + likely severity (score) + "did they say they own the appliance?" (yes/no).
  The verdict then follows the matched entry: a condition the owner must fix (with its conditions, e.g. 3+ units,
  checked against the household facts) -> likely violation; a not_covered entry -> its verdict; no match -> unclear.
- "Did the tenant mention it?" is ONE Decisions call per clip covering every video finding.
Never invent rules: card "Why" lines come from the entry's plain_rule only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import config
import models
import transcribe
from rules import RULE_LABELS

REF = json.loads((config.DATA_DIR / "hpd_reference.json").read_text(encoding="utf-8"))
CONDITIONS = {c["id"]: c for c in REF["conditions"]}
NOT_COVERED = {c["id"]: c for c in REF["not_covered_or_unclear"]}
DATASET_URL = REF.get("dataset_url", "https://data.cityofnewyork.us/d/wvxf-dwi5")
CIV14 = REF.get("tenant_rights", {}).get("CIV 14", "")
SEV_RANK = {"immediately hazardous": 0, "hazardous": 1, "non-hazardous": 2}
UNCLEAR_TEXT = "We're not sure this is covered. You can still report it to 311."
RULE_TO_ENTRY = {"WINDOW_GUARD": "window_guard", "HEAT_DAY": "heat", "HEAT_NIGHT": "heat",
                 "SMOKE_CO": "smoke_detector", "LEAK": "water_leak", "MOLD": "mold"}


# ---------------------------------------------------------------- reference helpers

def entry_for_rule(rule: str, household: dict | None) -> dict | None:
    """Reference entry for a rules.py rule (peeling paint -> lead entry if a child under 6 lives there)."""
    if rule == "PEELING_PAINT":
        return CONDITIONS["peeling_paint_lead" if (household or {}).get("child_under_6") else "plaster_paint"]
    return CONDITIONS.get(RULE_TO_ENTRY.get(rule, ""))


def source_line(entry: dict | None) -> tuple[str | None, str | None]:
    """('HPD order 790 · §27-2043.1', url) for a condition entry; (None, None) otherwise."""
    if not entry or not entry.get("hpd_orders"):
        return None, None
    o = entry["hpd_orders"][0]
    m = re.search(r"§\s*([\d][\d\-\.,]*)", o.get("law_text", ""))
    line = f"HPD order {o['order']}" + (f" · §{m.group(1).rstrip(',.')}" if m else "")
    urls = [s for s in entry.get("sources", []) if s.startswith("http")]
    return line, (urls[0] if urls else DATASET_URL)


def _entry_option(e: dict) -> str:
    """Short description of a reference entry for the Decisions choice."""
    return f"{e['title']}. {e['plain_rule']}"[:300]


def household_text(h: dict | None) -> str:
    """One line of household facts for Decisions input."""
    if not h or h.get("unknown"):
        return "Household facts: unknown."
    ages = h.get("children_ages") or []
    kids = f"children aged {ages}" if ages else ("a child 10 or younger" if h.get("child_10_or_younger") else "no children mentioned")
    units = "3 or more apartments" if h.get("building_3plus_units") else "fewer than 3 apartments"
    return f"Household (as the tenant said): {kids}; building has {units}."


def conditions_met(entry: dict, household: dict | None) -> tuple[bool, str | None]:
    """Check the entry's household conditions (child ages, 3+ units). Unknown household -> assume met."""
    if not household or household.get("unknown"):
        return True, None
    conds = entry.get("conditions") or []
    if "building_3plus_units" in conds and household.get("building_3plus_units") is False:
        return False, "This rule only applies to buildings with 3 or more apartments."
    if "child_10_or_younger" in conds and household.get("child_10_or_younger") is False:
        return False, "This rule only applies if a child 10 or younger lives there."
    return True, None


def room_of(frame_checks: list[dict], frame_path: str | None) -> str | None:
    """Room type for an evidence frame from the frame checks (bathroom/kitchen/bedroom), if clear."""
    fc = next((f for f in frame_checks if f["frame_path"] == frame_path), None)
    if not fc:
        return None
    best = max(("bathroom", "kitchen", "bedroom"), key=lambda k: fc["p"].get(k, 0))
    return best if fc["p"].get(best, 0) >= 0.5 else None


# ---------------------------------------------------------------- C: did the tenant mention it?

def mention_checks(visual: list[dict], reported: list[dict], transcript: dict | None,
                   household: dict | None) -> dict[str, dict]:
    """ONE Decisions call: for each video finding, did the tenant mention it, and which report matches?"""
    if not visual:
        return {}
    if not household or household.get("unknown") or not transcript or not transcript.get("segments"):
        return {f["id"]: {"mentioned": False, "p": None, "report_id": None} for f in visual}
    said = transcribe.format_for_prompt(transcript)
    listed = "; ".join(household.get("stated_problems") or []) or "none"
    questions = {}
    for f in visual:
        label = RULE_LABELS[f["rule"]]
        questions[f"m_{f['id']}"] = {
            "type": "noul",
            "instructions": f"Did the tenant mention this problem (in any words or language): {label}?",
            "criteria": {"true_description": "They talked about this problem or this kind of problem.",
                         "false_description": "They never mentioned it; it was only seen on the video."}}
        if reported:
            opts = {r["id"]: r["text"][:120] for r in reported}
            opts["none"] = "None of these."
            questions[f"r_{f['id']}"] = {"type": "choice",
                                         "instructions": f"Which of the tenant's reported problems is the same as: {label}?",
                                         "criteria": opts}
    a = models.decide(f"What the tenant said (any language):\n{said}\nProblems they listed: {listed}\n", questions,
                      step="mention", purpose=f"did the tenant mention each of {len(visual)} video finding(s)?",
                      key=lambda a: ", ".join(f"{k[2:]}: P(mentioned)={v.noul:.2f}" for k, v in a.items() if k.startswith("m_")))
    out = {}
    for f in visual:
        m = a[f"m_{f['id']}"].noul
        rid = a[f"r_{f['id']}"].choice if f"r_{f['id']}" in a else "none"
        out[f["id"]] = {"mentioned": m >= 0.5, "p": round(m, 3), "report_id": None if rid == "none" else rid}
    return out


# ---------------------------------------------------------------- B: verdict for reported problems

def reported_verdict(item: dict, household: dict | None, work_dir: Path) -> dict:
    """ONE Decisions call: closest reference entry + likely severity + tenant-owned appliance?; then the verdict."""
    inp = [{"type": "text", "text":
            f"A NYC tenant reported this problem in their apartment: {item['text']}\n"
            f"Their words: \"{item.get('quote') or item['text']}\"\n"
            f"Visible on their video: {'yes' if item.get('visible') else 'no or not shown'}.\n{household_text(household)}\n"}]
    if item.get("evidence_frame"):
        inp.append(models.media_block(work_dir / item["evidence_frame"], resolution=config.EVIDENCE_RESOLUTION))
    options = {**{cid: _entry_option(c) for cid, c in CONDITIONS.items()},
               **{nid: _entry_option(n) for nid, n in NOT_COVERED.items()},
               "none": "None of these fits the problem."}
    a = models.decide(inp, {
        "entry": {"type": "choice",
                  "instructions": "Which HPD reference entry is the closest match for this problem? "
                                  "A leak from an appliance is a water leak. Noise or slowness alone is 'noisy or slow appliance'.",
                  "criteria": options},
        "severity": {"type": "score", "instructions": "How severe is this problem for the people living here?",
                     "criteria": config.SEVERITY_LEVELS},
        "tenant_owns": {"type": "noul", "instructions": "Did the tenant say they bought or own this appliance themselves?",
                        "criteria": {"true_description": "They clearly said it is theirs (they bought it).",
                                     "false_description": "They didn't say, or it came with the apartment."}},
    }, step="verdict", purpose=f"match the reported problem \"{item['text']}\" to the HPD reference",
        key=lambda a: f"entry={a['entry'].choice} ({a['entry'].confidence:.2f}), severity={a['severity'].score:.2f}")
    ent, sev = a["entry"], a["severity"]
    level = int(max(sev.probabilities, key=sev.probabilities.get))
    out = {"entry_id": ent.choice, "entry_confidence": round(ent.confidence, 3),
           "likely_severity": config.SEVERITY_NAMES[level], "severity_score": round(sev.score, 3)}
    if ent.choice in CONDITIONS:
        e = CONDITIONS[ent.choice]
        met, why_not = conditions_met(e, household)
        line, url = source_line(e)
        verdict = "likely_violation" if (e["owner_responsible"] == "yes" and met) else "likely_not_violation"
        return {**out, "verdict": verdict, "plain_rule": e["plain_rule"], "not_met": why_not,
                "source_line": line, "source_url": url, "category": e["hpd_orders"][0]["category"] if e["hpd_orders"] else None}
    if ent.choice in NOT_COVERED:
        n = NOT_COVERED[ent.choice]
        verdict = n["verdict"]
        if ent.choice == "appliance_not_owner_supplied" and a["tenant_owns"].noul < 0.5:
            verdict = "unclear"                       # per the entry's note: unclear unless the tenant says who owns it
        return {**out, "verdict": verdict, "plain_rule": n["plain_rule"], "source_line": "HPD order list: no matching order",
                "source_url": DATASET_URL}
    return {**out, "verdict": "unclear", "plain_rule": UNCLEAR_TEXT, "source_line": None, "source_url": None}


# ---------------------------------------------------------------- buckets

def _card(kind: str, bucket: str, base: dict, frame_checks: list[dict], **extra) -> dict:
    """Common card fields."""
    ev = base.get("evidence_frames") or ([base["evidence_frame"]] if base.get("evidence_frame") else [])
    return {"id": base["id"], "kind": kind, "bucket": bucket, "rule": base.get("rule"),
            "label": RULE_LABELS.get(base.get("rule"), base.get("text")), "t_start": base.get("t_start"),
            "t_end": base.get("t_end"), "evidence_frames": ev, "evidence_frame": ev[0] if ev else None,
            "room": room_of(frame_checks, ev[0]) if ev else None, "source_url": base.get("source_url"),
            "source_line": base.get("source_line"), "likely_severity": base.get("likely_severity"),
            "confidence": base.get("confidence"), "decided_by": base.get("decided_by", "decisions"),
            "quote": base.get("tenant_quote") or base.get("quote"), "facts": base.get("explanation"),
            "plain_rule": base.get("plain_rule"), **extra}


def build(routed: dict, reported_items: list[dict], frame_checks: list[dict], transcript: dict | None,
          household: dict | None, work_dir: Path) -> dict:
    """Sort everything into buckets. Returns {"buckets": {...}, "mentions": {...}}."""
    reported = list(reported_items)
    visual = routed["findings"] + routed["not_violations"]
    mentions = mention_checks(visual, reported, transcript, household)
    b = {"reported_violation": [], "reported_not_violation": [], "found_violation": [], "confirm": [], "other_checks": []}

    def rule_ref(f: dict) -> dict:
        """plain_rule + HPD order source for a rules.py finding."""
        e = entry_for_rule(f["rule"], household)
        line, url = source_line(e)
        return {"plain_rule": e["plain_rule"] if e else None, "source_line": line,
                "source_url": url or f.get("source_url"), "entry_id": e["id"] if e else None}

    matched = set()
    for f in visual:
        m = mentions.get(f["id"], {"mentioned": False, "p": None, "report_id": None})
        rep = next((r for r in reported if r["id"] == m["report_id"]), None)
        if rep:
            matched.add(rep["id"])
        said = m["mentioned"] or bool(rep)
        if f in routed["findings"]:
            bucket = "reported_violation" if said else "found_violation"
        else:
            bucket = "reported_not_violation" if said else "other_checks"
        # Quote only what's relevant: the child quote for window guards, or the tenant's words when they mentioned it
        # (rules.py attaches nearby narration, which can be about something else).
        quote = (household or {}).get("child_quote") if f["rule"] == "WINDOW_GUARD" else None
        if said:
            quote = (rep or {}).get("quote") or f.get("tenant_quote") or quote
        b[bucket].append(_card("rule", bucket, {**f, **rule_ref(f), "tenant_quote": quote, "quote": None},
                               frame_checks, mentioned=said, p_mentioned=m["p"], is_violation=f["is_violation"]))
    for f in routed["refilm"]:
        b["confirm"].append(_card("rule", "confirm", {**f, **rule_ref(f), "tenant_quote": None, "quote": None},
                                  frame_checks, reason=f.get("refilm_reason") or f.get("explanation")))

    for r in reported:
        if r["id"] in matched:
            continue
        try:
            v = reported_verdict(r, household, work_dir)
        except Exception as exc:
            v = {"verdict": "unclear", "plain_rule": UNCLEAR_TEXT, "error": type(exc).__name__}
        base = {**r, "likely_severity": v.get("likely_severity"), "confidence": v.get("entry_confidence"),
                "source_url": v.get("source_url"), "source_line": v.get("source_line"), "decided_by": "decisions",
                "plain_rule": v.get("plain_rule"), "explanation": v.get("not_met") or v.get("plain_rule")}
        bucket = {"likely_violation": "reported_violation", "likely_not_violation": "reported_not_violation"}.get(
            v["verdict"], "confirm")
        b[bucket].append(_card("reported", bucket, base, frame_checks, verdict=v["verdict"],
                               entry_id=v.get("entry_id"), category=v.get("category"), visible=r.get("visible"),
                               p_visible=r.get("p_visible"), t_said=r.get("t_said"),
                               reason=(v.get("plain_rule") if v.get("entry_id") in NOT_COVERED else UNCLEAR_TEXT)
                               if bucket == "confirm" else None))
    for k in b:
        b[k].sort(key=lambda c: (SEV_RANK.get(c.get("likely_severity"), 3), -(c.get("confidence") or 0)))
    return {"buckets": b, "mentions": mentions}
