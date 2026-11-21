"""Verdicts + the 3 report buckets.

Buckets (each sorted most serious first):
  1 reported_violation      "You reported it, and it's likely a violation"
  2 reported_not_violation  "You reported it, but it's likely not a violation"
  3 found_violation         "We found it on your video, and it's likely a violation"
  + confirm                 "Please confirm or film again" (unclear / film again)
  + other_checks            things we checked on video that you didn't mention and that aren't violations

Rule-based findings keep their rules.py verdict. Tenant-reported problems get ONE Decisions call each:
verdict (choice: likely_violation / likely_not_violation / unclear) + best-matching HPD reference row
(choice) + likely severity (score), using the official-source table in hpd_reference.json.
"Did the tenant mention it?" is ONE Decisions call per clip covering every visual finding.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
import models
import transcribe
from rules import RULE_LABELS

REFERENCE = json.loads((config.ROOT / "hpd_reference.json").read_text(encoding="utf-8"))["rows"]
REF_BY_ID = {r["id"]: r for r in REFERENCE}
SEV_RANK = {"immediately hazardous": 0, "hazardous": 1, "non-hazardous": 2}
UNCLEAR_TEXT = "We're not sure this is covered. You can still report it to 311."


def reference_text() -> str:
    """The reference table as compact text for Decisions input."""
    return "\n".join(f"[{r['id']}] {r['condition']} | owner must fix: {r['owner_responsible']} | {r['plain_rule']}"
                     for r in REFERENCE)


def household_text(h: dict | None) -> str:
    """One line of household facts for Decisions input."""
    if not h or h.get("unknown"):
        return "Household facts: unknown."
    ages = h.get("children_ages") or []
    kids = f"children aged {ages}" if ages else ("a child 10 or younger" if h.get("child_10_or_younger") else "no children mentioned")
    units = "3 or more apartments" if h.get("building_3plus_units") else "fewer than 3 apartments"
    return f"Household (as the tenant said): {kids}; building has {units}."


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
    """ONE Decisions call: for each visual finding, did the tenant mention it, and which report matches?"""
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
        if len(reported) >= 1:
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
    """ONE Decisions call: likely violation? + matching reference row + likely severity."""
    inp = [{"type": "text", "text":
            f"A NYC tenant reported this problem: {item['text']}\n"
            f"Their words: \"{item.get('quote') or item['text']}\"\n"
            f"Visible on their video: {'yes' if item.get('visible') else 'no or not shown'}.\n"
            f"{household_text(household)}\n\nHPD reference (official NYC sources):\n{reference_text()}\n"}]
    if item.get("evidence_frame"):
        inp.append(models.media_block(work_dir / item["evidence_frame"], resolution=config.EVIDENCE_RESOLUTION))
    a = models.decide(inp, {
        "verdict": {"type": "choice",
                    "instructions": "Based on this HPD reference, is this likely a housing code violation the owner must fix?",
                    "criteria": {
                        "likely_violation": "Some part of the problem matches a reference row where the owner must fix it "
                                            "(owner must fix: yes, or depends and the conditions are met). Example: an "
                                            "appliance that leaks water is still a water leak, which the owner must fix.",
                        "likely_not_violation": "The reference says it is the tenant's job or not required, or it is only a "
                                                "comfort or convenience annoyance (noise, slowness, looks) that no reference "
                                                "row says the owner must fix.",
                        "unclear": "A real repair problem (something broken or not working) that the reference marks "
                                   "unclear or does not cover, or it can't be judged from what was said."}},
        "reference": {"type": "choice", "instructions": "Which reference row is closest to this problem?",
                      "criteria": {**{r["id"]: r["condition"] for r in REFERENCE}, "none": "No row fits."}},
        "severity": {"type": "score", "instructions": "How severe is this problem for the people living here?",
                     "criteria": config.SEVERITY_LEVELS},
    }, step="verdict", purpose=f"is the reported problem \"{item['text']}\" likely a violation?",
        key=lambda a: f"{a['verdict'].choice} ({a['verdict'].confidence:.2f}), ref={a['reference'].choice}, "
                      f"severity={a['severity'].score:.2f}")
    v, ref, sev = a["verdict"], a["reference"], a["severity"]
    level = int(max(sev.probabilities, key=sev.probabilities.get))
    row = REF_BY_ID.get(ref.choice)
    if v.choice == "likely_violation" and (not row or row["owner_responsible"] not in ("yes", "depends")):
        # cite the most likely row the owner must fix (e.g. a leaking appliance -> "leaks"), not an "unclear" row
        owner_rows = [k for k in ref.probabilities if k in REF_BY_ID and REF_BY_ID[k]["owner_responsible"] in ("yes", "depends")]
        if owner_rows:
            row = REF_BY_ID[max(owner_rows, key=ref.probabilities.get)]
    return {"verdict": v.choice, "verdict_confidence": round(v.confidence, 3),
            "reference_id": row["id"] if row else None, "plain_rule": row["plain_rule"] if row else None,
            "source_url": row["source_url"] if row else None,
            "likely_severity": config.SEVERITY_NAMES[level], "severity_score": round(sev.score, 3)}


# ---------------------------------------------------------------- buckets

def _card(kind: str, bucket: str, base: dict, frame_checks: list[dict], **extra) -> dict:
    """Common card fields."""
    ev = base.get("evidence_frames") or ([base["evidence_frame"]] if base.get("evidence_frame") else [])
    return {"id": base["id"], "kind": kind, "bucket": bucket, "rule": base.get("rule"),
            "label": RULE_LABELS.get(base.get("rule"), base.get("text")), "t_start": base.get("t_start"),
            "t_end": base.get("t_end"), "evidence_frames": ev, "evidence_frame": ev[0] if ev else None,
            "room": room_of(frame_checks, ev[0]) if ev else None, "source_url": base.get("source_url"),
            "likely_severity": base.get("likely_severity"), "confidence": base.get("confidence"),
            "decided_by": base.get("decided_by", "decisions"), "quote": base.get("tenant_quote") or base.get("quote"),
            "facts": base.get("explanation"), **extra}


def build(routed: dict, reported_items: list[dict], frame_checks: list[dict], transcript: dict | None,
          household: dict | None, work_dir: Path) -> dict:
    """Sort everything into buckets. Returns {"buckets": {...}, "mentions": {...}}."""
    reported = list(reported_items)
    visual = routed["findings"] + routed["not_violations"]
    mentions = mention_checks(visual, reported, transcript, household)
    b = {"reported_violation": [], "reported_not_violation": [], "found_violation": [], "confirm": [], "other_checks": []}

    matched = set()
    for f in visual:
        m = mentions.get(f["id"], {"mentioned": False, "p": None, "report_id": None})
        rep = next((r for r in reported if r["id"] == m["report_id"]), None)
        if rep:
            matched.add(rep["id"])
        quote = f.get("tenant_quote") or (rep or {}).get("quote")
        if f in routed["findings"]:
            bucket = "reported_violation" if (m["mentioned"] or rep) else "found_violation"
        else:
            bucket = "reported_not_violation" if (m["mentioned"] or rep) else "other_checks"
        b[bucket].append(_card("rule", bucket, {**f, "tenant_quote": quote}, frame_checks,
                               mentioned=m["mentioned"] or bool(rep), p_mentioned=m["p"],
                               is_violation=f["is_violation"]))
    for f in routed["refilm"]:
        b["confirm"].append(_card("rule", "confirm", f, frame_checks, reason=f.get("refilm_reason") or f.get("explanation")))

    for r in reported:
        if r["id"] in matched:
            continue
        try:
            v = reported_verdict(r, household, work_dir)
        except Exception as exc:
            v = {"verdict": "unclear", "verdict_confidence": None, "error": type(exc).__name__}
        base = {**r, "likely_severity": v.get("likely_severity"), "confidence": v.get("verdict_confidence"),
                "source_url": v.get("source_url"), "decided_by": "decisions",
                "explanation": v.get("plain_rule") or ""}
        bucket = {"likely_violation": "reported_violation", "likely_not_violation": "reported_not_violation"}.get(
            v["verdict"], "confirm")
        b[bucket].append(_card("reported", bucket, base, frame_checks, verdict=v["verdict"],
                               reference_id=v.get("reference_id"), plain_rule=v.get("plain_rule"),
                               visible=r.get("visible"), p_visible=r.get("p_visible"), t_said=r.get("t_said"),
                               reason=UNCLEAR_TEXT if bucket == "confirm" else None))
    for k in b:
        b[k].sort(key=lambda c: (SEV_RANK.get(c.get("likely_severity"), 3), -(c.get("confidence") or 0)))
    return {"buckets": b, "mentions": mentions}
