"""Steps 10-11: NEON writes the rights explanation (tenant's language) and complaint claims (English);
DECISIONS checks every claim against its evidence frame + measured context + tenant's words, and
unsupported claims are removed before the complaint is assembled.

The complaint is assembled in code from verified claims, so nothing unverified reaches the draft.
The "not a violation, and why" section is filled from rules.py explanations (deterministic).
"""
from __future__ import annotations

import json
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

import config
import models
import transcribe
from rules import RULE_LABELS

LANG_NAMES = {"en": "English", "es": "Spanish", "hi": "Hindi", "zh": "Chinese", "ru": "Russian",
              "bn": "Bengali", "fr": "French", "ht": "Haitian Creole", "ko": "Korean", "ar": "Arabic",
              "ur": "Urdu", "pl": "Polish", "yi": "Yiddish", "it": "Italian", "pt": "Portuguese"}


class Claim(BaseModel):
    """One factual sentence of the complaint, tied to a finding."""
    finding_id: str
    text: str


class Writing(BaseModel):
    """Neon's writing output."""
    rights_explanation: str
    complaint_intro: str
    claims: list[Claim]


PROMPT = """You help a New York City tenant understand what their apartment video shows.
You are NOT a lawyer: never give legal advice, never promise outcomes, never make health claims
(for mold say only "condition observed"). Use only the facts in the JSON below; invent nothing.

Write:
1. rights_explanation, in {lang_name} (language code "{lang}"), plain and short (under 220 words):
   - for each item in "findings": what was seen, at what time in the video, and which NYC rule it relates to;
   - for each item in "not_violations": what was seen and why it is NOT a violation;
   - for each item in "refilm": what to film again or confirm;
   - how to file: call 311 or use portal.311.nyc.gov; HPD inspects and decides. This is not legal advice.
2. complaint_intro, in English: 1-2 sentences from the tenant to NYC HPD requesting an inspection.
   Use "[ADDRESS, APT]" as a placeholder for the address. No claims about specific conditions here.
3. claims, in English: for each item in "findings", 1-3 short factual sentences, each a separate claim
   with that finding's id. State what is visible (and, if relevant, what the tenant said, e.g. a child's
   age, or the measured time/outdoor temperature). One fact per claim. No legal conclusions.

JSON:
{data}
"""


def _slim(f: dict) -> dict:
    """Finding fields Neon needs (no frame paths or internals)."""
    keys = ("id", "rule", "t_start", "t_end", "explanation", "tenant_quote", "likely_severity",
            "category", "indoor_f", "outdoor_f", "local_time", "refilm_reason", "lead_concern")
    d = {k: f[k] for k in keys if f.get(k) is not None}
    d["condition"] = RULE_LABELS[f["rule"]]
    return d


def write(routed: dict, household: dict | None, context: dict) -> dict:
    """NEON call: returns Writing dict (rights_explanation, complaint_intro, claims)."""
    lang = (household or {}).get("language") or "en"
    data = {"household": household, "context": context,
            "findings": [_slim(f) for f in routed["findings"]],
            "not_violations": [_slim(f) for f in routed["not_violations"]],
            "refilm": [_slim(f) for f in routed["refilm"]]}
    resp = models.neon(
        PROMPT.format(lang=lang, lang_name=LANG_NAMES.get(lang, lang),
                      data=json.dumps(data, ensure_ascii=False, indent=1)),
        types.GenerateContentConfig(response_mime_type="application/json", response_schema=Writing),
        step="write",
    )
    return models.parse_json(resp)


def verify(claims: list[dict], findings: list[dict], context: dict, transcript: dict | None,
           work_dir: Path) -> list[dict]:
    """ONE Decisions call per finding: a yes/no per claim, "supported by the evidence?".

    Evidence = the finding's frame + measured context + the tenant's own words.
    Returns claims with p_supported and kept fields. Claims with unknown finding ids are dropped.
    """
    by_id = {f["id"]: f for f in findings}
    narration = transcribe.format_for_prompt(transcript or {}) or "(none)"
    results = []
    for fid in dict.fromkeys(c["finding_id"] for c in claims):
        group = [c for c in claims if c["finding_id"] == fid]
        f = by_id.get(fid)
        if f is None:
            results += [{**c, "p_supported": 0.0, "kept": False} for c in group]
            continue
        measured = {k: f[k] for k in ("indoor_f", "outdoor_f", "local_time") if f.get(k) is not None}
        text = (f"Evidence frame from the tenant's video at {f['t_start']:.0f}s.\n"
                f"Measured by the system: {json.dumps(measured) if measured else 'nothing'}; date {context.get('date')}.\n"
                f"Tenant's words (any language):\n{narration}\n")
        questions = {
            f"c{i}": {"type": "noul",
                      "instructions": f"Is this claim supported by the evidence (the image, the measured values, "
                                      f"or the tenant's words)? Claim: \"{c['text']}\"",
                      "criteria": {"true_description": "The evidence shows or states this.",
                                   "false_description": "Not shown, contradicted, or goes beyond the evidence."}}
            for i, c in enumerate(group)}
        answers = models.decide(
            [{"type": "text", "text": text},
             models.media_block(work_dir / f["evidence_frame"], resolution=config.FRAME_RESOLUTION)],
            questions, step="verify")
        for i, c in enumerate(group):
            p = round(answers[f"c{i}"].noul, 3)
            results.append({**c, "p_supported": p, "kept": p >= config.VERIFY_MIN_P})
    return results


def assemble_complaint(intro: str, verified: list[dict], routed: dict, household: dict | None,
                       context: dict) -> str:
    """Build the English complaint draft from verified claims + not-violation explanations."""
    borough = (household or {}).get("borough") or "[BOROUGH]"
    lines = ["To: NYC Department of Housing Preservation and Development (via 311)",
             f"Re: Housing maintenance complaint, [ADDRESS, APT], {borough}", "", intro.strip(), ""]
    kept = [c for c in verified if c["kept"]]
    if routed["findings"]:
        lines.append(f"Conditions recorded on video ({context.get('date')} at {context.get('local_time')}):")
        for f in routed["findings"]:
            fc = [c["text"].strip() for c in kept if c["finding_id"] == f["id"]]
            if not fc:
                continue
            sev = f" (likely severity: {f['likely_severity']})" if f.get("likely_severity") else ""
            lines.append(f"- {RULE_LABELS[f['rule']].capitalize()}{sev}, video {f['t_start']:.0f}-{f['t_end']:.0f}s: "
                         + " ".join(fc))
    else:
        lines.append("No violations were identified in this video.")
    if routed["not_violations"]:
        lines += ["", "Items checked that are not violations, and why:"]
        lines += [f"- {RULE_LABELS[f['rule']].capitalize()}: {f['explanation']}" for f in routed["not_violations"]]
    lines += ["", "Video evidence with timestamps is available on request. I request an inspection.", "",
              "Drafted with Walkthrough from the tenant's video; reviewed and filed by the tenant. Not legal advice."]
    return "\n".join(lines)
