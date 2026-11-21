"""Steps 10-11: NEON writes plain-language card text, the rights explanation (tenant's language) and the
complaint claims (English, buckets 1 and 3 only); DECISIONS checks every claim against its evidence
(frame + measured values + the tenant's words). Unsupported claims are removed; a problem with no
surviving claim moves to "Please confirm or film again".

Plain style everywhere: about an 8th-grade reading level, short sentences, everyday words, no rule numbers,
"likely" (never an official HPD class), no health claims.
"""
from __future__ import annotations

import json
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

import config
import models
import transcribe

LANG_NAMES = {"en": "English", "es": "Spanish", "hi": "Hindi", "zh": "Chinese", "ru": "Russian",
              "bn": "Bengali", "fr": "French", "ht": "Haitian Creole", "ko": "Korean", "ar": "Arabic",
              "ur": "Urdu", "pl": "Polish", "yi": "Yiddish", "it": "Italian", "pt": "Portuguese"}
DISCLAIMER = "This is not legal advice. HPD decides on inspection. You review and file this yourself through 311."
COMPLAINT_BUCKETS = ("reported_violation", "found_violation")
CONTRADICTS_WHY = "Because the video shows it working, this likely isn't a violation right now."
CONTRADICTS_TODO = "If it only fails sometimes, film it when it happens and run a new walkthrough."


class Card(BaseModel):
    """Four short lines for one problem card (English)."""
    id: str
    what: str
    where: str
    why: str
    todo: str


class Claim(BaseModel):
    """One factual sentence of the complaint, tied to a card id."""
    finding_id: str
    text: str


class Writing(BaseModel):
    """Neon's writing output."""
    cards: list[Card]
    rights_explanation: str
    complaint_intro: str
    claims: list[Claim]


PROMPT = """You help a New York City tenant understand a report about their apartment video.
Write for someone with no legal knowledge, at about an 8th-grade reading level: short sentences,
everyday words, no jargon, no law or rule numbers, no violation classes. Say "likely", never "definitely".
Never give legal advice, never promise outcomes, never make health claims (for mold say "mold", nothing about health).
Use ONLY the facts in the JSON below. Invent nothing.

Buckets:
- reported_violation: the tenant mentioned it, and it is likely something the owner must fix.
- reported_not_violation: the tenant mentioned it, but it is likely not something the owner must fix.
- found_violation: seen on the video, the tenant did not mention it, likely something the owner must fix.
- confirm: unclear, or needs to be filmed again.

1. cards: one per item in "items", same id, in English, each line ONE short sentence:
   - what: the problem in plain words, e.g. "Your window has no safety guard."
     If video_check is "contradicts": "You said <problem>, but in your video it <works normally>.", e.g.
     "You said the toilet doesn't flush, but in your video it flushes normally." (use video_reason).
   - where: just the place, e.g. "Bedroom window" or "Kitchen ceiling" (use "room" if given; no time).
   - why: why it is likely (or likely not) the owner's job. Base it ONLY on "plain_rule" and "facts"; never add any
     rule that is not in plain_rule. Mention the tenant's own words if relevant (e.g. "You said your son is 6.").
     For reported_not_violation, explain kindly,
     e.g. "A loud dishwasher is annoying, but it isn't listed as something the landlord must fix. You can still ask them."
     For confirm items whose reason says we're not sure, say: "We're not sure this is covered. You can still report it to 311."
   - todo: one or two next steps, e.g. "Ask your landlord to install guards. If they don't, report it to 311."
     For likely violations you may add: "If your landlord says it's fixed but it isn't, you can challenge it when
     HPD mails you a notice."
2. rights_explanation: return an empty string (it is written later from the final buckets).
3. complaint_intro, in English, plain: 1-2 sentences from the tenant asking HPD to inspect. Use "[ADDRESS, APT]".
4. claims, in English, plain, FIRST PERSON as the tenant writing to HPD ("My bedroom window has no guard.",
   "My son is 6.", "The video shows water stains on my ceiling."), never "the tenant": ONLY for items in buckets
   reported_violation and found_violation: 1-3 short factual sentences each, one fact per claim, tagged with the
   item id (what is visible, what I said, measured values).

JSON:
{data}
"""


def _item(c: dict) -> dict:
    """Card facts Neon needs."""
    keys = ("id", "bucket", "label", "room", "quote", "facts", "plain_rule", "reason", "likely_severity",
            "visible", "t_start", "video_check", "video_reason")
    return {k: c[k] for k in keys if c.get(k) not in (None, "")}


def write(buckets: dict, household: dict | None, context: dict) -> dict:
    """NEON call: card text + rights explanation + complaint intro + claims."""
    lang = (household or {}).get("language") or "en"
    items = [_item(c) for k in ("reported_violation", "reported_not_violation", "found_violation", "confirm")
             for c in buckets[k]]
    data = {"household": household, "context": context, "items": items}
    resp = models.neon(
        PROMPT.format(lang=lang, lang_name=LANG_NAMES.get(lang, lang), data=json.dumps(data, ensure_ascii=False, indent=1)),
        types.GenerateContentConfig(response_mime_type="application/json", response_schema=Writing),
        step="write", purpose=f"write plain-language cards, rights explanation ({LANG_NAMES.get(lang, lang)}) and complaint",
        key=lambda r: (lambda d: f"{len(d.get('cards', []))} cards, {len(d.get('claims', []))} claims")(models.parse_json(r)),
    )
    return models.parse_json(resp)


RIGHTS_PROMPT = """Write a short explanation for a New York City tenant, in {lang_name} (language code "{lang}"),
at about an 8th-grade reading level: short sentences, everyday words, no jargon, no law numbers, say "likely".
No legal advice, no health claims. Use ONLY the final lists below; do not add or drop any item; invent nothing.

- likely_landlords_job: say these are likely the landlord's job to fix (name each one).
- likely_not_covered: say these are likely not covered by housing rules, kindly (name each one). Skip if empty.
- not_sure ({n_confirm} item(s)): {confirm_rule}
Then: how to report: call 311 or use portal.311.nyc.gov; HPD inspects and decides.
End with: "This is not legal advice." Under 170 words.

{data}
"""


def _names(items: list[dict]) -> list[str]:
    """Plain item names for the rights explanation."""
    return [((c.get("text") or {}).get("what") or c.get("label") or "").strip() for c in items]


def write_rights(buckets: dict, household: dict | None) -> str:
    """NEON: rights explanation in the tenant's language, generated ONLY from the FINAL buckets."""
    lang = (household or {}).get("language") or "en"
    n = len(buckets.get("confirm", []))
    confirm_rule = (f'say exactly (translated if needed): "We weren\'t sure about {n} thing(s): please film again or '
                    f'confirm." and name them. Never say nothing needs to be filmed again.' if n else
                    "there are none: do not mention filming again.")
    data = {"likely_landlords_job": _names(buckets["reported_violation"] + buckets["found_violation"]),
            "likely_not_covered": _names(buckets["reported_not_violation"]), "not_sure": _names(buckets["confirm"])}
    resp = models.neon(RIGHTS_PROMPT.format(lang=lang, lang_name=LANG_NAMES.get(lang, lang), n_confirm=n,
                                            confirm_rule=confirm_rule, data=json.dumps(data, ensure_ascii=False, indent=1)),
                       step="rights", purpose=f"rights explanation ({LANG_NAMES.get(lang, lang)}) from the final buckets",
                       key=lambda r: f"{len(r.text or '')} chars")
    text = (resp.text or "").strip()
    if n and lang == "en" and "weren't sure" not in text.lower():   # guarantee the sentence in English
        text += f"\n\nWe weren't sure about {n} thing(s): please film again or confirm."
    return text


def verify(claims: list[dict], items: list[dict], context: dict, transcript: dict | None,
           work_dir: Path) -> list[dict]:
    """ONE Decisions call per item: a yes/no per claim, "supported by the evidence?".

    Evidence = the item's frame (if any) + measured values + the tenant's own words.
    Returns claims with p_supported and kept. Claims for unknown items are dropped.
    """
    by_id = {c["id"]: c for c in items}
    narration = transcribe.format_for_prompt(transcript or {}) or "(none)"
    results = []
    for fid in dict.fromkeys(c["finding_id"] for c in claims):
        group = [c for c in claims if c["finding_id"] == fid]
        f = by_id.get(fid)
        if f is None:
            results += [{**c, "p_supported": 0.0, "kept": False} for c in group]
            continue
        text = (f"Evidence about this problem: {f.get('label')}. "
                f"{'Frame from the video at %.0f s.' % f['t_start'] if f.get('evidence_frame') and f.get('t_start') is not None else 'No video frame.'}\n"
                f"Facts from the system: {f.get('facts') or 'none'}; date {context.get('date')}, time {context.get('local_time')}, "
                f"outdoor {context.get('outdoor_f')}°F.\nTenant's words (any language):\n{narration}\n")
        inp = [{"type": "text", "text": text}]
        if f.get("evidence_frame"):
            inp.append(models.media_block(work_dir / f["evidence_frame"], resolution=config.EVIDENCE_RESOLUTION))
        questions = {
            f"c{i}": {"type": "noul",
                      "instructions": f"Is this claim supported by the evidence (the image, the system facts, "
                                      f"or the tenant's words)? Claim: \"{c['text']}\"",
                      "criteria": {"true_description": "The evidence shows or states this.",
                                   "false_description": "Not shown, contradicted, or goes beyond the evidence."}}
            for i, c in enumerate(group)}
        answers = models.decide(
            inp, questions, step="verify",
            purpose=f"check {len(group)} complaint claim(s) about {f.get('label')} against the evidence",
            key=lambda a: (f"{sum(x.noul >= config.VERIFY_MIN_P for x in a.values())}/{len(a)} supported "
                           f"(min P={min(x.noul for x in a.values()):.2f})"))
        for i, c in enumerate(group):
            p = round(answers[f"c{i}"].noul, 3)
            results.append({**c, "p_supported": p, "kept": p >= config.VERIFY_MIN_P})
    return results


def assemble_complaint(intro: str, verified: list[dict], buckets: dict, household: dict | None,
                       context: dict, address: str | None = None) -> str:
    """Plain-language English complaint: buckets 1 and 3 only, verified claims only."""
    addr = (address or "").strip() or "[ADDRESS, APT]"
    boroughs = ("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island")
    in_addr = next((b for b in boroughs if b.lower() in addr.lower()), None)
    borough = (household or {}).get("borough") or in_addr or "[BOROUGH]"
    where = addr if in_addr else f"{addr}, {borough}"
    kept = [c for c in verified if c["kept"]]
    lines = ["To: NYC Housing Preservation and Development (HPD), through 311",
             f"About: repairs needed at {where}", "", intro.strip().replace("[ADDRESS, APT]", addr), ""]
    sections = (("reported_violation", "Problems I reported that likely need repair:"),
                ("found_violation", "Other problems seen in my video that likely need repair:"))
    any_items = False
    for key, title in sections:
        rows = []
        for c in buckets.get(key, []):
            txt = " ".join(x["text"].strip() for x in kept if x["finding_id"] == c["id"])
            if not txt:   # no claim written for it: still list it, from its card / label (every bucket 1+3 item appears)
                label = str(c.get("label") or "problem").strip().rstrip(".")
                txt = f"{label[:1].upper()}{label[1:]}."
                if c.get("quote") and c["kind"] == "reported":
                    txt += f' I said: "{c["quote"]}"'
            when = f" (video {int(c['t_start']) // 60}:{int(c['t_start']) % 60:02d})" if c.get("t_start") is not None else ""
            rows.append(f"- {txt}{when}")
        if rows:
            any_items = True
            lines += [title, *rows, ""]
    if not any_items:
        lines += ["No problems that likely need repair were confirmed in this video.", ""]
    lines += [f"I recorded this on {context.get('date')} at {context.get('local_time')}. "
              "I have the video with timestamps. Please inspect.", "", DISCLAIMER]
    return chr(10).join(lines)
