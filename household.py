"""Step 3: NEON turns the transcript into HouseholdFacts (PLAN.md Section 7).

Solo mode: the Decisions cross-check is cut. Cautious defaults instead: if something is not
clearly stated, pick the answer that keeps a possible violation visible and set
needs_confirmation=true so the tenant confirms it in the UI.
"""
from __future__ import annotations

from typing import Optional

from google.genai import types
from pydantic import BaseModel

import models


class HouseholdFacts(BaseModel):
    """Facts the tenant states about who lives in the apartment and the building."""
    language: str
    child_10_or_younger: bool
    child_under_6: bool
    children_ages: list[int]
    building_3plus_units: bool
    borough: Optional[str]
    stated_problems: list[str]
    needs_confirmation: bool
    child_quote: Optional[str]      # exact words (original language) supporting the child facts
    units_quote: Optional[str]      # exact words supporting the building-size fact


PROMPT = """You extract facts from what a New York City tenant said while filming their apartment.
The transcript may be in any language. Answer from what was SAID only; never guess from names.

Rules:
- children_ages: ages of children who live in or regularly spend time in the apartment.
  Babies/infants count as age 0.
- child_10_or_younger: true if any such child is 10 or younger.
- child_under_6: true if any such child is under 6.
- building_3plus_units: true if the building has 3 or more apartments. If the tenant does not say,
  set it to true AND set needs_confirmation to true (most NYC rentals are in 3+ unit buildings).
- If children are mentioned but an age is unclear, set child_10_or_younger and child_under_6 to true
  and needs_confirmation to true. If the tenant says no children live there, both are false.
- borough: one of Manhattan, Brooklyn, Queens, Bronx, Staten Island, or null if not said.
- stated_problems: short English phrases for problems the tenant mentions (e.g. "no heat at night").
- language: ISO 639-1 code of the language the tenant spoke: {language}
- child_quote / units_quote: copy the exact words (original language) that support those facts, or null.
- needs_confirmation: also true if anything above is ambiguous.

Transcript:
{transcript}
"""


def extract(transcript_text: str, language: str = "en") -> dict:
    """Return a HouseholdFacts dict from transcript text (or an intro_override string)."""
    resp = models.neon(
        PROMPT.format(language=language, transcript=transcript_text or "(no speech)"),
        types.GenerateContentConfig(response_mime_type="application/json", response_schema=HouseholdFacts),
        step="household",
    )
    facts = models.parse_json(resp)
    # Enforce consistency in code rather than trusting the model.
    ages = facts.get("children_ages") or []
    if ages:
        facts["child_10_or_younger"] = facts["child_10_or_younger"] or min(ages) <= 10
        facts["child_under_6"] = facts["child_under_6"] or min(ages) < 6
    if facts["child_under_6"]:
        facts["child_10_or_younger"] = True
    return facts
