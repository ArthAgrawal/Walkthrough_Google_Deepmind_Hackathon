"""Launch the caseworker Managed Agent (Antigravity, remote sandbox) for one address + findings.

The agent gets AGENTS.md + the hpd-lookup skill (SKILL.md + script) as inline environment sources,
runs the lookup in its sandbox, matches our findings against open violations, and returns JSON.
If the managed agent fails (not enabled, quota, timeout, bad output), we fall back to running the
same lookup script directly and say so in the result ("mode": "fallback_direct_api").

Docs: https://aistudio.google.com/docs/custom-agents , /docs/managed-agents-quickstart
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import models
from caseworker.skills_path import HPD_DIR
from rules import RULE_LABELS

log = logging.getLogger("caseworker")
ROOT = Path(__file__).resolve().parent
BOROUGHS = ("MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND")


def _sources() -> list[dict]:
    """Inline environment sources: AGENTS.md + the hpd-lookup skill files."""
    return [
        {"type": "inline", "target": ".agents/AGENTS.md", "content": (ROOT / "AGENTS.md").read_text(encoding="utf-8")},
        {"type": "inline", "target": ".agents/skills/hpd-lookup/SKILL.md",
         "content": (HPD_DIR / "SKILL.md").read_text(encoding="utf-8")},
        {"type": "inline", "target": ".agents/skills/hpd-lookup/hpd_lookup.py",
         "content": (HPD_DIR / "hpd_lookup.py").read_text(encoding="utf-8")},
    ]


def borough_of(address: str, household_borough: str | None) -> str | None:
    """Borough named in the typed address wins; else the one the tenant said."""
    up = address.upper()
    return next((b for b in BOROUGHS if b in up), None) or (household_borough or "").upper() or None


def _slim_findings(report: dict) -> list[dict]:
    """Findings + reported problems in a compact form for the agent prompt."""
    items = [{"finding_id": f["id"], "condition": RULE_LABELS[f["rule"]], "category": f.get("category")}
             for f in report.get("findings", [])]
    items += [{"finding_id": r["id"], "condition": r["text"], "tenant_reported": True}
              for r in report.get("reported", [])]
    return items


def _parse_json(text: str) -> dict:
    """Extract the JSON object from the agent's final text."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON object in agent output")
    return json.loads(m.group(0))


def run_managed_agent(address: str, borough: str, findings: list[dict]) -> dict:
    """Run the managed agent (blocking; minutes). Returns its parsed JSON + interaction metadata."""
    client = models.get_client()
    prompt = (f"Address: {address}\nBorough: {borough}\n"
              f"Walkthrough findings JSON:\n{json.dumps(findings, ensure_ascii=False)}\n\n"
              "Follow AGENTS.md: run the hpd-lookup skill, match findings to open violations, reply with the JSON only.")
    t0 = time.perf_counter()
    record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": "agent", "step": "caseworker",
              "model": config.MANAGED_AGENT, "purpose": f"caseworker: HPD building history for {address}",
              "prompt": prompt}
    try:
        interaction = client.interactions.create(
            agent=config.MANAGED_AGENT, input=prompt, environment={"type": "remote", "sources": _sources()})
        out = _parse_json(interaction.output_text)
        record.update(ok=True, ms=round((time.perf_counter() - t0) * 1000), output=interaction.output_text,
                      key_output=f"open violations: {out.get('open_violations')}", interaction_id=interaction.id)
        models._write_log(record)
        return {**out, "mode": "managed_agent", "interaction_id": interaction.id,
                "ms": record["ms"]}
    except Exception as exc:
        record.update(ok=False, ms=round((time.perf_counter() - t0) * 1000), error=f"{type(exc).__name__}: {str(exc)[:500]}")
        models._write_log(record)
        raise


def run_fallback(address: str, borough: str, findings: list[dict], reason: str) -> dict:
    """Direct Open Data lookup with the same script (no agent); no automatic finding matching."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("hpd_lookup", HPD_DIR / "hpd_lookup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.lookup(address, borough)
    m = r["matched"]
    return {"matched_address": f"{m['housenumber']} {m['streetname']}, {m['boro']}",
            "total_violations": r["total"], "open_violations": r["open_total"],
            "open_by_class": r["open_by_class"], "recent_open": r["recent_open"], "findings": [],
            "summary": f"This building has {r['open_total']} open HPD violation(s) on record.",
            "mode": "fallback_direct_api", "fallback_reason": reason}


def run(address: str, household_borough: str | None, report: dict) -> dict:
    """Caseworker entry point: managed agent first, direct API fallback. Never raises."""
    borough = borough_of(address, household_borough)
    if not borough:
        return {"status": "error", "summary": "Add the borough to the address (e.g. '..., Manhattan') to look up building history."}
    findings = _slim_findings(report)
    try:
        return {"status": "done", **run_managed_agent(address, borough, findings)}
    except Exception as exc:
        reason = f"managed agent unavailable ({type(exc).__name__}: {str(exc)[:160]})"
        log.warning("caseworker fallback: %s", reason)
        try:
            return {"status": "done", **run_fallback(address, borough, findings, reason)}
        except Exception as exc2:
            return {"status": "error", "summary": f"Building lookup failed ({type(exc2).__name__})."}
