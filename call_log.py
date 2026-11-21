"""Per-job model-call log: "How this report was made" (UI section + data/work/<job>/model_log.json)."""
from __future__ import annotations

import json
from pathlib import Path

import config

MODEL_NAMES = {config.HYDROGEN: "Hydrogen", config.NEON: "Neon", config.CARBON: "Carbon",
               config.DECISION: "Decisions"}
SLIM_KEYS = ("ts", "step", "purpose", "questions", "ms", "attempts", "ok", "key_output", "error")


def summarize(calls: list[dict]) -> dict:
    """Per-model counts: {"Hydrogen": {"calls": 1, "ms": ...}, ..., "Decisions": {..., "questions": N}}."""
    by = {name: {"calls": 0, "ms": 0} for name in MODEL_NAMES.values()}
    by["Decisions"]["questions"] = 0
    for c in calls:
        name = MODEL_NAMES.get(c.get("model"), c.get("model"))
        b = by.setdefault(name, {"calls": 0, "ms": 0})
        b["calls"] += 1
        b["ms"] += c.get("ms") or 0
        if name == "Decisions":
            b["questions"] += c.get("questions") or 0
    return by


def summary_line(by: dict) -> str:
    """'Hydrogen 1 call · Neon 3 calls · Carbon 1 call · Decisions 34 calls / 612 questions'."""
    def n(k: int, word: str) -> str:
        return f"{k} {word}{'' if k == 1 else 's'}"
    parts = [f"{name} {n(b['calls'], 'call')}" for name, b in by.items() if name != "Decisions"]
    d = by["Decisions"]
    parts.append(f"Decisions {n(d['calls'], 'call')} / {n(d['questions'], 'question')}")
    return " · ".join(parts)


def build(calls: list[dict], work_dir: Path, filename: str = "model_log.json") -> dict:
    """Sort calls by time, write the full log (media already stripped) to work_dir/filename.

    Returns {"summary", "by_model", "calls": slim entries for the UI, "file": filename}.
    """
    calls = sorted(calls, key=lambda c: c.get("ts", ""))
    by = summarize(calls)
    line = summary_line(by)
    (Path(work_dir) / filename).write_text(
        json.dumps({"summary": line, "by_model": by, "calls": calls}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    slim = [{"model": MODEL_NAMES.get(c.get("model"), c.get("model")), "model_id": c.get("model"),
             **{k: c.get(k) for k in SLIM_KEYS}} for c in calls]
    return {"summary": line, "by_model": by, "calls": slim, "file": filename}
