"""Clip eval (READY, NOT RUN YET): full pipeline vs video-only vs one-call Neon baseline on labelled clips.

eval/labels.csv columns (PLAN.md Section 9, plus optional `video`):
  clip_id,source,intro_text,expected_rules,expected_times,is_dev,video
  - video: file in data/raw/ (default: <clip_id>.mp4 / .webm / .mov). Several rows may share one video
    with different intro_text ("my son is 6" vs "no kids live here").
  - intro_text: typed intro used instead of the spoken one (empty = use the audio).
  - expected_rules / expected_times: ';'-separated, aligned, e.g. "WINDOW_GUARD;LEAK" and "5-9;20-25".
    Empty = clean clip (no violations expected).
  - is_dev: 1 = used for tuning; excluded from reported numbers unless --include-dev.

Match rule (Section 10): same rule AND time ranges within ±3 s of each other.
Methods:
  full        pipeline.run(video, intro_override=intro_text)
  video_only  pipeline.run(..., video_only=True) reusing the full run's frame checks (only spoken facts differ)
  one_call    ONE Neon call with the whole clip: "List all NYC HPD housing code violations with timestamps."

Usage:  python eval/eval_clips.py [--include-dev] [--only venue_01,venue_02] [--methods full,video_only,one_call]
Outputs: eval/results_clips.csv (per clip x method), eval/results_clips.md (summary table).
Budget: ~40 Decisions + ~6 other calls per clip for `full`, ~5 for `video_only`, 1 for `one_call`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import models  # noqa: E402
import pipeline  # noqa: E402
from rules import RULE_LABELS  # noqa: E402

EVAL = Path(__file__).resolve().parent
TOL_S = 3.0
RULE_IDS = list(RULE_LABELS)


# ---------------------------------------------------------------- labels

def load_labels(path: Path) -> list[dict]:
    """Read labels.csv into rows with parsed expected [(rule, t0, t1)]."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("clip_id") or r["clip_id"].startswith("#"):
                continue
            rules_ = [x.strip() for x in (r.get("expected_rules") or "").split(";") if x.strip()]
            times = [x.strip() for x in (r.get("expected_times") or "").split(";") if x.strip()]
            exp = []
            for i, rule in enumerate(rules_):
                t0, t1 = (times[i].split("-") + [None])[:2] if i < len(times) else ("0", "9999")
                exp.append((rule, float(t0), float(t1 or t0)))
            rows.append({**r, "expected": exp, "is_dev": r.get("is_dev", "0").strip() == "1"})
    return rows


def video_for(row: dict) -> Path:
    """Resolve the clip's video file in data/raw/."""
    if row.get("video"):
        return config.RAW_DIR / row["video"]
    for ext in (".mp4", ".webm", ".mov"):
        p = config.RAW_DIR / f"{row['clip_id']}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No video for {row['clip_id']} in {config.RAW_DIR}")


# ---------------------------------------------------------------- one-call baseline

class BaselineFinding(BaseModel):
    """One violation as reported by the one-call baseline."""
    rule: Literal["WINDOW_GUARD", "HEAT_DAY", "HEAT_NIGHT", "SMOKE_CO", "LEAK", "MOLD", "PEELING_PAINT"]
    t_start: float
    t_end: float
    explanation: str


BASELINE_PROMPT = """List all NYC HPD housing code violations in this apartment video, with timestamps.
Only use these rule ids: {rules}.
{intro}Return a JSON list of {{"rule", "t_start", "t_end", "explanation"}} (seconds). Empty list if none."""


def one_call(video: Path, intro_text: str | None) -> list[dict]:
    """ONE Neon call on the whole clip (inline if small, else Files API)."""
    data = video.read_bytes()
    mime = "video/webm" if video.suffix.lower() == ".webm" else "video/mp4"
    if len(data) < 18_000_000:
        part = types.Part.from_bytes(data=data, mime_type=mime)
    else:
        f = models.get_client().files.upload(file=str(video))
        while getattr(f.state, "name", str(f.state)) == "PROCESSING":
            time.sleep(2)
            f = models.get_client().files.get(name=f.name)
        part = f
    intro = f"The tenant says: \"{intro_text}\"\n" if intro_text else ""
    resp = models.neon([part, BASELINE_PROMPT.format(rules=", ".join(RULE_IDS), intro=intro)],
                       types.GenerateContentConfig(response_mime_type="application/json",
                                                   response_schema=list[BaselineFinding]),
                       step="eval_one_call", purpose=f"one-call baseline on {video.name}")
    return models.parse_json(resp) or []


# ---------------------------------------------------------------- scoring

def match(pred: list[tuple], exp: list[tuple]) -> tuple[int, int, int]:
    """Greedy match (same rule, intervals within ±TOL_S). Returns (tp, fp, fn)."""
    used, tp = set(), 0
    for rule, a0, a1 in pred:
        for j, (erule, b0, b1) in enumerate(exp):
            if j in used or rule != erule:
                continue
            if a0 <= b1 + TOL_S and b0 <= a1 + TOL_S:
                used.add(j)
                tp += 1
                break
    return tp, len(pred) - tp, len(exp) - tp


def preds_from_report(rep: dict) -> list[tuple]:
    """Violations the pipeline reports (findings list only)."""
    return [(f["rule"], f["t_start"], f["t_end"]) for f in rep.get("findings", [])]


def main() -> None:
    """Run the selected methods on labelled clips and write results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(EVAL / "labels.csv"))
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--methods", default="full,video_only,one_call")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    methods = args.methods.split(",")
    rows = [r for r in load_labels(Path(args.labels)) if args.include_dev or not r["is_dev"]]
    if args.only:
        rows = [r for r in rows if r["clip_id"] in args.only.split(",")]
    print(f"{len(rows)} clips x {methods}")

    out = []
    for r in rows:
        video, intro = video_for(r), (r.get("intro_text") or "").strip() or None
        work = config.WORK_DIR / f"eval_{r['clip_id']}"
        for m in methods:
            t0 = time.perf_counter()
            with models.collect_calls() as calls:
                try:
                    if m == "full":
                        preds = preds_from_report(pipeline.run(video, intro_override=intro, work_dir=work))
                    elif m == "video_only":
                        preds = preds_from_report(pipeline.run(video, video_only=True, work_dir=work,
                                                               reuse_frames=(work / "steps_1_7.json").exists()))
                    else:
                        preds = [(f["rule"], float(f["t_start"]), float(f["t_end"])) for f in one_call(video, intro)]
                    err = ""
                except Exception as exc:
                    preds, err = [], f"{type(exc).__name__}: {str(exc)[:200]}"
            tp, fp, fn = match(preds, r["expected"])
            out.append({"clip_id": r["clip_id"], "source": r.get("source"), "method": m, "clean": not r["expected"],
                        "tp": tp, "fp": fp, "fn": fn, "n_calls": len(calls),
                        "seconds": round(time.perf_counter() - t0, 1), "predicted": json.dumps(preds), "error": err})
            print(f"{r['clip_id']:<14} {m:<10} tp={tp} fp={fp} fn={fn} calls={len(calls)} {out[-1]['seconds']}s {err}")

    with (EVAL / "results_clips.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ["clip_id"])
        w.writeheader()
        w.writerows(out)
    lines = ["# Clip eval", "", "| method | clips | recall | precision | false alarms / clean clip | calls / clip | s / clip |",
             "|---|---|---|---|---|---|---|"]
    for m in methods:
        rs = [o for o in out if o["method"] == m]
        if not rs:
            continue
        tp, fp, fn = (sum(o[k] for o in rs) for k in ("tp", "fp", "fn"))
        clean = [o for o in rs if o["clean"]]
        lines.append(f"| {m} | {len(rs)} | {tp / (tp + fn) if tp + fn else float('nan'):.0%} | "
                     f"{tp / (tp + fp) if tp + fp else float('nan'):.0%} | "
                     f"{(sum(o['fp'] for o in clean) / len(clean)) if clean else float('nan'):.2f} | "
                     f"{sum(o['n_calls'] for o in rs) / len(rs):.0f} | {sum(o['seconds'] for o in rs) / len(rs):.0f} |")
    lines += ["", f"Match: same rule, time within ±{TOL_S:.0f} s. Labels written before running the system."]
    (EVAL / "results_clips.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
