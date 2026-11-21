"""HPD eval (no videos needed): Decisions API vs a Neon baseline on real NYC HPD records.

Two tasks, both on public NYC Open Data:
  1. Likely severity: 200 HPD violation descriptions (wvxf-dwi5, inspected May 2025),
     stratified ~67/67/66 over class A/B/C. Truth = HPD class: A -> non-hazardous, B -> hazardous,
     C -> immediately hazardous (HPD's own class definitions). The violations dataset has no category.
  2. Category: 100 HPD complaint problems (ygpa-z7cr, received Jan-Jun 2025), stratified over
     major_category. Input = problem code + location (NOT minor_category); truth = major_category.

Methods:
  - Decisions API: ONE call per record, the same question the pipeline uses (score over
    config.SEVERITY_LEVELS, or choice over config.HPD_CATEGORIES). Confidence used for calibration.
  - Neon baseline: one Neon call labels a batch of 20 records (JSON out), thinking LOW.
Call budget: ~200 + 100 Decisions + ~15 Neon = ~315.

Usage:  python eval/eval_hpd.py            (fetches + caches data in eval/data/, runs, writes outputs)
Outputs: eval/results.md, eval/charts/category_accuracy.png, eval/charts/calibration.png,
         eval/hpd_predictions.csv
"""
from __future__ import annotations

import contextvars
import csv
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import models  # noqa: E402
from google.genai import types  # noqa: E402

EVAL = Path(__file__).resolve().parent
DATA, CHARTS = EVAL / "data", EVAL / "charts"
CLASS_TO_SEV = {"A": 0, "B": 1, "C": 2}
SEED = 7
BATCH = 20


# ---------------------------------------------------------------- data

def _get(url: str, params: dict) -> list[dict]:
    """GET a Socrata endpoint."""
    r = requests.get(url, params=params, timeout=150)
    r.raise_for_status()
    return r.json()


def clean_description(text: str) -> str:
    """Strip the legal citation prefix ('§ 27-2005 ADM CODE ...') and location suffix noise."""
    t = re.sub(r"^.*?(ADM CODE|M/D LAW|REGULATIONS|REGS\.?)\s*(AND DEPT\.? RULES AND REG\w*\.?)?\s*", "", text or "")
    t = re.sub(r"\s+LOCATED AT .*$", "", t)
    return re.sub(r"\s+", " ", t).strip()[:400] or (text or "")[:400]


def load_violations(n: int = 200) -> list[dict]:
    """Stratified sample of violations (A/B/C), cached in eval/data/violations.json."""
    path = DATA / "violations.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    rows = []
    for cls in "ABC":
        got = _get("https://data.cityofnewyork.us/resource/wvxf-dwi5.json", {
            "$select": "violationid, class, novdescription", "$limit": 1500,
            "$where": f"class='{cls}' AND inspectiondate between '2025-05-01' and '2025-06-01'"})
        seen, uniq = set(), []
        for g in got:                                   # de-duplicate near-identical descriptions
            d = clean_description(g.get("novdescription", ""))
            key = d[:80]
            if d and key not in seen:
                seen.add(key)
                uniq.append({"id": g["violationid"], "class": cls, "text": d})
        rows += rng.sample(uniq, k={"A": 67, "B": 67, "C": 66}[cls])
    rng.shuffle(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows[:n]


def load_problems(n: int = 100) -> list[dict]:
    """Stratified sample of complaint problems over major_category, cached in eval/data/problems.json."""
    path = DATA / "problems.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    cats = list(config.HPD_CATEGORIES)
    per = {c: max(2, round(n / len(cats))) for c in cats}
    rows = []
    for c in cats:
        got = _get("https://data.cityofnewyork.us/resource/ygpa-z7cr.json", {
            "$select": "problem_id, major_category, problem_code, space_type, unit_type", "$limit": 400,
            "$where": f"major_category='{c}' AND received_date between '2025-01-01' and '2025-07-01'"})
        uniq = {}
        for g in got:
            uniq.setdefault((g.get("problem_code"), g.get("space_type")), g)
        pick = rng.sample(list(uniq.values()), k=min(per[c], len(uniq)))
        rows += [{"id": g["problem_id"], "category": c,
                  "text": f"Problem: {g.get('problem_code', '').strip()}. Location: "
                          f"{(g.get('space_type') or '').strip().lower()} ({(g.get('unit_type') or '').strip().lower()})."}
                 for g in pick]
    rng.shuffle(rows)
    rows = rows[:n]
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows


# ---------------------------------------------------------------- methods

def decide_severity(row: dict) -> dict:
    """Decisions: likely severity (score over SEVERITY_LEVELS) for one violation description."""
    a = models.decide(f"NYC HPD housing-code violation: {row['text']}\n",
                      {"severity": {"type": "score",
                                    "instructions": "How severe is this condition for the people living here?",
                                    "criteria": config.SEVERITY_LEVELS}},
                      step="eval_hpd_severity", purpose=f"eval: severity of violation {row['id']}")["severity"]
    return {"pred": int(max(a.probabilities, key=a.probabilities.get)), "conf": a.confidence, "score": a.score}


def decide_category(row: dict) -> dict:
    """Decisions: HPD category (choice) for one complaint problem."""
    a = models.decide(f"NYC tenant housing complaint. {row['text']}\n",
                      {"category": {"type": "choice", "instructions": "Which NYC HPD complaint category fits this problem?",
                                    "criteria": config.HPD_CATEGORIES}},
                      step="eval_hpd_category", purpose=f"eval: category of problem {row['id']}")["category"]
    return {"pred": a.choice, "conf": a.confidence}


def neon_batch(rows: list[dict], task: str) -> dict[str, str | int]:
    """Neon baseline: label a batch of records in ONE call; returns {id: label}."""
    if task == "severity":
        options = "\n".join(f"{i}: {s}" for i, s in enumerate(config.SEVERITY_LEVELS))
        ask = "For each NYC HPD violation below, give the severity level number (0, 1 or 2):\n" + options
    else:
        options = "\n".join(f"- {k}: {v}" for k, v in config.HPD_CATEGORIES.items())
        ask = "For each NYC tenant complaint below, give the HPD category (exact key):\n" + options
    items = "\n".join(f"[{r['id']}] {r['text']}" for r in rows)
    resp = models.neon(f"{ask}\n\nReply as JSON: a list of {{\"id\": ..., \"label\": ...}}, one per item.\n\n{items}",
                       types.GenerateContentConfig(response_mime_type="application/json"),
                       step=f"eval_hpd_neon_{task}", purpose=f"eval baseline: Neon labels {len(rows)} records ({task})")
    out = {}
    for x in models.parse_json(resp):
        lab = x.get("label")
        out[str(x.get("id"))] = int(lab) if task == "severity" and str(lab).strip().isdigit() else lab
    return out


def run_all(rows: list[dict], fn) -> list[dict]:
    """Run a per-row Decisions function with DECISION_WORKERS in parallel; failures -> None."""
    def safe(r):
        try:
            return fn(r)
        except Exception as exc:
            return {"pred": None, "conf": None, "error": type(exc).__name__}
    with ThreadPoolExecutor(max_workers=config.DECISION_WORKERS) as pool:   # copy_context: calls reach the collector
        return [f.result() for f in [pool.submit(contextvars.copy_context().run, safe, r) for r in rows]]


def run_neon(rows: list[dict], task: str) -> dict:
    """Neon baseline over all rows in batches of BATCH (4 batches in parallel)."""
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    def safe(b):
        try:
            return neon_batch(b, task)
        except Exception:
            return {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        merged = {}
        for d in [f.result() for f in [pool.submit(contextvars.copy_context().run, safe, b) for b in batches]]:
            merged.update(d)
    return merged


# ---------------------------------------------------------------- metrics + outputs

def accuracy(preds: list, truth: list) -> tuple[float, int]:
    """(accuracy over rows with a prediction, n answered)."""
    pairs = [(p, t) for p, t in zip(preds, truth) if p is not None]
    return (sum(p == t for p, t in pairs) / len(pairs) if pairs else 0.0), len(pairs)


def calibration(confs: list, correct: list, edges=(0.0, .4, .5, .6, .7, .8, .9, 1.01)) -> list[dict]:
    """Bucket confidences; return [{lo, hi, n, mean_conf, acc}]."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        idx = [i for i, c in enumerate(confs) if c is not None and lo <= c < hi]
        if idx:
            out.append({"lo": lo, "hi": min(hi, 1.0), "n": len(idx),
                        "mean_conf": sum(confs[i] for i in idx) / len(idx),
                        "acc": sum(correct[i] for i in idx) / len(idx)})
    return out


def confusion(preds: list, truth: list, labels: list) -> list[list[int]]:
    """Confusion matrix rows=truth, cols=pred."""
    m = [[0] * len(labels) for _ in labels]
    for p, t in zip(preds, truth):
        if p in labels and t in labels:
            m[labels.index(t)][labels.index(p)] += 1
    return m


def charts(res: dict) -> None:
    """Write category_accuracy.png and calibration.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CHARTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    tasks = [f"Category\n({res['category']['n']} complaints, 14 classes)",
             f"Likely severity\n({res['severity']['n']} violations, vs class A/B/C)"]
    dec = [res["category"]["decisions_acc"], res["severity"]["decisions_acc"]]
    neo = [res["category"]["neon_acc"], res["severity"]["neon_acc"]]
    x = range(len(tasks))
    b1 = ax.bar([i - 0.18 for i in x], dec, 0.36, label="Decisions API (1 call / record)", color="#2453c7")
    b2 = ax.bar([i + 0.18 for i in x], neo, 0.36, label="Neon baseline (20 records / call)", color="#9aa6bf")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.0%}", ha="center", fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks); ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy")
    ax.axhline(1 / 3, ls=":", color="#999"); ax.text(1.45, 1 / 3 + 0.01, "chance (severity)", fontsize=8, color="#777", ha="right")
    ax.set_title("HPD records: accuracy vs ground truth"); ax.legend(loc="upper right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(CHARTS / "category_accuracy.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", label="perfect calibration")
    for key, color, label in (("category", "#2453c7", "category"), ("severity", "#c76b24", "likely severity")):
        cal = res[key]["calibration"]
        ax.plot([c["mean_conf"] for c in cal], [c["acc"] for c in cal], marker="o", color=color, label=f"Decisions: {label}")
        for c in cal:
            ax.annotate(f"n={c['n']}", (c["mean_conf"], c["acc"]), textcoords="offset points", xytext=(4, -10), fontsize=7, color=color)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0, 1.02); ax.set_xlabel("Decisions API confidence (bucket mean)")
    ax.set_ylabel("actual accuracy"); ax.set_title("Do the confidence scores mean anything?")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(CHARTS / "calibration.png", dpi=150); plt.close(fig)


def write_results(res: dict, n_calls: str) -> None:
    """Write eval/results.md."""
    sev, cat = res["severity"], res["category"]
    def cal_rows(cal):
        return "\n".join(f"| {c['lo']:.1f}–{c['hi']:.1f} | {c['n']} | {c['mean_conf']:.2f} | {c['acc']:.0%} |" for c in cal)
    sev_labels = config.SEVERITY_NAMES
    cm = sev["confusion"]
    md = f"""# Eval results: HPD records (no video)

Run on {date.today().isoformat()}. Data: NYC Open Data, public. Calls: {n_calls}.

| Task | n | Truth | Decisions API | Neon baseline |
|---|---|---|---|---|
| HPD category | {cat['n']} | `major_category` of real HPD complaint problems (ygpa-z7cr) | **{cat['decisions_acc']:.0%}** | {cat['neon_acc']:.0%} |
| Likely severity | {sev['n']} | HPD class of real violations (wvxf-dwi5): A→non-hazardous, B→hazardous, C→immediately hazardous | **{sev['decisions_acc']:.0%}** | {sev['neon_acc']:.0%} |

Chance for severity is 33% (balanced A/B/C sample).

## Method
- **Severity:** {sev['n']} violation descriptions (inspected May 2025), stratified ~67/67/66 over class A/B/C,
  legal-citation prefix stripped. Decisions: one call per record, the same score question the pipeline uses
  (3 ordered levels, `config.SEVERITY_LEVELS`); predicted level = most probable level.
- **Category:** {cat['n']} complaint problems (received Jan–Jun 2025), stratified over the 14 real
  categories. Input = problem code + location only (not `minor_category`). Decisions: one choice call per record
  over the 14 categories (`config.HPD_CATEGORIES`, now the real dataset values).
- **Neon baseline:** Neon (thinking LOW) labels 20 records per call from the same option lists (JSON output).
  Batching keeps us inside the call budget; it is a cheaper-per-record setup than Decisions' one call per record.
- Text only: no images here. In the app, classification also sees the evidence frame.

## Severity confusion (Decisions; rows = HPD truth, cols = predicted)
| truth \\ pred | {' | '.join(sev_labels)} |
|---|---|---|---|
""" + "\n".join(f"| {sev_labels[i]} (class {'ABC'[i]}) | " + " | ".join(str(v) for v in cm[i]) + " |" for i in range(3)) + f"""

## Calibration (Decisions confidence vs actual accuracy)
Category:

| confidence | n | mean conf | accuracy |
|---|---|---|---|
{cal_rows(cat['calibration'])}

Likely severity:

| confidence | n | mean conf | accuracy |
|---|---|---|---|
{cal_rows(sev['calibration'])}

Charts: `charts/category_accuracy.png`, `charts/calibration.png`. Per-record predictions: `hpd_predictions.csv`.

## Caveats
- HPD class is assigned by inspectors under legal definitions; our "likely severity" is an estimate from text,
  and we never present it as the official class.
- Category truth is the category the complaint was filed under; some problem codes fit more than one category.
- Single run, small samples ({cat['n']} / {sev['n']}); differences of a few points are within noise.
"""
    (EVAL / "results.md").write_text(md, encoding="utf-8")


def main() -> None:
    """Fetch data, run both methods on both tasks, write outputs."""
    sys.stdout.reconfigure(encoding="utf-8")
    viol, probs = load_violations(), load_problems()
    print(f"data: {len(viol)} violations, {len(probs)} complaint problems")
    with models.collect_calls() as calls:
        d_sev = run_all(viol, decide_severity)
        d_cat = run_all(probs, decide_category)
        n_sev = run_neon(viol, "severity")
        n_cat = run_neon(probs, "category")
    import call_log
    summary = call_log.summary_line(call_log.summarize(calls))
    print("calls:", summary)

    res = {}
    t_sev = [CLASS_TO_SEV[r["class"]] for r in viol]
    p_sev, p_nsev = [d["pred"] for d in d_sev], [n_sev.get(str(r["id"])) for r in viol]
    acc, n = accuracy(p_sev, t_sev)
    res["severity"] = {"n": len(viol), "decisions_acc": acc, "decisions_answered": n,
                       "neon_acc": accuracy(p_nsev, t_sev)[0],
                       "calibration": calibration([d["conf"] for d in d_sev], [p == t for p, t in zip(p_sev, t_sev)]),
                       "confusion": confusion(p_sev, t_sev, [0, 1, 2])}
    t_cat = [r["category"] for r in probs]
    p_cat, p_ncat = [d["pred"] for d in d_cat], [n_cat.get(str(r["id"])) for r in probs]
    acc, n = accuracy(p_cat, t_cat)
    res["category"] = {"n": len(probs), "decisions_acc": acc, "decisions_answered": n,
                       "neon_acc": accuracy(p_ncat, t_cat)[0],
                       "calibration": calibration([d["conf"] for d in d_cat], [p == t for p, t in zip(p_cat, t_cat)])}

    with (EVAL / "hpd_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "id", "input", "truth", "decisions_pred", "decisions_conf", "neon_pred"])
        for r, d, nv in zip(viol, d_sev, p_nsev):
            w.writerow(["severity", r["id"], r["text"], CLASS_TO_SEV[r["class"]], d["pred"],
                        None if d["conf"] is None else round(d["conf"], 3), nv])
        for r, d, nv in zip(probs, d_cat, p_ncat):
            w.writerow(["category", r["id"], r["text"], r["category"], d["pred"],
                        None if d["conf"] is None else round(d["conf"], 3), nv])
    (EVAL / "hpd_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    charts(res)
    write_results(res, summary)
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "decisions_acc", "neon_acc")} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
