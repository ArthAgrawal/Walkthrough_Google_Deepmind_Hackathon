# Eval results: HPD records (no video)

Run on 2026-11-21. Data: NYC Open Data, public. Calls: Decisions 288 calls / 288 questions · Neon 15 calls (counted from logs/model_calls.jsonl; 0 failures).

| Task | n | Truth | Decisions API | Neon baseline |
|---|---|---|---|---|
| HPD category | 88 | `major_category` of real HPD complaint problems (ygpa-z7cr) | 67% | 74% |
| Likely severity | 200 | HPD class of real violations (wvxf-dwi5): A→non-hazardous, B→hazardous, C→immediately hazardous | 55% | 62% |

Chance for severity is 33% (balanced A/B/C sample).

**Honest read: on plain text, the Neon baseline beat the Decisions API on both tasks.** What the Decisions API adds is a
confidence score that means something for category:

| Decisions API, selective | records kept | accuracy |
|---|---|---|
| Category, confidence ≥ 0.9 | 41 / 88 (47%) | **98%** |
| Category, all | 88 / 88 | 67% |
| Likely severity, confidence ≥ 0.7 | 24 / 200 | 62% |

- Category: high-confidence answers are almost always right; the misses are inputs that are ambiguous without HPD's
  minor category (e.g. "Problem: BROKEN OR MISSING. Location: lobby"). This is why the app routes on confidence.
- Likely severity is the weak spot: Decisions under-calls class C (36 of 66 class C violations predicted "hazardous"),
  and its severity confidence barely tracks accuracy. Our three severity descriptions are written around our 6
  conditions (window guards, heat, leaks...), not HPD's class C examples (fire damage, sprinkler obstructions, etc.).
  We did NOT retune them on this test set; see FAILURES.md.

## Method
- **Severity:** 200 violation descriptions (inspected May 2025), stratified ~67/67/66 over class A/B/C,
  legal-citation prefix stripped. Decisions: one call per record, the same score question the pipeline uses
  (3 ordered levels, `config.SEVERITY_LEVELS`); predicted level = most probable level.
- **Category:** 88 complaint problems (received Jan–Jun 2025), stratified over the 14 real
  categories. Input = problem code + location only (not `minor_category`). Decisions: one choice call per record
  over the 14 categories (`config.HPD_CATEGORIES`, now the real dataset values).
- **Neon baseline:** Neon (thinking LOW) labels 20 records per call from the same option lists (JSON output).
  Batching keeps us inside the call budget; it is a cheaper-per-record setup than Decisions' one call per record.
- Text only: no images here. In the app, classification also sees the evidence frame.

## Severity confusion (Decisions; rows = HPD truth, cols = predicted)
| truth \ pred | non-hazardous | hazardous | immediately hazardous |
|---|---|---|---|
| non-hazardous (class A) | 38 | 28 | 1 |
| hazardous (class B) | 15 | 46 | 6 |
| immediately hazardous (class C) | 4 | 36 | 26 |

## Calibration (Decisions confidence vs actual accuracy)
Category:

| confidence | n | mean conf | accuracy |
|---|---|---|---|
| 0.0–0.4 | 10 | 0.27 | 20% |
| 0.4–0.5 | 4 | 0.44 | 50% |
| 0.5–0.6 | 6 | 0.57 | 50% |
| 0.6–0.7 | 10 | 0.65 | 50% |
| 0.7–0.8 | 10 | 0.75 | 40% |
| 0.8–0.9 | 7 | 0.82 | 43% |
| 0.9–1.0 | 41 | 0.97 | 98% |

Likely severity:

| confidence | n | mean conf | accuracy |
|---|---|---|---|
| 0.0–0.4 | 3 | 0.38 | 67% |
| 0.4–0.5 | 46 | 0.47 | 46% |
| 0.5–0.6 | 83 | 0.54 | 58% |
| 0.6–0.7 | 44 | 0.65 | 55% |
| 0.7–0.8 | 18 | 0.74 | 50% |
| 0.8–0.9 | 5 | 0.82 | 100% |
| 0.9–1.0 | 1 | 0.92 | 100% |

Charts: `charts/category_accuracy.png`, `charts/calibration.png`. Per-record predictions: `hpd_predictions.csv`.

## Caveats
- HPD class is assigned by inspectors under legal definitions; our "likely severity" is an estimate from text,
  and we never present it as the official class.
- Category truth is the category the complaint was filed under; some problem codes fit more than one category.
- Single run, small samples (88 / 200); differences of a few points are within noise.
