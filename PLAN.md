# Walkthrough: Team Build Plan (Gemini 4 NYC Hackathon)

> **For Claude / any coding agent reading this:** this is the source of truth for the project.
> Read the whole file before writing code. Gemini 4 and the Decisions API launched in Oct–Nov 2026,
> after most models' training data. **Do not guess API shapes or model names.** Anything marked
> `VERIFY` must be checked against the official docs linked in Section 11 before you rely on it.
> Keep each file small (one feature per file), add a docstring to every function, and log every
> model call (model, prompt, config, output, but strip inline media bytes).

Deadline: **submissions due 5:00 PM ET today.** Demo: 3 min live + 2 min Q&A.

---

## 1. What we're building (read this first)

**One line:** a tenant says who lives in their NYC apartment (in any language), films the apartment,
and gets back which problems break NYC housing rules and which don't, ranked by likely severity,
with timestamped evidence and a complaint draft they review and file through 311 themselves.

**Pitch:** "Landlords have lawyers. Tenants have a phone." Walkthrough is a witness in your pocket.

**Problem:** NYC tenants often don't know which problems the law actually covers (window guards
are only required if a child 10 or younger lives there; heat rules depend on time of day and
outdoor temperature), and a 311 complaint without evidence is just the tenant's word. It's heat
season now (Oct 1 – May 31).

**Hackathon track:** Problem Statement Two ("Element 119"). The core input is raw video + audio,
and the result depends on combining both. Same video + different spoken words = different answer.

**What it does NOT do (say this plainly):** it checks 6 common conditions, not all housing law.
It isn't legal advice. It never makes health claims (for mold, say "condition observed" only).
It never files automatically: the tenant reviews and files.

---

## 2. Hackathon rules we must follow

- Repo must be **public**. **Never commit the API key.** If a key is pushed, organizers kill all keys.
  `.env` goes in `.gitignore` BEFORE the first commit.
- New work only: all code written today.
- Must use **≥2 Gemini 4 models, each with a clear job**, plus the **Decisions API**.
- Banned project types we must not look like: image analyzer, dashboard as main feature, medical
  advice, basic RAG, Streamlit apps. So: no Streamlit; lead with the video+voice → rules → complaint
  flow; keep the trace panel small.
- Don't abuse the temporary accounts (thousands of requests can mean disqualification). The whole
  eval must stay in the **hundreds of calls**. Develop on 5–10 items; run the full eval once or twice.
- The temporary accounts are deleted tomorrow. Push to GitHub all day.

**Judging (Round 1):** Live demo on unrehearsed inputs 35% · Proof it works (baselines, failures,
self-checks) 25% · Creativity 20% · Gemini 4 + Decisions API use 20%.
Extra prize we're targeting: **Best use of Managed Agents ($5,000).**

---

## 3. Environment and setup

- **Coding:** each person uses Claude Code locally. **Runtime:** Gemini API key from the
  temporary hackathon account (each teammate uses their own account/key).
- Python 3.11+, `ffmpeg` installed.
- `pip install google-genai fastapi uvicorn python-multipart python-dotenv pandas matplotlib pillow requests reportlab`
- `.env`:
  ```
  GEMINI_API_KEY=...
  ```
- `.gitignore`: `.env`, `__pycache__/`, `data/raw/`, `*.mp4`, `*.webm` (don't push friends' videos).
- Optional but recommended: install Google's Gemini skills so agents know current APIs:
  `npx skills add google-gemini/gemini-skills` (VERIFY the exact command in the Gemini docs).
- Every model call goes through `models.py` with **retry + backoff** (3 tries: 1 s, 2 s, 4 s).
  New models return 503 "overloaded" errors. A 429 usually means the spend cap at
  aistudio.google.com/spend was hit (raise it a little).

---

## 4. Models and their jobs

All model names live in `config.py` and nowhere else.

```python
# config.py
HYDROGEN = "gemini-4-hydrogen"    # fast: frame scan support, transcription
NEON     = "gemini-4-neon"        # understanding + writing: household facts, thermostat reading, complaint text
CARBON   = "gemini-4-carbon"      # hard cases only: uncertain findings, contradictions
DECISION = "gemini-decision-1"    # Decisions API: every judgment call
MANAGED_AGENT = "antigravity-preview-05-2026"   # VERIFY current agent id in Managed Agents docs

CONF_HIGH = 0.70   # include finding
CONF_LOW  = 0.40   # below this, drop; between LOW and HIGH → Carbon re-check or "film again"
FPS = 1            # frames per second sampled from video
```

| Model | Job (one sentence each, for judges) |
|---|---|
| **Hydrogen** | Watches everything fast: transcribes the audio (original language, with timestamps). |
| **Decisions API** | Decides everything: frame quality, what's in each frame, category, severity, routing, final claim check, with calibrated confidence. |
| **Neon** | Understands what the tenant says (household facts), reads the thermostat number, writes the rights explanation (tenant's language) and the complaint (English). |
| **Carbon** | Only steps in when the system is unsure: re-checks the ±5 s around an uncertain finding or a contradiction. |
| **Managed Agent ("caseworker")** | Background, multi-step: looks up the building's HPD violation history on NYC Open Data, compares it with our findings, builds the evidence-packet PDF. |

---

## 5. The rules (encode in `rules.py`, plain Python, no LLM)

Each finding cites its official source URL.

| ID | Rule | Visual fact (Decisions API) | Spoken/context fact |
|---|---|---|---|
| `WINDOW_GUARD` | Buildings with 3+ units must have approved window guards in any apartment where a child 10 or younger lives or spends time. | window visible AND no guard | `child_10_or_younger` AND `building_3plus_units` |
| `HEAT_DAY` | 6 AM–10 PM: indoor ≥ 68°F when outdoor < 55°F (Oct 1–May 31). | thermostat visible; reading from Neon | local time from clock; outdoor temp from Open-Meteo |
| `HEAT_NIGHT` | 10 PM–6 AM: indoor ≥ 62°F regardless of outdoor temp (Oct 1–May 31). | thermostat reading | local time |
| `SMOKE_CO` | Smoke and CO detectors required (inspectors check them). | ceiling visible AND no smoke detector (and/or no CO detector) | room type, if said |
| `LEAK` | Water leak / water damage condition. | water stain OR active drip | duration, if said |
| `MOLD` | Mold condition (NO health claims). | dark growth | location/duration, if said |
| `PEELING_PAINT` | Peeling paint; flag as **possible lead-paint concern** if a child under 6 lives there. | peeling paint | `child_under_6` |

Sources:
- Heat: https://portal.311.nyc.gov/article/?kanumber=KA-01036 and https://www.nyc.gov/site/hpd/services-and-information/heat-and-hot-water-information.page
- Window guards: https://www.nyc.gov/site/hpd/news/014-25/save-lives-install-window-guards and https://www.nyc.gov/site/doh/about/press/pr2023/help-prevent-window-falls-as-weather-gets-warmer.page
- What inspectors check on heat inspections (smoke/CO detectors, lead paint if child under 6, window guards): https://www.nyc.gov/site/hpd/news/043-22/hpd-start-heat-season-new-york-city

Wording rule: we output **"likely severity"**, never the official HPD class (HPD assigns that on inspection).

---

## 6. Pipeline (exact flow)

```
INPUT: one video file (webm/mp4) with audio. The first ~10 s is the spoken intro:
       who lives there (children's ages), how many apartments in the building, borough,
       optional known problems. Narration during the walkthrough is optional.

1. ingest.py      ffmpeg → frames at FPS=1 (JPEG, 768px wide) + audio track (wav/mp3)
2. transcribe     HYDROGEN: transcript with timestamps, in the ORIGINAL language, plus detected language
3. household      NEON: from the transcript → HouseholdFacts JSON (Section 7)
                  DECISIONS (cross-check): on the intro audio, yes/no "child 10 or younger mentioned?",
                  "child under 6 mentioned?", "3+ units mentioned?" → if Neon and Decisions disagree,
                  take the cautious answer and set needs_confirmation=true
4. frame checks   DECISIONS, one call per frame, ALL questions in ONE request (Section 8):
                  quality questions + ~20 visual yes/no questions → FrameCheck JSON
                  skip frames that fail quality checks
5. thermostat     if any frame has P(thermostat visible) ≥ 0.7: NEON reads the number on the best frame
6. context        local time (system clock, America/New_York); outdoor temp (Open-Meteo, NYC lat/lon)
7. rules          rules.py combines frame evidence + household + context → candidate findings
                  (merge consecutive frames with the same condition into one finding with a time range)
8. classify       DECISIONS per finding: choice → HPD category; score → likely severity
9. route          conf ≥ 0.70 → include
                  0.40–0.70 → CARBON re-checks frames ±5 s + transcript; if still unsure → "film this spot again"
                  < 0.40 → drop
                  DECISIONS yes/no "does the narration contradict what's visible?" → if yes → CARBON
10. write         NEON: (a) rights explanation in the tenant's language, (b) complaint draft in English
                  ("not a violation, and why" items included too)
11. verify        DECISIONS: one yes/no per claim in the draft, "Is this claim supported by the
                  evidence frame?" → remove unsupported claims
12. caseworker    (background) MANAGED AGENT: building history lookup + evidence-packet PDF
OUTPUT: Report JSON (Section 7) → frontend
```

**Call budget per 30 s clip:** ~30 Decisions calls (frames) + ~10 (findings/verify) + 1 Hydrogen + 2–3 Neon + 0–2 Carbon.

---

## 7. Data contracts (so the three of us can work in parallel)

```python
# HouseholdFacts
{"language": "es", "child_10_or_younger": true, "child_under_6": false, "children_ages": [6],
 "building_3plus_units": true, "borough": "Brooklyn", "stated_problems": ["no heat at night"],
 "needs_confirmation": false}

# FrameCheck (one per frame)
{"t": 7.0, "frame_path": "frames/0007.jpg", "usable": true,
 "p": {"window_visible": 0.96, "window_guard_installed": 0.04, "smoke_detector_visible": 0.10, ...}}

# Finding
{"id": "f1", "rule": "WINDOW_GUARD", "t_start": 5.0, "t_end": 9.0, "evidence_frame": "frames/0007.jpg",
 "confidence": 0.91, "category": "DOOR/WINDOW", "likely_severity": "immediately hazardous",
 "decided_by": "decisions", "escalated": false, "is_violation": true,
 "explanation": "Window visible with no guard; tenant said a 6-year-old lives here.",
 "tenant_quote": "mi hijo tiene seis años", "source_url": "https://..."}

# Report (what /analyze returns)
{"household": HouseholdFacts, "context": {"local_time": "19:42", "outdoor_f": 41},
 "findings": [Finding], "not_violations": [Finding], "refilm": [Finding],
 "rights_explanation": "... (tenant's language)", "complaint_draft_en": "...",
 "trace": [{"step": "frame_checks", "model": "gemini-decision-1", "calls": 18, "ms": 210}, ...],
 "caseworker_job_id": "..."}
```

---

## 8. Decisions API usage (the core of our "model use" score)

`VERIFY` the exact request format against the docs. What we know from Google's launch post:
endpoint `client.decisions.create(model="gemini-decision-1", input=[...], ...)` in the `v1beta`
API version (`genai.Client(http_options={"api_version": "v1beta"})`); question types are
**yes/no**, **multiple choice (2–255 options)** and **rubric score**; up to **1,000 questions per
request** against one shared multimodal input (text, image, audio, video, PDF; up to 256K tokens);
every answer has a 0–1 confidence; latency ~150–250 ms, nearly flat from 1 to 100 questions.
**Put all of a frame's questions in ONE call.** Never one call per question.

Frame questions (step 4), all yes/no:
- Quality: `too_dark`, `too_blurry`, `camera_blocked`
- Visual: `window_visible`, `window_guard_installed`, `ceiling_visible`, `smoke_detector_visible`,
  `co_detector_visible`, `thermostat_visible`, `radiator_visible`, `water_stain`, `active_drip`,
  `dark_growth_on_surface`, `peeling_paint`, `rust_stain` (to separate rust from water damage),
  `bathroom`, `kitchen`, `bedroom`

Use criteria text where the API allows it (e.g. "window_guard_installed: TRUE if metal bars/guard
fixed across the lower part of the window; FALSE for plain glass, blinds or screens").

HPD category options (step 8): build the list from the **distinct category values in the HPD
dataset** we download (Section 9); don't hardcode guesses.

**For the pitch:** "Every judgment (frame quality, what's in each frame, category, severity,
routing, and the final check against made-up claims) is a calibrated Decisions API call. The text
models only write; the Decisions API decides."

---

## 9. Data (evaluation only, NO training)

We don't train anything. Data is for testing. Tune prompts/thresholds on a small **dev set**
(8 venue clips) and report results on everything else.

| Set | What | Size | Owner |
|---|---|---|---|
| **A: HPD violations** | NYC Open Data "Housing Maintenance Code Violations" https://data.cityofnewyork.us/d/wvxf-dwi5 (API: `https://data.cityofnewyork.us/resource/wvxf-dwi5.json?$limit=2000`). VERIFY columns first (expect a violation description text and a class A/B/C column). Sample ~300 rows across categories/classes | ~300 rows | Person C |
| **B: venue clips** | 10–20 s clips around the venue: windows, thermostat, ceilings, bathroom. Film each scene once; pair with different spoken intros ("my son is 6" vs "no kids live here"; times of day). Include ~5 clean clips | ~20–25 | Person C |
| **C: friends' apartments** | 60 s walkthroughs with spoken intro, sent by friends (consent; blur faces/mail/addresses; never commit them) | 6–7 | Person C |
| **D: Veo hard cases** | Generated in AI Studio: active ceiling drip, mold corner, rust vs. water stain, window without guard. Max 5–10 (expensive). Report separately as synthetic | 5–10 | Person C |
| **E: other languages** | 5 venue clips with the intro spoken in Spanish/Hindi/etc. | 5 | Person C |

**No YouTube videos** (rights rule), unless Creative Commons-licensed with credit.

`eval/labels.csv`:
```
clip_id,source,intro_text,expected_rules,expected_times,is_dev
venue_01,venue,"my son is 6",WINDOW_GUARD,5-9,1
venue_02,venue,"no kids live here",,,0
```
Labels written by someone who did NOT build the pipeline.

---

## 10. Evaluation (our "proof it works" score)

Baselines:
1. **One-call baseline:** one NEON call with the whole clip: "List all NYC HPD housing code violations with timestamps." (Parse into the same Finding format.)
2. **Video-only:** our own pipeline with the household facts removed. **This is our best slide.**

Metrics:
| Metric | Data |
|---|---|
| Violations caught (recall) and false ones (precision); match = same rule, time within ±3 s | B (non-dev), C, D, E |
| False alarms per clean clip | clean clips in B, C |
| HPD category accuracy + likely-severity accuracy vs. class A/B/C | A |
| Calibration chart (confidence buckets vs. actual accuracy) for Decisions API answers | A + B |
| Carbon escalation rate and how much it helped | B, C |
| Time and cost per clip | B |

Outputs: `eval/results.csv`, `eval/charts/*.png`, and a **failure log** (`eval/FAILURES.md`) with at
least 2 entries: what broke → what we changed → what improved (with numbers).

---

## 11. Official docs (check these, don't guess)

- Decisions API launch post: https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/
- Decisions API codelab: https://codelabs.developers.google.com/getting-started-with-gemini-decisions-api
- Managed Agents overview: https://ai.google.dev/gemini-api/docs/agents
- Managed Agents quickstart: https://ai.google.dev/gemini-api/docs/managed-agents-quickstart
- Building custom managed agents (AGENTS.md / SKILL.md): https://ai.google.dev/gemini-api/docs/custom-agents
- Gemini 4 Neon announcement: https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-neon/

---

## 12. Managed Agent "caseworker" (for the Managed Agents prize)

A custom managed agent (Antigravity agent in a Google-hosted Linux sandbox) defined by files in
`caseworker/`:
- `AGENTS.md`: instructions: "You are a tenant caseworker. Given findings JSON and an NYC address,
  (1) query NYC Open Data HPD violations for that building, (2) mark which findings are already open
  violations vs. new, (3) build an evidence-packet PDF. Never give legal or health advice. Never file anything."
- `skills/hpd-lookup/SKILL.md` + script: query the HPD violations API by address/borough.
- `skills/evidence-pdf/SKILL.md` + script: build the PDF (reportlab) from findings + frames.

Rules: run it in the **background** after findings are ready (it's slow and can use 100K–3M
tokens per run). Never inside the eval loop. Only a handful of runs today. VERIFY whether we can
select a Gemini 4 model for it (docs say Managed Agents default to Gemini 3.6 Flash and support model
selection). For the demo, use a real public address (e.g., the venue: 220 W 26th St, Manhattan).

---

## 13. Repo structure

```
walkthrough/
  config.py            # model names, thresholds (Person A)
  models.py            # wrappers + retry for Hydrogen/Neon/Carbon/Decisions (Person A)
  ingest.py            # ffmpeg frames + audio (Person A)
  household.py         # transcript → HouseholdFacts (Person A)
  frame_checks.py      # Decisions per-frame checks (Person A)
  rules.py             # the 6 rules (Person A)
  classify_route.py    # category, severity, routing, Carbon (Person A)
  write_verify.py      # Neon writing + Decisions claim check (Person A)
  pipeline.py          # glues steps 1–11, returns Report (Person A)
  server.py            # FastAPI: POST /analyze, GET /caseworker/{id} (Person B)
  static/index.html    # record/upload, analyzing trace, results, complaint (Person B)
  caseworker/          # managed agent files + launcher (Person B)
  eval/                # eval.py, labels.csv, results, charts, FAILURES.md (Person C)
  DESIGN.md            # keep updated: what each file does
  README.md            # how to run, models used and why, eval results, limitations
  .env  .gitignore
```

To work in parallel before the pipeline is ready, Person B uses a **fake Report JSON** (Section 7)
and Person C writes the eval against the `pipeline.run(video_path, intro_override=None) -> Report` interface.

---

## 14. Team split and timeline

| Person | Owns |
|---|---|
| **A, pipeline** | config, models (+retry), ingest, household, frame checks, rules, classify/route, Carbon, write/verify, `pipeline.run` |
| **B, product + agent** | FastAPI server, the web page (3 screens), small trace panel, Managed Agent caseworker, PDF |
| **C, data + eval + story** | text friends NOW, download HPD data, film/label venue clips, Veo hard cases, eval.py, baselines, charts, failure log, slides, demo rehearsal |

| Time | Milestone |
|---|---|
| 12:35 | Repo + `.gitignore` + `.env`. Everyone: one successful test call to Hydrogen, Neon, Carbon, Decisions with retry. C texts friends, downloads HPD data, checks columns |
| 12:35–1:45 | A: pipeline end-to-end on one venue clip. B: page on fake Report JSON + server stub. C: venue clips filmed + labels; HPD eval running on 20 rows |
| **1:45** | **Integration #1:** real pipeline behind `/analyze`; the window demo works end to end |
| 1:45–2:45 | A: Carbon routing, contradiction check, claim verification, multilingual. B: Managed Agent caseworker + PDF. C: full eval (A, B, C, D, E) + both baselines |
| **2:45** | **Integration #2:** caseworker runs in background from the page; eval numbers in hand |
| 2:45–3:45 | Fix the worst failure, re-run eval once, FAILURES.md, charts, 4 slides (story, how it works, proof, failures) |
| 3:45–4:30 | Rehearse demo twice with a non-team person; warm-up script; backup clip ready |
| 4:30–5:00 | 1-min video, README, check repo is public and has NO key, submit |

**Cut order if behind:** heat log → Veo hard cases → caseworker PDF (keep the building lookup) →
contradiction check → trace panel.
**Never cut:** window-guard check with vs. without the child sentence, frame checks + routing via
Decisions API, HPD category eval, video-only baseline, complaint draft.

---

## 14b. SOLO MODE (overrides Section 14): one person, 12:40 → 4:50 PM

Scope is cut to what one person can finish. Build in this order; don't start a step until the previous one works.

1. Setup + smoke test (4 models, with retry).
2. Pipeline steps 1–7 (ingest, transcribe, household, frame checks, thermostat, context, rules).
3. Steps 8–11 (category, severity, routing + Carbon, Neon writing incl. tenant-language explanation, claim verification) + `server.py` + a simple `index.html` (upload/record → results + complaint).
4. Eval: HPD category accuracy (~200 rows), venue clips: full vs. video-only vs. one-call baseline, calibration chart, FAILURES.md.
5. Managed Agent caseworker: **building-history lookup only** (no PDF unless time remains).
6. README, slides, submission video.

**Cut in solo mode:** heat log, Veo clips, evidence PDF, contradiction check, Decisions household cross-check (step 3 cross-check), video timeline markers (a plain list of findings with evidence frames is enough).
**Labels:** the builder writes `eval/labels.csv` BEFORE running the system on any clip and says so honestly in the README ("labels written before running the system").

## 15. Frontend (3 screens, one page, plain HTML/JS, no framework)

1. **Record:** live camera preview, prompt "Start by saying who lives here and how many apartments are in your building, then show us the problems." Record button (MediaRecorder, video+audio) or upload.
2. **Analyzing:** stage lines updating from `trace`: "Hydrogen: transcript → Decisions API: 360 checks → Neon: household facts → Carbon: re-checking 1 finding."
3. **Results:** household facts (as heard) · video timeline with markers (click → evidence frame) · findings (rule, likely severity, time, confidence, decided by) · "not a violation, and why" · "film this spot again" buttons · rights explanation in the tenant's language · complaint draft (copy / download) with "Review, then file through 311" · caseworker panel ("building history: 3 open violations", PDF link) when ready.

---

## 16. Demo script (3 minutes)

| Time | What happens |
|---|---|
| 0:00 | "Landlords have lawyers. Tenants have a phone." Heat season; a complaint without evidence is your word against theirs. |
| 0:20 | Ask a judge: "Do you speak another language?" They say "my son is 6" in it, then film the room's window for 10 s → missing window guard, evidence frame, explanation in their language, complaint in English. |
| 0:55 | Same window, "no kids live here" → not flagged. "Same video, different words, different answer." |
| 1:20 | Second judge films the thermostat → heat check with reading, time, outdoor temp, rule applied. |
| 1:45 | Complaint draft + caseworker: "this building already has N open violations" + evidence packet. |
| 2:10 | Proof: full vs. video-only vs. one-call baseline; calibration chart; HPD category accuracy. |
| 2:40 | Failure slide: what broke, what we changed, the number that improved. |

Before the slot: warm-up call to every model; laptop + phone on a hotspot; a backup clip ready.

**Q&A answers:**
- Why not one call? → baseline numbers.
- Do the confidence scores mean anything? → calibration chart.
- What if the tenant lies? → "We draft; HPD inspects. Evidence makes the complaint stronger, not automatic."
- Isn't this image analysis? → "Remove the spoken facts and accuracy drops from X to Y."
- Mold and health? → "No health claims. We describe the condition observed."
