"""Central config: model names, thresholds, paths. Model names live here and nowhere else."""
from pathlib import Path

# --- Models (PLAN.md Section 4) ---
# Verified: "gemini-4-neon" (Neon launch post / AI Studio link) and "gemini-decision-1"
# (Decisions API launch post + docs). HYDROGEN / CARBON ids are not published in the docs we
# read; scripts/smoke_test.py confirms them (and lists available gemini-4 models on failure).
HYDROGEN = "gemini-4-hydrogen"    # fast: transcription
NEON     = "gemini-4-neon"        # understanding + writing: household facts, thermostat, complaint
CARBON   = "gemini-4-carbon"      # hard cases only: uncertain findings, contradictions
DECISION = "gemini-decision-1"    # Decisions API: every judgment call
MANAGED_AGENT = "antigravity-preview-09-2026"   # verified: aistudio.google.com/docs/managed-agents-quickstart

API_VERSION = "v1beta"            # Decisions API requires v1beta

# Thinking levels (google-genai ThinkingLevel: MINIMAL, LOW, MEDIUM, HIGH, XHIGH).
# LOW by default for speed (Neon at its default took ~13 s for "pong"); Carbon re-checks think harder.
# Measured on one re-check (3-6 frames): HIGH 265 s, MEDIUM 17 s, LOW 4.5 s, all the same verdict.
THINKING_DEFAULT = "LOW"
THINKING_CARBON = "MEDIUM"
CALL_TIMEOUT_S = 90               # per text-model request; timeouts are not retried
CARBON_TIMEOUT_S = 45             # Carbon re-check; on timeout the finding goes to "film again"

# Decisions API image resolution for per-frame checks
FRAME_RESOLUTION = "low"          # per-frame checks
EVIDENCE_RESOLUTION = "medium"    # evidence frames: classify, reported-problem checks, claim verification
FRAME_WIDTH = 768                 # px, frames extracted by ingest.py
DECISION_WORKERS = 8              # parallel per-frame Decisions calls (keep modest: temp accounts)

# --- Context (step 6) ---
TIMEZONE = "America/New_York"
NYC_LAT, NYC_LON = 40.7128, -74.0060
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# --- Rules (step 7) ---
MERGE_GAP_S = 2.0                 # frames this close (same rule) merge into one finding
MIN_CEILING_FRAMES = 2            # SMOKE_CO needs this many clear ceiling frames before flagging
THERMOSTAT_MIN_P = 0.70           # step 5: read thermostat only if a frame has P(visible) >= this

# --- Classify / route / verify (steps 8-11) ---
# HPD_CATEGORIES is built at the bottom of this file from data/hpd_reference.json.
# Likely severity (score levels 0..2). We never output official HPD classes.
SEVERITY_LEVELS = [
    "non-hazardous: minor condition that should be fixed (e.g. small area of peeling paint, old stain)",
    "hazardous: affects health or safety (e.g. active leak, mold patch, low daytime heat)",
    "immediately hazardous: urgent danger (e.g. no window guard where a young child lives, "
    "no heat at night, peeling paint where a child under 6 lives)",
]
SEVERITY_NAMES = ["non-hazardous", "hazardous", "immediately hazardous"]
CARBON_WINDOW_S = 5.0             # Carbon re-checks frames within ±5 s of a finding
CARBON_MAX_FRAMES = 3             # at most this many frames sent to Carbon per re-check
CARBON_MAX_PER_CLIP = 3           # budget: more uncertain findings than this go to "film again"
VERIFY_MIN_P = 0.50               # claims with P(supported) below this are removed from the draft

# --- Thresholds ---
CONF_HIGH = 0.70   # include finding
CONF_LOW  = 0.35   # below this, drop; between LOW and HIGH -> Carbon re-check or "film again"
                   # (low on purpose: favour catching problems; Carbon + claim check filter false alarms)
FPS = 1            # frames per second for clips <= MAX_FRAMES seconds
MAX_FRAMES = 40    # longer clips are sampled evenly down to this many frames
DEDUP_DIFF = 3.0   # mean abs pixel diff (0-255, 64x36 grayscale) below this = near-duplicate frame
DEDUP_MAX_RUN = 2  # never skip more than this many frames in a row

# --- Retry (PLAN.md Section 3): first try + retries after 1 s, 2 s, 4 s ---
RETRY_DELAYS_S = (1, 2, 4)
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WORK_DIR = DATA_DIR / "work"      # per-clip frames/audio/outputs (gitignored)
LOG_DIR = ROOT / "logs"
MODEL_LOG = LOG_DIR / "model_calls.jsonl"


# --- HPD categories (Decisions choice options) ---
# The distinct HPD order "category" values used by the conditions in data/hpd_reference.json (from HPD's data
# dictionary); description = the plain titles of the conditions in that category.
def _hpd_categories() -> dict:
    """{category: 'title; title'} from data/hpd_reference.json."""
    import json
    ref = json.loads((DATA_DIR / "hpd_reference.json").read_text(encoding="utf-8"))
    cats: dict = {}
    for c in ref["conditions"]:
        for o in c["hpd_orders"]:
            cats.setdefault(o["category"], [])
            if c["title"] not in cats[o["category"]]:
                cats[o["category"]].append(c["title"])
    return {k: "; ".join(v)[:250] for k, v in cats.items()}


HPD_CATEGORIES = _hpd_categories()
