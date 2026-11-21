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
MANAGED_AGENT = "antigravity-preview-05-2026"   # VERIFY in Managed Agents docs (step 5)

API_VERSION = "v1beta"            # Decisions API requires v1beta

# --- Thresholds ---
CONF_HIGH = 0.70   # include finding
CONF_LOW  = 0.40   # below this, drop; between LOW and HIGH -> Carbon re-check or "film again"
FPS = 1            # frames per second sampled from video

# --- Retry (PLAN.md Section 3): first try + retries after 1 s, 2 s, 4 s ---
RETRY_DELAYS_S = (1, 2, 4)
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = ROOT / "logs"
MODEL_LOG = LOG_DIR / "model_calls.jsonl"
