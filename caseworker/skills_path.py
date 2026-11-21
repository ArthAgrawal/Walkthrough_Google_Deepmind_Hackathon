"""Path to the hpd-lookup skill folder (its name has a hyphen, so it isn't importable directly)."""
from pathlib import Path

HPD_DIR = Path(__file__).resolve().parent / "skills" / "hpd-lookup"
