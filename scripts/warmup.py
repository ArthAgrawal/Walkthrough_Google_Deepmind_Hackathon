"""Warm-up before the demo: ONE tiny call to each model (Hydrogen, Neon, Carbon, Decisions), in parallel.

Run from the repo root:  python scripts/warmup.py
Prints each model's latency so you know everything is awake (new models can be slow on first call).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import models  # noqa: E402

CHECKS = {
    "Hydrogen": lambda: models.hydrogen("Reply with: ok", step="warmup").text,
    "Neon": lambda: models.neon("Reply with: ok", step="warmup").text,
    "Carbon": lambda: models.carbon("Reply with: ok", step="warmup", thinking="LOW").text,
    "Decisions": lambda: f"P(yes)={models.decide('The radiator is cold.', {'cold': {'type': 'noul', 'instructions': 'Is the radiator cold?'}}, step='warmup')['cold'].noul:.2f}",
}


def check(name: str) -> str:
    """Run one warm-up call; return a status line."""
    t0 = time.perf_counter()
    try:
        out = (CHECKS[name]() or "").strip()[:30]
        return f"[OK]   {name:<10} {time.perf_counter() - t0:5.1f} s  {out!r}"
    except Exception as exc:
        return f"[FAIL] {name:<10} {time.perf_counter() - t0:5.1f} s  {type(exc).__name__}: {str(exc)[:120]}"


if __name__ == "__main__":
    print(f"models: {config.HYDROGEN}, {config.NEON}, {config.CARBON}, {config.DECISION}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        lines = list(pool.map(check, CHECKS))
    print("\n".join(lines))
    sys.exit(0 if all(line.startswith("[OK]") for line in lines) else 1)
