"""Smoke test: ONE call to each of the 4 models (Hydrogen, Neon, Carbon, Decisions), via models.py.

Run from the repo root:  python scripts/smoke_test.py
If a text model id is wrong (404), prints the gemini-4* model ids the key can see.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import models  # noqa: E402


def list_gemini4_models() -> list[str]:
    """Return model ids containing 'gemini-4' or 'decision' visible to this API key."""
    names = []
    for m in models.get_client().models.list():
        name = (m.name or "").removeprefix("models/")
        if "gemini-4" in name or "decision" in name:
            names.append(name)
    return sorted(names)


def check_text(label: str, fn) -> bool:
    """Make one tiny call with a text model wrapper; print the reply. True on success."""
    try:
        resp = fn("Reply with exactly the word: pong", step="smoke_test")
        print(f"[OK]   {label}: {(resp.text or '').strip()[:60]!r}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {type(exc).__name__}: {str(exc)[:300]}")
        return False


def check_decisions() -> bool:
    """Make one Decisions API call with one yes/no question; print P(yes) and confidence."""
    try:
        answers = models.decide(
            "The radiator is cold and the thermostat reads 58°F at 9 PM in January.",
            {"heat_problem": {
                "type": "noul",
                "instructions": "Does the text describe an apartment heating problem?",
                "criteria": {"true_description": "Cold radiator or low indoor temperature.",
                             "false_description": "No heating issue described."},
            }},
            step="smoke_test",
        )
        a = answers["heat_problem"]
        print(f"[OK]   DECISION ({config.DECISION}): P(yes)={a.noul:.2f} confidence={a.confidence:.2f}")
        return True
    except Exception as exc:
        print(f"[FAIL] DECISION ({config.DECISION}): {type(exc).__name__}: {str(exc)[:300]}")
        return False


def main() -> int:
    """Run all four checks; exit non-zero if any fail."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = [
        check_text(f"HYDROGEN ({config.HYDROGEN})", models.hydrogen),
        check_text(f"NEON ({config.NEON})", models.neon),
        check_text(f"CARBON ({config.CARBON})", models.carbon),
        check_decisions(),
    ]
    if not all(results):
        try:
            print("Visible gemini-4 / decision models:", list_gemini4_models())
        except Exception as exc:
            print(f"(could not list models: {exc})")
    print(f"Log: {config.MODEL_LOG}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
