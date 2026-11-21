"""Step 4: ONE Decisions API call per frame with ALL quality + visual yes/no questions (Section 8)."""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
import models


def _q(instructions: str, true_d: str, false_d: str) -> dict:
    """Build one yes/no ("noul") Decisions question with true/false criteria."""
    return {"type": "noul", "instructions": instructions,
            "criteria": {"true_description": true_d, "false_description": false_d}}


QUALITY = {
    "too_dark": _q("Is this frame too dark to see what is in the room?",
                   "Mostly black or so underexposed that objects can't be recognized.",
                   "Objects and surfaces can be recognized, even if dim."),
    "too_blurry": _q("Is this frame too blurry to recognize objects?",
                     "Heavy motion blur or out of focus; objects can't be recognized.",
                     "Objects can be recognized, even with slight blur."),
    "camera_blocked": _q("Is the camera lens blocked or covered?",
                         "A finger, hand, pocket or object covers most of the frame.",
                         "The room is visible."),
}

VISUAL = {
    "window_visible": _q("Is a window visible in this frame?",
                         "A window (glass pane with frame) is clearly visible.",
                         "No window, or only a door, mirror, picture or screen."),
    "window_guard_installed": _q("Is a window guard installed on a visible window?",
                                 "Metal bars or a metal guard fixed across the lower part of the window.",
                                 "Plain glass, blinds, curtains, screens, or no window visible."),
    "ceiling_visible": _q("Is the ceiling clearly visible in this frame?",
                          "A substantial area of the ceiling is in view.",
                          "Ceiling not in view, or only a thin edge."),
    "smoke_detector_visible": _q("Is a smoke detector visible?",
                                 "A round or square smoke alarm mounted on the ceiling or high on a wall.",
                                 "No smoke alarm visible (light fixtures and sprinklers don't count)."),
    "co_detector_visible": _q("Is a carbon monoxide (CO) detector visible?",
                              "A CO alarm or combination smoke/CO alarm (often labeled CO) is visible.",
                              "No CO alarm visible."),
    "thermostat_visible": _q("Is a thermostat or wall temperature display visible?",
                             "A wall thermostat or room temperature display is visible.",
                             "No thermostat visible (light switches and outlets don't count)."),
    "radiator_visible": _q("Is a radiator or heating unit visible?",
                           "A radiator, baseboard heater or heating/AC unit is visible.",
                           "No heating unit visible."),
    "water_stain": _q("Is there a water stain or water damage on a wall or ceiling?",
                      "Brown/yellow ring-shaped stains, bubbling or sagging from water.",
                      "Clean surface, shadows, or normal texture."),
    "active_drip": _q("Is water actively dripping or leaking?",
                      "Visible drops, a stream of water, or a fresh wet patch spreading.",
                      "No visible dripping or leaking water."),
    "dark_growth_on_surface": _q("Is there dark growth (spots or patches) on a surface?",
                                 "Clusters of black, green or gray spots or fuzzy patches on walls, ceiling, grout or caulk.",
                                 "Clean surfaces, shadows, dirt smudges or patterned decor."),
    "peeling_paint": _q("Is there peeling, flaking or chipping paint?",
                        "Paint lifting, curling, flaking or chipped off the wall, ceiling, window or door.",
                        "Intact painted surface."),
    "rust_stain": _q("Is there a rust stain (orange-brown, from metal)?",
                     "Orange-brown staining next to or below metal pipes, fixtures, radiators or bars.",
                     "No rust staining."),
    "bathroom": _q("Is this a bathroom?", "Toilet, bathtub, shower or bathroom sink visible.",
                   "Not a bathroom."),
    "kitchen": _q("Is this a kitchen?", "Stove, oven, kitchen sink, fridge or kitchen counters visible.",
                  "Not a kitchen."),
    "bedroom": _q("Is this a bedroom?", "A bed is the main furniture in the room.", "Not a bedroom."),
}

QUESTIONS = {**QUALITY, **VISUAL}
QUALITY_FAIL_P = 0.5


def frame_key(answers: dict) -> str:
    """Key output for the call log: quality failures, else notable visual answers (P >= 0.4)."""
    p = {k: a.noul for k, a in answers.items()}
    bad = [f"{k}={p[k]:.2f}" for k in QUALITY if p.get(k, 0) >= QUALITY_FAIL_P]
    if bad:
        return "unusable: " + ", ".join(bad)
    notable = sorted(((v, k) for k, v in p.items() if k in VISUAL and v >= 0.4), reverse=True)
    parts = [f"P({k})={v:.2f}" for v, k in notable[:5]]
    if p.get("window_visible", 0) >= 0.4 and "window_guard_installed" not in [k for _, k in notable[:5]]:
        parts.append(f"P(window_guard_installed)={p['window_guard_installed']:.2f}")
    return ", ".join(parts) or "nothing notable"


def check_frame(frame: dict, work_dir: Path) -> dict:
    """Run all questions on one frame in ONE Decisions call; return a FrameCheck dict."""
    block = models.media_block(Path(work_dir) / frame["frame_path"], resolution=config.FRAME_RESOLUTION)
    idx = Path(frame["frame_path"]).stem.lstrip("0") or "0"
    answers = models.decide(
        [{"type": "text", "text": "One frame from a tenant's video walkthrough of their NYC apartment.\n"},
         block],
        QUESTIONS,
        step="frame_checks",
        purpose=f"{len(QUESTIONS)} yes/no checks on frame {idx} ({frame['t']:.0f} s)",
        key=frame_key,
    )
    p = {k: round(answers[k].noul, 3) for k in QUESTIONS if k in answers}
    usable = all(p.get(k, 0.0) < QUALITY_FAIL_P for k in QUALITY)
    return {"t": frame["t"], "frame_path": frame["frame_path"], "usable": usable, "p": p}


def check_frames(frames: list[dict], work_dir: Path) -> list[dict]:
    """Check every frame (parallel, config.DECISION_WORKERS at a time); returns FrameChecks sorted by t.

    A frame whose call fails after retries is returned with usable=False and an "error" field.
    """
    def safe(frame: dict) -> dict:
        try:
            return check_frame(frame, work_dir)
        except Exception as exc:
            return {"t": frame["t"], "frame_path": frame["frame_path"], "usable": False, "p": {},
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    with ThreadPoolExecutor(max_workers=config.DECISION_WORKERS) as pool:
        # copy_context() per task (in this thread) so worker calls reach the per-job call log
        futures = [pool.submit(contextvars.copy_context().run, safe, f) for f in frames]
        results = [fut.result() for fut in futures]
    return sorted(results, key=lambda r: r["t"])
