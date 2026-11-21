"""Run pipeline steps 1-7 on one video and print HouseholdFacts + candidate findings.

Usage (repo root):
  python scripts/run_pipeline.py data/raw/venue_01.mp4
  python scripts/run_pipeline.py clip.mp4 --intro "my son is 6, 20 apartments in the building"
  python scripts/run_pipeline.py clip.mp4 --video-only
  python scripts/run_pipeline.py clip.mp4 --time 23:30 --outdoor 40
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402


def main() -> int:
    """Parse args, run steps 1-7, print a summary; full JSON is saved in the clip's work dir."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--intro", help="use this text instead of the spoken intro")
    ap.add_argument("--video-only", action="store_true", help="drop spoken facts (baseline)")
    ap.add_argument("--time", help="override local time HH:MM")
    ap.add_argument("--outdoor", type=float, help="override outdoor temp °F")
    ap.add_argument("-q", "--quiet", action="store_true", help="hide per-call log lines")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")

    r = pipeline.run_steps_1_7(args.video, intro_override=args.intro, video_only=args.video_only,
                               local_time=args.time, outdoor_f=args.outdoor)

    usable = sum(f["usable"] for f in r["frame_checks"])
    errors = sum("error" in f for f in r["frame_checks"])
    print(f"\n=== {r['video']}  ({r['duration']:.1f}s, {len(r['frame_checks'])} frames, "
          f"{usable} usable, {errors} errors)")
    if r["transcript"]:
        print(f"\n--- Transcript ({r['transcript']['language']}) ---\n{r['transcript']['text'] or '(no speech)'}")
    print("\n--- HouseholdFacts ---")
    print(json.dumps(r["household"], ensure_ascii=False, indent=2) if r["household"] else "(video only: none)")
    print("\n--- Context ---\n" + json.dumps(r["context"]))
    if r["thermostat"]:
        print("--- Thermostat ---\n" + json.dumps({k: v for k, v in r["thermostat"].items() if k != "raw"}))
    print(f"\n--- Candidate findings ({len(r['candidates'])}) ---")
    for f in r["candidates"]:
        flag = {True: "VIOLATION", False: "not a violation", None: "UNDETERMINED"}[f["is_violation"]]
        print(f"{f['id']:>3} {f['rule']:<14} {f['t_start']:>5.1f}-{f['t_end']:<5.1f}s conf={f['confidence']:.2f} "
              f"[{flag}] {f['evidence_frame']}\n      {f['explanation']}"
              + (f"\n      quote: {f['tenant_quote']}" if f["tenant_quote"] else ""))
    print("\n--- Trace ---")
    for t in r["trace"]:
        print(f"  {t['step']:<13} {t['model'] or '-':<20} calls={t['calls']:<3} {t['ms']} ms")
    print(f"\nFull JSON: {Path(r['work_dir']) / ('steps_1_7_video_only.json' if r['video_only'] else 'steps_1_7.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
