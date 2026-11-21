"""Regression check for the 3(+1)-bucket report: runs the full pipeline on videos and prints
problem | source | video check | HPD verdict | bucket | decided by, then checks the expected buckets.

Usage (repo root):  python scripts/check_buckets.py                 (every video in data/raw/)
                    python scripts/check_buckets.py data/raw/flush_18s.mp4 ...
Expected results live in EXPECTED below (keyed by file name). Exit code 1 if any expectation fails.
Each run makes ~50-80 model calls per video.
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import pipeline  # noqa: E402

BUCKET_NAMES = {"reported_violation": "1 reported, likely violation", "reported_not_violation": "2 reported, likely NOT",
                "found_violation": "3 found on video, likely violation", "confirm": "4 confirm / film again",
                "other_checks": "- other checks (not violations)"}
BUCKET_NUM = {"reported_violation": "1", "reported_not_violation": "2", "found_violation": "3", "confirm": "4",
              "other_checks": "-"}

# (regex on the problem text/label, expected bucket number, expected video check or None)
EXPECTED = {
    "flush_18s.mp4": [(r"flush|toilet", "2", "contradicts")],
    "hotwater_flush_37s.mp4": [(r"hot water", "1", None), (r"flush|toilet", "1", None)],
    "venue_01.mp4": [(r"wash|dryer", "1", None), (r"window.*(jam|open|stuck)|jam", "1", None),
                     (r"dishwasher", "2", None), (r"stove|burner", "4", None)],
}
# videos where no LEAK may appear in buckets 1-4 (water sitting in a sink is normal use, not a leak)
NO_LEAK = {"hotwater_flush_37s.mp4"}


def rows_for(report: dict) -> list[dict]:
    """Flatten buckets into table rows."""
    rows = []
    for key, items in report["buckets"].items():
        for c in items:
            src = "reported" if c["kind"] == "reported" else ("reported+video" if c.get("mentioned") else "video")
            if c["kind"] == "reported" and c.get("merged"):
                src = "reported+video"
            rows.append({"problem": (c.get("label") or "")[:48], "source": src,
                         "video": c.get("video_check") or ("seen" if c["kind"] == "rule" else "-"),
                         "hpd": c.get("verdict") or ("rule: violation" if c.get("is_violation") else
                                                     "rule: not violation" if c.get("is_violation") is False else "rule: unsure"),
                         "bucket": BUCKET_NUM[key], "by": c.get("decided_by") or "", "rule": c.get("rule")})
    return rows


def print_table(rows: list[dict]) -> None:
    """Fixed-width table."""
    hdr = ("problem", "source", "video check", "HPD verdict", "bucket", "decided by")
    w = (48, 15, 12, 20, 6, 50)
    print("  " + " | ".join(h.ljust(n) for h, n in zip(hdr, w)))
    print("  " + "-+-".join("-" * n for n in w))
    for r in sorted(rows, key=lambda r: r["bucket"]):
        print("  " + " | ".join(str(v)[:n].ljust(n) for v, n in
                                zip((r["problem"], r["source"], r["video"], r["hpd"], r["bucket"], r["by"]), w)))


def check(name: str, rows: list[dict], report: dict) -> list[str]:
    """Compare against EXPECTED; return failure messages."""
    fails = []
    for pat, bucket, video in EXPECTED.get(name, []):
        hits = [r for r in rows if re.search(pat, r["problem"], re.I)]
        if not hits:
            fails.append(f"{name}: no row matches /{pat}/ (expected bucket {bucket})")
            continue
        if not any(r["bucket"] == bucket and (video is None or r["video"] == video) for r in hits):
            got = ", ".join(f"bucket {r['bucket']} / video {r['video']}" for r in hits)
            fails.append(f"{name}: /{pat}/ expected bucket {bucket}" + (f" with video {video}" if video else "") + f", got {got}")
    if name in NO_LEAK:
        leaks = [r for r in rows if r["rule"] == "LEAK" and r["bucket"] != "-"]
        if leaks:
            fails.append(f"{name}: water in the sink was treated as a leak (bucket {leaks[0]['bucket']})")
    return fails


def main() -> int:
    """Run the pipeline on each video, print the table, check expectations."""
    sys.stdout.reconfigure(encoding="utf-8")
    videos = [Path(a) for a in sys.argv[1:]] or sorted(p for p in config.RAW_DIR.iterdir()
                                                       if p.suffix.lower() in (".mp4", ".webm", ".mov"))
    all_fails = []
    for v in videos:
        t0 = time.time()
        rep = pipeline.run(v, work_dir=config.WORK_DIR / f"check_{v.stem}")
        rows = rows_for(rep)
        print(f"\n=== {v.name}  ({time.time() - t0:.0f} s; {rep['model_log']['summary']})")
        if rep.get("dropped"):
            normal = [d for d in rep["dropped"] if "normal use" in (d.get("route_reason") or "")]
            if normal:
                print(f"  dropped as normal-use water: {', '.join(d['id'] + ' @' + str(round(d['t_start'])) + 's' for d in normal)}")
        print_table(rows)
        fails = check(v.name, rows, rep)
        for f in fails:
            print("  MISMATCH:", f)
        if v.name in EXPECTED or v.name in NO_LEAK:
            print("  RESULT:", "PASS" if not fails else "FAIL")
        all_fails += fails
    print(f"\n{'ALL EXPECTATIONS MET' if not all_fails else f'{len(all_fails)} MISMATCH(ES)'}")
    return 1 if all_fails else 0


if __name__ == "__main__":
    sys.exit(main())
