# Failure log

What broke → what we changed → what improved (with numbers). Newest last.

## 1. Neon was too slow for a live demo
- **Broke:** a one-word smoke-test reply from Neon took **13 s** at its default thinking level.
- **Changed:** default `thinking_level=LOW` for Hydrogen and Neon (`config.THINKING_DEFAULT`), overridable per call.
- **Improved:** same call **2.5 s**. Household facts ~6 s, complaint writing ~12 s.

## 2. Carbon re-check took 4.5 minutes
- **Broke:** the first end-to-end run hung in "route": one Carbon re-check at HIGH thinking with 6 frames took **265 s**
  (no timeout was set, so the whole report waited).
- **Changed:** Carbon at MEDIUM thinking, 3 frames, 45 s server-side deadline; on timeout/failure → "film this spot again".
  Measured on the same uncertain finding: HIGH 265 s, MEDIUM **17 s**, LOW 4.5 s, all with the same verdict
  ("absent": it was a water stain, not mold).
- **Improved:** end-to-end on the same clip went from ~5 min to **36 s**.

## 3. Our retry made timeouts worse
- **Broke:** a request that hit its own deadline came back as `504 DEADLINE_EXCEEDED`, which our retry treated as
  transient: 4 attempts × 10 s = 44 s before failing.
- **Changed:** `DEADLINE_EXCEEDED` (and client-side timeouts) are not retried; 408/429/5xx still retry after 1/2/4 s.
  Also disabled the SDK's own Decisions retries (4 by default) so they don't stack with ours.
- **Improved:** a deadline failure now fails fast in **10 s** and the caller falls back.

## 4. Video-only mode flagged unconfirmed windows as violations
- **Broke:** in video-only mode, two "window without a guard" findings had every complaint claim rejected by the
  Decisions claim check (P(supported) 0.47 and 0.43) but still showed as violations on the page.
- **Changed:** a finding whose claims all fail verification moves to "Please confirm or film again".
- **Improved:** on venue_01 video-only, violations 2 → **1**; the other two windows became "film again".

## 5. Likely severity under-calls the worst violations (OPEN)
- **Broke:** on 200 real HPD violations, Decisions' likely-severity accuracy is **55%** (Neon baseline 62%, chance 33%).
  36 of 66 class C ("immediately hazardous") violations were called "hazardous". Severity confidence barely tracks
  accuracy (see `charts/calibration.png`).
- **Why we think so:** our three severity level descriptions use our 6 conditions as examples; HPD's class C covers
  many other things (fire damage, painted sprinkler heads, egress). 
- **Not changed yet:** rewriting the levels from HPD's class definitions and re-scoring on this same sample would be
  tuning on the test set. Next step: rewrite on a separate dev sample, then re-run once on this one.
- Category, by contrast, is well calibrated: confidence ≥ 0.9 → **98%** accurate on 47% of records.
