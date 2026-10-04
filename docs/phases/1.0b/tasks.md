# Phase 1.0b — Tasks (demo page)

> Spec: [`spec_details.md`](./spec_details.md) · Conversation: `docs/archive/conversation-phase1.0b-demo-real-datasets.txt`

## T0 — Design consensus
- [x] T0.1 Prototype "The Verdict" (`prototype/`), real cached verdicts embedded; screenshots desktop/dark/mobile.
- [x] T0.2 User: proceed to wire the real API with the cold-start handling (2026-10-04).

## T1 — Build
- [x] T1.1 `serve/export_demo_samples.py` → `space/samples.json` (from the dumps; Bitext placeholders excluded).
- [x] T1.2 `space/index.html`: the prototype design on real data (fetch `samples.json`), real `/predict` wiring,
      `?api=` override, copy fix ("7 intents · Voice commands").
- [x] T1.3 Cold-start + recovery: `/health` probe, waking-up message + elapsed counter, 30 s / 180 s timeouts,
      one automatic retry, retry button.
- [x] T1.4 Seed + drift test follow the new page (own-labels default example).
- [x] T1.5 `space/README.md` updated.

## T2 — Tests
- [x] T2.1 AC-8 every sampled (text, labels) is a hit on the live API (7 × 5 via HTTP).
- [x] T2.2 AC-1/2/3/6/7 browser E2E against a local API on a copy of the production cache; screenshots in
      `e2e-screenshots/`.
- [x] T2.3 AC-4 cold start against a local API with the model unloaded.
- [x] T2.4 AC-5 recovery against a failing API stub.
- [x] T2.5 AC-9 unit test.

### Results (2026-10-04)
- AC-8: 35/35 sampled (sentence, description labels) pairs are hits on the live production API (≤ 220 ms).
- Browser E2E (`e2e/demo_e2e.py`, real Chrome, real local API on the seeded cache, models starting cold): 19/19 —
  all 7 domains answered from the cache (K = 7…151 ranked), cold start shows the waking message + clock and returns
  a live verdict after 31–39 s, edited > 50-label sentence blocked with a reason, SNIPS edit computed live, one
  automatic retry after an injected 503, calm error + working "Try again", desktop dark / mobile / mobile dark
  without overflow, reduced motion shows the final state, no console errors. Screenshots: `e2e-screenshots/`.
- Fixed during testing: the footnote reported the 350 ms deliberation animation as response time — it now reports
  the real API latency (cache hits 0.01–0.08 s locally).
- Unit tests: 46 passed (drift test now guards the page's own-labels default).

## T3 — Go-live (must-ask)
- [x] T3.1 User approved go-live (2026-10-04). Published by the research session (it holds the HF token) from main
      c013b4e: Space commit dc580a2 (index.html, samples.json, README.md; sha256 matched), then 771efb1 removed the
      template's unused style.css. Space repo now: .gitattributes, README.md, index.html, samples.json.
- [x] T3.2 Live Space verified in real Chrome against the production API (`e2e/verify_live.py`): 9/9 — new page
      live on desktop + mobile; default Banking, CLINC150, SNIPS, Bitext and the own-labels default answered from
      the cache in 0.03–0.28 s; no console errors.
