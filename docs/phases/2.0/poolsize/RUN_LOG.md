# Banking77 pool size — run log (paper side)

Events relevant to the order of work (protocol §8), in time order (UTC). Commit identifiers are on this repo's
`main` or on the research branches `research/poolsize-code-review` / `research/phase2.0-paper-evidence`, as marked.

| when | event | where |
|---|---|---|
| 2026-10-12, before any code | Pre-registration (systems by user decision: open stack, pplx-decider v1.1, Clef-Flash; no Clef 27B, no Jev), then pre-code review edits (descriptive pool-fit arm) | research d637927, 7232258 |
| 2026-10-12, before any cell | Code including the §7 analysis script (`scripts/poolsize_analysis.py`, sha `149fc720`), then a docstring and metadata-string edit to `poolsize_open.py` (final temperature recorded, not applied; no behaviour change) | research a4dad90, 1c3d3fd |
| before any cell | Pools, manifests (sha-pinned in `manifest_meta.json`) | research branch, `results/poolsize/` |
| 10-13 12:28–14:37 | pplx-decider v1.1, 11 cells, GB300 serving engine (R16 A3 configuration), concurrency 2 | summaries |
| 10-13 13:43 – 10-14 03:04 | Clef-Flash, 11 cells, release code, GB10, batch 8 | summaries |
| 10-14 03:36–10:17 | Open stack, 11 cells, and its pool-fit arm on the 10 small pools, GB10 | summaries |
| before the §7 run | **Disclosure.** While checking status, the research session tailed the open-stack log, whose last line printed one cell's accuracy (open_poolfit pc8_d5, 0.8253). This was after the analysis script was committed; no other cell was seen and nothing changed since. | research session report |
| 10-17 | All 43 cells committed (0 failures, 0 reruns, 0 unanswered, 0 truncated); the single §7 run → `results/poolsize/analysis.json` | research 047d708, 91a5dda |
| 10-17 | Verification on this repo: `scripts/poolsize_bodies.py verify` rebuilds every body of every pool with matching sha256 (33,880 bodies); `paper/scripts/numbers_poolsize.py` independently recomputes the pools from the protocol text, every recorded retrieval with its own vectorised BM25 (33,860 identical, 20 differing only in the order of exactly tied scores), every summary's manifest sha, and every accuracy, ECE, Brier, McNemar pair, Holm value, rank order, size effect and pool-fit difference of `analysis.json` (exact, or within 1.5e-4 where the analysis averages or subtracts values already rounded to 4 decimals). The open stack's full cell equals the stable-tie re-score of the clean run (same retrieval, 0 predictions differ). | main (this commit) |
