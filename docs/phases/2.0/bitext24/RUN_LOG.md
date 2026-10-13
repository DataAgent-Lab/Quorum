# R16 Bitext 24-shot — run log (paper side)

Events relevant to the order of work (protocol §10), in time order (UTC, 2026-10-12). The commit identifiers are on
this repo's `main` or on the research branch `research/phase2.0-paper-evidence`, as marked.

| when | event | where |
|---|---|---|
| before any run | Pre-registration, then amendment A1 (recorded retrieval), the code, and the manifest (sha256 `1420b379…`) | research 9d6dc44, 3d84cc3, 837018a, ce92f03 |
| before any pplx call | The research session's amendment on the pplx serving configuration (first numbered A2, renumbered A3) | research 34f5615 |
| 02:5x, before any scored Jev call | Amendment A2: non-near-duplicate subset (2,484 ids), floor row, Jev send order and cap rule, Jev prediction = choice, exposure disclosure | main f4ac106 |
| 02:51 | Jev smoke: 3 pool rows, self excluded, outputs discarded | main (ledger only) |
| 02:5x–03:02 | Jev scored arm: 5,375 sent and answered on the first attempt, 0 failed, 0 unsent; served `jev-1.13.0` throughout. Counts only; no accuracy computed | main 1624980 |
| after the §9 script | §9 analysis script with A2 folded in | research ed2962d, 1da31ee |
| after the §9 script | pplx arm copied from the GPU host. **Disclosure:** the copy step's leak check printed the arm's summary block (headline accuracy, unanswered count, prompt-token sizes) before the joint §9 run. This was after the §9 script was committed, so §10 holds. It broke only an informal agreement between the sessions not to read any arm before the joint run. No code, amendment or analysis has changed since. | research 39774b0 |
| 08:10–09:03 | Clef-Flash arm, release code, one GB10 (smoke excluded the self row) | research f41ce62 |
| 09:09–11:56 | Clef arm, release code, one GB10 | research f41ce62 |
| 11:58–12:52 | Open stack: smoke; selection on S (12:25; 38 of 48 candidates tied at 324/324, pre-declared tie-break → M2, names, 1 order); single test run 12:25–12:52 | research f41ce62 |
| after the §9 script, before §9 ran | **Disclosure.** While checking the chain, the research session tailed the open-stack test log, which printed the headline test accuracy. As with the pplx summary above, this was after the §9 script was committed; nothing has changed since. | research session report |
| after all five arms | The single §9 run (`scripts/r16_analysis.py` @ 1da31ee, run once, pinned dataset offline) → `results/r16/analysis.json` | research 3a1a2b8 |
| after §9 | Independent recomputation on this repo: `paper/scripts/numbers_bitext24.py` reproduces every accuracy, McNemar pair, Holm value, ECE, Brier, calibration difference and the floor row of `analysis.json` exactly; bootstrap intervals agree with an independent bootstrap within 0.002 | main (this commit) |
