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
