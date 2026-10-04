# Phase 2.0 — Requests to the research session (model runs + provenance)

> Spec: [`spec_details.md`](./spec_details.md) · Tasks: T0.5 · Delivery: a branch named
> **`research/phase2.0-paper-evidence`** on this repo (do not push `main`). This session validates and merges.
> Public repo: keep every delivered file de-identified (no internal names, hosts, paths or IPs).
> **Order = priority.** R1 decides whether the paper's main 24-shot claim stands, so it comes first.

## R1 — Selection provenance of the 0.938 configuration (gap G11) — answer first, no compute needed
For the configuration reported as "0.938, McNemar p = 0.0014 (b = 109, c = 66)", described in the README as
"per-class descriptions and a calibration-selected combination", and "survives Bonferroni":
1. Which split was used to choose it (exact item ids/indices; size; how it was carved out of the Banking77 train set;
   whether those items were also in the retrieval pool at test time)?
2. The full list of configurations evaluated (on any split, **including any evaluated on the test set**), with scores.
3. The selection rule (metric, split, tie-break) and the "calibration" step (what was fitted, on which data).
4. The family used for "survives Bonferroni" (which comparisons, how many).
5. How these constants were set: `COSINE_SCALE = 20` (quorum/ensemble.py), `KNN_SCALE = 20`, `KNN_FLOOR = -1`,
   `perms = 3` (quorum/fewshot.py), and the reader prompt/task description.
Deliver as `provenance/banking77_24shot_selection.md` (+ any logs). **If any choice touched test labels, say so
plainly** — we will then re-select on a held-out split and run the test once (R1b, requested separately).

## R2 — Per-item 24-shot dumps for 0.932 (repo default) and 0.938 (study best) (G3, G1)
`results/predictions/banking77_24shot_{default,best}.jsonl.gz`, one row per official test item, **in official
test order (3,080 rows)**:
`{"idx", "text", "gold", "pred", "probs": [K], "reader_probs": [K], "knn_probs": [K], "retrieved_idx": [24],
"perm_reader_probs": [[K] × perms]}`
plus `banking77_24shot_{default,best}_meta.json`: config (label text, task description, retrieval settings,
scales/floor, perms, seeds), quorum git commit, base model + adapter revision (hash), torch/transformers versions,
device, wall time. Acceptance here: accuracy recomputed from `pred` equals the meta; `probs` == geometric mean of
reader/knn within 1e-5; row texts match the official test order.

## R3 — Per-member probabilities for the zero-shot dumps (G1)
For all 7 datasets × {names, descriptions}: add `"member_probs": [[K],[K],[K]]` (PrismNLI, bge-large, bge-base;
post-softmax, T = 1) to rows matching the existing `results/predictions/*` files (same idx/order, same commit
config). Acceptance: geometric mean of `member_probs` reproduces existing `probs` within 1e-5; `argmax` equals
`member_picks`.

## R4 — Adapter training record (G6, Appendix A)
For `Quorum-Reader-Qwen3-4B-Adapter`: training code, dataset list with versions/configs/splits and example counts,
the held-out list (all 7 eval targets?), contamination checks performed, hyper-parameters, seed(s), base-model
revision, compute. Deliver `provenance/adapter_training.md` (+ code under `scripts/` if it can be public).

## R5 — 24-shot ablations and retrieval-only baselines (G12)
Full Banking77 test, per-item dumps (R2 schema where applicable): (a) reader alone, (b) kNN alone, (c) base
Qwen3-4B **without** the LoRA (same prompt/scoring), (d) BM25 label vote over the 24 retrieved examples
(majority + similarity-weighted).

## R6 — Protocol-parity sensitivity runs (G4)
The public reproduction (github.com/simonmesmith/jev-banking77-experiment @ 5cac4ff) retrieves with BM25
word + bigram, **at most four examples per class**, up to 24. Run our best and default configs with (a) that
retrieval rule; (b) if feasible, its label definitions as label text (note: that repo has **no license** — read
at run time for a sensitivity analysis only, do not copy the file into this repo). Per-item dumps.

## R7 — Description authoring log (G8)
For each of the 7 description sets (`examples/*_descriptions.py`): who/what produced them (human, which LLM +
prompt), what information was visible (label names only? training examples? any test results?), how many
revisions, and whether any revision followed an evaluation on a test split. Deliver
`provenance/descriptions_authoring.md`.

## R8 — GPU latency/throughput (G9)
Zero-shot ensemble (batch 1 and batched) and 24-shot pipeline: ms/item with hardware spec (GPU model, CPU, RAM),
precision, batch sizes, warm vs cold.

## R9 — Variance (G5)
24-shot: 3 seeds × option-order permutations for default and best (per-item dumps or at least per-run accuracy +
McNemar vs the reproduction). Zero-shot is deterministic — confirm (fp32) with one repeat on one dataset.

## Hand-back checklist (the research session ticks these in the branch's PR description)
- [ ] R1 answered (blocking).  - [ ] R2 dumps + meta.  - [ ] R3 member probs.  - [ ] R4 adapter record.
- [ ] R5 ablations.  - [ ] R6 parity runs.  - [ ] R7 authoring log.  - [ ] R8 latency.  - [ ] R9 variance.
