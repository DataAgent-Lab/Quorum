# Phase 2.0 — Tasks (TACL paper) · v2 after self-review

> Spec: [`spec_details.md`](./spec_details.md) · Venues: [`venues.md`](./venues.md) (SSoT:
> [`docs/references/quorum-paper-venues/quorum-paper-venues.md`](../../references/quorum-paper-venues/quorum-paper-venues.md))
> Conversation: `docs/archive/conversation-phase2.0-paper-publication.txt`
> Model runs → the research session (git branch, validated here). Nothing outward-facing without the user's OK.
> Order rule: **no section is drafted before its ledger rows are VERIFIED** (T5 depends on T1/T2 rows).

## T0 — Venue, toolchain, day-1 requests
- [x] T0.1 Venue decision + format table → `venues.md` (TACL primary; D2).
- [x] T0.1b Self-review (independent reviewer) → spec/tasks v2 (spec "Self-review record").
- [x] T0.2 Vendor the current TACL style + formatting PDF; record exact rules (A4, pages, appendix policy incl.
      "will not be reviewed", desk-reject items, header, line numbers, no supplements/links, author info at
      submission; 2024 announcement supersedes the PDF's appendix sentence).
- [x] T0.3 Pin a TeX engine fitting the 5.8 GB-free disk (log-and-proceed) and compile the unmodified TACL sample.
- [~] T0.4 Scaffold `paper/` (style, scripts, generated, tools, refs-candidates.bib done; main.tex/sections at T4) (style, `main.tex`, `sections/`, `figures/`, `tables/`, `refs.bib`, `numbers.tex`
      (generated), `scripts/`, `tools/`).
- [ ] T0.5 **Day-1 research-session request** → `research-requests.md` + message: (1) G11 selection provenance
      for 0.938 (dev split, configs tried, rule, Bonferroni family, constants' origin); (2) per-item 24-shot dumps
      for 0.932 and 0.938 (same schema as `results/predictions/*` + member scores); (3) per-member probability
      dumps, 7 datasets × names/descriptions; (4) adapter training code + data list + held-out list; (5) G12
      ablations; (6) G4 matched-retrieval / matched-label-text runs; (7) G8 description authoring log; (8) GPU
      latency; (9) 24-shot permutation/seed variance. Branch name + schemas + acceptance checks specified.
- [ ] T0.6 Fix the venue reference (SSoT): TACL uses its own submission system (not OpenReview); move the prestige
      tier table + sources into the reference; venues.md cites it (no second copy of the evidence).

## T1 — Claims ledger (fact-first gate)
- [x] T1.1 `claims-ledger.md`: claim → macro name → dump/URL@commit → command → VERIFIED / GAP / DROPPED.
- [x] T1.2 `paper/scripts/zero_shot_tables.py` (reuses `quorum.metrics`): accuracies, member accuracies from
      `member_picks`, exact McNemar ensemble-vs-best-member and names-vs-descriptions, **unrounded p** → macros.
- [x] T1.3 G2: zero-shot Jev 0.801 → DROPPED (sole source a blog with no n/protocol; ledger Z5).
- [ ] T1.4 G7 split audit: our splits (CLINC plus/OOS, Bitext 20% stratified, SNIPS 1400) vs each cited baseline's split.
- [x] T1.5 C5 evidence (traps 1, 3, 4, 5 measured → ledger M1–M4; trap 2 citation verified → M5): before/after numbers for each "trap" (or the section shrinks to a paragraph).

## T2 — Evidence-gap closure
- [x] T2.1 G1 zero-shot calibration from dumps (ECE-10 as the reproduction, adaptive ECE, Brier, log loss, reliability).
- [x] T2.2 G1/G3 validate research-session dumps (clean + default + R3 VERIFIED 2026-10-05; branch merged into main 4f32702) (schema, accuracies match, row alignment) when delivered.
- [ ] T2.3 G3 `paper/scripts/mcnemar_vs_jev.py`: fetch the reproduction's `results/predictions.csv` @ `5cac4ff`
      (no redistribution), row-index join + case-folded labels, `quorum.metrics.mcnemar` for 0.932 and 0.938.
- [ ] T2.4 G1 24-shot calibration vs the reproduction's Jev ECE/Brier/log loss (same definitions).
- [x] T2.5 G11 gate: R1 delivered (research branch 8bea0c0) and verified (exact p 0.001428; Bonferroni crossover
      m=35/36); outcome recorded in spec; user chose the clean re-selection protocol (D3 revised).
- [x] T2.5b G11b: review the clean protocol committed by the research session BEFORE it runs (pre-registration
      evidence = its commit timestamp); go/changes; then validate the single test-run dump (R2 schema) + McNemar.
- [ ] T2.5c Disclosure paragraph for the 0.938 exploratory history (R1 §8 list) in the main text.
- [ ] T2.6 G4 parity table; G12 ablation table (from delivered runs); disclose the 25 overlaps.
- [ ] T2.7 G5 Holm across the declared family; bootstrap CIs; 24-shot variance.
- [ ] T2.8 G6 contamination table; Qwen3 probe decision [unproven — measure first].
- [ ] T2.9 G8 descriptions: review the authoring log; blind re-author + re-evaluate if iteration cannot be excluded.
- [ ] T2.10 G9 CPU latency (time 1 item first; fp32 serve path) + GPU numbers from the research session.
- [x] T2.11 G10 licence table (→ licences-contamination.md).
- [x] T2.11b Data notice for results/predictions (results/predictions/DATA_NOTICE.md) (MTOP CC-BY-SA-4.0, Bitext CDLA-Sharing-1.0 share-alike) — push needs user OK.
- [x] T2.17 R13 contamination-robustness (→ ledger Z8/Z9: C1 8/8 on unexposed sets with a clean NLI member; calibration 14/14 for both clean ensembles): zero-shot ensemble with a documented-clean NLI member (deberta-v3-large-zeroshot-v2.0-c); re-verify C1 + calibration on clean data; split zero-shot reporting into clean vs exposed datasets.

## T2b — Reviewer-risk closures from the novelty memo (added 2026-10-05)
- [x] T2.12 Combiner ablation (done 2026-10-05 → claims Z6/Z7) (CPU, needs R3 member probs): geometric mean vs true product of experts vs arithmetic
      mean vs majority vote, 7 datasets × 2 label texts; per-member entropy / effective weight. (Note: true product
      and geometric mean share the argmax — they differ only in sharpness, i.e. calibration.)
- [x] T2.13 Baselines a reviewer will ask for (R11 reranker → ledger B1; R10 fixed-budget fine-tune → ledger F12) (research session R10/R11): a sub-1B cross-encoder reranker as a
      zero-shot classifier; BTZSC intent subsets if comparable; Qwen3-4B fine-tuned on Banking77 train as an upper
      reference for the 24-shot result. (bge-large + descriptions alone = Hu et al. 2024-style baseline — already in
      the member columns.)
- [x] T2.14 Calibration confirmation (done: T*=0.508, ECE down 14/14, pre-declared b93b37d) (research session R12): global temperature fitted on zero-shot outputs for
      TRAIN-split samples only (no test data in fitting), frozen, then applied once to all 14 test conditions —
      turns the exploratory LODO result (Z4b) into a clean one. Bootstrap CI on the ECE gap vs Jev (24-shot).
- [~] T2.16 Retrieval tie-break sensitivity (R6c done → F14: 2 predictions change, headline unchanged; cross-platform identity verified) — remaining: + library root fix after the evidence freeze (stable sort, index tie-break, sorted term order); report platform in the paper.
- [x] T2.15 Contamination (→ licences-contamination.md; embedders clean†; **PrismNLI train-exposed to Banking77/MASSIVE via its base**, HWU64 indirect; R13 clean-NLI rerun requested): bge-v1.5 training/fine-tuning data vs our eval sets (MTEB includes
      Banking77/MASSIVE/MTOP *evaluation* tasks — check whether any *training* data overlaps); split wording
      (HWU64/Bitext have no official test split; CLINC150 variant = plus).

## T3 — Novelty, literature, bibliography
- [x] T3.0 **G13 novelty memo** → `novelty-memo.md` (verdict: empirical/methodological paper; new = pre-registered paired accuracy+calibration comparison with the closed system; frame Jev as a case study) — (before T4): positioning of C1/C2 vs PIE, dataless, NLI zero-shot, log-linear
      pooling/PoE; Jev-replication landscape (disambiguate the 2026-09 open "Jev" clones). Venue still right? (must-ask if not).
- [~] T3.1 Related-work map (in the memo). T3.2 `refs.bib` (candidates: `paper/refs-candidates.bib`, 152 entries, 12/12 sampled ids resolve; full check in T7.6) (resolvable entries only; baseline numbers with budget **and** split).

## T4 — Outline
- [ ] T4.1 Title candidates; section outline with ledger rows per paragraph; **page budget per section**; figure/table list with generators (colour-blind-safe palette).
- [ ] T4.2 Outline review by a role-isolated TACL-persona reviewer; revise.

## T5 — Drafting (each section only after its rows are VERIFIED)
- [ ] T5.1 Method + setup (incl. description authoring protocol, splits, selection protocol).
- [ ] T5.2 Results (macros only; parity + ablation tables).
- [ ] T5.3 What did not work + evaluation lessons.
- [ ] T5.4 Related work.
- [ ] T5.5 Introduction + abstract + title (last).
- [ ] T5.6 Limitations, ethics, release statement (AC-15), AI-assistance disclosure (AC-16); Appendices A/B.
- [ ] T5.7 Copy-edit pass (English, consistency, terminology); self-overlap check vs the README text.

## T6 — Phase 1.2 decision gate
- [ ] T6.1 At the cut-off set in T4.1: passed → section; failed → negative result; not run → omitted (future work). Record (AC-11).

## T7 — Tests / verification (one per AC)
- [x] T7.0 Tool fixtures (paper/tools/test_checks.py, 8 passed — layout good/bad/over-limit/exact-limit, anonymity, numbers, bib real/fake): each of `check_layout` / `check_anonymity` / `check_numbers` / `check_bib` run on a
      planted-violation sample (must flag) and a clean sample (must not flag).
- [ ] T7.1 AC-1 `check_layout` on the built PDF.
- [ ] T7.2 AC-2 `check_anonymity` + manual pass.
- [ ] T7.3 AC-3 `check_numbers`.
- [ ] T7.4 AC-4 statistics checklist.
- [ ] T7.5 AC-5 REAL: CPU zero-shot subset (SNIPS or Banking77 subset via `serve/spot_check.py`); McNemar vs pinned
      public predictions; research-session reproduction of 0.932/0.938 dumps.
- [~] T7.6 AC-6 `check_bib` — candidate bibliography resolves 152/152 (existence; titles/authors get a manual pass when refs.bib is finalised).
- [ ] T7.7 AC-7 calibration section backed by T2.1/T2.4 or removed; README correction proposal.
- [ ] T7.8 AC-8 parity table vs the reproduction's PROTOCOL.md/README.
- [ ] T7.9 AC-9 contamination table vs sources.
- [ ] T7.10 AC-10 review round 1 (2 TACL personas + claims auditor) → `reviews/round1.md` + responses.
- [ ] T7.11 AC-10 review round 2 + ≥1 human reader → `reviews/round2.md`; no open Critical/Major.
- [ ] T7.12 AC-11 Phase 1.2 gate recorded.
- [ ] T7.13 AC-12 arXiv build + license + endorsement status.
- [ ] T7.14 AC-13 submission packet + checklist.
- [ ] T7.15 AC-14 G11 outcome reflected in C3 wording.
- [ ] T7.16 AC-15 / AC-16 release statement + AI disclosure present and anonymous.

## T8 — Must-ask items (BLOCKED in the conversation file; never done silently)
- [ ] 🔲 blocked: T8.1 Author name(s), email, country, affiliation — **required at TACL submission** (BLOCKED #1).
- [ ] 🔲 blocked: T8.2 Adapter access: reviewers cannot get it during review (no links allowed); ungate at/after acceptance? (BLOCKED #2).
- [ ] 🔲 blocked: T8.3 System name in the review version (keep "Quorum" vs neutral name) + disclosure of the public repo/Space/HF pages as prior non-archival versions (BLOCKED #4).
- [ ] 🔲 blocked: T8.4 Employer publication / IP clearance before choosing the affiliation line (BLOCKED #5).
- [ ] T8.5 README corrections once evidence is in — push is must-ask: "Jev's own protocol", "on CPU in ~120 ms",
      "calibrated", "survives Bonferroni", "trained only on public intent data … Banking77 excluded" (CLINC/HWU/SNIPS
      + non-intent tasks are in the mixture), "full training pool" (9,079 for the study runs), 0.938 framing.
- [ ] T8.6 arXiv endorser — user action; posting is must-ask.
- [ ] T8.7 Submission to TACL — must-ask, after AC-1…AC-16 pass.
