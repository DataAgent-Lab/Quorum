# Phase 2.0 — Paper publication: a submission-ready TACL paper on Quorum

> **Status**: IN PROGRESS — evidence phase (2026-10-05): toolchain ready, zero-shot numbers + calibration verified, 24-shot headline verified; numbers live in [`claims-ledger.md`](./claims-ledger.md). Spec + tasks v2 (2026-10-04) after independent self-review (REVISE: 2 Critical, 15
> Warning, 7 Note — all applied or routed to must-ask; see "Self-review record") · **Predecessors**: Phases
> 1.0–1.1 (results), 1.0a/1.0b (demo, per-item dumps) · **Runs in parallel with**: Phase 1.2 (not started)
> **Venues & formats**: [`venues.md`](./venues.md) (summarised from the SSoT
> [`docs/references/quorum-paper-venues/quorum-paper-venues.md`](../../references/quorum-paper-venues/quorum-paper-venues.md))
> **Conversation**: `docs/archive/conversation-phase2.0-paper-publication.txt`

> **Commit hashes:** the public history was rewritten on 2026-10-05 (author identity only; trees, dates and order unchanged). Hashes cited in this phase's documents before that date are pre-rewrite — resolve them with [`provenance/commit_hash_map.md`](../../../provenance/commit_hash_map.md) (e.g. protocol 8455e7e→0c8602f, selection d513aa9→fcfc351, single test run 09217ba→66f5768). The paper cites post-rewrite hashes.

## Goal

Produce a **formal, venue-compliant paper** about Quorum, ready for the user to submit to **TACL**: an anonymised
PDF in the official TACL format and its LaTeX source, plus a de-anonymised arXiv build — where **every number is
generated from a reproducible artifact**, the protocol a claim depends on is stated in the reviewed main text, and
the draft has survived independent critical review. **Quality and completeness come first; no deadline drives a
cut.**

## Definition of Done (each item is observable)

1. `paper/` builds the anonymised PDF from source with the pinned engine; `tools/check_layout.py` passes (A4,
   ≤10 content pages, appendices A ≤5 / B ≤3, font sizes, line numbers, confidential header, embedded fonts,
   clean PDF metadata) — AC-1.
2. `tools/check_anonymity.py` passes on PDF text + source (incl. the system name decision, org/host/author strings)
   and a manual pass is recorded — AC-2.
3. Every number in the results sections comes from the generated `paper/numbers.tex` macros (built from the
   per-item dumps); `tools/check_numbers.py` finds no literal numbers outside allowed places; every external
   number carries a citation — AC-3.
4. Ledger rows for every claim are VERIFIED or the claim is DROPPED; the 0.938 selection-provenance gate (G11) has
   a recorded outcome — AC-4, AC-14.
5. Real-path reproductions recorded (zero-shot subset on CPU; McNemar vs the pinned public predictions; 0.932 and
   0.938 per-item dumps from the research session) — AC-5.
6. `refs.bib` fully resolvable — AC-6; release statement, AI-assistance disclosure, limitations and ethics
   present — AC-15, AC-16.
7. Two independent review rounds with no open Critical/Major; Phase 1.2 gate recorded — AC-10, AC-11.
8. Submission packet (PDF, source, Comments-to-the-Editor text incl. author list and prior non-archival
   versions) and arXiv build ready — AC-12, AC-13.
9. Docs synced: tasks.md, this spec, the conversation file, and the venue reference (SSoT) where facts changed.

## User decisions (2026-10-04)

| # | Decision |
|---|---|
| D1 | Phase number **2.0**. |
| D2 | **Primary venue: TACL** (Tier 1; rolling monthly; 10 pages + appendices; free). Fallbacks: ARR → ACL 2027 / EMNLP 2027; TMLR (Tier 2) backup. NAACL Demo only after the TACL outcome. |
| D3 | **24-shot is a main result.** ~~0.938 as the headline~~ → **revised 2026-10-05 after G11 (user decision): the headline is the single test run of a CLEAN, pre-registered re-selection protocol**, reported whatever its value and significance. 0.938 is reported as exploratory history with full disclosure (R1 §8). |
| D4 | **Phase 1.2 in parallel**; gate before the final draft with three outcomes: passed → section; failed → measured negative; **not run by the cut-off → omitted** (mentioned as future work). |
| D5 | Quality first; ARR 2026-10-12 and Industry 2026-10-31 are not targets. |

## TACL requirements this paper must meet (verified 2026-10-04 on transacl.org — see venues.md)

- Official TACL style, **A4**; **7–10 content pages**, references excluded. Appendices outside the 10 pages
  (policy since 2024-03-01): **≤5 pp replication details, ≤3 pp complementary results** — and **"Appendices will
  not be reviewed."** → every protocol detail a claim depends on goes in the main text. *(The older formatting PDF
  says the limit "includes any appendices"; T0.2 records that the 2024 announcement supersedes it.)*
- Formatting desk-reject rules (formatting PDF): paper size, ≥11 pt text / ≥10 pt captions, line numbers, the
  "Confidential TACL submission. DO NOT DISTRIBUTE." header, DOIs in references, colour-blind-safe figures.
- **No supplementary material and no links to it, "anonymized or not".** Instead: anonymous text stating whether
  and how software/data will be released.
- **Double-blind**: no names/affiliations/identifying acknowledgments; third-person self-references; "Authors are
  cautioned not to overpublicize the work"; prior non-archival versions (arXiv, README/Space/HF pages) disclosed in
  Comments to the Editor.
- **Author information is required at submission** ("email, country, and affiliation" of all authors; listed again
  in Comments to the Editor) — "Your paper will NOT proceed to review without this information."
- **No dual submission.** Submitted through TACL's own system (not OpenReview). Deadline: the 1st of each month,
  23:59 Honolulu time. Free.
- **ACL policy on AI writing assistance**: generative-AI use beyond language polishing must be disclosed in the
  Acknowledgements (worded anonymously for review).

## Working thesis and contributions (confirmed or cut by the claims ledger, T1/T2)

> *A geometric-mean jury of three sub-1B open models is a cheap, training-free zero-shot intent classifier that
> reliably beats its own strongest member across domains; and with the training set as a retrieval pool,
> 24 in-context examples and no Banking77 weight update, an open stack outperforms a public reproduction of a
> closed commercial decision API on Banking77.*

| # | Contribution | Current evidence (verify in T1/T2) |
|---|---|---|
| C1 | Ensemble beats its best member on most intent datasets (paired exact McNemar) | Names: Banking77 + 5/6 significant, SNIPS n.s. loss; **descriptions: SNIPS also a significant win** (b=34, c=16, p=0.015, from dumps) |
| C2 | Label descriptions as a free lever, and when it helps | Banking77 +0.018, SNIPS +0.091, CLINC +0.030 (Phase 1.1) — subject to G8 (authoring provenance) |
| C3 | Retrieval-augmented 24-shot open stack vs the reproduced Jev under the **reproducer's** selected setup ("retrieved24", Jev `jev-1.13.0`) — not "Jev's own protocol" | **VERIFIED headline (pre-registered, single test run): 93.57% vs 92.40%, exact McNemar b=119/c=83, p=0.0136**; repo default 93.21% is n.s. (p=0.066); 0.938 = disclosed exploratory history. **Attribution (R5):** kNN over the retrieved examples alone already matches Jev (93.02%, n.s.); the reader adds +0.55 pt (p_Holm 0.023); the adapter's own share is n.s. — never credit the fine-tuned reader alone. **Retrieval (R6):** under the reproduction's retrieval rule the open stack ties Jev on accuracy — the accuracy gain is from the retrieval rule — mainly the absent per-class cap (F13, post hoc); the **calibration (log-loss) advantage holds under parity** (paired, F9). See claims-ledger F1–F9. |
| C4 | Measured negative results | data-scaling saturation, bigger embedder hurts, reranking fails, Dialog2Flow member hurts, (Phase 1.2 if negative) |
| C5 | Evaluation lessons — reframed: practices are textbook, the contribution is **measured flips** (each trap's before/after numbers) | README "Five ways we caught ourselves" — needs the before/after numbers per trap, or it is cut to a short paragraph |
| C6 | Cost/latency | GPU ≈122 ms/item (README L98); **CPU latency is not 120 ms** (README L23 conflates) → re-measure (G9) |

## Evidence gaps — closed, or the claim is dropped

Single source of numbers: **the per-item dumps** (`results/predictions/*`, fp32, commit 04e6709) and research-session
dumps delivered the same way. `results/*.json` (older runs; e.g. SNIPS McNemar p 0.057 vs 0.072 from dumps) and
hand-entered references (`jev_reproduction_reference: 0.801`, the `--jev-ref 0.924` default) are **not** sources.

| ID | Gap | Closure |
|---|---|---|
| **G1** | "Calibrated" never measured. | ECE (10-bin, as the reproduction; + adaptive), Brier, log loss, reliability — zero-shot from dumps (CPU); **24-shot from its per-item dump**, compared with the reproduction's Jev values (`results/results.json`: ECE 0.0341, Brier 0.1193, log loss 0.784; same definitions). Ensemble-vs-member needs per-member probs (research session). Poor calibration → the word is dropped. |
| **G2** | Zero-shot Jev ≈ 0.801 unsourced (reproduction lists "definitions" 79.22% / "static" 81.82% on 154 items only). | Find a citable source or DROP. |
| **G3** | 0.938 config not public; no 24-shot per-item dump at all (`scripts/eval_24shot.py` writes aggregates only). | Research session delivers config + code + **per-item dumps for 0.932 and 0.938**; McNemar recomputed with `quorum.metrics.mcnemar` vs the reproduction's `results/predictions.csv` at pinned commit `5cac4ff` (row-index join, case-folded labels; file unlicensed → fetched, never redistributed). |
| **G4** | Protocol parity. | Table: retrieval (theirs BM25 word+bigram, **≤4 examples per class**; ours plain top-24), pool (both full 10,003 train), label text (theirs training-data-derived definitions + original label strings; ours descriptions), dev selection (theirs 154/616 pre-registered), weights (ours LoRA on other public intent data), model version (`jev-1.13.0`, date). Matched-retrieval / matched-label-text sensitivity runs (research session). Disclose the 25 normalized train/test overlaps. |
| **G5** | Statistics. | Exact McNemar (unrounded p — `quorum/metrics.py` rounds to 6 dp, which prints Banking77 as 0.0); Holm across the declared family; bootstrap 95% CIs; 24-shot permutation/seed variance. |
| **G6** | Contamination. | Per model × dataset table from cited sources; Qwen3 pretraining undisclosed → a measured probe or an explicit limitation [unproven — measure first]; LoRA contaminated for CLINC/HWU/SNIPS. |
| **G7** | Baselines, related work, **splits**. | Quote budget **and** split: CLINC uses the `plus` config (5,500 test incl. 1,000 out-of-scope as a class); Bitext has **no official test** (deterministic stratified 20%); SNIPS n=1400 (vs the common 700). Only compare to numbers on the same split, else label "not comparable". Full-data references in the 24-shot table (BERT 93.66%, Casanueva et al. 2020). |
| **G8** | Description provenance. | Authoring log from the research session (bitext/hwu64/massive/mtop descriptions first appear in commit 40fa9e2). If iteration against test results cannot be excluded → re-author blind (label names only, fixed procedure), freeze, re-evaluate [unproven — measure first]. Protocol in main text. |
| **G9** | Efficiency. | Re-measure CPU (fp32 serve path) and take GPU numbers from the research session, with hardware specs. |
| **G10** | Licenses, ethics, limitations. | License table (incl. the reproduction's CC-BY-4.0 data, its unlicensed code/predictions); limitations; ethics; commercial system named factually ("a public reproduction of Jev"). |
| **G11** | **Selection provenance of 0.938** (README: "per-class descriptions and a calibration-selected combination"; "survives Bonferroni"). | Research session supplies: the dev split, the list of configurations tried, the selection rule, the Bonferroni family, and the provenance of fixed constants (`COSINE_SCALE=20`, `KNN_SCALE=20`, `KNN_FLOOR=-1`, `perms=3`). Test-set selection → re-select on held-out data and run the test once; still significant → keep; else must-ask (D3). |
| **G11 — outcome (R1, research branch 8bea0c0, verified here)** | 0.938 was **selected on a 924-row cal split from train** (seed 0, disjoint from pool and test; member sets + rule fixed in the run script before results), and is **not test-best** (prob_mean 0.9396 not adopted). But: ~40 configurations were scored on test; the 3-order averaging and answer scoring were chosen on the first 200 **test** items (debug probe); descriptions were adopted after zero-shot test results; "survives Bonferroni" had no defined family; "pre-registered anchor" was named ~21 h after its test score; the fitted temperatures (T_reader 1.2724, T_kNN 0.4255) act as a ~3:1 mixing weight; the pool was **9,079** rows (not 10,003); a single cal-based choice across later stages gives 0.9377. | → **G11b**: clean protocol (fresh stratified split, pinned base/adapter revisions, pre-declared grid + rule committed before any run, train-side choice of orders/scoring, one test run). Protocol reviewed here before it runs (AC-14). |
| **G6 addendum (R1)** | Adapter training mixture excluded **only Banking77**; **CLINC150, HWU64, SNIPS are in it**, plus non-intent tasks → "trained only on public intent data" is inaccurate; clean 24-shot cross-domain sets are **MASSIVE, MTOP, Bitext** only. Qwen3-4B base revision was never pinned; public adapter ≡ study checkpoint unverified. | Contamination table + wording; G11b pins revisions and uses the public adapter. |
| **G12** | Missing 24-shot ablations/baselines. | Reader alone, kNN alone, base Qwen3-4B without LoRA, BM25 label vote (retrieval-only) — research session. |
| **G13** | Novelty & positioning (TACL judges originality). | Novelty memo **before** drafting: C1/C2 vs PIE, dataless, NLI zero-shot, log-linear pooling/PoE; Jev-replication landscape (several open "Jev" clones from 2026-09 — Zefan-Cai/Open-Jev, SiliconLabAI/OpenJev, TheoLeeCJ/SemIf-OpenJev, Heman10x-NGU/openJev-verdict-2.0, featherless-ai/simple-jev — disambiguated and cited properly). Lead with the scientific contribution; Jev is a secondary benchmark. If the memo finds the contribution too thin for TACL → must-ask (venue). |

## Planned manuscript structure (TACL, 10 content pages; appendices unreviewed)

1. **Introduction** — contributions C1–C6 (led by C1/C2; C3 as a strong benchmark result).
2. **Related work** — zero-shot intent detection, pooling/ensembles, retrieval ICL, calibration, evaluation methodology.
3. **Method** — members, template, per-member softmax (T=1), geometric mean; descriptions + **authoring protocol**; the 24-shot pipeline.
4. **Experimental setup** — datasets **with exact splits** (G7), selection protocol (G11), significance protocol (G5), contamination (G6), hardware.
5. **Results** — ensemble vs members; descriptions; calibration (if supported); 24-shot with **protocol-parity table** + ablations (G4, G12); cost/latency.
6. **What did not work**. 7. **Evaluation lessons** (measured flips only). 8. **Limitations**, **Ethics**, **release statement**, **AI-assistance disclosure** (anonymous wording).
- Appendix A (≤5 pp, unreviewed): full hyper-parameters, prompts, the description lists, adapter training data list. Appendix B (≤3 pp): per-dataset tables, reliability diagrams.
- A **page budget per section** is set in T4.1; overflow is resolved by moving *non-load-bearing* detail to appendices.

## Artifacts & locations

| Artifact | Location | Notes |
|---|---|---|
| LaTeX source, figures, bib | `paper/` (new) | Official TACL style vendored; pinned engine (T0.3). |
| Number macros | `paper/numbers.tex` (generated) | Built by `paper/scripts/` from the dumps; results sections use macros only. |
| Analysis/figure scripts | `paper/scripts/` | Reuse `quorum.metrics` (accuracy, exact McNemar) and `quorum.data`; no second implementation. |
| Claims ledger | `docs/phases/2.0/claims-ledger.md` | Claim → macro → dump/URL@commit → command → status. |
| Research-session requests | `docs/phases/2.0/research-requests.md` | Schemas, deliverables, branch name; sent on day 1 (T0.5). |
| Compliance tools + fixtures | `paper/tools/` | `check_layout`, `check_anonymity`, `check_numbers`, `check_bib`, each with a planted-violation fixture test. |
| Review log | `docs/phases/2.0/reviews/` | Simulated reviews + responses. |

## Acceptance criteria

- **AC-1 Layout** — built PDF passes `check_layout` (A4, page budgets, ≥11/10 pt, line numbers, confidential header, embedded fonts, metadata stripped).
- **AC-2 Anonymity** — `check_anonymity` passes on PDF text + source (system name per BLOCKED #4, org/hosts/author strings, first-person self-citation); manual pass recorded.
- **AC-3 Numbers** — no literal numbers in results/abstract outside generated macros or cited externals (`check_numbers`).
- **AC-4 Statistics** — every comparative claim: paired test, family correction, CI, n.
- **AC-5 Reproducibility (real path)** — (a) zero-shot on CPU for SNIPS (7 labels) or a fixed Banking77 subset via `serve/spot_check.py`, after timing one item; (b) McNemar vs the pinned public predictions; (c) 0.932 and 0.938 per-item dumps reproduced by the research session from the delivered branch.
- **AC-6 Bibliography** — every entry resolves (DOI / Anthology / arXiv) via `check_bib`.
- **AC-7 Calibration** — measured for zero-shot and 24-shot (vs the reproduction's Jev ECE/Brier) or the claim is removed; README correction proposal filed (push is must-ask).
- **AC-8 Protocol parity** — table complete per G4; differences matched by sensitivity runs or stated.
- **AC-9 Contamination** — per-model table with sources; Qwen3 handled per G6.
- **AC-10 Review** — ≥2 rounds of role-isolated reviewers (TACL persona ×2 + claims auditor) + ≥1 human reader; no open Critical/Major.
- **AC-11 Phase 1.2 gate** — recorded with one of three outcomes.
- **AC-12 arXiv build** — de-anonymised build, license chosen, endorsement status recorded.
- **AC-13 Submission packet** — PDF + source + Comments-to-the-Editor (author list, prior non-archival versions) + checklist.
- **AC-14 Selection provenance** — G11 outcome recorded; the paper's C3 wording matches it.
- **AC-15 Release statement** — anonymous text on code/data/adapter release (no links).
- **AC-16 AI-assistance disclosure** — present, anonymous, consistent with ACL policy.

## Decision authority for this phase

- **autonomous**: wording/structure within the plan, tooling details, analysis scripts, file layout.
- **log-and-proceed**: TeX engine; statistical procedures; dropping a claim that fails its gate; section changes; research-session requests.
- **must-ask**: outward-facing actions (public push, README/Space, adapter ungating, contacting third parties, arXiv posting, submission); author identity/affiliation and employer clearance; system-name decision for the review version; demoting 0.938 (G11); changing the primary venue (G13).

## Self-review record (independent reviewer, 2026-10-04 — REVISE → v2)

| Finding | Root fix applied | Status |
|---|---|---|
| C1 TACL forbids supplements and links (confirmed on transacl.org) | Supplement removed; AC-15 release statement; load-bearing protocol in main text; BLOCKED #2/#3 restated | applied |
| C2 0.938 selection provenance ungated | G11 + AC-14 + day-1 request; demotion must-ask | applied |
| W1 research requests incomplete/late | T0.5 day-1 request incl. 24-shot per-item dumps, adapter recipe | applied |
| W2 "24-shot" understates the data budget | thesis/C3 reworded; full-data baselines | applied |
| W3 missing ablations | G12 | applied |
| W4 parity table incomplete; "Jev's own protocol" | G4 extended; wording fixed | applied |
| W5 no 24-shot/Jev calibration | G1 extended | applied |
| W6 two number sources disagree | dumps = single source; `numbers.tex` | applied |
| W7 descriptions possibly iterated on test | G8 authoring log; blind re-author if needed | applied [unproven — measure first] |
| W8 splits not comparable | G7 split audit | applied |
| W9 desk-reject formatting | AC-1 extended | applied |
| W10 author info required at submission; employer clearance | BLOCKED #1 now blocks submission; BLOCKED #5 | must-ask |
| W11 public footprint vs double-blind | disclosure + BLOCKED #4 (system name) + anonymity scan scope | must-ask |
| W12 AI-assistance disclosure | AC-16 | applied |
| W13 novelty unchecked before drafting | G13 memo before T4 | applied |
| W14 CPU reproduction too heavy; CPU-120 ms claim wrong | AC-5(a) subset; C6/G9 | applied |
| W15 1.2 "not run" outcome | D4 third outcome + cut-off | applied |
| N1–N7 | title/page budget/copy-edit/colour-blind/self-overlap/human reader; tool fixtures; reuse `quorum.metrics`; SSoT fix in venue reference (TACL uses its own system; tier table moved to the reference); Qwen3 probe; SNIPS-descriptions win; README correction task | applied |

## Out of scope

Heavy model jobs on this shared box (research session); the NAACL Demo paper (later phase); Phase 1.2's own work.
