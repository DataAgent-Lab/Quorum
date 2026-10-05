# Phase 2.0 — Claims ledger (fact-first gate)

> Rule: no number enters the manuscript unless its row is **VERIFIED** (recomputed here from the artifact) and the
> manuscript uses the generated macro. **REPORTED** = delivered by the research session with provenance, not
> independently recomputable here; the paper attributes it accordingly. **REFUTED** = an existing public claim the
> evidence contradicts (feeds the README correction list, T8.5).
> Sources: zero-shot per-item dumps `results/predictions/*` (main, commit 04e6709 config); 24-shot artifacts on
> `research/phase2.0-paper-evidence` (not yet merged); Jev reproduction `github.com/simonmesmith/jev-banking77-experiment`
> @ `5cac4ff` (read at run time, never redistributed).

## Zero-shot ensemble (generator: `paper/scripts/zero_shot_tables.py` → `paper/generated/{zero_shot.json,numbers_zeroshot.tex}`)

| ID | Claim | Value | Status |
|---|---|---|---|
| Z1 | Ensemble accuracy, 7 datasets × {names, descriptions}, full test sets | e.g. Banking77 75.6 / 77.4; SNIPS 85.3 / 94.4; MTOP 61.4 / 72.1 (macros `\zs<DS><Name|Desc>Acc`, 95% bootstrap CI `…AccLo/Hi`) | **VERIFIED** — equals dump meta exactly; equals README (rounded) |
| Z2 | Ensemble beats its best single member (family F1, 14 exact McNemar tests, Holm) | names **6/7** significant wins (SNIPS n.s., p_Holm 0.14); descriptions **6/7** (HWU64 n.s., p_Holm 0.14; SNIPS win p_Holm 0.046) | **VERIFIED** |
| Z3 | Descriptions beat names (family F2, 7 tests, Holm) | significant on **6/7**; largest MTOP **+10.7**, SNIPS +9.1, Bitext +4.5, MASSIVE +3.8, CLINC +3.0, Banking77 +1.8; HWU64 +1.9 **n.s.** (p_Holm 0.19) | **VERIFIED** |
| Z4 | "Calibrated" (README) | Raw ensemble ECE-10 **0.09–0.37**; **under-confident on all 14** (mean confidence < accuracy; Banking77 0.54 vs 0.76) | **REFUTED** as stated |
| Z4b | One task-agnostic temperature restores calibration | **Confirmatory (pre-declared `b93b37d`, run once):** one global T* = **0.508** fitted ONLY on the pre-registered train-side sample (14,000 rows, R12) → test ECE-10 falls on **14/14** conditions (Banking77 0.217→0.079, CLINC 0.288→0.039, MTOP 0.362→0.035, SNIPS-desc 0.095→0.023); bootstrap 95% CIs of raw vs T* do not overlap on any condition; Brier and log loss improve 14/14; accuracy unchanged by construction; per-label-text temperatures 0.505 / 0.511 (label text does not matter). The earlier test-side LODO analysis (exploratory, T 0.51–0.56) agrees. (`paper/scripts/calibration_confirm.py` → `paper/generated/calibration_confirm.json`) | **VERIFIED — confirmatory** |
| Z5 | Zero-shot Jev ≈ 0.801 | no citable source (the reproduction lists "definitions" 79.22% / "static" 81.82% on a 154-item screen only) | **GAP → DROP** unless sourced |
| Z6 | Where the under-confidence comes from (R3 member probabilities, verified: geo-mean identity ≤5.4e-7, member argmax == member_picks, other fields identical on 47,582 rows) | NLI member ECE-10 **0.01–0.14** and sharp (entropy 0.11–0.49 of uniform); the two bi-encoders are near-flat (entropy 0.48–0.82 of uniform) and under-confident (ECE **0.16–0.40**) → the geometric mean inherits their flatness; true product of experts (same argmax) ECE **0.04–0.16** | **VERIFIED** (`paper/scripts/combiner_ablation.py` → `paper/generated/combiners.json`) |
| Z7 | Combiner ablation (family F3, 28 exact McNemar, Holm) | geometric mean vs arithmetic mean: geo higher on 10/14, **significant on 5** (Banking77 ×2, HWU64 desc, MTOP desc, Bitext desc), arithmetic never significantly better; vs majority vote: geo higher on 14/14, **significant on 9**; per-item most-confident-member is worse than geo everywhere | **VERIFIED** |

## Banking77 retrieval-augmented 24-shot (research branch)

| ID | Claim | Value | Status |
|---|---|---|---|
| F1 | **Headline:** clean pre-registered configuration vs the Jev reproduction | **93.57%** (2,882/3,080) vs 92.40%; exact McNemar **b=119, c=83, p=0.0136** | **VERIFIED** — recomputed here from the dump + the reproduction's `predictions.csv` (truth labels matched on all 3,080 rows) |
| F1a | Pre-registration chain | protocol `8455e7e` (04:01) + addendum `f8f0476` (04:11) → run script `9363066`/fixes `2423006`/`8df80ae` → selection `d513aa9` (06:48, the only commit touching `selection_scores.json`; no test field) → single test run `09217ba` (07:22, the only commit touching the clean dump); `rerun=false` | **VERIFIED** (git history + file contents) |
| F1b | Selected configuration | {reader, kNN}, weighted probability mean, descriptions, 1 option order; out-of-fold 867/924; three candidates at 866 (disclose) | **VERIFIED** — winner recomputed from the 48 candidate scores with the declared tie-break |
| F2 | Secondary (pre-declared) | bootstrap 95% CI [92.69, 94.42]; macro-F1 0.9357; final-temperature ECE-10 **0.0105**, Brier 0.0995, log loss 0.255 vs Jev's reported 0.034 / 0.119 / 0.784 | **REPORTED** (meta); stored-probability ECE 0.0171 / Brier 0.0999 / log loss 0.255 **VERIFIED** here; final-temperature values to be recomputed by the paper's 24-shot generator |
| F3 | Repo default (class names, T=1, 3 orders, full pool) | 93.21% (2,871/3,080); vs Jev b=98, c=73, **p=0.066 — not significant** | **VERIFIED** |
| F4 | Clean vs repo default | b=57, c=46, p=0.32 — not significant (descriptive) | **VERIFIED** |
| F5 | Exploratory history (0.938) | cal-selected, not test-best, but ~40 configs scored on test; p=0.001428 survives Bonferroni only for m ≤ 35 | **VERIFIED** (R1 provenance + recomputation) — appears only as disclosed history |
| F6 | Prompt lengths / no truncation | clean test max 1,996 tokens; default max 1,802 (< 2,048) | **REPORTED** (meta) |
| F7 | Ablations (reader/kNN alone, base without LoRA, BM25 vote), parity retrieval, seed variance, latency | R5/R6/R8/R9 running (script `cce33f4`) | pending |

## Provenance facts the paper must state

| ID | Fact | Status |
|---|---|---|
| P1 | Adapter = byte-identical to the study checkpoint (sha256 `6ef3f783…c381`) | **REPORTED** (R1/R4) |
| P2 | Adapter training mixture: 14 tasks, train splits only; Banking77 train+test excluded (leakage gate exact + MinHash 0.8; 4 CLINC rows identical to Banking77 purged); includes CLINC150/HWU64/SNIPS/ATIS + topic/sentiment/emotion tasks → "public intent data only" is inaccurate | **REPORTED** (R4) |
| P3 | Exact-match overlap of eval test sets with the adapter mixture: **MASSIVE 318/2,974 (10.7%)** (311 via HWU64 train), MTOP 11/4,386, Banking77 0, Bitext 0 (lower bound) → MASSIVE is **not** a clean held-out set for adapter-based results; zero-shot results unaffected | **REPORTED** (R4) |
| P4 | Description authorship: six sets written by an LLM coding agent (Claude Opus 4.8, the research session) from label names only, one pass, measured once, never revised; Banking77 by an earlier agent session (model not recorded), hand-authored from HF label names, never edited; names-only test accuracies existed beforehand; per-class test errors not consulted for the six, unknown for Banking77 | **REPORTED** (R7, A3) — must be disclosed (AC-16 / G8) |
| P5 | Banking77 train/test: 25 normalized-text overlaps (reproduction's report) | **REPORTED** (reproduction) — disclose |

## README claims refuted or needing correction (→ T8.5, push is must-ask)

"calibrated" (Z4) · "on CPU in ~120 ms" (GPU number) · "0.932 … above the reproduced Jev" read as a win (F3 n.s.) ·
"significantly beat … 0.938 … survives Bonferroni" (F5) · "trained only on public intent data … Banking77 excluded"
(P2) · "full training pool" (9,079 in the study runs) · "Jev's own protocol" (the reproducer's selected setup) ·
"clean 24-shot sets MASSIVE/MTOP/Bitext" (P3) · zero-shot Jev ~0.801 (Z5).
