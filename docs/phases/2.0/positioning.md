# Paper positioning after the open decision models (2026-10-11)

> Agreed between this session and the research session. **USER decisions (2026-10-11): P1 adopted; venue TMLR; GO for Clef/Clef-Flash b24B on any idle GPU; GO for Bitext as the second 24-shot dataset with Jev ≤ US$1 (one arm); de-identification of internal findings approved.** Pending: whether the 13-task code/workflow comparison joins this paper (recommended — see below). Numbers:
> `claims-ledger.md` (F1–F15, J1–J3, K1–K5, L1–L2). Internal deployment details may enter the paper only with the
> author's explicit go, de-identified.

## Why the positioning must change

| Banking77 24-shot, identical evidence | accuracy | ECE / Brier |
|---|---|---|
| reproduction's retrieval: Clef-Flash · Clef · pplx · Jev · ours¹ | 95.49 · 94.32 · 94.19 · 92.40 · 92.34 | 0.176/0.127 · 0.066/0.093 · 0.015/0.092 · 0.034/0.119 · 0.024/0.116 |
| our retrieval: pplx · ours · Jev (Clef pending) | 94.64 · 93.57 · 93.47 | 0.016/0.084 · 0.017/0.100 · 0.029/0.105 |

*(Jev under the reproduction's retrieval corrected 2026-10-11 to 0.034/0.119 — the table previously repeated its Arm-B values.)*

¹ ours with the reproduction's full retrieval rule over all 10,003 rows (F13), not our clean run.
Scoring provenance: pplx b24 release code (A6), pplx b24B parity-gated SGLang (A7); Clef b24 release code (R14).

→ "Quorum as a competitive system" is no longer viable: three open 9–27B decision models beat it on accuracy, and
pplx matches its ECE and beats its Brier. Zero-shot, every decision model is far ahead (with exposure caveats).

## Agreed direction (P1): an evaluation-methodology paper on the "decision model" class

**Contribution = the protocol**, then the findings it produces. Quorum is the low-cost open baseline and testbed.

- **Protocol:** identical, sha256-pinned request bodies; per-item paired exact tests with Holm; pre-registration
  before every scored run; a serving-engine parity gate; over-length handling that follows each release code;
  exact replication of a closed API's published result (Jev 92.40%).
- **Findings (measured):**
  1. **Retrieval matters as much as the model.** Uncapped retrieval gives Jev +1.07 pt, pplx +0.45 n.s. (Clef
     pending). Pool-size curve for pplx (b24B recipe): ~9k pool 94.51 · 24/class 92.11 · 8/class 89.77 ·
     24/class names-only 91.79 · zero-shot 80.7; with a strong decision model, retrieval-augmented few-shot from a
     small labelled pool beats training a small head on the same labels (0.844–0.888 at 24/class) — one dataset,
     one random draw.
  2. **Accuracy ≠ calibration:** Clef-Flash is the most accurate and the worst calibrated.
  3. **Undisclosed or exposed training data makes zero-shot numbers uninterpretable** (Clef undisclosed; pplx's
     manifest lists MASSIVE/MTOP/SNIPS).
  4. **Metric pitfalls:** Jev's 0.01 rounding invalidates log loss (J2).
  5. **Reproducibility hazards:** platform-dependent BM25 tie-breaking (F10/F14); serving-engine parity
     (one model failed the gate on 1/3,080 items); serving input limits differing from release limits (context −
     6 tokens silently refused prompts the release accepts); over-length policies differing by model, which
     flipped one task's ranking. *(The serving-engine findings come from internal deployment work → USER go +
     de-identification required.)*
  6. **The self-caught traps**, incl. contamination via model lineage (P6).
- **Counter-argument to answer:** "a benchmark of other people's fast-moving models on one dataset". Answer:
  the protocol is the contribution; widen the evidence; ship results as a versioned leaderboard-style appendix;
  submit sooner rather than wait for full coverage.

## Essential experiments, in order (each needs USER go; GPU is the bottleneck)

1. **Clef / Clef-Flash b24B** (pre-registered as A4) — without it finding 1 rests on two models. GPU: an idle GB10
   (≈1.5 s/item for Clef-Flash on the release code) or a short maintenance swap on the production host.
2. **ECE/Brier for every model on identical evidence** — done except Clef b24B (filled by item 1).
3. **A second 24-shot dataset** — Bitext (clean for our adapter; pplx/Clef need GPU; Jev needs paid calls and
   USER approval; never publish spend — MCA §14.1). Our own 24-shot system on Bitext also needs a pre-registered
   selection. MTOP is a fallback but exposed for pplx.
4. **Latency/throughput on one hardware type** — consistent single-engine numbers for the three open models exist
   from internal work (USER go + de-identification); Jev's API latency is not comparable.
5. **Pool-size curve for ≥2 models** (pplx done).

**Revised recommendation (2026-10-11): include** the 13-task code/workflow comparison of the three open models (pplx +9, Clef −1, Clef-Flash −8 Holm wins − losses) as a second evidence block: it answers the main counter-argument (one dataset), several methodology findings come from it, and TMLR has no page limit. Pending: its protocol/pre-registration location, dataset licences, per-item dumps for independent verification, and the author's final OK.

## Venue (USER)

| venue | fit for P1 | trade-off |
|---|---|---|
| **TMLR** | correctness + interest criterion fits an evaluation-methodology paper; rolling → can submit soon | recognition Tier 2 (weak evidence); its LLM policy allows LLMs as assistive tools with full author responsibility |
| TACL | possible | Tier 1, but reviewers may ask "what is the NLP contribution"; strict ACL AI-writing rules (dictation workflow) |
| an evaluations / datasets-and-benchmarks track | good topical fit | fixed annual deadlines (e.g. NeurIPS 2027 spring) |

Constraints carried everywhere: never tune on Jev outputs; do not publish Jev spend; no internal hosts or company
names in public files.
