# Addendum 1 — extend the zero-shot Jev comparison to every clean dataset (committed before any call of these arms)

**Requested by the user (2026-10-06):** "請在乾淨資料集上跟 Jev 公平比" — compare with Jev on the clean datasets.
**Base protocol:** `PROTOCOL.md` (unchanged; its arms and analyses stay as declared).

## Added arms (same design as the base protocol)

| dataset (full test) | arms | label text |
|---|---|---|
| CLINC150 (5,500; 151 classes incl. out-of-scope as a class, the `plus` configuration) | `names`, `descriptions` | humanized names / `examples/clinc_descriptions.py` |
| SNIPS 7-intent (1,400) | `names`, `descriptions` | humanized names / `examples/snips_descriptions.py` |
| Bitext customer support (5,375; our stratified 20% test split — no official test split exists) | `names`, `descriptions` | humanized names / `examples/bitext_descriptions.py` |

"Clean" = no documented training exposure for any zero-shot ensemble member (`../../licences-contamination.md`):
together with MTOP these are the four clean datasets. Same model (`jev-1.13.0`, asserted), endpoint, generic
instructions, no examples, pacing, retries, raw-response saving and request manifest as the base protocol.

## Pre-declared analyses for the added arms

- **Family E (declared separately, Holm over 6):** exact McNemar, Jev vs the clean-NLI ensemble (`clean_c`), same label
  text, CLINC150 / SNIPS / Bitext × {names, descriptions}. Kept separate so the base protocol's primary family is not
  altered after it was declared. A **clean-datasets summary** (MTOP + these three = 8 tests) is reported descriptively.
- Secondary (vs the original ensemble — clean on these datasets too), calibration (paired bootstrap with each
  ensemble's train-side T*) and failure handling: as in the base protocol.

## Budget

Estimated 24,550 calls, ≈ US$1.9 (CLINC150 dominates: 151 criteria ≈ 3k tokens per call). Separate cap for these
arms: **US$3.5**; the whole zero-shot run (base + addendum) stays under US$6.

## Not added

A clean 24-shot Jev comparison on Bitext would need our own clean 24-shot system on Bitext, which does not exist
(the pre-registered 24-shot protocol covers Banking77 only) — out of scope for this addendum.
