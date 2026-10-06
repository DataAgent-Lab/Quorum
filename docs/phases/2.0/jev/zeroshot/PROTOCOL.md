# Pre-registered protocol: zero-shot Jev vs the zero-shot ensemble (Banking77 + MTOP)

**Status: pre-registration** — committed and pushed before any test-set call. Smoke tests used TRAIN items only
(MTOP train #123 with names and with descriptions; Banking77 train-00019 with the reproduction's definitions): all
returned `jev-1.13.0` with full probability vectors; ≈2.0–2.6k input tokens per call.

## Question

The paper's zero-shot results have no external reference point (the old "~80% zero-shot Jev" was a blog figure and
was dropped). How does the closed model do zero-shot on the same items and label text, and how calibrated is it?

## Arms (TypeSafe API, `POST https://api.typesafe.ai/v1/systemone`, model `jev-1.13.0` requested and asserted on every response)

| dataset (full official test) | arm | criteria (label → text) | instructions |
|---|---|---|---|
| Banking77 (3,080, 77 classes) | `names` | humanized class name → itself (what the ensemble's `names` condition sees) | generic (below) |
| Banking77 | `descriptions` | humanized name → our one-line description (`examples/banking77_descriptions.py`) | generic |
| Banking77 | `repro_definitions` | the reproduction's own definitions (`configs/descriptions.json` @ 5cac4ff, written from training data — NOT strictly zero-shot), raw labels | the reproduction's own instructions; = its "definitions" variant, run on the full test (it was only screened on 154 items, 79.22%) |
| MTOP en (4,386, 113 classes; clean for every ensemble member) | `names` | humanized name → itself | generic |
| MTOP | `descriptions` | humanized name → our one-line description (`examples/mtop_descriptions.py`) | generic |

Generic instructions (verbatim): ``Which ONE intent best describes `customer_message`? Every message belongs to exactly one of these categories.``
State: `{"customer_message": <test text>}`; no examples. The reproduction's files are fetched at run time and never committed.

## Settings

Concurrency 8, ≥0.12 s between request starts, up to 3 attempts on 429/5xx/network; a failed item counts as wrong.
Budget: hard cap US$5 for all arms (estimate ≈ US$1.7). Every call records the sha256 of the exact bytes sent; the
**full raw response** is saved (gzipped) alongside the per-item results in this folder.

## Pre-declared analyses (`paper/scripts/jev_zeroshot_analysis.py`, committed with this protocol)

1. **Primary (Holm over 4):** exact McNemar, Jev vs **our clean-NLI ensemble** (R13 `clean_c`, conservative — its NLI
   member has no exposure to either dataset) with the SAME label text: Banking77 × {names, descriptions},
   MTOP × {names, descriptions}.
2. Secondary: the same against the original ensemble (PrismNLI member; exposure-affected on Banking77, clean on MTOP).
3. Calibration: paired bootstrap (10,000 resamples, seed 20261012; ECE 2,000) of ours − Jev for per-item log loss
   (clip 1e-15), Brier, ECE-10 — ours = the ensemble's probabilities after its single train-side global temperature T*
   (`paper/generated/calibration_confirm*.json`; fitted without any test data). Both clean and original ensembles.
4. Descriptive: Banking77 `repro_definitions` accuracy and calibration (full-test value of the reproduction's
   154-item screen).
Expected (stated beforehand): Jev, a large model, probably leads zero-shot accuracy; all outcomes are reported.
