# Phase 1.2 — Train our own intent-specialised encoder (PIE-style): feasibility, benefit, and ship-or-not

> **Status**: PLANNED — not started (execution deferred; owner will trigger) · **Predecessor**: Phase 1.1.

## Why this exists

Phase 1.1 found two learnable levers behind the external zero-shot methods and measured both:
- **Descriptions (P1)** help significantly but their gain scales with name opacity — they *clear* the reference
  on SNIPS (+9.1) yet only *narrow* it on CLINC150 (0.652→0.682, still below dataless 0.815 / PIE 0.831).
- **A specialised encoder (P2)** is the other lever, but it is **inaccessible off-the-shelf**: PIE ships no
  downloadable weights, and the one usable open dialogue-specialised encoder (Dialog2Flow) *hurt* as a member
  (its utterance↔dialogue-action objective ≠ utterance↔intent-name matching).

So the ONLY path to the specialisation lever is to **train our own** intent-specialised bi-encoder. This phase
evaluates whether that is worth it, and ships it only if the benefit is real and regression-free.

## Hypothesis (to be tested, not assumed)

A small bi-encoder trained so that an utterance's embedding is pulled toward its **intent-name** embedding
(the exact cosine-to-label-text use in Quorum) — pretrained on public intent data with the eval targets held
out — added as a 4th ensemble member, will lift the fine-grained sets (CLINC/SNIPS, names AND descriptions)
without hurting Banking77/others, recovering part of the residual descriptions left.

## Feasibility axes (to establish in T1)

- **Method**: start with the SIMPLEST that could work — supervised contrastive (utterance, intent-name)
  positives with in-batch negatives on a strong small base (bge-base / MiniLM). Only escalate to PIE's full
  recipe (key-phrase tagger + pseudo-label contrastive, arXiv:2305.14827) if the simple version underperforms.
  Train the objective to match the SERVING use (utterance↔intent-name cosine), explicitly avoiding the
  Dialog2Flow failure mode (utterance↔utterance).
- **Data**: public intent corpora ONLY, with every Quorum eval target (Banking77, CLINC150, SNIPS, HWU64,
  MASSIVE, MTOP, Bitext) **held out**. Contamination-checked per [[feedback_check_zero_shot_training_lists]]
  (verify each source's overlap; mark CLEAN/UNVERIFIED before training on it).
- **Compute**: a sub-1B encoder, a few GPU-hours on the available box. Weights are ours → publishable.
- **License**: base-model license must permit redistribution of the fine-tune.

## Benefit gate + kill criteria (honesty-first — this is the point of the phase)

- **Ship IF**: as a 4th member it beats the current 3-member ensemble by a **significant** margin (paired
  McNemar, full test) on **≥2 datasets**, AND does **not** significantly hurt any dataset.
- **Do NOT ship (document as measured-negative, like Dialog2Flow) IF**: no significant gain, or any significant
  regression. A specialised encoder that only ties is not worth the extra model/latency.
- Target for a "clearly worth it" outcome: close **≥half** the CLINC residual (≈0.682 → ≈0.75+) without
  Banking77 regression. (Target, not a promise.)

## Evaluation protocol

Full official test sets only; paired McNemar vs the 3-member ensemble; names AND descriptions conditions;
targets held out of training; re-verify any surprising number. Same discipline as Phase 1.1
([[feedback_verify_before_claiming_results]]).

## Acceptance criteria

- **AC-1**: training data is public and contamination-checked, with all eval targets held out (documented).
- **AC-2**: benefit measured on the FULL test as a standalone member AND in the ensemble, with McNemar, vs the
  3-member baseline — reported honestly either way.
- **AC-3**: an explicit ship / don't-ship decision is made against the gate above and recorded; no silent ship.
- **AC-4**: if it ships — the encoder weights + a model card are published (reproducible), and it becomes an
  OPTIONAL ensemble member (default stays the 3-member; the specialised member is opt-in), consistent with
  [[feedback_ship_the_validated_best_not_a_floor]].

## Out of scope

- Large (>1B) encoders; retraining bge from scratch; using any eval-target training split; shipping on a
  tie (see kill criteria).
