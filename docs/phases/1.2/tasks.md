# Phase 1.2 — Tasks (train our own intent-specialised encoder)

> Spec: `spec_details.md`. Log: `../../archive/conversation-phase1.2-train-intent-encoder.txt`.
> **Status: PLANNED — not started.** Execution deferred until the owner triggers it.

## T1 — Feasibility spike (data + a first encoder)
- [ ] T1.1 Assemble public intent training data with ALL Quorum eval targets held out; contamination-check each
  source (card + any linked training list) and mark CLEAN/UNVERIFIED. Record counts + sources.
- [ ] T1.2 Train the SIMPLEST candidate first: supervised contrastive (utterance, intent-name) with in-batch
  negatives on a small base (bge-base / MiniLM). Objective = utterance↔intent-name cosine (the serving use),
  NOT utterance↔utterance (the Dialog2Flow failure mode). A few GPU-hours.

## T2 — Measure (standalone + as a member)
- [ ] T2.1 Standalone: the new encoder's zero-shot accuracy (cosine to label text) on SNIPS / CLINC150 /
  Banking77, full test, names + descriptions.
- [ ] T2.2 As a 4th ensemble member: full-test accuracy + paired McNemar vs the current 3-member ensemble, on
  each dataset, names + descriptions.

## T3 — Decision gate (ship or document-negative)
- [ ] T3.1 Apply the benefit gate (spec): ship only if significant gain on ≥2 datasets AND no significant
  regression; else record as a measured-negative (like Dialog2Flow). No silent ship.
- [ ] T3.2 If underperforming, decide whether PIE's fuller recipe (key-phrase tagger + pseudo-label
  contrastive) is worth a second attempt, or stop.

## T4 — Ship (only if T3 says ship)
- [ ] T4.1 Publish the encoder weights + a model card (reproducible); wire it as an OPTIONAL ensemble member
  (default stays 3-member); update README + examples; record the honest before/after.
