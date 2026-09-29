# Phase 1.1 — Tasks (learn from PIE & dataless)

> Spec: `spec_details.md`. Log: `../../archive/conversation-phase1.1-learn-from-pie-dataless.txt`.

## T1 — Root-cause (done in spec)
- [x] T1.1 Analyse why PIE (intent-specialised encoder) and dataless (descriptions) beat Quorum's default on
  CLINC150/SNIPS → two learnable levers (P1 descriptions, P2 specialised encoder member). See spec.

## T2 — P1: descriptions as first-class label text
- [x] T2.1 Validate on SNIPS (full test): ensemble with per-class **descriptions** vs **names**.
  **RESULT (2026-09-29, SNIPS full 1400, cuda): names=0.8529 → descriptions=0.9443, delta=+0.0914;
  McNemar b=154 c=26 p≈0 → descriptions significantly better.** Descriptions not only close the gap to the
  dataless reference (0.9257) but clear it (+0.0186). AC-1 met. No new inference code — `label_texts` only.
- [x] T2.2 It helps → shipped `examples/snips_descriptions.py` (the exact 7 descriptions used) and documented
  descriptions as the "+quality" path in the README (default stays names = zero-authoring).
- [x] T2.3 Authored 151 disambiguating CLINC150 descriptions and MEASURED on the full test (5500):
  **names=0.6518 → descriptions=0.6816 (+0.0298); McNemar b=518 c=354 p≈0 → significantly better.**
  Descriptions help significantly on CLINC too, but — unlike SNIPS — do NOT close the gap to the external
  references (dataless 0.815 / PIE 0.831): a one-line description per class recovers only part of a 151-way
  fine-grained gap. The residual is consistent with the encoder-specialisation lever (P2), for which no usable
  open checkpoint exists (T3). Honest: descriptions are a real lever whose *magnitude* depends on name opacity
  (huge on SNIPS, modest on CLINC), not a universal fix.

## T3 — P2: intent-specialised encoder member (MEASURED — negative)
- [x] T3.1 Searched for an open intent-specialised encoder. **Findings:** (a) **PIE (arXiv:2305.14827,
  amazon-science/intent-aware-encoder, Apache-2.0) releases NO pretrained weights** — training code only, repo
  archived 2024-08; using it would require pre-training an encoder ourselves (out of scope). (b) The "dataless"
  method's encoder is `bge-large` — which we ALREADY have as a member; its edge is descriptions (= P1), not a
  special encoder. (c) The only usable open English dialogue/intent-specialised, ST-compatible checkpoint found
  is `sergioburdisso/dialog2flow-joint-bert-base` (EMNLP'24, MIT).
- [x] T3.2 Added Dialog2Flow as a 4th ensemble member and MEASURED on SNIPS names (full 1400):
  **3-member 0.8529 → +Dialog2Flow 0.8350 (delta −0.0179); McNemar b=20 c=45 p=0.0026 → the 3-member is
  SIGNIFICANTLY better.** It HURTS. Root cause (as predicted): Dialog2Flow optimises utterance↔dialogue-action
  similarity, not utterance↔intent-name matching, so its cosine-to-label-text is a noisy member. **Verdict:
  no usable open intent-specialised encoder improves the ensemble; the specialisation lever (PIE) is
  inaccessible without training an encoder ourselves (out of scope).** AC-3 satisfied as a measured negative,
  AC-4 honoured (honest where it doesn't help).

## T4 — Docs
- [ ] T4.1 Record results (spec/tasks/conversation); update README if P1 ships; keep the "known levers, honest
  where they help / don't" framing (AC-4).
