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
- [ ] T2.3 (optional, if it pays) author + measure descriptions for CLINC150 (151) — the biggest gap.
  Deferred: SNIPS already proves the lever; CLINC's 151-intent authoring is a larger effort, tracked as a
  future direction rather than done speculatively.

## T3 — P2: intent-specialised encoder member (spec only unless a checkpoint is found)
- [ ] T3.1 Find whether a suitable OPEN intent-specialised encoder exists (PIE checkpoint on the Hub, or a
  strong intent embedder). Record candidates + licenses.
- [ ] T3.2 If found: add it as an optional ensemble member (same cosine-to-label-text interface as the bge
  members) and measure on CLINC150/SNIPS. If not found: leave as a documented future direction.

## T4 — Docs
- [ ] T4.1 Record results (spec/tasks/conversation); update README if P1 ships; keep the "known levers, honest
  where they help / don't" framing (AC-4).
