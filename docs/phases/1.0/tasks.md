# Phase 1.0 — Tasks (Quorum live demo Space)

> Spec: `spec_details.md`. Conversation log: `../../archive/conversation-phase1.0-hf-space-demo.txt`.

## T0 — Fact-first gate
- [x] T0.1 Verify CPU feasibility + latency of `ZeroShotEnsemble` (load 22 s, ~2–5 s small-label, RAM 3.3 GB;
  fits free CPU tier). → conversation DECISION #1.

## T1 — Build the Gradio app
- [ ] T1.1 `space/app.py` — inputs (message + candidate labels), module-level ensemble singleton (CPU),
  output = top label + confidence + ranked distribution + per-member breakdown; input guards.
- [ ] T1.2 `space/requirements.txt` — transformers / sentence-transformers / torch (CPU) / numpy + the
  `quorum` package (vendored copy under `space/quorum/` so the Space is self-contained, or pip-from-git).
- [ ] T1.3 `space/README.md` — Space card (sdk: gradio, title/emoji, links to repo + collection, honest note).

## T2 — Tests (one per AC)
- [ ] T2.1 (AC-F1/F3) defaults → valid distribution summing ~1, "card arrival" top.
- [ ] T2.2 (AC-F2) per-member breakdown shows 3 members' picks.
- [ ] T2.3 (AC-F4) <2 labels → friendly error; >20 labels → rejected.
- [ ] T2.4 (AC-P1) 6-label prediction < ~5 s warm on CPU.
- [ ] T2.5 (AC-I3) internal-trace scan of `space/` = clean.
- [ ] T2.6 (AC-I1/I2) Space boots on CPU tier + is listed in the Quorum collection (verified post-deploy).

## T3 — Deploy
- [ ] T3.1 Create HF Space `DataAgent/Quorum-Demo` (sdk gradio, CPU), push app + requirements + card.
- [ ] T3.2 Add the Space to the `Quorum` collection.
- [ ] T3.3 Link the Space from the GitHub README.

## T4 — Docs sync
- [ ] T4.1 Update spec/tasks statuses; append `[STATUS SUMMARY]` to the conversation file.
