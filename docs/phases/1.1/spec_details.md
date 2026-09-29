# Phase 1.1 — Learn from PIE & dataless (why they win, and folding it into Quorum)

> **Status**: P1 done (AC-1/AC-2 met), P2 speced (AC-3) · **Projected-date**: 2026-09-29 · **Predecessor**: Phase 1.0.

## Root-cause analysis (why two zero-shot methods beat Quorum's default on CLINC150 / SNIPS)

| method | what it is | root cause of its edge |
|---|---|---|
| **PIE** (arXiv:2305.14827) | a bi-encoder *pre-trained to be intent-aware* — contrastive pretraining that pulls an utterance's embedding toward its intent-label embedding, on other intent data; zero-shot = embed utterance + label names, cosine | **encoder specialisation** for utterance↔intent matching (Quorum's bge members are general-purpose) |
| **dataless** (arXiv:2407.17862) | bge-large + per-class natural-language *descriptions*, cosine-matched (no examples) | **richer label text** — descriptions carry far more signal than bare names |

Quorum's default: general-purpose members (PrismNLI + bge) + **label names only**. So on label-rich / naming-
opaque sets it trails: CLINC150 0.652 (vs PIE 0.831 / dataless 0.815), SNIPS 0.853 (vs dataless 0.9257). This
is expected, and **both edges are known, learnable levers**, not a mysterious architecture.

## Principles to learn in

- **P1 — Descriptions as first-class label text.** `ZeroShotEnsemble.predict_proba(..., label_texts=…)` and the
  `serve` API already accept arbitrary label text, so a per-class *description* needs **no new inference code** —
  it needs: (a) enabling/documenting descriptions as the recommended path when they exist, (b) shipping example
  descriptions, and (c) **measuring** how much of the PIE/dataless gap descriptions close.
- **P2 — An intent-specialised encoder member (PIE-style).** Add an optional ensemble member that is an
  intent-specialised open encoder (PIE itself if a suitable checkpoint is public, else a strong intent embedder),
  since specialisation — not size — is what wins on these sets.

## Result (T2.1, 2026-09-29)

SNIPS full test (1400), same ensemble, `label_texts` = per-class descriptions vs bare names, on cuda:

| label text | accuracy | vs names | vs dataless ref (0.9257) |
|---|---|---|---|
| names (default) | 0.8529 | — | −0.0728 |
| **descriptions** | **0.9443** | **+0.0914** | **+0.0186** |

McNemar (descriptions vs names): b=154, c=26, p≈0 → descriptions **significantly** better. Descriptions close
the SNIPS gap to the dataless method *and clear it*, with **no new inference code** — the same ensemble, only
richer `label_texts`. This confirms P1: the dataless edge is a label-text lever Quorum already supports, not a
missing capability. (SNIPS is a favourable case — 7 opaque names; the CLINC 151-intent case is unmeasured, T2.3.)

## Acceptance criteria

- **AC-1 (P1 validated)** ✅: SNIPS full test, descriptions 0.9443 vs names 0.8529 (+0.0914, McNemar p≈0),
  above the dataless reference 0.9257. See Result above.
- **AC-2 (P1 documented)**: descriptions are a documented first-class option (README + a shipped example
  description set); the default stays names (zero-authoring) with descriptions as the "+quality" path.
- **AC-3 (P2 speced)**: a concrete, sourced plan to add an intent-specialised encoder member (candidate model,
  how it plugs into the ensemble, how to evaluate) — implemented only if a suitable open checkpoint is found.
- **AC-4 (honest scope)**: no overclaim — descriptions/PIE are *known* levers; report where they help and where
  they don't; the cross-domain "ensemble beats its own parts" claim is unaffected.

## Out of scope
- Re-training an encoder ourselves (P2 is *use* an existing specialised encoder, not train one).
- Authoring descriptions for all 151 CLINC intents up front (validate on SNIPS first; roll out if it pays).
