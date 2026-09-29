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
missing capability.

**CLINC150 (T2.3, full test 5500), 151 authored descriptions:**

| label text | accuracy | vs names | vs dataless 0.815 / PIE 0.831 |
|---|---|---|---|
| names | 0.6518 | — | −0.163 / −0.179 |
| **descriptions** | **0.6816** | **+0.0298** (McNemar p≈0) | −0.133 / −0.149 |

So the lever's *magnitude* depends on how opaque the names are: it **clears** the reference on SNIPS (7 opaque
names) but only **narrows** the gap on CLINC (151 fine-grained intents — a one-line description per class
recovers part, not all). The residual on CLINC is consistent with the encoder-specialisation lever (P2), for
which no usable open checkpoint exists (see AC-3). Honest scope: descriptions are a real, significant,
zero-new-code lever — not a universal fix.

## Acceptance criteria

- **AC-1 (P1 validated)** ✅: SNIPS full test, descriptions 0.9443 vs names 0.8529 (+0.0914, McNemar p≈0),
  above the dataless reference 0.9257; CLINC150 full test, descriptions 0.6816 vs names 0.6518 (+0.0298,
  McNemar p≈0) — significant on both, magnitude scales with name opacity. See Result above.
- **AC-2 (P1 documented)**: descriptions are a documented first-class option (README + a shipped example
  description set); the default stays names (zero-authoring) with descriptions as the "+quality" path.
- **AC-3 (P2)** ✅ (measured negative): PIE has no downloadable weights (training-code-only, archived); the
  only usable open specialised checkpoint, `sergioburdisso/dialog2flow-joint-bert-base`, was added as a 4th
  member and **significantly HURT** SNIPS names (0.8529→0.8350, McNemar p=0.0026) — its dialogue-action
  objective ≠ intent-name matching. No usable open intent-specialised encoder improves the ensemble; PIE-style
  specialisation is inaccessible without training our own encoder (out of scope). See T3 in tasks.md.
- **AC-4 (honest scope)**: no overclaim — descriptions/PIE are *known* levers; report where they help and where
  they don't; the cross-domain "ensemble beats its own parts" claim is unaffected.

## Out of scope
- Re-training an encoder ourselves (P2 is *use* an existing specialised encoder, not train one).
- Authoring descriptions for all 151 CLINC intents up front (validate on SNIPS first; roll out if it pays).
