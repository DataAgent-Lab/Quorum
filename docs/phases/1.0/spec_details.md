# Phase 1.0 — Quorum live demo (Hugging Face Space)

> **Status**: in progress · **Projected-date**: 2026-09-29
> **Goal**: an interactive Space where anyone can try Quorum's zero-shot decision service on their own text.
> **Predecessor**: the shipped zero-shot ensemble (`quorum.ZeroShotEnsemble`) and 24-shot pipeline.
> **Linked-code**: `quorum/ensemble.py`, `space/app.py` (new).

## Context

Quorum's zero-shot service is a geometric-mean ensemble of three sub-1B open models (PrismNLI-0.4B +
bge-large + bge-base). To let people *feel* it — and to anchor the write-up with a hands-on link — we publish
a Gradio Space under the `Quorum` collection. The demo serves the **zero-shot** tier (CPU-friendly). The
24-shot pipeline is out of scope here (its gated Qwen3-4B + LoRA adapter is too heavy for a free CPU Space).

## Verified facts (fact-first gate, 2026-09-29 — see conversation DECISION #1)

- `ZeroShotEnsemble(device="cpu")`: model load ~22 s (once at boot); peak RAM ~3.3 GB (fits free CPU 16 GB).
- Latency (CPU): ~1.8–2.4 s for 5 candidate labels, ~4.6 s for 20; ~15 s+ for the full 77 (one NLI pair/label).
- ⇒ Target the **free CPU tier**; UX uses a **small candidate-label set** (default ~6), warns above ~15.

## Development items

1. **`space/app.py`** — a Gradio app:
   - Inputs: `message` (textbox); `candidate labels` (one per line, default ~6 illustrative banking intents).
   - Loads `ZeroShotEnsemble(device="cpu")` once at import (module-level singleton).
   - Output: (a) top label + calibrated confidence; (b) the full ranked distribution (Gradio `Label`);
     (c) a per-member breakdown (each of the 3 members' top pick) so users see the "jury" voting.
   - Guardrails: cap labels (reject > 20, warn > 15); require ≥ 2 labels; strip blank lines.
2. **`space/requirements.txt`** — `quorum` deps (transformers, sentence-transformers, torch CPU, numpy) + the
   `quorum` package itself (installed from the GitHub repo, or vendored — see task).
3. **`space/README.md`** — the Space card (YAML: `sdk: gradio`, `title`, `emoji`, `pinned`, links to the repo
   + collection), with an honest note: zero-shot demo; not open zero-shot SOTA; 24-shot needs the gated adapter.
4. **Deploy**: create HF Space `DataAgent/Quorum-Demo` (sdk gradio, CPU), push the three files, **add it to the
   `Quorum` collection**.

## Acceptance criteria

**Functional**
- AC-F1: given a message + ≥2 labels, the app returns a probability distribution over exactly those labels that
  sums to ~1.0, plus the argmax label and its confidence.
- AC-F2: the per-member breakdown shows each of the 3 members' top label.
- AC-F3: sensible output on a clear case — e.g. message "when will my card arrive?" with labels including
  "card arrival" ⇒ top label "card arrival".
- AC-F4: input guards — < 2 labels or empty message ⇒ a friendly error, not a crash; > 20 labels ⇒ rejected
  with a message.

**Performance**
- AC-P1: with ≤ 6 labels, a prediction returns in < ~5 s on the free CPU tier (after warm-up).

**Integration**
- AC-I1: the Space boots on the free CPU tier (peak RAM < 16 GB; ensemble loads once at startup).
- AC-I2: the Space is listed in the `Quorum` collection (DataAgent/quorum-…).
- AC-I3: the Space card links back to the GitHub repo and the collection; no private-infrastructure references.

**UX**
- AC-U1: defaults are pre-filled (a message + ~6 labels) so a visitor gets a result in one click.
- AC-U2: the UI states this is the zero-shot tier and links to the 24-shot adapter for the stronger pipeline.

## Test cases (real environment)

- T1 (AC-F1/F3/U1): load the app, click Submit on the defaults ⇒ valid distribution, "card arrival" top.
- T2 (AC-F2): breakdown shows 3 members' picks.
- T3 (AC-F4): 1 label ⇒ friendly error; 25 labels ⇒ rejected.
- T4 (AC-P1): time a 6-label prediction on CPU ⇒ < ~5 s warm.
- T5 (AC-I1/I2/I3): Space builds + runs on CPU tier; appears in the collection; card links correct; trace-scan clean.

## Out of scope
- The 24-shot pipeline in the live demo (gated adapter, too heavy for free CPU).
- Any paid GPU Space tier (revisit if demand warrants).
