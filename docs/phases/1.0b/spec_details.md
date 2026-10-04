# Phase 1.0b — Demo page: real benchmark examples, the verdict reveal, and cold-start handling

> **Status**: built + tested (AC-1…AC-9 PASS); awaiting go-live approval · **Projected-date**: 2026-10-06 · **Predecessor**: Phase 1.0a (cache + lazy worker)
> **Conversation**: `docs/archive/conversation-phase1.0b-demo-real-datasets.txt` · **Prototype**: `prototype/`
> **Go-live**: publishing the Space needs the user's approval in this session (and an HF write token — not on the
> demo host).

## Goal
Every example on the demo page is a real benchmark test sentence answered instantly from the permanent cache, the
result reveal feels refined, and a visitor who triggers a cold model load is told what is happening and gets an
answer instead of a timeout.

## Design (agreed with the user, 2026-10-04)
- One message box + one labels box. Chip row = the 7 benchmark domains (friendly title + dataset · label count)
  + "Your own labels". A domain chip fills the same boxes with a random real test sentence (🎲 for another one);
  default on load = Banking + a random sentence. Domain labels are shown as a one-line summary (expandable list).
- Labels sent to the API = the **description** variant (exact strings from the research dumps' meta); results show
  name (normal) + description (small). Descriptions beat names on all 7 datasets.
- "Your own labels": editable labels textarea, default = the former "Support triage" preset (seeded → instant).
- Result: 3 juror cards flip in → verdict label + seal stamp with counting confidence → ranked bars (top 5,
  expandable) → dataset-answer line (✓ / ✗ + the real answer + the domain's full-set accuracy) → confidence gate.
- Visual language from the prototype ("The Verdict"): paper/ink + one vermilion accent; Instrument Serif / Hanken
  Grotesk / IBM Plex Mono; light + dark; mobile; `prefers-reduced-motion` respected.
- The page states that visitors' own sentences are kept on the demo server (user decision in 1.0a).

## Data — `space/samples.json`
Built by `python -m serve.export_demo_samples` from `results/predictions/*` (single source: the same strings that
were seeded). Per dataset: key, title, dataset name, `n_test`, `accuracy` (descriptions, from meta), labels
`[{name, desc}]` (the desc strings are exactly what is sent), and a random sample of test sentences `{t, g}` (text
as cached = stripped; gold index). Bitext sentences containing `{{…}}` template placeholders are excluded (they read
as broken; editing them would break the cache key). Samples per dataset: 300 (deterministic seed).

## API wiring + cold-start handling (the user's explicit ask)
- `POST /predict {message, labels}` with `labels` = the description strings; header `X-Quorum-Cache` decides the
  footnote ("answered from the cache" / "computed live").
- **Known-cached requests** (an unedited domain sample, or the default own-labels example) → fetch with a 30 s
  timeout; no special waiting UI.
- **Requests that may need live computation** (edited sentence, own labels): before sending, `GET /health`; if
  `model_loaded` is false, the deliberating panel says the jury is waking up (models load on demand, the first
  answer can take ~1–2 minutes) and shows an elapsed-seconds counter. Fetch timeout 180 s (the server's cold-start
  budget is 300 s, typical 30–90 s).
- **Recovery**: on a network timeout, 503 (busy / waking) or 504, retry once after 3 s — the server stores a finished
  result, so the retry usually hits. If the retry also fails, show a calm error with a "Try again" button.
- 422 (e.g. > 50 labels live) cannot normally happen (the UI blocks it first); if it does, show the server's message.
- `?api=<url>` query parameter overrides the API base (for self-hosters and local testing); default unchanged.

## Acceptance criteria
| # | Criterion | Measurement |
|---|---|---|
| AC-1 | Default load shows Banking + a real test sentence; "Hear the verdict" returns a real cached verdict (`hit`) with name + description, ✓/✗ against the dataset answer, the domain accuracy line | browser E2E (real API) + screenshot |
| AC-2 | Each of the 7 domains: chip selects it, 🎲 draws another real sentence, the verdict is a cache `hit` with K = the domain's label count | browser E2E, all 7 |
| AC-3 | Editing a sentence in a > 50-label domain disables the button with an explanation; in SNIPS/Bitext (≤ 50) an edited sentence is computed live | browser E2E |
| AC-4 | Cold start: with the model unloaded, an own-labels request shows the waking-up message + elapsed counter and ends in a verdict (no client timeout) | browser E2E against a cold local API |
| AC-5 | Recovery: a failed first attempt (503/504/timeout) is retried once automatically; a second failure shows a retry button | browser E2E with a failing API stub |
| AC-6 | Visuals: desktop light/dark + mobile render without overflow; reveal choreography plays; reduced-motion shows the final state immediately | screenshots (desktop light, dark, mobile) |
| AC-7 | No console errors; keyboard reachable chips/buttons with visible focus; result panel is `aria-live` | browser E2E |
| AC-8 | `samples.json` is generated from the dumps: every sample's (text, description labels) is a cache hit on the live API (spot-check 7 × 5 via HTTP) | script check |
| AC-9 | The seeded own-labels default and the drift test follow the new page (no stale preset list) | unit test |
