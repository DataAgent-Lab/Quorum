# Addendum to the Jev retrieval-swap protocol (route change) — committed before any test-set call

**Base protocol:** `provenance/jev_retrieval_swap_protocol.md` (pre-registered; old hash 747a393 — see
`provenance/commit_hash_map.md`). Arms, inputs, analyses and the budget cap are **unchanged**. Only the route changes.

## Why

The Vercel AI Gateway route returned HTTP 403 ("free tier users do not have access to this model") and never served
a single call. The user switched to TypeSafe's own API (key in the gitignored `service/.env`).

## Route (replaces "Fixed settings → Route")

- Endpoint `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer <key>` — the endpoint the public
  reproduction used.
- Model: the request's `model` field is sent **unchanged** (`jev-1.13.0` in every reproduction request), and every
  response must report `model == "jev-1.13.0"`; any other served version aborts the run. This makes Arm A an exact
  replication of the reproduction's model version, not a drift check.
- Smoke test (train-split item `train-00019`, the reproduction's confirm-run request; no test data): HTTP 200, served
  `jev-1.13.0`, choice `card_arrival`, confidence 0.97, 77 probabilities summing to 1.0 — identical to the
  reproduction's recorded answer for that item. 3,658 input tokens.
- Runner: `scripts/jev/run_typesafe.py` (Python stdlib, replaces `scripts/jev/run.mjs` for this route). Same arms,
  same per-item construction (Arm B = the reproduction's request with `state.labeled_examples` replaced by the clean
  run's 24 retrieved examples, byte limit 30,000 by dropping from the end), interleaved A then B per item,
  concurrency 4 with ≥0.12 s between request starts (the reproduction's pacing), up to 3 attempts on 429/5xx/network,
  US$5 hard cap at $0.042 per million input tokens (output tokens are not billed per the reproduction's ledger).
- Raw responses are kept outside the repository; committed outputs are per-item `choice`, `probabilities`,
  `confidence`, served `model`, `usage`, attempts and the number of examples kept.
- TypeSafe terms: outputs are not used to train any model.
