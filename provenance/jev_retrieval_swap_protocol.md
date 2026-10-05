# Pre-registered protocol: Jev with the open stack's retrieval (Banking77, single run per arm)

**Status: pre-registration.** Committed before any Jev call under this protocol. The only call made so far was a
smoke test on ONE train-split item (the reproduction's `runs/confirm-retrieved24/train-00019/request.json`); it
failed with HTTP 403 (gateway billing tier) — no Jev output exists yet.

## Question

The clean pre-registered run beats the public Jev reproduction (93.57% vs 92.40%, McNemar p = 0.0136), but the R6
parity runs show the accuracy gain comes from the retrieval rule. Does Jev gain the same from that rule? Only running
Jev with our retrieved examples answers it.

## Arms (same model, same time window, same requests except the examples)

- **Arm A — replication.** For each of the 3,080 test items, send the reproduction's own request **unchanged**
  (`runs/test-v1/test-NNNNN/request.json` @ `5cac4ff`: its instructions, its 77 class definitions, its 24
  retrieved examples). Measures today's Jev against the published 92.40% (model-version drift check).
- **Arm B — retrieval swap.** The same request with `state.labeled_examples` replaced by the clean run's 24 retrieved
  training examples for that item (`results/predictions/banking77_24shot_clean.jsonl.gz`, field `retrieved_idx`,
  in the recorded order), each as `{"message": <train text>, "intent": <pinned dataset label name>}` (the
  reproduction's example format). If the payload exceeds the reproduction's 30,000-byte limit, examples are dropped
  from the end until it fits (the reproduction's rule); the count kept is recorded per item.

The reproduction's files are fetched at run time into a cache outside the repository; the repository has no licence,
so nothing from it is committed. Only our derived per-item outputs (prediction, probabilities, usage) are written.

## Fixed settings

- Route: Vercel AI Gateway, model `typesafe-ai/jev`, AI SDK `experimental_evaluate` (`ai` 7.0.127, pinned in
  `scripts/jev/package.json`). The served model id/version returned by the gateway is recorded per call; the
  reproduction used `jev-1.13.0`.
- Concurrency 4; up to 3 attempts per item on retryable errors (HTTP 429/5xx, network); a failed item is recorded as
  failed and kept in the denominator as wrong (the reproduction's convention: 0 failures there).
- Budget: hard cap US$5 for both arms together (the reproduction reports US$0.44 per full test run).
- Arms run interleaved by item (A then B for each item) so both see the same model state.

## Pre-declared analyses (nothing selected after seeing results)

1. **Primary:** exact McNemar, clean run vs **Arm B** (equal retrieval) — model-level accuracy comparison.
2. Secondary: exact McNemar, Arm B vs Arm A — effect of the retrieval rule on Jev itself.
3. Secondary: Arm A accuracy vs the published 2,846/3,080 (exact McNemar vs the published per-item correctness) —
   version drift.
4. Secondary: paired bootstrap (10,000 resamples, seed 20261011) of ours − Jev-B per-item log loss (clip 1e-15) and
   Brier, and ECE-10 difference (2,000 resamples) — the same definitions as `paper/scripts/jev_paired_calibration.py`.
Whatever the outcome, all four are reported.
