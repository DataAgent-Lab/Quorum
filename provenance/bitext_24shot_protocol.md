# Pre-registered protocol: Bitext 24-shot, the open stack and four decision models on identical evidence

**Status: pre-registration.** This file is committed before any model is run under it. Until §10's code is
committed, nothing is scored on the selection split or on the test set: no model run of any kind. The only work
done so far is the read-only pin lookups in §1.

Purpose: a second 24-shot dataset for the paper's equal-evidence comparison (R16), answering "one dataset"
(Banking77). It mirrors `banking77_24shot_clean_protocol.md` for the open stack (§2–§7) and the Arm-B
equal-retrieval design (`jev_retrieval_swap_protocol.md`, R14/R15 `b24B`) for the decision models (§8).

## 1. Pins (looked up read-only before this commit)

| item | pin |
|---|---|
| Dataset | `Bitext/Bitext-customer-support-llm-chatbot-training-dataset` @ `430d1a89bd93bd1fa23c16f29dd53e73f0087443` (current head, 2024-07-18; licence CDLA-Sharing-1.0); 26,872 rows, 27 intents |
| Split | this repo's loader (`quorum/data.py`, `bitext`): no official test split, so a deterministic stratified 20 % test per intent (`np.random.default_rng(20250926)`); the rest is `T` |
| Open stack | the same pins as the Banking77 clean protocol §1: Qwen3-4B @ `1cfa9a72…`, adapter @ `e0765b76…`, bge-large, zero-shot ensemble defaults. The adapter's training mixture contains no Bitext data, so Bitext is clean for it (`provenance/adapter_training.md`) |
| Decision models | Jev `jev-1.13.0` (TypeSafe API); Clef @ `2f3de3d`, Clef-Flash @ `17f0b0a` (release code); pplx-decider v1.1 @ `3b45dea` (the A7 parity-gated SGLang engine) |

**Stated plainly:** the test set is our own 20 % stratified split. Bitext publishes no test split.

## 2. Selection split for the open stack (fresh, stratified)

1. `T` = the loader's train rows in loader order, indexed 0..|T|−1.
2. `S`: 12 rows per intent (27 × 12 = 324), drawn with one generator `rng = np.random.default_rng(20261012)`. For
   each intent label `c` in ascending order, take the indices of `T` with label `c`, sorted ascending, permute them
   with `rng.permutation`, and keep the first 12.
3. Pool `P = T \ S`. The same pool is used for selection scoring, the test run, and every decision-model body.
4. The run script writes the index list of `S` to `results/bitext24/selection_split_idx.json` before any model is
   loaded.

## 3. Fixed settings for the open stack (declared, not searched)

As Banking77 clean §3. The only changes are the dataset and the reader task description:
"Classify the customer's support message into the intent it expresses."

Unchanged from Banking77 clean §3:
- shots 24;
- BM25 (k1 1.5, b 0.75, `quorum/fewshot.py`), top 24 from `P`, the same 24 for the reader and kNN;
- kNN = bge-large, max per class, scale 20, floor −1;
- answer scoring by option-marker logits;
- order seeds `random.Random(1000 + i)`;
- max length 2,048, batch 8, bf16.

## 4. Candidate grid (48) and §5 fitting, §6 selection rule

Identical to Banking77 clean §4–§6:
- **Grid:** member sets M2/M3, × 4 combination rules, × 2 option texts, × 1/3/7 orders.
- **Option texts:** names are this repo's humanized intent names; descriptions are `examples/bitext_descriptions.py`
  (generic one-line rephrasings authored without the data).
- **Fitting:** 5-fold stratified cross-fitting on `S` (fold seed 20261013).
- **Selection:** by out-of-fold correct count, with the same total tie-break order.

## 7. The single test run (open stack)

Only the selected candidate runs, once, on all test rows in loader order, with retrieval from `P`.

Per-item dump `results/predictions/bitext_24shot_clean.jsonl.gz`:
`{idx, text, gold, pred, probs[27], reader_probs, knn_probs, zs_probs (M3 only), retrieved_idx[24], perm_reader_probs}`.

The meta holds the commit hashes, pins, configuration, fitted parameters, versions, device and wall time.

## 8. Decision-model arms: identical request bodies, built once

**Body for test item i:**

```
{"model": "jev-1.13.0",
 "state": {"customer_message": <test text>,
           "labeled_examples": [{"message": <T text>, "intent": <raw intent name>} for the BM25 top 24 from P]},
 "questions": {"intent": {"type": "choice",
                          "instructions": "Which ONE intent best describes the customer message? The labeled examples show messages with their intents. Every message belongs to exactly one of the listed intents.",
                          "criteria": {<raw intent name>: <description>, … all 27, sorted by raw name>}}}}
```

**Example format.** The examples use the same format as the Banking77 Arm B bodies (`message` / `intent`). The
example list is the same top-24 list as the open stack's `retrieved_idx`, in rank order.

**Criteria.** The descriptions are `examples/bitext_descriptions.py`, keyed by humanized name and mapped to the raw
intent name.

**Size cap.** Examples are dropped from the end until `json.dumps(body, ensure_ascii=True)` is ≤ 30,000 bytes. This
is the reproduction's rule, and the kept count is recorded.

**Model field.** `"model": "jev-1.13.0"` is sent to every model, so the bodies are byte-identical. The open models
ignore the field.

**Built once.**
- `scripts/bitext24_bodies.py` (committed per §10) builds every body before any model call.
- It writes `results/bitext24/request_manifest.jsonl.gz`, one line per test item:
  `{idx, sha256, bytes, examples_kept}`. That file holds hashes only; the bodies are rebuilt deterministically.
- The manifest's own sha256 is recorded in `results/bitext24/manifest_meta.json` and committed before any arm runs.
- Every arm rebuilds each body, checks its sha256 against the manifest, and aborts on any mismatch.

**Arms.** One scored run per model.

| arm | run by | stack |
|---|---|---|
| Jev | the paper session | TypeSafe API, served model `jev-1.13.0` recorded per call, concurrency 4, ≤ 3 attempts on 429/5xx/network; budget cap US$1 (≈ 5.4 k calls; not published) |
| Clef-Flash 9B and Clef 27B | the research session | release code on one GB10, batch 8 / 4, with the A1 memory watchdog (`clef_eval_protocol.md` A1) |
| pplx-decider v1.1 | the research session | the A7 parity-gated SGLang engine on one GB300, concurrency 2 |

Rules for every arm:
- **Failures.** A failed item counts as wrong; there are no re-runs of completed arms.
- **Probabilities.** Scores come from each model's full probability vector over the 27 intents.
- **Length limits.**
  - The pplx release refuses questions over 8,192 tokens, and such items would be unanswered (= wrong).
  - Clef truncates the state to 16,384 tokens.
  - Expected: none at the median Bitext body size. The count is computed with each tokenizer before any call and
    reported.

## 9. Pre-declared analyses (all reported, whatever the outcome)

1. **Primary — accuracy.** Exact two-sided McNemar for every pair of the five systems on identical test items: the
   open stack, Jev, Clef, Clef-Flash and pplx, 10 pairs. Holm correction over the 10.
2. **Calibration.** ECE-10 (probability of the predicted intent) and Brier (summed over the 27 intents). Each model
   is compared with the open stack by paired bootstrap: 10,000 resamples, seed 20261012; 2,000 for ECE.
   - Log loss is not compared with Jev, because Jev rounds its probabilities to 0.01 (ledger J2).
3. **Descriptive.** Accuracy, macro-F1, and per-intent accuracy for each arm; the examples-kept distribution; and,
   for the open stack, the selection scores of all 48 candidates. No test score of a non-selected candidate is
   ever produced.

## 10. Code, smoke tests and order of work

**Order:**
1. this protocol;
2. code (`scripts/bitext24_bodies.py` and the open-stack run script, generalising `scripts/clean_24shot.py` to
   take the dataset as a parameter without changing the Banking77 path), committed after review;
3. bodies + manifest, committed;
4. arms (any order; each records its start time);
5. analyses (§9 script committed before the first arm result is read).

**Smoke tests.** They use rows of `P` that are not in `S`, never selection or test rows, and their outputs are
discarded.

## 11. Constraints carried

- TypeSafe MCA: Jev outputs are never used to tune or train anything, and Jev spend is not published.
- The dataset licence (CDLA-Sharing-1.0) allows sharing derived data under the same terms. This repo stores only
  derived per-item outputs, hashes and code.

## Amendment A1 (2026-10-12, before any code commit or model run): test-row retrieval is computed once and recorded

**Why.** Retrieval must be the same in every arm, because "identical evidence" is the point of R16.
`quorum/fewshot.py` `BM25.top` ranks with an unstable `np.argsort`. R6c measured the effect on Banking77: tie order
differs across platforms (x86 vs aarch64), and 410–427 test items had a different retrieved set. The bodies are built
on one machine, while the open stack runs on another, so the arms could otherwise see different examples.

**Change.** Nothing else changes: items, pool, BM25 parameters and body format are as declared.
1. `scripts/bitext24_bodies.py` computes the top 24 from `P` for every test row **once**. It uses the same BM25 scores
   as `quorum/fewshot.py` (same tokenizer, k1 1.5, b 0.75). Ties are broken by a **stable** sort on pool position.
2. It records the result per test item in the manifest as `retrieved_t_idx[24]` (indices into `T`, in rank order),
   next to `sha256`, `bytes` and `examples_kept`.
3. **Every arm consumes those recorded indices:**
   - the decision models through the rebuilt, sha-checked bodies;
   - the open stack's **test run** through a retrieval override that returns the recorded indices for test rows.
4. The open stack's selection phase on `S` keeps the library's own retrieval. The selection scores therefore use the
   same code path as Banking77, and selection never touches test rows.
5. The open stack's dump field `retrieved_idx` is therefore identical to the manifest by construction. The run asserts
   it.

## Amendment A3 (2026-10-12, before any pplx-decider call): the exact pplx serving engine

*Numbering note.* This was first committed as "Amendment A2" (34f5615). The paper side committed its own
`docs/phases/2.0/bitext24/AMENDMENT_A2.md` (main f4ac106: the non-near-duplicate secondary subset, the floor row,
the Jev sending rules, exposure disclosure). Both were made before any arm result was read. To keep one sequence,
that file is **A2**, and this section is **A3**. The content is unchanged.

§8 says "the A7 parity-gated SGLang engine". The pplx arm runs on that engine's current serving configuration on one
GB300. That configuration differs from the A7 run only in serving flags. It passed the same per-item parity gate.

**Code.**
- The same SGLang build (digest `868b0bd8…`, nightly-dev-cu13-20261007-f1065423).
- The same noncausal patch (sha256 `fa7f6afc…`), now baked into the image instead of applied at start.
- Attention flags unchanged: triton, no chunked prefill, no radix cache, no prefill CUDA graph.

**Serving flags that differ from A7.**
- `--context-length 16391`: the engine accepts ≤ 16,384 prompt tokens. Under A7 it was ≤ 8,192.
- `--max-running-requests 16`, `--max-queued-requests 256`, `--mem-fraction-static 0.55`.

**Gate on this exact configuration.** The b24 Banking77 bodies (3,080) were scored against the release-code dump at
concurrency 1 and 16.
- Both PASS: agreement 0.9994, max |Δp| 0.060.
- Argmax agreement with the A7 engine's own concurrency-1 output is 1.0, with Δp 0.

**Release rule.** It is applied as declared in §8: a question over 8,192 tokens (the engine reports `input_tokens`)
is scored as unanswered. The larger window therefore cannot give pplx an answer its release code would refuse.

**Concurrency.** Concurrency 2, as in §8.
