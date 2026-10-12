# Pre-registered protocol: does the evidence budget change the ranking? (Banking77 pool size, identical evidence)

**Status: pre-registration.** This file is committed before any model is run under it. Nothing has been scored
under this protocol. An earlier exploratory single-model measurement exists (pplx-decider v1.1, one draw per pool
size, no full probability vectors); it was not pre-registered, it is not evidence for this protocol, and none of its
numbers or seeds are reused here.

## 1. Question

The equal-retrieval 24-shot comparisons (Banking77 `b24B`, Bitext R16) give every system the same 24 examples,
retrieved from a large labelled pool (9,079 rows for Banking77). A developer often has far fewer labelled rows.
**When the pool the examples come from shrinks, does the ranking of the systems change?** Every system sees the
same retrieved examples from each pool.

## 2. Systems (user decision 2026-10-12: the open stack, pplx-decider v1.1 and Clef-Flash; no Clef 27B, no Jev)

| system | revision | engine |
|---|---|---|
| open stack | the Banking77 clean selection, unchanged (§5) | release code of this repo, one GB10 |
| pplx-decider v1.1 27B | `perplexity-ai/pplx-decider-v1.1-27b` @ `3b45dea` | the serving configuration of R16 amendment A3 (same image, patch, flags), one GB300, concurrency 2 |
| Clef-Flash 9B | `Cloudflare/clef-flash` @ `17f0b0a` | release code, one GB10, batch 8, with the A1 memory watchdog of `clef_eval_protocol.md` |

Not included: Clef 27B (GPU time, user decision) and Jev (3,080 calls × 11 pools is outside the approved budget).

## 3. Data, pools and draws (fixed)

- **Dataset:** `legacy-datasets/banking77` @ `f54121560de48f2852f90be299010d1d6dc612ec`; the official test split,
  all 3,080 items, in loader order. Labels and train rows are those of `docs/phases/2.0/jev/train_pinned.json`
  (10,003 rows, the same pinned revision).
- **P:** train minus the clean protocol's 924-row selection split (`results/clean/selection_split_idx.json`), i.e.
  the 9,079-row pool of the clean run and of `b24B`. All pools below are subsets of P.
- **Pools** (11 in total):
  - `full`: P itself (1 pool).
  - `pc24_d{1..5}` and `pc8_d{1..5}`: for draw d = 1..5, `rng = np.random.default_rng(20261020 + d)`. For each
    label in ascending label id, take that label's indices in P (indices into the pinned train, sorted ascending),
    permute them once with `rng.permutation`, and keep the first 24 (`pc24_d`) and the first 8 (`pc8_d`). So
    `pc8_d ⊂ pc24_d`: within a draw the smaller pool is nested in the larger one, and the size effect is not mixed
    with a different random draw. A label with fewer rows in P than the pool size keeps all of its rows; the pool
    sizes are recorded.
  - Each pool is stored as a sorted list of train indices.
- **Pool sizes (computed before any call, from `train_pinned.json` and the selection split):** in P, labels have
  23–175 rows. Exactly one label has fewer than 24 rows: `contactless_not_working` (23). So every `pc24_d` pool
  has **1,847** rows, and that label keeps the same 23 rows in every draw. No label has fewer than 8 rows, so every
  `pc8_d` pool has **616** rows (8 per label). The actual sizes are recorded in `manifest_meta.json`.

## 4. Retrieval and request bodies (built once, recorded, sha-checked)

- **Retrieval:** for each test item and each pool, the top 24 pool rows by BM25 with the scores of
  `quorum/fewshot.py` (same tokenizer, k1 1.5, b 0.75, idf computed on that pool), ties broken by a stable sort on
  pool position (as R16 amendment A1; `scripts/bitext24_bodies.py` `StableBM25`). It is computed **once** per
  pool and recorded as `retrieved_t_idx[24]` (train indices, rank order).
- **Decision-model bodies:** the Banking77 Arm-B / `b24B` body (`scripts/r14_clef.py` `armb_bodies` logic): the
  reproduction's instruction and 77 definitions (read at run time, never stored in this repository), with
  `state.labeled_examples` = the recorded 24 examples as `{"message": <train text>, "intent": <pinned label
  name>}`, trimmed from the end only if the body exceeds 30,000 bytes. The same bytes go to pplx-decider and
  Clef-Flash.
- **Manifest:** one file per pool, `results/poolsize/manifest_<pool>.jsonl.gz`, one line per test item:
  `{idx, sha256, bytes, examples_kept, retrieved_t_idx}`, gzip mtime 0. A `manifest_meta.json` records each pool's
  train indices, its size, and the sha256 of every manifest file. All are committed before any model call. Every
  arm rebuilds each body, checks its sha256 and aborts on a mismatch.
- **Open stack:** its test pass consumes the same recorded indices (the R16 `RecordedRetrieval` override); its kNN
  member indexes the pool's rows only. The dump field `retrieved_idx` must equal the manifest (asserted).

Note on criteria text (as in `b24B`): the decision models receive the reproduction's 77 definitions; the open stack
uses the option text its clean selection chose. Evidence (the examples) is identical; option text is each system's
own, unchanged from the `b24B` comparison.

## 5. The open stack: fixed configuration, no re-selection

The open stack runs the candidate selected by the Banking77 clean protocol (`results/clean/selection_scores.json`:
M2 = reader + kNN, `weighted_prob_mean`, descriptions, 1 order) with its fitted temperatures, weights and final
temperature, **unchanged for every pool**. Only the pool (retrieval and the kNN index) changes.

**Stated limitation.** Those parameters were fitted on the 924-row selection split with retrieval from the full P.
A developer with 24 or 8 rows per label would not have 924 extra labelled rows to fit on. The decision models use no
fitting data at all. This favours the open stack on the small pools, by an amount this protocol does not measure,
and the write-up must say so next to every small-pool result.

**Pool-fit sensitivity arm (descriptive, measures that limitation).** "Open stack, pool-fit" runs the same selected
candidate (M2, `weighted_prob_mean`, descriptions, 1 order) on the 10 small pools, with its fitted parameters
(member temperatures, weights, final temperature) **re-fitted on the pool itself, with no use of the selection
split**:
- Each pool row is scored by the reader and kNN members with **leave-one-out retrieval**: the BM25 top 24 (same
  scores, stable ties) from the pool minus that row's own position.
- The parameters are fitted on all pool rows exactly as the clean run's final refit
  (`clean_24shot.py` `fit_params` with `weighted_prob_mean`, then `fit_temperature` of the combined
  distribution): what a developer holding only the pool could do.
- The test pass is the unchanged arm's test pass (same recorded retrieval, same member logits); only the fitted
  parameters differ.
- It is in **no** Holm family. Reported per pool: accuracy, ECE-10, Brier, and the difference from the unchanged
  arm. The `full` pool has no pool-fit cell (fitting on 9,079 rows is the full-data case already covered by S).

One confound remains in both arms and must be named in the write-up: the **candidate itself** (member set, rule,
option text, orders) was chosen on the 924-row selection split with full-P retrieval.

## 6. Scoring

- Per item: the predicted label is the argmax of the full probability vector over the 77 labels (open stack: of
  the combined distribution, as the clean run). Probabilities are stored over the 77 labels in label-id order.
- pplx-decider: an item whose prompt exceeds 8,192 tokens (server `usage.input_tokens`) is unanswered (= wrong),
  the release rule (R16 A3). Clef-Flash truncates the state to 16,384 tokens as its release code does. Expected: no
  item near either limit (`b24B` max prompt 3,996 Clef tokens); the counts are reported.
- A failed item counts as wrong. One scored run per system × pool; no re-runs of completed cells.
- Per-item dumps (R-format): `results/poolsize/predictions/<system>_<pool>.jsonl.gz` with
  `{idx, gold, pred, probs[77], unanswered, input_tokens}` (open stack: no `input_tokens`, plus `retrieved_idx`;
  the pool-fit arm as `open_poolfit_<pool>`, with its fitted parameters in the summary).

## 7. Pre-declared analyses (all reported, whatever the outcome)

1. **Primary — pairwise accuracy per pool.** For each of the 11 pools, exact two-sided McNemar for the 3 pairs
   (open–pplx, open–Clef-Flash, pplx–Clef-Flash) on the identical 3,080 items. **Holm families by pool size:**
   `full` (3 tests), `pc24` (5 draws × 3 = 15 tests), `pc8` (15 tests).
2. **Ranking stability (descriptive, the question of §1).** For each pool: each system's accuracy and the rank
   order of the 3 systems. For each pool size: mean, min and max accuracy across the 5 draws per system; in how
   many draws the rank order equals the `full` order; and for each pair, in how many draws the Holm-significant
   winner is each system (or none).
   **Ties:** two systems tie in a pool when their correct counts are equal. A pool whose order contains a tie is
   its own outcome ("tie"): it is counted neither as matching nor as differing from the `full` order, and is
   reported as such. If `full` itself contains a tie, the comparison is reported the same way.
3. **Size effect per system (descriptive).** For each system and draw: accuracy `pc24_d` − `full` and `pc8_d` −
   `full`, with the across-draw mean, min and max. No test is run on these; draws are not independent of `full`.
4. **Calibration (descriptive).** ECE-10 (probability of the predicted label) and Brier (summed over the 77
   labels) per system × pool. No log loss across engines that round.
5. **Pool-fit sensitivity (descriptive, §5).** Per small pool: the pool-fit arm's accuracy, ECE-10 and Brier, and
   its difference from the unchanged open-stack arm; across draws, mean, min and max. In no Holm family.
6. **Descriptive.** Pool sizes, examples kept, unanswered / truncated counts, and per system the agreement of the
   `full` cell with that system's existing `b24B` / clean-run cell (same pool, retrieval with stable instead of
   library tie order), as a consistency check only.

No pooled test across draws, no names-only arm, no Jev arm.

## 8. Order of work

1. This protocol.
2. Code (`scripts/poolsize_bodies.py`, `scripts/poolsize_arms.py`, `scripts/poolsize_open.py`,
   `scripts/poolsize_analysis.py`), committed after review. The analysis script is committed before any cell is run.
3. Pools, manifests and meta (CPU only), committed.
4. Smokes: the first 8 rows of the 924-row selection split (in no pool, never test rows) as queries, retrieving
   from `pc8_d1`; outputs discarded.
5. Runs, after R16 and its single §9 run. GB10 queue: Clef-Flash then the open stack, pool order `full`,
   `pc24_d1..5`, `pc8_d1..5`; the open stack computes its pool-fit arm in the same pass. pplx on the GB300 serving engine in the same pool order.
6. The analysis runs once, after all 33 cells exist.

Estimated cost (measured rates): Clef-Flash ~80 min per pool (~15 h); open stack ~31 min per pool (~6 h); pplx
~10 min per pool (~2 h); open-stack pool-fit reader + kNN passes over the pool rows (~2 h).

## 9. Constraints carried

- Never tune or train on Jev outputs (TypeSafe MCA); no Jev arm here.
- The reproduction's instruction and definitions are read at run time and never committed (the reproduction has no
  licence).
- Hardware is named only as the device type.
