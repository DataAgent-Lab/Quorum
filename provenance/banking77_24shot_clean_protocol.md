# Pre-registered protocol: clean selection and single test run for the Banking77 24-shot result

**Status: pre-registration.** This file is committed before any model is run under this protocol. Its commit
timestamp is the pre-registration evidence. Until the run script is committed (§9), nothing is scored on the
selection split or on the test set: no model run of any kind. The only work done so far is the read-only pin
lookups in §1.

The single test accuracy produced in §7 becomes the paper's 24-shot main result, whatever its value or
significance. The earlier 0.938 is reported only as exploratory history (see `banking77_24shot_selection.md`).

## 1. Pins (looked up read-only before this commit)

| item | pin |
|---|---|
| Dataset | `legacy-datasets/banking77` @ `f54121560de48f2852f90be299010d1d6dc612ec`: train 10,003, test 3,080 (40 per class) |
| Base model | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`, bf16 |
| Adapter | `DataAgent/Quorum-Reader-Qwen3-4B-Adapter` @ `e0765b768e04ff8dc83a8682963349bd6291ffa3` |
| Adapter vs the study's checkpoint | **Identical.** `adapter_model.safetensors` sha256 `6ef3f783cdd25d889004146e780a0ae39dd7c87541bbac418b4aaa058529c381` in both, and `adapter_config.json` identical field by field (r=64, alpha=128, dropout 0.05, all linear projections). Not retrained. |
| kNN embedder | `BAAI/bge-large-en-v1.5`, revision recorded at run time in the meta |
| Zero-shot members | the repo's `ZeroShotEnsemble` defaults at Quorum commit 04e6709 (PrismNLI-0.4B + bge-large + bge-base). Revisions recorded in the meta. |
| Code | Quorum `f21a4a0` (the base of this branch) plus the run script committed per §9 |

**Train order.** The study's loader (`banking77`) and this repo's loader (`legacy-datasets/banking77`) return
identical train rows in identical order. All 924 rows of the study's old selection split are present.

## 2. Selection split: fresh, stratified, disjoint from the old one

1. Let `T` be the 10,003 train rows in the pinned dataset order, indexed 0–10,002.
2. Old split: `O = np.random.default_rng(0).permutation(10003)[:924]`, the study's 924 rows.
3. New split `S`, 12 rows per class (77 × 12 = 924), drawn with one generator `rng = np.random.default_rng(20261005)`:
   for each class label `c = 0..76` in ascending order, take the train indices with label `c` not in `O`, sorted
   ascending, permute them with `rng.permutation`, and keep the first 12.
4. Every class has at least 32 eligible rows, so the draw is always feasible.

**Why 12 per class (equal allocation) rather than proportional.** The test set is exactly balanced (40 per
class). A balanced selection split matches the distribution that will be evaluated, and gives every class the
same weight in selection.

**Pool.** `P = T \ S`: 9,079 rows, with the old 924 rows back in. The same pool is used when scoring the
selection split and in the test run, so fitted parameters are applied under the same retrieval conditions they
were fitted under. This pool is 924 rows smaller than the reference protocol's full training set. That can
only disadvantage our system, never favour it.

The index list of `S` is written to `results/clean/selection_split_idx.json` by the run script before any
model is loaded.

## 3. Fixed settings (declared, not searched)

| setting | value |
|---|---|
| Shots | 24 per query |
| Retrieval | BM25 (k1 = 1.5, b = 0.75, `quorum/fewshot.py`), top 24 from `P`; the same 24 for the reader and kNN |
| kNN member | bge-large cosine to the 24 retrieved examples; max per class; scale 20; floor −1 for classes absent from the 24 |
| Reader prompt | `quorum/prompting.py`, with task description "Classify the customer's banking message into the intent it expresses." (the string in `scripts/eval_24shot.py`) |
| Answer scoring | next-token logit of single-token option markers (A–Z, a–z, AA–ZZ), un-permuted per order |
| Order seeds | `random.Random(1000 + i)` for order i = 0..6 (the repo's convention) |
| Max length | 2,048 tokens |
| Batch size | 8 |
| Reader precision | bf16 |
| Zero-shot member | the repo default for this device (fp16 on GPU) |

**Why answer scoring is fixed rather than searched.** The adapter was trained to answer with the option-marker
token. Any other scoring would be off its training distribution. So the method is fixed by construction, no
data is used to choose it, and test is never consulted.

## 4. Candidate grid (48 candidates)

| dimension | values |
|---|---|
| Member set | M2 = {reader, kNN}; M3 = {reader, kNN, zero-shot}. The zero-shot member is the repo's zero-shot ensemble distribution, used as one member. |
| Combination rule | `logprob_mean`, `prob_mean`, `rank_mean`, `weighted_prob_mean` |
| Option text | class names (the repo's humanized names); per-class descriptions (`examples/banking77_descriptions.py`). Applies to the reader options and to the zero-shot label text. |
| Option orders (reader) | 1, 3, 7, averaging the first k of the 7 seeded orders |

Grid size: 2 member sets × 4 rules × 2 option texts × 3 order counts = **48 candidates**.

## 5. Fitting and selection score (selection split only)

1. **Logits.** Compute member logits on the 924 selection rows. Retrieval comes from `P`, and each row is
   scored with the **same number of orders as the candidate**. This fixes the old cal/test mismatch, where cal
   was scored at 1 order and test at 3.
2. **Fitted parameters** for every candidate:
   - one temperature per member: golden-section search on negative log-likelihood over [0.05, 10];
   - for `weighted_prob_mean` only, simplex weights;
   - a final temperature, reported but argmax-neutral.
3. **Cross-fitting.** The parameters are fitted by **5-fold stratified cross-fitting** within the selection
   split (fold seed 20261006). Each candidate's selection score is its **out-of-fold** count of correct rows
   out of 924. In-sample accuracy is also recorded, for information only.
   **Why cross-fitting.** It removes the in-sample advantage of rules with more fitted parameters.
4. **Disclosure.** Under the geometric mean the per-member temperatures act as a fitted mixing weight, and are
   reported as such.

## 6. Selection rule (declared now)

Select the candidate with the highest out-of-fold correct count. Exact ties are broken in this order:
1. fewer members;
2. fewer orders;
3. class names over descriptions;
4. rule in the order `logprob_mean`, `prob_mean`, `rank_mean`, `weighted_prob_mean`.

The order is total, so the choice is deterministic. The selected candidate's parameters are then refit on all
924 selection rows for the test run.

## 7. The single test run

- **What runs.** Only the selected candidate, **once**, on all 3,080 test rows in the pinned dataset order,
  with retrieval from `P` and the refit parameters.
- **Isolation.** No other candidate is ever scored on test. The run script never loads test labels before
  selection is final and written to disk.
- **Outputs, per-item dump** `results/predictions/banking77_24shot_clean.jsonl.gz`, one line per test row:
  `{idx, text, gold, pred, probs[77], reader_probs[77], knn_probs[77], zs_probs[77] (M3 only),
  retrieved_idx[24] (indices into T), perm_reader_probs[[77] × orders]}`.
- **Outputs, meta** `banking77_24shot_clean_meta.json`: this protocol's commit hash, the run-script commit hash,
  all pins, the selected configuration, the fitted parameters, library versions, device, and wall time.
- **Comparison.**
  - Paired exact McNemar against the public reproduction's per-item predictions:
    `github.com/simonmesmith/jev-banking77-experiment` @ `5cac4ff`, `predictions.csv`.
  - That file is read at run time only and never copied into this repo.
  - Rows are joined to the official test order. The matching method and any unmatched rows are reported.
  - No multiplicity correction applies: this is one pre-registered test.
- **Reporting.**
  - `results/clean/selection_scores.json` lists all 48 candidates' selection-split scores (out-of-fold and
    in-sample).
  - It contains **no test scores for non-selected candidates**.

## 8. Failure handling

- **Interrupted runs.** Per-order logits are saved as they are computed, so an interrupted run resumes without
  changing anything.
- **Crash during the test run.** If the test run crashes, the same selected configuration is re-run with the
  same seeds, and the crash is disclosed. A different configuration is never tried.
- **Determinism.** bf16 GPU kernels may not be bit-reproducible across reruns. Any rerun is disclosed together
  with both outputs.

## 9. Code and smoke tests

- **Code commit.** The run script implementing exactly this protocol is committed to this branch after review
  and before any run. Its hash goes into the meta.
- **Smoke tests.** Pipeline checks use only rows of `P` that are not in `S`, never selection-split or test rows.
  Their outputs are discarded.

## 10. Deviations from the request this protocol answers

- **Answer scoring is fixed (M1), not searched.** See §3.
- **Cross-fitting** of the fitted parameters. See §5.
- **Equal-allocation stratification**, 12 per class. See §2.
- **The zero-shot member** is the repo's zero-shot ensemble used as one member.
