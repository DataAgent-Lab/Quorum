# Provenance: how the Banking77 24-shot configuration reported as 0.938 was selected

Prepared by the research session from the original experiment records: run scripts, their commit history, and
the per-run result files written at the time. Every number below is copied from those result files, not
recomputed. Where the records do not settle a question, this document says **unknown**.

## Verdict

- **The 0.938 configuration was selected on a calibration split carved from the training data, not by test
  score.** Its member set and combination rule were fixed in the run script before any result existed. Both
  were then chosen by calibration accuracy.
- **It is not the test-best candidate.** Among the configurations available at that stage, a different
  combination rule scored 0.9396 on test and was not adopted.
- **But several statements that accompanied the result overstate its rigour.** Do not repeat them as written.
  - "survives Bonferroni": the correction family was never defined.
  - "pre-registered": the anchor was designated after its own test score was known.
  - "calibration fits temperatures only": in a geometric mean those temperatures act as a fitted mixing weight.
  - "BM25 over the full training pool": the pool was 9,079 rows, not 10,003.
- **Two upstream design choices touched Banking77 test data.** Both were made before any ensemble existed.
  - The number of option orders (3) and the answer scoring method were chosen on the first 200 test items.
  - Per-class descriptions were adopted because zero-shot test results showed descriptions helped.

Details follow. Section 8 gives what the paper can claim as-is, and what a fully clean result would need.

## 1. The selection (calibration) split

- **Where it comes from.** 924 rows drawn from the Banking77 **train** split: a uniform random permutation
  (numpy `default_rng(0)`, first 924 indices), not stratified by class.
- **Disjointness.** These rows are removed from the retrieval pool, so the selection split is disjoint from
  both the pool and the test set.
- **Side effect on the pool.** The pool used for retrieval at test time is therefore **9,079** rows
  (10,003 − 924), not the full 10,003-row training set.
- **Reproducing the ids.** Load the Hugging Face `banking77` dataset and take
  `np.random.default_rng(0).permutation(10003)[:924]`. No dataset revision was pinned. A later parity check
  confirmed that the regenerated split's labels matched the saved ones.
- **Scoring difference between cal and test.** On the selection split the in-context member was scored with
  **1** option order. On test it used **3** orders, averaged. So its fitted temperature was learned on
  single-order logits and applied to averaged ones. The effect of this mismatch was not measured.

## 2. Every configuration scored, with the split it was scored on

The combiner computes four combination rules for every member set:
- arithmetic mean of probabilities (`prob_mean`);
- geometric mean (`logprob_mean`);
- mean rank (`rank_mean`);
- a weighted probability mean with weights fitted on cal (`weighted_prob_mean`).

For each rule it reports both calibration and test accuracy, and selects the rule with the highest
calibration accuracy. So **test accuracy was computed and printed for every rule of every set below.**

Members:
- **in-context**: the 4B reader with the LoRA adapter, reading the 24 retrieved examples.
- **kNN**: cosine to the same 24 examples, using bge-large unless noted.
- **zero-shot**: PrismNLI plus a bge zero-shot member.

### Stage D (the stage that produced 0.938)

| member set | prob_mean (cal / test) | logprob_mean | rank_mean | weighted_prob_mean | selected by cal |
|---|---|---|---|---|---|
| in-context + kNN | 0.9286 / **0.9396** | **0.9329 / 0.938** | 0.9242 / 0.9269 | 0.9307 / 0.9354 | logprob_mean → **0.938** |
| in-context + kNN + 2 zero-shot | 0.9232 / 0.9334 | 0.9113 / 0.9273 | 0.8561 / 0.8744 | 0.9286 / 0.9341 | weighted_prob_mean → 0.9341 |

- **How the pair was chosen.** Pair and 4-member set were compared on the selected rules' calibration accuracy:
  0.9329 for the pair vs 0.9286 for the 4-member set. The pair won.
- **Timing.** Both member sets were written into the run script before the results existed. The script was
  committed 2026-09-24 02:47 UTC and the result files were written at 04:18 UTC.
- **A caveat.** The pair-vs-4-member comparison was not coded into the script. It was made by hand afterwards,
  with both test scores already printed. The comparison itself used the calibration accuracies above.

### Later stages (run after 0.938 was known)

Stage F tried more option orders (7) and a stronger kNN embedder (Qwen3-Embedding-8B):

| member set | selected rule (cal / test) | test-best rule on that set |
|---|---|---|
| A: the 0.938 set (anchor) | logprob 0.9329 / 0.938 | prob_mean 0.9396 |
| B: 7 orders + bge kNN | logprob 0.9329 / 0.939 | prob_mean 0.9403 |
| C: 3 orders + Qwen3-8B kNN | weighted 0.9275 / 0.9334 | prob_mean 0.9347 |
| D: 7 orders + Qwen3-8B kNN | weighted 0.9275 / 0.9325 | logprob 0.9357 |
| E: 7 orders + bge kNN + Qwen3-8B kNN | **logprob 0.9351 / 0.938** (stage cal-winner) | prob_mean 0.939 |

Stage G added an instruction-prefixed query to the kNN embedder:

| member set | selected rule (cal / test) | test-best rule on that set |
|---|---|---|
| bge + instruction kNN | **logprob 0.9361 / 0.9377** (stage cal-winner) | prob_mean 0.9393 |
| 7 orders + bge + instruction kNN | logprob 0.9361 / 0.9377 | prob_mean 0.9399 |
| instruction pair | weighted 0.9307 / 0.9351 | prob_mean 0.9367 |

### Single members, each scored on test

| member | test accuracy |
|---|---|
| in-context, 3 orders averaged | 0.9195 |
| in-context, single order | 0.9162 |
| in-context, 7 orders | 0.9205 |
| kNN bge-large | 0.9279 |
| kNN Qwen3-Embedding-8B | 0.9256 |
| kNN Qwen3-8B with instruction | 0.925 |
| untrained base model, same 24-shot prompt | 0.8529 |
| 24 synthetic shots, adapter | 0.7302 |
| 24 synthetic shots, untrained | 0.5519 |

**Never run, as far as the records show:**
- kNN scale or floor variants;
- kNN mean aggregation;
- class names instead of descriptions inside the 24-shot reader;
- capped-per-class or random-shot retrieval.

Counting distinct configurations scored on test gives about 40: 9 distinct member sets × 4 rules = 36, plus
the single members above.

## 3. What was selected, and what "calibration" fitted

All of the following were fitted on the 924-row selection split only:
- one temperature per member (golden-section search on negative log-likelihood, range [0.05, 10]);
- simplex weights, used only by the weighted rule;
- the rule itself (highest calibration accuracy);
- a final temperature, which does not change the argmax.

Fitted values for the 0.938 configuration: T_reader = 1.2724, T_kNN = 0.4255, T_final = 1.5744.

**These temperatures matter for accuracy, not just calibration.** Under a geometric mean the combined score is
proportional to logit_reader / T_reader + logit_kNN / T_kNN. So the two temperatures act as a mixing weight
fitted on cal. Here it is about 3:1 in favour of kNN (2.35 vs 0.79 per unit logit). "Calibration fits
temperatures only" is literally true, but the temperatures are a fitted weight in all but name.

Stage F's script declared the 0.938 set an "anchor" and said a pre-chosen anchor would be reported alongside the
calibration winner. That declaration was written about 21 hours after the anchor's own test score was
committed, so it fixes the anchor relative to stages F and G only.

Stage G's own calibration rule would have selected 0.9377, one test item fewer than 0.938. The headline stayed
at the anchor. A single calibration-based choice across all stages (D, F and G) would also give 0.9377.

## 4. The Bonferroni statement

- **What the record says.** "Survives Bonferroni/3". No document or script defines which three comparisons form
  the family. The most likely reading is the three McNemar tests run in stage D: in-context alone, the
  4-member set, and the pair.
- **Exact numbers.** McNemar of the pair vs the public Jev reproduction, b = 109 and c = 66, gives an exact
  two-sided p = 0.001428.
- **Sensitivity.**

| correction family | Bonferroni-adjusted p | significant at 0.05? |
|---|---|---|
| m ≤ 35 comparisons | ≤ 0.05 | yes |
| m = 6 (every 24-shot system compared against the reproduction) | 0.0086 | yes |
| m ≈ 40 (every configuration scored on test) | ≈ 0.057 | **no** |

"Survives correcting for the handful of configurations we tried" is therefore not supported as written.

## 5. How the fixed settings were set

| setting | how it was set | touched Banking77 test? |
|---|---|---|
| kNN scale 20, floor −1 | Defaults in the first version of the kNN member, never varied. −1 is the minimum cosine. Scale 20 was copied from the zero-shot embedding member. | No variants were run. |
| Zero-shot cosine scale 20 (`COSINE_SCALE`) | The same constant, set when the zero-shot embedding member was first written. | No variants were run. |
| 24 shots | Jev's protocol. | No. |
| 3 option orders, averaged; answer-token scoring of single-token option markers | Chosen in an early probe on the **first 200 Banking77 test items** with the untrained base model. | **Yes, mildly** (a debug slice of test). |
| Task description and prompt | Written before the probe. No variants found. | No. |
| Option markers (A–Z, a–z, AA–ZZ) | Tokenizer single-token check. | No. |
| Per-class descriptions used as options | Written once, never edited afterwards. | No sign the text was tuned on test results. Whether the author looked at test confusions while writing them is **unknown**. The decision to use descriptions rather than names followed zero-shot **test** results, on other models, showing that descriptions helped. |

## 6. 0.938 vs the public default (0.932)

| | 0.938 (study) | 0.932 (this repo's `scripts/eval_24shot.py`) |
|---|---|---|
| Option text | per-class descriptions | class names |
| Combination | cal-fitted per-member temperatures, then geometric mean | geometric mean at T = 1, no calibration |
| Retrieval pool | 9,079 rows (selection split removed) | full train, 10,003 rows |
| Adapter | the study's training run | `DataAgent/Quorum-Reader-Qwen3-4B-Adapter`; identity with the study's checkpoint **not verified** |
| Same in both | 3 orders, kNN scale 20 and floor −1, BM25, bge-large | |

The contribution of each difference was never isolated.

## 7. Other facts the paper needs

- **Adapter training data.** The only exclusion was Banking77 (train and test). **CLINC150, HWU64 and SNIPS are
  in the training mixture.** MASSIVE, MTOP and Bitext are not. 24-shot results on CLINC150, HWU64 or SNIPS
  with this adapter are therefore not clean held-out results. The study's cross-domain 24-shot claims were
  restricted to MASSIVE, MTOP and Bitext for this reason. The mixture also contains non-intent tasks, so
  "trained only on public intent data" is inaccurate.
- **Base model revision.** `Qwen/Qwen3-4B` was used without a pinned revision. The exact revision is
  **unknown**.
- **Retrieval check.** No check was made that our BM25 (k1 = 1.5, b = 0.75, own implementation) returns the
  same 24 examples as the Jev reproduction's retrieval. Request R6 addresses this.

## 8. What the paper can claim, and what a clean main result needs

**As-is, defensible:**
- 0.938 was selected by calibration accuracy among member sets and rules that were fixed before the run.
- It is not the test-best of those candidates.
- Against the public Jev reproduction, McNemar p = 0.0014 (b = 109, c = 66).
- That p remains below 0.05 under a Bonferroni correction for up to 35 comparisons, including all six 24-shot
  systems compared against the reproduction.
- A single calibration-based choice across all later stages gives 0.9377.

The paper must also disclose:
- the 200-item test probe;
- the description decision;
- the 9,079-row pool;
- that the temperatures act as a fitted mixing weight;
- that about 40 configurations were scored on test overall.

**For a fully clean main result (proposed protocol, to run only with approval):**
1. Carve a fresh selection split from Banking77 train, disjoint from the old one, and remove it from the pool.
2. Pre-declare the candidate grid and the selection rule, and commit both before any run.
3. Choose permutations and scoring on train-only data, not on test.
4. Select on the new split only.
5. Run the single selected configuration once on the full test set.
6. Report that one number with its McNemar against the reproduction.
