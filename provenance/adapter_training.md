# Provenance: training record of `DataAgent/Quorum-Reader-Qwen3-4B-Adapter`

Prepared by the research session from the study's training scripts, data manifest, leakage reports and training
log. Items the records do not settle are marked **unknown**. The training code lives in a private research
codebase and is not published (see the last section). Everything needed to judge the result is described here.

## Identity

- **The published adapter is the study's checkpoint.**
  - `adapter_model.safetensors` sha256 `6ef3f783cdd25d889004146e780a0ae39dd7c87541bbac418b4aaa058529c381` is
    identical in the study checkpoint and in the published repo @ `e0765b768e04ff8dc83a8682963349bd6291ffa3`.
  - `adapter_config.json` is identical field by field.
- **Base model:** `Qwen/Qwen3-4B`. The revision used **during training was not pinned and is unknown**. All
  evaluations in this repo pin `1cfa9a7208912126459214e8b04321603b3df60c`.

## Training data

A 14-task mixture, drawn **from each task's train split only** (Python `random.seed(0)`). 5% per task is held
out as a development set.

| task | source | train rows | dev rows | classes |
|---|---|---|---|---|
| CLINC150 | `clinc/clinc_oos` (plus) | 3,800 | 200 | 151 |
| HWU64 | `DeepPavlov/hwu64` | 2,850 | 150 | 64 |
| SNIPS | `DeepPavlov/snips` | 1,900 | 100 | 7 |
| ATIS | `tuetschek/atis` | 1,425 | 75 | 16 |
| AG News | `fancyzhx/ag_news` | 2,375 | 125 | 4 |
| DBpedia-14 | `fancyzhx/dbpedia_14` | 2,375 | 125 | 14 |
| Yahoo Answers | `community-datasets/yahoo_answers_topics` | 2,375 | 125 | 10 |
| TREC fine | `SetFit/TREC-QC` | 2,375 | 125 | 50 |
| TREC coarse | `SetFit/TREC-QC` | 1,425 | 75 | 6 |
| SST-2 | `stanfordnlp/sst2` | 2,375 | 125 | 2 |
| SST-5 | `SetFit/sst5` | 2,375 | 125 | 5 |
| IMDB | `stanfordnlp/imdb` | 1,900 | 100 | 2 |
| Amazon polarity | `fancyzhx/amazon_polarity` | 2,375 | 125 | 2 |
| Emotion | `dair-ai/emotion` | 2,375 | 125 | 6 |

- **Totals:** 32,300 train and 1,700 dev rows, then 32,296 train after the leakage purge below.
- **Not only intent data.** The mixture includes topic, question-type, sentiment and emotion tasks, so describing
  it as "trained only on public intent data" is inaccurate.

## Held-out status of the seven evaluation sets

| eval set | in the training mixture? | test sentences found verbatim in the mixture (normalised exact match) |
|---|---|---|
| **Banking77** | **excluded** (train and test) | **0 / 3,080** |
| Bitext | no | 0 / 5,375 |
| MTOP | no | 11 / 4,386 (0.25%) |
| HWU64 | **yes** (train split) | 4 / 1,076 (0.37%) |
| SNIPS | **yes** (train split) | 6 / 1,400 (0.43%) |
| CLINC150 | **yes** (train split) | 22 / 5,500 (0.40%) |
| **MASSIVE** | not as a task, **but shares HWU64's intent ontology** | **318 / 2,974 (10.69%)**: 311 from HWU64 train rows, 7 from CLINC150 |

**What this means.**
- The 24-shot reader's results are cleanly held out only on **Banking77, Bitext and MTOP**. MTOP has a negligible
  0.25% overlap.
- HWU64, SNIPS and CLINC150 were seen during training.
- **MASSIVE is contaminated:** about one in nine of its test sentences appears verbatim in the training data,
  via HWU64. This was not detected at training time. Earlier statements that MASSIVE was a clean held-out set
  for the adapter are wrong.
- The zero-shot ensemble does not use the adapter, so its results are unaffected.

The counts are exact matches after lowercasing, keeping only alphanumerics, and collapsing whitespace. They are a
**lower bound**: fuzzy near-duplicates were measured only against Banking77 (next section).

## Contamination checks

**At training time, against Banking77 only.**
- The mixture was compared against all 13,083 Banking77 train and test texts by exact match and by MinHash-LSH
  (5-gram Jaccard ≥ 0.8).
- **The first run failed:** 3 exact matches and 1 near-duplicate, all CLINC150 rows from `card_declined`,
  `report_lost_card` and `card_not_working`. For example, "why was my card declined" and "someone stole my card"
  appear in both datasets.
- The 4 rows were removed and the check re-run: 0 hits, PASS.

**Post hoc, for this record.** The normalised exact-match check of the final mixture against all seven
evaluation test sets, shown in the table above.

## Training procedure and hyper-parameters

| item | value |
|---|---|
| Method | LoRA on all attention and MLP projections (q, k, v, o, gate, up, down) |
| LoRA settings | r = 64, alpha = 128, dropout 0.05 |
| Objective | proper scoring loss on the K-way option distribution: cross-entropy + 1.0 × Brier |
| Prompt and scoring | the same as at inference (`quorum/prompting.py`): options carry single-token markers, the next-token logits at the markers form the distribution, options are shuffled per example |
| In-context examples per training item | drawn from {0, 0, 0, 4, 8, 24}, so half the items are zero-shot |
| Steps | 2,926 |
| Batch | 4 per step × 2 gradient-accumulation steps = 8 |
| Learning rate | 1e-4, warm-up 3% |
| Max sequence length | 1,536 tokens, with gradient checkpointing |
| Seed | 0 |
| Checkpoint | the final one. No early stopping and no selection on any evaluation set; the health check used the mixture's dev set, not Banking77. |
| Compute | one NVIDIA GB10, about 10.4 s per step, so about 8.5 GPU-hours |

## Code

- **Not published.** The training script and the mixture builder are part of a private research codebase.
- **Not ported.** A clean-room port into this repo has not been done. It would cover the mixture builder, the
  training loop with the CE + Brier objective, and shot sampling.
- **What is reproducible.** The procedure above is specified completely enough to re-implement. The inference
  side (prompt construction and scoring) already ships in this repo, and the released weights are the exact ones
  used.

## Known gaps

- The base-model revision used in training is **unknown**.
- Fuzzy (near-duplicate) overlap against the non-Banking77 evaluation sets was **not measured**. The exact-match
  counts above are a lower bound.
