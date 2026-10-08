# R14 protocol — Cloudflare Clef / Clef-Flash under this study's standard

Committed before any scored run. Script: `scripts/r14_clef.py`.

## Models

| tag | repository | revision | backbone |
|---|---|---|---|
| `clef` | `Cloudflare/clef` | `2f3de3dd85f379784083b0814d997ab627200f0c` | Qwen3.8-27B + joint schema head |
| `clef_flash` | `Cloudflare/clef-flash` | `17f0b0ad64efb65d273590632833508766b2aae6` | Qwen3.5-9B + joint schema head |

Apache-2.0; loaded with the repositories' own `joint_schema_model.py` (read before use; its sha256 is recorded in
every summary), bf16, on one GB10. Scores are the model's per-option logits with a softmax per question, not the
4-decimal `systemone` response.

## Training-data status: UNVERIFIED

The model cards do not list training data. The announcement says post-training "leverages our own internal
synthetic datasets permutating field orders, prompts, and schema structures", and does not say whether the
synthetic data were derived from, or exclude, Banking77, CLINC150, HWU64, MASSIVE, MTOP, SNIPS or Bitext. The
cards report BANKING77 macro-F1 94.2 / 90.9 and CLINC150+OOS 97.4 / 66.8 (Clef / Clef-Flash) on an undisclosed
protocol, far above the open zero-shot results in this study. Every Clef number is therefore reported as
"training data undisclosed; exposure to the evaluation datasets cannot be ruled out", never as zero-shot evidence.
The `train` setting is a memorisation probe, not proof either way.

## Settings

1. **zs** — the 7 datasets (Banking77, CLINC150, HWU64, MASSIVE, MTOP, SNIPS, Bitext), full official test sets,
   loaded with `quorum.data.load` exactly as R3. Request: `state = {"message": text}`, one `choice` question with
   instruction "Which ONE intent best describes `message`? Every message belongs to exactly one of the listed
   categories." and criteria mapping each class name to itself (**names**) or to this repository's description
   (**descriptions**). The instruction is identical for all datasets. Difference from R3 recorded: in the
   descriptions variant Clef also sees the class name as the option id (natural use of the Jev/Clef API), while
   the R3 ensemble sees only the description.
2. **b24** — Banking77 full test (3,080), the public reproduction's `retrieved24` request body for each item: 24
   training examples (word+bigram BM25, <=4 per class, all 10,003 rows, its tie order), its instruction, its 77
   definitions, its 30,000-byte cap. Rebuilt clean-room; phase `parity` showed all 3,080 bodies byte-identical to
   the reproduction's own builder (`results/r14/parity.json`). These are the exact requests whose Jev answers the
   reproduction reports (0.924).
3. **defs** — Banking77 full test, the reproduction's `definitions` body (its definitions, no examples).
4. **train** — the R12 pre-registered train-side sample (7 x 1,000, `results/r12/sample_ids.json`) with the `zs`
   request, names and descriptions.

## Metrics and comparisons

Per setting x dataset: accuracy, macro-F1, log loss (clip 1e-15), Brier (summed over classes), ECE (10 bins), max
prompt tokens, ms/item. Exact two-sided McNemar:
- b24 and defs vs the reproduction's per-item Jev predictions (`results/predictions.csv`, read at run time);
- b24 vs the clean 24-shot run (0.9357) and the parity run R6 `best_parityret` (0.9234);
- zs vs the R3 zero-shot ensemble and the R13 clean-NLI ensemble (`clean_c`), same items.

## Run discipline

- One scored run per model x setting; no prompt, instruction or option-format tuning on test.
- Smoke runs use `--limit` and write `_limitN` files, which are not results.
- Batch size: the largest that passes `batchcheck` (batch-1 vs batch-8 argmax identical on 16 items; logit
  differences reported). Right padding with a causal backbone; batching is not expected to change outputs beyond
  bf16 noise.
- All comparisons beyond the reproduction McNemar are secondary and uncorrected.

## Amendment A1 (2026-10-07, before any Clef 27B result)

The first Clef 27B b24 run (batch 8) was **stopped by the operator before completion** at 1 h 47 min. Its
process held 100 GB of GPU memory and still rising (length-sorted batches; caching-allocator growth), with host
MemAvailable down to 13.5 GB on the 119 GB unified-memory GB10. A second GB10 host had hung earlier under a
similar load. No output was written, so there is no partial result to discard or select from.

Changes for all remaining runs. None of them alters any item, request body, model or scoring:
- Clef 27B uses batch 4 (Clef-Flash keeps batch 8). Batching only adds bf16-level noise: right padding with a
  causal backbone; Clef-Flash batch-1 vs batch-8 gave 0 argmax changes. `batchcheck` is rerun for Clef 27B
  before its runs.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, an allocator setting, to limit fragmentation.
- A host memory watchdog: below 20 GB (first host) / 25 GB (second host) MemAvailable it kills the eval process. A
  killed run is rerun from scratch with a smaller batch, never resumed or partially reported.
- Timing (ms/item) of a run is reported together with its batch size. Latency is not compared across batch
  sizes.

## Amendment A2 (2026-10-07, before any pplx-decider run): add `perplexity-ai/pplx-decider-v1-27b`

Model `perplexity-ai/pplx-decider-v1-27b` @ `5117a6c7fe73b19308dc1a6b0fb529a40c2ecad4`: Apache-2.0, Qwen3.8-27B
fine-tuned, bf16, ~52 GB. Script `scripts/r15_pplx_decider.py`, reusing R14's request builders, metrics and
McNemar code unchanged.

How the model scores (read in its release code):
- prompt `State / Question / Options`, with the options listed as single-token letter codes **in the criteria's
  own order**;
- one logit per code from a readout over the last hidden state;
- softmax at the release's saved temperature (2.2076), exactly as its own `predict` does.

Settings and items are identical to R14:
- zs: same state/instruction/criteria; the order is the dataset's label order;
- b24 and defs: the reproduction's bodies verbatim, i.e. its criteria order, as sent to Jev;
- train: the R12 sample.

Option order is part of the request and is not tuned.

**Training-data status: PARTLY DOCUMENTED.** The release ships its data builder (`source/src/autojev/data.py`).
Its tasks: vitaminc, **MASSIVE en-US (10,000 train rows)** and de-DE, boolq, squad2, paws, multinli,
civil_comments, aegis2, pubmedqa. Benchmark test splits are family-separated. The released checkpoint was trained
on a curated 73,000-row subset that is **not** bundled. Reading for our evaluation:
- **MASSIVE: exposed** (its train split);
- **HWU64: indirectly exposed** (43.7 % of HWU64 test utterances occur verbatim in MASSIVE train);
- **Banking77, CLINC150, MTOP, SNIPS, Bitext:** not among the documented sources. Because the curated subset is
  undisclosed, this is "not documented", not "verified clean".

Execution: batch 4 and the A1 memory watchdog, with the pattern widened to this script. The backbone is loaded
straight onto the GPU instead of CPU-then-GPU (device placement only), because CPU-then-GPU would briefly need
~2x the weights on unified memory.

## Amendment A3 (2026-10-07, before any pplx-decider run): pplx-decider v1 → v1.1

R15 now evaluates `perplexity-ai/pplx-decider-v1.1-27b` @ `3b45dead91dfa6d95aad6b95764a606fab2bf7a6` instead of
v1. No v1 run had started, so no result is discarded or selected. Reason: v1.1 is the current release and the
one on the public Decision Index board. Same backbone, same 255 letter codes and token ids, same prompt builder.

Differences read in the release code and `decision_config.json`:
- the 16 full-attention layers run **non-causally** (`attention_mode: noncausal_full_attention`); the
  linear-attention layers keep their recurrence; readout on the last token (`pooling: last`);
- calibration temperature 1.0087 (v1: 2.2076), applied by the release's own `predict`, unchanged;
- the card states that default causal inference does not reproduce the evaluated checkpoint, so scores come only
  from the release `DecisionModel` (transformers 5.17.0, SDPA), as for v1.

**Training-data status: PARTLY DOCUMENTED, more exposure than v1.** `training/data-manifest.json` lists tasksource
`multilingual/massive` (535 rows), `multilingual/mtop` (582) and `snips_built_in_intents` (396); the bundled
builder still includes MASSIVE en-US train. Reading for this evaluation:
- **MASSIVE, MTOP, SNIPS: exposed** (splits not stated);
- **HWU64: indirectly exposed** (MASSIVE overlap, as in A2);
- **Banking77, CLINC150, Bitext:** not in the manifest by name; reported as "not documented", not "clean".

Everything else in A2 (settings, items, request bodies, batch 4, watchdog, direct-to-GPU loading) is unchanged.

## Amendment A4 (2026-10-07, before any Arm-B run): equal-retrieval 24-shot setting `b24B`

Adds one setting for Clef-Flash, Clef and pplx-decider v1.1: Banking77 full test (3,080) with Jev **Arm B** request
bodies, i.e. equal retrieval with the clean 24-shot run (`results/predictions/banking77_24shot_clean.jsonl.gz`)
and with Jev Arm B (`docs/phases/2.0/jev/24shot/`, F15). The existing `b24` setting is the reproduction-retrieval
(Arm A) condition; both are reported, never pooled (ledger K4).

Bodies: the `b24` body with `state.labeled_examples` replaced by the clean run's 24 retrieved training examples
(`docs/phases/2.0/jev/train_pinned.json`), trimmed from the end only if the body exceeds 30,000 bytes — the logic of
`scripts/jev/run_typesafe.py`. **Every body's sha256 must equal the request manifest's Arm B hash, or the run
aborts.** Checked before this amendment on CPU: 3,080/3,080 Arm A and Arm B hashes match, retrieved ids match the
manifest, and Jev Arm B accuracy recomputed from `armB.jsonl.gz` = 93.47 % (ledger F15).

Comparisons (exact two-sided McNemar, same items): vs Jev Arm B; vs the clean 24-shot run (93.57 %); vs the same
model's `b24` (does the uncapped retrieval also help it?). Metrics as before (accuracy, macro-F1, ECE-10, Brier);
no log-loss comparison with Jev (ledger J2). Clef / Clef-Flash run on the release code; pplx-decider v1.1 on the
release code unless the Phase 20.8 SGLang parity gate has passed and a further amendment says otherwise.

Execution order (operator, 2026-10-07): after the serving parity work; Clef-Flash first. Batch sizes and the memory
watchdog as in A1.

## Amendment A5 (2026-10-08): remaining R14/R15 cells deferred; Clef 27B zero-shot run stopped by the operator

The Clef 27B `zs` run on CLINC150 + MTOP was **stopped by the operator** at 10 h 41 min (process killed,
exit 143). The reason is speed only: on the reference kernels it runs ~4.6 s/item, and the serving work in progress
(an SGLang path, gated on per-item parity with these release-code dumps) may make the remaining cells much faster.
The decision was not based on any result.
- The stop came during `clinc150_descriptions`. That cell has no output; nothing from it is used.
- `clinc150_names` had finished before the stop (all 5,500 items). Its dump was written when the cell finished and
  is kept: `results/predictions/clinc150_names_r14_clef_zs.jsonl.gz` (sha256 `443ebab4…`), metrics in
  `results/r14/summary_clef_zs_part-clinc150_names.json` (accuracy 0.9658, recomputed from the dump). The run's own
  summary file was never written, so that summary was written afterwards from the printed line and the dump.
- Not run yet and deferred: Clef 27B `zs` on CLINC150 descriptions, MTOP, and part B; `defs` and `train` for
  both models; Clef-Flash `zs` part B; all of R15 (pplx-decider v1.1); `b24B` (A4).
- These cells will run either on the release code (as before) or on a serving engine. A serving engine is used
  only if, for that model, it has passed the per-item parity gate against these release-code dumps, and only
  after a further amendment naming the engine, its version and the parity result. Every result records its engine.

## Amendment A6 (2026-10-08, before any scored R15 run): R15 runs on one NVIDIA GB300

All R15 (pplx-decider v1.1) runs, starting with `b24`, execute on **one NVIDIA GB300** instead of a GB10, for speed.
This changes only the hardware and software stack. The release `DecisionModel` code is unchanged, and so are the
model revision, bf16, batch 4, GPU-direct loading, items, request bodies and scoring.
- Environment: the pinned SGLang image `lmsysorg/sglang@sha256:868b0bd8…a8b77f` used only as a Python environment.
  It has torch 2.14.1+cu130 and transformers 5.17.0. `flash-linear-attention` and `causal-conv1d` are both absent,
  so the GDN layers use transformers' reference path. This also avoids the known causal-conv1d wrong-output issue
  on this GPU generation.
- The GPU is shared with another served model, so ms/item from these runs is not compared with GB10 timings.
- Every summary records the device name and package versions, as before.
- The 32-item smoke reference used to check the serving patch was produced on a GB10. It is a smoke, not a result,
  and nothing in R15 is computed from it.

## Amendment A7 (2026-10-08, before any R15 cell other than b24): remaining R15 cells are scored by a serving engine that passed the parity gate

Decided by the user after the gate result below. It applies to every R15 (pplx-decider v1.1) cell except `b24`,
which stays the release-code headline and the gate's reference. That is `zs` (7 datasets × names/descriptions),
`defs`, `train` and `b24B`.

- **Engine**: SGLang from the pinned image `lmsysorg/sglang@sha256:868b0bd8…a8b77f`, plus a 61-line patch
  (sha256 `fa7f6afc…4167`). The patch runs the checkpoint's full-attention layers non-causally, as its
  `decision_config.json` requires; stock SGLang ignores that field and runs them causally. Flags:
  `--attention-backend triton --chunked-prefill-size -1 --disable-radix-cache --disable-prefill-cuda-graph`.
  Hardware: one GB300 (A6).
- **Gate passed before this amendment.** The engine was run on the 3,080 `b24` bodies against the release-code dump
  `banking77_r15_pplx_decider_b24.jsonl.gz` (sha256 `b88be647…`), one request at a time:
  - argmax agreement 99.94 % (2 items differ);
  - accuracy difference +0.03 pt, McNemar b/c 1/0, p = 1;
  - max |Δp| 0.060, mean 4e-5;
  - 0 failed requests.
  These meet the pre-set thresholds (agreement ≥ 99.5 %, |Δacc| ≤ 0.5 pt with p ≥ 0.05, max |Δp| ≤ 0.10).
- **How cells run**: `scripts/r15_pplx_decider.py run … --engine-url … --engine-meta …`. Jobs, items, request
  bodies, metrics and dump format are unchanged.
  - Requests go one at a time, the gate's condition.
  - Any failed request aborts the whole run; no partial results.
  - Output files carry the suffix `_sgl`, and each summary records the engine (image, patch, gate summary).
- **Reporting**: release-code and engine results are labelled by engine and never pooled without that label. The
  2 / 3,080 argmax disagreements bound how much engine choice can move a cell.
