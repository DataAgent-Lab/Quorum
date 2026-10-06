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
