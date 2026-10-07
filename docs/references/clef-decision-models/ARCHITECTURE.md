# Cloudflare Clef / Clef-Flash — architecture and training, dissected

Reference for R14 (`scripts/r14_clef.py`, protocol `provenance/clef_eval_protocol.md`). Written 2026-10-07.

Every statement is tagged with its basis:
- **[code]** read in the release's `joint_schema_model.py`;
- **[config]** from the release's `config.json` / `joint_head_config.json`;
- **[blog]** Cloudflare's announcement, not independently verifiable;
- **[measured]** our own runs.

Sources:
- `Cloudflare/clef` @ `2f3de3dd85f379784083b0814d997ab627200f0c`;
- `Cloudflare/clef-flash` @ `17f0b0ad64efb65d273590632833508766b2aae6`. Both releases ship the same `joint_schema_model.py`, byte-identical [code];
- <https://blog.cloudflare.com/clef-decision-models> (read 2026-10-06/07).

## 1. What it is

Clef is a "decision model". Given a `state` (text, JSON, images or video) and a schema of typed questions (`choice`, `score`, `noul` = true/false), it returns **one logit per allowed option of every question in a single forward pass**. There is no text generation and no output parsing. The API is the Jev/SystemOne request/response body [code, blog].

Inference runs in two parts [code]:
1. a **prefill-only** pass of a Qwen backbone (`use_cache=False`, final hidden states only);
2. a small **joint schema head** that turns those hidden states into option logits.

The decision step is non-autoregressive [blog, code].

## 2. Input encoding [code]

One flat token sequence:

```
<|im_start|>system
Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>
<|im_start|>user
STATE:
<state, rendered as compact JSON with sorted keys, or the raw string>
[image/video placeholders, if any]

SCHEMA FIELDS:

FIELD 1
ID: <question id>
TYPE: <choice|score|noul>
INSTRUCTION: <instructions, or the question id if empty>      <- question span
ALLOWED OPTIONS:
OPTION 1: {"description": "...", "option_id": "..."}          <- option span (one per option)
...
END FIELD
...
<|im_end|>
<|im_start|>assistant
<think>

</think>

JOINT SCHEMA DECISIONS:
```

- **`choice` options are sorted by option id.** Order in the request does not matter, and the model never sees a caller-chosen order.
- `score` options are indexed 0..n-1.
- `noul` options default to fixed descriptions for `true` and `false`.
- Default `max_length` is **16,384 tokens**.
  - The schema must fit entirely; otherwise the request raises an error.
  - When the input is too long, the `state` is cut from its end.
  - The blog advertises a 64k context. The release code defaults to 16k.

## 3. Backbone [config, code]

| | Clef | Clef-Flash |
|---|---|---|
| base (per card) | Qwen3.8-27B | Qwen3.5-9B |
| class | `Qwen3_5ForConditionalGeneration` (includes vision encoder) | same |
| layers | 64 = **48 linear-attention (Gated DeltaNet) + 16 full-attention** (full every 4th) | 32 = 24 + 8 |
| hidden size | 5,120 | 4,096 |
| attention heads / KV heads / head dim | 24 / 4 / 256 | 16 / 4 / 256 |
| GDN value / key heads, head dim | 48 / 16, 128 | 32 / 16, 128 |
| MLP intermediate | 17,408 | 12,288 |
| vocab | 248,320 | 248,320 |
| release size (bf16) | 55.0 GB | 19.1 GB |

For text-only records the head reads `model.language_model` directly. With images or video, the full multimodal model runs [code].

- Per the loader docstring, the shipped backbone is "merged". The blog says rank-256 LoRA adapters were trained; the release ships them merged into the weights [code docstring + blog].
- We have **not** checked how far the merged weights differ from the public Qwen base.

## 4. Joint schema head [code, config]

Config, identical for both models apart from the input size: `width 1024, routing_layers 2, layers 4, heads 16, feedforward 4096`.

Parameter count computed from these dimensions and the module definitions:
- Clef: 128.1 M;
- Clef-Flash: 121.8 M.

At bf16 this is 256.1 MB / 243.5 MB, exactly the size of the released `joint_head.safetensors`. That confirms the head structure described below.

Per record:
1. **Memory.** `LayerNorm(hidden_states)` of the whole sequence (state + schema), projected to width 1024.
2. **Question vector** = mean of the hidden states over the instruction span. **Global vector** = hidden state of the last token.
3. **Option representations.** Each option gets two vectors:
   - *context*: the mean hidden state over its `{"description", "option_id"}` span;
   - *lexical*: the mean of the **output-embedding (LM-head) rows** of its tokens.

   Option query = `proj(context) + proj(lexical) + proj(question)`.
4. **Option-specific evidence routing.** 2 pre-norm cross-attention + FFN layers. In each, *all options of all questions* attend to the full-sequence memory, so each option pulls the state evidence relevant to it.
5. **Field vectors.** For each question:
   - `proj(question)`
   - `+ LayerNorm(softmax-weighted summary of its routed options)`
   - `+ proj(global)`
   - `+ type embedding (noul/choice/score)`.
6. **Joint cross-field attention.** 4 `TransformerDecoderLayer`s (pre-norm, GELU). The field vectors self-attend to each other, so questions in one request influence one another, and they cross-attend back to the memory.
7. **Scoring.** For each option of a field:

   `logit = s_p · cos(lexical_anchor, normalize(question + global)) + sigmoid(g) · ( s_j · cos(field, option) + MLP([f, o, f⊙o, |f−o|]) )`

   - The first term is a **lexical prior**: how well the option's own token embedding matches the question/state summary.
   - The second term is a gated **joint** score from the routed representations.
   - The learned scales `s_p` and `s_j` are clamped to ≤ 100.
8. **Probabilities.** Softmax over the options of each question. There is no temperature or other post-hoc calibration step in the release.

Implementation notes [code]:
- The head runs a Python loop over the records in a batch, and over the questions and options within each record.
- Batches are right-padded; the causal backbone makes padding harmless. [measured] Batch-1 vs batch-8 on 16 items: 0 argmax changes, max |Δp| 7.6e-3 (bf16).

## 5. Training (as stated, not verifiable) [blog]

- **Base frozen; trained parts:** Qwen3.8-27B (Clef) / Qwen3.5-9B (Clef-Flash) is frozen. The routing head is "jointly optimized alongside rank-256 low-rank adapters".
- **Losses:** label-smoothed cross-entropy over the valid schema outputs, plus a Brier loss "to refine probability calibration".
- **Data:** "our own internal synthetic datasets permutating field orders, prompts, and schema structures". **No source datasets are named.** There is no statement that the public benchmarks they report (BANKING77, CLINC150+OOS, …) or their training splits were excluded.
- **RLCD**, "Reinforcement Learning for Calibrated Decisions", is a secondary objective. It:
  - gives partial credit to adjacent ordinal choices;
  - rewards fully correct records;
  - applies a reference penalty "to prevent distribution shift".
- **Lineage:** it follows Cloudflare's earlier DiffusionGemma-based prototype, which exposed per-option logprobs of an LLM. Clef replaces that with the dedicated head.

Consequence for evaluation: **training data status UNVERIFIED** (see the protocol). The cards report BANKING77 macro-F1 94.2 / 90.9 and CLINC150+OOS 97.4 / 66.8. These come from an undisclosed protocol (Decision Index 0.2.1, `multimodalart/jev-decision-index`), and the same table lists Jev at 79.7 on BANKING77, versus the public reproduction's 0.924 accuracy under 24-shot retrieval. The numbers are therefore not comparable with ours as published.

## 6. How it differs from the approaches in this repository

| aspect | Clef | Quorum 24-shot reader (this repo) |
|---|---|---|
| backbone | Qwen3.8-27B / Qwen3.5-9B (merged LoRA r256) | Qwen3-4B + LoRA |
| scoring | head over pooled hidden states of each option's description span + lexical prior | logit of a single marker token per option at the answer position |
| option order | canonical (sorted ids); order-invariant by construction | randomly permuted orders, averaged (1 order in the clean config) |
| multiple questions | joint, one pass | one question per pass |
| retrieval / examples | none built in; examples only if the caller puts them in `state` | BM25-retrieved shots in the prompt + a kNN member over the same shots |
| calibration | Brier term in training; no post-hoc step | temperature/weights fitted on a held-out selection split |

Both train with cross-entropy plus a Brier term.

## 7. What we measured so far [measured, R14]

- **Banking77, the reproduction's exact `retrieved24` request bodies** (byte-identical to what Jev received, 3,080 items), Clef-Flash:
  - accuracy 0.9549, macro-F1 0.9546;
  - McNemar vs Jev 115/20 (p = 2e-17);
  - log loss 0.390, **ECE 0.176**: accurate but poorly calibrated in this setting;
  - training-data caveat applies.
- **Speed on GB10 (sm_121a):** Clef-Flash 1.55 s/item at ≤ 3.7k prompt tokens, batch 8, using transformers' **reference PyTorch** GDN / causal-conv1d paths. `flash-linear-attention` 0.5.2 fails to compile here (Triton `ptxas --gpu-name=sm_121a`: "Internal Triton PTX codegen error"). GB300 (sm_103) is untested.
- **Cross-host determinism:** identical image, model files and data on two GB10 hosts give bit-identical probabilities on 64 items.

## 8. Open questions

1. Training data provenance: were benchmark train or test splits used to generate the "synthetic" data?
2. How far do the merged backbone weights move from the public Qwen base? Checkable by comparing tensors; not done.
3. The release default context is 16k; the blog claims 64k.
4. The lexical prior uses LM-head rows of the option tokens. Option ids and descriptions therefore carry signal through two paths, embedding identity and contextual hidden states. This is one reason the `names` and `descriptions` variants of the zero-shot setting are reported separately.
