---
title: Quorum
emoji: ⚖️
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: mit
---

# Quorum — ask the jury

A jury of three tiny open models reads a message, each casts a vote, and Quorum returns the verdict with a confidence
you can act on. Every example on the page is a **real test sentence** from one of seven public intent benchmarks
(Banking77, CLINC150, HWU64, MASSIVE, MTOP, SNIPS, Bitext), so you can see when the jury gets it right — and when it
doesn't. You can also type your own sentence and labels.

- The page is static (`index.html` + `samples.json`) and calls a self-hosted Quorum API. Benchmark examples are
  answered instantly from the API's cache; your own sentences are computed live (the first one after a quiet spell
  takes a minute or two while the models wake up). Sentences you type are kept on the demo server so repeats answer
  instantly.
- Run your own API: see [`serve/`](https://github.com/DataAgent-Lab/Quorum/tree/main/serve) and set `DEFAULT_API` in
  your copy of `index.html` (when the page is served from localhost, `?api=https://your-api.example` also works).
- `samples.json` is generated from the per-item benchmark predictions: `python -m serve.export_demo_samples`.
- Code + study: https://github.com/DataAgent-Lab/Quorum ·
  Collection: https://huggingface.co/collections/DataAgent/quorum-6abb3255e0fab20d66add39b

**Honest note.** This is the zero-shot tier: three models under 1B that run on a CPU, but *not* open zero-shot
state-of-the-art. Measured on the same test sentences and label text, the closed model Jev is clearly ahead zero-shot
on all five benchmarks we ran (Banking77: 0.803 vs 0.716 for this jury with an NLI juror never trained on Banking77).
Banking77, MASSIVE and HWU64 are not fully held out here: the NLI juror's base model was trained on the Banking77 and
MASSIVE train splits, and 44% of HWU64's test sentences appear in MASSIVE's train split. Only the 24-shot pipeline,
with a 4B reader (gated LoRA adapter `DataAgent/Quorum-Reader-Qwen3-4B-Adapter`), ties Jev on Banking77 (0.932 vs
0.924, not significant; 0.936 vs 0.935 given the same retrieved examples). Details:
https://academy.idataagent.com/research/quorum
