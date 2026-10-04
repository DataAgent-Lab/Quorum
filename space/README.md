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
- Run your own API: see [`serve/`](https://github.com/DataAgent-Lab/Quorum/tree/main/serve) and open the page with
  `?api=https://your-api.example`.
- `samples.json` is generated from the per-item benchmark predictions: `python -m serve.export_demo_samples`.
- Code + study: https://github.com/DataAgent-Lab/Quorum ·
  Collection: https://huggingface.co/collections/DataAgent/quorum-6abb3255e0fab20d66add39b

**Honest note.** This is the zero-shot tier — cheap and calibrated, but *not* open zero-shot state-of-the-art: the
closed API Jev (an independent reproduction, ~0.801 on Banking77) still leads zero-shot. The 24-shot pipeline that
edges it (0.932 vs 0.924 on Banking77) uses a gated LoRA adapter, `DataAgent/Quorum-Reader-Qwen3-4B-Adapter`.
