---
title: Quorum
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Quorum — live demo

A jury of tiny open models decides an intent and tells you how sure it is. Type a message and a few candidate
intent labels; a **zero-shot** ensemble of three sub-1B open models — `Jaehun/PrismNLI-0.4B` +
`BAAI/bge-large-en-v1.5` + `BAAI/bge-base-en-v1.5` (geometric mean) — returns a calibrated distribution and
each member's vote. **No training, no calibration**, runs on CPU.

- Code + study: https://github.com/DataAgent-Lab/Quorum
- Collection: https://huggingface.co/collections/DataAgent/quorum-6abb3255e0fab20d66add39b

**Honest note.** This is the zero-shot tier — cheap and calibrated, but *not* open zero-shot state-of-the-art
(a 7–9B LLM beats it on some datasets). The stronger 24-shot pipeline that edges the closed API Jev
(0.932 vs 0.924 on Banking77, reproducible; 0.938 with paired-significant p=0.0014 in the study) uses a gated
LoRA adapter — see `DataAgent/Quorum-Reader-Qwen3-4B-Adapter`.
