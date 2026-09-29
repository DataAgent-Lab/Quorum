---
title: Quorum
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
license: mit
---

# Quorum — live demo (static frontend)

A jury of tiny open models decides an intent and tells you how sure it is. This Space is a **static** browser
frontend (`index.html`): you paste the URL of a running Quorum API and it calls it live.

- Run the API yourself: see [`serve/`](https://github.com/DataAgent-Lab/Quorum/tree/main/serve) in the repo
  (`pip install -e ".[serve]"` → `uvicorn serve.app:app`), then expose it over HTTPS and paste the URL here.
- Code + study: https://github.com/DataAgent-Lab/Quorum
- Collection: https://huggingface.co/collections/DataAgent/quorum-6abb3255e0fab20d66add39b

**Why static?** Hosting a compute (Gradio) Space needs a paid HF plan; a static Space is free and calls your
own self-hosted API — so the demo stays free and you keep the inference on your machine.

**Honest note.** This is the zero-shot tier — cheap and calibrated, but *not* open zero-shot state-of-the-art.
The 24-shot pipeline that edges the closed API Jev (0.932 vs 0.924 on Banking77) uses a gated LoRA adapter,
`DataAgent/Quorum-Reader-Qwen3-4B-Adapter`.
