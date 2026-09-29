# Quorum Decision Service

A self-hosted decision model: **define a decision, (optionally) upload labels, and predict typed, calibrated
decisions** — the full loop, on your own box, no data leaves. It wraps the `quorum` library behind a small
FastAPI + a guided web console.

## Run

```bash
# Docker (one command)
cd service && docker compose up --build
# open http://localhost:8080

# or from a Python env at the repo root
pip install -e ".[service]"
uvicorn service.app:app --host 0.0.0.0 --port 8080
```

First run downloads the zero-shot models (~1.7 GB) into the HF cache; after that it is offline. Trained tasks
persist under `service/data/registry`.

The console walks you through: **1** define the decision + labels → *Use now* predicts immediately (zero-shot,
no training) → **2** paste `text | label` examples and train a calibrated head → **3** Playground: type an
input, see the calibrated distribution + confidence + which layer served, and slide the **confidence gate** to
decide auto-act vs escalate-to-human.

## Three layers

- **Layer 1 — zero-shot, no training (default).** `quorum.ZeroShotEnsemble`: `PrismNLI-0.4B` (NLI) + `bge-large`
  + `bge-base` (cosine), logprob-mean, no per-task calibration. Full Banking77 test **0.780**, above the best
  single member (0.736), and it beats its own best component on 5 of 6 different-domain intent datasets. All
  members are sub-1B → runs on CPU. A lighter tier (`LAYER1_MODE=single`, NLI only) trades accuracy for speed.
  *Optional `descriptions`*: pass a natural-language description per class as the label text — the gain scales
  with how opaque the names are (small when names are already descriptive; up to +9 points when they are short
  or ambiguous).
- **Layer 1.5 — 24-shot retrieval anchor, no head training (GPU tier).** Give it a labeled example POOL and it
  serves `quorum.fewshot.RetrievalEnsemble`: 24 BM25-retrieved examples read in-context by the gated 4B reader
  (`Qwen/Qwen3-4B` + the released LoRA adapter) + a `bge-large` kNN over the same 24, geometric mean. On the
  full Banking77 test this code path scores **0.932 — above the reproduced closed API (Jev) 0.924** (our
  study's best configuration reached **0.938** and *significantly* beat that reproduction, paired McNemar
  p=0.0014; see the repo README). Needs a GPU and access to the gated adapter (`RLCD_ADAPTER`, default
  `DataAgent/Quorum-Reader-Qwen3-4B-Adapter`). Layers 1/2 stay on CPU and never load it.
- **Layer 2 — trained on your labels.** A frozen embedder (default `bge-base`) + a linear head + temperature
  calibration. `predict` auto-serves Layer 2 once a task is trained, else Layer 1.5 if a pool is defined, else
  Layer 1.

## API

| method | path | body | returns |
|---|---|---|---|
| POST | `/api/tasks/{name}/define` | `{classes[], primitive, descriptions?:{class: text}}` | task meta (Layer 1) |
| POST | `/api/tasks/{name}/define_pool` | `{classes[], examples:[{text,label}], primitive, descriptions?}` | task meta (Layer 1.5) |
| POST | `/api/tasks/{name}/train` | `{classes[], examples:[{text,label}], primitive}` | metrics (accuracy, ECE, temperature) |
| POST | `/api/tasks/{name}/predict` | `{text, tau}` or `{text, target_precision}` | `{label, confidence, distribution, abstain, gate}` |
| GET | `/api/tasks` / `/api/tasks/{name}` | — | task registry |
| GET | `/health` | — | status |

```bash
curl -s localhost:8080/api/tasks/support-intent/predict \
  -H 'content-type: application/json' -d '{"text":"charged twice","tau":0.7}'
```

Primitives: `choice` (pick one of N), `score` (ordered levels low→high, returns an expected level), `noul`
(yes/no, returns `probability_true`).

## Confidence gate — two modes

- **Raw** (`tau`): auto-act when confidence ≥ τ. Fast, but only "calibrated on average".
- **Conformal** (`target_precision`, e.g. 0.95): the service returns a threshold that **GUARANTEES** the
  auto-acted precision ≥ your target with high probability (finite-sample — Wilson lower bound + Bonferroni over
  the threshold grid), estimated on a held-out split of your labels. If the calibration set is too small to
  guarantee the target, it says so instead of pretending.
