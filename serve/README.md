# Quorum service — self-host the API

A tiny FastAPI wrapper (`serve/app.py`) around the zero-shot ensemble. Run it on any CPU machine, expose it
over HTTPS, and point the [static demo Space](../space/) — or your own app — at it. It imports the public
`quorum` package; no private code is involved.

## Run it

```bash
git clone https://github.com/DataAgent-Lab/Quorum.git && cd Quorum
pip install -e ".[serve]"
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

The models are not loaded at startup: the first prediction that is not in the cache downloads them (~1 GB, once)
and loads them in a worker process (see *Cache and memory* below).

## Endpoints

- `GET /health` → `{ ok, service, members, model_loaded, cache: {seed, live} }` (never loads the models)
- `POST /predict` → body `{ "message": "...", "labels": ["...", "..."] }` (2–50 labels computed live; up to 200
  labels are accepted for requests that are already in the cache)
  returns `{ label, confidence, distribution, members: [{model, pick}] }` and the header
  `X-Quorum-Cache: hit|miss`

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"message":"when will my card arrive?","labels":["card arrival","change pin","exchange rate"]}'
```

## Cache and memory

Every answer goes through a **permanent cache** (SQLite, `serve/cache.py`): a request that was answered before —
same message, same set of labels in any order — comes back in milliseconds without touching the models. Entries
never expire. On a miss, the ensemble runs in **one child process** (`serve/worker.py`) that is started on demand
and stopped after an idle window, so a quiet server holds ~70 MB instead of the ~2.5–4 GB the models take.

- **Cold start.** The first miss after startup or after an idle unload spawns the worker and loads the models:
  ~40–80 s on a busy CPU host. Later misses take the steady-state time below. Seeded requests are instant either way.
- **Load control.** One prediction runs at a time. At most `QUORUM_MAX_PENDING` misses are admitted (running +
  waiting); beyond that the API answers 503 at once, so slow misses never stall cache hits. Timeouts: 504.
- **Safety.** The cache stores a fingerprint of the model configuration (members, template, scale, dtype); opening it
  with a different configuration empties it, so an outdated answer is never served.

### Seeding

```bash
# benchmark predictions: per-item dumps in results/predictions/ ({dataset}_meta.json + {dataset}_{variant}.jsonl.gz)
python -m serve.seed_cache --predictions results/predictions          # --dry-run to validate + report only
# the demo page's example presets (computed locally)
python -m serve.seed_cache --demo
# spot-check dumped predictions against this machine's live model
python -m serve.spot_check --dataset banking77 --variant names --n 30
```

The importer refuses dumps produced with a different model configuration, validates every row, and prints the exact
accuracy per dataset/variant next to the published `results/*.json` number.

## Performance

Cost is dominated by the NLI member, which scores each candidate label, so latency scales ~linearly with the
number of labels. Measured on a shared CPU box (steady state): **~1.5–2 s at ~4 labels, ~5 s at 20 labels**. The
NLI is loaded in **fp32 on CPU** (the fp16 checkpoint has no fast path on x86 — fp32 is ~4× faster there and
identical in accuracy) and fp16 on GPU. Set `QUORUM_DEVICE=cuda` on a GPU box for **sub-second** responses. On a
busy shared machine expect variance from CPU contention — give it its own box (or a small GPU) if latency matters.
Cache hits take a few milliseconds.

## Expose it publicly (for the browser demo)

The static demo Space runs in the visitor's browser, so the API must be:

1. **Public + HTTPS.** Behind NAT? Use a tunnel that gives an HTTPS URL, e.g.
   - `cloudflared tunnel --url http://localhost:8000`  (Cloudflare Quick Tunnel), or
   - `ngrok http 8000`.
   Browsers block a HTTPS page calling an HTTP API (mixed content) — the URL must be `https://`.
2. **CORS-enabled.** Set the allowed origin(s) so the Space's JS may call it:
   ```bash
   CORS_ORIGINS="https://<your-space-host>" uvicorn serve.app:app --host 0.0.0.0 --port 8000
   ```
   `CORS_ORIGINS` defaults to `*` (open) — fine for a quick demo; tighten it to the Space origin in production.
3. **Availability.** The demo only works while this process + the tunnel are up.

Then point the demo at your API: the hosted Space defaults to a shared endpoint, so to use your own, fork the
Space (or edit `space/index.html`'s `API_BASE`) and set it to your `https://…` URL.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `CORS_ORIGINS` | `*` | allowed browser origins (comma-separated) |
| `QUORUM_DEVICE` | `cpu` | `cpu` or `cuda` |
| `PORT` | `8000` | when run as `python -m serve.app` |
| `QUORUM_CACHE_PATH` | `serve/data/quorum_cache.sqlite3` | cache file — keep it on persistent storage |
| `QUORUM_CACHE_LIVE` | `persist` | `persist`: also cache results for visitors' own inputs (their text is then kept in the cache file); `off`: only seeded rows |
| `QUORUM_CACHE_MAX_LIVE` | `20000` | max stored live results (rows > 16 KB are never stored); seeded rows are unlimited |
| `QUORUM_IDLE_UNLOAD_SECONDS` | `900` | stop the model worker after this idle time (`<= 0`: never) |
| `QUORUM_LOAD_TIMEOUT_SECONDS` | `300` | budget for a cold start (spawn + load + warm-up) |
| `QUORUM_JOB_TIMEOUT_SECONDS` | `120` | budget for one prediction |
| `QUORUM_QUEUE_TIMEOUT_SECONDS` | `60` | how long a miss waits for its turn before a 503 |
| `QUORUM_MAX_PENDING` | `4` | misses running + waiting before new ones get a 503 |
| `QUORUM_PRELOAD` | unset | `1` to start the model worker at boot |

## Tests

```bash
pip install -r serve/requirements-dev.txt
pytest serve/tests                              # fast suite (stub models, real worker processes)
QUORUM_REAL_MODEL_TEST=1 pytest serve/tests     # + the real-model end-to-end test (~2-3 min on CPU)
```
