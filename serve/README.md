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

First start downloads the three models (~1 GB) and loads them (~20 s on CPU).

## Endpoints

- `GET /health` → `{ ok, service, members }`
- `POST /predict` → body `{ "message": "...", "labels": ["...", "..."] }` (2–50 labels)
  returns `{ label, confidence, distribution, members: [{model, pick}] }`

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"message":"when will my card arrive?","labels":["card arrival","change pin","exchange rate"]}'
```

## Performance

Cost is dominated by the NLI member, which scores each candidate label, so latency scales ~linearly with the
number of labels. Measured on a shared CPU box (steady state): **~1.5–2 s at ~4 labels, ~5 s at 20 labels**; the
first request after startup warms up (~12 s), then it's steady. The NLI is loaded in **fp32 on CPU** (the fp16
checkpoint has no fast path on x86 — fp32 is ~4× faster there and identical in accuracy) and fp16 on GPU. Set
`QUORUM_DEVICE=cuda` on a GPU box for **sub-second** responses. On a busy shared machine expect variance from
CPU contention — give it its own box (or a small GPU) if latency matters.

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

Env: `CORS_ORIGINS` (default `*`), `QUORUM_DEVICE` (`cpu`/`cuda`), `PORT` (default 8000).
