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

Then paste the public `https://…` URL into the demo Space's "API URL" box (or set it as the Space default).

Env: `CORS_ORIGINS` (default `*`), `QUORUM_DEVICE` (`cpu`/`cuda`), `PORT` (default 8000).
