"""Quorum-as-a-service: a tiny FastAPI wrapper around the zero-shot ensemble.

Self-host this (any machine with CPU), expose it over HTTPS (a tunnel is fine), and point the static demo
Space — or your own app — at it. It imports the public `quorum` package, so no private code is involved.

    pip install -e ".[serve]"      # from the repo root
    uvicorn serve.app:app --host 0.0.0.0 --port 8000
    # then expose :8000 over HTTPS (e.g. a Cloudflare / ngrok tunnel) and set CORS_ORIGINS

Env:
    CORS_ORIGINS   comma-separated allowed origins for the browser demo (default "*"); set it to your Space
                   URL in production, e.g. https://huggingface.co  (or the exact Space origin).
    QUORUM_DEVICE  "cpu" (default) or "cuda".
"""
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quorum import ZeroShotEnsemble

MAX_LABELS = 50
_CLF = ZeroShotEnsemble(device=os.environ.get("QUORUM_DEVICE", "cpu"))
_MEMBERS = ["PrismNLI-0.4B", "bge-large", "bge-base"]

app = FastAPI(title="Quorum service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"], allow_headers=["*"],
)


class PredictReq(BaseModel):
    message: str
    labels: List[str] = Field(..., min_length=2, max_length=MAX_LABELS)


@app.get("/health")
def health():
    return {"ok": True, "service": "quorum", "members": _CLF.model_names}


@app.post("/predict")
def predict(req: PredictReq):
    message = (req.message or "").strip()
    labels, seen = [], set()
    for l in req.labels:
        s = (l or "").strip()
        if s and s not in seen:
            seen.add(s); labels.append(s)
    if not message:
        raise HTTPException(422, "empty message")
    if len(labels) < 2:
        raise HTTPException(422, "need at least 2 distinct labels")
    members, ens = _CLF.member_and_ensemble_proba(message, list(range(len(labels))), label_texts=labels)
    order = sorted(range(len(labels)), key=lambda i: -float(ens[i]))
    dist = {labels[i]: round(float(ens[i]), 4) for i in order}
    top = order[0]
    return {
        "label": labels[top],
        "confidence": round(float(ens[top]), 4),
        "distribution": dist,
        "members": [{"model": _MEMBERS[m], "pick": labels[int(members[m].argmax())]} for m in range(len(members))],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
