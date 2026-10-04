"""Quorum-as-a-service: a tiny FastAPI wrapper around the zero-shot ensemble.

Self-host this (any machine with CPU), expose it over HTTPS (a tunnel is fine), and point the static demo
Space — or your own app — at it. It imports the public `quorum` package, so no private code is involved.

    pip install -e ".[serve]"      # from the repo root
    uvicorn serve.app:app --host 0.0.0.0 --port 8000
    # then expose :8000 over HTTPS (e.g. a Cloudflare / ngrok tunnel) and set CORS_ORIGINS

Every answer goes through a permanent cache (serve/cache.py): a repeated or pre-seeded request is answered
without touching the models. On a miss the ensemble runs in a child process (serve/worker.py) that is started on
demand and stopped after an idle window, so a quiet server holds almost no memory.

Env:
    CORS_ORIGINS                comma-separated allowed origins for the browser demo (default "*"); set it to
                                your Space origin in production.
    QUORUM_DEVICE               "cpu" (default) or "cuda".
    QUORUM_CACHE_PATH           SQLite cache file (default serve/data/quorum_cache.sqlite3). Keep it on
                                persistent storage: entries never expire.
    QUORUM_CACHE_LIVE           "persist" (default): also store results computed for public requests (their
                                message + labels are kept in the cache file); "off": only seeded rows are served
                                from the cache, live results are never written.
    QUORUM_CACHE_MAX_LIVE       max stored live results (default 20000; seeded rows are unlimited).
    QUORUM_IDLE_UNLOAD_SECONDS  stop the model worker after this many idle seconds (default 900; <= 0 = never).
    QUORUM_JOB_TIMEOUT_SECONDS  timeout of one running prediction (default 120).
    QUORUM_LOAD_TIMEOUT_SECONDS timeout of a cold start: spawn + model load + warm-up (default 300).
    QUORUM_QUEUE_TIMEOUT_SECONDS  how long a request waits for its turn before a 503 (default 60).
    QUORUM_MAX_PENDING          predictions running + waiting before new misses get a 503 (default 4).
    QUORUM_PRELOAD              "1" to start the model worker at boot instead of on the first miss.
"""
import logging
import os
import sqlite3
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from quorum.ensemble import DEFAULT_EMBEDDERS, DEFAULT_NLI
from serve.cache import PredictionCache, build_response, cache_key, dump_response, normalize
from serve.worker import ModelWorker, WorkerBusy, WorkerCrashed, WorkerTimeout

MAX_LABELS = 50            # largest label set computed live on a cache miss
MAX_LABELS_REQUEST = 200   # largest label set accepted at all; > MAX_LABELS is served only from the cache
MODEL_IDS = [DEFAULT_NLI, *DEFAULT_EMBEDDERS]
CACHE_HEADER = "X-Quorum-Cache"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "data" / "quorum_cache.sqlite3"

_log = logging.getLogger("quorum.serve")


class PredictReq(BaseModel):
    message: str
    labels: List[str] = Field(..., min_length=2, max_length=MAX_LABELS_REQUEST)


def _cache_from_env() -> PredictionCache:
    return PredictionCache(os.environ.get("QUORUM_CACHE_PATH", str(DEFAULT_CACHE_PATH)),
                           max_live=int(os.environ.get("QUORUM_CACHE_MAX_LIVE", "20000")))


def _worker_from_env() -> ModelWorker:
    return ModelWorker(device=os.environ.get("QUORUM_DEVICE", "cpu"),
                       idle_unload_seconds=float(os.environ.get("QUORUM_IDLE_UNLOAD_SECONDS", "900")),
                       job_timeout_seconds=float(os.environ.get("QUORUM_JOB_TIMEOUT_SECONDS", "120")),
                       load_timeout_seconds=float(os.environ.get("QUORUM_LOAD_TIMEOUT_SECONDS", "300")),
                       queue_timeout_seconds=float(os.environ.get("QUORUM_QUEUE_TIMEOUT_SECONDS", "60")),
                       max_pending=int(os.environ.get("QUORUM_MAX_PENDING", "4")))


def create_app(cache: Optional[PredictionCache] = None, worker: Optional[ModelWorker] = None,
               persist_live: Optional[bool] = None) -> FastAPI:
    """Build the API. A cache/worker passed in stays owned by the caller; ones built here are closed on shutdown."""
    owns_cache, owns_worker = cache is None, worker is None
    cache = cache if cache is not None else _cache_from_env()
    worker = worker if worker is not None else _worker_from_env()
    if persist_live is None:
        persist_live = os.environ.get("QUORUM_CACHE_LIVE", "persist").lower() != "off"
    inflight: Dict[str, Future] = {}  # single-flight: concurrent identical misses share one computation
    inflight_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start_reaper()
        if os.environ.get("QUORUM_PRELOAD") == "1":
            threading.Thread(target=_preload, args=(worker,), name="quorum-preload", daemon=True).start()
        yield
        if owns_worker:
            worker.close()
        if owns_cache:
            cache.close()

    app = FastAPI(title="Quorum service", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
        allow_methods=["*"], allow_headers=["*"], expose_headers=[CACHE_HEADER],
    )
    app.state.cache, app.state.worker = cache, worker

    @app.get("/health")
    def health():
        return {"ok": True, "service": "quorum", "members": MODEL_IDS,
                "model_loaded": worker.loaded, "cache": cache.counts()}

    @app.post("/predict")
    def predict(req: PredictReq):
        message, labels = normalize(req.message, req.labels)
        if not message:
            raise HTTPException(422, "empty message")
        if len(labels) < 2:
            raise HTTPException(422, "need at least 2 distinct labels")
        key = cache_key(message, labels)
        body = cache.get(key)
        if body is not None:
            return Response(body, media_type="application/json", headers={CACHE_HEADER: "hit"})
        if len(labels) > MAX_LABELS:
            raise HTTPException(422, f"at most {MAX_LABELS} labels can be computed live on this server; "
                                     "larger label sets are only served for cached examples")
        with inflight_lock:
            shared = inflight.get(key)
            if shared is None:
                inflight[key] = mine = Future()
        if shared is not None:
            return _respond(shared.result())  # an identical request is computing it right now
        try:
            body = _compute(message, labels)
            if persist_live:  # stored before the in-flight entry is released, so no identical request recomputes
                try:
                    cache.put(key, body, "live")
                except sqlite3.Error:  # a cache problem must never cost the user their answer
                    _log.exception("could not store a live result in the cache")
            mine.set_result(body)
        except BaseException as e:
            mine.set_exception(e)
            raise
        finally:
            with inflight_lock:
                inflight.pop(key, None)
        return _respond(body)

    def _compute(message: str, labels: List[str]) -> str:
        try:
            ens, picks = worker.run(message, labels)
        except WorkerBusy:
            raise HTTPException(503, "the demo server is busy; please retry in a few seconds")
        except WorkerCrashed:
            _log.exception("model worker failed")
            raise HTTPException(503, "the model worker failed; please retry")
        except WorkerTimeout:
            _log.warning("prediction timed out (%d labels)", len(labels))
            raise HTTPException(504, "the prediction timed out; please retry")
        return dump_response(build_response(labels, ens, picks))

    def _respond(body: str) -> Response:
        return Response(body, media_type="application/json", headers={CACHE_HEADER: "miss"})

    return app


def _preload(worker: ModelWorker) -> None:
    try:
        worker.run("warm up", ["a", "b"])
    except Exception:
        _log.exception("model preload failed")


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
