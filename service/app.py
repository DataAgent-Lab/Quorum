"""Quorum Decision Service — FastAPI: define a task, (optionally) upload labels, train a calibrated head, and
predict typed, calibrated decisions. Serves the guided web console at /. One command via docker compose, or
`uvicorn service.app:app`. Everything runs on your own box; no data leaves.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from service import core

app = FastAPI(title="Quorum Decision Service", version="0.1.0")
WEB = Path(__file__).resolve().parent / "web"


class Example(BaseModel):
    text: str
    label: str


class DefineReq(BaseModel):
    classes: list[str]
    primitive: str = "choice"
    descriptions: dict[str, str] | None = None   # optional per-class natural-language descriptions (Layer 1)


class TrainReq(BaseModel):
    classes: list[str]
    examples: list[Example]
    primitive: str = "choice"


class DefinePoolReq(BaseModel):
    classes: list[str]
    examples: list[Example]                       # labeled example POOL (retrieved from, no head training)
    primitive: str = "choice"
    descriptions: dict[str, str] | None = None


class PredictReq(BaseModel):
    text: str
    tau: float = 0.0
    target_precision: float | None = None   # e.g. 0.95 -> use the conformal precision-guaranteed threshold


@app.get("/health")
def health():
    return {"ok": True, "embed_model": core.EMBED_MODEL, "n_tasks": len(core.list_tasks())}


@app.get("/api/tasks")
def list_tasks():
    return core.list_tasks()


@app.get("/api/tasks/{name}")
def get_task(name: str):
    try:
        return core.get_task(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/tasks/{name}/define")
def define(name: str, req: DefineReq):
    """Layer 1: register labels; predict works immediately (zero-shot), no training."""
    try:
        return core.define_task(name, req.classes, req.primitive, req.descriptions)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tasks/{name}/define_pool")
def define_pool(name: str, req: DefinePoolReq):
    """Layer 1.5: register a labeled example pool; predict serves the 24-shot retrieval anchor (GPU tier)."""
    try:
        return core.define_pool_task(name, req.classes, [e.model_dump() for e in req.examples],
                                     req.primitive, req.descriptions)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tasks/{name}/train")
def train(name: str, req: TrainReq):
    try:
        return core.train_task(name, req.classes, [e.model_dump() for e in req.examples], req.primitive)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/tasks/{name}/predict")
def predict(name: str, req: PredictReq):
    try:
        return core.predict_task(name, req.text, req.tau, req.target_precision)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
