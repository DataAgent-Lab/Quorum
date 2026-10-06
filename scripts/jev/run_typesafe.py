#!/usr/bin/env python3
"""Jev retrieval-swap run via TypeSafe's API — implements provenance/jev_retrieval_swap_protocol.md + its addendum.

    python scripts/jev/run_typesafe.py --cache <dir outside the repo> [--limit N]

Arm A: the reproduction's own per-item request, unchanged (model jev-1.13.0). Arm B: the same request with
state.labeled_examples replaced by the clean run's 24 retrieved training examples. Resumable: items already written
to results/jev/arm{A,B}.jsonl are skipped. The key (service/.env, JEV_API_KEY) is never printed or written.
"""
from __future__ import annotations
import argparse, gzip, json, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jev"
REPRO = "https://raw.githubusercontent.com/simonmesmith/jev-banking77-experiment/5cac4ff/runs/test-v1"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
BYTE_LIMIT, WORKERS, ATTEMPTS, SPACING = 30_000, 4, 3, 0.12
CAP_USD, RATE = 5.0, 0.042 / 1_000_000
RETRYABLE = {429, 500, 502, 503, 504, 529}

env = dict(l.split("=", 1) for l in (ROOT / "service" / ".env").read_text().splitlines() if "=" in l)
KEY = env["JEV_API_KEY"].strip().strip('"').strip("'")
lock = threading.Lock(); state = {"spent": 0.0, "next": 0.0}


def redact(s: str) -> str:
    return str(s).replace(KEY, "[REDACTED]")


def write(name: str, rec: dict):
    with lock, (OUT / name).open("a") as f:
        f.write(json.dumps(rec) + "\n")


def repro_request(cache: Path, i: int) -> dict:
    f = cache / f"request-{i:05d}.json"
    if not f.exists():
        f.write_bytes(urllib.request.urlopen(f"{REPRO}/test-{i:05d}/request.json", timeout=60).read())
    return json.loads(f.read_text())


def swapped(req: dict, retrieved: list, train: dict) -> tuple[dict, int]:
    out = json.loads(json.dumps(req)); ex = [{"message": train["text"][j], "intent": train["label_name"][j]} for j in retrieved]
    while True:
        out["state"]["labeled_examples"] = ex
        if len(json.dumps(out, ensure_ascii=True).encode()) <= BYTE_LIMIT or not ex:
            return out, len(ex)
        ex = ex[:-1]


def call(arm: str, i: int, payload: dict, kept: int) -> dict:
    for attempt in range(1, ATTEMPTS + 1):
        with lock:
            if state["spent"] + 0.01 > CAP_USD:
                raise RuntimeError("budget cap reached")
            wait = max(0.0, state["next"] - time.monotonic()); state["next"] = max(state["next"], time.monotonic()) + SPACING
        time.sleep(wait); t0 = time.time(); status = None
        try:
            req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read()); status = r.status
            if resp.get("model") != MODEL:
                raise SystemExit(f"served model {resp.get('model')!r} != {MODEL!r} — aborting (protocol addendum)")
            a = resp["answers"]["intent"]; toks = int(resp.get("usage", {}).get("input_tokens", 0))
            with lock:
                state["spent"] += toks * RATE
            write("ledger.jsonl", {"arm": arm, "idx": i, "attempt": attempt, "ok": True, "ms": int(1000 * (time.time() - t0)),
                                   "input_tokens": toks, "cost_usd": toks * RATE, "model": resp.get("model")})
            return {"idx": i, "arm": arm, "choice": a["choice"], "confidence": a.get("confidence"),
                    "probabilities": a.get("probabilities"), "model": resp.get("model"), "usage": resp.get("usage"),
                    "attempts": attempt, "kept": kept}
        except urllib.error.HTTPError as e:
            status = e.code; err = redact(e.read().decode(errors="replace"))[:300]
        except (urllib.error.URLError, TimeoutError) as e:
            err = redact(e)[:300]
        write("ledger.jsonl", {"arm": arm, "idx": i, "attempt": attempt, "ok": False, "status": status, "error": err})
        if status is not None and status not in RETRYABLE:
            break
        time.sleep(2 ** attempt)
    return {"idx": i, "arm": arm, "failed": True, "status": status, "kept": kept}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--limit", type=int, default=3080)
    a = ap.parse_args(); cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    train = json.loads((OUT / "train_pinned.json").read_text())
    clean = [json.loads(l)["retrieved_idx"] for l in gzip.open(ROOT / "results/predictions/banking77_24shot_clean.jsonl.gz", "rt")]
    done = {arm: {json.loads(l)["idx"] for l in (OUT / f"arm{arm}.jsonl").read_text().splitlines()} if (OUT / f"arm{arm}.jsonl").exists() else set()
            for arm in "AB"}
    if (OUT / "ledger.jsonl").exists():
        state["spent"] = sum(json.loads(l).get("cost_usd", 0) for l in (OUT / "ledger.jsonl").read_text().splitlines())

    def item(i):
        req = repro_request(cache, i)
        if i not in done["A"]:
            write("armA.jsonl", call("A", i, req, len(req["state"].get("labeled_examples", []))))
        if i not in done["B"]:
            payload, kept = swapped(req, clean[i], train)
            write("armB.jsonl", call("B", i, payload, kept))
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(item, range(a.limit)))
    print(json.dumps({"items": a.limit, "spent_usd": round(state["spent"], 4)}))


if __name__ == "__main__":
    main()
