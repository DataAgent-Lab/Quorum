#!/usr/bin/env python3
"""R16 Bitext 24-shot, the Jev arm via TypeSafe's API — implements the research branch's
provenance/bitext_24shot_protocol.md §8 (+ A1) and docs/phases/2.0/bitext24/AMENDMENT_A2.md.

    python scripts/jev/run_bitext24.py --research-root <checkout> smoke   # 3 pool rows (not S), self excluded; discarded
    python scripts/jev/run_bitext24.py --research-root <checkout> run     # the one scored run (resumable)
    python scripts/jev/run_bitext24.py --research-root <checkout> finish  # per-item file + summary

Every body is rebuilt and sha-checked against the committed manifest (verified_bodies), and the exact bytes whose
sha256 is in the manifest are the bytes sent. Items are sent in a seeded random order (A2); if the US$1 cap stops the
run, unsent items are excluded from the Jev pairs, while attempted-and-failed items count as wrong. `finish` reports
counts only; accuracy and every comparison come from the §9 analysis script (committed before results are read). The key
(service/.env, JEV_API_KEY) is never printed or written. Spend stays in the ledger and is not published in the paper.
"""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.util, json, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "phases" / "2.0" / "jev" / "bitext24"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
WORKERS, ATTEMPTS, SPACING = 4, 3, 0.12
CAP_USD, RATE = 1.0, 0.042 / 1_000_000
ORDER_SEED = 20261012
RETRYABLE = {429, 500, 502, 503, 504, 529}

env = dict(l.split("=", 1) for l in (ROOT / "service" / ".env").read_text().splitlines() if "=" in l)
KEY = env["JEV_API_KEY"].strip().strip('"').strip("'")
lock = threading.Lock(); state = {"spent": 0.0, "next": 0.0, "capped": False}


def redact(s) -> str:
    return str(s).replace(KEY, "[REDACTED]")


def write(name: str, rec: dict):
    with lock, (OUT / name).open("a") as f:
        f.write(json.dumps(rec) + "\n")


def research(rr: Path):
    sys.path.insert(0, str(rr))
    spec = importlib.util.spec_from_file_location("bb", rr / "scripts" / "bitext24_bodies.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def call(i: int, data: bytes, ledger: str) -> dict:
    for attempt in range(1, ATTEMPTS + 1):
        with lock:
            if state["spent"] + 0.01 > CAP_USD:
                state["capped"] = True
                return {"idx": i, "unsent": True}
            wait = max(0.0, state["next"] - time.monotonic()); state["next"] = max(state["next"], time.monotonic()) + SPACING
        time.sleep(wait); t0 = time.time(); status = None
        try:
            req = urllib.request.Request(ENDPOINT, data=data, method="POST",
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read()); status = r.status
            if resp.get("model") != MODEL:
                raise SystemExit(f"served model {resp.get('model')!r} != {MODEL!r} — aborting")
            a = resp["answers"]["intent"]; toks = int(resp.get("usage", {}).get("input_tokens", 0))
            with lock:
                state["spent"] += toks * RATE
            write(ledger, {"idx": i, "attempt": attempt, "ok": True, "ms": int(1000 * (time.time() - t0)),
                           "input_tokens": toks, "cost_usd": toks * RATE, "model": resp.get("model")})
            return {"idx": i, "choice": a["choice"], "confidence": a.get("confidence"),
                    "probabilities": a.get("probabilities"), "model": resp.get("model"), "usage": resp.get("usage"),
                    "attempts": attempt}
        except urllib.error.HTTPError as e:
            status = e.code; err = redact(e.read().decode(errors="replace"))[:300]
        except (urllib.error.URLError, TimeoutError) as e:
            err = redact(e)[:300]
        write(ledger, {"idx": i, "attempt": attempt, "ok": False, "status": status, "error": err})
        if status is not None and status not in RETRYABLE:
            break
        time.sleep(2 ** attempt)
    return {"idx": i, "failed": True, "status": status, "attempts": attempt}


def smoke(bb):
    raw, names, T, S, P, test = bb.split(); desc = bb.descriptions(names)
    bm = bb.StableBM25([T[i]["text"] for i in P]); out = []
    for pos in (0, 7000, 14000):                                   # pool rows; P excludes S by construction
        r = T[P[pos]]; top = [P[x] for x in bm.top(r["text"], bb.SHOTS + 1) if x != pos][:bb.SHOTS]
        b, _ = bb.body(r["text"], [{"message": T[j]["text"], "intent": raw[T[j]["label"]]} for j in top], raw, desc)
        res = call(-1 - pos, json.dumps(b).encode(), "smoke_ledger.jsonl")
        assert set(res.get("probabilities") or {}) == set(raw), "probabilities do not cover the 27 raw intents"
        out.append(res["usage"]["input_tokens"])
    print(json.dumps({"smoke_input_tokens": out, "projected_usd_for_5375": round(float(np.mean(out)) * RATE * 5375, 3)}))


def run(bb):
    test, raw, bodies, man = bb.verified_bodies()                  # rebuilt and sha-checked (aborts on mismatch)
    datas = []
    for b, m in zip(bodies, man):
        d = json.dumps(b).encode()
        assert hashlib.sha256(d).hexdigest() == m["sha256"], f"bytes to send differ from the manifest at {m['idx']}"
        datas.append(d)
    order = [int(i) for i in np.random.default_rng(ORDER_SEED).permutation(len(bodies))]
    done = set()
    if (OUT / "jev.jsonl").exists():
        done = {json.loads(l)["idx"] for l in (OUT / "jev.jsonl").read_text().splitlines()
                if not json.loads(l).get("unsent")}
    for led in ("ledger.jsonl", "smoke_ledger.jsonl"):             # the cap covers the smoke calls too
        if (OUT / led).exists():
            state["spent"] += sum(json.loads(l).get("cost_usd", 0) for l in (OUT / led).read_text().splitlines())

    def item(i):
        if i not in done:
            write("jev.jsonl", call(i, datas[i], "ledger.jsonl"))
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(item, order))
    print(json.dumps({"items": len(order), "capped": state["capped"]}))


def finish(bb):
    test, raw, bodies, man = bb.verified_bodies()
    recs = {}
    for l in (OUT / "jev.jsonl").read_text().splitlines():
        r = json.loads(l); recs[r["idx"]] = r                        # last record per item wins (resume)
    pos = {k: j for j, k in enumerate(raw)}
    with gzip.open(OUT / "bitext_r16_jev.jsonl.gz", "wt") as f:
        for i, t in enumerate(test):
            r = recs.get(i, {"unsent": True})
            p = r.get("probabilities")
            f.write(json.dumps({"idx": i, "gold": int(t["label"]),
                                "pred": pos.get(r.get("choice"), -1) if "choice" in r else None,
                                "probs": [float(p[k]) for k in raw] if p else None,
                                "served_model": r.get("model"), "attempts": r.get("attempts"),
                                "failed": bool(r.get("failed")), "unsent": bool(r.get("unsent"))}) + "\n")
    led = [json.loads(l) for l in (OUT / "ledger.jsonl").read_text().splitlines()]
    sent = [r for r in recs.values() if not r.get("unsent")]
    ok = [r for r in sent if "choice" in r]
    meta = {"protocol": "research provenance/bitext_24shot_protocol.md @ 9d6dc44 + A1 3d84cc3; AMENDMENT_A2 (this repo)",
            "manifest_sha256": json.loads((Path(bb.OUT) / "manifest_meta.json").read_text())["manifest_sha256"],
            "model": MODEL, "served_models": sorted({r.get("model") for r in ok}), "order_seed": ORDER_SEED,
            "items": len(test), "sent": len(sent), "answered": len(ok), "failed": sum(bool(r.get("failed")) for r in sent),
            "unsent": len(test) - len(sent), "attempts_total": len(led),
            "retried_items": sum(r.get("attempts", 1) > 1 for r in sent),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "summary_bitext_r16_jev.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--research-root", required=True)
    ap.add_argument("phase", choices=["smoke", "run", "finish"]); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); bb = research(Path(a.research_root).resolve())
    {"smoke": smoke, "run": run, "finish": finish}[a.phase](bb)
