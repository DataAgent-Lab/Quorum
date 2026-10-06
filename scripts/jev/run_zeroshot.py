#!/usr/bin/env python3
"""Zero-shot Jev run — implements docs/phases/2.0/jev/zeroshot/PROTOCOL.md (pre-registered).

    python scripts/jev/run_zeroshot.py --cache <dir outside the repo>

Writes, per arm, docs/phases/2.0/jev/zeroshot/{ds}_{arm}.jsonl (per-item result), raw_{ds}_{arm}.jsonl (full raw
response), plus ledger.jsonl and request_manifest.jsonl (sha256 of the exact bytes sent). Resumable. The key
(service/.env, JEV_API_KEY) is never printed or written.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum import data  # noqa: E402

OUT = ROOT / "docs" / "phases" / "2.0" / "jev" / "zeroshot"
ENDPOINT, MODEL = "https://api.typesafe.ai/v1/systemone", "jev-1.13.0"
REPRO = "https://raw.githubusercontent.com/simonmesmith/jev-banking77-experiment/5cac4ff"
INSTR = ("Which ONE intent best describes `customer_message`? "
         "Every message belongs to exactly one of these categories.")
WORKERS, ATTEMPTS, SPACING, CAP_USD, RATE = 8, 3, 0.12, 5.0, 0.042 / 1_000_000
RETRYABLE = {429, 500, 502, 503, 504, 529}
ARMS = [("banking77", "names"), ("banking77", "descriptions"), ("banking77", "repro_definitions"),
        ("mtop", "names"), ("mtop", "descriptions")]

env = dict(l.split("=", 1) for l in (ROOT / "service" / ".env").read_text().splitlines() if "=" in l)
KEY = env["JEV_API_KEY"].strip().strip('"').strip("'")
lock = threading.Lock(); state = {"spent": 0.0, "next": 0.0}


def redact(s):
    return str(s).replace(KEY, "[REDACTED]")


def write(name, rec):
    with lock, (OUT / name).open("a") as f:
        f.write(json.dumps(rec) + "\n")


def descriptions(ds):
    spec = importlib.util.spec_from_file_location("d", ROOT / "examples" / f"{ds}_descriptions.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return next(v for k, v in vars(m).items() if k.endswith("_DESCRIPTIONS"))


def build_arm(ds, arm, cache):
    d = data.load(ds); test = d["test"]
    if arm == "repro_definitions":
        f = cache / "repro_definitions.json"; g = cache / "repro_instructions.json"
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(f"{REPRO}/configs/descriptions.json", timeout=60).read())
        if not g.exists():            # the reproduction's instructions, taken from one of its own test requests
            req = json.loads(urllib.request.urlopen(f"{REPRO}/runs/test-v1/test-00000/request.json", timeout=60).read())
            g.write_text(json.dumps(req["questions"]["intent"]["instructions"]))
        crit, instr = json.loads(f.read_text()), json.loads(g.read_text())
        assert set(crit) == set(d["raw_labels"]), "reproduction definitions must cover the raw label set"
        gold = [d["raw_labels"][r["label"]] for r in test]
    else:
        labs = d["labels"]; D = descriptions(ds) if arm == "descriptions" else None
        # description files are keyed by raw labels (banking77) or humanized labels (mtop): resolve either form
        crit = {l: ((D[l] if l in D else D[raw]) if D else l) for l, raw in zip(labs, d["raw_labels"])}; instr = INSTR
        gold = [labs[r["label"]] for r in test]
    return [(i, {"model": MODEL, "state": {"customer_message": r["text"]},
                 "questions": {"intent": {"type": "choice", "instructions": instr, "criteria": crit}}}, gold[i])
            for i, r in enumerate(test)]


def call(ds, arm, i, payload, gold):
    body = json.dumps(payload).encode(); sha = hashlib.sha256(body).hexdigest(); status = None; err = ""
    for attempt in range(1, ATTEMPTS + 1):
        with lock:
            if state["spent"] + 0.01 > CAP_USD:
                raise RuntimeError("budget cap reached")
            wait = max(0.0, state["next"] - time.monotonic()); state["next"] = max(state["next"], time.monotonic()) + SPACING
        time.sleep(wait); t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
                    "Authorization": "Bearer " + KEY, "Content-Type": "application/json"}), timeout=60) as r:
                resp = json.loads(r.read()); status = r.status
            if resp.get("model") != MODEL:
                raise SystemExit(f"served model {resp.get('model')!r} != {MODEL!r} — aborting (protocol)")
            a = resp["answers"]["intent"]; toks = int(resp.get("usage", {}).get("input_tokens", 0))
            with lock:
                state["spent"] += toks * RATE
            write("ledger.jsonl", {"ds": ds, "arm": arm, "idx": i, "attempt": attempt, "ok": True,
                                   "ms": int(1000 * (time.time() - t0)), "input_tokens": toks, "cost_usd": toks * RATE})
            write(f"raw_{ds}_{arm}.jsonl", {"idx": i, "response": resp})
            write("request_manifest.jsonl", {"ds": ds, "arm": arm, "idx": i, "sha256": sha, "bytes": len(body)})
            return {"idx": i, "gold": gold, "choice": a["choice"], "correct": a["choice"] == gold,
                    "confidence": a.get("confidence"), "probabilities": a.get("probabilities"),
                    "model": resp.get("model"), "usage": resp.get("usage"), "attempts": attempt}
        except urllib.error.HTTPError as e:
            status = e.code; err = redact(e.read().decode(errors="replace"))[:300]
        except (urllib.error.URLError, TimeoutError) as e:
            err = redact(e)[:300]
        write("ledger.jsonl", {"ds": ds, "arm": arm, "idx": i, "attempt": attempt, "ok": False, "status": status, "error": err})
        if status is not None and status not in RETRYABLE:
            break
        time.sleep(2 ** attempt)
    return {"idx": i, "gold": gold, "failed": True, "status": status, "correct": False}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "ledger.jsonl").exists():
        state["spent"] = sum(json.loads(l).get("cost_usd", 0) for l in (OUT / "ledger.jsonl").read_text().splitlines())
    for ds, arm in ARMS:
        f = OUT / f"{ds}_{arm}.jsonl"
        done = {json.loads(l)["idx"] for l in f.read_text().splitlines()} if f.exists() else set()
        todo = [x for x in build_arm(ds, arm, cache) if x[0] not in done]
        with ThreadPoolExecutor(WORKERS) as ex:
            for rec in ex.map(lambda x: call(ds, arm, *x), todo):
                write(f"{ds}_{arm}.jsonl", rec)
        print(json.dumps({"arm": f"{ds}/{arm}", "spent_usd": round(state["spent"], 4)}), flush=True)


if __name__ == "__main__":
    main()
