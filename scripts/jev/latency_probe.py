#!/usr/bin/env python3
"""Jev API latency at long inputs (requested by the user 2026-10-14, for comparison with pplx on one GB300).

    python scripts/jev/latency_probe.py              # first run: repeated identical bodies per size
    python scripts/jev/latency_probe.py --distinct   # rerun (user request): every body distinct, sizes from plan.json
    python scripts/jev/latency_probe.py --distinct --top-only --scale 0.93
        # shuffled text has ~7% more tokens per character, so the distinct top-size bodies (~33.6k) were refused;
        # this repeats only the top size with 7% fewer characters (failed records stay in the log)
    python scripts/jev/latency_probe.py --distinct --top-only --scale 0.93 --seed-offset 1000
        # the previous line restarted seeds at 1, so its bodies shared prefixes with earlier calls' bodies (same
        # shuffled order, shorter); superseded by this run, whose seeds are used by no other call

States are concatenated public Bitext training utterances (CDLA-Sharing-1.0); one small choice question per call.
Records per call: target size, attempt order, client wall latency (send → full response), HTTP status,
usage.input_tokens, served model id, and for errors the first 300 characters of the error body. Jev's answer text is
never stored. Evaluation only; nothing is trained or tuned on outputs. The key (service/.env, JEV_API_KEY) is never
printed or written. A US$0.20 cap stops the probe; spend stays out of the paper.
"""
from __future__ import annotations
import argparse, json, random, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bitext24_bodies as bb                                        # noqa: E402  (public Bitext split)

OUT = ROOT / "docs" / "phases" / "2.0" / "jev" / "latency"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
CAP_USD, RATE = 0.20, 0.042 / 1_000_000
SIZES = [1000, 4000, 8000, 16000, 24000, 31500]
OVER = [33000, 40000]
REPS, CONC = 5, (5, 16)

env = dict(l.split("=", 1) for l in (ROOT / "service" / ".env").read_text().splitlines() if "=" in l)
KEY = env["JEV_API_KEY"].strip().strip('"').strip("'")
lock = threading.Lock(); spent = [0.0]; LOG = ["calls.jsonl"]; DISTINCT = [False]; counter = [0]
SCALE = [1.0]; TOP_ONLY = [False]
QUESTION = {"topic": {"type": "choice",
                      "instructions": "Which topic do most of the customer messages in the state concern?",
                      "criteria": {"orders": "orders, delivery, shipping or cancellation",
                                   "account": "account, password or registration",
                                   "payment": "payment, refund or invoice",
                                   "other": "anything else"}}}


def body(n_utts: int, texts: list[str]) -> bytes:
    return json.dumps({"model": MODEL, "state": {"customer_messages": "\n".join(texts[:n_utts])},
                       "questions": QUESTION}).encode()


def call(tag: str, target: int, n_utts: int, texts: list[str]) -> dict:
    with lock:
        if spent[0] + 40_000 * RATE > CAP_USD:
            return {"tag": tag, "target": target, "skipped": "cap"}
    if DISTINCT[0]:                                                 # a fresh order per call: no two bodies share a prefix
        with lock:
            counter[0] += 1; seed = counter[0]
        goal = SCALE[0] * len("\n".join(texts[:n_utts]))             # the first run's characters, times the scale
        order = random.Random(seed).sample(texts, len(texts)); acc, n_utts = 0, 0
        while acc < goal:
            acc += len(order[n_utts]) + 1; n_utts += 1
        texts = order
    data = body(n_utts, texts); t0 = time.perf_counter(); rec = {"tag": tag, "target": target, "body_bytes": len(data),
                                                                 "sent_at": time.strftime("%H:%M:%S", time.gmtime())}
    try:
        req = urllib.request.Request(ENDPOINT, data=data, method="POST",
                                     headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read()); rec["status"] = r.status
        toks = int(resp.get("usage", {}).get("input_tokens", 0))
        with lock:
            spent[0] += toks * RATE
        rec |= {"input_tokens": toks, "model": resp.get("model"),
                "answered": "topic" in (resp.get("answers") or {})}
    except urllib.error.HTTPError as e:
        rec |= {"status": e.code, "error": e.read().decode(errors="replace").replace(KEY, "[REDACTED]")[:300]}
    except (urllib.error.URLError, TimeoutError) as e:
        rec |= {"status": None, "error": str(e).replace(KEY, "[REDACTED]")[:300]}
    rec["latency_s"] = round(time.perf_counter() - t0, 3)
    if DISTINCT[0]:
        rec |= {"seed": seed, "char_scale": SCALE[0]}
    with (OUT / LOG[0]).open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, names, T, S, P, test = bb.split()
    texts = [T[i]["text"] for i in P]
    if DISTINCT[0]:
        sizes = {int(k): v for k, v in json.loads((OUT / "plan.json").read_text())["utts"].items()}
        return measure(sizes, texts)
    cal = call("calibrate", 1000, 60, texts)                        # tokens per utterance from one call
    assert cal.get("status") == 200, cal
    per_utt = cal["input_tokens"] / 60; base = 0                    # question overhead is folded into the estimate
    small = call("calibrate2", 4000, 240, texts); per_utt = (small["input_tokens"] - cal["input_tokens"]) / 180
    base = cal["input_tokens"] - 60 * per_utt

    def n_for(target):
        return max(1, round((target - base) / per_utt))
    sizes = {t: n_for(t) for t in SIZES + OVER}
    big = call("calibrate3", 31500, sizes[31500], texts)           # refine at the top size (it must stay under 32k)
    if big.get("status") == 200:
        per_utt = (big["input_tokens"] - base) / sizes[31500]
        sizes = {t: n_for(t) for t in SIZES + OVER}
    (OUT / "plan.json").write_text(json.dumps({"base_tokens": base, "tokens_per_utt": per_utt, "utts": sizes,
                                               "network": "client on a GCP VM in asia-east1 (Taiwan), public internet"}))
    measure(sizes, texts)


def measure(sizes, texts):
    for t in (SIZES[-1:] if TOP_ONLY[0] else SIZES):                # sequential, one at a time
        for _ in range(REPS):
            call("seq", t, sizes[t], texts)
    for c in CONC:                                                  # concurrent at the top size
        with ThreadPoolExecutor(c) as ex:
            list(ex.map(lambda _: call(f"conc{c}", 31500, sizes[31500], texts), range(c)))
        time.sleep(5)
    for t in ([] if TOP_ONLY[0] else OVER):
        call("over", t, sizes[t], texts)
    print(json.dumps({"done": True, "spent_usd_private": round(spent[0], 4)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--distinct", action="store_true")
    ap.add_argument("--top-only", action="store_true"); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()
    if a.distinct:
        DISTINCT[0] = True; LOG[0] = "calls_distinct.jsonl"; SCALE[0] = a.scale; TOP_ONLY[0] = a.top_only
        counter[0] = a.seed_offset
    main()
