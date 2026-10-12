#!/usr/bin/env python3
"""Decision-model study, block E runner (`provenance/decision13_protocol.md` §3–§7): send frozen item bodies to a
`/v1/systemone` engine and score them. Stdlib only.

    python3 scripts/decision13/run_block_e.py --items <set>.jsonl.gz --manifest results/decision13/manifests/<set>_test_manifest.json \
        --set patchnet --model pplx_v11 --engine-url http://HOST:PORT --engine-meta engine.json --out-dir out/ \
        --expect-model pplx-decider-v1.1-27b [--concurrency 16] [--breakdown category] [--limit N]

The items file is what the set's builder (scripts/decision13/builders/) writes; its per-body sha256 and the aggregate
`items_sha256` must equal the committed manifest.

Rules implemented: every body's sha256 must equal the item's `body_sha256` and the set's `items_sha256` must equal
the manifest (else abort before any call); unanswered (rejected / failed request) = wrong (§5); 25 consecutive
server-side failures (connection, timeout, 5xx — not 4xx refusals) abort the run (a dead server, not a result); noul → P(true) from `answers[q].noul`; choice → argmax of
`answers[q].probabilities`, correct if in `gold_accept` when present; Brier summed over options, ECE-10 on the
predicted option's probability (as R14). --limit writes `_limitN` files (smoke, never a result).
"""
import argparse, gzip, hashlib, json, math, os, time, urllib.error, urllib.request
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--set", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--engine-url", required=True); ap.add_argument("--engine-meta", required=True)
ap.add_argument("--out-dir", required=True); ap.add_argument("--breakdown", default="")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--timeout", type=float, default=900)
ap.add_argument("--concurrency", type=int, default=1, help="in-flight requests (amendment 2 decides 1 or 16)")
ap.add_argument("--rescore", default="", help="amendment 4: re-score an existing per-item file, no calls")
ap.add_argument("--unanswer-idx", default="", help="comma-separated item idx that the release code would refuse (over its limit)")
ap.add_argument("--resend-idx", default="", help="amendment 6: with --rescore, re-send only these idx and replace their records")
ap.add_argument("--expect-model", required=True, help="substring the server's model path must contain (e.g. pplx-decider-v1.1-27b)")
a = ap.parse_args()

items = [json.loads(l) for l in gzip.open(a.items, "rt")]
man = json.load(open(a.manifest))
hashes = [hashlib.sha256(json.dumps(it["body"]).encode()).hexdigest() for it in items]
bad = [it["idx"] for it, h in zip(items, hashes) if h != it["body_sha256"]]
agg = hashlib.sha256("".join(hashes).encode()).hexdigest()
if bad or agg != man["items_sha256"] or len(items) != man["items"]:
    raise SystemExit(f"ABORT: frozen items do not match the manifest (bad bodies {len(bad)}, items_sha256 ok {agg == man['items_sha256']})")
if a.limit:
    items = items[: a.limit]
url = a.engine_url.rstrip("/") + "/v1/systemone"


def served_model():
    """The model path the SGLang server reports (/get_model_info); abort if it is not the expected checkpoint."""
    info = {}
    for ep in ("/model_info", "/get_model_info"):                  # newer name first; the old one is deprecated
        try:
            with urllib.request.urlopen(a.engine_url.rstrip("/") + ep, timeout=30) as r:
                info = json.loads(r.read()); break
        except urllib.error.HTTPError:
            continue
    path = str(info.get("model_path") or info.get("model") or "")
    if a.expect_model not in path:
        raise SystemExit(f"ABORT: server serves {path!r}, expected a path containing {a.expect_model!r}")
    return path


served = served_model() if (not a.rescore or a.resend_idx) else "(rescore: no calls)"
tag = f"{a.set}_{a.model}" + (f"_limit{a.limit}" if a.limit else "")
os.makedirs(a.out_dir, exist_ok=True)
part = os.path.join(a.out_dir, f"{tag}.jsonl.gz.partial")


def call(body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        return json.loads(r.read())


import threading
from concurrent.futures import ThreadPoolExecutor
fail_lock, fails = threading.Lock(), [0]


def one(it):
    body = it["body"]; (qid, q), = body["questions"].items(); keys = list(q.get("criteria") or {})
    t0 = time.perf_counter(); rec = {"idx": it["idx"], "gold": it["gold"]}
    if a.breakdown:
        rec["group"] = it.get(a.breakdown)
    try:
        resp = call(body); ans = resp["answers"][qid]
        rec["input_tokens"] = (resp.get("usage") or {}).get("input_tokens")   # flags Clef's release-style truncation
        with fail_lock:
            fails[0] = 0
        if q["type"] == "noul":
            pt = float(ans["noul"]); probs = {"true": pt, "false": 1.0 - pt}; pred = pt >= 0.5
        else:
            probs = {k: float(ans["probabilities"][k]) for k in keys}; pred = max(keys, key=probs.__getitem__)
        accept = it.get("gold_accept") or [it["gold"]]
        rec.update({"pred": pred, "probs": probs, "unanswered": False, "correct": pred in accept})
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        refused = isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500   # e.g. over the release limit
        with fail_lock:
            if refused:
                fails[0] = 0                                        # a refusal proves the server is alive
            else:
                fails[0] += 1
            n_fail = fails[0]
        rec.update({"pred": None, "probs": None, "unanswered": True, "correct": False,
                    "status": getattr(e, "code", None), "error": repr(e)[:200]})
        if n_fail >= 25:
            raise SystemExit(f"ABORT: {n_fail} consecutive failed requests (last: {rec['error']}) — server down?")
    rec["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    return rec


recs, t_start = [], time.time()
if a.rescore:                                                         # amendment 4 length rule, applied post hoc
    drop = {int(x) for x in a.unanswer_idx.split(",") if x}
    for r in map(json.loads, gzip.open(a.rescore, "rt")):
        if r["idx"] in drop:
            r.update({"pred": None, "probs": None, "unanswered": True, "correct": False, "status": "over_release_limit"})
        recs.append(r)
    resend = {int(x) for x in a.resend_idx.split(",") if x}
    if resend:                                                        # amendment 6: engine refused below the release limit
        by_idx = {it["idx"]: it for it in items}
        recs = [one(by_idx[r["idx"]]) if r["idx"] in resend else r for r in recs]
        tag += "_amend6"
        with gzip.open(os.path.join(a.out_dir, f"{tag}.jsonl.gz"), "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    else:
        tag += "_lengthrule"
else:
    with gzip.open(part, "wt") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for rec in ex.map(one, items):                                    # results come back in item order
            recs.append(rec); f.write(json.dumps(rec) + "\n")
            if len(recs) % 1000 == 0:
                print(f"{tag}: {len(recs)}/{len(items)} done, unanswered {sum(r['unanswered'] for r in recs)}", flush=True)
    os.replace(part, part[: -len(".partial")])


def scores(rs):
    n = len(rs); acc = sum(r["correct"] for r in rs) / n if n else None
    ans = [r for r in rs if not r["unanswered"]]
    labels = sorted({json.dumps(r["gold"]) for r in rs} | {json.dumps(r["pred"]) for r in ans}, key=str)
    f1s = []
    for lab in labels:
        tp = sum(json.dumps(r["pred"]) == lab and json.dumps(r["gold"]) == lab for r in ans)
        fp = sum(json.dumps(r["pred"]) == lab and json.dumps(r["gold"]) != lab for r in ans)
        fn = sum(json.dumps(r["gold"]) == lab and (r["unanswered"] or json.dumps(r["pred"]) != lab) for r in rs)
        if tp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    out = {"n": n, "answered": len(ans), "unanswered": n - len(ans), "accuracy": round(acc, 6) if acc is not None else None,
           "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else None}
    if rs and isinstance(rs[0]["gold"], bool):
        tp = sum(r["pred"] is True and r["gold"] is True for r in ans); fp = sum(r["pred"] is True and r["gold"] is False for r in ans)
        fn = sum(r["gold"] is True and r["pred"] is not True for r in rs)
        out["f1_positive"] = round(2 * tp / (2 * tp + fp + fn), 6) if tp + fp + fn else None
    if ans:
        brier, bins = 0.0, [[] for _ in range(10)]
        for r in ans:
            gold = (r["gold"] if not isinstance(r["gold"], bool) else ("true" if r["gold"] else "false"))
            if r.get("correct") and not isinstance(r["gold"], bool):
                gold = r["pred"]                                   # multi-accept golds: the accepted letter chosen
            brier += sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in r["probs"].items())
            conf = max(r["probs"].values()); bins[min(9, int(conf * 10))].append((conf, r["correct"]))
        out["brier"] = round(brier / len(ans), 6)
        out["ece10"] = round(sum(len(b) / len(ans) * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b))
                                 for b in bins if b), 6)
    return out


summary = {"set": a.set, "model": a.model, "limit": a.limit, "items_sha256": agg, "manifest_items_sha256": man["items_sha256"],
           "engine": json.load(open(a.engine_meta)), "served_model_path": served, "concurrency": a.concurrency, "wall_seconds": round(time.time() - t_start, 1),
           "ms_per_item": round(sum(r["ms"] for r in recs) / len(recs), 1), "overall": scores(recs),
           "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
if a.breakdown:
    g = defaultdict(list)
    for r in recs:
        g[str(r.get("group"))].append(r)
    summary["by_" + a.breakdown] = {k: scores(v) for k, v in sorted(g.items())}
json.dump(summary, open(os.path.join(a.out_dir, f"summary_{tag}.json"), "w"), indent=1)
print("BLOCK_E_DONE", tag, json.dumps(summary["overall"]), flush=True)
