#!/usr/bin/env python3
"""Pool size (Banking77), decision-model arms (`provenance/banking77_pool_size_protocol.md` §2, §4, §6): pplx-decider
v1.1 (R16 A3 serving engine) and Clef-Flash (release code) on the byte-identical, sha-checked bodies of every pool.
One scored run per model x pool; a pool whose dump exists is skipped, never re-run.

    python scripts/poolsize_arms.py run --model clef_flash --bs 8                       # all 11 pools, run order
    python scripts/poolsize_arms.py run --model pplx --engine-url http://HOST:PORT --engine-meta engine.json
    python scripts/poolsize_arms.py smoke --model ...      # 8 selection-split rows retrieving from pc8_d1; discarded

Probabilities are stored over the 77 labels in dataset label-id order.
pplx: a question over 8,192 prompt tokens (server `usage.input_tokens`) is unanswered (= wrong), the release rule.
"""
import argparse, gzip, hashlib, importlib.util, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PPLX_LIMIT = 8192
PPLX = ("perplexity-ai/pplx-decider-v1.1-27b", "3b45dead91dfa6d95aad6b95764a606fab2bf7a6")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


PB = _load("poolsize_bodies")
OUTP = PB.OUT / "predictions"


def engine_probs(url, bodies, raw, conc=2):
    import urllib.request

    def one(b):
        req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:             # any error aborts the run
            out = json.loads(r.read())
        p = out["answers"]["intent"]["probabilities"]
        if set(p) != set(b["questions"]["intent"]["criteria"]):
            raise SystemExit("engine returned option ids that differ from the request's criteria")
        return np.array([float(p[k]) for k in raw]), int(out.get("usage", {}).get("input_tokens", 0))
    with ThreadPoolExecutor(conc) as ex:
        return list(ex.map(one, bodies))


def release_probs(R14, loaded, bodies, raw, bs):
    J, model, proc = loaded
    res = R14.score(J, model, proc, bodies, bs)                         # (option_ids, probs, prompt tokens) per body
    out = []
    for opt_ids, p, ntok in res:
        pos = {o: j for j, o in enumerate(opt_ids)}
        out.append((np.array([p[pos[k]] for k in raw]), int(ntok)))
    return out


def run(args, smoke=False):
    R14 = _load("r14_clef")
    train, raw = PB.train_rows()
    loaded, code_sha = None, None
    if args.model == "clef_flash":
        J, model, proc, code_sha = R14.load_model("clef_flash"); loaded = (J, model, proc)
    score = (lambda bodies: engine_probs(args.engine_url, bodies, raw)) if args.model == "pplx" else \
            (lambda bodies: release_probs(R14, loaded, bodies, raw, args.bs))
    if smoke:
        S = json.loads(PB.SPLIT.read_text())["selection_idx"][:8]; pool = PB.pools(train)["pc8_d1"]
        assert not set(S) & set(pool)
        top = PB.retriever(train, pool); rq = R14.ReproRequests()
        bodies = [PB.body(rq, train[i]["text"], top(train[i]["text"]), train)[0] for i in S]
        res = score(bodies)
        assert len(res) == 8 and all(p.shape == (PB.K,) and abs(p.sum() - 1) < 1e-3 for p, _ in res)
        print("SMOKE_OK (outputs discarded)", args.model, "max tokens", max(n for _, n in res), flush=True); return
    test = PB.test_rows(); gold = np.array([r["label"] for r in test])
    engine = json.loads(Path(args.engine_meta).read_text()) if args.model == "pplx" else \
        {"name": "release code", "batch": args.bs}
    model_id = PPLX if args.model == "pplx" else R14.MODELS["clef_flash"]
    OUTP.mkdir(parents=True, exist_ok=True)
    for name in PB.pools(train):
        dump = OUTP / f"{args.model}_{name}.jsonl.gz"; summ = PB.OUT / f"summary_{args.model}_{name}.json"
        if dump.exists() or summ.exists():
            print(f"skip {name}: output exists (one scored run per cell)", flush=True); continue
        bodies, man = PB.verified_bodies(name)
        t0 = time.time(); res = score(bodies)
        P_ = np.stack([p for p, _ in res]); ntok = np.array([n for _, n in res])
        unans = (ntok > PPLX_LIMIT) if args.model == "pplx" else np.zeros(len(test), bool)
        pred = P_.argmax(1); hit = (pred == gold) & ~unans
        Pm = np.where(unans[:, None], 1 / PB.K, P_)                     # unanswered: uniform for calibration
        rec = R14.metrics(Pm, gold)
        rec.update({"accuracy": round(float(hit.mean()), 4), "correct": int(hit.sum()), "unanswered": int(unans.sum()),
                    "truncated_at_16384": int((ntok >= 16384).sum()) if args.model == "clef_flash" else None,
                    "prompt_tokens": {"median": int(np.median(ntok)), "max": int(ntok.max())},
                    "seconds": round(time.time() - t0, 1)})
        tmp = dump.with_suffix(".partial")
        with gzip.open(tmp, "wt") as f:
            for i in range(len(test)):
                f.write(json.dumps({"idx": i, "gold": int(gold[i]), "pred": None if unans[i] else int(pred[i]),
                                    "probs": R14.g7(P_[i]), "unanswered": bool(unans[i]),
                                    "input_tokens": int(ntok[i])}) + "\n")
        tmp.rename(dump)
        meta = {"protocol": PB.PROTOCOL, "pool": name, "pool_rows": len(json.loads((PB.OUT / "pools.json").read_text())["pools"][name]),
                "manifest_sha256": json.loads((PB.OUT / "manifest_meta.json").read_text())["pools"][name]["manifest_sha256"],
                "model": model_id[0], "revision": model_id[1], "engine": engine, "custom_code_sha256": code_sha,
                "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                for f in ("scripts/poolsize_arms.py", "scripts/poolsize_bodies.py", "scripts/r14_clef.py")},
                "label_order": "dataset label ids", "results": rec,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        summ.write_text(json.dumps(meta, indent=1))
        print("POOLSIZE_CELL_DONE", args.model, name, json.dumps({k: rec[k] for k in ("unanswered", "seconds")}), flush=True)
    print("POOLSIZE_ARM_DONE", args.model, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["run", "smoke"])
    ap.add_argument("--model", choices=["clef_flash", "pplx"], required=True)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--engine-url", default=""); ap.add_argument("--engine-meta", default="")
    a = ap.parse_args(); run(a, smoke=a.phase == "smoke")
