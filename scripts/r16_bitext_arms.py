#!/usr/bin/env python3
"""R16 Bitext 24-shot, decision-model arms (`provenance/bitext_24shot_protocol.md` §8 + A1): Clef, Clef-Flash
(release code) and pplx-decider v1.1 (the A7 parity-gated serving engine), each on the byte-identical, sha-checked
bodies of the R16 manifest. One scored run per model.

    python scripts/r16_bitext_arms.py run --model clef_flash --bs 8
    python scripts/r16_bitext_arms.py run --model clef --bs 4
    python scripts/r16_bitext_arms.py run --model pplx --engine-url http://HOST:PORT --engine-meta engine.json
    python scripts/r16_bitext_arms.py smoke --model ...      # 8 pool rows not in S (§10); output discarded

Probabilities are stored over the 27 raw intent names in sorted order (= the dataset's label ids).
pplx: a question over 8,192 prompt tokens is unanswered (= wrong), because the release code refuses it (§8). The
engine reports `usage.input_tokens`, and its prompt equals the release prompt (A7 parity gate).
"""
import argparse, gzip, hashlib, importlib.util, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PPLX_LIMIT = 8192


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


B = _load("bitext24_bodies")
OUTP, OUT = ROOT / "results" / "predictions", ROOT / "results" / "r16"


def engine_probs(url, bodies, conc=2):
    import urllib.request

    def one(b):
        req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:            # any error aborts the run (as A7)
            out = json.loads(r.read())
        p = out["answers"]["intent"]["probabilities"]; keys = list(b["questions"]["intent"]["criteria"])
        if set(p) != set(keys):
            raise SystemExit("engine returned option ids that differ from the request's criteria")
        return np.array([float(p[k]) for k in keys]), int(out.get("usage", {}).get("input_tokens", 0))
    with ThreadPoolExecutor(conc) as ex:
        return list(ex.map(one, bodies))


def release_probs(model_tag, bodies, bs):
    R14 = _load("r14_clef")
    J, model, proc, code_sha = R14.load_model(model_tag)
    res = R14.score(J, model, proc, bodies, bs)                       # (option_ids, probs, prompt tokens) per body
    out = []
    for b, (opt_ids, p, ntok) in zip(bodies, res):
        pos = {o: j for j, o in enumerate(opt_ids)}
        out.append((np.array([p[pos[k]] for k in b["questions"]["intent"]["criteria"]]), int(ntok)))
    return out, code_sha, R14.MODELS[model_tag]


def run(args, smoke=False):
    R14 = _load("r14_clef")
    if smoke:
        raw, names, T, S, P, test = B.split(); desc = B.descriptions(names)
        bm = B.StableBM25([T[i]["text"] for i in P]); rows = [T[i] for i in P[:8]]
        bodies = [B.body(r["text"], [{"message": T[j]["text"], "intent": raw[T[j]["label"]]}
                                     for j in [P[x] for x in bm.top(r["text"], B.SHOTS)]], raw, desc)[0] for r in rows]
    else:
        rows, raw, bodies, man = B.verified_bodies()                  # sha-checked against the committed manifest
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    assert order == list(range(len(raw))), "raw labels are expected in sorted order"
    t0 = time.time()
    if args.model == "pplx":
        res = engine_probs(args.engine_url, bodies); engine = json.loads(Path(args.engine_meta).read_text())
        model_id = ("perplexity-ai/pplx-decider-v1.1-27b", "3b45dead91dfa6d95aad6b95764a606fab2bf7a6"); code_sha = None
    else:
        res, code_sha, model_id = release_probs(args.model, bodies, args.bs); engine = {"name": "release code", "batch": args.bs}
    if smoke:
        assert len(res) == 8 and all(p.shape == (len(raw),) for p, _ in res)
        print("SMOKE_OK (outputs discarded)", flush=True); return
    gold = np.array([r["label"] for r in rows]); P_ = np.zeros((len(rows), len(raw))); unans = np.zeros(len(rows), bool)
    ntok = np.array([n for _, n in res])
    for i, (p, n) in enumerate(res):
        P_[i] = p
        if args.model == "pplx" and n > PPLX_LIMIT:
            unans[i] = True                                            # release refusal (§8)
    pred = P_.argmax(1); hit = (pred == gold) & ~unans
    rec = R14.metrics(np.where(unans[:, None], 0, P_) + np.where(unans[:, None], 1 / len(raw), 0), gold)
    rec.update({"accuracy": round(float(hit.mean()), 4), "correct": int(hit.sum()), "unanswered": int(unans.sum()),
                "prompt_tokens": {"median": int(np.median(ntok)), "max": int(ntok.max())},
                "seconds": round(time.time() - t0, 1), "ms_per_item": round(1000 * (time.time() - t0) / len(rows), 1)})
    OUTP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTP / f"bitext_r16_{args.model}.jsonl.gz", "wt") as f:
        for i in range(len(rows)):
            f.write(json.dumps({"idx": i, "gold": int(gold[i]), "pred": None if unans[i] else int(pred[i]),
                                "probs": R14.g7(P_[i]), "input_tokens": int(ntok[i]), "unanswered": bool(unans[i])}) + "\n")
    meta = {"protocol_commit": B.PROTOCOL_COMMIT, "amendment_A1": "3d84cc3a4f13ce52355d46ae3b76134a0edf8ada",
            "manifest_sha256": json.loads((B.OUT / "manifest_meta.json").read_text())["manifest_sha256"],
            "model": model_id[0], "revision": model_id[1], "engine": engine, "custom_code_sha256": code_sha,
            "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                            for f in ("scripts/r16_bitext_arms.py", "scripts/bitext24_bodies.py", "scripts/r14_clef.py")},
            "label_order": "raw intent names, sorted (= dataset label ids)", "results": rec,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / f"summary_bitext_r16_{args.model}.json").write_text(json.dumps(meta, indent=1))
    print("R16_DONE", args.model, json.dumps(rec), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["run", "smoke"])
    ap.add_argument("--model", choices=["clef", "clef_flash", "pplx"], required=True)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--engine-url", default=""); ap.add_argument("--engine-meta", default="")
    a = ap.parse_args(); run(a, smoke=a.phase == "smoke")
