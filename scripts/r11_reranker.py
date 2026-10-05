#!/usr/bin/env python3
"""R11 — a sub-1B cross-encoder reranker used as a zero-shot classifier, and as an optional 4th ensemble member.

The reranker scores (message, "This message is about {label}.") for every label; softmax at T=1 gives the class
distribution. Run on all 7 datasets x {class names, descriptions}, full official test sets. The 4-member ensemble is
the geometric mean (T=1) of the three existing members (taken from the R3 dumps' `member_probs`, so those models
are not re-run) and the reranker.

    python scripts/r11_reranker.py [--r3-dir PATH]

Writes results/predictions/{dataset}_{variant}_reranker.jsonl.gz and ..._ensemble4.jsonl.gz, plus
results/r11/summary.json (accuracies; exact McNemar of the 4-member vs the 3-member ensemble).
"""
import argparse, gzip, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RERANKER, RERANKER_REV = "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
TEMPLATE = "This message is about {label}."
DATASETS = ["banking77", "clinc150", "hwu64", "massive", "mtop", "snips", "bitext"]
DESC_MODULE = {"clinc150": "clinc"}
OUTP, OUT = ROOT / "results" / "predictions", ROOT / "results" / "r11"
g7 = lambda v: [float(f"{x:.7g}") for x in v]


def load_desc(ds, labels):
    mod = DESC_MODULE.get(ds, ds)
    spec = importlib.util.spec_from_file_location(mod, ROOT / "examples" / f"{mod}_descriptions.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    d = next(v for k, v in vars(m).items() if k.endswith("_DESCRIPTIONS") and isinstance(v, dict))
    norm = {str(k).replace("_", " ").strip().lower(): v for k, v in d.items()}
    return [norm[str(l).replace("_", " ").strip().lower()] for l in labels]


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r3-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--pair-batch", type=int, default=256); a = ap.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from quorum import data
    tok = AutoTokenizer.from_pretrained(RERANKER, revision=RERANKER_REV)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER, revision=RERANKER_REV,
                                                               torch_dtype=torch.float16).to("cuda").eval()
    OUTP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    summary, t_all = {}, time.time()
    for ds in DATASETS:
        d = data.load(ds); labels = list(d["labels"]); test = d["test"]; K = len(labels)
        gold = np.array([r["label"] for r in test])
        for variant, lt in [("names", labels), ("descriptions", load_desc(ds, labels))]:
            hyps = [TEMPLATE.format(label=t) for t in lt]; L = np.zeros((len(test), K)); t0 = time.time()
            per = max(1, a.pair_batch // K)                              # messages per forward batch
            with torch.no_grad():
                for i in range(0, len(test), per):
                    msgs = [r["text"] for r in test[i:i + per]]
                    enc = tok([m for m in msgs for _ in hyps], hyps * len(msgs), padding=True, truncation=True,
                              max_length=512, return_tensors="pt").to("cuda")
                    L[i:i + len(msgs)] = model(**enc).logits[:, 0].float().cpu().numpy().reshape(len(msgs), K)
            P = softmax(L); pred = P.argmax(1); acc = float((pred == gold).mean())
            with gzip.open(OUTP / f"{ds}_{variant}_reranker.jsonl.gz", "wt") as f:
                for i, r in enumerate(test):
                    f.write(json.dumps({"idx": i, "text": r["text"], "gold": int(gold[i]), "pred": int(pred[i]),
                                        "probs": g7(P[i])}) + "\n")
            rec = {"reranker_accuracy": round(acc, 4), "seconds": round(time.time() - t0, 1)}
            r3 = Path(a.r3_dir) / f"{ds}_{variant}.jsonl.gz"
            if r3.exists():
                rows = [json.loads(l) for l in gzip.open(r3, "rt")]
                assert [x["text"] for x in rows] == [r["text"] for r in test], f"{ds}/{variant}: R3 row order differs"
                if "member_probs" in rows[0]:
                    M = np.array([x["member_probs"] for x in rows])                # (N, 3, K)
                    e3 = np.array([x["probs"] for x in rows]); p3 = e3.argmax(1)
                    lg = np.concatenate([np.log(M + 1e-12), np.log(P + 1e-12)[:, None, :]], 1).mean(1)
                    e4 = np.exp(lg - lg.max(1, keepdims=True)); e4 /= e4.sum(1, keepdims=True); p4 = e4.argmax(1)
                    with gzip.open(OUTP / f"{ds}_{variant}_ensemble4.jsonl.gz", "wt") as f:
                        for i, r in enumerate(test):
                            f.write(json.dumps({"idx": i, "text": r["text"], "gold": int(gold[i]), "pred": int(p4[i]),
                                                "probs": g7(e4[i]),
                                                "member_picks": [int(m) for m in M[i].argmax(1)] + [int(pred[i])]}) + "\n")
                    h3, h4 = p3 == gold, p4 == gold; b, c = int((h4 & ~h3).sum()), int((~h4 & h3).sum())
                    best_single = max(float((M[:, j].argmax(1) == gold).mean()) for j in range(3))
                    rec.update({"ensemble3_accuracy": round(float(h3.mean()), 4),
                                "ensemble4_accuracy": round(float(h4.mean()), 4),
                                "best_existing_single_member": round(best_single, 4),
                                "mcnemar_ensemble4_vs_ensemble3": {"b": b, "c": c, "p_two_sided_exact": mcnemar_exact(b, c)}})
                else:
                    rec["ensemble4"] = "skipped: R3 dump has no member_probs"
            else:
                rec["ensemble4"] = f"skipped: {r3} not found"
            summary[f"{ds}/{variant}"] = rec; print(f"[{ds}/{variant}] {json.dumps(rec)}", flush=True)
    import transformers
    meta = {"reranker": RERANKER, "reranker_revision": RERANKER_REV, "template": TEMPLATE, "precision": "fp16",
            "max_length": 512, "softmax_temperature": 1.0, "ensemble4": "geometric mean (T=1) of PrismNLI, bge-large, "
            "bge-base (R3 member_probs) and the reranker", "results": summary,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "device": torch.cuda.get_device_name(0), "wall_seconds": round(time.time() - t_all, 1),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "summary.json").write_text(json.dumps(meta, indent=1)); print("R11_DONE", flush=True)


if __name__ == "__main__":
    main()
