#!/usr/bin/env python3
"""Reproduce the 24-shot result that beats the closed API (Jev) on Banking77.

Retrieve 24 examples per query with BM25 from the training set, run the retrieval ensemble (in-context reader +
bge-large kNN over the same 24, geometric mean), no weight update. Reports full-test accuracy; Jev's reference
on this protocol is 0.924.

    python scripts/eval_24shot.py --dataset banking77 --adapter <hf-id-or-path> --device cuda

The `--adapter` is the gated 24-shot LoRA (Hugging Face id, e.g. DataAgent-Lab/quorum-24shot-adapter, or a
local path). Requires access to the gated model.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import data, metrics
from quorum.fewshot import RetrievalEnsemble

TASK_DESC = {"banking77": "Classify the customer's banking message into the intent it expresses."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77")
    ap.add_argument("--adapter", default="DataAgent/Quorum-Reader-Qwen3-4B-Adapter", help="gated 24-shot LoRA: HF id or local path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="cap test items (debug only)")
    ap.add_argument("--jev-ref", type=float, default=0.924, help="Jev's reference accuracy on this protocol")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = data.load(args.dataset)
    labels, label_texts = list(range(len(d["labels"]))), d["labels"]
    pool, test = d["train"], d["test"]
    if args.limit:
        test = test[:args.limit]
    y = np.array([r["label"] for r in test])
    task_desc = TASK_DESC.get(args.dataset, "Classify the message into the intent it expresses.")
    print(f"[{args.dataset}] labels={len(labels)} pool={len(pool)} test={len(test)} perms={args.perms}", flush=True)

    eng = RetrievalEnsemble(label_texts, task_desc, adapter=args.adapter, perms=args.perms, device=args.device)
    eng.index([{"text": r["text"], "label": r["label"]} for r in pool])
    t0 = time.time()
    P = eng.predict_proba(test)
    pred = P.argmax(1)
    sec = time.time() - t0
    acc = metrics.accuracy(pred, y)
    rep = {"dataset": args.dataset, "n_labels": len(labels), "n_test": len(test), "perms": args.perms,
           "ensemble_accuracy": round(acc, 4), "jev_reference": args.jev_ref,
           "delta_vs_jev": round(acc - args.jev_ref, 4), "adapter": args.adapter,
           "ms_per_item": round(1000 * sec / len(test), 1), "full_test": bool(args.limit == 0)}
    print(f"[{args.dataset}] 24-shot ensemble = {acc:.4f}  (Jev ref {args.jev_ref}, Δ {acc-args.jev_ref:+.4f})",
          flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
