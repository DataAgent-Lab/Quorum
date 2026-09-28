#!/usr/bin/env python3
"""Reproduce the zero-shot result: the cheap open ensemble vs its best single member, on a public intent
benchmark, full official test set, with a paired McNemar test.

    python scripts/eval_zeroshot.py --dataset banking77 --device cpu
    python scripts/eval_zeroshot.py --dataset massive  --device cuda --out results/massive.json

Datasets: banking77, clinc150, massive, mtop, hwu64, snips, bitext.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import ZeroShotEnsemble, data, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="cap test items (debug only — never for a headline)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = data.load(args.dataset)
    labels, label_texts, test = list(range(len(d["labels"]))), d["labels"], d["test"]
    if args.limit:
        test = test[:args.limit]
    y = np.array([r["label"] for r in test])
    print(f"[{args.dataset}] labels={len(labels)} test={len(test)} device={args.device}", flush=True)

    clf = ZeroShotEnsemble(device=args.device)
    print(f"  ensemble = {clf.model_names}", flush=True)
    n_members = len(clf.members)
    ens_pred = np.zeros(len(test), int)
    mem_pred = np.zeros((n_members, len(test)), int)
    t0 = time.time()
    for i, r in enumerate(test):
        mprobs, ep = clf.member_and_ensemble_proba(r["text"], labels, label_texts)
        ens_pred[i] = int(np.argmax(ep))
        for m in range(n_members):
            mem_pred[m, i] = int(np.argmax(mprobs[m]))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(test)}", flush=True)
    sec = time.time() - t0

    ens_acc = metrics.accuracy(ens_pred, y)
    singles = {clf.model_names[m]: metrics.accuracy(mem_pred[m], y) for m in range(n_members)}
    best_m = max(range(n_members), key=lambda m: metrics.accuracy(mem_pred[m], y))
    mc = metrics.mcnemar(ens_pred, mem_pred[best_m], y)
    rep = {"dataset": args.dataset, "n_labels": len(labels), "n_test": len(test),
           "ensemble_accuracy": round(ens_acc, 4), "single_member_accuracy": {k: round(v, 4) for k, v in singles.items()},
           "best_single_member": clf.model_names[best_m],
           "mcnemar_ensemble_vs_best_single": mc, "ms_per_item": round(1000 * sec / len(test), 1),
           "full_test": bool(args.limit == 0)}
    print(f"[{args.dataset}] ensemble={ens_acc:.4f}  best_single={singles[clf.model_names[best_m]]:.4f}  "
          f"McNemar {mc['verdict']} (p={mc['p_two_sided']}, b={mc['b_a_wins']}/c={mc['c_b_wins']})", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
