"""Spot-check dumped benchmark predictions against the served code path, live on this machine.

    python -m serve.spot_check --predictions results/predictions --dataset banking77 --variant names --n 30

Samples N items from {dataset}_{variant}.jsonl.gz, recomputes each with the exact code the API runs on a cache
miss (serve.worker.init_model / predict), and reports:
  - agreement with the dump on the predicted label, and the largest per-label probability difference
    (GPU-vs-CPU float noise is ~1e-5; a model/config mismatch shows up far larger)
  - accuracy on the N items with a 95% Wilson interval (N=30 is only ±~0.15 — a sanity check, not a measurement)
  - the exact full-set accuracy from the dump vs the published results/*.json number
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from pathlib import Path

from serve.seed_cache import REPO, published_accuracy

DEFAULT_DATASETS = ("banking77", "clinc150", "hwu64", "massive", "mtop", "snips", "bitext")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_dump(pred_dir: Path, dataset: str, variant: str):
    meta = json.loads((pred_dir / f"{dataset}_meta.json").read_text())
    labels = meta[f"labels_{variant}"]
    with gzip.open(pred_dir / f"{dataset}_{variant}.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return labels, rows, meta


def check(pred_dir: Path, dataset: str, variant: str, n: int, seed: int, results_dir: Path) -> dict:
    from serve import worker as w
    labels, rows, _ = load_dump(pred_dir, dataset, variant)
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    agree = live_correct = 0
    max_dp = 0.0
    secs = []
    for row in sample:
        t = time.perf_counter()
        probs, _ = w.predict(row["text"].strip(), labels)
        secs.append(time.perf_counter() - t)
        live_pred = max(range(len(probs)), key=lambda i: probs[i])
        agree += int(live_pred == row["pred"])
        live_correct += int(live_pred == row["gold"])
        max_dp = max(max_dp, max(abs(a - b) for a, b in zip(probs, row["probs"])))
    lo, hi = wilson(live_correct, len(sample))
    full = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
    return {"dataset": dataset, "variant": variant, "n": len(sample), "argmax_agree": agree,
            "max_abs_dp": max_dp, "live_acc": live_correct / len(sample), "live_acc_ci95": (lo, hi),
            "full_set_acc_from_dump": full, "full_set_n": len(rows),
            "published": published_accuracy(dataset, variant, results_dir),
            "sec_per_item": sum(secs) / len(secs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=REPO / "results" / "predictions")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--dataset", action="append", help="repeatable; default = every dataset with a dump")
    ap.add_argument("--variant", default="names", choices=("names", "descriptions"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    datasets = args.dataset or [d for d in DEFAULT_DATASETS
                                if (args.predictions / f"{d}_{args.variant}.jsonl.gz").exists()]
    from serve import worker as w
    w.init_model(args.device)
    ok = True
    for d in datasets:
        r = check(args.predictions, d, args.variant, args.n, args.seed, args.results)
        pub = "n/a" if r["published"] is None else f"{r['published']:.4f}"
        print(f"{d:10s} {args.variant:12s} n={r['n']:3d} argmax_agree={r['argmax_agree']}/{r['n']} "
              f"max|Δp|={r['max_abs_dp']:.2e} live_acc={r['live_acc']:.3f} "
              f"(95% CI {r['live_acc_ci95'][0]:.3f}–{r['live_acc_ci95'][1]:.3f}) "
              f"full_set={r['full_set_acc_from_dump']:.4f} on {r['full_set_n']} published={pub} "
              f"{r['sec_per_item']:.1f}s/item", flush=True)
        ok &= r["argmax_agree"] >= r["n"] - 1 and r["max_abs_dp"] < 1e-3
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
