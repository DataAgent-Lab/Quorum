#!/usr/bin/env python3
"""Combiner ablation + member calibration for the zero-shot ensemble (T2.12, claims Z6), from per-member dumps.

Combiners (all training-free, computed from the same three member distributions):
  geo   — geometric mean of member probabilities, renormalised (the method; = log-linear pool, weights 1/3)
  poe   — product of experts, renormalised (= softmax of the summed log-probs; same argmax as geo, sharper)
  arith — arithmetic mean of member probabilities
  vote  — majority vote over member argmaxes; when all three disagree, the pick with the highest arithmetic-mean
          probability among the three picks (declared tie rule)
  maxconf — the single most confident member decides (per item)
Family F3 (Holm): geo vs arith and geo vs vote, 7 datasets x 2 label texts = 28 exact McNemar tests.

    python paper/scripts/combiner_ablation.py --dumps <dir with *_{names,descriptions}.jsonl.gz incl. member_probs>
"""
from __future__ import annotations
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quorum.metrics import calibration, mcnemar_exact_p  # noqa: E402
from zero_shot_tables import DATASETS, CONDS, holm, fmt_p, pct  # noqa: E402

MEMBERS = ["PrismNLI-0.4B", "bge-large", "bge-base"]


def combine(M):
    """M: (n, 3, K) member probabilities -> dict of (n, K) score matrices (probability matrices except vote)."""
    L = np.log(np.clip(M, 1e-12, 1.0))
    geo = np.exp(L.mean(1)); geo /= geo.sum(1, keepdims=True)
    s = L.sum(1); s -= s.max(1, keepdims=True); poe = np.exp(s); poe /= poe.sum(1, keepdims=True)
    arith = M.mean(1)
    picks = M.argmax(2)                                                     # (n, 3)
    n, K = arith.shape; vote = np.zeros((n, K))
    for i in range(n):
        vals, cnt = np.unique(picks[i], return_counts=True)
        if cnt.max() >= 2:
            vote[i, vals[cnt.argmax()]] = 1.0
        else:
            vote[i, picks[i][np.argmax(arith[i, picks[i]])]] = 1.0
    conf = M.max(2); best_m = conf.argmax(1)
    maxconf = M[np.arange(n), best_m]
    return {"geo": geo, "poe": poe, "arith": arith, "vote": vote, "maxconf": maxconf}


def entropy_bits(P):
    return float(np.mean(-(P * np.log2(np.clip(P, 1e-12, 1))).sum(1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dumps", required=True); a = ap.parse_args()
    out, fam = {}, []
    for ds, _ in DATASETS:
        out[ds] = {}
        for cond, _ in CONDS:
            rows = [json.loads(l) for l in gzip.open(Path(a.dumps) / f"{ds}_{cond}.jsonl.gz", "rt")]
            M = np.array([r["member_probs"] for r in rows]); y = np.array([r["gold"] for r in rows])
            C = combine(M); pred = {k: v.argmax(1) for k, v in C.items()}
            assert (pred["geo"] == np.array([r["pred"] for r in rows])).all(), "geo must equal the dumped prediction"
            assert (pred["geo"] == pred["poe"]).all(), "poe and geo must share the argmax"
            res = {"n": len(y), "acc": {k: float((p == y).mean()) for k, p in pred.items()},
                   "member_acc": {m: float((M[:, j].argmax(1) == y).mean()) for j, m in enumerate(MEMBERS)},
                   "calibration": {k: calibration(C[k], y) for k in ("geo", "poe", "arith")},
                   "member_calibration": {m: calibration(M[:, j], y) for j, m in enumerate(MEMBERS)},
                   "member_entropy_bits": {m: entropy_bits(M[:, j]) for j, m in enumerate(MEMBERS)},
                   "uniform_entropy_bits": float(np.log2(M.shape[2])), "tests": {}}
            for other in ("arith", "vote"):
                ga, ob = pred["geo"] == y, pred[other] == y
                b, c = int((ga & ~ob).sum()), int((~ga & ob).sum())
                res["tests"][f"geo_vs_{other}"] = {"b": b, "c": c, "p": mcnemar_exact_p(b, c)}
                fam.append((ds, cond, f"geo_vs_{other}"))
            out[ds][cond] = res
    for (ds, cond, t), adj in zip(fam, holm([out[d][c]["tests"][t]["p"] for d, c, t in fam])):
        out[ds][cond]["tests"][t]["p_holm_F3"] = adj
    out["_declared"] = {"family_F3": "geo vs arith, geo vs vote; 7 datasets x 2 label texts (28), Holm",
                        "vote_tie_rule": "highest arithmetic-mean probability among the three picks",
                        "source": str(a.dumps)}
    gen = ROOT / "paper" / "generated"; gen.mkdir(parents=True, exist_ok=True)
    (gen / "combiners.json").write_text(json.dumps(out, indent=1))

    print(f"{'dataset':10s} {'cond':5s} | {'geo':>6s} {'arith':>6s} {'vote':>6s} {'maxcf':>6s} best1 | "
          f"geo-arith p_H  geo-vote p_H | ECE geo  poe  arith | member ECE (NLI, bgeL, bgeB) | member H/Hmax")
    for ds, _ in DATASETS:
        for cond, _ in CONDS:
            r = out[ds][cond]; A = r["acc"]; t = r["tests"]
            ga, gv = t["geo_vs_arith"], t["geo_vs_vote"]
            mece = " ".join(f"{r['member_calibration'][m]['ece_10_bins']:.2f}" for m in MEMBERS)
            ment = " ".join(f"{r['member_entropy_bits'][m] / r['uniform_entropy_bits']:.2f}" for m in MEMBERS)
            print(f"{ds:10s} {cond[:5]:5s} | {A['geo']:.4f} {A['arith']:.4f} {A['vote']:.4f} {A['maxconf']:.4f} "
                  f"{max(r['member_acc'].values()):.4f} | {100*(A['geo']-A['arith']):+5.1f} {ga['p_holm_F3']:.1e} "
                  f"{100*(A['geo']-A['vote']):+5.1f} {gv['p_holm_F3']:.1e} | "
                  f"{r['calibration']['geo']['ece_10_bins']:.3f} {r['calibration']['poe']['ece_10_bins']:.3f} "
                  f"{r['calibration']['arith']['ece_10_bins']:.3f} | {mece} | {ment}")


if __name__ == "__main__":
    main()
