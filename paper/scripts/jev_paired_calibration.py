#!/usr/bin/env python3
"""Paired calibration comparison with the public Jev reproduction — DECLARED BEFORE RUNNING (committed first).

Jev's per-item probabilities are published by the reproduction (github.com/simonmesmith/jev-banking77-experiment
@ 5cac4ff, runs/test-v1/test-NNNNN/prediction.json, field answer.probabilities, 77 labels, 2-decimal rounding).
That repository has no licence: the files are fetched at run time into a cache OUTSIDE this repository and never
committed or redistributed.

Declared procedure:
  0. Instrument control (abort on failure): recompute Jev's ECE-10, Brier and log loss (clip 1e-15) from the
     per-item probabilities AS PUBLISHED with quorum.metrics.calibration; they must equal the reproduction's
     reported results.json values (0.0341 / 0.1193 / 0.784) within 1e-3; Jev's argmax accuracy must equal its
     predictions.csv accuracy (0.9240).
  1. Systems (ours, full Banking77 test, rows joined by official test index, gold labels checked):
       primary  — clean pre-registered run (results/predictions/banking77_24shot_clean)
       secondary — best configuration under the reproduction's retrieval rule (…_r6_best_parityret)
       descriptive — repo default (…_default), kNN alone (…_r5_knn_alone)
  2. Differences ours − Jev for log loss (per-item, clipped 1e-15), Brier (per-item, summed over classes) and
     ECE-10; paired bootstrap, 10,000 row resamples, seed 20261010, 95% percentile CI. "Better calibrated" = the
     CI lies entirely below 0. Reported for every system whatever the outcome.
  3. Sensitivity (declared): Jev's 2-decimal rounding produces exact zeros that dominate its log loss. Repeat step 2
     with Jev's probabilities floored at 0.005 (half the rounding unit) and renormalised — this can only favour Jev.

    python paper/scripts/jev_paired_calibration.py --jev-cache <dir with test-NNNNN.json files>
"""
from __future__ import annotations
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum.metrics import calibration  # noqa: E402

REPORTED = {"ece_10_bins": 0.0341, "brier_score": 0.1193, "log_loss_clipped_1e_15": 0.784, "accuracy": 0.9240}
SYSTEMS = [("primary", "clean"), ("secondary", "r6_best_parityret"), ("descriptive", "default"),
           ("descriptive", "r5_knn_alone")]
BOOT_N, SEED, FLOOR = 10_000, 20261010, 0.005


def ours(name):
    rows = [json.loads(l) for l in gzip.open(ROOT / "results/predictions" / f"banking77_24shot_{name}.jsonl.gz", "rt")]
    return np.array([r["probs"] for r in rows], float), np.array([r["gold"] for r in rows])


def jev(cache, label_names):
    idx = {n: i for i, n in enumerate(label_names)}; P = np.zeros((3080, 77)); pred_ok = []
    for i in range(3080):
        a = json.loads((Path(cache) / f"test-{i:05d}.json").read_text())["answer"]
        for k, v in a["probabilities"].items():
            P[i, idx[k]] = v
        pred_ok.append(a["choice"])
    return P, pred_ok


def per_item(P, y):
    ll = -np.log(np.maximum(P[np.arange(len(y)), y], 1e-15))
    oh = np.zeros_like(P); oh[np.arange(len(y)), y] = 1
    return ll, ((P - oh) ** 2).sum(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jev-cache", required=True); a = ap.parse_args()
    from datasets import load_dataset
    names = load_dataset("legacy-datasets/banking77", revision="f54121560de48f2852f90be299010d1d6dc612ec",
                         split="test").features["label"].names
    _, y = ours("clean")
    Pj, choices = jev(a.jev_cache, names)
    cj = calibration(Pj, y); acc_choice = float(np.mean([names[t] == c for t, c in zip(y, choices)]))
    print(f"control — Jev recomputed: ECE {cj['ece_10_bins']:.4f} Brier {cj['brier_score']:.4f} "
          f"log loss {cj['log_loss_clipped_1e_15']:.3f} acc(choice) {acc_choice:.4f}  | reported {REPORTED}")
    for k in ("ece_10_bins", "brier_score", "log_loss_clipped_1e_15"):
        assert abs(cj[k] - REPORTED[k]) < 1e-3, f"instrument control failed on {k}"
    assert abs(acc_choice - REPORTED["accuracy"]) < 1e-4, "instrument control failed on accuracy"
    Pj_floor = np.maximum(Pj, FLOOR); Pj_floor /= Pj_floor.sum(1, keepdims=True)
    rng = np.random.default_rng(SEED); n = len(y); boots = [rng.integers(0, n, n) for _ in range(BOOT_N)]
    out = {"control": {"jev_recomputed": cj, "jev_choice_accuracy": acc_choice, "reported": REPORTED}, "results": {}}
    for variant, PJ in (("as_published", Pj), (f"jev_floor_{FLOOR}", Pj_floor)):
        llj, brj = per_item(PJ, y)
        for role, name in SYSTEMS:
            Po, yo = ours(name); assert (yo == y).all()
            llo, bro = per_item(Po, y); co = calibration(Po, y); cjv = calibration(PJ, y)
            d_ll, d_br = llo - llj, bro - brj
            ci = lambda d: [float(x) for x in np.percentile([d[i].mean() for i in boots], [2.5, 97.5])]
            d_ece = [calibration(Po[i], y[i])["ece_10_bins"] - calibration(PJ[i], y[i])["ece_10_bins"] for i in boots[:2000]]
            r = {"role": role, "ours": co, "jev": cjv,
                 "diff_log_loss": float(d_ll.mean()), "ci_log_loss": ci(d_ll),
                 "diff_brier": float(d_br.mean()), "ci_brier": ci(d_br),
                 "diff_ece": co["ece_10_bins"] - cjv["ece_10_bins"],
                 "ci_ece": [float(x) for x in np.percentile(d_ece, [2.5, 97.5])], "ece_boot_n": 2000}
            out["results"][f"{variant}/{name}"] = r
            flag = lambda c: "better" if c[1] < 0 else ("worse" if c[0] > 0 else "n.s.")
            print(f"{variant:16s} {name:20s} ({role:11s}) acc {co['accuracy']:.4f} | Δlogloss {r['diff_log_loss']:+.3f} "
                  f"[{r['ci_log_loss'][0]:+.3f},{r['ci_log_loss'][1]:+.3f}] {flag(r['ci_log_loss'])} | Δbrier "
                  f"{r['diff_brier']:+.4f} [{r['ci_brier'][0]:+.4f},{r['ci_brier'][1]:+.4f}] {flag(r['ci_brier'])} | "
                  f"ΔECE {r['diff_ece']:+.4f} [{r['ci_ece'][0]:+.4f},{r['ci_ece'][1]:+.4f}] {flag(r['ci_ece'])}")
    (ROOT / "paper" / "generated" / "jev_paired_calibration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
