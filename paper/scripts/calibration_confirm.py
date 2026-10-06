#!/usr/bin/env python3
"""Confirmatory calibration analysis (T2.14) — DECLARED BEFORE RUNNING; committed before its first execution.

Question: the zero-shot ensemble is under-confident (claims Z4). An exploratory leave-one-dataset-out analysis on
test sets (Z4b) suggested one task-agnostic temperature fixes it. This script confirms or refutes that WITHOUT any
test data in the fitting.

Primary procedure (fixed):
  1. Inputs for fitting: the pre-registered train-side sample (results/r12/*_train1000.jsonl.gz; 7 datasets x
     {names, descriptions} x 1,000 rows = 14,000 rows; ids committed at 3ee8fee before the run).
  2. Fit ONE temperature T* on the ensemble's log-probabilities by minimising the pooled negative log-likelihood
     over all 14,000 rows (golden-section search on [0.05, 10], 100 iterations). Nothing else is fitted.
  3. Apply T* once to each of the 14 full test conditions (results/predictions/*); report ECE-10 (the reproduction's
     definition), Brier and log loss, raw vs T*. Accuracy is unchanged by construction (temperature keeps the argmax).
  4. Success criterion (declared): ECE-10 decreases on all 14 test conditions. Reported as-is whatever happens.
Secondary (declared, descriptive): bootstrap 95% CIs on ECE-10 (raw and T*), 2,000 row resamples, seed 20261009;
sensitivity: one temperature per label-text condition (fit on that condition's 7,000 train rows only).

    python paper/scripts/calibration_confirm.py                 # original ensemble (PrismNLI member)
    python paper/scripts/calibration_confirm.py --tag clean_c   # R13: same procedure, documented-clean NLI member
    python paper/scripts/calibration_confirm.py --tag mnli_snli # R13: same procedure, SNLI+MNLI NLI member
The --tag runs (added 2026-10-06, committed before running them) apply the IDENTICAL declared procedure to the R13
ensembles: train-side inputs results/r13/<ds>_<cond>_train1000_<tag>.jsonl.gz (same pre-registered ids), test inputs
results/predictions/<ds>_<cond>_r13_<tag>.jsonl.gz, output paper/generated/calibration_confirm_<tag>.json.
"""
from __future__ import annotations
import gzip, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from quorum.metrics import calibration  # noqa: E402
from zero_shot_tables import DATASETS, CONDS  # noqa: E402

BOOT_N, BOOT_SEED = 2000, 20261009


def logp(path):
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    return np.log(np.clip(np.array([r["probs"] for r in rows]), 1e-12, 1.0)), np.array([r["gold"] for r in rows])


def softmax_T(L, T):
    Z = L / T; Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def pooled_nll(pairs, T):
    tot = n = 0
    for L, y in pairs:
        P = softmax_T(L, T); tot += -np.log(np.maximum(P[np.arange(len(y)), y], 1e-15)).sum(); n += len(y)
    return tot / n


def fit_T(pairs):
    a, b, g = 0.05, 10.0, (math.sqrt(5) - 1) / 2
    for _ in range(100):
        c, d = b - g * (b - a), a + g * (b - a)
        if pooled_nll(pairs, c) < pooled_nll(pairs, d):
            b = d
        else:
            a = c
    return (a + b) / 2


def boot_ece(P, y, rng):
    n = len(y); vals = []
    for _ in range(BOOT_N):
        i = rng.integers(0, n, n); vals.append(calibration(P[i], y[i])["ece_10_bins"])
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); tag = ap.parse_args().tag
    tr = (lambda ds, c: ROOT / "results" / "r13" / f"{ds}_{c}_train1000_{tag}.jsonl.gz") if tag else \
         (lambda ds, c: ROOT / "results" / "r12" / f"{ds}_{c}_train1000.jsonl.gz")
    te = (lambda ds, c: ROOT / "results" / "predictions" / f"{ds}_{c}_r13_{tag}.jsonl.gz") if tag else \
         (lambda ds, c: ROOT / "results" / "predictions" / f"{ds}_{c}.jsonl.gz")
    train = {(ds, c): logp(tr(ds, c)) for ds, _ in DATASETS for c, _ in CONDS}
    T_star = fit_T(list(train.values()))
    T_cond = {c: fit_T([v for (d, cc), v in train.items() if cc == c]) for c, _ in CONDS}
    rng = np.random.default_rng(BOOT_SEED); res = {"T_star": T_star, "T_per_condition": T_cond, "tests": {}}
    wins = 0
    print(f"T* (one global temperature, train-side only) = {T_star:.4f}; per-condition: "
          + ", ".join(f"{c} {t:.4f}" for c, t in T_cond.items()))
    print(f"{'dataset':10s} {'cond':6s} {'ECE raw [95% CI]':>24s} {'ECE T* [95% CI]':>24s} {'Brier raw→T*':>16s} {'LL raw→T*':>14s}  ECE(T_cond)")
    for ds, _ in DATASETS:
        for c, _ in CONDS:
            L, y = logp(te(ds, c))
            P0, P1, P2 = softmax_T(L, 1.0), softmax_T(L, T_star), softmax_T(L, T_cond[c])
            raw, cal, cal_c = calibration(P0, y), calibration(P1, y), calibration(P2, y)
            ci0, ci1 = boot_ece(P0, y, rng), boot_ece(P1, y, rng)
            assert (P0.argmax(1) == P1.argmax(1)).all()
            wins += cal["ece_10_bins"] < raw["ece_10_bins"]
            res["tests"][f"{ds}/{c}"] = {"raw": raw, "T_star": cal, "T_condition": cal_c,
                                         "ece_ci95_raw": ci0, "ece_ci95_T_star": ci1}
            print(f"{ds:10s} {c[:6]:6s} {raw['ece_10_bins']:.4f} [{ci0[0]:.3f},{ci0[1]:.3f}]   "
                  f"{cal['ece_10_bins']:.4f} [{ci1[0]:.3f},{ci1[1]:.3f}]   {raw['brier_score']:.3f}→{cal['brier_score']:.3f}"
                  f"   {raw['log_loss_clipped_1e_15']:.3f}→{cal['log_loss_clipped_1e_15']:.3f}   {cal_c['ece_10_bins']:.4f}")
    res["success_criterion"] = {"rule": "ECE-10 decreases on all 14 test conditions", "met": wins == 14,
                                "decreased_on": int(wins)}
    print(f"success criterion (ECE decreases on all 14): {'MET' if wins == 14 else 'NOT MET'} ({wins}/14)")
    (ROOT / "paper" / "generated" / (f"calibration_confirm_{tag}.json" if tag else "calibration_confirm.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
