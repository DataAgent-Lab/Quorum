"""Evaluation helpers: accuracy and the paired McNemar exact test (for comparing two classifiers on the same
test set — the right test when both predict on identical items)."""
from __future__ import annotations
from math import comb
import numpy as np


def accuracy(pred, y) -> float:
    return float((np.asarray(pred) == np.asarray(y)).mean())


def calibration(probs, y, n_bins: int = 10) -> dict:
    """Calibration of a (N, K) probability matrix against gold labels y, with the definitions used by the public
    Banking77 reproduction of the closed API we compare against (its frozen-v1/src/score.py):
      - ece_10_bins: confidence = max probability; equal-width bins [i/10, (i+1)/10), last bin closed at 1;
        sum over bins of (bin share) * |mean confidence - accuracy|
      - brier_score: per item sum over classes of (p - onehot)^2, averaged
      - log_loss_clipped_1e_15: mean of -log(max(1e-15, p_gold))
    plus adaptive_ece_10 (10 equal-mass bins over sorted confidence) and mean confidence."""
    P = np.asarray(probs, float); y = np.asarray(y); n, k = P.shape
    conf = P.max(1); hit = (P.argmax(1) == y).astype(float)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= 1.0))
        if m.any():
            ece += m.sum() / n * abs(conf[m].mean() - hit[m].mean())
    order = np.argsort(conf); aece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            aece += len(chunk) / n * abs(conf[chunk].mean() - hit[chunk].mean())
    onehot = np.zeros_like(P); onehot[np.arange(n), y] = 1.0
    return {"ece_10_bins": float(ece), "adaptive_ece_10": float(aece),
            "brier_score": float(((P - onehot) ** 2).sum(1).mean()),
            "log_loss_clipped_1e_15": float(np.mean(-np.log(np.maximum(1e-15, P[np.arange(n), y])))),
            "mean_confidence": float(conf.mean()), "accuracy": float(hit.mean())}


def mcnemar_exact_p(n_b: int, n_c: int) -> float:
    """Unrounded exact two-sided McNemar p-value from the discordant counts (binomial, p = 0.5). Exact integer
    arithmetic, so very small p-values are not lost to rounding (report them as-is or as a bound)."""
    n = n_b + n_c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(n_b, n_c) + 1)) / (2 ** n))


def mcnemar(pred_a, pred_b, y) -> dict:
    """Exact two-sided McNemar test between systems A and B on the same items. b = A right & B wrong,
    c = A wrong & B right. Returns the counts, accuracies, difference and exact p-value (rounded to 6 dp here;
    use `mcnemar_exact_p` for the unrounded value)."""
    y = np.asarray(y); a = np.asarray(pred_a) == y; b = np.asarray(pred_b) == y
    n_b = int((a & ~b).sum()); n_c = int((~a & b).sum()); n = n_b + n_c
    p = mcnemar_exact_p(n_b, n_c)
    return {"acc_a": round(float(a.mean()), 4), "acc_b": round(float(b.mean()), 4),
            "diff": round(float(a.mean() - b.mean()), 4),
            "b_a_wins": n_b, "c_b_wins": n_c, "n_discordant": n, "p_two_sided": round(p, 6),
            "verdict": ("A significantly better" if n_b > n_c and p <= 0.05 else
                        "B significantly better" if n_c > n_b and p <= 0.05 else "no significant difference")}
