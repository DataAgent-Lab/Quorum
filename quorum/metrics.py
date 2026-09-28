"""Evaluation helpers: accuracy and the paired McNemar exact test (for comparing two classifiers on the same
test set — the right test when both predict on identical items)."""
from __future__ import annotations
from math import comb
import numpy as np


def accuracy(pred, y) -> float:
    return float((np.asarray(pred) == np.asarray(y)).mean())


def mcnemar(pred_a, pred_b, y) -> dict:
    """Exact two-sided McNemar test between systems A and B on the same items. b = A right & B wrong,
    c = A wrong & B right. Returns the counts, accuracies, difference and exact p-value."""
    y = np.asarray(y); a = np.asarray(pred_a) == y; b = np.asarray(pred_b) == y
    n_b = int((a & ~b).sum()); n_c = int((~a & b).sum()); n = n_b + n_c
    if n == 0:
        p = 1.0
    else:
        k = min(n_b, n_c)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"acc_a": round(float(a.mean()), 4), "acc_b": round(float(b.mean()), 4),
            "diff": round(float(a.mean() - b.mean()), 4),
            "b_a_wins": n_b, "c_b_wins": n_c, "n_discordant": n, "p_two_sided": round(p, 6),
            "verdict": ("A significantly better" if n_b > n_c and p <= 0.05 else
                        "B significantly better" if n_c > n_b and p <= 0.05 else "no significant difference")}
