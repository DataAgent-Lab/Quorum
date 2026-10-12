"""Statistics shared by the decision-model study (`provenance/decision13_protocol.md` §7): exact two-sided McNemar
and Holm adjustment."""
import math


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=ps.__getitem__); adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj
