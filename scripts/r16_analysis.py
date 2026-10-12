#!/usr/bin/env python3
"""R16 Bitext 24-shot, the pre-declared analyses of `provenance/bitext_24shot_protocol.md` §9. Committed before any
arm's result is read jointly. Every analysis is reported whatever its outcome.

    python scripts/r16_analysis.py      # -> results/r16/analysis.json

Arms and their per-item dumps (results/predictions/):
  open   bitext_24shot_clean.jsonl.gz      (the open stack's single test run)
  jev    bitext_r16_jev.jsonl.gz
  clef, clef_flash, pplx    bitext_r16_<arm>.jsonl.gz
A missing arm is listed as missing, never imputed.

1. Primary: exact two-sided McNemar for every pair of the arms present (10 pairs when all 5 are present), with Holm
   correction over that family.
2. Calibration: ECE-10 on the predicted intent's probability, and Brier summed over the 27 intents. Each arm is
   compared with the open stack by paired bootstrap (seed 20261012; 10,000 resamples for Brier, 2,000 for ECE).
   No log loss against Jev (ledger J2).
3. Descriptive: accuracy, macro-F1, per-intent accuracy, unanswered count, and the examples kept (from the manifest).

An unanswered item counts as wrong. For calibration it carries a uniform distribution and appears in the unanswered
count.
"""
import gzip, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED, OUT = ROOT / "results" / "predictions", ROOT / "results" / "r16"
ARMS = {"open": "bitext_24shot_clean.jsonl.gz", "jev": "bitext_r16_jev.jsonl.gz", "clef": "bitext_r16_clef.jsonl.gz",
        "clef_flash": "bitext_r16_clef_flash.jsonl.gz", "pplx": "bitext_r16_pplx.jsonl.gz"}
K, SEED = 27, 20261012


def load(fn, raw=None):
    rows = [json.loads(l) for l in gzip.open(PRED / fn, "rt")]
    rows.sort(key=lambda r: r["idx"])
    gold = np.array([r["gold"] for r in rows]); P = np.full((len(rows), K), 1 / K); un = np.zeros(len(rows), bool)
    pred = np.full(len(rows), -1)
    for i, r in enumerate(rows):
        if r.get("unanswered") or r.get("failed") or r.get("probs") is None:
            un[i] = True; continue
        P[i] = np.asarray(r["probs"], float)
        p = r["pred"]
        pred[i] = p if isinstance(p, int) else int(np.argmax(P[i]))
    return [r["idx"] for r in rows], gold, P, pred, un


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def holm(ps):
    order = sorted(range(len(ps)), key=ps.__getitem__); adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


def ece10(P, hit):
    conf = P.max(1); e = 0.0
    for i in range(10):
        m = (conf >= i / 10) & ((conf < (i + 1) / 10) if i < 9 else (conf <= 1.0))
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def brier_items(P, gold):
    return ((P - np.eye(K)[gold]) ** 2).sum(1)


def macro_f1(pred, gold):
    f = []
    for c in range(K):
        tp = int(((pred == c) & (gold == c)).sum()); a = int((gold == c).sum()); p = int((pred == c).sum())
        if a:
            f.append(2 * tp / (a + p) if a + p else 0.0)
    return float(np.mean(f))


def main():
    present = {a: load(fn) for a, fn in ARMS.items() if (PRED / fn).exists()}
    res = {"protocol": "provenance/bitext_24shot_protocol.md (9d6dc44 + A1 3d84cc3 + A2)", "arms_present": list(present),
           "arms_missing": [a for a in ARMS if a not in present], "per_arm": {}, "pairs": [], "calibration_vs_open": {}}
    ids0, gold0 = None, None
    for a, (ids, gold, P, pred, un) in present.items():
        if ids0 is None:
            ids0, gold0 = ids, gold
        assert ids == ids0 and (gold == gold0).all(), f"{a}: items or gold differ from the other arms"
        hit = (pred == gold) & ~un
        res["per_arm"][a] = {"n": len(ids), "accuracy": round(float(hit.mean()), 4), "correct": int(hit.sum()),
                             "unanswered": int(un.sum()), "macro_f1": round(macro_f1(np.where(un, -1, pred), gold), 4),
                             "ece10": round(ece10(P, hit), 4), "brier": round(float(brier_items(P, gold).mean()), 4),
                             "per_intent_accuracy": [round(float(hit[gold == c].mean()), 4) for c in range(K)]}
    hits = {a: ((v[3] == v[1]) & ~v[4]) for a, v in present.items()}
    names = list(present); pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    ps = []
    for x, y in pairs:
        b = int((hits[x] & ~hits[y]).sum()); c = int((hits[y] & ~hits[x]).sum()); p = mcnemar_exact(b, c); ps.append(p)
        res["pairs"].append({"a": x, "b": y, "a_only": b, "b_only": c, "p": p})
    for r, ph in zip(res["pairs"], holm(ps)):
        r["p_holm"] = ph; r["holm_family"] = len(ps)
    if "open" in present:
        rng = np.random.default_rng(SEED); _, gold, Po, _, _ = present["open"]; n = len(gold)
        bo, ho = brier_items(Po, gold), hits["open"]
        for a in names:
            if a == "open":
                continue
            _, _, Pa, _, _ = present[a]; ba, ha = brier_items(Pa, gold), hits[a]
            d_b = np.concatenate([(lambda ix: ba[ix].mean(1) - bo[ix].mean(1))(rng.integers(0, n, (1000, n)))
                                  for _ in range(10)])                 # 10,000 resamples in chunks
            idx_e = rng.integers(0, n, (2000, n))
            d_e = np.array([ece10(Pa[ix], ha[ix]) - ece10(Po[ix], ho[ix]) for ix in idx_e])
            res["calibration_vs_open"][a] = {
                "brier_diff_arm_minus_open": round(float(ba.mean() - bo.mean()), 4),
                "brier_ci95": [round(float(x), 4) for x in np.percentile(d_b, [2.5, 97.5])],
                "ece_diff_arm_minus_open": round(ece10(Pa, ha) - ece10(Po, ho), 4),
                "ece_ci95": [round(float(x), 4) for x in np.percentile(d_e, [2.5, 97.5])],
                "log_loss": "not compared" if a == "jev" else "not part of the declared analyses"}
    man = [json.loads(l) for l in gzip.open(ROOT / "results" / "bitext24" / "request_manifest.jsonl.gz", "rt")]
    kept = [m["examples_kept"] for m in man]
    res["examples_kept"] = {"min": min(kept), "median": int(np.median(kept)), "all_24": sum(k == 24 for k in kept)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({a: v["accuracy"] for a, v in res["per_arm"].items()}), "missing:", res["arms_missing"], flush=True)


if __name__ == "__main__":
    main()
