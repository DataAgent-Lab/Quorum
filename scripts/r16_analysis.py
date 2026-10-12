#!/usr/bin/env python3
"""R16 Bitext 24-shot: the pre-declared analyses of `provenance/bitext_24shot_protocol.md` §9, plus the paper-side
amendment A2 (`docs/phases/2.0/bitext24/AMENDMENT_A2.md`). Committed before any arm's result is read jointly.
Everything below is reported whatever its outcome.

    python scripts/r16_analysis.py      # -> results/r16/analysis.json

Arms and their per-item dumps:
  open                     results/predictions/bitext_24shot_clean.jsonl.gz   (the single test run)
  clef, clef_flash, pplx   results/predictions/bitext_r16_<arm>.jsonl.gz
  jev                      docs/phases/2.0/jev/bitext24/bitext_r16_jev.jsonl.gz (A2 fields: pred = returned choice id,
                           -1 if unknown; unsent; failed)
A missing arm is listed as missing and never imputed.

Families, each with its own Holm correction over its pairs (10 when all 5 arms are present):
  primary    §9.1, all 5,375 test items;
  subset     A2, the 2,484 non-near-duplicate items (docs/phases/2.0/bitext24/subset_a2.json).

Per family:
  - exact two-sided McNemar for every pair;
  - ECE-10 (probability of the predicted intent) and Brier (summed over the 27 intents) per arm;
  - paired bootstrap of each arm against the open stack (seed 20261012; 10,000 resamples for Brier, 2,000 for ECE);
  - no log loss against Jev (ledger J2).
Descriptive, outside every family: accuracy, macro-F1, per-intent accuracy, unanswered / failed / unsent counts, Jev
items where choice != probability argmax, the BM25 top-1 floor row (A2.2), the duplicate-audit counts (A2.3) and the
examples kept.

Item rules:
  - unanswered (pplx over the release limit) and failed (Jev) items count as wrong; for calibration they carry a
    uniform distribution;
  - unsent Jev items (A2.4, budget cap) are excluded from every pair and every calibration comparison that involves
    Jev.
"""
import gzip, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PRED, OUT = ROOT / "results" / "predictions", ROOT / "results" / "r16"
ARMS = {"open": PRED / "bitext_24shot_clean.jsonl.gz", "jev": ROOT / "docs/phases/2.0/jev/bitext24/bitext_r16_jev.jsonl.gz",
        "clef": PRED / "bitext_r16_clef.jsonl.gz", "clef_flash": PRED / "bitext_r16_clef_flash.jsonl.gz",
        "pplx": PRED / "bitext_r16_pplx.jsonl.gz"}
SUBSET = ROOT / "docs/phases/2.0/bitext24/subset_a2.json"
K, SEED, N = 27, 20261012, 5375


def load(arm, path):
    rows = sorted((json.loads(l) for l in gzip.open(path, "rt")), key=lambda r: r["idx"])
    assert [r["idx"] for r in rows] == list(range(N)), f"{arm}: expected idx 0..{N - 1}"
    gold = np.array([r["gold"] for r in rows]); P = np.full((N, K), 1 / K)
    pred = np.full(N, -1); wrong_forced = np.zeros(N, bool); sent = np.ones(N, bool); choice_ne_argmax = 0
    for i, r in enumerate(rows):
        if r.get("unsent"):
            sent[i] = False; continue
        if r.get("unanswered") or r.get("failed") or r.get("probs") is None:
            wrong_forced[i] = True; continue
        P[i] = np.asarray(r["probs"], float)
        p = r.get("pred")
        if arm == "jev":                                          # A2.5: the returned choice is the prediction
            pred[i] = int(p) if p is not None else -1
            choice_ne_argmax += int(pred[i] != int(np.argmax(P[i])))
        else:
            pred[i] = int(p) if p is not None else int(np.argmax(P[i]))
    hit = (pred == gold) & ~wrong_forced
    return {"gold": gold, "P": P, "pred": pred, "hit": hit, "sent": sent, "forced_wrong": wrong_forced,
            "choice_ne_argmax": choice_ne_argmax if arm == "jev" else None}


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


def family(name, items, A, rng):
    """All pairs + calibration vs the open stack on `items` (a boolean mask), Jev's unsent items excluded."""
    out = {"family": name, "n_items": int(items.sum()), "per_arm": {}, "pairs": [], "calibration_vs_open": {}}
    for a, d in A.items():
        m = items & d["sent"]; hit, P, gold = d["hit"][m], d["P"][m], d["gold"][m]
        out["per_arm"][a] = {"n": int(m.sum()), "accuracy": round(float(hit.mean()), 4), "correct": int(hit.sum()),
                             "forced_wrong": int(d["forced_wrong"][m].sum()),
                             "macro_f1": round(macro_f1(np.where(d["forced_wrong"][m], -1, d["pred"][m]), gold), 4),
                             "ece10": round(ece10(P, hit), 4), "brier": round(float(brier_items(P, gold).mean()), 4),
                             "per_intent_accuracy": [round(float(hit[gold == c].mean()), 4) if (gold == c).any() else None
                                                     for c in range(K)]}
    names = list(A); pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    ps = []
    for x, y in pairs:
        m = items & A[x]["sent"] & A[y]["sent"]; hx, hy = A[x]["hit"][m], A[y]["hit"][m]
        b, c = int((hx & ~hy).sum()), int((hy & ~hx).sum()); p = mcnemar_exact(b, c); ps.append(p)
        out["pairs"].append({"a": x, "b": y, "n": int(m.sum()), "a_only": b, "b_only": c, "p": p})
    for r, ph in zip(out["pairs"], holm(ps)):
        r["p_holm"] = ph; r["holm_family"] = f"{name} ({len(ps)} pairs)"
    if "open" in A:
        for a in names:
            if a == "open":
                continue
            m = items & A[a]["sent"]; n = int(m.sum()); gold = A["open"]["gold"][m]
            Po, ho, Pa, ha = A["open"]["P"][m], A["open"]["hit"][m], A[a]["P"][m], A[a]["hit"][m]
            bo, ba = brier_items(Po, gold), brier_items(Pa, gold)
            d_b = np.concatenate([(lambda ix: ba[ix].mean(1) - bo[ix].mean(1))(rng.integers(0, n, (1000, n)))
                                  for _ in range(10)])
            d_e = np.array([ece10(Pa[ix], ha[ix]) - ece10(Po[ix], ho[ix]) for ix in rng.integers(0, n, (2000, n))])
            out["calibration_vs_open"][a] = {
                "n": n, "brier_diff_arm_minus_open": round(float(ba.mean() - bo.mean()), 4),
                "brier_ci95": [round(float(x), 4) for x in np.percentile(d_b, [2.5, 97.5])],
                "ece_diff_arm_minus_open": round(ece10(Pa, ha) - ece10(Po, ho), 4),
                "ece_ci95": [round(float(x), 4) for x in np.percentile(d_e, [2.5, 97.5])]}
    return out


def floor_row(subset_mask):
    """A2.2: the label of the BM25 top-1 example (manifest retrieved_t_idx[0]) on the full set and the subset."""
    from scripts.bitext24_bodies import split
    raw, names, T, S, P_, test = split()
    man = [json.loads(l) for l in gzip.open(ROOT / "results" / "bitext24" / "request_manifest.jsonl.gz", "rt")]
    gold = np.array([r["label"] for r in test]); top1 = np.array([T[m["retrieved_t_idx"][0]]["label"] for m in man])
    hit = top1 == gold
    return {"full": round(float(hit.mean()), 4), "subset": round(float(hit[subset_mask].mean()), 4)}


def main():
    A = {a: load(a, p) for a, p in ARMS.items() if p.exists()}
    golds = [d["gold"] for d in A.values()]
    assert all((g == golds[0]).all() for g in golds), "arms disagree on gold labels"
    sub = json.loads(SUBSET.read_text()); sub_mask = np.zeros(N, bool); sub_mask[sub["subset_idx"]] = True
    assert sub["n_subset"] == int(sub_mask.sum()) and sub["manifest_sha256"].startswith("1420b379")
    rng = np.random.default_rng(SEED)
    res = {"protocol": "provenance/bitext_24shot_protocol.md (9d6dc44; A1 3d84cc3; A3 engine) + "
                       "docs/phases/2.0/bitext24/AMENDMENT_A2.md (f4ac106)",
           "arms_present": list(A), "arms_missing": [a for a in ARMS if a not in A],
           "primary": family("primary", np.ones(N, bool), A, rng),
           "secondary_subset_A2": family("subset_A2", sub_mask, A, rng),
           "jev_unsent": int((~A["jev"]["sent"]).sum()) if "jev" in A else None,
           "jev_choice_ne_argmax": A["jev"]["choice_ne_argmax"] if "jev" in A else None,
           "duplicate_audit_A2": {k: sub[k] for k in ("exact_duplicates", "cos_ge_threshold", "threshold", "n_subset")}}
    try:
        res["floor_bm25_top1_A2"] = floor_row(sub_mask)
    except Exception as e:                                      # needs the pinned dataset; reported if unavailable
        res["floor_bm25_top1_A2"] = {"error": f"{type(e).__name__}: {e}"}
    man = [json.loads(l) for l in gzip.open(ROOT / "results" / "bitext24" / "request_manifest.jsonl.gz", "rt")]
    kept = [m["examples_kept"] for m in man]
    res["examples_kept"] = {"min": min(kept), "median": int(np.median(kept)), "all_24": sum(k == 24 for k in kept)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({f: {a: v["accuracy"] for a, v in res[f]["per_arm"].items()} for f in ("primary", "secondary_subset_A2")}),
          "missing:", res["arms_missing"], flush=True)


if __name__ == "__main__":
    main()
