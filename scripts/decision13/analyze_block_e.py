#!/usr/bin/env python3
"""Decision-model study, block E analysis (`provenance/decision13_protocol.md` §5–§7, §9, amendments 4–7).

    python scripts/decision13/analyze_block_e.py      # -> results/decision13/analysis_block_e.json

Inputs (all under results/decision13/):
  predictions/<set>_<model>.jsonl.gz   per-item records {idx, gold, group?, pred, probs, unanswered, correct,
                                       input_tokens, status?}; pplx-decider EquiBench uses the `_amend6` file;
  summary_<set>_<model>.json           the runner's own summary, checked against the recomputed full-set metrics;
  restricted_aggregates.json           sets whose per-item data is not redistributable (TOSEM-DBRD): counts only.

- Per set x model: full-set metrics (unanswered = wrong) and like-for-like subset metrics (amendment 5: items pplx
  answered and no Clef model truncated, i.e. both Clef input_tokens < 16,384).
- Pairwise exact two-sided McNemar on identical items, Holm across all pairs x decision sets (PAWS excluded).
- Pairwise paired bootstrap of ECE-10 and Brier differences (10,000 resamples, seed 20261008) on the items both models
  answered; two-sided bootstrap p, Holm across pairs x decision sets.
- Primary score (§7.2): Holm-significant accuracy wins - losses; exposure sensitivity (§9) without PatchNet and
  SecVulEval. Block C is added by analyze_block_c.py.

Restricted sets. For DBRD only b/c counts, n and accuracies are published, so its accuracy tests are recomputed from
the counts (exact McNemar needs nothing else). Its calibration tests need per-item probabilities and cannot be
recomputed here: the ECE-10 Holm family below therefore covers the redistributable decision sets only (8 sets x 3
pairs). The bootstrap generator is shared across sets in protocol order, so for DBRD the script draws exactly the
resamples the full analysis drew (same shapes, from `n_both_answered`) and discards them; every later set therefore
sees the same random stream as the full analysis.
"""
import gzip, json, os, sys
from itertools import combinations
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import holm, mcnemar_exact

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "decision13"
MODELS = ["pplx_v11", "clef_flash", "clef"]
DECISION_SETS = ["patchnet", "dbrd", "equibench", "poj104", "primevul_pair", "primevul_func", "svtrusteval", "secvuleval", "juliet"]
CONTROL_SETS = ["paws"]
EXPOSED_ALL = {"patchnet", "secvuleval"}                          # §9: assume exposure for every model
CLEF_MAX = 16384
PATCHNET_PPLX_OVER = {3386}                                       # amendment 4 item 2
SEED, B = 20261008, 10000


def load(s, m):
    path = RES / "predictions" / f"{s}_{m}.jsonl.gz"
    if (s, m) == ("equibench", "pplx_v11"):
        path = RES / "predictions" / f"{s}_{m}_amend6.jsonl.gz"  # amendment 6
    recs = {}
    for r in map(json.loads, gzip.open(path, "rt")):
        if m == "pplx_v11" and s == "patchnet" and r["idx"] in PATCHNET_PPLX_OVER:
            r = dict(r, pred=None, probs=None, unanswered=True, correct=False, status="over_release_limit")
        recs[r["idx"]] = r
    return recs, path.name


def gold_key(r):
    g = r["gold"]
    if isinstance(g, bool):
        return "true" if g else "false"
    return r["pred"] if r.get("correct") else g                   # multi-accept golds: the accepted letter chosen


def brier_ece(rs):
    rs = [r for r in rs if not r["unanswered"]]
    if not rs:
        return None, None
    brier = sum(sum((p - (1.0 if k == gold_key(r) else 0.0)) ** 2 for k, p in r["probs"].items()) for r in rs) / len(rs)
    bins = [[] for _ in range(10)]
    for r in rs:
        c = max(r["probs"].values()); bins[min(9, int(c * 10))].append((c, r["correct"]))
    ece = sum(len(b) / len(rs) * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b)) for b in bins if b)
    return brier, ece


def metrics(rs):
    n = len(rs); ans = [r for r in rs if not r["unanswered"]]
    labels = sorted({json.dumps(r["gold"]) for r in rs} | {json.dumps(r["pred"]) for r in ans}, key=str)
    f1s = []
    for lab in labels:
        tp = sum(json.dumps(r["pred"]) == lab and json.dumps(r["gold"]) == lab for r in ans)
        fp = sum(json.dumps(r["pred"]) == lab and json.dumps(r["gold"]) != lab for r in ans)
        fn = sum(json.dumps(r["gold"]) == lab and (r["unanswered"] or json.dumps(r["pred"]) != lab) for r in rs)
        if tp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    brier, ece = brier_ece(rs)
    out = {"n": n, "answered": len(ans), "unanswered": n - len(ans), "accuracy": sum(r["correct"] for r in rs) / n,
           "macro_f1": sum(f1s) / len(f1s) if f1s else None, "brier": brier, "ece10": ece}
    if rs and isinstance(rs[0]["gold"], bool):
        tp = sum(r["pred"] is True and r["gold"] is True for r in ans); fp = sum(r["pred"] is True and r["gold"] is False for r in ans)
        fn = sum(r["gold"] is True and r["pred"] is not True for r in rs)
        out["f1_positive"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return out


def check_against_summary(s, m, got):
    """The runner's own summary for the file used must agree with the recomputed full-set metrics."""
    tag = f"{s}_{m}" + {("patchnet", "pplx_v11"): "_lengthrule", ("equibench", "pplx_v11"): "_amend6"}.get((s, m), "")
    ref = json.loads((RES / f"summary_{tag}.json").read_text())["overall"]
    for k in ("n", "answered", "accuracy", "macro_f1", "brier", "ece10"):
        assert abs((ref[k] or 0) - (got[k] or 0)) < 2e-6, f"{tag}: {k} summary {ref[k]} vs recomputed {got[k]}"


def arrays(rs):
    """Per item: confidence of the predicted option, correct (0/1), ECE bin, Brier term."""
    conf = np.array([max(r["probs"].values()) for r in rs]); hit = np.array([float(r["correct"]) for r in rs])
    bins = np.minimum(9, (conf * 10).astype(int))
    br = np.array([sum((p - (1.0 if k == gold_key(r) else 0.0)) ** 2 for k, p in r["probs"].items()) for r in rs])
    return conf, hit, bins, br


def resample_shapes(n):
    return [(min(500, B - start), n) for start in range(0, B, 500)]


def boot_calib(ra, rb, rng):
    """Paired bootstrap over items both models answered: diff = A - B for ECE-10 and Brier (lower is better)."""
    A, Bm = arrays(ra), arrays(rb); n = len(ra)
    onehot = [np.eye(10)[x[2]] for x in (A, Bm)]
    gaps = [onehot[i] * (x[0] - x[1])[:, None] for i, x in enumerate((A, Bm))]
    d_ece, d_bri = [], []
    for shape in resample_shapes(n):
        idx = rng.integers(0, n, size=shape)
        w = np.stack([np.bincount(row, minlength=n) for row in idx]).astype(float)
        tot = w.sum(1)
        e = [np.abs(w @ g).sum(1) / tot for g in gaps]
        br = [(w @ x[3]) / tot for x in (A, Bm)]
        d_ece.append(e[0] - e[1]); d_bri.append(br[0] - br[1])
    out = {}
    for name, d in (("ece10", np.concatenate(d_ece)), ("brier", np.concatenate(d_bri))):
        lo, hi = np.percentile(d, [2.5, 97.5]); p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out[name] = {"ci95": [float(lo), float(hi)], "p_boot": float(p)}
    ea, eb = brier_ece(ra)[1], brier_ece(rb)[1]; ba, bb = brier_ece(ra)[0], brier_ece(rb)[0]
    out["ece10"]["diff"] = ea - eb; out["brier"]["diff"] = ba - bb
    return out


def restricted_entry(s, agg, rng):
    """Accuracy tests from published counts; the bootstrap draws are replayed so the random stream stays aligned."""
    entry = {k: agg[k] for k in ("items_sha256", "n", "like_for_like_n", "clef_truncated", "full", "like_for_like",
                                 "without_intermittent") if k in agg}
    entry["per_item_data"] = "not redistributable; counts only (results/decision13/restricted_aggregates.json)"
    entry["pairs"] = {}
    for x, y in combinations(MODELS, 2):
        src = agg["pairs"][f"{x}__{y}"]; pair = {}
        for name in ("full", "like_for_like"):
            b, c = src[name]["b_x_only"], src[name]["c_y_only"]
            pair[name] = {"b_x_only": b, "c_y_only": c, "p": mcnemar_exact(b, c)}
        for shape in resample_shapes(src["n_both_answered"]):
            rng.integers(0, src["n_both_answered"], size=shape)
        pair["calibration_both_answered"] = {"n": src["n_both_answered"], "not_recomputable": "needs per-item probabilities"}
        entry["pairs"][f"{x}__{y}"] = pair
    return entry


def main():
    restricted = json.loads((RES / "restricted_aggregates.json").read_text())["block_e"]
    rng = np.random.default_rng(SEED)
    res = {"protocol": "provenance/decision13_protocol.md (amendments 1-9)", "seed": SEED, "bootstrap_resamples": B,
           "sets": {}}
    tests_acc, tests_cal = [], []                                 # (set, a, b, p, better)
    for s in DECISION_SETS + CONTROL_SETS:
        if s in restricted:
            entry = restricted_entry(s, restricted[s], rng)
            for x, y in combinations(MODELS, 2):
                f = entry["pairs"][f"{x}__{y}"]["full"]
                tests_acc.append([s, x, y, f["p"], x if f["b_x_only"] > f["c_y_only"] else y])
            res["sets"][s] = entry; print(s, "done (counts only)", flush=True)
            continue
        R, files = {}, {}
        for m in MODELS:
            R[m], files[m] = load(s, m)
        ids = sorted(R["pplx_v11"])
        assert all(sorted(R[m]) == ids for m in MODELS), f"{s}: item ids differ across models"
        lfl = [i for i in ids if not R["pplx_v11"][i]["unanswered"]
               and all((R[m][i].get("input_tokens") or 0) < CLEF_MAX for m in ("clef_flash", "clef"))]
        entry = {"files": files, "n": len(ids), "like_for_like_n": len(lfl),
                 "clef_truncated": {m: sum((R[m][i].get("input_tokens") or 0) >= CLEF_MAX for i in ids) for m in ("clef_flash", "clef")},
                 "full": {m: metrics([R[m][i] for i in ids]) for m in MODELS},
                 "like_for_like": {m: metrics([R[m][i] for i in lfl]) for m in MODELS}, "pairs": {}}
        for m in MODELS:
            check_against_summary(s, m, entry["full"][m])
        groups = {r.get("group") for r in R["pplx_v11"].values()} - {None}
        if groups:
            entry["by_group"] = {g: {m: metrics([R[m][i] for i in ids if R[m][i].get("group") == g]) for m in MODELS} for g in sorted(groups)}
        for x, y in combinations(MODELS, 2):
            pair = {}
            for name, sub in (("full", ids), ("like_for_like", lfl)):
                b = sum(R[x][i]["correct"] and not R[y][i]["correct"] for i in sub)
                c = sum(R[y][i]["correct"] and not R[x][i]["correct"] for i in sub)
                pair[name] = {"b_x_only": b, "c_y_only": c, "p": mcnemar_exact(b, c)}
            both = [i for i in ids if not R[x][i]["unanswered"] and not R[y][i]["unanswered"]]
            pair["calibration_both_answered"] = {"n": len(both), **boot_calib([R[x][i] for i in both], [R[y][i] for i in both], rng)}
            entry["pairs"][f"{x}__{y}"] = pair
            if s in DECISION_SETS:
                f = pair["full"]; tests_acc.append([s, x, y, f["p"], x if f["b_x_only"] > f["c_y_only"] else y])
                e = pair["calibration_both_answered"]["ece10"]; tests_cal.append([s, x, y, e["p_boot"], x if e["diff"] < 0 else y])
        res["sets"][s] = entry
        print(s, "done", flush=True)

    for tests, key in ((tests_acc, "accuracy"), (tests_cal, "ece10")):
        for t, p_adj in zip(tests, holm([t[3] for t in tests])):
            t.append(p_adj)
        res[f"holm_{key}"] = [dict(zip(["set", "a", "b", "p", "better", "p_holm"], t)) for t in tests]
    res["holm_ece10_family"] = "redistributable decision sets only (DBRD calibration needs per-item probabilities)"

    def score(tests, drop=()):
        sc = {m: 0 for m in MODELS}
        for s, x, y, p, better, p_holm in tests:
            if s in drop or p_holm >= 0.05:
                continue
            worse = y if better == x else x; sc[better] += 1; sc[worse] -= 1
        return sc

    res["primary_score_block_e"] = score(tests_acc)
    res["primary_score_block_e_without_exposed"] = score(tests_acc, EXPOSED_ALL)
    res["ece_score_block_e_redistributable_sets"] = score(tests_cal)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "analysis_block_e.json").write_text(json.dumps(res, indent=1))
    print("primary (block E):", res["primary_score_block_e"], "| without exposed:", res["primary_score_block_e_without_exposed"],
          "| ECE wins-losses (redistributable sets):", res["ece_score_block_e_redistributable_sets"])


if __name__ == "__main__":
    main()
