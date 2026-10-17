#!/usr/bin/env python3
"""Zero-shot comparison and jury-appendix numbers for the paper (claims ledger Z4b, Z6, Z7, Z8, Z9, K5, L2, M1).

Writes paper/generated/zeroshot_compare.json and paper/generated/numbers_zsx.tex (macros \\zx...). Accuracies are
recomputed from the per-item dumps; calibration-temperature and combiner values are read from the JSON outputs of the
committed scripts calibration_confirm.py and combiner_ablation.py.

    python paper/scripts/numbers_zeroshot_compare.py
"""
from __future__ import annotations
import gzip, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum.metrics import mcnemar_exact_p  # noqa: E402

PRED = ROOT / "results" / "predictions"
GEN = ROOT / "paper" / "generated"
DATASETS = [("banking77", "Banking"), ("clinc150", "Clinc"), ("hwu64", "Hwu"), ("massive", "Massive"),
            ("mtop", "Mtop"), ("snips", "Snips"), ("bitext", "Bitext")]
CONDS = [("names", "Name"), ("descriptions", "Desc")]
VARIANTS = {"Orig": "", "CleanNli": "_r13_clean_c", "SnliNli": "_r13_mnli_snli", "Pplx": "_r15_pplx_decider_zs_sgl",
            "ClefBig": "_r14_clef_zs"}


def load(ds, cond, suffix):
    f = PRED / f"{ds}_{cond}{suffix}.jsonl.gz"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in gzip.open(f, "rt")]
    assert [r["idx"] for r in rows] == list(range(len(rows))), f
    return np.array([r["gold"] for r in rows]), np.array([r["pred"] for r in rows]), rows


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def main():
    out, mac = {}, {}

    def put(k, v, m=None):
        out[k] = v
        if m is not None:
            mac[k] = m

    for ds, D in DATASETS:
        for cond, C in CONDS:
            gold = None
            for V, suf in VARIANTS.items():
                r = load(ds, cond, suf)
                if r is None:
                    continue
                y, p, _ = r
                if gold is None:
                    gold = y
                assert len(y) == len(gold) and (y == gold).all(), f"{ds}/{cond}/{V}: gold differs"
                put(f"{D}{C}{V}Acc", float((p == y).mean()), pct((p == y).mean()))

    # zero-shot Jev accuracy (one decimal, for the table), from its per-item files; a failed item counts as wrong
    for ds, D in DATASETS:
        for cond, C in CONDS:
            f = ROOT / "docs/phases/2.0/jev/zeroshot" / f"{ds}_{cond}.jsonl.gz"
            if f.exists():
                rows = [json.loads(l) for l in gzip.open(f, "rt")]
                ref = load(ds, cond, "")[0]
                assert len(rows) == len(ref), f
                acc = sum(bool(r.get("correct")) for r in rows) / len(ref)
                put(f"{D}{C}JevAcc", acc, pct(acc))

    # M1: smoke-test slice (first 100 test rows of SNIPS) vs the full test set, original ensemble
    for cond, C in CONDS:
        y, p, _ = load("snips", cond, "")
        put(f"SmokeSnips{C}Acc", float((p[:100] == y[:100]).mean()), pct((p[:100] == y[:100]).mean()))
        put(f"SmokeSnips{C}FullAcc", float((p == y).mean()), pct((p == y).mean()))
        put("SmokeSnipsClasses", int(len(set(y[:100].tolist()))), str(len(set(y[:100].tolist()))))
        put("SnipsClasses", int(len(set(y.tolist()))), str(len(set(y.tolist()))))

    # Z4b / Z9: one train-side global temperature (pre-declared, run once per ensemble)
    for tag, V in [("", "Orig"), ("_clean_c", "CleanNli"), ("_mnli_snli", "SnliNli")]:
        cc = json.loads((GEN / f"calibration_confirm{tag}.json").read_text())
        put(f"Tstar{V}", cc["T_star"], f"{cc['T_star']:.3f}")
        tests = cc["tests"]; n_better = sum(t["T_star"]["ece_10_bins"] < t["raw"]["ece_10_bins"] for t in tests.values())
        put(f"Tstar{V}Better", n_better, str(n_better)); put(f"Tstar{V}Of", len(tests), str(len(tests)))
        if V == "Orig":
            under = sum(t["raw"]["mean_confidence"] < t["raw"]["accuracy"] for t in tests.values())
            put("RawUnderconfident", under, str(under))
            eces = [t["raw"]["ece_10_bins"] for t in tests.values()]
            put("RawEceMin", min(eces), f"{min(eces):.2f}"); put("RawEceMax", max(eces), f"{max(eces):.2f}")
            nov = sum(t["ece_ci95_T_star"][1] < t["ece_ci95_raw"][0] for t in tests.values())
            put("TstarCiDisjoint", nov, str(nov))
            for key, D, C in [("banking77/names", "Banking", "Name"), ("clinc150/names", "Clinc", "Name"),
                              ("mtop/names", "Mtop", "Name"), ("snips/descriptions", "Snips", "Desc")]:
                t = tests[key]
                put(f"{D}{C}RawEce", t["raw"]["ece_10_bins"], f"{t['raw']['ece_10_bins']:.3f}")
                put(f"{D}{C}TstarEce", t["T_star"]["ece_10_bins"], f"{t['T_star']['ece_10_bins']:.3f}")
                put(f"{D}{C}RawConf", t["raw"]["mean_confidence"], f"{t['raw']['mean_confidence']:.2f}")
                put(f"{D}{C}RawAcc", t["raw"]["accuracy"], f"{t['raw']['accuracy']:.2f}")

    # Z6 / Z7: combiner ablation (geometric vs arithmetic mean vs majority vote), from combiner_ablation.py output
    cb = json.loads((GEN / "combiners.json").read_text())
    put("CombinerDeclared", cb.get("_declared"))
    geo_gt_arith = geo_gt_vote = 0
    for ds, _ in DATASETS:
        for cond, _ in CONDS:
            a = cb[ds][cond]["acc"]
            geo_gt_arith += a["geo"] > a["arith"]; geo_gt_vote += a["geo"] > a["vote"]
    put("GeoGtArith", geo_gt_arith, str(geo_gt_arith)); put("GeoGtVote", geo_gt_vote, str(geo_gt_vote))
    sig = {k: 0 for k in ("GeoSigArith", "ArithSigGeo", "GeoSigVote", "VoteSigGeo")}   # family F3, Holm over 28
    for ds, _ in DATASETS:
        for cond, _ in CONDS:
            for test, a, b in (("geo_vs_arith", "GeoSigArith", "ArithSigGeo"), ("geo_vs_vote", "GeoSigVote", "VoteSigGeo")):
                t = cb[ds][cond]["tests"][test]
                if t["p_holm_F3"] < 0.05:
                    sig[a if t["b"] > t["c"] else b] += 1
    for k, v in sig.items():
        put(k, v, str(v))

    # Z8: clean-NLI jury vs its best member, paired exact McNemar, Holm over all 14 conditions; wins counted on the
    # conditions unexposed for every member (CLINC150, MTOP, SNIPS, Bitext) and overall
    res = []
    for ds, D in DATASETS:
        for cond, C in CONDS:
            y, p, rows = load(ds, cond, "_r13_clean_c")
            mem = np.array([r["member_picks"] for r in rows]).T
            best = mem[int(np.argmax([(m == y).mean() for m in mem]))]
            a, b = p == y, best == y
            nb, nc = int((a & ~b).sum()), int((~a & b).sum())
            res.append((ds, nb, nc, mcnemar_exact_p(nb, nc)))
    m = len(res); order = np.argsort([r[3] for r in res]); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * res[i][3])); adj[i] = run
    unexp = {"clinc150", "mtop", "snips", "bitext"}
    wins = [(r[1] > r[2]) and ph < 0.05 for r, ph in zip(res, adj)]
    losses = [(r[1] < r[2]) and ph < 0.05 for r, ph in zip(res, adj)]
    put("CleanNliWins", sum(wins), str(sum(wins))); put("CleanNliLosses", sum(losses), str(sum(losses)))
    put("CleanNliUnexposedWins", sum(w for w, r in zip(wins, res) if r[0] in unexp), str(sum(w for w, r in zip(wins, res) if r[0] in unexp)))
    put("CleanNliUnexposedOf", sum(r[0] in unexp for r in res), str(sum(r[0] in unexp for r in res)))

    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "zeroshot_compare.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = ["% GENERATED by paper/scripts/numbers_zeroshot_compare.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\zx{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    (GEN / "numbers_zsx.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(mac)} macros")


if __name__ == "__main__":
    main()
