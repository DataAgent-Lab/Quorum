#!/usr/bin/env python3
"""Zero-shot numbers for the paper, computed ONLY from the per-item dumps in results/predictions/.

Writes paper/generated/zero_shot.json (every value) and paper/generated/numbers_zeroshot.tex (LaTeX macros —
the manuscript never types a zero-shot number by hand). Reuses quorum.metrics (accuracy, exact McNemar).

Families for multiple-comparison control (declared here, Holm within each family):
  F1 (C1, ensemble vs its best single member): 7 datasets x {names, descriptions} = 14 tests
  F2 (C2, descriptions vs names, ensemble):    7 tests
Bootstrap: 10,000 row resamples, seed 20261008, percentile 95% CI on accuracy.

    python paper/scripts/zero_shot_tables.py
"""
from __future__ import annotations
import gzip, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum.metrics import accuracy, mcnemar_exact_p  # noqa: E402

DUMPS = ROOT / "results" / "predictions"
OUT = ROOT / "paper" / "generated"
DATASETS = [("banking77", "Banking"), ("clinc150", "Clinc"), ("hwu64", "Hwu"), ("massive", "Massive"),
            ("mtop", "Mtop"), ("snips", "Snips"), ("bitext", "Bitext")]
CONDS = [("names", "Name"), ("descriptions", "Desc")]
MEMBER_SHORT = {"Jaehun/PrismNLI-0.4B": ("Nli", "PrismNLI-0.4B"), "BAAI/bge-large-en-v1.5": ("BgeL", "bge-large"),
                "BAAI/bge-base-en-v1.5": ("BgeB", "bge-base")}
BOOT_N, BOOT_SEED, ALPHA = 10_000, 20261008, 0.05


def load(ds, cond):
    rows = [json.loads(l) for l in gzip.open(DUMPS / f"{ds}_{cond}.jsonl.gz", "rt")]
    meta = json.loads((DUMPS / f"{ds}_meta.json").read_text())
    assert [r["idx"] for r in rows] == list(range(len(rows))), f"{ds}/{cond}: idx not 0..n-1"
    assert len(rows) == meta["n_test"], f"{ds}/{cond}: n mismatch"
    y = np.array([r["gold"] for r in rows]); ens = np.array([r["pred"] for r in rows])
    mem = np.array([r["member_picks"] for r in rows]).T                       # (3, n)
    probs = np.array([r["probs"] for r in rows])
    assert (probs.argmax(1) == ens).all(), f"{ds}/{cond}: pred != argmax(probs)"
    return meta, y, ens, mem


def discordant(pa, pb, y):
    a, b = pa == y, pb == y
    return int((a & ~b).sum()), int((~a & b).sum())


def boot_ci(hit, rng):
    n = len(hit); acc = np.empty(BOOT_N)
    for s in range(0, BOOT_N, 500):                                           # chunked: bounded memory
        idx = rng.integers(0, n, size=(min(500, BOOT_N - s), n))
        acc[s:s + len(idx)] = hit[idx].mean(1)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i])); adj[i] = running
    return adj.tolist()


def fmt_p(p):
    """LaTeX math-mode p-value: three decimals when >= 0.001, otherwise m x 10^e (never '0.0')."""
    if p >= 0.001:
        return f"{p:.3f}"
    e = math.floor(math.log10(p)); m = p / 10 ** e
    return f"{m:.1f}\\times10^{{{e}}}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    res, f1, f2 = {}, [], []
    for ds, _ in DATASETS:
        res[ds] = {}
        for cond, _ in CONDS:
            meta, y, ens, mem = load(ds, cond)
            members = meta["members"]
            accs = {m: accuracy(mem[i], y) for i, m in enumerate(members)}
            best = max(members, key=lambda m: accs[m])
            b, c = discordant(ens, mem[members.index(best)], y)
            lo, hi = boot_ci((ens == y).astype(float), rng)
            res[ds][cond] = {"n": len(y), "k": meta["n_labels"], "acc": accuracy(ens, y), "acc_ci95": [lo, hi],
                             "member_acc": accs, "best_member": best, "mc_vs_best": {"b": b, "c": c,
                             "p": mcnemar_exact_p(b, c)}, "meta_acc": meta[f"accuracy_{cond}"]}
            f1.append((ds, cond))
        _, y, en, _ = load(ds, "names"); _, y2, ed, _ = load(ds, "descriptions")
        assert (y == y2).all()
        b, c = discordant(ed, en, y)
        res[ds]["desc_vs_names"] = {"b": b, "c": c, "p": mcnemar_exact_p(b, c),
                                    "gain": accuracy(ed, y) - accuracy(en, y)}
        f2.append(ds)
    for (ds, cond), a in zip(f1, holm([res[d][c]["mc_vs_best"]["p"] for d, c in f1])):
        res[ds][cond]["mc_vs_best"]["p_holm_F1"] = a
    for ds, a in zip(f2, holm([res[d]["desc_vs_names"]["p"] for d in f2])):
        res[ds]["desc_vs_names"]["p_holm_F2"] = a
    res["_declared"] = {"families": {"F1": "ensemble vs best single member, 7 datasets x 2 label texts (14)",
                                     "F2": "descriptions vs names, ensemble, 7 datasets (7)"},
                        "correction": "Holm", "alpha": ALPHA, "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED},
                        "source": "results/predictions/*_{names,descriptions}.jsonl.gz + *_meta.json"}
    (OUT / "zero_shot.json").write_text(json.dumps(res, indent=1))

    L = ["% GENERATED by paper/scripts/zero_shot_tables.py from results/predictions/ -- do not edit by hand."]
    def mac(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    wins = {c: 0 for c, _ in CONDS}
    for ds, D in DATASETS:
        r = res[ds]; mac(f"zs{D}N", f"{r['names']['n']:,}"); mac(f"zs{D}K", r["names"]["k"])
        for cond, C in CONDS:
            x = r[cond]; mc = x["mc_vs_best"]; p = f"zs{D}{C}"
            mac(p + "Acc", pct(x["acc"])); mac(p + "AccLo", pct(x["acc_ci95"][0])); mac(p + "AccHi", pct(x["acc_ci95"][1]))
            for m, a in x["member_acc"].items():
                mac(p + "Acc" + MEMBER_SHORT[m][0], pct(a))
            mac(p + "Best", MEMBER_SHORT[x["best_member"]][1]); mac(p + "BestAcc", pct(x["member_acc"][x["best_member"]]))
            mac(p + "Delta", f"{100 * (x['acc'] - x['member_acc'][x['best_member']]):+.1f}")
            mac(p + "McB", mc["b"]); mac(p + "McC", mc["c"]); mac(p + "McP", fmt_p(mc["p"]))
            mac(p + "McPHolm", fmt_p(mc["p_holm_F1"]))
            sig = mc["p_holm_F1"] < ALPHA
            mac(p + "Verdict", ("win" if mc["b"] > mc["c"] else "loss") if sig else "n.s.")
            wins[cond] += int(sig and mc["b"] > mc["c"])
        dv = r["desc_vs_names"]; p = f"zs{D}DescVsName"
        mac(p + "Gain", f"{100 * dv['gain']:+.1f}"); mac(p + "B", dv["b"]); mac(p + "C", dv["c"])
        mac(p + "P", fmt_p(dv["p"])); mac(p + "PHolm", fmt_p(dv["p_holm_F2"]))
    mac("zsWinsNames", wins["names"]); mac("zsWinsDesc", wins["descriptions"]); mac("zsNumDatasets", len(DATASETS))
    (OUT / "numbers_zeroshot.tex").write_text("\n".join(L) + "\n")

    print(f"{'dataset':10s} {'cond':5s} {'n':>5s} {'ens':>6s}  95% CI        best member      {'best':>6s}  "
          f"b/c       p(exact)   p(Holm)  | desc-names  p(Holm)")
    for ds, _ in DATASETS:
        for cond, _ in CONDS:
            x = res[ds][cond]; mc = x["mc_vs_best"]
            extra = ""
            if cond == "descriptions":
                dv = res[ds]["desc_vs_names"]; extra = f"| {100*dv['gain']:+5.1f}  {dv['p_holm_F2']:.2e}"
            print(f"{ds:10s} {cond[:5]:5s} {x['n']:5d} {x['acc']:.4f} [{x['acc_ci95'][0]:.3f},{x['acc_ci95'][1]:.3f}] "
                  f"{MEMBER_SHORT[x['best_member']][1]:14s} {x['member_acc'][x['best_member']]:.4f}  "
                  f"{mc['b']:4d}/{mc['c']:<4d} {mc['p']:.2e}  {mc['p_holm_F1']:.2e} {extra}")
    print(f"significant ensemble wins after Holm (F1): names {wins['names']}/7, descriptions {wins['descriptions']}/7")


if __name__ == "__main__":
    main()
