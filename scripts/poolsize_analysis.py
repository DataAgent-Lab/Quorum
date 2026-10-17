#!/usr/bin/env python3
"""Pool size (Banking77): the pre-declared analyses of `provenance/banking77_pool_size_protocol.md` §7.
Committed before any cell is run. Everything below is reported whatever its outcome.

    python scripts/poolsize_analysis.py      # -> results/poolsize/analysis.json

Cells: results/poolsize/predictions/<system>_<pool>.jsonl.gz, systems open / pplx / clef_flash (+ the descriptive
open_poolfit on the 10 small pools). A missing cell is listed as missing and never imputed.

  §7.1 primary   per pool, exact two-sided McNemar for the 3 pairs; Holm families by pool size:
                 full (3), pc24 (15), pc8 (15)
  §7.2           ranking per pool (ties = equal correct counts, their own outcome); per size: accuracy mean/min/max
                 across draws, draws whose order equals full, Holm-significant winners per pair
  §7.3           size effect per system and draw (pool − full), descriptive
  §7.4           ECE-10 (probability of the predicted label) and Brier (summed over 77), descriptive
  §7.5           pool-fit sensitivity: open_poolfit vs open per small pool, descriptive, in no Holm family
  §7.6           pool sizes, examples kept, unanswered / truncated, and the full cells vs the existing b24B / clean
                 cells (argmax agreement), as a consistency check
Unanswered items count as wrong and carry a uniform distribution for calibration.
"""
import gzip, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PS = ROOT / "results" / "poolsize"; PRED = PS / "predictions"
SYSTEMS = ("open", "pplx", "clef_flash")
PAIRS = [("open", "pplx"), ("open", "clef_flash"), ("pplx", "clef_flash")]
K, N = 77, 3080
EXISTING = {"open": ROOT / "results/predictions/banking77_24shot_clean.jsonl.gz",
            "pplx": ROOT / "results/predictions/banking77_r15_pplx_decider_b24B_sgl.jsonl.gz",
            "clef_flash": ROOT / "results/predictions/banking77_r14_clef_flash_b24B.jsonl.gz"}


def load(path):
    rows = sorted((json.loads(l) for l in gzip.open(path, "rt")), key=lambda r: r["idx"])
    assert [r["idx"] for r in rows] == list(range(N)), f"{path.name}: expected idx 0..{N - 1}"
    gold = np.array([r["gold"] for r in rows]); P = np.full((N, K), 1 / K); pred = np.full(N, -1)
    unans = np.array([bool(r.get("unanswered")) for r in rows])
    for i, r in enumerate(rows):
        if not unans[i]:
            P[i] = np.asarray(r["probs"], float); pred[i] = int(r["pred"]) if r.get("pred") is not None else int(np.argmax(P[i]))
    tok = [r.get("input_tokens") for r in rows]
    return {"gold": gold, "P": P, "pred": pred, "hit": (pred == gold) & ~unans, "unanswered": int(unans.sum()),
            "max_tokens": max(t for t in tok) if all(t is not None for t in tok) else None}


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


def brier(P, gold):
    return float(((P - np.eye(K)[gold]) ** 2).sum(1).mean())


def order_of(correct):
    """Rank order as a tuple of systems, best first, or 'tie' if any two systems have equal correct counts."""
    vals = [correct[s] for s in SYSTEMS]
    if len(set(vals)) < len(vals):
        return "tie"
    return tuple(sorted(SYSTEMS, key=lambda s: -correct[s]))


def main():
    pools = list(json.loads((PS / "pools.json").read_text())["pools"])
    meta = json.loads((PS / "manifest_meta.json").read_text())
    cells, missing = {}, []
    for s in SYSTEMS + ("open_poolfit",):
        for p in pools:
            if s == "open_poolfit" and p == "full":
                continue
            f = PRED / f"{s}_{p}.jsonl.gz"
            if f.exists():
                cells[(s, p)] = load(f)
            else:
                missing.append(f"{s}_{p}")
    golds = [c["gold"] for c in cells.values()]
    assert all((g == golds[0]).all() for g in golds), "cells disagree on gold labels"
    size_of = lambda p: "full" if p == "full" else p.split("_")[0]
    res = {"protocol": "provenance/banking77_pool_size_protocol.md", "missing_cells": missing, "per_pool": {},
           "families": {}, "ranking": {}, "size_effect": {}, "pool_fit": {}, "descriptive": {}}
    # §7.1 + §7.4 per pool
    fam = {"full": [], "pc24": [], "pc8": []}
    for p in pools:
        row = {"pool_rows": meta["pools"][p]["rows"], "systems": {}, "pairs": []}
        for s in SYSTEMS:
            if (s, p) in cells:
                c = cells[(s, p)]
                row["systems"][s] = {"accuracy": round(float(c["hit"].mean()), 4), "correct": int(c["hit"].sum()),
                                     "unanswered": c["unanswered"], "ece10": round(ece10(c["P"], c["hit"]), 4),
                                     "brier": round(brier(c["P"], c["gold"]), 4), "max_input_tokens": c["max_tokens"]}
        for x, y in PAIRS:
            if (x, p) in cells and (y, p) in cells:
                hx, hy = cells[(x, p)]["hit"], cells[(y, p)]["hit"]
                b, c_ = int((hx & ~hy).sum()), int((hy & ~hx).sum())
                pr = {"a": x, "b": y, "a_only": b, "b_only": c_, "p": mcnemar_exact(b, c_)}
                row["pairs"].append(pr); fam[size_of(p)].append(pr)
        if len(row["systems"]) == 3:
            row["order"] = order_of({s: row["systems"][s]["correct"] for s in SYSTEMS})
        res["per_pool"][p] = row
    for f, prs in fam.items():
        for pr, ph in zip(prs, holm([pr["p"] for pr in prs])):
            pr["p_holm"] = ph; pr["holm_family"] = f"{f} ({len(prs)} tests)"
        res["families"][f] = len(prs)
    # §7.2 ranking stability, §7.3 size effect
    full_order = res["per_pool"]["full"].get("order")
    for size in ("pc24", "pc8"):
        ps = [p for p in pools if size_of(p) == size]
        acc = {s: [res["per_pool"][p]["systems"][s]["accuracy"] for p in ps if s in res["per_pool"][p]["systems"]]
               for s in SYSTEMS}
        outcome = {"matches_full": 0, "differs_from_full": 0, "tie": 0, "incomplete": 0}
        for p in ps:
            o = res["per_pool"][p].get("order")
            if o is None:
                outcome["incomplete"] += 1
            elif o == "tie" or full_order in (None, "tie"):
                outcome["tie"] += 1
            else:
                outcome["matches_full" if o == full_order else "differs_from_full"] += 1
        winners = {f"{x}-{y}": {x: 0, y: 0, "none": 0} for x, y in PAIRS}
        for p in ps:
            for pr in res["per_pool"][p]["pairs"]:
                w = "none" if pr["p_holm"] >= 0.05 else (pr["a"] if pr["a_only"] > pr["b_only"] else pr["b"])
                winners[f"{pr['a']}-{pr['b']}"][w] += 1
        res["ranking"][size] = {"full_order": full_order, "outcomes": outcome, "holm_winners": winners,
                                "accuracy": {s: {"mean": round(float(np.mean(v)), 4), "min": min(v), "max": max(v)}
                                             for s, v in acc.items() if v}}
        eff = {}
        for s in SYSTEMS:
            if s in res["per_pool"]["full"]["systems"]:
                base = res["per_pool"]["full"]["systems"][s]["accuracy"]
                d = [round(a - base, 4) for a in acc[s]]
                if d:
                    eff[s] = {"per_draw": d, "mean": round(float(np.mean(d)), 4), "min": min(d), "max": max(d)}
        res["size_effect"][size] = eff
    # §7.5 pool-fit sensitivity
    for p in pools:
        if p != "full" and ("open_poolfit", p) in cells and ("open", p) in cells:
            a, u = cells[("open_poolfit", p)], cells[("open", p)]
            res["pool_fit"][p] = {"accuracy": round(float(a["hit"].mean()), 4), "ece10": round(ece10(a["P"], a["hit"]), 4),
                                  "brier": round(brier(a["P"], a["gold"]), 4),
                                  "minus_unchanged": {"accuracy": round(float(a["hit"].mean() - u["hit"].mean()), 4),
                                                      "ece10": round(ece10(a["P"], a["hit"]) - ece10(u["P"], u["hit"]), 4),
                                                      "brier": round(brier(a["P"], a["gold"]) - brier(u["P"], u["gold"]), 4)}}
    for size in ("pc24", "pc8"):
        d = [v["minus_unchanged"]["accuracy"] for p, v in res["pool_fit"].items() if size_of(p) == size]
        if d:
            res["pool_fit"][f"{size}_across_draws_accuracy_diff"] = {"mean": round(float(np.mean(d)), 4), "min": min(d), "max": max(d)}
    # §7.6 descriptive
    res["descriptive"]["pools"] = {p: {k: meta["pools"][p][k] for k in ("rows", "per_label_min", "per_label_max", "examples_kept")}
                                   for p in pools}
    cons = {}
    for s, f in EXISTING.items():
        if (s, "full") in cells and f.exists():
            old = load(f); new = cells[(s, "full")]
            assert (old["gold"] == new["gold"]).all(), f"{f.name}: gold differs from the full cell"
            cons[s] = {"existing_file": str(f.relative_to(ROOT)), "argmax_agreement": round(float((old["pred"] == new["pred"]).mean()), 4),
                       "existing_accuracy": round(float(old["hit"].mean()), 4), "full_accuracy": round(float(new["hit"].mean()), 4)}
    res["descriptive"]["full_vs_existing"] = cons
    (PS / "analysis.json").write_text(json.dumps(res, indent=1, default=list))
    print(json.dumps({p: {s: v["accuracy"] for s, v in r["systems"].items()} for p, r in res["per_pool"].items()}),
          "missing:", len(missing), flush=True)


if __name__ == "__main__":
    main()
