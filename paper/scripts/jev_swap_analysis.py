#!/usr/bin/env python3
"""Pre-declared analyses for the Jev retrieval-swap run (provenance/jev_retrieval_swap_protocol.md §"Pre-declared
analyses" + addendum). Written and committed BEFORE the run's results were available.

  1. PRIMARY: exact McNemar, clean run vs Arm B (Jev with the clean run's retrieved examples) — equal retrieval.
  2. Exact McNemar, Arm B vs Arm A — effect of the retrieval rule on Jev itself.
  3. Exact McNemar, Arm A vs the reproduction's published per-item correctness (results/predictions.csv @ 5cac4ff,
     fetched at run time) — run-to-run agreement of the same pinned model.
  4. Paired bootstrap (10,000 resamples, seed 20261011; ECE 2,000) of ours − Arm B: per-item log loss (clip 1e-15),
     Brier (summed over classes), ECE-10 — the definitions of paper/scripts/jev_paired_calibration.py.
Failure handling (declared here, before results): a failed call counts as WRONG for accuracy (protocol); it is
EXCLUDED from the calibration metrics of analysis 4 (no probabilities exist), and the number excluded is reported.

    python paper/scripts/jev_swap_analysis.py --repro-csv <path to the reproduction's predictions.csv>
"""
from __future__ import annotations
import argparse, csv, gzip, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum.metrics import calibration, mcnemar_exact_p  # noqa: E402

BOOT_N, ECE_BOOT_N, SEED = 10_000, 2_000, 20261011


def arm(name, names):
    rows = {}
    src = ROOT / "results" / "jev" / f"arm{name}.jsonl"          # committed gzipped (size); plain file if present
    text = src.read_text() if src.exists() else gzip.open(str(src) + ".gz", "rt").read()
    for l in text.splitlines():
        r = json.loads(l); rows[r["idx"]] = r
    assert sorted(rows) == list(range(3080)), f"arm {name}: expected 3,080 items, got {len(rows)}"
    idx = {n: i for i, n in enumerate(names)}
    choice = np.array([idx.get(rows[i].get("choice"), -1) for i in range(3080)])
    P = np.full((3080, 77), np.nan)
    for i in range(3080):
        pr = rows[i].get("probabilities")
        if pr:
            P[i] = 0.0
            for k, v in pr.items():
                P[i, idx[k]] = v
    failed = np.array([bool(rows[i].get("failed")) for i in range(3080)])
    return choice, P, failed


def mc(a_ok, b_ok):
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return {"b": b, "c": c, "p": mcnemar_exact_p(b, c), "acc_a": float(a_ok.mean()), "acc_b": float(b_ok.mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repro-csv", required=True); a = ap.parse_args()
    from datasets import load_dataset
    names = load_dataset("legacy-datasets/banking77", revision="f54121560de48f2852f90be299010d1d6dc612ec",
                         split="test").features["label"].names
    rows = [json.loads(l) for l in gzip.open(ROOT / "results/predictions/banking77_24shot_clean.jsonl.gz", "rt")]
    y = np.array([r["gold"] for r in rows]); ours_ok = np.array([r["pred"] for r in rows]) == y
    Po = np.array([r["probs"] for r in rows], float)
    cA, PA, fA = arm("A", names); cB, PB, fB = arm("B", names)
    okA, okB = cA == y, cB == y                                    # failed / unknown choice → wrong
    rep = {int(r["id"].split("-")[1]): r for r in csv.DictReader(open(a.repro_csv))}
    assert all(rep[i]["truth"] == names[y[i]] for i in range(3080))
    pub_ok = np.array([rep[i]["correct"] == "True" for i in range(3080)])
    out = {"n": 3080, "failed": {"A": int(fA.sum()), "B": int(fB.sum())},
           "kept_examples_B": None,
           "1_primary_clean_vs_armB": mc(ours_ok, okB),
           "2_armB_vs_armA": mc(okB, okA),
           "3_armA_vs_published": mc(okA, pub_ok),
           "agreement_armA_vs_published_choice": float(np.mean([names[cA[i]] == rep[i]["prediction"] if cA[i] >= 0 else False
                                                                for i in range(3080)]))}
    keep = ~fB & ~np.isnan(PB).any(1)
    yy, Pk, Ok = y[keep], PB[keep], Po[keep]
    ll = lambda P: -np.log(np.maximum(P[np.arange(len(yy)), yy], 1e-15))
    br = lambda P: ((P - np.eye(77)[yy]) ** 2).sum(1)
    d_ll, d_br = ll(Ok) - ll(Pk), br(Ok) - br(Pk)
    rng = np.random.default_rng(SEED); n = len(yy); boots = [rng.integers(0, n, n) for _ in range(BOOT_N)]
    ci = lambda d: [float(x) for x in np.percentile([d[i].mean() for i in boots], [2.5, 97.5])]
    ece = [calibration(Ok[i], yy[i])["ece_10_bins"] - calibration(Pk[i], yy[i])["ece_10_bins"] for i in boots[:ECE_BOOT_N]]
    out["4_calibration_ours_minus_armB"] = {
        "n_used": int(n), "excluded_failed": int((~keep).sum()),
        "ours": calibration(Ok, yy), "jev_armB": calibration(Pk, yy), "jev_armA": calibration(PA[~np.isnan(PA).any(1)], y[~np.isnan(PA).any(1)]),
        "diff_log_loss": float(d_ll.mean()), "ci_log_loss": ci(d_ll),
        "diff_brier": float(d_br.mean()), "ci_brier": ci(d_br),
        "diff_ece": calibration(Ok, yy)["ece_10_bins"] - calibration(Pk, yy)["ece_10_bins"],
        "ci_ece": [float(x) for x in np.percentile(ece, [2.5, 97.5])]}
    (ROOT / "paper" / "generated" / "jev_swap_analysis.json").write_text(json.dumps(out, indent=1))
    p = out["1_primary_clean_vs_armB"]; q = out["2_armB_vs_armA"]; r = out["3_armA_vs_published"]; c = out["4_calibration_ours_minus_armB"]
    print(f"failed calls: A {out['failed']['A']}, B {out['failed']['B']}")
    print(f"1 PRIMARY ours {p['acc_a']:.4f} vs Jev-B {p['acc_b']:.4f}: b={p['b']} c={p['c']} p={p['p']:.4g}")
    print(f"2 Jev-B {q['acc_a']:.4f} vs Jev-A {q['acc_b']:.4f}: b={q['b']} c={q['c']} p={q['p']:.4g}")
    print(f"3 Jev-A {r['acc_a']:.4f} vs published {r['acc_b']:.4f}: b={r['b']} c={r['c']} p={r['p']:.4g}; choice agreement {out['agreement_armA_vs_published_choice']:.4f}")
    print(f"4 ours−JevB: Δlogloss {c['diff_log_loss']:+.3f} {c['ci_log_loss']} | Δbrier {c['diff_brier']:+.4f} {c['ci_brier']} | "
          f"ΔECE {c['diff_ece']:+.4f} {c['ci_ece']} (n={c['n_used']}, excluded {c['excluded_failed']})")


if __name__ == "__main__":
    main()
