#!/usr/bin/env python3
"""Pre-declared analyses for the zero-shot Jev run (docs/phases/2.0/jev/zeroshot/PROTOCOL.md). Committed before
results. Failed calls count as wrong for accuracy and are excluded (and counted) for calibration.

    python paper/scripts/jev_zeroshot_analysis.py
"""
from __future__ import annotations
import gzip, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from quorum import data  # noqa: E402
from quorum.metrics import calibration, mcnemar_exact_p  # noqa: E402
from zero_shot_tables import holm  # noqa: E402

Z = ROOT / "docs" / "phases" / "2.0" / "jev" / "zeroshot"
GEN = ROOT / "paper" / "generated"
BOOT_N, ECE_BOOT_N, SEED = 10_000, 2_000, 20261012
PRIMARY = [("banking77", "names"), ("banking77", "descriptions"), ("mtop", "names"), ("mtop", "descriptions")]


def lines(path: Path):
    if path.exists():
        return path.read_text().splitlines()
    return gzip.open(str(path) + ".gz", "rt").read().splitlines()


def jev(ds, arm, keys):
    rows = {json.loads(l)["idx"]: json.loads(l) for l in lines(Z / f"{ds}_{arm}.jsonl")}
    n = len(data.load(ds)["test"]); assert sorted(rows) == list(range(n)), f"{ds}/{arm}: {len(rows)} of {n}"
    pos = {k: j for j, k in enumerate(keys)}; P = np.full((n, len(keys)), np.nan); ok = np.zeros(n, bool); failed = 0
    for i in range(n):
        r = rows[i]; ok[i] = bool(r.get("correct"))
        if r.get("failed") or not r.get("probabilities"):
            failed += 1; continue
        P[i] = 0.0
        for k, v in r["probabilities"].items():
            P[i, pos[k]] = v
    return ok, P, failed


def ours(ds, cond, tag):
    f = ROOT / "results" / "predictions" / (f"{ds}_{cond}_r13_{tag}.jsonl.gz" if tag else f"{ds}_{cond}.jsonl.gz")
    rows = [json.loads(l) for l in gzip.open(f, "rt")]
    y = np.array([r["gold"] for r in rows]); P = np.array([r["probs"] for r in rows], float)
    T = json.loads((GEN / (f"calibration_confirm_{tag}.json" if tag else "calibration_confirm.json")).read_text())["T_star"]
    L = np.log(np.clip(P, 1e-12, 1)) / T; L -= L.max(1, keepdims=True); Pc = np.exp(L); Pc /= Pc.sum(1, keepdims=True)
    return y, P.argmax(1) == y, Pc, T


def paired_cal(Po, Pj, y, rng):
    keep = ~np.isnan(Pj).any(1); yy, A, B = y[keep], Po[keep], Pj[keep]; n = len(yy)
    ll = lambda P: -np.log(np.maximum(P[np.arange(n), yy], 1e-15)); br = lambda P: ((P - np.eye(P.shape[1])[yy]) ** 2).sum(1)
    d_ll, d_br = ll(A) - ll(B), br(A) - br(B); boots = [rng.integers(0, n, n) for _ in range(BOOT_N)]
    ci = lambda d: [float(x) for x in np.percentile([d[i].mean() for i in boots], [2.5, 97.5])]
    ece = [calibration(A[i], yy[i])["ece_10_bins"] - calibration(B[i], yy[i])["ece_10_bins"] for i in boots[:ECE_BOOT_N]]
    return {"n_used": int(n), "excluded": int((~keep).sum()), "ours": calibration(A, yy), "jev": calibration(B, yy),
            "diff_log_loss": float(d_ll.mean()), "ci_log_loss": ci(d_ll), "diff_brier": float(d_br.mean()), "ci_brier": ci(d_br),
            "diff_ece": calibration(A, yy)["ece_10_bins"] - calibration(B, yy)["ece_10_bins"],
            "ci_ece": [float(x) for x in np.percentile(ece, [2.5, 97.5])]}


def main():
    rng = np.random.default_rng(SEED); out = {"primary": {}, "secondary": {}, "calibration": {}, "descriptive": {}}
    prim_p = []
    for ds, cond in PRIMARY:
        labs = data.load(ds)["labels"]
        okJ, PJ, failed = jev(ds, cond, labs)
        for tag, bucket in (("clean_c", "primary"), ("", "secondary")):
            y, okO, Pc, T = ours(ds, cond, tag)
            b, c = int((okO & ~okJ).sum()), int((~okO & okJ).sum())
            out[bucket][f"{ds}/{cond}"] = {"ours": "clean-NLI ensemble" if tag else "original ensemble (PrismNLI)",
                                           "acc_ours": float(okO.mean()), "acc_jev": float(okJ.mean()), "jev_failed": failed,
                                           "b_ours_right_jev_wrong": b, "c_ours_wrong_jev_right": c, "p": mcnemar_exact_p(b, c)}
            out["calibration"][f"{ds}/{cond}/{'clean_c' if tag else 'original'}"] = {"T_star": T, **paired_cal(Pc, PJ, y, rng)}
            if bucket == "primary":
                prim_p.append(out[bucket][f"{ds}/{cond}"]["p"])
    for (ds, cond), ph in zip(PRIMARY, holm(prim_p)):
        out["primary"][f"{ds}/{cond}"]["p_holm"] = ph
    d = data.load("banking77"); okR, PR, failed = jev("banking77", "repro_definitions", d["raw_labels"])
    y = np.array([r["label"] for r in d["test"]])
    out["descriptive"]["banking77/repro_definitions"] = {"acc": float(okR.mean()), "failed": failed,
        "calibration": calibration(PR[~np.isnan(PR).any(1)], y[~np.isnan(PR).any(1)]),
        "reference_154_item_screen": 0.7922}
    (GEN / "jev_zeroshot_analysis.json").write_text(json.dumps(out, indent=1))
    for k, v in out["primary"].items():
        print(f"PRIMARY {k:24s} clean-NLI {v['acc_ours']:.4f} vs Jev {v['acc_jev']:.4f} | b/c {v['b_ours_right_jev_wrong']}/{v['c_ours_wrong_jev_right']} p={v['p']:.3g} p_Holm={v['p_holm']:.3g}")
    for k, v in out["secondary"].items():
        print(f"second. {k:24s} original  {v['acc_ours']:.4f} vs Jev {v['acc_jev']:.4f} | b/c {v['b_ours_right_jev_wrong']}/{v['c_ours_wrong_jev_right']} p={v['p']:.3g}")
    for k, v in out["calibration"].items():
        print(f"calib   {k:34s} T*={v['T_star']:.3f} Δlogloss {v['diff_log_loss']:+.3f} {[round(x,3) for x in v['ci_log_loss']]} "
              f"Δbrier {v['diff_brier']:+.4f} {[round(x,4) for x in v['ci_brier']]} ΔECE {v['diff_ece']:+.4f} {[round(x,4) for x in v['ci_ece']]} "
              f"(Jev ECE {v['jev']['ece_10_bins']:.4f}, ours {v['ours']['ece_10_bins']:.4f})")
    r = out["descriptive"]["banking77/repro_definitions"]
    print(f"descr.  banking77/repro_definitions acc {r['acc']:.4f} (154-item screen 0.7922) ECE {r['calibration']['ece_10_bins']:.4f} failed {r['failed']}")


if __name__ == "__main__":
    main()
