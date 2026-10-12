#!/usr/bin/env python3
"""Banking77 24-shot and decision-model numbers for the paper, computed from the per-item dumps.

Writes paper/generated/fewshot.json (every value) and paper/generated/numbers_fewshot.tex (LaTeX macros — the
manuscript never types one of these numbers by hand). Claims ledger rows: F1, F3, F4, F7, F8, F11–F15, K1–K3, L1,
L2 (definitions), M2, M3 (accuracy parts), J1/J3 (from the pre-declared analysis JSON).

Inputs
  results/predictions/banking77_*.jsonl.gz        our runs (clean, default, R5, R6, R6b, R6c, R9, R10) and the
                                                  open decision models (R14 Clef incl. b24B, R15 pplx-decider)
  docs/phases/2.0/jev/24shot/arm{A,B}.jsonl.gz    Jev with the reproduction's (A) and our (B) retrieved examples
  docs/phases/2.0/jev/zeroshot/banking77_repro_definitions.jsonl.gz
  paper/generated/jev_swap_analysis.json          pre-declared analysis (F15), produced by jev_swap_analysis.py
  paper/generated/jev_paired_calibration.json     pre-declared analysis (F9), produced by jev_paired_calibration.py
  paper/generated/jev_zeroshot_analysis.json      pre-declared analysis (J1, J3), produced by jev_zeroshot_analysis.py
  --repro-csv                                     the reproduction's results/predictions.csv @ 5cac4ff (read at
                                                  run time, never committed)

Families (Holm within each; post hoc ones are labelled post hoc in the text):
  R5 (post hoc): clean vs each of the 6 ablation variants
  R6b (post hoc): the 6 factorial contrasts (cap, tokenisation, pool)
Bootstrap: 10,000 item resamples, seed 20261011 (accuracy CI; paired log-loss differences).

    python paper/scripts/numbers_24shot.py --repro-csv <path>
"""
from __future__ import annotations
import argparse, csv, gzip, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum.metrics import calibration, mcnemar_exact_p  # noqa: E402

PRED = ROOT / "results" / "predictions"
JEV = ROOT / "docs" / "phases" / "2.0" / "jev"
GEN = ROOT / "paper" / "generated"
N, K = 3080, 77
BOOT_N, SEED = 10_000, 20261011

# macro stem -> dump stem (results/predictions/banking77_<stem>.jsonl.gz)
OURS = {
    "Clean": "24shot_clean", "Default": "24shot_default",
    "ReaderAlone": "24shot_r5_reader_alone", "KnnAlone": "24shot_r5_knn_alone",
    "BaseKnn": "24shot_r5_base_nolora", "BaseReaderAlone": "24shot_r5_base_nolora_reader_alone",
    "VoteMaj": "24shot_r5_bm25vote_majority", "VoteWtd": "24shot_r5_bm25vote_weighted",
    "ParBest": "24shot_r6_best_parityret", "ParDefault": "24shot_r6_default_parityret",
    "ParBestDefs": "24shot_r6_best_parityret_reprodefs", "ParDefaultDefs": "24shot_r6_default_parityret_reprodefs",
    "CapUni": "24shot_r6b_cap4_unigram_P", "NocapBi": "24shot_r6b_nocap_bigram_P", "CapBi": "24shot_r6b_cap4_bigram_P",
    "Stable": "24shot_r6c_stable_tiebreak",
    "SeedB": "24shot_r9_best_seed2000", "SeedC": "24shot_r9_best_seed3000",
    "DefSeedB": "24shot_r9_default_seed2000", "DefSeedC": "24shot_r9_default_seed3000",
    "Ft": "finetune_lora",
    "Clef": "r14_clef_b24", "Flash": "r14_clef_flash_b24", "ClefB": "r14_clef_b24B", "FlashB": "r14_clef_flash_b24B",
    "Pplx": "r15_pplx_decider_b24", "PplxB": "r15_pplx_decider_b24B_sgl", "PplxDefs": "r15_pplx_decider_defs_sgl",
}
R5 = ["ReaderAlone", "KnnAlone", "BaseKnn", "BaseReaderAlone", "VoteMaj", "VoteWtd"]
R6B = [("CapEffUni", "Clean", "CapUni"), ("CapEffBi", "NocapBi", "CapBi"), ("BiEffNocap", "Clean", "NocapBi"),
       ("BiEffCap", "CapUni", "CapBi"), ("PoolEff", "CapBi", "ParBest"), ("BothEff", "Clean", "CapBi")]


def load_dump(stem):
    rows = [json.loads(l) for l in gzip.open(PRED / f"banking77_{stem}.jsonl.gz", "rt")]
    assert [r["idx"] for r in rows] == list(range(N)), stem
    return (np.array([r["gold"] for r in rows]), np.array([r["pred"] for r in rows]),
            np.array([r["probs"] for r in rows], float))


def load_jev(path, names):
    rows = {}
    for l in gzip.open(path, "rt"):
        r = json.loads(l); rows[r["idx"]] = r
    assert sorted(rows) == list(range(N)), path
    idx = {n: i for i, n in enumerate(names)}
    pred = np.array([idx.get(rows[i].get("choice"), -1) for i in range(N)])
    P = np.zeros((N, K))
    for i in range(N):
        for k, v in (rows[i].get("probabilities") or {}).items():
            P[i, idx[k]] = v
    assert not any(rows[i].get("failed") for i in range(N)), path
    return pred, P


def mc(pa, pb, y):
    a, b = pa == y, pb == y
    nb, nc = int((a & ~b).sum()), int((~a & b).sum())
    return {"b": nb, "c": nc, "p": mcnemar_exact_p(nb, nc)}


def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def wilson(k, n, z=1.959963984540054):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def item_logloss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1.0))


def paired_boot(d, rng):
    idx = rng.integers(0, len(d), size=(BOOT_N, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), *map(float, np.percentile(m, [2.5, 97.5]))


# ---------------------------------------------------------------- formatting
def fp(p):
    """p-value for math mode: 2 decimals >= 0.1; 3 significant digits in [0.001, 0.1); m x 10^e below."""
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.{2 - math.floor(math.log10(p))}f}"
    if p < 1e-300:
        return "10^{-300}"
    e = math.floor(math.log10(p)); m = p / 10 ** e
    if round(m, 1) >= 10:
        m, e = 1.0, e + 1
    return f"{m:.1f}\\times10^{{{e}}}"


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def num(x, d=3):
    return f"{x:.{d}f}"


def signed(x, d=3):
    return f"{x:+.{d}f}".replace("-", "-")


def thousands(n):
    return f"{n:,}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repro-csv", required=True); a = ap.parse_args()
    from datasets import load_dataset
    names = load_dataset("legacy-datasets/banking77", revision="f54121560de48f2852f90be299010d1d6dc612ec",
                         split="test").features["label"].names
    idx = {n: i for i, n in enumerate(names)}

    sys_pred, sys_prob, y = {}, {}, None
    for m, stem in OURS.items():
        yy, p, P = load_dump(stem)
        if y is None:
            y = yy
        assert (yy == y).all(), f"{stem}: gold labels differ"
        sys_pred[m], sys_prob[m] = p, P
    with open(a.repro_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == N and [int(r["id"].split("-")[1]) for r in rows] == list(range(N))
    assert all(idx[r["truth"]] == y[i] for i, r in enumerate(rows)), "reproduction truth != dataset gold"
    sys_pred["JevPub"] = np.array([idx.get(r["prediction"], -1) for r in rows])
    for arm in ("A", "B"):
        sys_pred[f"Jev{arm}"], sys_prob[f"Jev{arm}"] = load_jev(JEV / "24shot" / f"arm{arm}.jsonl.gz", names)
    sys_pred["JevDefs"], sys_prob["JevDefs"] = load_jev(JEV / "zeroshot" / "banking77_repro_definitions.jsonl.gz", names)

    out, mac = {}, {}

    def put(key, val, macro=None):
        out[key] = val
        if macro is not None:
            mac[key] = macro

    # accuracy for every system
    for m, p in sys_pred.items():
        k = int((p == y).sum())
        put(f"{m}Acc", k / N, pct(k / N)); put(f"{m}Correct", k, thousands(k))

    # calibration for every system with probabilities (ECE-10 equal-width, Brier summed over classes, log loss)
    for m, P in sys_prob.items():
        c = calibration(P, y)
        put(f"{m}Ece", c["ece_10_bins"], num(c["ece_10_bins"])); put(f"{m}Brier", c["brier_score"], num(c["brier_score"]))
        put(f"{m}LogLoss", c["log_loss_clipped_1e_15"], num(c["log_loss_clipped_1e_15"]))
        put(f"{m}MeanConf", c["mean_confidence"], num(c["mean_confidence"]))

    def contrast(key, a_, b_):
        r = mc(sys_pred[a_], sys_pred[b_], y)
        put(f"{key}B", r["b"], str(r["b"])); put(f"{key}C", r["c"], str(r["c"])); put(f"{key}P", r["p"], fp(r["p"]))
        return r

    # vs the published Jev reproduction (per-item correctness from its predictions.csv)
    for m in ["Clean", "Default", "Stable", "SeedB", "SeedC", "DefSeedB", "DefSeedC", "Ft", *R5,
              "ParBest", "ParDefault", "ParBestDefs", "ParDefaultDefs", "JevA"]:
        contrast(f"{m}VsJev", m, "JevPub")
    contrast("CleanVsDefault", "Clean", "Default")
    contrast("CleanVsParBest", "Clean", "ParBest")
    contrast("CleanVsFt", "Clean", "Ft")
    contrast("CleanVsStable", "Clean", "Stable")
    contrast("CleanVsSeedB", "Clean", "SeedB"); contrast("CleanVsSeedC", "Clean", "SeedC")
    # the retrieval swap (identical to the pre-declared F15 script; recomputed here as a control)
    contrast("CleanVsJevB", "Clean", "JevB"); contrast("JevBVsJevA", "JevB", "JevA")
    # open decision models on identical bodies
    contrast("FlashVsJevA", "Flash", "JevA"); contrast("ClefVsJevA", "Clef", "JevA"); contrast("PplxVsJevA", "Pplx", "JevA")
    contrast("FlashVsClean", "Flash", "Clean"); contrast("ClefVsClean", "Clef", "Clean")
    contrast("ParBestVsFlash", "ParBest", "Flash"); contrast("ParBestVsClef", "ParBest", "Clef")
    contrast("PplxVsParBest", "Pplx", "ParBest"); contrast("PplxVsClef", "Pplx", "Clef"); contrast("PplxVsFlash", "Pplx", "Flash")
    contrast("PplxBVsJevB", "PplxB", "JevB"); contrast("PplxBVsClean", "PplxB", "Clean"); contrast("PplxBVsPplx", "PplxB", "Pplx")
    contrast("PplxDefsVsJevDefs", "PplxDefs", "JevDefs")
    # Clef and Clef-Flash on our retrieval (Jev Arm-B bodies; R14 amendment A4)
    for m in ("ClefB", "FlashB"):
        contrast(f"{m}VsJevB", m, "JevB"); contrast(f"{m}VsClean", m, "Clean"); contrast(f"{m}Vs{m[:-1]}", m, m[:-1])
    contrast("FlashBVsClefB", "FlashB", "ClefB"); contrast("PplxBVsClefB", "PplxB", "ClefB")
    contrast("PplxBVsFlashB", "PplxB", "FlashB")

    # R5 family (post hoc): clean vs each ablation variant, Holm over 6
    r5 = [contrast(f"CleanVs{m}", "Clean", m) for m in R5]
    for m, ph in zip(R5, holm([r["p"] for r in r5])):
        put(f"CleanVs{m}PHolm", ph, fp(ph))
    # R6b factorial (post hoc), Holm over 6
    r6b = [contrast(key, a_, b_) for key, a_, b_ in R6B]
    for (key, _, _), ph in zip(R6B, holm([r["p"] for r in r6b])):
        put(f"{key}PHolm", ph, fp(ph))
    for m, stem in [("CapUni", OURS["CapUni"]), ("NocapBi", OURS["NocapBi"]), ("CapBi", OURS["CapBi"])]:
        meta = json.loads((PRED / f"banking77_{stem}_meta.json").read_text())
        put(f"{m}Classes", meta["retrieval_stats"]["mean_distinct_classes"],
            f"{meta['retrieval_stats']['mean_distinct_classes']:.1f}")
        if m == "CapUni":
            cs = meta["clean_retrieval_stats"]
            put("CleanClasses", cs["mean_distinct_classes"], f"{cs['mean_distinct_classes']:.1f}")
            put("CleanGoldAmong", cs["gold_class_among_24"], pct(cs["gold_class_among_24"], 1))
    pm = json.loads((PRED / f"banking77_{OURS['ParBest']}_meta.json").read_text())
    put("ParBestMeta", {k: pm[k] for k in pm if "retrieval" in k})

    # accuracy differences in percentage points (a - b)
    for key, a_, b_ in [("JevBMinusJevA", "JevB", "JevA"), ("PplxBMinusPplx", "PplxB", "Pplx"),
                        ("CleanMinusParBest", "Clean", "ParBest"), ("CleanMinusJevPub", "Clean", "JevPub"),
                        ("CleanMinusKnnAlone", "Clean", "KnnAlone"), ("CleanMinusBaseKnn", "Clean", "BaseKnn"),
                        ("CleanMinusJevB", "Clean", "JevB"), ("FlashMinusJevA", "Flash", "JevA"),
                        ("PplxBMinusClean", "PplxB", "Clean"), ("CleanMinusCapUni", "Clean", "CapUni"),
                        ("ClefBMinusClef", "ClefB", "Clef"), ("FlashBMinusFlash", "FlashB", "Flash")]:
        d = out[f"{a_}Acc"] - out[f"{b_}Acc"]; put(f"{key}Pt", d, f"{100 * d:.2f}")

    # dataset and pool facts (from the dumps' metadata and the dataset itself)
    put("N", N, thousands(N)); put("K", K, str(K))
    full = load_dataset("legacy-datasets/banking77", revision="f54121560de48f2852f90be299010d1d6dc612ec", split="train")
    put("FullTrain", len(full), thousands(len(full)))
    sel = json.loads((ROOT / "results/clean/selection_split_idx.json").read_text())
    put("SelN", sel["n"], thousands(sel["n"])); put("PoolN", len(full) - sel["n"], thousands(len(full) - sel["n"]))
    ftm = json.loads((PRED / "banking77_finetune_lora_meta.json").read_text())
    assert ftm["train_rows"] == len(full) - sel["n"]
    put("FtGpuHours", ftm["gpu_hours"]["train"], f"{ftm['gpu_hours']['train']:.1f}")
    cm = json.loads((PRED / "banking77_24shot_clean_meta.json").read_text())
    put("CleanMeta", {k: cm[k] for k in ("protocol_commit", "rerun", "dataset_revision", "base_revision", "adapter_revision")})

    # pplx-decider serving-engine parity gate (A7), as recorded in the engine-mode summary
    g = json.loads((ROOT / "results/r15/summary_pplx_decider_b24B_sgl.json").read_text())["engine"]["gate"]
    assert g["verdict"] == "PASS"
    put("PplxGateAgree", g["argmax_agreement"], pct(g["argmax_agreement"])); put("PplxGateMaxDp", g["max_abs_dp"], f"{g['max_abs_dp']:.2f}")

    # Jev call ledgers: number of scored calls, served model(s), failures
    for tag_, path in [("Fs", JEV / "24shot" / "ledger.jsonl.gz"), ("Zs", JEV / "zeroshot" / "ledger.jsonl.gz")]:
        led = [json.loads(l) for l in gzip.open(path, "rt")]
        ok = [r for r in led if r.get("ok")]
        if tag_ == "Fs":
            models = sorted({r.get("model") for r in ok})
        else:                                     # the zero-shot ledger has no model field; the item files do
            models = sorted({json.loads(l).get("model") for f in sorted((JEV / "zeroshot").glob("*.jsonl.gz"))
                             if not f.name.startswith(("raw_", "ledger", "request_manifest")) for l in gzip.open(f, "rt")})
        put(f"Jev{tag_}Calls", len(ok), thousands(len(ok))); put(f"Jev{tag_}Models", models)
        put(f"Jev{tag_}FailedAttempts", len(led) - len(ok), str(len(led) - len(ok)))
    assert out["JevFsModels"] == ["jev-1.13.0"] and out["JevZsModels"] == ["jev-1.13.0"]

    # headline extras: bootstrap accuracy CI, Wilson intervals (M2), changed predictions (R6c, R9)
    rng = np.random.default_rng(SEED)
    hit = (sys_pred["Clean"] == y).astype(float)
    boots = hit[rng.integers(0, N, size=(BOOT_N, N))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    put("CleanAccLo", lo, pct(lo)); put("CleanAccHi", hi, pct(hi))
    for m in ("Clean", "JevPub"):
        wl, wh = wilson(out[f"{m}Correct"], N)
        put(f"{m}WilsonLo", wl, pct(wl)); put(f"{m}WilsonHi", wh, pct(wh))
    def retrieved(stem):
        return [json.loads(l)["retrieved_idx"] for l in gzip.open(PRED / f"banking77_{stem}.jsonl.gz", "rt")]
    rc, rs = retrieved(OURS["Clean"]), retrieved(OURS["Stable"])
    sd = sum(set(a_) != set(b_) for a_, b_ in zip(rc, rs)); od = sum(a_ != b_ and set(a_) == set(b_) for a_, b_ in zip(rc, rs))
    put("StableSetDiff", sd, thousands(sd)); put("StableSetDiffPct", sd / N, pct(sd / N, 0))
    put("StableOrderOnlyDiff", od, thousands(od))
    for m in ("Stable", "SeedB", "SeedC"):
        d = int((sys_pred[m] != sys_pred["Clean"]).sum()); put(f"{m}Changed", d, str(d))

    # paired log-loss differences for the open models (unrounded probabilities; descriptive), ours - model
    ll = {m: item_logloss(sys_prob[m], y) for m in ("Clean", "ParBest", "Flash", "Clef", "Pplx", "PplxB", "ClefB", "FlashB")}
    for key, a_, b_ in [("LlCleanMinusFlash", "Clean", "Flash"), ("LlCleanMinusClef", "Clean", "Clef"),
                        ("LlParMinusFlash", "ParBest", "Flash"), ("LlParMinusClef", "ParBest", "Clef"),
                        ("LlCleanMinusPplxB", "Clean", "PplxB"), ("LlCleanMinusClefB", "Clean", "ClefB"),
                        ("LlCleanMinusFlashB", "Clean", "FlashB")]:
        d, l_, h_ = paired_boot(ll[a_] - ll[b_], rng)
        put(key, d, signed(d)); put(f"{key}Lo", l_, signed(l_)); put(f"{key}Hi", h_, signed(h_))

    # pre-declared analyses (values taken from the committed scripts' JSON outputs)
    sw = json.loads((GEN / "jev_swap_analysis.json").read_text())
    assert sw["1_primary_clean_vs_armB"]["b"] == out["CleanVsJevBB"] and sw["1_primary_clean_vs_armB"]["c"] == out["CleanVsJevBC"]
    assert sw["2_armB_vs_armA"]["b"] == out["JevBVsJevAB"] and sw["3_armA_vs_published"]["b"] == out["JevAVsJevB"]
    cal = sw["4_calibration_ours_minus_armB"]
    for k_, name in [("log_loss", "LogLoss"), ("brier", "Brier"), ("ece", "Ece")]:
        put(f"SwapDiff{name}", cal[f"diff_{k_}"], signed(cal[f"diff_{k_}"]))
        put(f"SwapDiff{name}Lo", cal[f"ci_{k_}"][0], signed(cal[f"ci_{k_}"][0]))
        put(f"SwapDiff{name}Hi", cal[f"ci_{k_}"][1], signed(cal[f"ci_{k_}"][1]))
    put("JevAAgreement", sw["agreement_armA_vs_published_choice"], pct(sw["agreement_armA_vs_published_choice"]))
    pc = json.loads((GEN / "jev_paired_calibration.json").read_text())
    ctl = pc["control"]["jev_recomputed"]
    put("JevPubEce", ctl["ece_10_bins"], num(ctl["ece_10_bins"])); put("JevPubBrier", ctl["brier_score"], num(ctl["brier_score"]))
    put("JevPubLogLoss", ctl["log_loss_clipped_1e_15"], num(ctl["log_loss_clipped_1e_15"]))
    r0 = pc["results"]["as_published/clean"]
    for k_, name in [("brier", "Brier"), ("ece", "Ece")]:
        put(f"PubDiff{name}", r0[f"diff_{k_}"], signed(r0[f"diff_{k_}"]))
        put(f"PubDiff{name}Lo", r0[f"ci_{k_}"][0], signed(r0[f"ci_{k_}"][0]))
        put(f"PubDiff{name}Hi", r0[f"ci_{k_}"][1], signed(r0[f"ci_{k_}"][1]))

    # J2: share of Jev Arm-B errors whose gold label got probability exactly 0, and the floored log loss
    PB = sys_prob["JevB"]; gold_p = PB[np.arange(N), y]; wrong = sys_pred["JevB"] != y
    z = int((gold_p == 0).sum()); put("JevBGoldZero", z, str(z)); put("JevBGoldZeroPct", z / N, pct(z / N, 1))
    put("JevBGoldZeroOfErrors", int(((gold_p == 0) & wrong).sum()) / int(wrong.sum()),
        pct(int(((gold_p == 0) & wrong).sum()) / int(wrong.sum()), 1))
    fl = np.maximum(PB, 1e-3); fl = fl / fl.sum(1, keepdims=True)
    d_fl = float((ll["Clean"] - item_logloss(fl, y)).mean())
    put("SwapDiffLogLossFloor", d_fl, signed(d_fl))

    # zero-shot Jev (J1, J3) from the pre-declared analysis
    zs = json.loads((GEN / "jev_zeroshot_analysis.json").read_text())
    tag = {"banking77": "Banking", "mtop": "Mtop", "clinc150": "Clinc", "snips": "Snips", "bitext": "Bitext"}
    for fam in ("primary", "family_E"):
        for key, r in zs[fam].items():
            ds, cond = key.split("/"); s = f"Zs{tag[ds]}{'Name' if cond == 'names' else 'Desc'}"
            put(f"{s}JevAcc", r["acc_jev"], pct(r["acc_jev"])); put(f"{s}CleanNliAcc", r["acc_ours"], pct(r["acc_ours"]))
            put(f"{s}JevPHolm", r["p_holm"], fp(r["p_holm"]))
    put("ZsDefsJevAcc", zs["descriptive"]["banking77/repro_definitions"]["acc"],
        pct(zs["descriptive"]["banking77/repro_definitions"]["acc"]))

    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "fewshot.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = ["% GENERATED by paper/scripts/numbers_24shot.py from the per-item dumps -- do not edit by hand."]
    for key in sorted(mac):
        lines.append(f"\\newcommand{{\\fs{key}}}{{{mac[key]}}}")
    (GEN / "numbers_fewshot.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(mac)} macros")


if __name__ == "__main__":
    main()
