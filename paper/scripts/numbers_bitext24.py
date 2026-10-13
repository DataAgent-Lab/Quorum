#!/usr/bin/env python3
"""Bitext 24-shot (R16) numbers for the paper, recomputed from the per-item dumps.

Writes paper/generated/bitext24.json (every value) and paper/generated/numbers_bitext24.tex (LaTeX macros \\bt...).
Claims ledger row B1.

Protocol: provenance/bitext_24shot_protocol.md (pre-registration 9d6dc44; amendment A1 3d84cc3; A3 pplx engine) and
docs/phases/2.0/bitext24/AMENDMENT_A2.md (main f4ac106). The pre-declared analysis is scripts/r16_analysis.py,
committed before the joint read and run once (results/r16/analysis.json). This script is an independent
recomputation: accuracy, every McNemar pair and Holm value, ECE and Brier per arm, and the point differences against
the open stack are recomputed here and asserted equal to analysis.json; the bootstrap intervals (one pre-declared
resampling run) are taken from analysis.json and checked against an independent bootstrap (tolerance 0.002).

Inputs
  results/predictions/bitext_24shot_clean.jsonl.gz          the open stack's single test run
  results/predictions/bitext_r16_{clef,clef_flash,pplx}.jsonl.gz
  docs/phases/2.0/jev/bitext24/bitext_r16_jev.jsonl.gz      Jev (pred = returned choice, A2.5)
  results/bitext24/request_manifest.jsonl.gz                 recorded retrieval (A1) and examples kept
  results/bitext24/selection_scores.json                     the open stack's 48 selection candidates
  docs/phases/2.0/bitext24/subset_a2.json                    A2 non-near-duplicate subset and duplicate audit
  the pinned Bitext dataset (via scripts/bitext24_bodies.py split()) for the BM25 top-1 floor row

    python paper/scripts/numbers_bitext24.py
"""
from __future__ import annotations
import gzip, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum.metrics import calibration, mcnemar_exact_p  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numbers_24shot import fp, holm, pct, num, signed, thousands  # noqa: E402

PRED = ROOT / "results" / "predictions"
GEN = ROOT / "paper" / "generated"
N, K = 5375, 27
# macro stem -> (analysis.json arm key, dump path)
ARMS = {"Open": ("open", PRED / "bitext_24shot_clean.jsonl.gz"),
        "Jev": ("jev", ROOT / "docs/phases/2.0/jev/bitext24/bitext_r16_jev.jsonl.gz"),
        "Clef": ("clef", PRED / "bitext_r16_clef.jsonl.gz"),
        "Flash": ("clef_flash", PRED / "bitext_r16_clef_flash.jsonl.gz"),
        "Pplx": ("pplx", PRED / "bitext_r16_pplx.jsonl.gz")}
KEY = {v[0]: k for k, v in ARMS.items()}


def load(m, path):
    rows = sorted((json.loads(l) for l in gzip.open(path, "rt")), key=lambda r: r["idx"])
    assert [r["idx"] for r in rows] == list(range(N)), m
    assert not any(r.get("unanswered") or r.get("failed") or r.get("unsent") for r in rows), f"{m}: forced-wrong items"
    y = np.array([r["gold"] for r in rows]); pred = np.array([r["pred"] for r in rows])
    P = np.array([r["probs"] for r in rows], float)
    assert P.shape == (N, K) and np.abs(P.sum(1) - 1).max() < 1e-3, m
    return y, pred, P, rows


def ece(P, hit):
    """ECE-10 on the probability of the predicted intent (equal-width bins; last bin closed)."""
    conf = P.max(1); e = 0.0
    for i in range(10):
        m = (conf >= i / 10) & ((conf < (i + 1) / 10) if i < 9 else (conf <= 1.0))
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def main():
    an = json.loads((ROOT / "results/r16/analysis.json").read_text())
    assert an["arms_missing"] == [] and sorted(an["arms_present"]) == sorted(KEY)
    sub = json.loads((ROOT / "docs/phases/2.0/bitext24/subset_a2.json").read_text())
    smask = np.zeros(N, bool); smask[sub["subset_idx"]] = True
    assert int(smask.sum()) == sub["n_subset"]

    pred, prob, hit, y = {}, {}, {}, None
    for m, (_, path) in ARMS.items():
        yy, p, P, rows = load(m, path)
        if y is None:
            y = yy
        assert (yy == y).all(), f"{m}: gold labels differ"
        if m == "Jev":                                      # A2.5: the prediction is the returned choice
            cne = int((p != P.argmax(1)).sum())
        else:
            assert (p == P.argmax(1)).all(), f"{m}: pred != argmax"
        pred[m], prob[m], hit[m] = p, P, p == y

    out, mac = {}, {}

    def put(key, val, macro=None):
        out[key] = val
        if macro is not None:
            mac[key] = macro

    names = list(ARMS)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    for fam, mask, tag, akey in [("primary", np.ones(N, bool), "", "primary"),
                                 ("subset", smask, "Sub", "secondary_subset_A2")]:
        A = an[akey]
        assert A["n_items"] == int(mask.sum())
        for m in names:
            h, P = hit[m][mask], prob[m][mask]
            acc, e = float(h.mean()), ece(P, h); br = float(((P - np.eye(K)[y[mask]]) ** 2).sum(1).mean())
            ref = A["per_arm"][ARMS[m][0]]
            assert ref["correct"] == int(h.sum()) and ref["ece10"] == round(e, 4) and ref["brier"] == round(br, 4), (fam, m)
            if m != "Jev" or cne == 0:                       # same ECE as the library definition when choice = argmax
                c = calibration(P, y[mask]); assert abs(c["ece_10_bins"] - e) < 1e-12 and abs(c["brier_score"] - br) < 1e-12
            put(f"{m}{tag}Acc", acc, pct(acc)); put(f"{m}{tag}Correct", int(h.sum()), thousands(int(h.sum())))
            put(f"{m}{tag}Err", int((~h).sum()), str(int((~h).sum())))
            put(f"{m}{tag}Ece", e, num(e, 4)); put(f"{m}{tag}Brier", br, num(br, 4))
        ps, recs = [], []
        for a_, b_ in pairs:
            ha, hb = hit[a_][mask], hit[b_][mask]
            nb, nc = int((ha & ~hb).sum()), int((~ha & hb).sum()); p = mcnemar_exact_p(nb, nc)
            ps.append(p); recs.append((a_, b_, nb, nc, p))
        for (a_, b_, nb, nc, p), ph in zip(recs, holm(ps)):
            ref = next(r for r in A["pairs"] if (r["a"], r["b"]) == (ARMS[a_][0], ARMS[b_][0]))
            assert (ref["a_only"], ref["b_only"]) == (nb, nc) and abs(ref["p"] - p) <= 1e-9 * max(p, 1e-300) + 1e-15
            assert abs(ref["p_holm"] - ph) <= 1e-9 * max(ph, 1e-300) + 1e-15, (fam, a_, b_)
            k = f"{a_}Vs{b_}{tag}"
            put(f"{k}B", nb, str(nb)); put(f"{k}C", nc, str(nc)); put(f"{k}P", p, fp(p)); put(f"{k}PHolm", ph, fp(ph))
        put(f"{tag or 'Full'}HolmSig", sum(h_ < 0.05 for h_ in holm(ps)), str(sum(h_ < 0.05 for h_ in holm(ps))))
        # calibration against the open stack: point differences recomputed; CIs from the pre-declared run, checked
        rng = np.random.default_rng(1)
        for m in names[1:]:
            ref = A["calibration_vs_open"][ARMS[m][0]]
            Po, Pa, yo = prob["Open"][mask], prob[m][mask], y[mask]
            bo, ba = ((Po - np.eye(K)[yo]) ** 2).sum(1), ((Pa - np.eye(K)[yo]) ** 2).sum(1)
            db = float(ba.mean() - bo.mean()); de = ece(Pa, hit[m][mask]) - ece(Po, hit["Open"][mask])
            assert round(db, 4) == ref["brier_diff_arm_minus_open"] and round(de, 4) == ref["ece_diff_arm_minus_open"], (fam, m)
            n = len(yo); ix = rng.integers(0, n, (4000, n))
            ci_b = np.percentile(ba[ix].mean(1) - bo[ix].mean(1), [2.5, 97.5])
            assert np.abs(ci_b - ref["brier_ci95"]).max() < 2e-3, (fam, m, ci_b, ref["brier_ci95"])
            put(f"{m}{tag}BrierDiff", db, signed(db, 4))
            put(f"{m}{tag}BrierDiffLo", ref["brier_ci95"][0], signed(ref["brier_ci95"][0], 4))
            put(f"{m}{tag}BrierDiffHi", ref["brier_ci95"][1], signed(ref["brier_ci95"][1], 4))
            put(f"{m}{tag}EceDiff", de, signed(de, 4))
            put(f"{m}{tag}EceDiffLo", ref["ece_ci95"][0], signed(ref["ece_ci95"][0], 4))
            put(f"{m}{tag}EceDiffHi", ref["ece_ci95"][1], signed(ref["ece_ci95"][1], 4))

    # A2.2 floor row: the label of the recorded BM25 top-1 example
    spec = importlib.util.spec_from_file_location("bb", ROOT / "scripts" / "bitext24_bodies.py")
    bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
    raw, _, T, S, P_, test = bb.split()
    man = [json.loads(l) for l in gzip.open(ROOT / "results/bitext24/request_manifest.jsonl.gz", "rt")]
    assert [m_["idx"] for m_ in man] == list(range(N)) and (np.array([t["label"] for t in test]) == y).all()
    top1 = np.array([T[m_["retrieved_t_idx"][0]]["label"] for m_ in man]) == y
    assert round(float(top1.mean()), 4) == an["floor_bm25_top1_A2"]["full"]
    assert round(float(top1[smask].mean()), 4) == an["floor_bm25_top1_A2"]["subset"]
    put("FloorAcc", float(top1.mean()), pct(float(top1.mean()))); put("FloorSubAcc", float(top1[smask].mean()), pct(float(top1[smask].mean())))
    opened = [json.loads(l)["retrieved_idx"] for l in gzip.open(ARMS["Open"][1], "rt")]
    assert all(o == m_["retrieved_t_idx"] for o, m_ in zip(opened, man)), "open stack did not use the recorded retrieval"
    kept = [m_["examples_kept"] for m_ in man]
    assert min(kept) == 24

    # dataset, split, audit, selection, run facts
    put("N", N, thousands(N)); put("K", K, str(K)); put("NSub", int(smask.sum()), thousands(int(smask.sum())))
    put("SubPct", smask.mean(), pct(float(smask.mean()), 0))
    put("TotalN", len(T) + N, thousands(len(T) + N)); put("TestPct", N / (len(T) + N), pct(N / (len(T) + N), 0))
    put("TrainN", len(T), thousands(len(T))); put("SelN", len(S), str(len(S))); put("PoolN", len(P_), thousands(len(P_)))
    put("ExactDup", sub["exact_duplicates"], str(sub["exact_duplicates"]))
    put("NearDup", sub["cos_ge_threshold"], thousands(sub["cos_ge_threshold"]))
    put("NearDupPct", sub["cos_ge_threshold"] / N, pct(sub["cos_ge_threshold"] / N, 0))
    meta = json.loads((PRED / "bitext_24shot_clean_meta.json").read_text())
    assert meta["pool_size"] == len(P_) and meta["n_test"] == N and meta["rerun"] is False
    sel = json.loads((ROOT / "results/bitext24/selection_scores.json").read_text())
    cands = sel["candidates"]; best = max(c["oof_correct"] for c in cands)
    put("Candidates", len(cands), str(len(cands)))
    put("SelTied", sum(c["oof_correct"] == best for c in cands), str(sum(c["oof_correct"] == best for c in cands)))
    put("SelBest", best, str(best)); put("Selected", meta["selected"])
    assert sel["selected"]["member_set"] == meta["selected"]["member_set"] == "M2"
    put("JevChoiceNeArgmax", cne, str(cne)); assert cne == an["jev_choice_ne_argmax"] and an["jev_unsent"] == 0
    toks = {}
    for m, stem in [("Clef", "clef"), ("Flash", "clef_flash"), ("Pplx", "pplx")]:
        s_ = json.loads((ROOT / f"results/r16/summary_bitext_r16_{stem}.json").read_text())["results"]
        assert s_["correct"] == out[f"{m}Correct"] and s_["unanswered"] == 0
        toks[m] = s_["prompt_tokens"]["max"]
    put("MaxPromptTokens", max(toks.values()), thousands(max(toks.values())))

    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "bitext24.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = ["% GENERATED by paper/scripts/numbers_bitext24.py from the per-item dumps -- do not edit by hand."]
    lines += [f"\\newcommand{{\\bt{k}}}{{{mac[k]}}}" for k in sorted(mac)]
    (GEN / "numbers_bitext24.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(mac)} macros")


if __name__ == "__main__":
    main()
