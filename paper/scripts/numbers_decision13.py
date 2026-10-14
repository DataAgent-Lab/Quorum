#!/usr/bin/env python3
"""Numbers for the thirteen code and workflow decision tasks (provenance/decision13_protocol.md, blocks E and C).

Writes paper/generated/decision13.json and paper/generated/numbers_d13.tex (macros \\dt...). Accuracy tests are
recomputed here, independently of scripts/decision13/analyze_block_{e,c}.py:
  - block E per-item files (results/decision13/predictions/), with the amendment 4 length rule applied to pplx
    patchnet idx 3386 and the amendment 6 file for pplx equibench; TOSEM-DBRD from restricted_aggregates.json;
  - block C invoice_processing from the per-decision rows (labels-consensus; status scored or prediction_error, the
    latter wrong); the three unlicensed workflows from restricted_aggregates.json.
Exact McNemar per pair, Holm within each block, primary score = Holm-significant wins minus losses over E (no PAWS)
plus C. The ECE-10 tie-break score (24 tests) and the analysis' own scores are read from analysis_block_e.json and
checked against the recomputation. Serving-gate and length-limit values that exist only in the protocol text are
transcribed below with their section.

    python paper/scripts/numbers_decision13.py
"""
from __future__ import annotations
import gzip, itertools, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum.metrics import mcnemar_exact_p  # noqa: E402

D = ROOT / "results" / "decision13"
GEN = ROOT / "paper" / "generated"
MODELS = [("pplx_v11", "Pplx"), ("clef", "Clef"), ("clef_flash", "Flash")]
M = [m for m, _ in MODELS]
SETS_E = [("patchnet", "Patchnet"), ("dbrd", "Dbrd"), ("equibench", "Equibench"), ("poj104", "Poj"),
          ("primevul_pair", "PrimevulPair"), ("primevul_func", "PrimevulFunc"), ("svtrusteval", "Svtrust"),
          ("secvuleval", "Secvul"), ("juliet", "Juliet")]
CONTROL = ("paws", "Paws")
EXPOSED = {"patchnet", "secvuleval"}                                  # protocol §9: assumed exposed for all models
WORKFLOWS = [("invoice_processing", "Invoice"), ("customer_service", "Customer"),
             ("security_incidents", "Security"), ("agent_trace_observability", "Trace")]
# Transcribed from the protocol text (no machine-readable source in the release):
GATE = {"Items": 3080, "ClefMaxDp": 0.1045, "ClefAgree": 0.99773, "ClefItemsAbove": 1, "Threshold": 0.10,
        "FlashMaxDp": 0.039}                                          # §2a, amendment 1
LIMITS = {"PplxRelease": 8192, "PplxServed": 8186, "ClefRelease": 16384, "AffectedItems": 1,
          "AffectedTokens": 8189,                                     # amendments 4-6
          "ClefHosted": 65536, "FlashHosted": 24576}                  # vendor's hosted-service model pages, 2026-10-14


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def sgn(x):
    return f"{x:+d}" if x else "0"


def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def load_correct(st, m):
    f = "equibench_pplx_v11_amend6.jsonl.gz" if (st, m) == ("equibench", "pplx_v11") else f"{st}_{m}.jsonl.gz"
    rows = [json.loads(l) for l in gzip.open(D / "predictions" / f, "rt")]
    assert [r["idx"] for r in rows] == list(range(len(rows))), f
    ok = np.array([bool(r["correct"]) and not r["unanswered"] for r in rows])
    unans = np.array([bool(r["unanswered"]) for r in rows])
    if (st, m) == ("patchnet", "pplx_v11"):                           # amendment 4: over the release limit
        ok[3386] = False; unans[3386] = True
    return ok, unans, rows


def main():
    out, mac = {}, {}

    def put(k, v, m=None):
        out[k] = v
        if m is not None:
            mac[k] = m

    ra = json.load(open(D / "restricted_aggregates.json"))
    ana = json.load(open(D / "analysis_block_e.json"))
    tests_e, cells = [], {}
    for st, S in SETS_E + [CONTROL]:
        if st == "dbrd":
            agg = ra["block_e"]["dbrd"]; n = agg["n"]
            acc = {m: agg["full"][m]["accuracy"] for m in M}
            lfl_n, lfl = agg["like_for_like_n"], {m: agg["like_for_like"][m]["accuracy"] for m in M}
            unans_p, trunc = 0, 0
            pairs = {(a, b): (agg["pairs"][f"{a}__{b}"]["full"]["b_x_only"], agg["pairs"][f"{a}__{b}"]["full"]["c_y_only"])
                     for a, b in itertools.combinations(["pplx_v11", "clef_flash", "clef"], 2)}
        else:
            cor, un = {}, {}
            for m in M:
                cor[m], un[m], _ = load_correct(st, m)
            n = len(cor["pplx_v11"]); acc = {m: float(cor[m].mean()) for m in M}
            info = ana["sets"][st]; trunc = info["clef_truncated"]["clef"]
            assert trunc == info["clef_truncated"]["clef_flash"]
            lfl_n, lfl = info["like_for_like_n"], {m: info["like_for_like"][m]["accuracy"] for m in M}
            unans_p = int(un["pplx_v11"].sum())
            assert not un["clef"].any() and not un["clef_flash"].any()
            pairs = {(a, b): (int((cor[a] & ~cor[b]).sum()), int((~cor[a] & cor[b]).sum()))
                     for a, b in itertools.combinations(["pplx_v11", "clef_flash", "clef"], 2)}
        for m, Mm in MODELS:
            put(f"{S}{Mm}Acc", acc[m], pct(acc[m])); put(f"{S}{Mm}LflAcc", lfl[m], pct(lfl[m]))
            assert abs(acc[m] - (ana["sets"][st]["full"][m]["accuracy"] if st != "dbrd" else acc[m])) < 1e-12
        put(f"{S}N", n, thousands(n)); put(f"{S}LflN", lfl_n, thousands(lfl_n))
        put(f"{S}PplxUnans", unans_p, thousands(unans_p)); put(f"{S}ClefTrunc", trunc, thousands(trunc))
        cells[st] = acc
        if st != CONTROL[0]:
            tests_e += [(st, a, b, bb, cc, mcnemar_exact_p(bb, cc)) for (a, b), (bb, cc) in pairs.items()]

    def score(tests, keep=lambda t: True):
        ts = [t for t in tests if keep(t)]; adj = holm([t[5] for t in ts]); s = {m: 0 for m in M}; sig = []
        for t, q in zip(ts, adj):
            if q < 0.05:
                w, l = (t[1], t[2]) if t[3] > t[4] else (t[2], t[1]); s[w] += 1; s[l] -= 1; sig.append((t[0], w, l))
        return s, sig

    sE, sigE = score(tests_e)
    sEx, _ = score(tests_e, lambda t: t[0] not in EXPOSED)
    assert sE == ana["primary_score_block_e"] and sEx == ana["primary_score_block_e_without_exposed"], (sE, sEx)

    # bold a cell when that model is Holm-significantly better than both others on the set
    for st, S in SETS_E:
        for m, Mm in MODELS:
            beats = {l for s_, w, l in sigE if s_ == st and w == m}
            v = pct(cells[st][m])
            put(f"{S}{Mm}Cell", None, f"\\textbf{{{v}}}" if len(beats) == 2 else v)
        put(f"{S}Exposed", st in EXPOSED, "$^\\times$" if st in EXPOSED else "")
    for m, Mm in MODELS:
        nb = sum(len({l for s_, w, l in sigE if s_ == st and w == m}) == 2 for st, _ in SETS_E)
        put(f"BestE{Mm}", nb, str(nb))

    # block C
    tests_c = []
    for w, W in WORKFLOWS:
        if w == "invoice_processing":
            rows = {m: {(r["case_id"], r["policy_id"]): r for r in map(json.loads, gzip.open(
                D / "workflowevals" / f"invoice_processing_{m}.jsonl.gz", "rt"))
                if r["label_set_id"] == "labels-consensus" and r["status"] in ("scored", "prediction_error")} for m in M}
            keys = sorted(rows["pplx_v11"]); assert all(set(rows[m]) == set(keys) for m in M)
            cor = {m: np.array([rows[m][k]["status"] == "scored" and bool(rows[m][k]["exact_actions"]) for k in keys]) for m in M}
            err = {m: sum(rows[m][k]["status"] == "prediction_error" for k in keys) for m in M}
            n = len(keys); acc = {m: float(cor[m].mean()) for m in M}
            pairs = {(a, b): (int((cor[a] & ~cor[b]).sum()), int((~cor[a] & cor[b]).sum()))
                     for a, b in itertools.combinations(["pplx_v11", "clef_flash", "clef"], 2)}
            alls = [k for k in keys if all(rows[m][k]["status"] == "scored" for m in M)]
            put("InvoiceAllScoredN", len(alls), str(len(alls)))
            for m, Mm in MODELS:
                a_ = float(np.mean([bool(rows[m][k]["exact_actions"]) for k in alls]))
                put(f"InvoiceAllScored{Mm}Acc", a_, pct(a_))
        else:
            agg = ra["block_c"][w]; n = agg["n"]
            acc = {m: agg["models"][m]["accuracy"] for m in M}; err = {m: agg["models"][m]["prediction_error"] for m in M}
            pairs = {(a, b): (agg["pairs"][f"{a}__{b}"]["b_x_only"], agg["pairs"][f"{a}__{b}"]["c_y_only"])
                     for a, b in itertools.combinations(["pplx_v11", "clef_flash", "clef"], 2)}
        put(f"{W}N", n, thousands(n))
        for m, Mm in MODELS:
            put(f"{W}{Mm}Acc", acc[m], pct(acc[m])); put(f"{W}{Mm}Err", err[m], thousands(err[m]))
        tests_c += [(w, a, b, bb, cc, mcnemar_exact_p(bb, cc)) for (a, b), (bb, cc) in pairs.items()]
    sC, sigC = score(tests_c)
    for w, W in WORKFLOWS:
        for m, Mm in MODELS:
            beats = {l for s_, w_, l in sigC if s_ == w and w_ == m}
            v = pct(out[f"{W}{Mm}Acc"])
            put(f"{W}{Mm}Cell", None, f"\\textbf{{{v}}}" if len(beats) == 2 else v)
    ac = json.load(open(D / "analysis_block_c.json"))
    assert sC == ac["score_block_c"], sC

    ece = ana["ece_score_block_e_redistributable_sets"]
    for m, Mm in MODELS:
        put(f"ScoreE{Mm}", sE[m], sgn(sE[m])); put(f"ScoreC{Mm}", sC[m], sgn(sC[m]))
        put(f"Score{Mm}", sE[m] + sC[m], sgn(sE[m] + sC[m]))
        put(f"ScoreNoExp{Mm}", sEx[m] + sC[m], sgn(sEx[m] + sC[m]))
        put(f"EceScore{Mm}", ece[m], sgn(ece[m]))
    put("TestsE", len(tests_e), str(len(tests_e))); put("TestsC", len(tests_c), str(len(tests_c)))
    put("EceTests", len(ana["holm_ece10"]), str(len(ana["holm_ece10"])))
    put("SigE", len(sigE), str(len(sigE))); put("SigC", len(sigC), str(len(sigC)))
    total = sum(out[f"{S}N"] for _, S in SETS_E) + sum(out[f"{W}N"] for _, W in WORKFLOWS)
    put("TotalItems", total, thousands(total)); put("Tasks", len(SETS_E) + len(WORKFLOWS), str(len(SETS_E) + len(WORKFLOWS)))

    conc = json.load(open(D / "concurrency_checks.json"))["per_model"]
    for m, Mm in MODELS:
        v = conc[m]; agree = min(x["argmax_agreement"] for x in v.values()); dp = max(x["max_abs_dp"] for x in v.values())
        put(f"Conc{Mm}Agree", agree, pct(agree)); put(f"Conc{Mm}MaxDp", dp, f"{dp:.4f}")
    put("ConcInFlight", 16, "16")
    put("GateItems", GATE["Items"], thousands(GATE["Items"])); put("GateClefMaxDp", GATE["ClefMaxDp"], f"{GATE['ClefMaxDp']:.4f}")
    put("GateClefAgree", GATE["ClefAgree"], pct(GATE["ClefAgree"], 2)); put("GateThreshold", GATE["Threshold"], f"{GATE['Threshold']:.2f}")
    put("GateFlashMaxDp", GATE["FlashMaxDp"], f"{GATE['FlashMaxDp']:.3f}"); put("GateClefItemsAbove", GATE["ClefItemsAbove"], str(GATE["ClefItemsAbove"]))
    for k, v in LIMITS.items():
        put(f"Limit{k}", v, thousands(v))

    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "decision13.json").write_text(json.dumps({k: v for k, v in out.items() if v is not None}, indent=1, default=float) + "\n")
    lines = ["% GENERATED by paper/scripts/numbers_decision13.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\dt{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    (GEN / "numbers_d13.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(mac)} macros; score E {sE}, C {sC}, E w/o exposed {sEx}")


if __name__ == "__main__":
    main()
