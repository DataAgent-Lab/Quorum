#!/usr/bin/env python3
"""Banking77 pool-size numbers for the paper, recomputed from the per-item dumps.

Writes paper/generated/poolsize.json (every value) and paper/generated/numbers_poolsize.tex (LaTeX macros \\ps...).
Claims ledger row P1.

Protocol: provenance/banking77_pool_size_protocol.md (pre-registration d637927, review edits 7232258; code a4dad90,
docstring-only edit 1c3d3fd). The pre-declared analysis is scripts/poolsize_analysis.py, committed before any cell
was run and run once (results/poolsize/analysis.json, research 91a5dda). This script is an independent
recomputation, asserted equal to analysis.json:
  - the 11 pools, recomputed from the pinned train split, the clean selection split and the draw seeds;
  - every recorded retrieval, recomputed with a vectorised BM25 (the scores of quorum/fewshot.py, stable ties)
    written here, not the study's retriever; mismatches are allowed only inside exact score ties (counted);
  - file integrity (pools, manifests and every summary's manifest sha against manifest_meta.json);
  - accuracy, ECE-10, Brier, every McNemar pair and Holm value, rank orders, size effects and the pool-fit arm;
  - the full-pool cells against the existing b24B / clean cells, and the open stack's full cell against the
    stable-tie re-score of the clean run (same retrieval, same parameters).
The bodies themselves are checked by `scripts/poolsize_bodies.py verify` (needs the reproduction's text at run time).

    python paper/scripts/numbers_poolsize.py
"""
from __future__ import annotations
import gzip, hashlib, json, math, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quorum.metrics import calibration, mcnemar_exact_p  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numbers_24shot import fp, holm, pct, num, thousands  # noqa: E402

PS = ROOT / "results" / "poolsize"; PRED = PS / "predictions"; GEN = ROOT / "paper" / "generated"
N, K, SHOTS, SEED0, DRAWS = 3080, 77, 24, 20261020, 5
SYS = {"Open": "open", "Pplx": "pplx", "Flash": "clef_flash"}
PAIRS = [("Open", "Pplx"), ("Open", "Flash"), ("Pplx", "Flash")]
POOLS = ["full"] + [f"pc24_d{d}" for d in range(1, DRAWS + 1)] + [f"pc8_d{d}" for d in range(1, DRAWS + 1)]
SIZE = {"pc24": "TwentyFour", "pc8": "Eight"}
EXISTING = {"Open": "results/predictions/banking77_24shot_clean.jsonl.gz",
            "Pplx": "results/predictions/banking77_r15_pplx_decider_b24B_sgl.jsonl.gz",
            "Flash": "results/predictions/banking77_r14_clef_flash_b24B.jsonl.gz"}
STABLE = "results/predictions/banking77_24shot_r6c_stable_tiebreak.jsonl.gz"
TOK = re.compile(r"[a-z0-9]+")                                        # quorum/fewshot.py tokenizer


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows_of(path):
    rows = sorted((json.loads(l) for l in gzip.open(ROOT / path, "rt")), key=lambda r: r["idx"])
    assert [r["idx"] for r in rows] == list(range(N)), path
    return rows


def cell(path):
    rows = rows_of(path)
    assert not any(r.get("unanswered") or r.get("failed") for r in rows), f"{path}: forced-wrong items"
    y = np.array([r["gold"] for r in rows]); pred = np.array([r["pred"] for r in rows])
    P = np.array([r["probs"] for r in rows], float)
    assert P.shape == (N, K) and np.abs(P.sum(1) - 1).max() < 2e-3, path
    assert (pred == P.argmax(1)).all(), f"{path}: pred != argmax"
    return {"y": y, "pred": pred, "P": P, "hit": pred == y, "rows": rows}


def ece(P, hit):
    conf = P.max(1); e = 0.0
    for i in range(10):
        m = (conf >= i / 10) & ((conf < (i + 1) / 10) if i < 9 else (conf <= 1.0))
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def brier(P, y):
    return float(((P - np.eye(K)[y]) ** 2).sum(1).mean())


def draw_pools(labels, P):
    """Protocol §3, written from the text: per label in ascending id, permute that label's sorted P indices once
    with default_rng(20261020 + d); keep the first 24 and the first 8."""
    lab = labels[P]; out = {"full": list(P)}
    for d in range(1, DRAWS + 1):
        rng = np.random.default_rng(SEED0 + d); a, b = [], []
        for c in range(K):
            perm = rng.permutation(np.sort(P[lab == c]))
            a += perm[:24].tolist(); b += perm[:8].tolist()
        out[f"pc24_d{d}"], out[f"pc8_d{d}"] = sorted(a), sorted(b)
    return out


def bm25_scores(docs, queries, k1=1.5, b=0.75):
    """BM25 of quorum/fewshot.py as a sparse product: idf log(1 + (N - n + 0.5)/(n + 0.5)), per-term saturation."""
    toks = [TOK.findall(d.lower()) for d in docs]; vocab = {}
    for t in toks:
        for w in t:
            vocab.setdefault(w, len(vocab))
    r, c, v = [], [], []
    for i, t in enumerate(toks):
        for w, f in Counter(t).items():
            r.append(i); c.append(vocab[w]); v.append(f)
    tf = sparse.csr_matrix((v, (r, c)), shape=(len(docs), len(vocab)), dtype=float)
    dl = np.array([len(t) for t in toks], float); avg = dl.mean(); n = len(docs)
    df = np.bincount(c, minlength=len(vocab)); idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    W = tf.copy(); rows = np.repeat(np.arange(n), np.diff(tf.indptr))
    W.data = idf[W.indices] * W.data * (k1 + 1) / (W.data + k1 * (1 - b + b * dl[rows] / avg))
    qr, qc = [], []
    for i, q in enumerate(queries):
        for w in set(TOK.findall(q.lower())):
            if w in vocab:
                qr.append(i); qc.append(vocab[w])
    Q = sparse.csr_matrix((np.ones(len(qr)), (qr, qc)), shape=(len(queries), len(vocab)))
    return (Q @ W.T).toarray()


def main():
    an = json.loads((PS / "analysis.json").read_text())
    meta = json.loads((PS / "manifest_meta.json").read_text())
    pools_rec = json.loads((PS / "pools.json").read_text())["pools"]
    assert an["missing_cells"] == [] and list(pools_rec) == POOLS
    assert sha_file(PS / "pools.json") == meta["pools_sha256"]
    out, mac = {}, {}

    def put(key, val, macro=None):
        out[key] = val
        if macro is not None:
            mac[key] = macro

    # --- pools, from the protocol text
    pin = json.loads((ROOT / "docs/phases/2.0/jev/train_pinned.json").read_text())
    names = sorted(set(pin["label_name"]))
    from datasets import load_dataset
    ds_test = load_dataset("legacy-datasets/banking77", revision=pin["revision"], split="test")
    raw = list(ds_test.features["label"].names); assert len(raw) == K
    labels = np.array([raw.index(n_) for n_ in pin["label_name"]]); assert len(labels) == 10003 and len(names) == K
    S = set(json.loads((ROOT / "results/clean/selection_split_idx.json").read_text())["selection_idx"])
    P = np.array([i for i in range(len(labels)) if i not in S])
    pools = draw_pools(labels, P)
    assert all(pools[p] == pools_rec[p] for p in POOLS), "pools differ from pools.json"
    for d in range(1, DRAWS + 1):
        assert set(pools[f"pc8_d{d}"]) <= set(pools[f"pc24_d{d}"])
    for p in POOLS:
        assert meta["pools"][p]["rows"] == len(pools[p]) == an["per_pool"][p]["pool_rows"]
    put("PoolFull", len(P), thousands(len(P))); put("PoolTwentyFour", len(pools["pc24_d1"]), thousands(len(pools["pc24_d1"])))
    put("PoolEight", len(pools["pc8_d1"]), str(len(pools["pc8_d1"]))); put("Draws", DRAWS, str(DRAWS))
    small = [c_ for c_ in range(K) if (labels[P] == c_).sum() < 24]
    put("ShortLabels", len(small), str(len(small)))

    # --- recorded retrieval, recomputed
    test_text = [r["text"] for r in ds_test]; texts = pin["text"]
    tie_only, exact = 0, 0
    for p in POOLS:
        man_path = PS / f"manifest_{p}.jsonl.gz"
        assert sha_file(man_path) == meta["pools"][p]["manifest_sha256"], p
        man = [json.loads(l) for l in gzip.open(man_path, "rt")]
        pool = np.array(pools[p]); sc = bm25_scores([texts[i] for i in pool], test_text)
        for i, m in enumerate(man):
            mine = pool[np.argsort(-sc[i], kind="stable")[:SHOTS]].tolist()
            if mine == m["retrieved_t_idx"]:
                exact += 1; continue
            pos = {t: j for j, t in enumerate(pool)}                  # differ only by reordering equal scores?
            rec = np.array([sc[i, pos[t]] for t in m["retrieved_t_idx"]])
            assert (np.diff(rec) <= 1e-9).all(), (p, i, "recorded list not in score order")
            kth = np.sort(sc[i])[::-1][SHOTS - 1]
            assert np.allclose(np.sort(rec)[::-1], np.sort(sc[i])[::-1][:SHOTS], atol=1e-9) and rec.min() >= kth - 1e-9, (p, i)
            tie_only += 1
    put("RetrievalExact", exact, thousands(exact)); put("RetrievalTieOnly", tie_only, str(tie_only))
    put("RetrievalTotal", exact + tie_only, thousands(exact + tie_only))

    # --- cells and summaries
    C = {}
    for m, s in SYS.items():
        for p in POOLS:
            C[(m, p)] = cell(f"results/poolsize/predictions/{s}_{p}.jsonl.gz")
    # the pool-fit cells are left out of the anonymous supplement (size); without them §7.5 is skipped
    fit = all((ROOT / f"results/poolsize/predictions/open_poolfit_{p}.jsonl.gz").exists() for p in POOLS[1:])
    for p in POOLS[1:] if fit else []:
        C[("Fit", p)] = cell(f"results/poolsize/predictions/open_poolfit_{p}.jsonl.gz")
    if not fit:
        print("pool-fit cells not present: §7.5 (Fit macros) skipped")
    y = C[("Open", "full")]["y"]
    assert all((c_["y"] == y).all() for c_ in C.values())
    for (m, p), c_ in C.items():
        stem = "open_poolfit" if m == "Fit" else SYS[m]
        sm = json.loads((PS / f"summary_{stem}_{p}.json").read_text())
        assert sm["manifest_sha256"] == meta["pools"][p]["manifest_sha256"], (m, p)
        assert sm["results"]["correct"] == int(c_["hit"].sum()), (m, p)
        if m in ("Open", "Fit"):
            man = [json.loads(l)["retrieved_t_idx"] for l in gzip.open(PS / f"manifest_{p}.jsonl.gz", "rt")]
            assert [r["retrieved_idx"] for r in c_["rows"]] == man, (m, p, "retrieved_idx != manifest")
        else:
            toks = [r["input_tokens"] for r in c_["rows"]]
            out[f"MaxTok{m}{p}"] = max(toks)
    put("MaxPromptTokens", max(v for k, v in out.items() if k.startswith("MaxTok")),
        thousands(max(v for k, v in out.items() if k.startswith("MaxTok"))))

    # --- §7.1/§7.4 per pool, Holm by size
    fam = {"full": [], "pc24": [], "pc8": []}
    for p in POOLS:
        A = an["per_pool"][p]
        for m, s in SYS.items():
            c_ = C[(m, p)]; acc, e, br = float(c_["hit"].mean()), ece(c_["P"], c_["hit"]), brier(c_["P"], y)
            lib = calibration(c_["P"], y); assert abs(lib["ece_10_bins"] - e) < 1e-12 and abs(lib["brier_score"] - br) < 1e-12
            ref = A["systems"][s]
            assert ref["correct"] == int(c_["hit"].sum()) and ref["ece10"] == round(e, 4) and ref["brier"] == round(br, 4), (p, m)
            out[f"{m}{p}"] = {"acc": acc, "correct": int(c_["hit"].sum()), "ece": e, "brier": br}
        for (a, b), ref in zip(PAIRS, A["pairs"]):
            assert (ref["a"], ref["b"]) == (SYS[a], SYS[b])
            ha, hb = C[(a, p)]["hit"], C[(b, p)]["hit"]
            bb, cc = int((ha & ~hb).sum()), int((hb & ~ha).sum())
            assert (bb, cc) == (ref["a_only"], ref["b_only"]) and abs(mcnemar_exact_p(bb, cc) - ref["p"]) <= 1e-12 * max(ref["p"], 1e-300)
            fam["full" if p == "full" else p.split("_")[0]].append({"pool": p, "pair": (a, b), "b": bb, "c": cc,
                                                                     "p": mcnemar_exact_p(bb, cc), "ref": ref})
        order = tuple(sorted(SYS, key=lambda m: -out[f"{m}{p}"]["correct"]))
        assert len({out[f"{m}{p}"]["correct"] for m in SYS}) == 3 and [SYS[m] for m in order] == A["order"], p
        out[f"Order{p}"] = order
    for f, items in fam.items():
        assert an["families"][f] == len(items)
        for it, ph in zip(items, holm([it["p"] for it in items])):
            assert abs(ph - it["ref"]["p_holm"]) <= 1e-9 * max(ph, 1e-300), (f, it["pool"], it["pair"])
            it["p_holm"] = ph
    for it in fam["full"]:
        a, b = it["pair"]; tag = f"{a}Vs{b}"
        put(f"Full{tag}B", it["b"], str(it["b"])); put(f"Full{tag}C", it["c"], str(it["c"]))
        put(f"Full{tag}PHolm", it["p_holm"], fp(it["p_holm"]))
    put("FullMaxPHolm", max(it["p_holm"] for it in fam["full"]), fp(max(it["p_holm"] for it in fam["full"])))
    for f in ("pc24", "pc8"):
        w = {pr: Counter() for pr in PAIRS}
        for it in fam[f]:
            win = "none" if it["p_holm"] >= 0.05 else (it["pair"][0] if it["b"] > it["c"] else it["pair"][1])
            w[it["pair"]][win] += 1
        ref = an["ranking"][f]["holm_winners"]
        for (a, b), cnt in w.items():
            r_ = ref[f"{SYS[a]}-{SYS[b]}"]
            assert r_ == {SYS[a]: cnt[a], SYS[b]: cnt[b], "none": cnt["none"]}, (f, a, b)
        assert all(w[(a, b)][b] == DRAWS for a, b in PAIRS)        # the better-ranked member wins in every draw
        mx = max(it["p_holm"] for it in fam[f])
        put(f"{SIZE[f]}MaxPHolm", mx, fp(mx)); put(f"{SIZE[f]}Tests", len(fam[f]), str(len(fam[f])))
        same = sum(out[f"Order{p}"] == out["Orderfull"] for p in POOLS if p.startswith(f + "_"))
        assert an["ranking"][f]["outcomes"] == {"matches_full": same, "differs_from_full": DRAWS - same, "tie": 0, "incomplete": 0}
        put(f"{SIZE[f]}SameOrder", same, str(same))
    assert out["Orderfull"] == ("Flash", "Pplx", "Open")

    # --- headline values, size effects (§7.2, §7.3)
    for m in SYS:
        f0 = out[f"{m}full"]
        put(f"{m}FullAcc", f0["acc"], pct(f0["acc"])); put(f"{m}FullEce", f0["ece"], num(f0["ece"], 3))
        put(f"{m}FullBrier", f0["brier"], num(f0["brier"], 3))
        for f in ("pc24", "pc8"):
            ps = [p for p in POOLS if p.startswith(f + "_")]
            acc = [out[f"{m}{p}"]["acc"] for p in ps]; eff = [a - f0["acc"] for a in acc]
            ra = an["ranking"][f]["accuracy"][SYS[m]]; re_ = an["size_effect"][f][SYS[m]]
            assert abs(float(np.mean(acc)) - ra["mean"]) <= 1e-4 and (round(min(acc), 4), round(max(acc), 4)) == (ra["min"], ra["max"])
            # the analysis subtracts accuracies already rounded to 4 decimals, so a draw can differ by 1e-4
            assert max(abs(e - r) for e, r in zip(eff, re_["per_draw"])) <= 1.5e-4 and abs(float(np.mean(eff)) - re_["mean"]) <= 1.5e-4
            t = f"{m}{SIZE[f]}"
            put(f"{t}Acc", float(np.mean(acc)), pct(np.mean(acc))); put(f"{t}AccMin", min(acc), pct(min(acc)))
            put(f"{t}AccMax", max(acc), pct(max(acc)))
            put(f"{t}Drop", -float(np.mean(eff)), pct(-np.mean(eff), 1))
            put(f"{t}DropMin", -max(eff), pct(-max(eff), 1)); put(f"{t}DropMax", -min(eff), pct(-min(eff), 1))
            eces = [out[f"{m}{p}"]["ece"] for p in ps]
            put(f"{t}EceMin", min(eces), num(min(eces), 3)); put(f"{t}EceMax", max(eces), num(max(eces), 3))

    for m in SYS:                                                   # ECE over all 11 pools (descriptive)
        eces = [out[f"{m}{p}"]["ece"] for p in POOLS]
        put(f"{m}EceMinAll", min(eces), num(min(eces), 3)); put(f"{m}EceMaxAll", max(eces), num(max(eces), 3))

    # --- pool-fit (§7.5)
    for f in ("pc24", "pc8") if fit else ():
        ps = [p for p in POOLS if p.startswith(f + "_")]; d = []
        for p in ps:
            a, u = C[("Fit", p)], C[("Open", p)]; ref = an["pool_fit"][p]
            dv = float(a["hit"].mean() - u["hit"].mean()); d.append(dv)
            assert round(dv, 4) == ref["minus_unchanged"]["accuracy"] and round(ece(a["P"], a["hit"]), 4) == ref["ece10"]
            assert round(brier(a["P"], y), 4) == ref["brier"]
        r_ = an["pool_fit"][f"{f}_across_draws_accuracy_diff"]
        assert abs(float(np.mean(d)) - r_["mean"]) <= 1e-4 and (round(min(d), 4), round(max(d), 4)) == (r_["min"], r_["max"])
        t = f"Fit{SIZE[f]}"
        sg = lambda x: f"{100 * x:+.1f}"                            # signed points, for math mode
        put(f"{t}Diff", float(np.mean(d)), sg(np.mean(d)))
        put(f"{t}DiffMin", min(d), sg(min(d))); put(f"{t}DiffMax", max(d), sg(max(d)))
        fit_acc = [float(C[("Fit", p)]["hit"].mean()) for p in ps]
        put(f"{t}Acc", float(np.mean(fit_acc)), pct(np.mean(fit_acc)))
        # does re-fitting change any order? (descriptive)
        flips = sum(tuple(sorted(["Fit", "Pplx", "Flash"], key=lambda m: -(C[(m, p)]["hit"].sum()))) != ("Flash", "Pplx", "Fit")
                    for p in ps)
        put(f"{t}OrderFlips", int(flips), str(int(flips)))

    # --- consistency (§7.6) and the stable-tie re-score
    for m, path in EXISTING.items():
        old = cell(path) if m != "Open" else {**cell(path)}
        agree = float((old["pred"] == C[(m, "full")]["pred"]).mean()); ref = an["descriptive"]["full_vs_existing"][SYS[m]]
        assert round(agree, 4) == ref["argmax_agreement"] and round(float(old["hit"].mean()), 4) == ref["existing_accuracy"]
        put(f"{m}FullAgree", agree, pct(agree)); put(f"{m}FullChanged", int((old["pred"] != C[(m, "full")]["pred"]).sum()),
                                                     str(int((old["pred"] != C[(m, "full")]["pred"]).sum())))
    st = rows_of(STABLE)
    assert [r["retrieved_idx"] for r in st] == [json.loads(l)["retrieved_t_idx"] for l in gzip.open(PS / "manifest_full.jsonl.gz", "rt")]
    stp = np.array([r["pred"] for r in st]); put("OpenFullVsStableChanged", int((stp != C[("Open", "full")]["pred"]).sum()),
                                                  str(int((stp != C[("Open", "full")]["pred"]).sum())))

    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "poolsize.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = ["% GENERATED by paper/scripts/numbers_poolsize.py from the per-item dumps -- do not edit by hand."]
    lines += [f"\\newcommand{{\\ps{k}}}{{{mac[k]}}}" for k in sorted(mac)]
    (GEN / "numbers_poolsize.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(mac)} macros; retrieval exact {exact}, tie-only {tie_only}")


if __name__ == "__main__":
    main()
