#!/usr/bin/env python3
"""R6c — POST HOC sensitivity of the clean run to BM25 tie-breaking.

quorum.fewshot.BM25.top ranks with np.argsort(-scores), numpy's default (unstable) sort, so rows whose scores tie
are ordered by an implementation detail that differs between platforms: recomputing the clean run's retrieval on
x86_64 (numpy 2.4.6) gives a different SET of 24 for 427-428 of the 3,080 test items and a different ORDER for
~1,760 more, every difference being a pure score tie (the clean dump was produced on aarch64, where the same code
reproduces every set). The frozen protocol code path is left unchanged. This run repeats the clean configuration
exactly (results/clean/selection_scores.json: reader + kNN, weighted_prob_mean, descriptions, 1 option order seeded
1000, the parameters fitted at selection, not refit) with a platform-independent retrieval:

    scores: per-term BM25 contributions (k1=1.5, b=0.75, unigrams, pool P) summed with terms in SORTED order
    ranking: np.argsort(-scores, kind="stable"), i.e. by (-score, pool index); top 24, no per-class cap

    PYTHONHASHSEED=0 python scripts/r6c_tiebreak.py

The meta records how many items' retrieval differs from the clean dump (set / order only), a check that every such
difference is a score tie, the numpy version and machine, accuracy, and exact McNemar against the clean run and the
reproduction. Writes results/predictions/banking77_24shot_r6c_stable_tiebreak.jsonl.gz + _meta.json.
"""
import gzip, hashlib, importlib.util, json, os, platform, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("paper_runs", ROOT / "scripts" / "paper_runs.py")
PR = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(PR)
C = PR.C
TAG = "r6c_stable_tiebreak"


def det_scores(bm, text):
    """BM25 scores as in quorum.fewshot.BM25.top, but summed over the query's terms in sorted order."""
    scores = np.zeros(bm.N)
    for w in sorted(set(re.findall(r"[a-z0-9]+", text.lower()))):
        idf = bm.idf.get(w)
        if idf is None:
            continue
        for i, c in enumerate(bm.tf):
            f = c.get(w, 0)
            if f:
                dl = len(bm.docs[i])
                scores[i] += idf * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * dl / bm.avgdl))
    return scores


def main():
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
    ctx = PR.Ctx(); PR.CACHE.mkdir(parents=True, exist_ok=True); cfg = PR.best_cfg(ctx)
    assert cfg["option_text"] == "descriptions" and cfg["orders"] == 1 and cfg["rule"] == "weighted_prob_mean"
    clean = [json.loads(l) for l in gzip.open(PR.OUTP / "banking77_24shot_clean.jsonl.gz", "rt")]
    clean_pred = np.array([r["pred"] for r in clean]); clean_ret = [r["retrieved_idx"] for r in clean]
    assert [r["text"] for r in clean] == [r["text"] for r in ctx.test]

    eng = ctx.engine("P"); P = eng.pool_to_train; inv = {t: j for j, t in enumerate(P)}
    ret_pool, set_diff, order_diff, non_tie = [], [], [], 0
    for i, r in enumerate(ctx.test):
        s = det_scores(eng.bm, r["text"]); top = [int(j) for j in np.argsort(-s, kind="stable")[:24]]
        ret_pool.append(top); old = [inv[t] for t in clean_ret[i]]
        if set(top) != set(old):
            set_diff.append(i); cut = s[top[-1]]
            non_tie += int(any(abs(s[j] - cut) > 1e-9 for j in set(top) ^ set(old)))
        elif top != old:
            order_diff.append(i)
            non_tie += int(any(abs(s[a] - s[b]) > 1e-9 for a, b in zip(top, old) if a != b))
        if (i + 1) % 500 == 0:
            print(f"  retrieval {i + 1}/3080: set diffs {len(set_diff)}, order-only diffs {len(order_diff)}", flush=True)
    print(f"retrieval vs clean dump: set differs {len(set_diff)}, order only {len(order_diff)}, non-tie {non_tie}",
          flush=True)
    assert non_tie == 0, "a retrieval difference is not explained by a score tie"

    memo = {r["text"]: x for r, x in zip(ctx.test, ret_pool)}
    eng._shots = lambda text: memo[text]
    per = PR.reader_logits(ctx, eng, TAG, ctx.test, ctx.descs, cfg["orders"])
    knn = C.cached(PR.CACHE / f"{TAG}_test_knn.npy", lambda: eng._knn_logits(ctx.test))
    comb, rp, kp = PR.combine_run(per, knn, cfg)
    eng._shots = eng.default_shots
    pred = comb.argmax(1); hit, hc = pred == ctx.gold, clean_pred == ctx.gold
    b, c = int((hit & ~hc).sum()), int((~hit & hc).sum())
    changed = sorted(set(set_diff) | set(order_diff))
    vs_clean = {"b_r6c_right_clean_wrong": b, "c_r6c_wrong_clean_right": c, "p_two_sided_exact": C.mcnemar_exact(b, c),
                "pred_changed_items": int((pred != clean_pred).sum()),
                "pred_changed_among_retrieval_changed": int((pred[changed] != clean_pred[changed]).sum()),
                "pred_changed_among_retrieval_unchanged": int(sum(pred[i] != clean_pred[i]
                                                                  for i in range(3080) if i not in set(changed)))}
    import torch
    PR.write_dump(ctx, TAG, comb, rp, kp, per, [[P[j] for j in x] for x in ret_pool],
                  {"description": "clean config with platform-independent BM25 tie-breaking (post hoc sensitivity)",
                   "retrieval": "unigram BM25 k1=1.5 b=0.75 over P, terms summed in sorted order, "
                                "np.argsort(-scores, kind='stable') (ties by pool index), top 24, no cap",
                   "post_hoc": True,
                   "retrieval_vs_clean_dump": {"set_differs": len(set_diff), "order_only_differs": len(order_diff),
                                               "differences_not_explained_by_ties": non_tie,
                                               "set_differs_items": set_diff},
                   "vs_clean_run": vs_clean,
                   "params": {"rule": cfg["rule"], "orders": cfg["orders"], "T": cfg["T"], "w": cfg["w"]},
                   "platform": {"machine": platform.machine(), "numpy": np.__version__, "python": platform.python_version(),
                                "torch": torch.__version__, "pythonhashseed": os.environ["PYTHONHASHSEED"]},
                   "r6c_code": {"sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    print("R6C vs clean:", json.dumps(vs_clean), flush=True); print("R6C_DONE", flush=True)


if __name__ == "__main__":
    main()
