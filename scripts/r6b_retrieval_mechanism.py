#!/usr/bin/env python3
"""R6b — POST HOC mechanism analysis: which part of the retrieval rule explains clean-run vs parity-run accuracy?

R6 showed that the clean configuration loses its advantage under the reproduction's retrieval rule. The two rules
differ in three ways: per-class cap (none vs <=4), tokenisation (unigrams vs words + adjacent bigrams) and pool
(P = train minus the selection split vs all 10,003 rows). This run separates all three, with
everything else identical to the clean configuration (results/clean/selection_scores.json: reader + kNN,
weighted_prob_mean, descriptions, 1 option order seeded 1000, parameters fitted at selection, not refit):

    r6b_cap4_unigram_P    the clean run's unigram BM25 over P, ranked by (-score, pool index), <=4 per class, 24
    r6b_nocap_bigram_P    the reproduction's word+bigram BM25 over P, ranked by (-score, pool index), no cap, 24
    r6b_cap4_bigram_P     the reproduction's full rule but over P (completes the 2x2 on P; vs R6 isolates the pool)

With the clean run (no cap, unigram, P) this is a full cap x tokenisation factorial on P, and r6b_cap4_bigram_P vs
r6_best_parityret (same rule over all 10,003 rows) isolates the pool.

Both members (reader prompt shots and kNN neighbours) use the same retrieved 24, as in every other run. Before any
GPU work two controls must pass, otherwise the script stops:
  (1) the unigram scores computed here, ranked as in quorum/fewshot.py, retrieve the same SET of 24 rows as the
      clean dump's retrieved_idx for every test item (order differences are counted and recorded, not fatal:
      BM25.top sums per-term scores in `set` iteration order, which depends on PYTHONHASHSEED, so near-tied rows
      can swap places between processes; in checks with three seeds this changed the order, never the set, of 1-2
      items);
  (2) the clean run's cached test logits, combined with the clean parameters, reproduce the clean dump's preds.

    PYTHONHASHSEED=0 python scripts/r6b_retrieval_mechanism.py      # the seed is required and recorded

Writes results/predictions/banking77_24shot_r6b_*.jsonl.gz + _meta.json and adds the runs to
results/paper/summary.json. Everything here is labelled post hoc; it is not part of the pre-registered protocol.
"""
import collections, gzip, hashlib, importlib.util, json, os, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("paper_runs", ROOT / "scripts" / "paper_runs.py")
PR = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(PR)
C = PR.C


def unigram_scores(bm, text):
    """Exactly the score vector quorum.fewshot.BM25.top computes (same loop, same operation order)."""
    q = re.findall(r"[a-z0-9]+", text.lower()); scores = np.zeros(bm.N)
    for w in set(q):
        idf = bm.idf.get(w)
        if idf is None:
            continue
        for i, c in enumerate(bm.tf):
            f = c.get(w, 0)
            if f:
                dl = len(bm.docs[i])
                scores[i] += idf * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * dl / bm.avgdl))
    return scores


def capped(order, labels, k=24, per_class=4):
    out, cnt = [], collections.Counter()
    for i in order:
        if cnt[labels[i]] >= per_class:
            continue
        out.append(int(i)); cnt[labels[i]] += 1
        if len(out) == k:
            break
    return out


def retrieval_stats(ret_train, train_labels, gold):
    return {"mean_distinct_classes": round(float(np.mean([len(set(train_labels[j] for j in x)) for x in ret_train])), 3),
            "max_same_class": int(max(max(collections.Counter(train_labels[j] for j in x).values()) for x in ret_train)),
            "gold_class_among_24": round(float(np.mean([g in {train_labels[j] for j in x}
                                                        for x, g in zip(ret_train, gold)])), 4)}


def main():
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (retrieval tie order depends on it)"
    ctx = PR.Ctx(); PR.CACHE.mkdir(parents=True, exist_ok=True); cfg = PR.best_cfg(ctx)
    assert cfg["option_text"] == "descriptions" and cfg["orders"] == 1 and cfg["rule"] == "weighted_prob_mean"
    clean = [json.loads(l) for l in gzip.open(PR.OUTP / "banking77_24shot_clean.jsonl.gz", "rt")]
    clean_pred = np.array([r["pred"] for r in clean]); clean_ret = [r["retrieved_idx"] for r in clean]
    train_labels = np.array([r["label"] for r in ctx.train])

    # control (2): CPU only, from the clean run's cache
    rc, kc = PR.CLEAN_CACHE / "test_reader_descriptions_o0.npy", PR.CLEAN_CACHE / "test_knn.npy"
    comb, _, _ = PR.combine_run(np.load(rc)[None], np.load(kc), cfg)
    ctl2 = int((comb.argmax(1) != clean_pred).sum())
    print(f"control 2 (clean cache -> clean preds): mismatches={ctl2}", flush=True)
    assert ctl2 == 0, "combine path does not reproduce the clean run"

    eng = ctx.engine("P"); P = eng.pool_to_train; pool_labels = eng.pool_lab
    pr = PR.ParityRetriever(eng.pool)
    uni_cap, bi_nocap, bi_cap, ctl1, ctl1_order = [], [], [], 0, []
    for i, r in enumerate(ctx.test):
        s = unigram_scores(eng.bm, r["text"]); mine = [P[j] for j in np.argsort(-s)[:24]]
        if set(mine) != set(clean_ret[i]):
            ctl1 += 1
        elif mine != clean_ret[i]:
            ctl1_order.append(i)
        order = sorted(range(len(s)), key=lambda j: (-s[j], j))
        uni_cap.append(capped(order, pool_labels))
        bi_nocap.append(pr.top(r["text"], k=24, per_class=24)); bi_cap.append(pr.top(r["text"], k=24, per_class=4))
        if (i + 1) % 500 == 0:
            print(f"  retrieval {i + 1}/3080 (control-1 mismatches so far {ctl1})", flush=True)
    print(f"control 1 (unigram ranking -> clean retrieved set): mismatches={ctl1}; order-only differences at "
          f"{ctl1_order}", flush=True)
    assert ctl1 == 0, "unigram scoring does not reproduce the clean run's retrieved sets"

    code = {"r6b_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    gold = [r["label"] for r in ctx.test]
    for tag, ret_pool, desc, rule in [
        ("r6b_cap4_unigram_P", uni_cap, "clean config; unigram BM25 over P with <=4 per class (post hoc)",
         "unigram BM25 k1=1.5 b=0.75 over P, ranked by (-score, pool index), <=4 per class, 24"),
        ("r6b_nocap_bigram_P", bi_nocap, "clean config; word+bigram BM25 over P without a per-class cap (post hoc)",
         "word+bigram BM25 k1=1.5 b=0.75 over P, ranked by (-score, pool index), no cap, 24"),
        ("r6b_cap4_bigram_P", bi_cap, "clean config; the reproduction's retrieval rule over P (post hoc)",
         "word+bigram BM25 k1=1.5 b=0.75 over P, ranked by (-score, pool index), <=4 per class, 24")]:
        memo = {r["text"]: x for r, x in zip(ctx.test, ret_pool)}
        eng._shots = lambda text, memo=memo: memo[text]
        per = PR.reader_logits(ctx, eng, tag, ctx.test, ctx.descs, cfg["orders"])
        knn = C.cached(PR.CACHE / f"{tag}_test_knn.npy", lambda: eng._knn_logits(ctx.test))
        comb, rp, kp = PR.combine_run(per, knn, cfg)
        ret_train = [[P[j] for j in x] for x in ret_pool]
        PR.write_dump(ctx, tag, comb, rp, kp, per, ret_train,
                      {"description": desc, "retrieval": rule, "post_hoc": True,
                       "retrieval_stats": retrieval_stats(ret_train, train_labels, gold),
                       "clean_retrieval_stats": retrieval_stats(clean_ret, train_labels, gold),
                       "controls": {"clean_cache_to_clean_preds_mismatches": ctl2,
                                    "unigram_ranking_to_clean_retrieved_set_mismatches": ctl1,
                                    "same_set_different_order_items": ctl1_order},
                       "pythonhashseed": os.environ["PYTHONHASHSEED"],
                       "params": {"rule": cfg["rule"], "orders": cfg["orders"], "T": cfg["T"], "w": cfg["w"]},
                       "note": "parameters are those fitted at selection (not refit for this retrieval)",
                       "r6b_code": code})
    eng._shots = eng.default_shots
    print("R6B_DONE", flush=True)


if __name__ == "__main__":
    main()
