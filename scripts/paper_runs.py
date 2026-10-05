#!/usr/bin/env python3
"""Paper evidence runs R5 (ablations), R6 (protocol-parity retrieval), R8 (latency), R9 (order-seed variance).

All runs use the pinned revisions and helpers of scripts/clean_24shot.py. "best" = the clean-selected
configuration (results/clean/selection_scores.json: reader + kNN, weighted_prob_mean, descriptions, 1 order,
parameters fitted on the selection split, pool = train minus the selection split). "default" = the repo default
(class names, T=1 geometric mean, 3 orders, full 10,003-row pool). Each run writes a per-item dump in the same
schema as the clean run plus a meta, and every run is summarised in results/paper/summary.json with its accuracy
and an exact McNemar against the public reproduction's per-item predictions (read at run time, never stored).

    python scripts/paper_runs.py r5     # derived reader/kNN-alone, base model without the LoRA, BM25 label votes
    python scripts/paper_runs.py r6     # reproduction's retrieval rule (word+bigram BM25, <=4 per class, 24)
    python scripts/paper_runs.py r9     # test-time option-order seeds 2000 and 3000 for best and default
    python scripts/paper_runs.py r8     # latency / throughput micro-benchmarks
"""
import argparse, collections, gzip, hashlib, importlib.util, json, math, re, sys, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("clean24", ROOT / "scripts" / "clean_24shot.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

OUTP, PAPER = ROOT / "results" / "predictions", ROOT / "results" / "paper"
CACHE = ROOT / "results" / "paper" / "cache"
CLEAN_CACHE = ROOT / "results" / "clean" / "cache"
REPRO_DEFS_URL = ("https://raw.githubusercontent.com/simonmesmith/jev-banking77-experiment/"
                  "5cac4ff/configs/descriptions.json")                     # read at run time, never stored
REPRO_TRAIN_URL = ("https://raw.githubusercontent.com/simonmesmith/jev-banking77-experiment/"
                   "5cac4ff/data/train.json")                              # alignment check only, never stored


# ---------------- shared setup ----------------
class Ctx:
    def __init__(self):
        from datasets import load_dataset
        from quorum.data import _humanize
        tr = load_dataset(C.DATASET, revision=C.DATASET_REV, split="train")
        self.raw = list(tr.features["label"].names); self.names = [_humanize(n) for n in self.raw]
        self.train = [{"text": r["text"], "label": int(r["label"])} for r in tr]
        self.test = [{"text": r["text"], "label": int(r["label"])}
                     for r in load_dataset(C.DATASET, revision=C.DATASET_REV, split="test")]
        self.gold = np.array([r["label"] for r in self.test])
        self.S = C.load_split(); Sset = set(self.S)
        self.P = [i for i in range(len(self.train)) if i not in Sset]
        self.descs = C.descriptions(self.raw)
        self.sel = json.loads((C.OUT / "selection_scores.json").read_text())
        self.repro = self._repro()
        self.engines = {}

    def _repro(self):
        try:
            import csv, io
            txt = urllib.request.urlopen(C.REPRO_URL, timeout=60).read().decode()
            rep = {int(r["id"].split("-")[1]): r for r in csv.DictReader(io.StringIO(txt))}
            assert len(rep) == 3080 and all(rep[i]["truth"] == self.raw[self.gold[i]] for i in range(3080))
            return np.array([rep[i]["correct"] == "True" for i in range(3080)])
        except Exception as e:
            print("reproduction predictions unavailable:", e, flush=True); return None

    def engine(self, pool_name):
        """One model load per pool; options are swapped per run."""
        if pool_name not in self.engines:
            from quorum.fewshot import RetrievalEnsemble
            idx = self.P if pool_name == "P" else list(range(len(self.train)))
            eng = RetrievalEnsemble(self.names, C.TASK_DESC, base=C.BASE, adapter=C.ADAPTER, embedder=C.KNN_EMBEDDER,
                                    shots=24, perms=1, device="cuda", base_revision=C.BASE_REV,
                                    adapter_revision=C.ADAPTER_REV, max_length=C.MAX_PROMPT_TOKENS, truncate=False)
            eng.index([self.train[i] for i in idx]); eng.pool_to_train = idx; eng.default_shots = eng._shots
            self.engines[pool_name] = eng
        return self.engines[pool_name]

    def mcnemar(self, pred):
        hit = np.asarray(pred) == self.gold
        out = {"accuracy": round(float(hit.mean()), 4), "correct": int(hit.sum())}
        if self.repro is not None:
            b, c = int((hit & ~self.repro).sum()), int((~hit & self.repro).sum())
            out.update({"b_ours_right_theirs_wrong": b, "c_ours_wrong_theirs_right": c,
                        "p_two_sided_exact": C.mcnemar_exact(b, c)})
        return out


def fingerprint():
    files = ["scripts/paper_runs.py", "scripts/clean_24shot.py", "quorum/fewshot.py", "quorum/prompting.py"]
    return {"git_head": C._read_git_head(),
            "sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}


def summary_add(tag, rec):
    PAPER.mkdir(parents=True, exist_ok=True); p = PAPER / "summary.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    s[tag] = rec; p.write_text(json.dumps(s, indent=1)); print(f"[{tag}] {json.dumps(rec)}", flush=True)


g7 = lambda v: [float(f"{x:.7g}") for x in v]


def write_dump(ctx, tag, probs, reader_p, knn_p, per_logits, retrieved, extra_meta):
    OUTP.mkdir(parents=True, exist_ok=True); pred = probs.argmax(1)
    with gzip.open(OUTP / f"banking77_24shot_{tag}.jsonl.gz", "wt") as f:
        for i, r in enumerate(ctx.test):
            row = {"idx": i, "text": r["text"], "gold": r["label"], "pred": int(pred[i]), "probs": g7(probs[i])}
            if reader_p is not None: row["reader_probs"] = g7(reader_p[i])
            if knn_p is not None: row["knn_probs"] = g7(knn_p[i])
            if retrieved is not None: row["retrieved_idx"] = retrieved[i]
            if per_logits is not None:
                row["perm_reader_probs"] = [g7(C.softmax(per_logits[o][i:i + 1])[0]) for o in range(len(per_logits))]
            f.write(json.dumps(row) + "\n")
    res = ctx.mcnemar(pred)
    meta = {"tag": tag, "code": fingerprint(), "dataset": C.DATASET, "dataset_revision": C.DATASET_REV,
            "base": C.BASE, "base_revision": C.BASE_REV, "adapter": C.ADAPTER, "adapter_revision": C.ADAPTER_REV,
            "result": res, "revisions": C.resolved_revisions(),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra_meta}
    (OUTP / f"banking77_24shot_{tag}_meta.json").write_text(json.dumps(meta, indent=1))
    summary_add(tag, {**res, **{k: extra_meta[k] for k in ("description",) if k in extra_meta}})
    return res


def best_cfg(ctx):
    w = ctx.sel["selected"]; f = ctx.sel["refit_on_full_selection_split"]
    return {"rule": w["rule"], "orders": w["orders"], "option_text": w["option_text"],
            "T": [f["temperatures"]["reader"], f["temperatures"]["knn"]], "w": f["weights"]}


def reader_logits(ctx, eng, tag, rows, opts, orders, order_base=1000, use_lora=True):
    eng.options = list(opts); outs = []
    for o in range(orders):
        def compute(o=o):
            if use_lora:
                return eng.incontext_logits_per_perm(rows, 1, bs=C.BS, perm_offset=order_base - 1000 + o)[0]
            with eng.model.disable_adapter():
                return eng.incontext_logits_per_perm(rows, 1, bs=C.BS, perm_offset=order_base - 1000 + o)[0]
        outs.append(C.cached(CACHE / f"{tag}_reader_o{o}.npy", compute))
    return np.stack(outs)


def combine_run(per, knn, cfg):
    Ls = [per.mean(0), knn]
    comb = C.combine(Ls, {"T": cfg["T"], "w": cfg["w"]}, cfg["rule"])
    return comb, C.softmax(Ls[0], cfg["T"][0]), C.softmax(Ls[1], cfg["T"][1])


# ---------------- R5: ablations ----------------
def r5():
    ctx = Ctx(); CACHE.mkdir(parents=True, exist_ok=True); cfg = best_cfg(ctx)
    clean = [json.loads(l) for l in gzip.open(OUTP / "banking77_24shot_clean.jsonl.gz", "rt")]
    rp = np.array([r["reader_probs"] for r in clean]); kp = np.array([r["knn_probs"] for r in clean])
    ret = [r["retrieved_idx"] for r in clean]
    write_dump(ctx, "r5_reader_alone", rp, rp, None, None, ret,
               {"description": "best config, reader member alone (derived from the clean dump)"})
    write_dump(ctx, "r5_knn_alone", kp, None, kp, None, ret,
               {"description": "best config, kNN member alone (derived from the clean dump)"})

    # BM25 label votes over the same 24 retrieved examples (pool = P), CPU only
    eng = ctx.engine("P"); labels = np.array([r["label"] for r in ctx.train])
    maj = np.zeros((3080, C.K)); wtd = np.zeros((3080, C.K))
    for i, r in enumerate(ctx.test):
        q = re.findall(r"[a-z0-9]+", r["text"].lower()); idx = eng.shots_idx(r["text"])
        for rank, j in enumerate(idx):
            lab = labels[eng.pool_to_train[int(j)]]
            maj[i, lab] += 1.0 - 1e-6 * rank                # ties -> the label retrieved first
            wtd[i, lab] += _bm25_score(eng, q, int(j))
    to_p = lambda M: (M + 1e-9) / (M + 1e-9).sum(1, keepdims=True)
    write_dump(ctx, "r5_bm25vote_majority", to_p(maj), None, None, None, ret,
               {"description": "majority label among the 24 BM25-retrieved examples (ties: earliest retrieved)"})
    write_dump(ctx, "r5_bm25vote_weighted", to_p(wtd), None, None, None, ret,
               {"description": "BM25-score-weighted label vote over the 24 retrieved examples"})

    # base model WITHOUT the LoRA, same prompt/scoring; parameters refit on the selection split with the same rule
    opts = ctx.descs if cfg["option_text"] == "descriptions" else ctx.names
    srows = [ctx.train[i] for i in ctx.S]; ys = np.array([r["label"] for r in srows])
    per_s = reader_logits(ctx, eng, "r5base_sel", srows, opts, cfg["orders"], use_lora=False)
    knn_s = C.cached(CLEAN_CACHE / "sel_knn.npy", lambda: eng._knn_logits(srows))
    prm = C.fit_params([per_s.mean(0), knn_s], ys, cfg["rule"])
    per_t = reader_logits(ctx, eng, "r5base_test", ctx.test, opts, cfg["orders"], use_lora=False)
    knn_t = C.cached(CLEAN_CACHE / "test_knn.npy", lambda: eng._knn_logits(ctx.test))
    bcfg = {**cfg, "T": prm["T"], "w": prm["w"]}
    comb, rpb, kpb = combine_run(per_t, knn_t, bcfg)
    write_dump(ctx, "r5_base_nolora", comb, rpb, kpb, per_t, ret,
               {"description": "best config with the base model (LoRA disabled); params refit on the selection split",
                "fitted": {"T": prm["T"], "w": prm["w"]}})
    write_dump(ctx, "r5_base_nolora_reader_alone", rpb, rpb, None, per_t, ret,
               {"description": "base model (LoRA disabled) reader alone, best-config prompt"})


def _bm25_score(eng, q_terms, j):
    bm = eng.bm; c = bm.tf[j]; dl = len(bm.docs[j]); s = 0.0
    for w in set(q_terms):
        idf = bm.idf.get(w); f = c.get(w, 0)
        if idf is not None and f:
            s += idf * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * dl / bm.avgdl))
    return s


# ---------------- R6: the reproduction's retrieval rule ----------------
class ParityRetriever:
    """Re-implementation of the reproduction's retrieval rule: BM25 (k1=1.5, b=0.75) over word unigrams +
    adjacent-word bigrams, ranked by (-score, row index), then walk the ranking keeping at most 4 examples per
    class until 24 are selected."""
    def __init__(self, rows):
        self.rows = rows; self.post = collections.defaultdict(list); self.len = []
        for i, r in enumerate(rows):
            ts = self.terms(r["text"]); self.len.append(len(ts))
            for t, n in collections.Counter(ts).items():
                self.post[t].append((i, n))
        self.avg = sum(self.len) / len(rows); N = len(rows)
        self.idf = {t: math.log(1 + (N - len(v) + .5) / (len(v) + .5)) for t, v in self.post.items()}

    @staticmethod
    def terms(text):
        w = re.findall(r"[a-z0-9]+", text.lower()); return w + [a + "_" + b for a, b in zip(w, w[1:])]

    def top(self, text, k=24, per_class=4):
        sc = collections.defaultdict(float)
        for t in set(self.terms(text)):
            for i, tf in self.post.get(t, []):
                sc[i] += self.idf[t] * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.len[i] / self.avg))
        ranked = sorted(range(len(self.rows)), key=lambda i: (-sc[i], i)); out, cnt = [], collections.Counter()
        for i in ranked:
            lab = self.rows[i]["label"]
            if cnt[lab] >= per_class: continue
            out.append(i); cnt[lab] += 1
            if len(out) == k: break
        return out


def r6():
    ctx = Ctx(); CACHE.mkdir(parents=True, exist_ok=True); cfg = best_cfg(ctx)
    eng = ctx.engine("T")                                          # the reproduction retrieves from all 10,003
    pr = ParityRetriever(ctx.train); memo = {}
    eng._shots = lambda text: memo.setdefault(text, pr.top(text))  # both members share this retrieval
    try:
        rdefs = json.loads(urllib.request.urlopen(REPRO_DEFS_URL, timeout=60).read().decode())
        defs = [rdefs[r] for r in ctx.raw]
    except Exception as e:
        print("reproduction label definitions unavailable:", e, flush=True); defs = None
    try:   # ties are broken by row index, so the support order must match the reproduction's
        rtr = json.loads(urllib.request.urlopen(REPRO_TRAIN_URL, timeout=120).read().decode())
        align = {"reproduction_support_rows": len(rtr), "our_rows": len(ctx.train),
                 "same_text_and_label_at_same_index": sum(a["text"] == b["text"] and a["label"] == ctx.raw[b["label"]]
                                                          for a, b in zip(rtr, ctx.train))}
    except Exception as e:
        align = {"error": f"{type(e).__name__}: {e}"}
    print("support-order alignment:", json.dumps(align), flush=True)
    ret = [[int(j) for j in eng._shots(r["text"])] for r in ctx.test]
    knn_t = C.cached(CACHE / "r6_test_knn.npy", lambda: eng._knn_logits(ctx.test))
    dflt = {"rule": "logprob_mean", "orders": 3, "T": [1.0, 1.0], "w": None}
    runs = [("r6_default_parityret", dflt, ctx.names, "names"),
            ("r6_best_parityret", cfg, ctx.descs if cfg["option_text"] == "descriptions" else ctx.names, cfg["option_text"])]
    if defs is not None:
        runs += [("r6_default_parityret_reprodefs", dflt, defs, "reproduction definitions"),
                 ("r6_best_parityret_reprodefs", cfg, defs, "reproduction definitions")]
    for tag, c, opts, otxt in runs:
        per = reader_logits(ctx, eng, tag, ctx.test, opts, c["orders"])
        comb, rp, kp = combine_run(per, knn_t, c)
        write_dump(ctx, tag, comb, rp, kp, per, ret,
                   {"description": f"{'best' if c is cfg else 'default'} config with the reproduction's retrieval "
                                   f"rule (word+bigram BM25, <=4 per class, 24) over all 10,003 train rows; "
                                   f"option text: {otxt}",
                    "retrieval": "word+bigram BM25 k1=1.5 b=0.75, <=4 per class, 24, ties by row index",
                    "support_order_alignment": align,
                    "label_text_source": REPRO_DEFS_URL if otxt == "reproduction definitions" else otxt,
                    "params": {"rule": c["rule"], "orders": c["orders"], "T": c["T"], "w": c["w"]},
                    "note": "best-config parameters are those fitted at selection (not refit for this retrieval)"})


# ---------------- R9: test-time option-order seeds ----------------
def r9():
    ctx = Ctx(); CACHE.mkdir(parents=True, exist_ok=True); cfg = best_cfg(ctx)
    dflt = {"rule": "logprob_mean", "orders": 3, "T": [1.0, 1.0], "w": None}
    for pool, c, opts, name in [("P", cfg, ctx.descs if cfg["option_text"] == "descriptions" else ctx.names, "best"),
                                ("T", dflt, ctx.names, "default")]:
        eng = ctx.engine(pool)
        knn_t = C.cached(CACHE / f"r9_{name}_test_knn.npy", lambda: eng._knn_logits(ctx.test))
        ret = [[eng.pool_to_train[int(j)] for j in eng.shots_idx(r["text"])] for r in ctx.test]
        for base in (2000, 3000):
            tag = f"r9_{name}_seed{base}"
            per = reader_logits(ctx, eng, tag, ctx.test, opts, c["orders"], order_base=base)
            comb, rp, kp = combine_run(per, knn_t, c)
            write_dump(ctx, tag, comb, rp, kp, per, ret,
                       {"description": f"{name} config, option orders seeded {base}+i (seed 1000 is the main run)",
                        "order_seed_base": base, "params": {"rule": c["rule"], "orders": c["orders"]}})


# ---------------- R8: latency / throughput ----------------
def r8():
    import torch
    from quorum import ZeroShotEnsemble
    ctx = Ctx(); rows = ctx.test[:64]; ids = list(range(C.K)); res = {}
    sync = lambda: torch.cuda.synchronize() if torch.cuda.is_available() else None
    t = time.time(); zs = ZeroShotEnsemble(device="cuda"); zs.members[0].model = zs.members[0].model.float(); sync()
    res["zero_shot_load_seconds"] = round(time.time() - t, 2)
    t = time.time(); zs.predict_proba(rows[0]["text"], ids, label_texts=ctx.names); sync()
    res["zero_shot_cold_first_call_seconds"] = round(time.time() - t, 3)
    lat = []
    for r in rows[1:33]:
        t = time.time(); zs.predict_proba(r["text"], ids, label_texts=ctx.names); sync(); lat.append(time.time() - t)
    res["zero_shot_warm_batch1"] = {"n": len(lat), "median_ms": round(1000 * float(np.median(lat)), 1),
                                    "p95_ms": round(1000 * float(np.percentile(lat, 95)), 1),
                                    "note": "77 labels; NLI cast to fp32, embedders fp32 (sentence-transformers default)"}
    del zs; torch.cuda.empty_cache()
    cfg = best_cfg(ctx); t = time.time(); eng = ctx.engine("P"); sync()
    res["fewshot_load_and_index_seconds"] = round(time.time() - t, 2)
    eng.options = list(ctx.descs)
    for bs in (1, 8):
        t = time.time(); eng.incontext_logits_per_perm(rows[:8], 1, bs=bs); sync(); warm = time.time() - t   # warm-up
        t = time.time(); eng.incontext_logits_per_perm(rows, 1, bs=bs); sync(); dt = time.time() - t
        res[f"fewshot_reader_bs{bs}"] = {"items": len(rows), "seconds": round(dt, 2),
                                         "ms_per_item_per_order": round(1000 * dt / len(rows), 1),
                                         "warmup_8_items_seconds": round(warm, 2)}
    t = time.time(); eng._knn_logits(rows); sync()
    res["fewshot_knn_ms_per_item"] = round(1000 * (time.time() - t) / len(rows), 2)
    p = torch.cuda.get_device_properties(0)
    res["hardware"] = {"gpu": torch.cuda.get_device_name(0), "memory_gb": round(p.total_memory / 1e9, 1),
                       "torch": torch.__version__, "cuda": torch.version.cuda,
                       "reader_precision": "bf16", "prompt": "best config (descriptions, 24 shots, 1 order)"}
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "latency.json").write_text(json.dumps({"code": fingerprint(), **res}, indent=1))
    print("R8:", json.dumps(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run", choices=["r5", "r6", "r8", "r9"])
    {"r5": r5, "r6": r6, "r8": r8, "r9": r9}[ap.parse_args().run]()
