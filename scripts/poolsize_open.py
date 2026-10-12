#!/usr/bin/env python3
"""Pool size (Banking77), the open stack (`provenance/banking77_pool_size_protocol.md` §4–§6):
the Banking77 clean selection unchanged (M2 = reader + kNN, weighted_prob_mean, descriptions, 1 order, fitted
parameters from `results/clean/selection_scores.json`), on every pool, with the test rows' retrieval taken from the
pool's recorded manifest. On the 10 small pools it also computes the descriptive pool-fit arm (§5): the same
candidate with its parameters re-fitted on the pool's own rows (leave-one-out retrieval), no use of the selection split.

    python scripts/poolsize_open.py run          # all 11 pools, run order; a cell whose dump exists is skipped
    python scripts/poolsize_open.py smoke        # 8 selection-split rows retrieving from pc8_d1; output discarded

It reuses `scripts/clean_24shot.py` (imported, never edited). The zero-shot member is not loaded: the selected
candidate (M2) does not use it.
"""
import argparse, gzip, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("clean_24shot")                                          # the Banking77 clean run's code, unchanged
PB = _load("poolsize_bodies")
B = _load("bitext24_bodies")                                       # StableBM25 (R16 A1)
OUTP = PB.OUT / "predictions"
M2_FILES = ["scripts/clean_24shot.py", "quorum/fewshot.py", "quorum/prompting.py", "quorum/data.py",
            "examples/banking77_descriptions.py"]                  # the code the M2 path runs (no quorum/ensemble.py)
g7 = lambda v: [float(f"{x:.7g}") for x in v]


class Recorded:
    """Test rows: the manifest's recorded top 24, as positions in the indexed pool."""

    def __init__(self, test, man, pool):
        pos = {t: j for j, t in enumerate(pool)}; self.by_text = {}
        for r, m in zip(test, man):
            rec = [pos[t] for t in m["retrieved_t_idx"]]
            assert self.by_text.setdefault(r["text"], rec) == rec, "same text, different recorded retrieval"

    def top(self, text, k):
        assert k == PB.SHOTS
        return list(self.by_text[text])


class LeaveOneOut:
    """Pool rows: BM25 top k from the pool minus the query row's own position (same scores, stable ties)."""

    def __init__(self, texts):
        self.bm = B.StableBM25(texts); self.own = {t: j for j, t in enumerate(texts)}
        assert len(self.own) == len(texts), "duplicate texts in the pool: own position would be ambiguous"

    def top(self, text, k):
        o = self.own[text]
        return [j for j in self.bm.top(text, k + 1) if j != o][:k]


def check_selection():
    sel_bytes = (C.OUT / "selection_scores.json").read_bytes(); sel = json.loads(sel_bytes)
    win, prm = sel["selected"], sel["refit_on_full_selection_split"]
    assert (win["member_set"], win["rule"], win["option_text"], win["orders"]) == ("M2", "weighted_prob_mean", "descriptions", 1)
    for f in M2_FILES:                                             # the M2 code is the code that made the selection
        assert sel["code"]["sha256"][f] == hashlib.sha256((ROOT / f).read_bytes()).hexdigest(), f"{f} changed"
    return win, prm, hashlib.sha256(sel_bytes).hexdigest()


def member_Ls(tag, rows, eng, opts):
    m = C.member_logits(tag, rows, eng, None, opts, "descriptions", 1, False)
    return C.cand_logits(m, "M2", 1)                               # [reader (N, K), knn (N, K)]


def fit_on(Ls, y):
    """The clean run's final refit (selection §5): member temperatures + weights, then the final temperature."""
    prm = C.fit_params(Ls, y, "weighted_prob_mean")
    tfin = C.fit_temperature(np.log(C.combine(Ls, prm, "weighted_prob_mean") + 1e-12), y)
    return {"temperatures": dict(zip(C.MEMBER_SETS["M2"], prm["T"])), "weights": prm["w"], "final_temperature": tfin}


def calib(P, gold):
    pred = P.argmax(1); hit = pred == gold; conf = P.max(1); e = 0.0
    for i in range(10):
        m = (conf >= i / 10) & ((conf < (i + 1) / 10) if i < 9 else (conf <= 1.0))
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return {"accuracy": round(float(hit.mean()), 4), "correct": int(hit.sum()), "ece10": round(float(e), 4),
            "brier": round(float(((P - np.eye(P.shape[1])[gold]) ** 2).sum(1).mean()), 4)}


def write_cell(tag, name, test, gold, comb, ret, extra):
    dump = OUTP / f"{tag}_{name}.jsonl.gz"; tmp = dump.with_suffix(".partial")
    with gzip.open(tmp, "wt") as f:
        for i in range(len(test)):
            f.write(json.dumps({"idx": i, "gold": int(gold[i]), "pred": int(np.argmax(comb[i])), "probs": g7(comb[i]),
                                "unanswered": False, "retrieved_idx": ret[i]}) + "\n")
    tmp.rename(dump)
    meta = {"protocol": PB.PROTOCOL, "system": tag, "pool": name, "results": calib(comb, gold), **extra,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (PB.OUT / f"summary_{tag}_{name}.json").write_text(json.dumps(meta, indent=1))
    print("POOLSIZE_CELL_DONE", tag, name, json.dumps(meta["results"]), flush=True)


def phase_run(smoke=False):
    win, prm_rec, sel_sha = check_selection()
    _, raw, names, train = C.load_train()
    tr_pb, raw_pb = PB.train_rows()
    assert raw_pb == raw and [r["text"] for r in tr_pb] == [r["text"] for r in train]
    opts = {"names": names, "descriptions": C.descriptions(raw)}
    pl = PB.pools(tr_pb)
    eng = C.build_engine(names, [train[i] for i in pl["full"]])
    prm = {"T": [prm_rec["temperatures"][n] for n in C.MEMBER_SETS["M2"]], "w": prm_rec["weights"]}
    if smoke:
        pool = pl["pc8_d1"]; rows = [train[i] for i in json.loads(PB.SPLIT.read_text())["selection_idx"][:8]]
        eng.index([train[i] for i in pool]); top = PB.retriever(tr_pb, pool); pos = {t: j for j, t in enumerate(pool)}
        eng.bm = type("R", (), {"top": lambda self, text, k: [pos[t] for t in top(text)]})()
        C.CACHE = PB.OUT / "cache_smoke"; C.CACHE.mkdir(parents=True, exist_ok=True)
        Ls = member_Ls("smoke", rows, eng, opts); comb = C.combine(Ls, prm, "weighted_prob_mean")
        assert comb.shape == (8, PB.K) and np.allclose(comb.sum(1), 1)
        loo = LeaveOneOut([train[i]["text"] for i in pool]); eng.bm = loo
        Lp = member_Ls("smoke_pool", [train[i] for i in pool[:16]], eng, opts)
        assert Lp[0].shape == (16, PB.K) and all(j not in loo.top(train[pool[j]]["text"], 24) for j in range(16))
        print("SMOKE_OK (outputs discarded)", flush=True); return
    test = PB.test_rows(); gold = np.array([r["label"] for r in test]); OUTP.mkdir(parents=True, exist_ok=True)
    common = {"selection_file_sha256": sel_sha, "selected": win, "code_sha256": {
        f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        for f in M2_FILES + ["scripts/poolsize_open.py", "scripts/poolsize_bodies.py", "scripts/bitext24_bodies.py"]}}
    import torch
    common["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    for name, pool in pl.items():
        want_fit = name != "full"
        done_u = (OUTP / f"open_{name}.jsonl.gz").exists(); done_f = (OUTP / f"open_poolfit_{name}.jsonl.gz").exists()
        if done_u and (done_f or not want_fit):
            print(f"skip {name}: output exists (one scored run per cell)", flush=True); continue
        t0 = time.time(); C.CACHE = PB.OUT / "cache" / name; C.CACHE.mkdir(parents=True, exist_ok=True)
        _, man = PB.verified_bodies(name)
        eng.index([train[i] for i in pool]); eng.bm = Recorded(test, man, pool)
        Ls = member_Ls("test", test, eng, opts)
        ret = [[pool[j] for j in eng.shots_idx(r["text"])] for r in test]
        assert ret == [m["retrieved_t_idx"] for m in man]            # identical to the manifest by construction
        base = {**common, "pool_rows": len(pool), "manifest_sha256": json.loads(
            (PB.OUT / "manifest_meta.json").read_text())["pools"][name]["manifest_sha256"]}
        if not done_u:
            write_cell("open", name, test, gold, C.combine(Ls, prm, "weighted_prob_mean"), ret,
                       {**base, "fitted": prm_rec, "fitted_on": "selection split (unchanged clean refit)",
                        "seconds": round(time.time() - t0, 1)})
        if want_fit and not done_f:
            t1 = time.time(); rows = [train[i] for i in pool]; y = np.array([r["label"] for r in rows])
            eng.bm = LeaveOneOut([r["text"] for r in rows])
            fit = fit_on(member_Ls("pool_loo", rows, eng, opts), y)
            pf = {"T": [fit["temperatures"][n] for n in C.MEMBER_SETS["M2"]], "w": fit["weights"]}
            write_cell("open_poolfit", name, test, gold, C.combine(Ls, pf, "weighted_prob_mean"), ret,
                       {**base, "fitted": fit, "fitted_on": f"the pool's own {len(rows)} rows, leave-one-out retrieval",
                        "seconds": round(time.time() - t1, 1)})
    print("POOLSIZE_OPEN_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["run", "smoke"])
    a = ap.parse_args(); phase_run(smoke=a.phase == "smoke")
