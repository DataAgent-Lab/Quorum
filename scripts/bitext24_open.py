#!/usr/bin/env python3
"""R16 Bitext 24-shot, the open stack: clean selection on S and ONE test run, as pre-registered in
`provenance/bitext_24shot_protocol.md` §2–§7 + amendment A1.

    python scripts/bitext24_open.py split     # writes results/bitext24/selection_split_idx.json; loads no model
    python scripts/bitext24_open.py smoke     # pipeline check on 8 pool rows not in S; output discarded
    python scripts/bitext24_open.py select    # 48-candidate selection on S (logits cached)
    python scripts/bitext24_open.py test      # refuses unless selection is written and the body manifest exists

It reuses the Banking77 clean run's code (`scripts/clean_24shot.py`, imported, never edited, so the Banking77 result
keeps its code fingerprint). It replaces only the dataset, the split and the descriptions. The test run takes each
test row's retrieval from the R16 body manifest (A1), so it sees exactly the examples every decision model sees.
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
B = _load("bitext24_bodies")                                       # R16 split, descriptions, manifest

C.PROTOCOL_COMMIT = B.PROTOCOL_COMMIT
C.DATASET, C.DATASET_REV = B.DATASET, B.DATASET_REV
C.TASK_DESC = "Classify the customer's support message into the intent it expresses."
C.SPLIT_SEED, C.FOLD_SEED, C.PER_CLASS, C.K = B.SPLIT_SEED_S, 20261013, B.PER_CLASS, B.K
C.OUT = ROOT / "results" / "bitext24"; C.CACHE = C.OUT / "cache"
_SPLIT = {}


def _split():
    if not _SPLIT:
        _SPLIT["v"] = B.split()
    return _SPLIT["v"]


def load_train():
    raw, names, T, S, P, test = _split()
    return None, raw, names, T


def make_split(train):
    raw, names, T, S, P, test = _split()
    assert train is T
    return S, []                                                   # no earlier selection split for Bitext


def descriptions(raw):
    return B.descriptions(_split()[1])


def code_fingerprint():
    files = ["scripts/bitext24_open.py", "scripts/bitext24_bodies.py", "scripts/clean_24shot.py", "quorum/fewshot.py",
             "quorum/prompting.py", "quorum/ensemble.py", "quorum/data.py", "examples/bitext_descriptions.py"]
    fp = C.__dict__["_orig_code_fingerprint"]()
    fp["sha256"] = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    return fp


C._orig_code_fingerprint = C.code_fingerprint
C.load_train, C.make_split, C.descriptions, C.code_fingerprint = load_train, make_split, descriptions, code_fingerprint


class RecordedRetrieval:
    """Returns the manifest's recorded top-24 for test rows (A1), as positions in the indexed pool P."""

    def __init__(self, test, man, P):
        pos = {t: j for j, t in enumerate(P)}
        self.by_text = {}
        for r, m in zip(test, man):
            rec = [pos[t] for t in m["retrieved_t_idx"]]
            assert self.by_text.setdefault(r["text"], rec) == rec, "same text, different recorded retrieval"

    def top(self, text, k):
        rec = self.by_text[text]
        assert k == len(rec) == B.SHOTS
        return list(rec)


def phase_test(args):
    sel_path = C.OUT / "selection_scores.json"
    if not sel_path.exists():
        sys.exit("refusing: selection_scores.json not written; run `select` first")
    OUTP = ROOT / "results" / "predictions"
    dump_path, meta_path = OUTP / "bitext_24shot_clean.jsonl.gz", OUTP / "bitext_24shot_clean_meta.json"
    if (dump_path.exists() or meta_path.exists()) and not args.rerun_after_crash:
        sys.exit("refusing: the single test run already produced output; --rerun-after-crash REASON only after a crash")
    sel_bytes = sel_path.read_bytes(); sel = json.loads(sel_bytes)
    assert sel["protocol_commit"] == C.PROTOCOL_COMMIT
    assert sel["code"]["sha256"] == code_fingerprint()["sha256"], "code differs from the code that made the selection"
    win = sel["selected"]; prm_rec = sel["refit_on_full_selection_split"]
    S = C.load_split(); raw, T, P, opts, eng, zs = C.setup(S)
    test, raw_b, bodies, man = B.verified_bodies()                 # test rows loaded only now; manifest checked
    assert raw_b == raw
    eng.bm = RecordedRetrieval(test, man, P)                       # A1: the recorded retrieval, nothing re-ranked
    t0 = time.time(); ms, text, k, rule = win["member_set"], win["option_text"], win["orders"], win["rule"]
    plen_test = C.length_stats(eng, test, opts, text, k)
    m = C.member_logits("test", test, eng, zs, opts, text, k, ms == "M3")
    Ls = C.cand_logits(m, ms, k)
    prm = {"T": [prm_rec["temperatures"][n] for n in C.MEMBER_SETS[ms]], "w": prm_rec["weights"]}
    comb_raw = C.combine(Ls, prm, rule)
    comb = comb_raw / comb_raw.sum(1, keepdims=True) if rule == "rank_mean" else comb_raw
    Pf = C.softmax(np.log(comb_raw + 1e-12) if rule != "rank_mean" else comb_raw, prm_rec["final_temperature"])
    g7 = lambda v: [float(f"{x:.7g}") for x in v]
    cal = [C.softmax(L, T_) for L, T_ in zip(Ls, prm["T"])]
    OUTP.mkdir(parents=True, exist_ok=True)
    preds, gold = [], []
    with gzip.open(dump_path, "wt") as f:
        for i, r in enumerate(test):
            pred = int(np.argmax(comb[i])); preds.append(pred); gold.append(r["label"])
            ret = [P[j] for j in eng.shots_idx(r["text"])]
            assert ret == man[i]["retrieved_t_idx"]                # A1 §5: identical to the manifest by construction
            row = {"idx": i, "text": r["text"], "gold": r["label"], "pred": pred, "probs": g7(comb[i]),
                   "reader_probs": g7(cal[0][i]), "knn_probs": g7(cal[1][i]), "retrieved_idx": ret,
                   "perm_reader_probs": [g7(C.softmax(m["reader_orders"][pi][i:i + 1])[0]) for pi in range(k)]}
            if ms == "M3":
                row["zs_probs"] = g7(cal[2][i])
            f.write(json.dumps(row) + "\n")
    acc = float((np.array(preds) == np.array(gold)).mean())
    import torch, transformers, peft, sentence_transformers
    meta = {"protocol_commit": C.PROTOCOL_COMMIT, "amendment_A1": "3d84cc3a4f13ce52355d46ae3b76134a0edf8ada",
            "code": code_fingerprint(), "selection_file_sha256": hashlib.sha256(sel_bytes).hexdigest(),
            "manifest_sha256": json.loads((C.OUT / "manifest_meta.json").read_text())["manifest_sha256"],
            "rerun": bool(args.rerun_after_crash), "rerun_reason": args.rerun_after_crash or None,
            "dataset": C.DATASET, "dataset_revision": C.DATASET_REV, "base": C.BASE, "base_revision": C.BASE_REV,
            "adapter": C.ADAPTER, "adapter_revision": C.ADAPTER_REV, "knn_embedder": C.KNN_EMBEDDER,
            "task_description": C.TASK_DESC, "selected": win, "fitted": prm_rec, "pool_size": len(P), "n_test": len(test),
            "accuracy": round(acc, 4), "correct": int(sum(np.array(preds) == np.array(gold))),
            "secondary": {"calibrated_final": C.secondary_metrics(Pf, preds, gold),
                          "uncalibrated_combined": C.secondary_metrics(comb, preds, gold)},
            "prompt_lengths_test": plen_test, "zero_shot_nli_dtype": "fp32", "max_prompt_tokens": C.MAX_PROMPT_TOKENS,
            "truncation": False, "revisions": C.resolved_revisions(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__, "sentence_transformers": sentence_transformers.__version__},
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "wall_seconds": round(time.time() - t0, 1), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"TEST: accuracy={acc:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["split", "smoke", "select", "test"])
    ap.add_argument("--rerun-after-crash", metavar="REASON", default=None)
    a = ap.parse_args()
    if a.phase == "test":
        phase_test(a)
    elif a.phase == "smoke":                                       # pool rows not in S only (§10)
        S = C.load_split(); raw, T, P, opts, eng, zs = C.setup(S)
        rows = [T[i] for i in P[:8]]; eng.options = list(opts["names"])
        r = eng.incontext_logits_per_perm(rows, 1, bs=C.BS)[0]; kn = eng._knn_logits(rows)
        assert r.shape == (8, C.K) and kn.shape == (8, C.K)
        print("SMOKE_OK (outputs discarded)", flush=True)
    else:
        {"split": C.phase_split, "select": C.phase_select}[a.phase]()
