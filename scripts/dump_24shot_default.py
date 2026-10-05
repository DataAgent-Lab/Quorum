#!/usr/bin/env python3
"""R2-default: per-item dump of this repo's DEFAULT 24-shot configuration (scripts/eval_24shot.py).

Configuration (unchanged from the default): class names as options; reader + bge-large kNN over the same 24
BM25-retrieved examples; geometric mean of the two members at T=1 (no calibration); 3 option orders; retrieval
from the full 10,003-row train set; library-default tokenisation (right-truncation at 2,048 tokens). Revisions
are pinned so the run is reproducible. Prompts longer than 2,048 tokens (i.e. truncated rows) are counted and
reported, as promised in the clean-protocol addendum (A1).

    python scripts/dump_24shot_default.py

Writes results/predictions/banking77_24shot_default.jsonl.gz (+ _meta.json), same per-item schema as the
clean run. Per-order logits are cached under results/default/cache/ so an interrupted run resumes.
"""
import csv, gzip, hashlib, importlib.util, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("clean24", ROOT / "scripts" / "clean_24shot.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)      # shared pins + helpers

PERMS, DEFAULT_MAX_LEN = 3, 2048
CACHE = ROOT / "results" / "default" / "cache"
OUTP = ROOT / "results" / "predictions"


def fingerprint():
    files = ["scripts/dump_24shot_default.py", "scripts/clean_24shot.py", "quorum/fewshot.py",
             "quorum/prompting.py", "quorum/data.py"]
    head = C._read_git_head()
    return {"git_head": head, "sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}


def main():
    from datasets import load_dataset
    from quorum.data import _humanize
    from quorum.fewshot import RetrievalEnsemble
    tr = load_dataset(C.DATASET, revision=C.DATASET_REV, split="train")
    raw = list(tr.features["label"].names); names = [_humanize(n) for n in raw]
    train = [{"text": r["text"], "label": int(r["label"])} for r in tr]
    test = [{"text": r["text"], "label": int(r["label"])}
            for r in load_dataset(C.DATASET, revision=C.DATASET_REV, split="test")]
    assert len(train) == 10003 and len(test) == 3080

    eng = RetrievalEnsemble(names, C.TASK_DESC, base=C.BASE, adapter=C.ADAPTER, embedder=C.KNN_EMBEDDER,
                            shots=24, perms=PERMS, device="cuda", base_revision=C.BASE_REV,
                            adapter_revision=C.ADAPTER_REV)                  # library default: truncate at 2,048
    eng.index(train)
    L = eng.prompt_lengths(test, PERMS)
    plen = {"n_prompts": int(L.size), "mean": round(float(L.mean()), 1), "p50": int(np.percentile(L, 50)),
            "p95": int(np.percentile(L, 95)), "max": int(L.max()),
            "over_2048_truncated_prompts": int((L > DEFAULT_MAX_LEN).sum()),
            "rows_with_any_truncated_order": int((L > DEFAULT_MAX_LEN).any(0).sum())}
    print("prompt lengths (default, test):", json.dumps(plen), flush=True)

    CACHE.mkdir(parents=True, exist_ok=True); t0 = time.time()
    per = np.stack([C.cached(CACHE / f"test_reader_names_o{pi}.npy",
                             lambda pi=pi: eng.incontext_logits_per_perm(test, 1, bs=C.BS, perm_offset=pi)[0])
                    for pi in range(PERMS)])
    knn = C.cached(CACHE / "test_knn.npy", lambda: eng._knn_logits(test))
    reader_p, knn_p = C.softmax(per.mean(0)), C.softmax(knn)
    g = np.exp(np.mean([np.log(reader_p + 1e-12), np.log(knn_p + 1e-12)], 0)); probs = g / g.sum(1, keepdims=True)

    g7 = lambda v: [float(f"{x:.7g}") for x in v]
    OUTP.mkdir(parents=True, exist_ok=True); preds, gold = [], []
    with gzip.open(OUTP / "banking77_24shot_default.jsonl.gz", "wt") as f:
        for i, r in enumerate(test):
            pred = int(np.argmax(probs[i])); preds.append(pred); gold.append(r["label"])
            f.write(json.dumps({"idx": i, "text": r["text"], "gold": r["label"], "pred": pred, "probs": g7(probs[i]),
                                "reader_probs": g7(reader_p[i]), "knn_probs": g7(knn_p[i]),
                                "retrieved_idx": [int(j) for j in eng.shots_idx(r["text"])],
                                "perm_reader_probs": [g7(C.softmax(per[pi][i:i + 1])[0]) for pi in range(PERMS)]}) + "\n")
    hit = np.array(preds) == np.array(gold); acc = float(hit.mean()); secs = time.time() - t0

    mc = {"source": C.REPRO_URL}
    try:
        txt = urllib.request.urlopen(C.REPRO_URL, timeout=60).read().decode()
        rep = {int(row["id"].split("-")[1]): row for row in csv.DictReader(io.StringIO(txt))}
        assert len(rep) == 3080 and all(rep[i]["truth"] == raw[gold[i]] for i in range(3080))
        theirs = np.array([rep[i]["correct"] == "True" for i in range(3080)])
        b, c = int((hit & ~theirs).sum()), int((~hit & theirs).sum())
        mc.update({"b_ours_right_theirs_wrong": b, "c_ours_wrong_theirs_right": c, "p_two_sided_exact": C.mcnemar_exact(b, c)})
    except Exception as e:
        mc["error"] = f"{type(e).__name__}: {e}"

    import torch, transformers, peft, sentence_transformers
    meta = {"configuration": "repo default (scripts/eval_24shot.py)", "code": fingerprint(),
            "dataset": C.DATASET, "dataset_revision": C.DATASET_REV, "base": C.BASE, "base_revision": C.BASE_REV,
            "adapter": C.ADAPTER, "adapter_revision": C.ADAPTER_REV, "knn_embedder": C.KNN_EMBEDDER,
            "option_text": "class names (humanized)", "task_description": C.TASK_DESC,
            "retrieval": {"method": "BM25 k1=1.5 b=0.75", "shots": 24, "pool": "full train (10,003 rows)"},
            "knn": {"scale": 20, "floor": -1, "aggregation": "max similarity per class"},
            "orders": PERMS, "order_seeds": [1000 + i for i in range(PERMS)], "combination": "geometric mean, T=1",
            "tokenisation": {"truncate": True, "max_length": DEFAULT_MAX_LEN, "side": "right"},
            "prompt_lengths_test": plen, "n_test": 3080, "accuracy": round(acc, 4), "correct": int(hit.sum()),
            "published_default_accuracy": 0.9321, "mcnemar_vs_reproduction": mc,
            "revisions": C.resolved_revisions(),
            "fields": {"probs": "normalised geometric mean of reader_probs and knn_probs",
                       "reader_probs": "softmax(mean over orders of reader logits), T=1",
                       "knn_probs": "softmax(kNN logits), T=1",
                       "perm_reader_probs": "softmax(per-order reader logits), T=1",
                       "retrieved_idx": "indices into the pinned train split"},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__, "sentence_transformers": sentence_transformers.__version__},
            "device": torch.cuda.get_device_name(0), "wall_seconds": round(secs, 1),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUTP / "banking77_24shot_default_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"DEFAULT: accuracy={acc:.4f} (published 0.9321) mcnemar={json.dumps(mc)}", flush=True)


if __name__ == "__main__":
    main()
