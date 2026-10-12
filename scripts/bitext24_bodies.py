#!/usr/bin/env python3
"""R16 Bitext 24-shot: build the decision-model request bodies once and record them in a sha256 manifest.

Implements `provenance/bitext_24shot_protocol.md` §1, §2, §8 and amendment A1. Runs on CPU only and loads no model.

    python scripts/bitext24_bodies.py build     # writes results/bitext24/{selection_split_idx.json,
                                                 #   request_manifest.jsonl.gz, manifest_meta.json}
    python scripts/bitext24_bodies.py verify    # rebuilds every body from the manifest and checks every sha256

Library (used by every arm):
    split()            -> raw, names, T, S, P, test           (pinned data + the protocol's split)
    verified_bodies()  -> test, raw, bodies, manifest rows     (rebuilt from the recorded indices, sha-checked)

The bodies are rebuilt from the recorded retrieval, never re-retrieved (A1). This repo stores only hashes and
indices; the dataset's own text is rebuilt from the pinned revision.
"""
import argparse, gzip, hashlib, importlib.util, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROTOCOL_COMMIT = "9d6dc44ef099a555d9b7bbd0b1778c46105a25f7"           # pre-registration; A1 = 3d84cc3
DATASET, DATASET_REV = "Bitext/Bitext-customer-support-llm-chatbot-training-dataset", "430d1a89bd93bd1fa23c16f29dd53e73f0087443"
SPLIT_SEED_TEST, SPLIT_SEED_S, PER_CLASS, K, SHOTS = 20250926, 20261012, 12, 27, 24
BYTE_LIMIT = 30000
INSTRUCTION = ("Which ONE intent best describes the customer message? The labeled examples show messages with their "
               "intents. Every message belongs to exactly one of the listed intents.")
OUT = ROOT / "results" / "bitext24"


def split():
    """The loader's split (quorum/data.py `bitext`) on the pinned revision, then the protocol's S and P."""
    from datasets import load_dataset
    from quorum import data
    ds = load_dataset(DATASET, revision=DATASET_REV)["train"]
    u, mp = data._index(ds["intent"])
    rows = [{"text": r["instruction"], "label": mp[r["intent"]]} for r in ds]
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["label"]].append(i)
    rng = np.random.default_rng(SPLIT_SEED_TEST)
    test_i, pool_i = [], []
    for lab in sorted(by):                                         # = quorum/data.py, bitext branch
        idx = np.array(by[lab]); rng.shuffle(idx); n = max(1, round(0.2 * len(idx)))
        test_i += idx[:n].tolist(); pool_i += idx[n:].tolist()
    T, test = [rows[i] for i in pool_i], [rows[i] for i in test_i]
    ref = data.load("bitext")                                       # the repo loader must agree row for row
    assert ref["raw_labels"] == u and ref["train"] == T and ref["test"] == test, "loader split differs"
    assert len(rows) == 26872 and len(u) == K
    labels = np.array([r["label"] for r in T]); rng = np.random.default_rng(SPLIT_SEED_S); S = []
    for c in range(K):                                              # protocol §2
        elig = np.array(sorted(int(i) for i in np.flatnonzero(labels == c)))
        S += [int(i) for i in rng.permutation(elig)[:PER_CLASS]]
    S = sorted(S); Sset = set(S)
    assert len(S) == K * PER_CLASS
    P = [i for i in range(len(T)) if i not in Sset]
    return u, ref["labels"], T, S, P, test


def descriptions(names):
    spec = importlib.util.spec_from_file_location("bxd", ROOT / "examples" / "bitext_descriptions.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return [m.BITEXT_DESCRIPTIONS[n] for n in names]


class StableBM25:
    """The scores of quorum/fewshot.py BM25 (same tokenizer, k1, b, idf); ties broken by a stable sort (A1)."""

    def __init__(self, docs):
        from quorum.fewshot import BM25
        self.bm = BM25(docs)

    def top(self, query, k):
        bm = self.bm; scores = np.zeros(bm.N)
        from quorum.fewshot import _TOK
        for w in sorted(set(_TOK.findall(query.lower()))):     # fixed order: deterministic float sums
            idf = bm.idf.get(w)
            if idf is None:
                continue
            for i, c in enumerate(bm.tf):
                f = c.get(w, 0)
                if f:
                    dl = len(bm.docs[i])
                    scores[i] += idf * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * dl / bm.avgdl))
        return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


def body(text, examples, raw, desc):
    b = {"model": "jev-1.13.0",
         "state": {"customer_message": text, "labeled_examples": []},
         "questions": {"intent": {"type": "choice", "instructions": INSTRUCTION,
                                  "criteria": {r: d for r, d in sorted(zip(raw, desc))}}}}
    ex = list(examples)
    while True:
        b["state"]["labeled_examples"] = ex
        if len(json.dumps(b, ensure_ascii=True).encode()) <= BYTE_LIMIT or not ex:
            return b, len(ex)
        ex = ex[:-1]


def sha(b):
    return hashlib.sha256(json.dumps(b).encode()).hexdigest()       # the hash convention of the Banking77 Arm B


def phase_build():
    raw, names, T, S, P, test = split(); desc = descriptions(names)
    OUT.mkdir(parents=True, exist_ok=True)
    sel = OUT / "selection_split_idx.json"
    rec = {"protocol_commit": PROTOCOL_COMMIT, "dataset": DATASET, "dataset_revision": DATASET_REV,
           "selection_idx": S, "n": len(S), "per_class": PER_CLASS, "split_seed": SPLIT_SEED_S, "pool_size": len(P)}
    if sel.exists():
        assert json.loads(sel.read_text())["selection_idx"] == S, "selection split differs from the written one"
    else:
        sel.write_text(json.dumps(rec | {"written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    man = OUT / "request_manifest.jsonl.gz"
    if man.exists():
        sys.exit("refusing: the manifest exists; bodies are built once (protocol §8)")
    bm = StableBM25([T[i]["text"] for i in P]); t0 = time.time(); lines = []
    for i, r in enumerate(test):
        top = [P[j] for j in bm.top(r["text"], SHOTS)]
        ex = [{"message": T[j]["text"], "intent": raw[T[j]["label"]]} for j in top]
        b, kept = body(r["text"], ex, raw, desc)
        lines.append({"idx": i, "sha256": sha(b), "bytes": len(json.dumps(b, ensure_ascii=True).encode()),
                      "examples_kept": kept, "retrieved_t_idx": top})
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(test)} bodies", flush=True)
    content = "".join(json.dumps(x) + "\n" for x in lines).encode()
    with open(man, "wb") as raw_f, gzip.GzipFile(fileobj=raw_f, mode="wb", mtime=0, filename="") as f:
        f.write(content)                                           # mtime 0: the file bytes are reproducible
    kept = [x["examples_kept"] for x in lines]; byt = [x["bytes"] for x in lines]
    meta = {"protocol_commit": PROTOCOL_COMMIT, "amendment_A1": "3d84cc3a4f13ce52355d46ae3b76134a0edf8ada",
            "dataset": DATASET, "dataset_revision": DATASET_REV, "n_test": len(test), "pool_size": len(P),
            "selection_size": len(S), "instruction": INSTRUCTION, "byte_limit": BYTE_LIMIT,
            "examples_kept": {"min": min(kept), "median": int(np.median(kept)), "all_24": sum(k == 24 for k in kept)},
            "bytes": {"min": min(byt), "median": int(np.median(byt)), "max": max(byt)},
            "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest(),
            "manifest_content_sha256": hashlib.sha256(content).hexdigest(),
            "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                            for f in ("scripts/bitext24_bodies.py", "quorum/fewshot.py", "quorum/data.py",
                                      "examples/bitext_descriptions.py")},
            "seconds": round(time.time() - t0, 1), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "manifest_meta.json").write_text(json.dumps(meta, indent=1))
    print("BODIES_OK", json.dumps({k: meta[k] for k in ("n_test", "examples_kept", "bytes", "manifest_sha256")}), flush=True)


def verified_bodies():
    """Rebuild every body from the recorded indices and check its sha256 against the committed manifest."""
    raw, names, T, S, P, test = split(); desc = descriptions(names)
    meta = json.loads((OUT / "manifest_meta.json").read_text())
    man_path = OUT / "request_manifest.jsonl.gz"
    assert hashlib.sha256(man_path.read_bytes()).hexdigest() == meta["manifest_sha256"], "manifest file changed"
    man = [json.loads(l) for l in gzip.open(man_path, "rt")]
    assert len(man) == len(test)
    out = []
    for m, r in zip(man, test):
        ex = [{"message": T[j]["text"], "intent": raw[T[j]["label"]]} for j in m["retrieved_t_idx"]]
        b, kept = body(r["text"], ex, raw, desc)
        if sha(b) != m["sha256"] or kept != m["examples_kept"]:
            raise SystemExit(f"body {m['idx']} differs from the manifest; aborting")
        out.append(b)
    return test, raw, out, man


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["build", "verify"])
    a = ap.parse_args()
    if a.phase == "build":
        phase_build()
    else:
        test, raw, bodies, man = verified_bodies(); print(f"VERIFY_OK {len(bodies)} bodies match the manifest", flush=True)
