#!/usr/bin/env python3
"""Pool size (Banking77): pools, recorded retrieval and decision-model request bodies, built once.

Implements `provenance/banking77_pool_size_protocol.md` §3–§4. CPU only; loads no model.

    python scripts/poolsize_bodies.py build     # results/poolsize/{pools.json, manifest_<pool>.jsonl.gz, manifest_meta.json}
    python scripts/poolsize_bodies.py verify    # rebuilds every body of every pool and checks every sha256

Library (used by every arm):
    pools()                 -> {pool name: sorted train indices}, in run order
    test_rows()             -> the 3,080 official test rows (pinned revision, loader order)
    verified_bodies(pool)   -> bodies, manifest rows       (rebuilt from the recorded indices, sha-checked)

The bodies are the Banking77 Arm-B / b24B body (`scripts/r14_clef.py` `armb_bodies`): the reproduction's
instruction and definitions (read at run time, never stored), with the recorded examples. This repo stores only
indices and hashes.
"""
import argparse, gzip, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROTOCOL = "provenance/banking77_pool_size_protocol.md"
DATASET, DATASET_REV = "legacy-datasets/banking77", "f54121560de48f2852f90be299010d1d6dc612ec"
K, SHOTS, BYTE_LIMIT, DRAW_SEED0, DRAWS = 77, 24, 30_000, 20261020, 5
OUT = ROOT / "results" / "poolsize"
TRAIN = ROOT / "docs" / "phases" / "2.0" / "jev" / "train_pinned.json"
SPLIT = ROOT / "results" / "clean" / "selection_split_idx.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def train_rows():
    """The pinned train split: [{'text', 'label' (id), 'label_name'}], checked against train_pinned.json."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=DATASET_REV, split="train")
    raw = list(ds.features["label"].names)
    rows = [{"text": r["text"], "label": int(r["label"]), "label_name": raw[int(r["label"])]} for r in ds]
    pin = json.loads(TRAIN.read_text())
    assert pin["revision"] == DATASET_REV and pin["text"] == [r["text"] for r in rows]
    assert pin["label_name"] == [r["label_name"] for r in rows] and len(rows) == 10003 and len(raw) == K
    return rows, raw


def test_rows():
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=DATASET_REV, split="test")
    rows = [{"text": r["text"], "label": int(r["label"])} for r in ds]
    assert len(rows) == 3080
    return rows


def pools(train=None):
    """§3: P = train minus the clean selection split; full + nested stratified draws at 24 and 8 per label."""
    train = train or train_rows()[0]
    S = set(json.loads(SPLIT.read_text())["selection_idx"])
    P = [i for i in range(len(train)) if i not in S]
    assert len(S) == 924 and len(P) == 9079
    lab = np.array([train[i]["label"] for i in P]); Pa = np.array(P)
    out = {"full": P}
    for d in range(1, DRAWS + 1):
        rng = np.random.default_rng(DRAW_SEED0 + d); p24, p8 = [], []
        for c in range(K):
            perm = [int(i) for i in rng.permutation(np.array(sorted(int(i) for i in Pa[lab == c])))]
            p24 += perm[:24]; p8 += perm[:8]
        out[f"pc24_d{d}"], out[f"pc8_d{d}"] = sorted(p24), sorted(p8)
    order = ["full"] + [f"pc24_d{d}" for d in range(1, DRAWS + 1)] + [f"pc8_d{d}" for d in range(1, DRAWS + 1)]
    return {k: out[k] for k in order}


def retriever(train, pool):
    B = _load("bitext24_bodies")
    bm = B.StableBM25([train[i]["text"] for i in pool])
    return lambda text: [pool[j] for j in bm.top(text, SHOTS)]


def body(rq, text, ex_idx, train):
    """`armb_bodies`: the reproduction's body with labeled_examples set to the recorded examples, trimmed to 30,000
    bytes. Built from the `definitions` variant (same instruction, criteria and key order, no retrieval pass); the
    bytes equal `armb_bodies` (checked: every Arm-B sha256 is reproduced from the clean run's retrieved_idx)."""
    b = json.loads(json.dumps(rq.body(text, "definitions")[0]))
    ex = [{"message": train[j]["text"], "intent": train[j]["label_name"]} for j in ex_idx]
    while True:
        b["state"]["labeled_examples"] = ex
        if len(json.dumps(b, ensure_ascii=True).encode()) <= BYTE_LIMIT or not ex:
            return b, len(ex)
        ex = ex[:-1]


def sha(b):
    return hashlib.sha256(json.dumps(b).encode()).hexdigest()     # the hash convention of Arm B / R16


def phase_build():
    train, raw = train_rows(); test = test_rows(); pl = pools(train)
    R14 = _load("r14_clef"); rq = R14.ReproRequests()
    assert rq.raw == raw
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest_meta.json").exists():
        sys.exit("refusing: the manifests exist; bodies are built once (protocol §4)")
    (OUT / "pools.json").write_text(json.dumps({"protocol": PROTOCOL, "pools": pl}))
    meta = {"protocol": PROTOCOL, "dataset": DATASET, "dataset_revision": DATASET_REV, "n_test": len(test),
            "pools_sha256": hashlib.sha256((OUT / "pools.json").read_bytes()).hexdigest(), "pools": {}}
    for name, pool in pl.items():
        t0 = time.time(); top = retriever(train, pool); lines = []
        for i, r in enumerate(test):
            idx = top(r["text"]); b, kept = body(rq, r["text"], idx, train)
            lines.append({"idx": i, "sha256": sha(b), "bytes": len(json.dumps(b, ensure_ascii=True).encode()),
                          "examples_kept": kept, "retrieved_t_idx": idx})
        content = "".join(json.dumps(x) + "\n" for x in lines).encode(); path = OUT / f"manifest_{name}.jsonl.gz"
        with open(path, "wb") as raw_f, gzip.GzipFile(fileobj=raw_f, mode="wb", mtime=0, filename="") as f:
            f.write(content)                                       # mtime 0: the file bytes are reproducible
        kept = [x["examples_kept"] for x in lines]; byt = [x["bytes"] for x in lines]
        per_label = np.bincount([train[i]["label"] for i in pool], minlength=K)
        meta["pools"][name] = {"rows": len(pool), "per_label_min": int(per_label.min()), "per_label_max": int(per_label.max()),
                               "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "examples_kept": {"min": min(kept), "all_24": sum(k == SHOTS for k in kept)},
                               "bytes": {"min": min(byt), "median": int(np.median(byt)), "max": max(byt)},
                               "seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(meta["pools"][name]), flush=True)
    meta["code_sha256"] = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                           for f in ("scripts/poolsize_bodies.py", "scripts/bitext24_bodies.py", "scripts/r14_clef.py",
                                     "quorum/fewshot.py")}
    meta["instruction"] = "the reproduction's (read at run time, not stored)"
    meta["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "manifest_meta.json").write_text(json.dumps(meta, indent=1))
    print("POOLSIZE_BODIES_OK", flush=True)


_CACHE = {}


def verified_bodies(name):
    """Rebuild every body of one pool from the recorded indices and check its sha256 against the committed manifest."""
    if not _CACHE:
        _CACHE["train"] = train_rows()[0]; _CACHE["test"] = test_rows()
        _CACHE["rq"] = _load("r14_clef").ReproRequests()
        _CACHE["meta"] = json.loads((OUT / "manifest_meta.json").read_text())
        assert hashlib.sha256((OUT / "pools.json").read_bytes()).hexdigest() == _CACHE["meta"]["pools_sha256"]
    train, test, rq, meta = _CACHE["train"], _CACHE["test"], _CACHE["rq"], _CACHE["meta"]
    path = OUT / f"manifest_{name}.jsonl.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["pools"][name]["manifest_sha256"], "manifest changed"
    pool = set(json.loads((OUT / "pools.json").read_text())["pools"][name])
    man = [json.loads(l) for l in gzip.open(path, "rt")]
    assert len(man) == len(test)
    out = []
    for m, r in zip(man, test):
        assert set(m["retrieved_t_idx"]) <= pool
        b, kept = body(rq, r["text"], m["retrieved_t_idx"], train)
        if sha(b) != m["sha256"] or kept != m["examples_kept"]:
            raise SystemExit(f"{name} body {m['idx']} differs from the manifest; aborting")
        out.append(b)
    return out, man


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["build", "verify"])
    a = ap.parse_args()
    if a.phase == "build":
        phase_build()
    else:
        names = json.loads((OUT / "pools.json").read_text())["pools"]
        for n in names:
            verified_bodies(n); print(f"VERIFY_OK {n}", flush=True)
