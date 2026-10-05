#!/usr/bin/env python3
"""R12 — train-side outputs of the zero-shot ensemble, for fitting ONE global temperature without test data.

    python scripts/r12_calibration.py ids    # writes results/r12/sample_ids.json; loads no model (pre-registration)
    python scripts/r12_calibration.py run    # refuses without the ids file; dumps full distributions

Sample: for each of the 7 datasets, 1,000 indices drawn uniformly without replacement from the dataset's TRAIN
side (for Bitext, the train part of this repo's deterministic 80/20 split), with
np.random.default_rng(R12_SEED + k) for the k-th dataset in the fixed order below. Indices refer to
`quorum.data.load(name)["train"]`. Configuration: the repo's ZeroShotEnsemble defaults (commit 04e6709) with the
NLI member in fp32, both label texts (class names; examples/<dataset>_descriptions.py). No test data is loaded.
"""
import argparse, gzip, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASETS = ["banking77", "clinc150", "hwu64", "massive", "mtop", "snips", "bitext"]
DESC_MODULE = {"clinc150": "clinc"}
R12_SEED, N_SAMPLE = 20261008, 1000
OUT = ROOT / "results" / "r12"


def ids_phase():
    from quorum import data
    rec = {"seed": R12_SEED, "n_per_dataset": N_SAMPLE, "rule": "default_rng(seed + k).choice(len(train), n, "
           "replace=False), sorted; k = dataset position in `datasets`", "datasets": DATASETS, "ids": {},
           "train_sizes": {}, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for k, ds in enumerate(DATASETS):
        n = len(data.load(ds)["train"]); rec["train_sizes"][ds] = n
        rec["ids"][ds] = sorted(int(i) for i in np.random.default_rng(R12_SEED + k).choice(n, N_SAMPLE, replace=False))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample_ids.json").write_text(json.dumps(rec, indent=1))
    print("R12 ids written:", rec["train_sizes"], flush=True)


def load_desc(ds, labels):
    mod = DESC_MODULE.get(ds, ds)
    spec = importlib.util.spec_from_file_location(mod, ROOT / "examples" / f"{mod}_descriptions.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    d = next(v for k, v in vars(m).items() if k.endswith("_DESCRIPTIONS") and isinstance(v, dict))
    norm = {str(k).replace("_", " ").strip().lower(): v for k, v in d.items()}
    return [norm[str(l).replace("_", " ").strip().lower()] for l in labels]


def run_phase():
    p = OUT / "sample_ids.json"
    if not p.exists():
        sys.exit("refusing: results/r12/sample_ids.json missing — run `ids` and commit it first")
    rec = json.loads(p.read_text()); assert rec["seed"] == R12_SEED
    from quorum import ZeroShotEnsemble, data
    clf = ZeroShotEnsemble(device="cuda"); clf.members[0].model = clf.members[0].model.float()
    g7 = lambda v: [float(f"{x:.7g}") for x in v]; summary = {}
    for ds in DATASETS:
        d = data.load(ds); labels = list(d["labels"]); train = d["train"]; ids = list(range(len(labels)))
        assert len(train) == rec["train_sizes"][ds], "train side changed since the ids were drawn"
        for variant, lt in [("names", labels), ("descriptions", load_desc(ds, labels))]:
            path = OUT / f"{ds}_{variant}_train1000.jsonl.gz"; correct = 0
            with gzip.open(path, "wt") as f:
                for i in rec["ids"][ds]:
                    r = train[i]; mp, ep = clf.member_and_ensemble_proba(r["text"], ids, lt)
                    pred = int(np.argmax(ep)); correct += int(pred == r["label"])
                    f.write(json.dumps({"train_idx": i, "text": r["text"], "gold": int(r["label"]), "pred": pred,
                                        "probs": g7(ep), "member_probs": [g7(m) for m in mp]}) + "\n")
            summary[f"{ds}/{variant}"] = round(correct / len(rec["ids"][ds]), 4)
            print(f"[{ds}/{variant}] train-sample accuracy {summary[f'{ds}/{variant}']}", flush=True)
    import torch, transformers
    meta = {"sample_ids_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "seed": R12_SEED,
            "config": "ZeroShotEnsemble defaults (PrismNLI-0.4B fp32 + bge-large + bge-base, T=1, geometric mean)",
            "members": clf.model_names, "train_sample_accuracy": summary,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "device": torch.cuda.get_device_name(0), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1)); print("R12_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["ids", "run"])
    {"ids": ids_phase, "run": run_phase}[ap.parse_args().phase]()
