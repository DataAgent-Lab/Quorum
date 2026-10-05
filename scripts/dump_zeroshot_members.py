#!/usr/bin/env python3
"""R3 — per-item zero-shot dump for all 7 datasets x {class names, descriptions}, with per-member distributions.

    python scripts/dump_zeroshot_members.py [--out results/predictions]

Schema per line: {idx, text, gold, pred, probs[K] (6 dp), member_picks[3], member_probs[3][K] (7 significant digits)}.
`probs` is the ensemble distribution (per-member softmax at T=1, geometric mean, renormalised); `pred` is the argmax of
the stored `probs`; `member_probs` are the three members' own softmax(T=1) distributions (PrismNLI, bge-large,
bge-base), stored with relative precision so small probabilities are not rounded to 0. One {dataset}_meta.json per
dataset records the configuration. The NLI member is forced to fp32 (the CPU serving path). The script is idempotent:
a complete output file is skipped, and files are written to .tmp and renamed only when complete.

Provenance: the committed results/predictions/{dataset}_{variant}.jsonl.gz files were produced by this script at
repo commit 04e6709; the executed copy differed only in its absolute path constants (sha256 of the executed copy:
488d1c732c392cde916a1bfbcaea7d39a575a024309eff245cb813bb2b9778f2). Their `idx`, `text`, `gold`, `pred`, `probs` and
`member_picks` fields are identical to the earlier dumps without `member_probs`; the run added `member_probs` only.
"""
import argparse, gzip, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GIT_COMMIT = "04e6709"
DATASETS = ["banking77", "clinc150", "hwu64", "massive", "mtop", "snips", "bitext"]
DESC_MODULE = {"clinc150": "clinc"}  # else: module name == dataset name


def load_desc(ds):
    mod_name = DESC_MODULE.get(ds, ds)
    spec = importlib.util.spec_from_file_location(f"{mod_name}_desc", ROOT / "examples" / f"{mod_name}_descriptions.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    d = next(v for k, v in vars(m).items() if k.endswith("_DESCRIPTIONS") and isinstance(v, dict))
    return {k.replace("_", " ").strip().lower(): v for k, v in d.items()}


def complete(path, n):
    if not os.path.exists(path):
        return False
    try:
        with gzip.open(path, "rt") as f:
            return sum(1 for _ in f) == n
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results" / "predictions"))
    out = ap.parse_args().out; os.makedirs(out, exist_ok=True)
    import torch, transformers
    from quorum import ZeroShotEnsemble, data
    from quorum.ensemble import TEMPLATE, COSINE_SCALE

    # ---- pre-flight: every dataset's descriptions must cover its labels (fail fast) ----
    pre = {}
    for ds in DATASETS:
        d = data.load(ds); labels = list(d["labels"])
        dirty = [l for l in labels if l != l.strip()]
        assert not dirty, f"[{ds}] labels have surrounding whitespace: {dirty[:5]}"
        assert len(labels) == len(set(labels)), f"[{ds}] duplicate labels present"
        norm = load_desc(ds)
        missing = [l for l in labels if l.replace("_", " ").strip().lower() not in norm]
        assert not missing, f"[{ds}] descriptions missing {len(missing)}: {missing[:8]}"
        pre[ds] = (d, labels, norm)
    print("PREFLIGHT_OK: all 7 datasets have full description coverage", flush=True)

    clf = ZeroShotEnsemble(device="cuda")
    clf.members[0].model = clf.members[0].model.float()   # fp32 NLI == cpu serve path
    assert next(clf.members[0].model.parameters()).dtype == torch.float32
    config = {"members": clf.model_names, "template": TEMPLATE, "cosine_scale": COSINE_SCALE,
              "combine": "per-member softmax(T=1) -> geometric mean (logprob-mean) -> renormalise",
              "quorum_git_commit": GIT_COMMIT, "torch_version": torch.__version__,
              "transformers_version": transformers.__version__, "device": "cuda (fp32 NLI == cpu fp32)",
              "nli_dtype": "fp32"}

    def run(ds, variant, labels, label_texts, test):
        path = f"{out}/{ds}_{variant}.jsonl.gz"
        if complete(path, len(test)):
            gold, pred = [], []
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line); gold.append(r["gold"]); pred.append(r["pred"])
            acc = float((np.array(pred) == np.array(gold)).mean())
            print(f"  [{ds}/{variant}] SKIP (complete) acc={acc:.4f}", flush=True)
            return acc
        ids = list(range(len(labels))); preds, gold = [], []; t0 = time.time()
        with gzip.open(path + ".tmp", "wt") as f:
            for i, r in enumerate(test):
                mprobs, ep = clf.member_and_ensemble_proba(r["text"], ids, label_texts)
                probs = [round(float(x), 6) for x in ep]          # store 6dp
                p = int(np.argmax(probs))                          # pred == argmax(stored probs), by construction
                preds.append(p); gold.append(int(r["label"]))
                f.write(json.dumps({"idx": i, "text": r["text"], "gold": int(r["label"]), "pred": p,
                                    "probs": probs,
                                    "member_picks": [int(np.argmax(m)) for m in mprobs],
                                    "member_probs": [[float(f"{x:.7g}") for x in m] for m in mprobs]}) + "\n")
                if (i + 1) % 1000 == 0:
                    print(f"    [{ds}/{variant}] {i+1}/{len(test)}", flush=True)
        os.replace(path + ".tmp", path)   # atomic: only a complete file appears
        acc = float((np.array(preds) == np.array(gold)).mean())
        print(f"  [{ds}/{variant}] DONE acc={acc:.4f} {round(time.time()-t0)}s", flush=True)
        return acc

    t_all = time.time()
    for ds in DATASETS:
        d, labels, norm = pre[ds]
        test = d["test"]
        desc_lt = [norm[l.replace("_", " ").strip().lower()] for l in labels]
        acc_n = run(ds, "names", labels, labels, test)
        acc_d = run(ds, "descriptions", labels, desc_lt, test)
        meta = {"dataset": ds, "n_labels": len(labels), "n_test": len(test),
                "raw_labels": labels, "labels_names": labels, "labels_descriptions": desc_lt,
                "accuracy_names": round(acc_n, 4), "accuracy_descriptions": round(acc_d, 4),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **config}
        with open(f"{out}/{ds}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{ds}] names={acc_n:.4f} descriptions={acc_d:.4f} (delta {acc_d-acc_n:+.4f})", flush=True)
    print(f"ALL DONE {round(time.time()-t_all)}s", flush=True)
    print("RESULT: R3_DUMP_DONE")


if __name__ == "__main__":
    main()
