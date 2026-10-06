#!/usr/bin/env python3
"""R13 — contamination robustness: the zero-shot ensemble with a documented-clean NLI member.

The default NLI member, Jaehun/PrismNLI-0.4B, is fine-tuned FROM MoritzLaurer/deberta-v3-large-zeroshot-v2.0 (its
model card: "Instead of starting from scratch, we start from deberta-v3-large-zeroshot-v2.0"), whose training list
marks banking77 and massive as used (train splits). This run swaps in NLI checkpoints whose cards document training
data that excludes those datasets, and keeps everything else identical:

    clean_c      MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c @ b2730f1 (synthetic Mixtral data + MNLI + FEVER-NLI)
    mnli_snli    cross-encoder/nli-deberta-v3-large @ bab4bc7 (SNLI + MultiNLI)

The NLI member is scored exactly as quorum.ensemble._NLIMember.logits does (premise = message, hypothesis =
"This message is about {label}.", max_length 256, the entailment logit per label, softmax at T=1), in fp32. The
entailment index is the label named exactly "entailment" (asserted). The two embedding members are not re-run:
their distributions are the committed R3 / R12 member_probs (bge-large, bge-base), so the ensemble is the
geometric mean of [new NLI, bge-large, bge-base], renormalised.

    python scripts/r13_clean_nli.py --model clean_c [--r3-dir PATH] [--r12-dir PATH]

Writes, per dataset x {names, descriptions}:
    results/predictions/{dataset}_{variant}_r13_{model}.jsonl.gz   test: ensemble probs/pred + member_probs/picks
    results/r13/{dataset}_{variant}_train1000_{model}.jsonl.gz     the R12 pre-registered train-side sample
and results/r13/summary_{model}.json (accuracies; exact McNemar of the new ensemble vs its best single member and
vs the original ensemble).
"""
import argparse, gzip, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = {"clean_c": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", "b2730f16019076bb0009481121efbe4705e0e378"),
          "mnli_snli": ("cross-encoder/nli-deberta-v3-large", "bab4bc7178836f731dcfd18c06ca9def0a137712")}
DATASETS = ["banking77", "clinc150", "hwu64", "massive", "mtop", "snips", "bitext"]
DESC_MODULE = {"clinc150": "clinc"}
OUTP, OUT = ROOT / "results" / "predictions", ROOT / "results" / "r13"
g7 = lambda v: [float(f"{x:.7g}") for x in v]


def load_desc(ds, labels):
    mod = DESC_MODULE.get(ds, ds)
    spec = importlib.util.spec_from_file_location(mod, ROOT / "examples" / f"{mod}_descriptions.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    d = next(v for k, v in vars(m).items() if k.endswith("_DESCRIPTIONS") and isinstance(v, dict))
    norm = {str(k).replace("_", " ").strip().lower(): v for k, v in d.items()}
    return [norm[str(l).replace("_", " ").strip().lower()] for l in labels]


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def softmax(z):
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e / e.sum()


def geo(P):
    g = np.exp(np.mean(np.log(np.asarray(P) + 1e-12), axis=0)); return g / g.sum()


def nli_member(name, rev):
    """A quorum.ensemble._NLIMember whose model/tokenizer are loaded at a pinned revision, in fp32 on cuda."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from quorum.ensemble import _NLIMember
    m = _NLIMember.__new__(_NLIMember)
    m.torch, m.device = torch, "cuda"
    m.tok = AutoTokenizer.from_pretrained(name, revision=rev)
    m.model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev, torch_dtype=torch.float32).to("cuda").eval()
    id2label = {int(k): str(v).lower() for k, v in m.model.config.id2label.items()}
    hits = [i for i, v in id2label.items() if v == "entailment"]
    assert len(hits) == 1, f"no unique 'entailment' label in {id2label}"
    m.entail = hits[0]
    return m, id2label


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--r3-dir", default=str(OUTP)); ap.add_argument("--r12-dir", default=str(ROOT / "results" / "r12"))
    a = ap.parse_args(); tag = a.model; name, rev = MODELS[tag]
    import torch, transformers
    from quorum import data
    nli, id2label = nli_member(name, rev)
    ids12 = json.loads((Path(a.r12_dir) / "sample_ids.json").read_text())
    OUTP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    summary, t_all = {}, time.time()
    for ds in DATASETS:
        d = data.load(ds); labels = list(d["labels"]); test, train = d["test"], d["train"]
        for variant, lt in [("names", labels), ("descriptions", load_desc(ds, labels))]:
            t0 = time.time()
            # ---- test: new NLI + R3 embedder members ----
            r3 = [json.loads(l) for l in gzip.open(Path(a.r3_dir) / f"{ds}_{variant}.jsonl.gz", "rt")]
            assert [x["text"] for x in r3] == [r["text"] for r in test], f"{ds}/{variant}: R3 row order differs"
            gold = np.array([r["label"] for r in test]); new, old_ens, picks = [], [], []
            with gzip.open(OUTP / f"{ds}_{variant}_r13_{tag}.jsonl.gz", "wt") as f:
                for i, r in enumerate(test):
                    pn = softmax(nli.logits(r["text"], lt)); emb = r3[i]["member_probs"][1:]
                    ep = geo([pn, *emb]); mp = [g7(pn), *emb]
                    row = {"idx": i, "text": r["text"], "gold": int(gold[i]), "pred": int(np.argmax(ep)), "probs": g7(ep),
                           "member_picks": [int(np.argmax(m)) for m in mp], "member_probs": mp}
                    f.write(json.dumps(row) + "\n"); new.append(row); old_ens.append(r3[i]["pred"])
            pe = np.array([x["pred"] for x in new]); mpk = np.array([x["member_picks"] for x in new]); po = np.array(old_ens)
            accs = [float((mpk[:, j] == gold).mean()) for j in range(3)]; best = int(np.argmax(accs))
            he, hb, ho = pe == gold, mpk[:, best] == gold, po == gold
            b1, c1 = int((he & ~hb).sum()), int((~he & hb).sum()); b2, c2 = int((he & ~ho).sum()), int((~he & ho).sum())
            # ---- R12 train-side sample: new NLI + R12 embedder members ----
            r12 = [json.loads(l) for l in gzip.open(Path(a.r12_dir) / f"{ds}_{variant}_train1000.jsonl.gz", "rt")]
            assert [x["train_idx"] for x in r12] == ids12["ids"][ds], f"{ds}/{variant}: R12 ids differ"
            tr_hit = []
            with gzip.open(OUT / f"{ds}_{variant}_train1000_{tag}.jsonl.gz", "wt") as f:
                for x in r12:
                    assert train[x["train_idx"]]["text"] == x["text"]
                    pn = softmax(nli.logits(x["text"], lt)); emb = x["member_probs"][1:]; ep = geo([pn, *emb])
                    tr_hit.append(int(np.argmax(ep)) == x["gold"])
                    f.write(json.dumps({"train_idx": x["train_idx"], "text": x["text"], "gold": x["gold"],
                                        "pred": int(np.argmax(ep)), "probs": g7(ep),
                                        "member_probs": [g7(pn), *emb]}) + "\n")
            rec = {"nli_alone_accuracy": round(accs[0], 4), "bge_large_accuracy": round(accs[1], 4),
                   "bge_base_accuracy": round(accs[2], 4), "ensemble_accuracy": round(float(he.mean()), 4),
                   "original_ensemble_accuracy": round(float(ho.mean()), 4),
                   "best_single_member": ["nli", "bge_large", "bge_base"][best],
                   "mcnemar_ensemble_vs_best_member": {"b": b1, "c": c1, "p_two_sided_exact": mcnemar_exact(b1, c1)},
                   "mcnemar_ensemble_vs_original_ensemble": {"b": b2, "c": c2, "p_two_sided_exact": mcnemar_exact(b2, c2)},
                   "train1000_ensemble_accuracy": round(float(np.mean(tr_hit)), 4), "seconds": round(time.time() - t0, 1)}
            summary[f"{ds}/{variant}"] = rec; print(f"[{ds}/{variant}] {json.dumps(rec)}", flush=True)
    meta = {"model": name, "revision": rev, "id2label": id2label, "entailment_index": nli.entail, "precision": "fp32",
            "template": "This message is about {label}.", "max_length": 256, "softmax_temperature": 1.0,
            "ensemble": "geometric mean (T=1) of the new NLI member and the committed R3/R12 bge-large, bge-base member_probs",
            "results": summary, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "device": torch.cuda.get_device_name(0), "wall_seconds": round(time.time() - t_all, 1),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / f"summary_{tag}.json").write_text(json.dumps(meta, indent=1)); print("R13_DONE", tag, flush=True)


if __name__ == "__main__":
    main()
