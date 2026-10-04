"""Build the Phase 1.0b design prototype: real benchmark test sentences, exact label names/descriptions and each
sample's REAL cached verdict (from the research session's per-item dumps), inlined into a single HTML file that opens
anywhere (no server, no API). Only the "Your own labels" path is simulated in the prototype.

    python docs/phases/1.0b/prototype/build_prototype.py      # -> docs/phases/1.0b/prototype/prototype.html

The production page fetches verdicts from the API instead of embedding them.
"""
import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SAMPLES_PER_DATASET = 40

# Friendly domain titles (what a visitor reads) + the dataset name in small print.
DATASETS = [
    ("banking77", "Banking", "Banking77"),
    ("clinc150", "Smart assistant", "CLINC150"),
    ("hwu64", "Home assistant", "HWU64"),
    ("massive", "Voice assistant", "MASSIVE"),
    ("mtop", "Everyday tasks", "MTOP"),
    ("snips", "Voice commands", "SNIPS"),
    ("bitext", "Customer support", "Bitext"),
]
def main():
    """Real data from the research session's per-item dumps (exact label strings sent to the API, real cached
    verdicts) — the prototype embeds each sample's real response, so no network is needed to review it."""
    import gzip, sys
    sys.path.insert(0, str(REPO))
    from serve.cache import build_response
    pred = REPO / "results" / "predictions"
    out = []
    for key, title, dsname in DATASETS:
        meta = json.loads((pred / f"{key}_meta.json").read_text())
        names, descs = meta["labels_names"], meta["labels_descriptions"]
        rows = [json.loads(l) for l in gzip.open(pred / f"{key}_descriptions.jsonl.gz", "rt")]
        pool = [r for r in rows if "{{" not in r["text"] and r["text"].strip()]
        rng = random.Random(7)
        samples = []
        for r in rng.sample(pool, SAMPLES_PER_DATASET):
            res = build_response(descs, r["probs"], r["member_picks"])
            samples.append({"t": r["text"].strip(), "g": r["gold"], "res": res})
        out.append({"key": key, "title": title, "dataset": dsname, "n_test": meta["n_test"],
                    "accuracy": meta["accuracy_descriptions"], "accuracy_variant": "descriptions",
                    "labels": [{"name": n, "desc": d} for n, d in zip(names, descs)], "samples": samples})
    html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps({"datasets": out}))
    (HERE / "prototype.html").write_text(html)
    print(f"wrote {HERE / 'prototype.html'} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
