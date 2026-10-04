"""Export the demo page's examples (space/samples.json) from the benchmark per-item dumps.

    python -m serve.export_demo_samples          # -> space/samples.json

The page sends exactly these strings, and the same dumps seeded the API cache (serve/seed_cache.py), so every
example the page offers is answered from the cache. Labels are the description variant (it beats the bare names on
all 7 datasets); each label carries its name for display.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

from serve.cache import normalize

REPO = Path(__file__).resolve().parent.parent
SAMPLES_PER_DATASET = 300

# (dataset key, the title a visitor reads, the dataset name shown in small print)
DOMAINS = [
    ("banking77", "Banking", "Banking77"),
    ("clinc150", "Smart assistant", "CLINC150"),
    ("hwu64", "Home assistant", "HWU64"),
    ("massive", "Voice assistant", "MASSIVE"),
    ("mtop", "Everyday tasks", "MTOP"),
    ("snips", "Voice commands", "SNIPS"),
    ("bitext", "Customer support", "Bitext"),
]


def build(pred_dir: Path, per_dataset: int = SAMPLES_PER_DATASET, seed: int = 7) -> dict:
    domains = []
    for key, title, dataset in DOMAINS:
        meta = json.loads((pred_dir / f"{key}_meta.json").read_text())
        names, descs = meta["labels_names"], meta["labels_descriptions"]
        if not descs or len(descs) != len(names):
            raise ValueError(f"{key}: meta has no description labels")
        with gzip.open(pred_dir / f"{key}_descriptions.jsonl.gz", "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        seen, pool = set(), []
        for r in rows:
            text, _ = normalize(r["text"], [])
            # Template placeholders ("{{Order Number}}") read as broken, and editing them would leave the cache.
            if text and "{{" not in text and text not in seen:
                seen.add(text)
                pool.append({"t": text, "g": r["gold"]})
        samples = random.Random(seed).sample(pool, min(per_dataset, len(pool)))
        domains.append({"key": key, "title": title, "dataset": dataset, "n_test": meta["n_test"],
                        "accuracy": meta["accuracy_descriptions"],
                        "labels": [{"name": n, "desc": d} for n, d in zip(names, descs)],
                        "samples": samples})
    return {"version": 1, "label_text": "descriptions", "domains": domains}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=REPO / "results" / "predictions")
    ap.add_argument("--out", type=Path, default=REPO / "space" / "samples.json")
    ap.add_argument("--per-dataset", type=int, default=SAMPLES_PER_DATASET)
    args = ap.parse_args(argv)
    data = build(args.predictions, args.per_dataset)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB): "
          + ", ".join(f"{d['key']}={len(d['samples'])}" for d in data["domains"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
