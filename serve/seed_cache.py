"""Seed the permanent prediction cache.

    # benchmark predictions (per-item dumps produced with scripts/eval_zeroshot.py on a GPU box)
    python -m serve.seed_cache --predictions results/predictions
    # the demo page's example presets (computed here with the local ensemble)
    python -m serve.seed_cache --demo
    # validate + report only, write nothing
    python -m serve.seed_cache --predictions results/predictions --dry-run

Dump layout (results/predictions/):
    {dataset}_meta.json              raw_labels, labels_names, labels_descriptions (null if none) — the exact label
                                     strings fed to the ensemble — plus the ensemble config and accuracies
    {dataset}_{variant}.jsonl.gz     one item per line: {idx, text, gold, pred, probs[K], member_picks[3]}
                                     variant = names | descriptions

Every item is stored under the key the API computes for {message: text, labels: <variant labels>}, with the body
the API would have returned for those probabilities — so a client sending that sentence with that label set gets
an instant cache hit. The report gives exact per-dataset accuracy and its gap to the published results/*.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from serve.cache import (MEMBER_DISPLAY_NAMES, PredictionCache, build_response, cache_key, config_fingerprint,
                         dump_response, normalize)

REPO = Path(__file__).resolve().parent.parent
VARIANTS = ("names", "descriptions")

# Must match EXAMPLES in space/index.html (guarded by serve/tests/test_seed.py).
DEMO_EXAMPLES = [
    ("when will my new card arrive?",
     ["card arrival", "card delivery estimate", "lost or stolen card", "change pin", "top up by card",
      "exchange rate"]),
    ("I was charged twice for the same order and want my money back",
     ["billing / double charge", "refund request", "order status", "cancel subscription", "technical bug",
      "general question"]),
    ("the new dashboard is gorgeous but it loads really slowly",
     ["praise", "performance complaint", "bug report", "feature request", "pricing concern", "churn risk"]),
]


class DumpError(ValueError):
    pass


@dataclass
class ImportReport:
    dataset: str
    variant: str
    n_items: int = 0
    n_unique_keys: int = 0
    n_correct: int = 0
    n_top_differs_from_pred: int = 0
    meta_accuracy: Optional[float] = None
    published_accuracy: Optional[float] = None
    errors: List[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_items if self.n_items else float("nan")

    def line(self) -> str:
        def vs(ref):
            return "n/a" if ref is None else f"{ref:.4f} (Δ {self.accuracy - ref:+.4f})"
        return (f"{self.dataset:10s} {self.variant:12s} items={self.n_items:5d} unique={self.n_unique_keys:5d} "
                f"acc={self.accuracy:.4f} meta={vs(self.meta_accuracy)} published={vs(self.published_accuracy)} "
                f"top≠pred={self.n_top_differs_from_pred} errors={len(self.errors)}")


def published_accuracy(dataset: str, variant: str, results_dir: Path) -> Optional[float]:
    """The repo's published full-test number for this dataset/variant, if recorded."""
    if variant == "names":
        p = results_dir / f"{dataset}.json"
        keys = ("ensemble_accuracy",)
    else:
        p = results_dir / f"{dataset}_descriptions.json"
        keys = ("ensemble_accuracy_descriptions", "ensemble_accuracy")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    for k in keys:
        if k in d:
            return float(d[k])
    return None


def check_meta_config(meta: dict, where: str) -> None:
    """A dump may only seed the cache if it was produced by the configuration this server runs."""
    fp = config_fingerprint()
    want = {"members": fp["members"], "template": fp["template"], "cosine_scale": fp["cosine_scale"],
            "nli_dtype": "fp32"}
    for k, v in want.items():
        got = meta.get(k)
        if k == "cosine_scale" and got is not None:
            got = float(got)
        if got != v:
            raise DumpError(f"{where}: meta {k}={got!r} but this server runs {v!r} — refusing to seed answers "
                            "from a different model configuration")


def discover(pred_dir: Path) -> Iterator[Tuple[str, str, List[str], Path, dict]]:
    """Yield (dataset, variant, labels, dump_path, meta) for every dump present in `pred_dir`."""
    for meta_path in sorted(pred_dir.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        check_meta_config(meta, meta_path.name)
        dataset = meta.get("dataset") or meta_path.name[: -len("_meta.json")]
        for variant in VARIANTS:
            dump = pred_dir / f"{dataset}_{variant}.jsonl.gz"
            labels = meta.get(f"labels_{variant}")
            if dump.exists():
                if not labels:
                    raise DumpError(f"{dump.name}: meta has no labels_{variant}")
                yield dataset, variant, list(labels), dump, meta


def iter_items(dataset: str, variant: str, labels: List[str], dump: Path, meta: dict,
               report: ImportReport) -> Iterator[Tuple[str, str]]:
    """Validate each dumped item and yield (cache key, serialised response). Problems go to report.errors."""
    k = len(labels)
    _, norm = normalize("x", labels)
    if norm != labels:
        raise DumpError(f"{dataset}/{variant}: label strings are not in API-normalised form (whitespace or "
                        "duplicates) — a client could never send this exact label set")
    keys = set()
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{dump.name}:{n}"
            message, _ = normalize(row.get("text"), [])
            probs, picks = row.get("probs"), row.get("member_picks")
            gold, pred = row.get("gold"), row.get("pred")
            if not message:
                report.errors.append(f"{where}: empty text")
                continue
            if not isinstance(probs, list) or len(probs) != k:
                report.errors.append(f"{where}: probs has {len(probs) if isinstance(probs, list) else '?'} "
                                     f"values, expected {k}")
                continue
            if not (isinstance(picks, list) and len(picks) == len(MEMBER_DISPLAY_NAMES)
                    and all(isinstance(p, int) and 0 <= p < k for p in picks)):
                report.errors.append(f"{where}: bad member_picks {picks!r}")
                continue
            if not (isinstance(gold, int) and isinstance(pred, int) and 0 <= gold < k and 0 <= pred < k):
                report.errors.append(f"{where}: bad gold/pred {gold!r}/{pred!r}")
                continue
            if probs[pred] < max(probs):  # equal is a 6-dp tie, not an error
                report.errors.append(f"{where}: pred={pred} is not the argmax of probs")
                continue
            resp = build_response(labels, probs, picks)
            report.n_items += 1
            report.n_correct += int(pred == gold)
            report.n_top_differs_from_pred += int(resp["label"] != labels[pred])
            key = cache_key(message, labels)
            keys.add(key)
            yield key, dump_response(resp)
    report.n_unique_keys = len(keys)


def import_predictions(cache: Optional[PredictionCache], pred_dir: Path, results_dir: Path) -> List[ImportReport]:
    reports = []
    for dataset, variant, labels, dump, meta in discover(pred_dir):
        rep = ImportReport(dataset, variant, meta_accuracy=meta.get(f"accuracy_{variant}"),
                           published_accuracy=published_accuracy(dataset, variant, results_dir))
        items = iter_items(dataset, variant, labels, dump, meta, rep)
        if cache is None:
            for _ in items:
                pass
        else:
            cache.put_many_seed(items)
        reports.append(rep)
    return reports


def seed_demo(cache: PredictionCache, device: str = "cpu") -> int:
    from serve import worker as w
    if w._MODEL is None:
        w.init_model(device)
    for message, labels in DEMO_EXAMPLES:
        msg, labs = normalize(message, labels)
        ens, picks = w.predict(msg, labs)
        cache.put(cache_key(msg, labs), dump_response(build_response(labs, ens, picks)), "seed")
    return len(DEMO_EXAMPLES)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.environ.get("QUORUM_CACHE_PATH",
                                                      str(REPO / "serve" / "data" / "quorum_cache.sqlite3")))
    ap.add_argument("--predictions", type=Path, help="directory with {dataset}_meta.json + dumps")
    ap.add_argument("--results", type=Path, default=REPO / "results", help="published results/*.json")
    ap.add_argument("--demo", action="store_true", help="compute + seed the demo page presets")
    ap.add_argument("--device", default=os.environ.get("QUORUM_DEVICE", "cpu"))
    ap.add_argument("--dry-run", action="store_true", help="validate and report only")
    args = ap.parse_args(argv)
    if not args.predictions and not args.demo:
        ap.error("nothing to do: pass --predictions DIR and/or --demo")

    cache = None if args.dry_run else PredictionCache(args.cache)
    failed = False
    if args.predictions:
        reports = import_predictions(cache, args.predictions, args.results)
        if not reports:
            print(f"no dumps found in {args.predictions}", file=sys.stderr)
            failed = True
        for r in reports:
            print(r.line())
            for e in r.errors[:5]:
                print("   ", e)
            failed |= bool(r.errors)
    if args.demo and cache is not None:
        print(f"demo presets seeded: {seed_demo(cache, args.device)}")
    if cache is not None:
        print(f"cache {args.cache}: {cache.counts()}")
        cache.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
