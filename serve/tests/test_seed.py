"""AC-9 (seed import: validation, keys a client will hit, exact accuracy report) on synthetic dumps, and AC-12
(the demo preset list cannot drift from the frontend)."""
import gzip
import json
import re
from pathlib import Path

import pytest

from serve.cache import build_response, cache_key, dump_response
from serve.seed_cache import DEMO_EXAMPLES, DumpError, import_predictions, published_accuracy

REPO = Path(__file__).resolve().parents[2]
NAMES = ["card arrival", "change pin", "exchange rate"]
DESCS = ["when a new card will arrive", "changing the card PIN", "the exchange rate used for payments"]
ROWS = [
    {"idx": 0, "text": "where is my card", "gold": 0, "pred": 0, "probs": [0.8, 0.15, 0.05], "member_picks": [0, 0, 1]},
    {"idx": 1, "text": "new pin please", "gold": 1, "pred": 1, "probs": [0.1, 0.85, 0.05], "member_picks": [1, 1, 1]},
    {"idx": 2, "text": "what rate do you use", "gold": 2, "pred": 1, "probs": [0.1, 0.5, 0.4], "member_picks": [1, 2, 2]},
]


def server_config():
    from serve.cache import config_fingerprint
    fp = config_fingerprint()
    return {"members": fp["members"], "template": fp["template"], "cosine_scale": fp["cosine_scale"],
            "nli_dtype": "fp32"}


def write_dump(d: Path, dataset="toy", rows=ROWS, variants=("names", "descriptions"), names=NAMES, descs=DESCS,
               config=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}_meta.json").write_text(json.dumps({
        "dataset": dataset, "raw_labels": names, "labels_names": names,
        "labels_descriptions": descs if "descriptions" in variants else None,
        "accuracy_names": 2 / 3, **(config if config is not None else server_config())}))
    for v in variants:
        with gzip.open(d / f"{dataset}_{v}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def test_import_seeds_keys_a_client_will_hit(tmp_path, cache, client):
    write_dump(tmp_path / "pred")
    reports = import_predictions(cache, tmp_path / "pred", tmp_path / "results")
    assert [(r.variant, r.n_items, r.n_unique_keys, r.errors) for r in reports] == [
        ("names", 3, 3, []), ("descriptions", 3, 3, [])]
    assert all(abs(r.accuracy - 2 / 3) < 1e-9 for r in reports)
    assert cache.counts() == {"seed": 6, "live": 0}
    # exactly what the browser sends for this dataset/variant -> instant hit with the dumped answer
    for labels in (NAMES, DESCS):
        r = client.post("/predict", json={"message": "what rate do you use", "labels": labels})
        assert r.headers["X-Quorum-Cache"] == "hit"
        assert r.text == dump_response(build_response(labels, ROWS[2]["probs"], ROWS[2]["member_picks"]))


def test_dry_run_validates_without_writing(tmp_path):
    write_dump(tmp_path / "pred")
    reports = import_predictions(None, tmp_path / "pred", tmp_path / "results")
    assert [r.n_items for r in reports] == [3, 3]


def test_bad_rows_are_reported_and_skipped(tmp_path, cache):
    bad = ROWS + [
        {"idx": 3, "text": "x", "gold": 0, "pred": 0, "probs": [1.0, 0.0], "member_picks": [0, 0, 0]},
        {"idx": 4, "text": "y", "gold": 0, "pred": 2, "probs": [0.9, 0.05, 0.05], "member_picks": [0, 0, 0]},
        {"idx": 5, "text": "  ", "gold": 0, "pred": 0, "probs": [0.9, 0.05, 0.05], "member_picks": [0, 0, 0]},
        {"idx": 6, "text": "z", "gold": 0, "pred": 0, "probs": [0.9, 0.05, 0.05], "member_picks": [0, 9, 0]},
    ]
    write_dump(tmp_path / "pred", rows=bad, variants=("names",))
    (rep,) = import_predictions(cache, tmp_path / "pred", tmp_path / "results")
    assert rep.n_items == 3 and len(rep.errors) == 4
    assert cache.counts()["seed"] == 3


def test_label_strings_must_be_sendable_by_a_client(tmp_path, cache):
    write_dump(tmp_path / "pred", names=["card arrival ", "change pin", "exchange rate"], variants=("names",))
    with pytest.raises(DumpError):
        import_predictions(cache, tmp_path / "pred", tmp_path / "results")


def test_duplicate_test_sentences_collapse_to_one_key(tmp_path, cache):
    rows = ROWS + [dict(ROWS[0], idx=9)]
    write_dump(tmp_path / "pred", rows=rows, variants=("names",))
    (rep,) = import_predictions(cache, tmp_path / "pred", tmp_path / "results")
    assert rep.n_items == 4 and rep.n_unique_keys == 3
    assert cache.get(cache_key("where is my card", NAMES)) is not None


def test_published_accuracy_lookup_uses_repo_results():
    assert published_accuracy("banking77", "names", REPO / "results") == 0.7555
    assert published_accuracy("banking77", "descriptions", REPO / "results") == 0.7737
    assert published_accuracy("no-such-dataset", "names", REPO / "results") is None


def test_demo_presets_match_the_frontend():
    html = (REPO / "space" / "index.html").read_text()
    own = re.search(r'const OWN = \{.*?msg:\s*"([^"]*)",\s*labels:\s*\[([^\]]*)\]', html, re.S)
    assert own, "the page's own-labels default (const OWN) was not found"
    assert [(own.group(1), json.loads(f"[{own.group(2)}]"))] == [(m, list(l)) for m, l in DEMO_EXAMPLES]


def test_dumps_from_a_different_configuration_are_refused(tmp_path, cache):
    for bad in ({**server_config(), "template": "Other {label}."}, {**server_config(), "nli_dtype": "fp16"},
                {k: v for k, v in server_config().items() if k != "members"}):
        d = tmp_path / str(abs(hash(json.dumps(bad, sort_keys=True))))
        write_dump(d, config=bad, variants=("names",))
        with pytest.raises(DumpError):
            import_predictions(cache, d, tmp_path / "results")
    assert cache.counts()["seed"] == 0


def test_a_client_sending_meta_labels_in_any_order_hits(tmp_path, cache, client):
    import random
    write_dump(tmp_path / "pred", variants=("names",))
    import_predictions(cache, tmp_path / "pred", tmp_path / "results")
    meta = json.loads((tmp_path / "pred" / "toy_meta.json").read_text())
    for seed_ in range(5):
        labels = meta["labels_names"][:]
        random.Random(seed_).shuffle(labels)
        for row in ROWS:
            r = client.post("/predict", json={"message": row["text"], "labels": labels})
            assert r.headers["X-Quorum-Cache"] == "hit"
