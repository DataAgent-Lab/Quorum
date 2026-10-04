"""AC-3 (key), AC-5 (response shape vs Phase 1.0), AC-6 (permanence + bounded live growth)."""
import numpy as np

from serve.cache import (MEMBER_DISPLAY_NAMES, PredictionCache, build_response, cache_key, dump_response,
                         normalize)


def legacy_response(labels, members, ens):
    """The Phase 1.0 construction, verbatim from serve/app.py@8a23bd4 lines 58-67."""
    order = sorted(range(len(labels)), key=lambda i: -float(ens[i]))
    dist = {labels[i]: round(float(ens[i]), 4) for i in order}
    top = order[0]
    return {
        "label": labels[top],
        "confidence": round(float(ens[top]), 4),
        "distribution": dist,
        "members": [{"model": MEMBER_DISPLAY_NAMES[m], "pick": labels[int(members[m].argmax())]}
                    for m in range(len(members))],
    }


def test_normalize_matches_api_rules():
    assert normalize("  hi there \n", [" a", "b ", "", "a", None, "c"]) == ("hi there", ["a", "b", "c"])
    assert normalize(None, []) == ("", [])


def test_key_ignores_label_order_but_not_content():
    m, labels = normalize("where is my card", ["card arrival", "change pin", "exchange rate"])
    k = cache_key(m, labels)
    assert cache_key(m, list(reversed(labels))) == k
    assert cache_key("where is my card?", labels) != k
    assert cache_key(m, labels[:2]) != k
    assert cache_key(m, labels + ["refund"]) != k
    # case is significant (the models are not guaranteed case-invariant)
    assert cache_key("Where is my card", labels) != k


def test_build_response_equals_phase_1_0_construction():
    rng = np.random.default_rng(0)
    for k in (2, 6, 77):
        labels = [f"label {i}" for i in range(k)]
        members = [rng.random(k) for _ in range(3)]
        ens = rng.random(k)
        ens = ens / ens.sum()
        picks = [int(m.argmax()) for m in members]
        assert build_response(labels, list(ens), picks) == legacy_response(labels, members, ens)


def test_response_body_is_fastapi_compatible_json():
    body = dump_response(build_response(["café", "b"], [0.25, 0.75], [1, 1, 0]))
    assert body.startswith('{"label":"b","confidence":0.75,"distribution":{"b":0.75,"café":0.25}')


def test_entries_survive_reopen_and_never_expire(tmp_path):
    path = tmp_path / "c.sqlite3"
    c = PredictionCache(path)
    c.put("k1", '{"x":1}', "live")
    c.put("k2", '{"x":2}', "seed")
    c.close()
    c2 = PredictionCache(path)
    assert c2.get("k1") == '{"x":1}' and c2.get("k2") == '{"x":2}'
    assert c2.counts() == {"seed": 1, "live": 1}


def test_live_cap_evicts_oldest_live_and_never_seed(tmp_path):
    c = PredictionCache(tmp_path / "c.sqlite3", max_live=3)
    c.put_many_seed([("s1", "1"), ("s2", "2")])
    for i in range(5):
        c.put(f"l{i}", str(i), "live")
    assert c.counts() == {"seed": 2, "live": 3}
    assert c.get("l0") is None and c.get("l1") is None
    assert [c.get(f"l{i}") for i in (2, 3, 4)] == ["2", "3", "4"]
    assert c.get("s1") == "1" and c.get("s2") == "2"


def test_live_never_overwrites_seed_but_seed_overwrites_live(tmp_path):
    c = PredictionCache(tmp_path / "c.sqlite3")
    c.put("a", "seed-body", "seed")
    c.put("a", "live-body", "live")
    assert c.get("a") == "seed-body"
    c.put("b", "live-body", "live")
    c.put("b", "seed-body", "seed")
    assert c.get("b") == "seed-body"
    assert c.counts() == {"seed": 2, "live": 0}


def test_different_model_configuration_empties_the_cache(tmp_path):
    from serve.cache import config_fingerprint
    path = tmp_path / "c.sqlite3"
    c = PredictionCache(path)
    c.put("k", "body", "seed")
    c.close()
    assert PredictionCache(path).get("k") == "body"  # same config: kept
    other = dict(config_fingerprint(), template="Something else {label}.")
    c2 = PredictionCache(path, fingerprint=other)
    assert c2.was_reset and c2.get("k") is None and c2.counts() == {"seed": 0, "live": 0}


def test_oversized_live_bodies_are_not_stored(tmp_path):
    c = PredictionCache(tmp_path / "c.sqlite3", max_body_bytes=100)
    assert c.put("small", "x" * 50, "live") is True
    assert c.put("big", "x" * 500, "live") is False
    assert c.get("big") is None
    assert c.put("big-seed", "x" * 500, "seed") is True  # seeds are imported deliberately


def test_lone_surrogates_are_sanitised_not_a_500():
    m, labels = normalize("bad \ud800 text", ["a\udfff", "b"])
    assert m == "bad ? text" and labels == ["a?", "b"]
    cache_key(m, labels)
    dump_response(build_response(labels, [0.5, 0.5], [0, 0, 0]))


def test_seeding_from_another_connection_does_not_break_live_writes(tmp_path):
    import threading
    path = tmp_path / "c.sqlite3"
    api, seeder = PredictionCache(path), PredictionCache(path)
    errors = []

    def seed_many():
        try:
            seeder.put_many_seed((f"s{i}", "x" * 200) for i in range(20000))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    t = threading.Thread(target=seed_many)
    t.start()
    for i in range(200):
        api.put(f"l{i}", "y", "live")
        assert api.get(f"l{i}") == "y"
    t.join()
    assert errors == []
    assert api.counts() == {"seed": 20000, "live": 200}
