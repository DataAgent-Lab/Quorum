"""AC-1 hit path, AC-2 miss path, AC-3 permuted hit, AC-4 label limits, AC-5 contract, AC-8 (API side), AC-11."""
from serve.app import CACHE_HEADER, MAX_LABELS, MAX_LABELS_REQUEST, MODEL_IDS
from serve.cache import build_response, cache_key, dump_response, normalize
from serve.worker import WorkerCrashed, WorkerTimeout

LABELS = ["card arrival", "change pin", "exchange rate"]
MSG = "when will my new card arrive?"


def seed(cache, message, labels, probs=None, picks=(0, 0, 0)):
    m, labs = normalize(message, labels)
    probs = probs or [1.0 / len(labs)] * len(labs)
    body = dump_response(build_response(labs, probs, list(picks)))
    cache.put(cache_key(m, labs), body, "seed")
    return body


def test_hit_returns_stored_body_without_touching_the_worker(client, cache, worker):
    body = seed(cache, MSG, LABELS, probs=[0.7, 0.2, 0.1])
    r = client.post("/predict", json={"message": MSG, "labels": LABELS})
    assert r.status_code == 200
    assert r.headers[CACHE_HEADER] == "hit"
    assert r.text == body
    assert worker.calls == [] and not worker.loaded


def test_miss_computes_once_stores_live_and_repeat_is_identical_hit(client, cache, worker):
    r1 = client.post("/predict", json={"message": MSG, "labels": LABELS})
    assert r1.status_code == 200 and r1.headers[CACHE_HEADER] == "miss"
    assert worker.calls == [(MSG, LABELS)]
    assert cache.counts() == {"seed": 0, "live": 1}
    r2 = client.post("/predict", json={"message": MSG, "labels": LABELS})
    assert r2.headers[CACHE_HEADER] == "hit"
    assert r2.content == r1.content
    assert len(worker.calls) == 1


def test_request_normalisation_and_permuted_labels_hit_the_same_entry(client, worker):
    client.post("/predict", json={"message": MSG, "labels": LABELS})
    r = client.post("/predict", json={"message": f"  {MSG} ", "labels": [" exchange rate", "card arrival",
                                                                          "change pin", "card arrival", ""]})
    assert r.headers[CACHE_HEADER] == "hit"
    r = client.post("/predict", json={"message": MSG + "!", "labels": LABELS})
    assert r.headers[CACHE_HEADER] == "miss"
    assert len(worker.calls) == 2


def test_more_than_50_labels_served_only_from_cache(client, cache, worker):
    big = [f"intent {i}" for i in range(77)]
    seed(cache, "my card has not arrived", big)
    r = client.post("/predict", json={"message": "my card has not arrived", "labels": big})
    assert r.status_code == 200 and r.headers[CACHE_HEADER] == "hit"
    r = client.post("/predict", json={"message": "a different sentence", "labels": big})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and str(MAX_LABELS) in r.json()["detail"]
    assert worker.calls == []
    # exactly MAX_LABELS is still computed live
    r = client.post("/predict", json={"message": "x", "labels": [f"l{i}" for i in range(MAX_LABELS)]})
    assert r.status_code == 200 and r.headers[CACHE_HEADER] == "miss"


def test_request_limits_and_legacy_errors(client, worker):
    too_many = [f"l{i}" for i in range(MAX_LABELS_REQUEST + 1)]
    assert client.post("/predict", json={"message": "x", "labels": too_many}).status_code == 422
    r = client.post("/predict", json={"message": "x", "labels": ["a", " a "]})
    assert r.status_code == 422 and r.json() == {"detail": "need at least 2 distinct labels"}
    r = client.post("/predict", json={"message": "   ", "labels": ["a", "b"]})
    assert r.status_code == 422 and r.json() == {"detail": "empty message"}
    assert client.post("/predict", json={"message": "x", "labels": ["a"]}).status_code == 422
    assert worker.calls == []


def test_success_body_contract_is_unchanged(client):
    d = client.post("/predict", json={"message": MSG, "labels": LABELS}).json()
    assert list(d) == ["label", "confidence", "distribution", "members"]
    assert d["label"] == "card arrival"
    assert d["confidence"] == round(d["confidence"], 4)
    vals = list(d["distribution"].values())
    assert vals == sorted(vals, reverse=True) and set(d["distribution"]) == set(LABELS)
    assert [m["model"] for m in d["members"]] == ["PrismNLI-0.4B", "bge-large", "bge-base"]
    assert all(m["pick"] in LABELS for m in d["members"])


def test_worker_failures_map_to_503_504_and_recover(cache):
    from fastapi.testclient import TestClient
    from serve.app import create_app
    from conftest import FakeWorker

    w = FakeWorker(fail=WorkerCrashed("died"))
    with TestClient(create_app(cache=cache, worker=w)) as c:
        assert c.post("/predict", json={"message": "a", "labels": LABELS}).status_code == 503
        assert c.post("/predict", json={"message": "a", "labels": LABELS}).status_code == 200
        w.fail = WorkerTimeout("slow")
        assert c.post("/predict", json={"message": "b", "labels": LABELS}).status_code == 504
        assert c.post("/predict", json={"message": "b", "labels": LABELS}).status_code == 200
    assert cache.counts()["live"] == 2  # failures are never cached


def test_health_keeps_phase_1_0_fields_and_does_not_load_models(client, cache, worker):
    seed(cache, MSG, LABELS)
    d = client.get("/health").json()
    assert d["ok"] is True and d["service"] == "quorum" and d["members"] == MODEL_IDS
    assert d["model_loaded"] is False and d["cache"] == {"seed": 1, "live": 0}
    assert worker.calls == []


def test_cache_header_is_exposed_to_browsers(client):
    r = client.post("/predict", json={"message": MSG, "labels": LABELS},
                    headers={"Origin": "https://dataagent-quorum-demo.static.hf.space"})
    assert CACHE_HEADER.lower() in r.headers.get("access-control-expose-headers", "").lower()


def _app(cache, worker, **kw):
    from fastapi.testclient import TestClient
    from serve.app import create_app
    return TestClient(create_app(cache=cache, worker=worker, **kw))


def test_concurrent_identical_misses_compute_once(cache):
    import threading
    from conftest import FakeWorker
    gate = threading.Event()
    w = FakeWorker(gate=gate)
    out = []
    with _app(cache, w) as c:
        threads = [threading.Thread(target=lambda: out.append(c.post("/predict", json={"message": MSG,
                                                                                       "labels": LABELS})))
                   for _ in range(4)]
        for t in threads:
            t.start()
        assert w.started.wait(5)
        import time
        time.sleep(0.3)  # let the others queue behind the in-flight computation
        gate.set()
        for t in threads:
            t.join()
    assert len(w.calls) == 1
    assert [r.status_code for r in out] == [200] * 4 and len({r.content for r in out}) == 1


def test_hits_stay_fast_while_a_miss_is_computing(cache):
    import threading, time
    from conftest import FakeWorker
    seed(cache, "cached sentence", LABELS)
    gate = threading.Event()
    w = FakeWorker(gate=gate)
    with _app(cache, w) as c:
        t = threading.Thread(target=lambda: c.post("/predict", json={"message": "slow", "labels": LABELS}))
        t.start()
        assert w.started.wait(5)
        t0 = time.perf_counter()
        r = c.post("/predict", json={"message": "cached sentence", "labels": LABELS})
        elapsed = time.perf_counter() - t0
        gate.set()
        t.join()
    assert r.headers[CACHE_HEADER] == "hit" and elapsed < 0.5


def test_busy_worker_is_a_503(cache):
    from conftest import FakeWorker
    from serve.worker import WorkerBusy
    with _app(cache, FakeWorker(fail=WorkerBusy("full"))) as c:
        r = c.post("/predict", json={"message": "x", "labels": LABELS})
    assert r.status_code == 503 and "busy" in r.json()["detail"]


def test_live_results_are_not_stored_when_persistence_is_off(cache):
    from conftest import FakeWorker
    w = FakeWorker()
    with _app(cache, w, persist_live=False) as c:
        assert c.post("/predict", json={"message": MSG, "labels": LABELS}).headers[CACHE_HEADER] == "miss"
        assert c.post("/predict", json={"message": MSG, "labels": LABELS}).headers[CACHE_HEADER] == "miss"
    assert len(w.calls) == 2 and cache.counts() == {"seed": 0, "live": 0}


def test_a_cache_write_failure_still_returns_the_answer(cache, monkeypatch):
    import sqlite3
    from conftest import FakeWorker

    def broken_put(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cache, "put", broken_put)
    with _app(cache, FakeWorker()) as c:
        r = c.post("/predict", json={"message": MSG, "labels": LABELS})
    assert r.status_code == 200 and r.json()["label"] == "card arrival"


def test_cors_origin_list_plus_regex(cache, worker, monkeypatch):
    from fastapi.testclient import TestClient
    from serve.app import create_app
    monkeypatch.setenv("CORS_ORIGINS", "https://site.example,http://localhost:3003")
    monkeypatch.setenv("CORS_ORIGIN_REGEX", r"^https://app-[a-z0-9-]+-team\.preview\.example$")
    with TestClient(create_app(cache=cache, worker=worker)) as c:
        def allowed(origin):
            r = c.options("/predict", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                "Access-Control-Request-Headers": "content-type"})
            return r.headers.get("access-control-allow-origin") == origin
        assert allowed("https://site.example") and allowed("http://localhost:3003")
        assert allowed("https://app-k2pmknqr6-team.preview.example")
        assert not allowed("https://evil.example")
        assert not allowed("https://app-x-team.preview.example.evil.example")  # must match in full
