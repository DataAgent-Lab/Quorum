"""REAL end-to-end test (no mocks): the actual ensemble in a spawned worker behind the actual API.

    QUORUM_REAL_MODEL_TEST=1 pytest serve/tests/test_real_e2e.py -s     # ~1-2 min on CPU, needs the HF models

Covers AC-2 (miss then byte-identical hit), AC-5 (served answer == the ensemble's own answer), AC-7 (idle unload
ends the worker; the API process never loads torch and stays small) and AC-12 (seeded demo presets are hits).
"""
import os
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("QUORUM_REAL_MODEL_TEST") != "1",
                                reason="set QUORUM_REAL_MODEL_TEST=1 to run the real-model test")


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024


def test_real_model_worker_through_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from serve.app import create_app
    from serve.cache import PredictionCache
    from serve.seed_cache import DEMO_EXAMPLES, seed_demo
    from serve.worker import ModelWorker

    cache = PredictionCache(tmp_path / "cache.sqlite3")
    worker = ModelWorker(device="cpu", idle_unload_seconds=3, reap_interval_seconds=0.5)
    msg, labels = "my card still has not arrived, where is it?", ["card arrival", "change pin", "exchange rate",
                                                                  "lost or stolen card"]
    with TestClient(create_app(cache=cache, worker=worker)) as c:
        t = time.perf_counter()
        r1 = c.post("/predict", json={"message": msg, "labels": labels})
        cold = time.perf_counter() - t
        assert r1.status_code == 200 and r1.headers["X-Quorum-Cache"] == "miss", r1.text
        assert c.get("/health").json()["model_loaded"] is True

        t = time.perf_counter()
        r2 = c.post("/predict", json={"message": msg, "labels": list(reversed(labels))})
        hit = time.perf_counter() - t
        assert r2.headers["X-Quorum-Cache"] == "hit" and r2.content == r1.content

        deadline = time.time() + 30
        while c.get("/health").json()["model_loaded"] and time.time() < deadline:
            time.sleep(0.5)
        assert c.get("/health").json()["model_loaded"] is False
        api_rss = rss_mb()
        print(f"\n  cold miss {cold:.1f}s, hit {hit * 1000:.1f}ms, API-process RSS after unload {api_rss:.0f} MB, "
              f"torch imported in API process: {'torch' in sys.modules}")
        assert "torch" not in sys.modules
        assert api_rss < 150

        # the served answer is exactly the ensemble's own answer (same code, same inputs, in this process now)
        from serve import worker as w
        w.init_model("cpu")
        probs, picks = w.predict(msg, labels)
        d = r1.json()
        assert d["label"] == labels[max(range(len(probs)), key=probs.__getitem__)]
        for i, lab in enumerate(labels):
            assert abs(d["distribution"][lab] - probs[i]) <= 5e-5 + 1e-6
        assert [m["pick"] for m in d["members"]] == [labels[p] for p in picks]

        # seeded demo presets are hits
        seed_demo(cache)
        for m, labs in DEMO_EXAMPLES:
            r = c.post("/predict", json={"message": m, "labels": labs})
            assert r.status_code == 200 and r.headers["X-Quorum-Cache"] == "hit"
    worker.close()
    cache.close()
