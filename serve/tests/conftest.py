import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# serve.app builds a module-level app at import: point its default cache at a throwaway file, never the repo.
os.environ.setdefault("QUORUM_CACHE_PATH", str(Path(tempfile.mkdtemp(prefix="quorum-test-")) / "cache.sqlite3"))

from fastapi.testclient import TestClient  # noqa: E402

from serve.app import create_app  # noqa: E402
from serve.cache import PredictionCache  # noqa: E402


class FakeWorker:
    """Stands in for ModelWorker: deterministic probabilities, records every call, never starts a process."""

    def __init__(self, fail=None, gate=None):
        self.calls = []
        self.fail = fail  # an exception instance to raise once
        self.gate = gate  # a threading.Event the job waits on (to hold a miss "in progress")
        self.started = __import__("threading").Event()
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def run(self, message, labels):
        self.calls.append((message, list(labels)))
        self.started.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail is not None:
            exc, self.fail = self.fail, None
            raise exc
        self._loaded = True
        k = len(labels)
        # label i gets weight k - i: the first label wins, all distinct
        w = [float(k - i) for i in range(k)]
        s = sum(w)
        return [x / s for x in w], [0, 1 % k, 0]

    def start_reaper(self):
        pass

    def close(self):
        pass


@pytest.fixture
def cache(tmp_path):
    c = PredictionCache(tmp_path / "cache.sqlite3")
    yield c


@pytest.fixture
def worker():
    return FakeWorker()


@pytest.fixture
def client(cache, worker):
    with TestClient(create_app(cache=cache, worker=worker)) as c:
        yield c
