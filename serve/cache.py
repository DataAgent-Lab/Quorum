"""Permanent prediction cache for the Quorum demo API.

A cache hit is answered without touching the models. Entries never expire. Rows come from two sources:

    seed  benchmark predictions and demo presets, imported ahead of time — never evicted
    live  results computed for public requests — capped at `max_live` rows (oldest live rows evicted first), and
          rows larger than `max_body_bytes` are not stored

Entries are only valid for the model configuration that produced them, so the cache stores a configuration
fingerprint; opening it with a different fingerprint empties it (re-seed afterwards) instead of ever serving answers
from an older configuration.

`normalize`, `cache_key` and `build_response` are shared by the API and the seeding tools, so a seeded entry is
found by exactly the request a client would send and is shaped exactly like a freshly computed one.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from quorum.ensemble import COSINE_SCALE, DEFAULT_EMBEDDERS, DEFAULT_NLI, TEMPLATE

MEMBER_DISPLAY_NAMES = ["PrismNLI-0.4B", "bge-large", "bge-base"]
KEY_VERSION = 1
SOURCES = ("seed", "live")
SEED_CHUNK = 2000

_log = logging.getLogger("quorum.cache")


def config_fingerprint() -> dict:
    """What a cached answer depends on. Update this whenever the ensemble's math or members change."""
    return {"key_version": KEY_VERSION, "members": [DEFAULT_NLI, *DEFAULT_EMBEDDERS], "template": TEMPLATE,
            "cosine_scale": float(COSINE_SCALE), "combine": "softmax(T=1) per member -> geometric mean",
            "nli_dtype_cpu": "fp32"}


def _clean(s: str) -> str:
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:  # lone surrogates from a JSON \ud800-style escape
        return s.encode("utf-8", "replace").decode("utf-8")


def normalize(message: Optional[str], labels: Iterable[Optional[str]]) -> Tuple[str, List[str]]:
    """The API's input rules: strip the message; strip labels, drop empty ones and duplicates (keep the first)."""
    msg = _clean((message or "").strip())
    out, seen = [], set()
    for l in labels:
        s = _clean((l or "").strip())
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return msg, out


def cache_key(message: str, labels: Sequence[str]) -> str:
    """Key over already-normalised inputs. Labels are sorted because the ensemble's result does not depend on the
    order of the candidate labels (every label is scored independently), so a permuted request is the same entry."""
    payload = json.dumps({"v": KEY_VERSION, "m": message, "l": sorted(labels)}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def build_response(labels: Sequence[str], ens_probs: Sequence[float], member_picks: Sequence[int]) -> dict:
    """The /predict response body: labels by descending probability (ties keep input order), 4-dp values, and each
    jury member's own top pick."""
    order = sorted(range(len(labels)), key=lambda i: -float(ens_probs[i]))
    top = order[0]
    return {
        "label": labels[top],
        "confidence": round(float(ens_probs[top]), 4),
        "distribution": {labels[i]: round(float(ens_probs[i]), 4) for i in order},
        "members": [{"model": MEMBER_DISPLAY_NAMES[m], "pick": labels[int(p)]} for m, p in enumerate(member_picks)],
    }


def dump_response(response: dict) -> str:
    """Serialise exactly like FastAPI's JSONResponse, so a stored body is byte-identical to a computed one."""
    return json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PredictionCache:
    """SQLite-backed key -> response-JSON store. One connection per thread (WAL: readers never wait on writers);
    concurrent writers — the API and an offline seeding run — wait up to `busy_timeout` seconds for each other."""

    def __init__(self, path, max_live: int = 20_000, max_body_bytes: int = 16_384,
                 fingerprint: Optional[dict] = None, busy_timeout: float = 5.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_live = max_live
        self.max_body_bytes = max_body_bytes
        self.busy_timeout = busy_timeout
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        db = self._conn()
        db.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            " key TEXT PRIMARY KEY,"
            " response TEXT NOT NULL,"
            " source TEXT NOT NULL CHECK (source IN ('seed','live')),"
            " created_at TEXT NOT NULL)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_source ON predictions(source)")
        db.execute("CREATE TABLE IF NOT EXISTS cache_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self.was_reset = self._check_fingerprint(fingerprint or config_fingerprint())

    def get(self, key: str) -> Optional[str]:
        row = self._conn().execute("SELECT response FROM predictions WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, body: str, source: str = "live") -> bool:
        """Store a serialised body; returns False when it is not stored. A seed row overwrites anything; a live row
        never overwrites an existing row and is skipped when larger than `max_body_bytes`."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        db = self._conn()
        if source == "seed":
            self._upsert_seed(db, key, body)
            return True
        if len(body.encode("utf-8")) > self.max_body_bytes:
            return False
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("INSERT OR IGNORE INTO predictions(key, response, source, created_at) VALUES (?,?,?,?)",
                       (key, body, "live", _now()))
            self._evict_live(db)
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        return True

    def put_many_seed(self, items: Iterable[Tuple[str, str]]) -> int:
        """Bulk-import (key, serialised body) pairs as seed rows, committing every SEED_CHUNK rows so a concurrent
        writer (the running API) never waits long. Returns the number written."""
        db, n = self._conn(), 0
        db.execute("BEGIN IMMEDIATE")
        try:
            for key, body in items:
                self._upsert_seed(db, key, body)
                n += 1
                if n % SEED_CHUNK == 0:
                    db.execute("COMMIT")
                    db.execute("BEGIN IMMEDIATE")
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        return n

    def counts(self) -> dict:
        rows = self._conn().execute("SELECT source, COUNT(*) FROM predictions GROUP BY source").fetchall()
        out = {s: 0 for s in SOURCES}
        out.update(dict(rows))
        return out

    def close(self) -> None:
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for c in conns:
            try:
                c.close()
            except sqlite3.Error:
                pass
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=self.busy_timeout, isolation_level=None,
                                check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
            with self._conns_lock:
                self._conns.append(c)
        return c

    def _check_fingerprint(self, fingerprint: dict) -> bool:
        db = self._conn()
        want = json.dumps(fingerprint, sort_keys=True)
        row = db.execute("SELECT v FROM cache_meta WHERE k = 'fingerprint'").fetchone()
        if row is not None and row[0] == want:
            return False
        reset = row is not None
        db.execute("BEGIN IMMEDIATE")
        try:
            if reset:
                db.execute("DELETE FROM predictions")
            db.execute("INSERT INTO cache_meta(k, v) VALUES ('fingerprint', ?) "
                       "ON CONFLICT(k) DO UPDATE SET v = excluded.v", (want,))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        if reset:
            _log.warning("cache %s was built with a different model configuration; emptied it (re-seed needed)",
                         self.path)
        return reset

    @staticmethod
    def _upsert_seed(db: sqlite3.Connection, key: str, body: str) -> None:
        db.execute(
            "INSERT INTO predictions(key, response, source, created_at) VALUES (?,?,'seed',?) "
            "ON CONFLICT(key) DO UPDATE SET response = excluded.response, source = 'seed', "
            "created_at = excluded.created_at",
            (key, body, _now()))

    def _evict_live(self, db: sqlite3.Connection) -> None:
        (n_live,) = db.execute("SELECT COUNT(*) FROM predictions WHERE source = 'live'").fetchone()
        excess = n_live - self.max_live
        if excess > 0:
            db.execute("DELETE FROM predictions WHERE rowid IN "
                       "(SELECT rowid FROM predictions WHERE source = 'live' ORDER BY rowid LIMIT ?)", (excess,))
