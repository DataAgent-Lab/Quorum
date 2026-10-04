"""Model worker: the ensemble runs in ONE child process that is started on the first cache miss and stopped when
idle, so the API process itself never imports torch and stays small while the demo is quiet.

Releasing model memory inside the API process is not enough: measured, `del` + gc + malloc_trim leaves ~750 MB
resident (torch's native libraries and arenas), whereas a process that never builds a model member stays ~50 MB.
Stopping the child returns everything to the OS.

Load control (this is a public endpoint on a small shared machine):
  - one job runs at a time; at most `max_pending` requests (running + waiting) are admitted, the rest get
    WorkerBusy immediately, so slow misses can never tie up the server's request threads and stall cache hits
  - a waiting request gives up after `queue_timeout_seconds` (WorkerBusy) without disturbing the running job
  - `job_timeout_seconds` bounds the running job only (including a cold model load); on expiry that child is killed
  - after a crash (e.g. the models fail to load) new jobs fail fast for `crash_backoff_seconds`

This module must stay importable without torch: the child imports it to unpickle `init_model` / `predict`.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, List, Optional, Sequence, Tuple

_log = logging.getLogger("quorum.worker")
_MODEL = None  # set in the child process only
_LABEL_CACHE_MAX = 256  # distinct label sets whose embeddings a member keeps; bounded so the child cannot grow


def init_model(device: str) -> None:
    """Child-process initializer: build the ensemble once per worker process."""
    global _MODEL
    from quorum import ZeroShotEnsemble
    _MODEL = ZeroShotEnsemble(device=device)


def predict(message: str, labels: List[str]) -> Tuple[List[float], List[int]]:
    """Child-process job: ensemble probabilities over `labels` (in order) and each member's argmax index."""
    for m in _MODEL.members:  # embedders memoise label vectors per label set; public input is unbounded
        lc = getattr(m, "_label_cache", None)
        if lc is not None and len(lc) > _LABEL_CACHE_MAX:
            lc.clear()
    members, ens = _MODEL.member_and_ensemble_proba(message, list(range(len(labels))), label_texts=labels)
    return [float(p) for p in ens], [int(m.argmax()) for m in members]


class WorkerCrashed(RuntimeError):
    """The worker process died, failed to load the models, or is in its post-crash backoff."""


class WorkerTimeout(RuntimeError):
    """The running job exceeded the job timeout; the worker was recycled."""


class WorkerBusy(RuntimeError):
    """Too many predictions are already running or waiting."""


def _stop_executor(executor: ProcessPoolExecutor, kill: bool, join_timeout: float = 15.0) -> None:
    # The pool only exposes its processes through a private attribute (stable across CPython 3.8–3.13). It is
    # needed to (a) kill a hung job, which shutdown() cannot do, and (b) wait until the child has really exited so
    # its memory is back with the OS.
    procs = list((getattr(executor, "_processes", None) or {}).values())
    if kill:
        for p in procs:
            p.kill()
    executor.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        p.join(join_timeout)
        if p.is_alive():
            p.kill()
            p.join(1.0)


class ModelWorker:
    """Lazily started, idle-stopped single-process model worker. Thread-safe: `run` is called from the API's
    request thread pool, `maybe_unload` from a background reaper thread."""

    def __init__(self, device: str = "cpu", idle_unload_seconds: float = 900.0,
                 job_timeout_seconds: float = 120.0, load_timeout_seconds: float = 300.0,
                 queue_timeout_seconds: float = 60.0, max_pending: int = 4, crash_backoff_seconds: float = 30.0,
                 init_fn: Callable = init_model, predict_fn: Callable = predict,
                 reap_interval_seconds: Optional[float] = None):
        self._init_fn = init_fn
        self._init_args = (device,)
        self._predict_fn = predict_fn
        self.idle_unload_seconds = idle_unload_seconds
        self.job_timeout_seconds = job_timeout_seconds
        self.load_timeout_seconds = load_timeout_seconds
        self.queue_timeout_seconds = queue_timeout_seconds
        self.crash_backoff_seconds = crash_backoff_seconds
        self._reap_interval = reap_interval_seconds or max(1.0, min(30.0, idle_unload_seconds / 4))
        self._slots = threading.BoundedSemaphore(max_pending)
        self._turn = threading.Lock()  # one job at a time
        self._lock = threading.Lock()  # guards the fields below
        self._executor: Optional[ProcessPoolExecutor] = None
        self._inflight = 0  # admitted requests: running + waiting for their turn
        self._last_used = time.monotonic()
        self._backoff_until = 0.0
        self._stop = threading.Event()
        self._reaper: Optional[threading.Thread] = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._executor is not None

    def run(self, message: str, labels: Sequence[str]) -> Tuple[List[float], List[int]]:
        if not self._slots.acquire(blocking=False):
            raise WorkerBusy("too many predictions are already running or waiting")
        with self._lock:
            self._inflight += 1
        try:
            if time.monotonic() < self._backoff_until:
                raise WorkerCrashed("the model worker failed recently; retrying shortly")
            if not self._turn.acquire(timeout=self.queue_timeout_seconds):
                raise WorkerBusy("timed out waiting for the model worker")
            try:
                return self._run_now(message, labels)
            finally:
                self._turn.release()
        finally:
            with self._lock:
                self._inflight -= 1
                self._last_used = time.monotonic()
            self._slots.release()

    def maybe_unload(self, now: Optional[float] = None) -> bool:
        """Stop the worker if it has been idle longer than the idle window. Never stops it while any admitted
        request is running or waiting."""
        if self.idle_unload_seconds <= 0:
            return False
        with self._lock:
            if self._executor is None or self._inflight > 0:
                return False
            if (time.monotonic() if now is None else now) - self._last_used < self.idle_unload_seconds:
                return False
            executor, self._executor = self._executor, None
        _stop_executor(executor, kill=False)
        return True

    def start_reaper(self) -> None:
        if self.idle_unload_seconds <= 0 or self._reaper is not None:
            return
        self._reaper = threading.Thread(target=self._reap_loop, name="quorum-worker-reaper", daemon=True)
        self._reaper.start()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            _stop_executor(executor, kill=True)

    # --- internals ---
    def _run_now(self, message: str, labels: Sequence[str]) -> Tuple[List[float], List[int]]:
        with self._lock:
            fresh = self._executor is None
            if fresh:
                self._executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                                     initializer=self._init_fn, initargs=self._init_args)
            executor = self._executor
        try:
            if fresh:
                # Spawn + imports + model load + first-inference warm-up take 35-80 s on a busy CPU host. They get
                # their own (longer) budget, so a slow cold start can never be killed by the per-job timeout and
                # leave the worker cold forever.
                stage, budget = "model load", self.load_timeout_seconds
                executor.submit(self._predict_fn, "warm up", ["a", "b"]).result(timeout=budget)
            stage, budget = "job", self.job_timeout_seconds
            return executor.submit(self._predict_fn, message, list(labels)).result(timeout=budget)
        except BrokenProcessPool as e:
            self._discard(executor, kill=False, backoff=True)
            raise WorkerCrashed(str(e) or "model worker process died") from e
        except FutureTimeout as e:
            self._discard(executor, kill=True, backoff=stage == "model load")
            raise WorkerTimeout(f"{stage} exceeded {budget:.0f}s") from e

    def _discard(self, executor: ProcessPoolExecutor, kill: bool, backoff: bool) -> None:
        with self._lock:
            if self._executor is executor:
                self._executor = None
            if backoff:
                self._backoff_until = time.monotonic() + self.crash_backoff_seconds
        _stop_executor(executor, kill=kill)

    def _reap_loop(self) -> None:
        while not self._stop.wait(self._reap_interval):
            try:
                self.maybe_unload()
            except Exception:  # the reaper must never die; log and retry on the next tick
                _log.exception("idle unload failed")
