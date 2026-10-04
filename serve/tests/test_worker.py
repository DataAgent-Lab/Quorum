"""AC-7 (idle unload really ends the child process; next miss restarts) and AC-8 (crash/timeout isolation; the
reaper never stops a worker with a job in flight). Uses real spawned processes with stub model functions."""
import os
import threading
import time

import pytest

from serve.tests import stub_worker as stub
from serve.worker import ModelWorker, WorkerBusy, WorkerCrashed, WorkerTimeout

FAR_FUTURE = time.monotonic() + 1e9


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def make(predict=stub.predict_stub, mode="ok", **kw):
    kw.setdefault("idle_unload_seconds", 60.0)
    kw.setdefault("crash_backoff_seconds", 0.0)
    return ModelWorker(device=mode, init_fn=stub.init_stub, predict_fn=predict, **kw)


def test_lazy_start_idle_unload_ends_child_and_restart_works():
    w = make(predict=stub.pid_stub)
    try:
        assert not w.loaded
        pid1 = w.run("x", ["a", "b"])
        assert w.loaded and alive(pid1)
        assert w.maybe_unload() is False  # not idle long enough
        assert w.maybe_unload(now=FAR_FUTURE) is True
        assert not w.loaded and not alive(pid1)
        pid2 = w.run("x", ["a", "b"])
        assert pid2 != pid1 and alive(pid2)
    finally:
        w.close()


def test_reaper_thread_unloads_on_its_own():
    w = make(predict=stub.pid_stub, idle_unload_seconds=0.5, reap_interval_seconds=0.1)
    try:
        w.start_reaper()
        pid = w.run("x", ["a", "b"])
        deadline = time.time() + 10
        # `loaded` flips as soon as the reaper detaches the worker; the child finishes exiting right after
        while (w.loaded or alive(pid)) and time.time() < deadline:
            time.sleep(0.1)
        assert not w.loaded and not alive(pid)
    finally:
        w.close()


def test_unload_disabled_when_idle_window_is_zero():
    w = make(idle_unload_seconds=0)
    try:
        w.run("x", ["a", "b"])
        assert w.maybe_unload(now=FAR_FUTURE) is False and w.loaded
    finally:
        w.close()


def test_never_unloads_with_a_job_in_flight():
    w = make()
    try:
        w.run("warm", ["a", "b"])
        t = threading.Thread(target=w.run, args=("sleep:1.5", ["a", "b"]))
        t.start()
        time.sleep(0.5)
        assert w.maybe_unload(now=FAR_FUTURE) is False and w.loaded
        t.join()
        assert w.maybe_unload(now=FAR_FUTURE) is True
    finally:
        w.close()


def test_crash_is_isolated_and_next_request_recovers():
    w = make()
    try:
        with pytest.raises(WorkerCrashed):
            w.run("crash", ["a", "b"])
        assert not w.loaded
        probs, picks = w.run("fine", ["a", "b"])
        assert probs == [0.5, 0.5] and picks == [0, 0, 0]
    finally:
        w.close()


def test_failed_model_load_reports_crash():
    w = make(mode="fail_init")
    try:
        with pytest.raises(WorkerCrashed):
            w.run("x", ["a", "b"])
        assert not w.loaded
    finally:
        w.close()


def test_timeout_recycles_the_worker():
    w = make(job_timeout_seconds=1.0)
    try:
        with pytest.raises(WorkerTimeout):
            w.run("sleep:30", ["a", "b"])
        assert not w.loaded
        assert w.run("fine", ["a", "b"])[0] == [0.5, 0.5]
    finally:
        w.close()


def test_admission_cap_rejects_immediately_when_full():
    w = make(max_pending=1)
    try:
        w.run("warm", ["a", "b"])
        t = threading.Thread(target=w.run, args=("sleep:1.5", ["a", "b"]))
        t.start()
        time.sleep(0.3)
        t0 = time.monotonic()
        with pytest.raises(WorkerBusy):
            w.run("x", ["a", "b"])
        assert time.monotonic() - t0 < 0.2
        t.join()
        assert w.run("x", ["a", "b"])[0] == [0.5, 0.5]
    finally:
        w.close()


def test_a_waiting_request_times_out_without_killing_the_running_job():
    w = make(queue_timeout_seconds=0.5, job_timeout_seconds=30)
    try:
        w.run("warm", ["a", "b"])
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("r", w.run("sleep:2", ["a", "b"])))
        t.start()
        time.sleep(0.3)
        with pytest.raises(WorkerBusy):
            w.run("x", ["a", "b"])
        t.join()
        assert result["r"][0] == [0.5, 0.5] and w.loaded  # the running job completed normally
    finally:
        w.close()


def test_after_a_crash_new_jobs_fail_fast_until_the_backoff_ends():
    w = make(crash_backoff_seconds=1.0)
    try:
        with pytest.raises(WorkerCrashed):
            w.run("crash", ["a", "b"])
        t0 = time.monotonic()
        with pytest.raises(WorkerCrashed):
            w.run("fine", ["a", "b"])
        assert time.monotonic() - t0 < 0.2 and not w.loaded  # no new process was spawned
        time.sleep(1.1)
        assert w.run("fine", ["a", "b"])[0] == [0.5, 0.5]
    finally:
        w.close()


def test_a_slow_cold_start_is_not_killed_by_the_job_timeout():
    w = make(mode="slow_init:2", job_timeout_seconds=1.0, load_timeout_seconds=30)
    try:
        assert w.run("x", ["a", "b"])[0] == [0.5, 0.5] and w.loaded
    finally:
        w.close()


def test_a_cold_start_over_the_load_budget_fails_with_backoff():
    w = make(mode="slow_init:5", load_timeout_seconds=1.0, crash_backoff_seconds=30)
    try:
        with pytest.raises(WorkerTimeout, match="model load"):
            w.run("x", ["a", "b"])
        assert not w.loaded
        with pytest.raises(WorkerCrashed):  # backing off instead of starting another 5 s load immediately
            w.run("x", ["a", "b"])
    finally:
        w.close()
