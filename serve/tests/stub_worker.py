"""Picklable stand-ins for serve.worker.init_model / predict, used to test ModelWorker's real process lifecycle
without loading any model. The `device` argument of ModelWorker selects a mode."""
import os
import time

MODE = None


def init_stub(mode):
    global MODE
    if mode == "fail_init":
        raise RuntimeError("model load failed")
    if mode.startswith("slow_init:"):
        time.sleep(float(mode.split(":", 1)[1]))
    MODE = mode


def predict_stub(message, labels):
    if message == "crash":
        os._exit(3)
    if message.startswith("sleep:"):
        time.sleep(float(message.split(":", 1)[1]))
    k = len(labels)
    return [1.0 / k] * k, [0, 0, 0]


def pid_stub(message, labels):
    return os.getpid()
