"""Fact-sheet measurements for Phase 1.0a (run from the repo root with the venv):

    python docs/phases/1.0a/measure/measure_lazy_unload.py

Measures, in ONE process:
  F1 RSS before any model load (serve deps imported, torch not yet imported)
  F2 cold ensemble load time + RSS after load
  F3 RSS after unload (del + gc.collect + glibc malloc_trim) -> does unloading give RAM back to the OS?
  F4 label-order invariance of the ensemble distribution (same message, permuted labels)
  F5 re-load time after an unload (the "cold miss after idle" cost)
"""
import ctypes, gc, os, time

import numpy as np


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


import fastapi, pydantic  # noqa: E402,F401  (what serve/app.py imports before the model)
from quorum import ZeroShotEnsemble  # noqa: E402

print(f"F1 rss_before_load_mb={rss_mb():.0f}  torch_imported={'torch' in __import__('sys').modules}", flush=True)

t = time.perf_counter()
clf = ZeroShotEnsemble(device="cpu")
print(f"F2 cold_load_s={time.perf_counter() - t:.1f}  rss_loaded_mb={rss_mb():.0f}", flush=True)

msg = "when will my new card arrive?"
labels = ["card arrival", "card delivery estimate", "lost or stolen card", "change pin", "top up by card", "exchange rate"]
t = time.perf_counter()
_, p = clf.member_and_ensemble_proba(msg, list(range(len(labels))), label_texts=labels)
print(f"   warm_predict_6_labels_s={time.perf_counter() - t:.2f}  rss_after_predict_mb={rss_mb():.0f}", flush=True)

perm = [3, 0, 5, 1, 4, 2]
_, q = clf.member_and_ensemble_proba(msg, list(range(len(labels))), label_texts=[labels[i] for i in perm])
q_by_label = {labels[perm[j]]: float(q[j]) for j in range(len(perm))}
max_abs = max(abs(float(p[i]) - q_by_label[labels[i]]) for i in range(len(labels)))
same_top = labels[int(np.argmax(p))] == max(q_by_label, key=q_by_label.get)
print(f"F4 label_order_max_abs_diff={max_abs:.2e}  same_top={same_top}", flush=True)

del clf
trim()
print(f"F3 rss_after_unload_mb={rss_mb():.0f}", flush=True)

t = time.perf_counter()
clf = ZeroShotEnsemble(device="cpu")
print(f"F5 reload_s={time.perf_counter() - t:.1f}  rss_reloaded_mb={rss_mb():.0f}", flush=True)
del clf
trim()
print(f"   rss_after_second_unload_mb={rss_mb():.0f}", flush=True)
