#!/usr/bin/env python3
"""R17 latency on one GPU type (descriptive; no pre-registration, design fixed in the request of 2026-10-17).

Systems, each on the same GB10 in bf16 with nothing else on the GPU, one process per system:
  clef_flash, clef  Cloudflare release code (`joint_schema_model.py`), Arm-B / b24B bodies
  pplx              perplexity-ai/pplx-decider-v1.1-27b release `autojev.model.DecisionModel` (not the serving engine)
  open              this repo's open stack, the clean Banking77 selection (M2 = reader + kNN, `scripts/clean_24shot.py`)

    python scripts/r17_latency.py run --system open|clef_flash|clef|pplx      # -> results/paper/latency_r17_<system>.json
    python scripts/r17_latency.py merge                                      # -> results/paper/latency_r17.json

Inputs: the first 200 Banking77 test items (b24B bodies, every sha256 checked against the Jev request manifest by
`r14_clef.armb_bodies`); the open stack gets the same 24 examples per item from the clean run's recorded retrieval.
Warm-up: 8 further test items (idx 200-207), discarded. Per system: model load time; batch-1 latency per item over
the 200 (wall time from input to probabilities, tokenisation included, GPU synchronised); throughput at batch 8 in
input order (the release code has no batch size of its own for Clef; pplx `predict` defaults to 8) over the same 200;
peak GPU memory. For the open stack the reader and kNN are timed separately. Sanity: the argmax of every timed
output is compared with the scored b24B / clean cell for the same item (agreement reported, accuracy not).
A batch-8 run of a 27B model is skipped for batch 4 when its estimated peak (weights + 8 x the batch-1 activation
peak) would exceed 85 GB, which is what the unified-memory host can hold next to the A1 memory watchdog.
"""
import argparse, gzip, hashlib, importlib.util, json, os, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "paper"
N, WARM, BS = 200, 8, 8
EST_PEAK_LIMIT_GB = 85.0
SCORED = {"clef": "banking77_r14_clef_b24B.jsonl.gz", "clef_flash": "banking77_r14_clef_flash_b24B.jsonl.gz",
          "pplx": "banking77_r15_pplx_decider_b24B_sgl.jsonl.gz", "open": "banking77_24shot_clean.jsonl.gz"}
SCORED_NOTE = {"pplx": "the scored pplx b24B cell was produced by the A7 serving engine, not the release code",
               "open": "the clean run's own test dump (library BM25 retrieval; this run replays its recorded retrieval)"}
M2_FILES = ["scripts/clean_24shot.py", "quorum/fewshot.py", "quorum/prompting.py", "quorum/data.py",
            "examples/banking77_descriptions.py"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class MemWatch:
    """Lowest MemAvailable (GB) seen while active; the host is unified-memory, so this is the real headroom."""

    def __enter__(self):
        self.min, self._stop = 1e9, False
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self

    def _run(self):
        while not self._stop:
            for l in open("/proc/meminfo"):
                if l.startswith("MemAvailable"):
                    self.min = min(self.min, int(l.split()[1]) / 1048576); break
            time.sleep(0.5)

    def __exit__(self, *a):
        self._stop = True; self._t.join()


def lat_stats(sec):
    ms = np.asarray(sec) * 1000
    return {"n": len(ms), "median_ms": round(float(np.median(ms)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1),
            "min_ms": round(float(ms.min()), 1), "max_ms": round(float(ms.max()), 1), "mean_ms": round(float(ms.mean()), 1)}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def context():
    import torch, transformers
    return {"gpu": torch.cuda.get_device_name(0), "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "driver": sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "cuda": torch.version.cuda, "torch": torch.__version__, "transformers": transformers.__version__,
            "dtype": "bf16", "numpy": np.__version__}


def gpu_idle():
    import torch
    free, tot = torch.cuda.mem_get_info()
    util = sh(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    return {"gpu_util_pct_before_load": util, "cuda_free_gb_before_load": round(free / 1e9, 1)}


def sync():
    import torch
    torch.cuda.synchronize()


def timed(fn, items):
    out, sec = [], []
    for it in items:
        sync(); t = time.perf_counter(); r = fn(it); sync(); sec.append(time.perf_counter() - t); out.append(r)
    return out, sec


def peak(reset=True):
    import torch
    v = {"max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return v


def agreement(system, pred):
    """Argmax of the 200 timed outputs vs the scored cell's `pred` for the same items."""
    rows = []
    for l in gzip.open(ROOT / "results" / "predictions" / SCORED[system], "rt"):
        rows.append(json.loads(l))
        if len(rows) == N:
            break
    ref = np.array([r["pred"] for r in rows])
    return {"scored_cell": SCORED[system], "note": SCORED_NOTE.get(system), "n": N,
            "argmax_agreement": round(float((np.asarray(pred) == ref).mean()), 4),
            "differing_items": [int(i) for i in np.flatnonzero(np.asarray(pred) != ref)]}


def data_and_bodies():
    from quorum import data
    rq = _load("r14_clef").ReproRequests(); d = data.load("banking77")
    rows = d["test"][: N + WARM]
    return rq, rows, _load("r14_clef").armb_bodies(rq, rows)           # raises unless every body matches the Jev manifest


def code_hashes(extra=()):
    fs = ["scripts/r17_latency.py", "scripts/r14_clef.py"] + list(extra)
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in fs}


# ---------------- Clef / Clef-Flash ----------------
def run_clef(tag):
    import torch
    R14 = _load("r14_clef"); rq, rows, bodies = data_and_bodies()
    res = {"system": tag, "release_code": "joint_schema_model.py (load_release_model, encode_record, collate_records)"}
    res["idle_check"] = gpu_idle(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); J, model, proc, code_sha = R14.load_model(tag); sync(); res["load_seconds"] = round(time.perf_counter() - t, 1)
    res["model"], res["revision"] = R14.MODELS[tag]; res["release_code_sha256"] = code_sha
    weights = torch.cuda.memory_allocated() / 1e9; res["weights_allocated_gb"] = round(weights, 2); res["after_load"] = peak()
    pos_cache = {}

    def to_label_order(r, i):
        opt_ids, p, _ = r
        pos = pos_cache.setdefault(tuple(opt_ids), {o: j for j, o in enumerate(opt_ids)})
        return np.array([p[pos[l]] for l in rq.raw])

    one = lambda i: R14.score(J, model, proc, [bodies[i]], 1)[0]
    with MemWatch() as mw:
        timed(one, range(N, N + WARM)); torch.cuda.reset_peak_memory_stats()
        outs, sec = timed(one, range(N))
    P1 = np.stack([to_label_order(r, i) for i, r in enumerate(outs)])
    res["batch1"] = {**lat_stats(sec), "per_item_ms": [round(s * 1000, 1) for s in sec], "prompt_tokens_median": int(np.median([r[2] for r in outs])),
                     "prompt_tokens_max": int(max(r[2] for r in outs)), "peak": peak(), "min_mem_available_gb": round(mw.min, 1)}
    est = weights + BS * (res["batch1"]["peak"]["max_allocated_gb"] - weights)
    bs = BS if est <= EST_PEAK_LIMIT_GB else 4
    res["batch_size_choice"] = {"wanted": BS, "estimated_peak_gb_at_bs8": round(est, 1), "limit_gb": EST_PEAK_LIMIT_GB, "used": bs,
                                "reason": None if bs == BS else "estimated bs8 peak above the limit; batch 4 (the batch size of the scored run) used"}

    def batched(idx):
        enc = [J.encode_record(proc.tokenizer, bodies[i], processor=None) for i in idx]; out = []
        for s in range(0, len(enc), bs):
            b = J.collate_records(enc[s:s + bs], proc.tokenizer.pad_token_id, torch.device("cuda"))
            with torch.inference_mode():
                lg = model(b)
            for e, l in zip(enc[s:s + bs], lg):
                out.append((e.questions[0].option_ids, l[0].float().softmax(-1).cpu().numpy(), len(e.input_ids)))
        return out

    with MemWatch() as mw:
        batched(range(N, N + bs)); torch.cuda.reset_peak_memory_stats(); sync()
        t = time.perf_counter(); outb = batched(range(N)); sync(); dt = time.perf_counter() - t
    Pb = np.stack([to_label_order(r, i) for i, r in enumerate(outb)])
    res["throughput"] = {"batch_size": bs, "order": "input order, no length bucketing", "items": N, "seconds": round(dt, 2),
                         "items_per_s": round(N / dt, 3), "peak": peak(), "min_mem_available_gb": round(mw.min, 1)}
    res["sanity"] = {"batch1": agreement(tag, P1.argmax(1)), f"batch{bs}": agreement(tag, Pb.argmax(1))}
    res["batch1_vs_batched_argmax_agreement"] = round(float((P1.argmax(1) == Pb.argmax(1)).mean()), 4)
    res["code_sha256"] = code_hashes()
    return res


# ---------------- pplx-decider v1.1 (release DecisionModel) ----------------
def run_pplx():
    import torch
    R15 = _load("r15_pplx_decider"); rq, rows, bodies = data_and_bodies(); prows = R15.rows_of(bodies)
    res = {"system": "pplx", "release_code": "autojev.model.DecisionModel.predict (not the SGLang serving engine)"}
    res["idle_check"] = gpu_idle(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); model, code_sha = R15.load(); sync(); res["load_seconds"] = round(time.perf_counter() - t, 1)
    res["load_note"] = "the backbone is loaded straight to the GPU (device_map), as in R15; same weights and numerics"
    res["model"], res["revision"] = R15.MODEL, R15.REV; res["release_model_py_sha256"] = code_sha
    weights = torch.cuda.memory_allocated() / 1e9; res["weights_allocated_gb"] = round(weights, 2); res["after_load"] = peak()

    def label_order(p, i):
        keys = list(bodies[i]["questions"]["intent"]["criteria"]); pos = {k: j for j, k in enumerate(keys)}
        return np.array([p[pos[l]] for l in rq.raw])

    one = lambda i: model.predict([prows[i]], batch_size=1)[0]
    with MemWatch() as mw:
        timed(one, range(N, N + WARM)); torch.cuda.reset_peak_memory_stats()
        outs, sec = timed(one, range(N))
    P1 = np.stack([label_order(p, i) for i, p in enumerate(outs)])
    res["batch1"] = {**lat_stats(sec), "per_item_ms": [round(s * 1000, 1) for s in sec], "peak": peak(), "min_mem_available_gb": round(mw.min, 1)}
    est = weights + BS * (res["batch1"]["peak"]["max_allocated_gb"] - weights)
    bs = BS if est <= EST_PEAK_LIMIT_GB else 4
    res["batch_size_choice"] = {"release_default": 8, "estimated_peak_gb_at_bs8": round(est, 1), "limit_gb": EST_PEAK_LIMIT_GB, "used": bs,
                                "reason": None if bs == BS else "estimated bs8 peak above the limit; batch 4 (the batch size of the scored run) used"}
    with MemWatch() as mw:
        model.predict(prows[N:N + bs], batch_size=bs); torch.cuda.reset_peak_memory_stats(); sync()
        t = time.perf_counter(); outb = model.predict(prows[:N], batch_size=bs); sync(); dt = time.perf_counter() - t
    Pb = np.stack([label_order(p, i) for i, p in enumerate(outb)])
    res["throughput"] = {"batch_size": bs, "order": "input order, as `predict` iterates", "items": N, "seconds": round(dt, 2),
                         "items_per_s": round(N / dt, 3), "peak": peak(), "min_mem_available_gb": round(mw.min, 1)}
    res["sanity"] = {"batch1": agreement("pplx", P1.argmax(1)), f"batch{bs}": agreement("pplx", Pb.argmax(1))}
    res["batch1_vs_batched_argmax_agreement"] = round(float((P1.argmax(1) == Pb.argmax(1)).mean()), 4)
    res["temperature"], res["attention_mode"], res["pooling"] = model.temperature, model.attention_mode, model.pooling
    res["code_sha256"] = code_hashes(["scripts/r15_pplx_decider.py"])
    return res


# ---------------- the open stack ----------------
def run_open():
    import torch
    C = _load("clean_24shot"); sel_bytes = (C.OUT / "selection_scores.json").read_bytes(); sel = json.loads(sel_bytes)
    win, prm_rec = sel["selected"], sel["refit_on_full_selection_split"]
    assert (win["member_set"], win["rule"], win["option_text"], win["orders"]) == ("M2", "weighted_prob_mean", "descriptions", 1)
    for f in M2_FILES:
        assert sel["code"]["sha256"][f] == hashlib.sha256((ROOT / f).read_bytes()).hexdigest(), f"{f} changed since the selection"
    prm = {"T": [prm_rec["temperatures"][n] for n in C.MEMBER_SETS["M2"]], "w": prm_rec["weights"]}
    from quorum import data
    d = data.load("banking77"); rows = d["test"][: N + WARM]
    clean = [json.loads(l) for l in gzip.open(ROOT / "results" / "predictions" / "banking77_24shot_clean.jsonl.gz", "rt")][: N + WARM]
    res = {"system": "open", "path": "scripts/clean_24shot.py selection (M2 = reader + kNN, weighted_prob_mean, descriptions, 1 order)"}
    res["idle_check"] = gpu_idle(); torch.cuda.reset_peak_memory_stats()
    _, raw, names, train = C.load_train(); S = C.load_split(); Sset = set(S); P = [i for i in range(len(train)) if i not in Sset]
    t = time.perf_counter(); eng = C.build_engine(names, [train[i] for i in P]); sync()
    res["load_and_index_seconds"] = round(time.perf_counter() - t, 1); res["after_load"] = peak()
    t = time.perf_counter(); eng.index([train[i] for i in P]); sync(); res["index_pool_seconds"] = round(time.perf_counter() - t, 1)
    res["model"] = {"base": C.BASE, "base_revision": C.BASE_REV, "adapter": C.ADAPTER, "adapter_revision": C.ADAPTER_REV, "knn_embedder": C.KNN_EMBEDDER}
    res["weights_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    eng.options = list(C.descriptions(raw)); wrows = rows[N:]; trows = rows[:N]
    t = time.perf_counter(); lib = [eng._shots(r["text"]) for r in trows]; res["retrieval_bm25_ms_per_item"] = round(1000 * (time.perf_counter() - t) / N, 2)
    res["retrieval_note"] = "library BM25 over the 9,079-row pool, CPU, measured on its own; the timed runs replay the clean run's recorded retrieval"
    ppos = {tj: k for k, tj in enumerate(P)}                                      # train index -> position in the indexed pool
    rec = {r["text"]: [ppos[j] for j in c["retrieved_idx"]] for r, c in zip(rows, clean)}
    eng.bm = type("Recorded", (), {"top": lambda self, text, k: list(rec[text])})()
    assert all(eng.shots_idx(r["text"]) == rec[r["text"]] for r in trows)
    reader = lambda r: eng.incontext_logits_per_perm([r], 1, bs=1)[0]            # (1, K)
    knn = lambda r: eng._knn_logits([r])                                         # (1, K)
    with MemWatch() as mw:
        timed(reader, wrows); timed(knn, wrows); torch.cuda.reset_peak_memory_stats()
        rl, rsec = timed(reader, trows); kl, ksec = timed(knn, trows)
    L_r, L_k = np.concatenate(rl), np.concatenate(kl)
    comb = C.combine([L_r, L_k], prm, "weighted_prob_mean"); pred1 = comb.argmax(1)
    res["batch1"] = {"reader": {**lat_stats(rsec), "per_item_ms": [round(s * 1000, 1) for s in rsec]},
                     "knn": {**lat_stats(ksec), "per_item_ms": [round(s * 1000, 1) for s in ksec]},
                     "reader_plus_knn": lat_stats(np.asarray(rsec) + np.asarray(ksec)),
                     "note": "reader and kNN timed in separate passes; reader_plus_knn adds the per-item times", "peak": peak(),
                     "min_mem_available_gb": round(mw.min, 1)}
    with MemWatch() as mw:
        eng.incontext_logits_per_perm(wrows, 1, bs=BS); eng._knn_logits(wrows); torch.cuda.reset_peak_memory_stats(); sync()
        t = time.perf_counter(); Lb_r = eng.incontext_logits_per_perm(trows, 1, bs=BS)[0]; sync(); dr = time.perf_counter() - t
        t = time.perf_counter(); Lb_k = eng._knn_logits(trows); sync(); dk = time.perf_counter() - t
    predb = C.combine([Lb_r, Lb_k], prm, "weighted_prob_mean").argmax(1)
    res["throughput"] = {"batch_size": BS, "items": N, "reader_seconds": round(dr, 2), "knn_seconds": round(dk, 2),
                         "items_per_s_reader_plus_knn": round(N / (dr + dk), 3), "peak": peak(), "min_mem_available_gb": round(mw.min, 1)}
    res["sanity"] = {"batch1": agreement("open", pred1), f"batch{BS}": agreement("open", predb)}
    res["batch1_vs_batched_argmax_agreement"] = round(float((pred1 == predb).mean()), 4)
    res["selection_file_sha256"] = hashlib.sha256(sel_bytes).hexdigest()
    res["code_sha256"] = code_hashes(M2_FILES)
    return res


def phase_run(a):
    out = {"run": {"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    out["context"] = context()
    res = {"clef": lambda: run_clef("clef"), "clef_flash": lambda: run_clef("clef_flash"), "pplx": run_pplx, "open": run_open}[a.system]()
    out.update(res); out["run"]["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"latency_r17_{a.system}.json").write_text(json.dumps(out, indent=1))
    b1 = out["batch1"]; med = b1["reader_plus_knn"]["median_ms"] if a.system == "open" else b1["median_ms"]
    print("R17_DONE", a.system, "batch1 median_ms", med, "load_s", out.get("load_seconds", out.get("load_and_index_seconds")), flush=True)


def phase_merge(a):
    parts = {s: json.loads((OUT / f"latency_r17_{s}.json").read_text()) for s in ("open", "clef_flash", "clef", "pplx")}
    ctx = {s: p["context"] for s, p in parts.items()}
    same = all(c == next(iter(ctx.values())) for c in ctx.values())
    (OUT / "latency_r17.json").write_text(json.dumps({"design": "scripts/r17_latency.py docstring; one GB10, bf16, one process per system, nothing else on the GPU",
                                                       "context_identical_across_systems": same, "systems": parts}, indent=1))
    print("merged; context identical across systems:", same)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["run", "merge"]); ap.add_argument("--system", choices=["open", "clef_flash", "clef", "pplx"])
    a = ap.parse_args(); {"run": phase_run, "merge": phase_merge}[a.phase](a)
