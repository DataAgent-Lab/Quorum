#!/usr/bin/env python3
"""Cost-section numbers (latency on one GPU type), recomputed from the per-item timings.

Writes paper/generated/cost.json and paper/generated/numbers_cost.tex (LaTeX macros \\ct...). Claims ledger row E2.

Inputs
  results/paper/latency_r17.json      R17: open stack, Clef-Flash, Clef, pplx-decider v1.1 on one GB10, bf16,
                                      release code, the first 200 Banking77 b24B bodies (scripts/r17_latency.py)
  docs/phases/2.0/jev/latency/calls_distinct.jsonl.gz   the closed API's latency probe (distinct bodies), used for
                                      one descriptive sentence only; not comparable (network, unknown hardware)

Every median, p95, minimum and maximum is recomputed from the per-item timings and asserted equal to the recorded
summary within 0.15 ms (the per-item times are stored rounded to 0.1 ms; the summary used unrounded times); the
reported values are the recorded ones; throughput is recomputed from items and seconds; the code hashes recorded in the file are checked against
this repository's files (current or any committed version); the argmax sanity checks are carried through.

    python paper/scripts/numbers_cost.py
"""
from __future__ import annotations
import gzip, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from numbers_24shot import thousands  # noqa: E402

GEN = ROOT / "paper" / "generated"
SYS = {"Open": "open", "Flash": "clef_flash", "Clef": "clef", "Pplx": "pplx"}


def versions(f):
    """sha256 of the file now and of every committed version (a later docstring edit must not fail the check)."""
    hs = {hashlib.sha256((ROOT / f).read_bytes()).hexdigest()}
    revs = subprocess.run(["git", "log", "--format=%H", "--", f], cwd=ROOT, capture_output=True, text=True).stdout.split()
    for rev in revs:
        blob = subprocess.run(["git", "show", f"{rev}:{f}"], cwd=ROOT, capture_output=True).stdout
        hs.add(hashlib.sha256(blob).hexdigest())
    return hs


def stats(ms):
    ms = np.asarray(ms, float)
    return {"n": len(ms), "median_ms": round(float(np.median(ms)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1),
            "min_ms": round(float(ms.min()), 1), "max_ms": round(float(ms.max()), 1)}


def main():
    d = json.loads((ROOT / "results/paper/latency_r17.json").read_text())
    assert d["context_identical_across_systems"]
    out, mac = {}, {}

    def put(key, val, macro=None):
        out[key] = val
        if macro is not None:
            mac[key] = macro

    ctx = None
    for m, s in SYS.items():
        r = d["systems"][s]
        ctx = ctx or r["context"]; assert r["context"] == ctx, m
        assert int(r["idle_check"]["gpu_util_pct_before_load"]) == 0, m
        for f, h in r["code_sha256"].items():                 # needs the repository's history (not in the supplement)
            assert not (ROOT / ".git").exists() or h in versions(f), (m, f)
        b1 = r["batch1"]
        if m == "Open":
            rd, kn = b1["reader"], b1["knn"]
            for part in (rd, kn):
                st = stats(part["per_item_ms"]); assert all(abs(st[k] - part[k]) <= 0.15 for k in st), (m, st)
            both = np.add(rd["per_item_ms"], kn["per_item_ms"]); st = stats(both)
            assert all(abs(st[k] - b1["reader_plus_knn"][k]) <= 0.15 for k in st if k != "n"), (st, b1["reader_plus_knn"])
            put("OpenReaderMedian", rd["median_ms"], f"{rd['median_ms']:.0f}"); put("OpenKnnMedian", kn["median_ms"], f"{kn['median_ms']:.0f}")
            put("OpenBmMs", r["retrieval_bm25_ms_per_item"], f"{r['retrieval_bm25_ms_per_item']:.0f}")
            load = r["load_and_index_seconds"]; tp = r["throughput"]
            thr = tp["items"] / (tp["reader_seconds"] + tp["knn_seconds"])
            assert abs(thr - tp["items_per_s_reader_plus_knn"]) < 2e-3
        else:
            st = stats(b1["per_item_ms"]); assert all(abs(st[k] - b1[k]) <= 0.15 for k in st), (m, st)
            load = r["load_seconds"]; tp = r["throughput"]
            thr = tp["items"] / tp["seconds"]; assert abs(thr - tp["items_per_s"]) < 2e-3
            assert r["batch_size_choice"]["used"] == tp["batch_size"]
        assert tp["batch_size"] == 8 and tp["items"] == 200
        rec = b1["reader_plus_knn"] if m == "Open" else b1
        med = rec["median_ms"]; p95 = rec["p95_ms"]
        put(f"{m}Load", load, f"{load:.0f}"); put(f"{m}Median", med, thousands(round(med)))
        put(f"{m}Pninetyfive", p95, thousands(round(p95))); put(f"{m}Thr", thr, f"{thr:.2f}")
        put(f"{m}PerHour", thr * 3600, thousands(int(round(thr * 3600, -1))))
        put(f"{m}MemOne", b1["peak"]["max_allocated_gb"], f"{b1['peak']['max_allocated_gb']:.1f}")
        put(f"{m}MemEight", tp["peak"]["max_allocated_gb"], f"{tp['peak']['max_allocated_gb']:.1f}")
        sa = r["sanity"]
        put(f"{m}AgreeOne", sa["batch1"]["argmax_agreement"], f"{100 * sa['batch1']['argmax_agreement']:.1f}")
        put(f"{m}AgreeEight", sa["batch8"]["argmax_agreement"], f"{100 * sa['batch8']['argmax_agreement']:.1f}")
        out[f"{m}N"] = st["n"]
    assert len({out[f"{m}N"] for m in SYS}) == 1
    put("N", out["OpenN"], str(out["OpenN"])); put("Batch", 8, "8")
    for m in ("Flash", "Clef", "Pplx"):                       # how many times slower than the open stack, batch 1
        put(f"{m}TimesOpen", out[f"{m}Median"] / out["OpenMedian"], f"{out[f'{m}Median'] / out['OpenMedian']:.1f}")
    toks = [d["systems"][s]["batch1"].get("prompt_tokens_max") for s in ("clef_flash", "clef")]
    put("PromptTokMax", max(toks), thousands(max(toks)))
    put("Gpu", ctx["gpu"], ctx["gpu"].replace("NVIDIA ", ""))

    # closed API, distinct-body probe at the size of these bodies (descriptive; not comparable)
    calls = [json.loads(l) for l in gzip.open(ROOT / "docs/phases/2.0/jev/latency/calls_distinct.jsonl.gz", "rt")]
    four = [c for c in calls if c["tag"] == "seq" and c["target"] == 4000 and c.get("char_scale", 1.0) == 1.0]
    assert len(four) == 5 and all(c["status"] == 200 for c in four)
    jm = float(np.median([c["latency_s"] for c in four])) * 1000
    put("JevFourKMedian", jm, f"{jm:.0f}")
    put("JevFourKTokLo", min(c["input_tokens"] for c in four), thousands(min(c["input_tokens"] for c in four)))
    put("JevFourKTokHi", max(c["input_tokens"] for c in four), thousands(max(c["input_tokens"] for c in four)))

    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "cost.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = ["% GENERATED by paper/scripts/numbers_cost.py from the per-item timings -- do not edit by hand."]
    lines += [f"\\newcommand{{\\ct{k}}}{{{mac[k]}}}" for k in sorted(mac)]
    (GEN / "numbers_cost.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(mac)} macros")


if __name__ == "__main__":
    main()
