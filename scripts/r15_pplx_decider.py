#!/usr/bin/env python3
"""R15 — perplexity-ai/pplx-decider-v1-27b under the same protocol as R14 (Clef): same items, same request
bodies, same metrics (provenance/clef_eval_protocol.md, amendment A2).

pplx-decider takes {state, question} with a Jev-style question ({type, instructions, criteria}). It lists the
options as letter codes in the criteria's own order and reads one logit per code at the last position. The scores
are its calibrated probabilities: the release's saved temperature, as its own `predict` applies it.

Settings, mirroring R14:
  zs     7 datasets x {names, descriptions}: state {"message": text}, the R14 instruction, criteria name->name
         or name->description in the dataset's label order;
  b24    Banking77: state and question taken verbatim from the reproduction's retrieved24 body (byte-identical to
         what Jev received; R14 phase parity);
  defs   Banking77: the reproduction's definitions body;
  train  the R12 pre-registered train-side sample with the zs request (memorisation probe).

    python scripts/r15_pplx_decider.py batchcheck
    python scripts/r15_pplx_decider.py run --setting b24 [--limit N] [--bs 4] [--datasets a,b]
"""
import argparse, gzip, hashlib, importlib.util, json, socket, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("r14", ROOT / "scripts" / "r14_clef.py")
R14 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R14)
MODEL, REV = "perplexity-ai/pplx-decider-v1-27b", "5117a6c7fe73b19308dc1a6b0fb529a40c2ecad4"
FILES = ["model-*.safetensors", "model.safetensors.index.json", "readout.safetensors", "config.json",
         "decision_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja", "source/src/autojev/__init__.py", "source/src/autojev/model.py",
         "source/src/autojev/types.py"]
TAG = "pplx_decider"


def load():
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(MODEL, revision=REV, allow_patterns=FILES))
    sys.path.insert(0, str(path / "source" / "src"))
    import autojev.model as AM                                                     # noqa: E402 (release code)
    code = hashlib.sha256((path / "source" / "src" / "autojev" / "model.py").read_bytes()).hexdigest()
    # The release loads the backbone on the CPU and then moves it to the GPU, which on a unified-memory host
    # briefly needs ~2x the weights. Load straight to the GPU instead (device placement only; same weights and
    # numerics), for the duration of construction.
    orig = AM.Qwen3_5Model.from_pretrained
    AM.Qwen3_5Model.from_pretrained = lambda *a, **k: orig(*a, **({"device_map": {"": "cuda"}} | k))
    try:
        model = AM.DecisionModel(checkpoint=path, device="cuda")
    finally:
        AM.Qwen3_5Model.from_pretrained = orig
    return model, code


def rows_of(bodies):
    """Jev/SystemOne request body -> pplx-decider input row (state + the single question, unchanged)."""
    out = []
    for b in bodies:
        (q,) = b["questions"].values()
        out.append({"state": b["state"], "question": q})
    return out


def score(model, bodies, bs):
    """(N, K) probabilities in each body's criteria order, scored in length-bucketed batches."""
    from autojev.model import decision_messages
    rows = rows_of(bodies)
    tok = model.processor.tokenizer
    lens = [len(tok(model.processor.apply_chat_template(decision_messages(r, model.codes), tokenize=False,
                                                        add_generation_prompt=True, enable_thinking=False),
                    add_special_tokens=False).input_ids) for r in rows]          # as the release's prepare()
    order = sorted(range(len(rows)), key=lambda i: lens[i]); res = [None] * len(rows)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        for i, p in zip(idx, model.predict([rows[i] for i in idx], batch_size=len(idx))):
            res[i] = np.asarray(p, float)
    return res, lens


def phase_batchcheck(args):
    model, _ = load()
    from quorum import data
    d = data.load("banking77"); names = [str(l) for l in d["labels"]]
    bodies = [R14.zs_body(r["text"], names, names) for r in d["test"][:16]]
    a, _ = score(model, bodies, 1); b, _ = score(model, bodies, 8)
    dev = max(float(np.abs(x - y).max()) for x, y in zip(a, b)); flips = sum(int(x.argmax() != y.argmax()) for x, y in zip(a, b))
    print(f"BATCHCHECK max|p_bs1 - p_bs8| = {dev:.2e}, argmax flips {flips}/16", flush=True)


def phase_run(args):
    import torch, transformers
    from quorum import data
    model, code_sha = load(); tag = f"{TAG}_{args.setting}"
    part = f"_part-{args.datasets.replace(',', '-')}" if args.datasets else ""
    lim = f"_limit{args.limit}" if args.limit else ""
    R14.OUTP.mkdir(parents=True, exist_ok=True); OUT = ROOT / "results" / "r15"; OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.setting in ("zs", "train"):
        ids12 = json.loads((ROOT / "results" / "r12" / "sample_ids.json").read_text())
        for ds in (args.datasets.split(",") if args.datasets else R14.DATASETS):
            assert ds in R14.DATASETS, ds
            d = data.load(ds); names = [str(l) for l in d["labels"]]
            rows = d["test"] if args.setting == "zs" else [d["train"][i] for i in ids12["ids"][ds]]
            rows = rows[: args.limit or len(rows)]
            for variant, lt in [("names", names), ("descriptions", R14.load_desc(ds, names))]:
                jobs.append((f"{ds}_{variant}", rows, [R14.zs_body(r["text"], names, lt) for r in rows], names))
    else:
        rq = R14.ReproRequests(); d = data.load("banking77"); rows = d["test"][: args.limit or len(d["test"])]
        variant = "retrieved24" if args.setting == "b24" else "definitions"
        jobs.append(("banking77", rows, [rq.body(r["text"], variant)[0] for r in rows], rq.raw))
    summary = {}
    for name, rows, bodies, label_order in jobs:
        t0 = time.time(); res, lens = score(model, bodies, args.bs)
        P = np.zeros((len(rows), len(label_order)))
        for i, (b, p) in enumerate(zip(bodies, res)):
            (q,) = b["questions"].values(); keys = list(q["criteria"])           # criteria order = option order
            pos = {k: j for j, k in enumerate(keys)}; P[i] = [p[pos[l]] for l in label_order]
        gold = np.array([int(r["label"]) for r in rows]); rec = R14.metrics(P, gold)
        rec.update({"prompt_tokens_max": int(max(lens)), "seconds": round(time.time() - t0, 1),
                    "ms_per_item": round(1000 * (time.time() - t0) / len(rows), 1)})
        fn = (R14.OUTP if args.setting != "train" else OUT) / f"{name}_r15_{tag}{lim}.jsonl.gz"
        with gzip.open(fn, "wt") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"idx": i, "text": r["text"], "gold": int(gold[i]), "pred": int(P[i].argmax()),
                                    "probs": R14.g7(P[i])}) + "\n")
        if name == "banking77" and not args.limit:
            th, h = R14.repro_hits(), P.argmax(1) == gold
            b, c = int((h & ~th).sum()), int((~h & th).sum())
            rec["mcnemar_vs_reproduction"] = {"b": b, "c": c, "p_two_sided_exact": R14.mcnemar_exact(b, c)}
        summary[name] = rec; print(f"[{tag}/{name}] {json.dumps(rec)}", flush=True)
    meta = {"model": MODEL, "revision": REV, "setting": args.setting, "limit": args.limit, "batch_size": args.bs,
            "datasets": args.datasets or "all", "temperature": model.temperature, "release_model_py_sha256": code_sha,
            "zs_instruction": R14.ZS_INSTRUCTION, "results": summary,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "r14_code_sha256": hashlib.sha256((ROOT / "scripts" / "r14_clef.py").read_bytes()).hexdigest(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "device": torch.cuda.get_device_name(0), "host": socket.gethostname(),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / f"summary_{tag}{part}{lim}.json").write_text(json.dumps(meta, indent=1)); print("R15_DONE", tag, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["batchcheck", "run"])
    ap.add_argument("--setting", choices=["zs", "b24", "defs", "train"]); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--datasets", default="")
    a = ap.parse_args(); {"batchcheck": phase_batchcheck, "run": phase_run}[a.phase](a)
