#!/usr/bin/env python3
"""R14 — Cloudflare Clef / Clef-Flash (open, Jev-API-compatible decision models) under this study's protocol.

Settings (all on the full official test sets unless noted; protocol: provenance/clef_eval_protocol.md):
  zs     7 datasets x {names, descriptions}: state {"message": text}; one `choice` question whose criteria map each
         class name to itself (names) or to this repo's description (descriptions); the same instruction for every
         dataset. Comparable with the zero-shot ensemble (R3) and the clean-NLI ensemble (R13).
  b24    Banking77, the public reproduction's `retrieved24` request body, rebuilt here and verified byte-identical
         to the reproduction's own builder (phase `parity`): 24 BM25 examples (word+bigram, <=4 per class) from all
         10,003 training rows, its instruction and its 77 definitions. Per-item comparable with the reproduction's
         Jev predictions and with the clean 24-shot run.
  defs   Banking77, the reproduction's `definitions` request body (its definitions, no examples).
  train  the R12 pre-registered train-side sample (7 x 1,000) with the `zs` request: a memorisation probe
         (train accuracy far above test accuracy would indicate training on these splits).

    python scripts/r14_clef.py parity                                     # CPU; reads the reproduction at run time
    python scripts/r14_clef.py batchcheck --model clef_flash              # batch-1 vs batch-8 logits
    python scripts/r14_clef.py run --model clef_flash --setting b24 [--limit N] [--bs 8]

Scores come from the model's per-option logits (softmax per question), not from the rounded `systemone` response.
The reproduction is read at run time and never stored in this repository.
"""
import argparse, collections, gzip, hashlib, importlib.util, io, json, math, os, re, sys, tempfile, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = {"clef": ("Cloudflare/clef", "2f3de3dd85f379784083b0814d997ab627200f0c"),
          "clef_flash": ("Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6")}
REPRO = "https://raw.githubusercontent.com/simonmesmith/jev-banking77-experiment/5cac4ff/"
REPRO_FILES = ["frozen-v1/src/method.py", "frozen-v1/src/data.py", "data/train.json", "data/categories.json",
               "data/test_inputs.json", "configs/descriptions.json"]
ZS_INSTRUCTION = ("Which ONE intent best describes `message`? Every message belongs to exactly one of the listed "
                  "categories.")
DATASETS = ["banking77", "clinc150", "hwu64", "massive", "mtop", "snips", "bitext"]
DESC_MODULE = {"clinc150": "clinc"}
OUTP, OUT = ROOT / "results" / "predictions", ROOT / "results" / "r14"
g7 = lambda v: [float(f"{x:.7g}") for x in v]


def fetch(rel):
    return urllib.request.urlopen(REPRO + rel, timeout=120).read()


def load_desc(ds, labels):
    mod = DESC_MODULE.get(ds, ds)
    spec = importlib.util.spec_from_file_location(mod, ROOT / "examples" / f"{mod}_descriptions.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    d = next(v for k, v in vars(m).items() if k.endswith("_DESCRIPTIONS") and isinstance(v, dict))
    norm = {str(k).replace("_", " ").strip().lower(): v for k, v in d.items()}
    return [norm[str(l).replace("_", " ").strip().lower()] for l in labels]


# ------------------------------------------------------------------ the reproduction's request bodies (clean-room)
def _terms(text):
    w = re.findall(r"[a-z0-9]+", text.lower()); return w + [a + "_" + b for a, b in zip(w, w[1:])]


class ReproRequests:
    """Rebuilds the reproduction's `definitions` / `retrieved24` request bodies from the pinned HF dataset plus the
    reproduction's definitions and instruction (read at run time). Equality with the reproduction's own builder is
    checked by phase `parity`."""
    INSTRUCTION = None

    def __init__(self):
        tr = __import__("datasets").load_dataset("legacy-datasets/banking77",
                                                 revision="f54121560de48f2852f90be299010d1d6dc612ec", split="train")
        self.raw = list(tr.features["label"].names)
        self.rows = [{"text": r["text"], "label": self.raw[int(r["label"])]} for r in tr]
        self.desc = json.loads(fetch("configs/descriptions.json"))
        self.labels = sorted(self.desc)
        assert set(self.labels) == set(self.raw)
        meth = fetch("frozen-v1/src/method.py").decode()
        m = re.search(r"'instructions':'(.*?)','criteria'", meth)
        type(self).INSTRUCTION = m.group(1)
        self.post, self.len = collections.defaultdict(list), []
        for i, r in enumerate(self.rows):
            ts = _terms(r["text"]); self.len.append(len(ts))
            for t, n in collections.Counter(ts).items():
                self.post[t].append((i, n))
        self.avg = sum(self.len) / len(self.rows); N = len(self.rows)
        self.idf = {t: math.log(1 + (N - len(v) + .5) / (len(v) + .5)) for t, v in self.post.items()}

    def _rank(self, text):
        sc = collections.defaultdict(float)
        for t in set(_terms(text)):
            for i, tf in self.post.get(t, []):
                sc[i] += self.idf[t] * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.len[i] / self.avg))
        return sorted(range(len(self.rows)), key=lambda i: (-sc[i], i))

    def body(self, text, variant):
        selected = []
        if variant == "retrieved24":
            cnt = collections.Counter()
            for i in self._rank(text):
                r = self.rows[i]
                if cnt[r["label"]] >= 4:
                    continue
                selected.append(i); cnt[r["label"]] += 1
                if len(selected) == 24:
                    break
        state = {"customer_message": text}
        q = {"type": "choice", "instructions": self.INSTRUCTION, "criteria": {k: self.desc[k] for k in self.labels}}
        payload = {"model": "jev-1.13.0", "state": state, "questions": {"intent": q}}
        while True:
            if selected:
                state["labeled_examples"] = [{"message": self.rows[i]["text"], "intent": self.rows[i]["label"]}
                                             for i in selected]
            else:
                state.pop("labeled_examples", None)
            if len(json.dumps(payload, ensure_ascii=True).encode()) <= 30000:
                break
            selected.pop()
        return payload, selected


def phase_parity(args):
    """Run the reproduction's OWN builder (fetched into a temp dir, executed, deleted) and compare every body."""
    rq = ReproRequests()
    with tempfile.TemporaryDirectory() as tmp:
        for rel in REPRO_FILES:
            p = Path(tmp) / rel.replace("frozen-v1/", ""); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(fetch(rel))
        sys.path.insert(0, str(Path(tmp) / "src"))
        import data as rdata, method as rmethod                      # noqa: E402  (reproduction code, run-time only)
        rdata.ROOT = Path(tmp); rmethod.ROOT = Path(tmp)
        support = rdata.read(Path(tmp) / "data/train.json")
        tests = rdata.read(Path(tmp) / "data/test_inputs.json")
        te = __import__("datasets").load_dataset("legacy-datasets/banking77",
                                                 revision="f54121560de48f2852f90be299010d1d6dc612ec", split="test")
        texts = [t["text"] for t in tests] if isinstance(tests[0], dict) else list(tests)
        assert texts == list(te["text"]), "reproduction test inputs differ from the pinned test split"
        res = {}
        for variant in ("definitions", "retrieved24"):
            M = rmethod.Method(support, variant); diff = 0
            for t in texts[: args.limit or len(texts)]:
                theirs, _ = M.payload(t); mine, _ = rq.body(t, variant)
                diff += int(json.dumps(theirs, sort_keys=True) != json.dumps(mine, sort_keys=True))
            res[variant] = {"items": min(len(texts), args.limit or len(texts)), "bodies_differing": diff}
            print(variant, res[variant], flush=True)
        sys.path.remove(str(Path(tmp) / "src"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity.json").write_text(json.dumps({"reproduction": REPRO, "result": res}, indent=1))
    assert all(v["bodies_differing"] == 0 for v in res.values()), "request bodies differ"
    print("PARITY_OK", flush=True)


# ------------------------------------------------------------------ model
def load_model(tag):
    from huggingface_hub import snapshot_download
    name, rev = MODELS[tag]
    path = snapshot_download(name, revision=rev)
    code = Path(path) / "joint_schema_model.py"
    sys.path.insert(0, path)
    import joint_schema_model as J                                    # noqa: E402
    model, proc = J.load_release_model(path, device="cuda")
    return J, model, proc, hashlib.sha256(code.read_bytes()).hexdigest()


def score(J, model, proc, bodies, bs):
    """Per-body probability vectors over the question's options, in option_ids order (sorted ids)."""
    import torch
    out, n_tok = [], []
    enc = [J.encode_record(proc.tokenizer, b, processor=None) for b in bodies]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i].input_ids))          # length-bucketed batches
    res = [None] * len(enc)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = J.collate_records([enc[i] for i in idx], proc.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            logits = model(batch)
        for i, lg in zip(idx, logits):
            res[i] = (enc[i].questions[0].option_ids, lg[0].float().softmax(-1).cpu().numpy(), len(enc[i].input_ids))
    return res


def zs_body(text, names, label_texts):
    return {"model": "clef", "state": {"message": text},
            "questions": {"intent": {"type": "choice", "instructions": ZS_INSTRUCTION,
                                     "criteria": {n: t for n, t in zip(names, label_texts)}}}}


def phase_batchcheck(args):
    J, model, proc, _ = load_model(args.model)
    from quorum import data
    d = data.load("banking77"); names = [str(l) for l in d["labels"]]
    bodies = [zs_body(r["text"], names, names) for r in d["test"][:16]]
    a = score(J, model, proc, bodies, 1); b = score(J, model, proc, bodies, 8)
    dev = max(float(np.abs(x[1] - y[1]).max()) for x, y in zip(a, b))
    flips = sum(int(x[1].argmax() != y[1].argmax()) for x, y in zip(a, b))
    print(f"BATCHCHECK max|p_bs1 - p_bs8| = {dev:.2e}, argmax flips {flips}/16", flush=True)


def metrics(P, gold):
    pred = P.argmax(1); K = P.shape[1]; hit = pred == gold
    f1 = []
    for c in range(K):
        tp = int(((pred == c) & (gold == c)).sum()); a = int((gold == c).sum()); p = int((pred == c).sum())
        if a:
            f1.append(2 * tp / (a + p) if a + p else 0.0)
    conf = P.max(1); ece = 0.0
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10; s = (conf >= lo) & ((conf < hi) if i < 9 else (conf <= hi))
        if s.any():
            ece += s.mean() * abs(hit[s].mean() - conf[s].mean())
    Pc = np.clip(P, 1e-15, 1)
    return {"accuracy": round(float(hit.mean()), 4), "correct": int(hit.sum()), "macro_f1": round(float(np.mean(f1)), 4),
            "log_loss": round(float(-np.mean(np.log(Pc[np.arange(len(gold)), gold]))), 4),
            "brier": round(float(np.mean(((P - np.eye(K)[gold]) ** 2).sum(1))), 4), "ece": round(float(ece), 4)}


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def repro_hits():
    import csv
    txt = fetch("results/predictions.csv").decode()
    rep = {int(r["id"].split("-")[1]): r for r in csv.DictReader(io.StringIO(txt))}
    return np.array([rep[i]["correct"] == "True" for i in range(3080)])


def phase_run(args):
    import torch, transformers
    from quorum import data
    J, model, proc, code_sha = load_model(args.model); tag = f"{args.model}_{args.setting}"
    OUTP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    jobs = []                                                         # (name, rows[(text,gold)], bodies, label order)
    if args.setting in ("zs", "train"):
        ids12 = json.loads((ROOT / "results" / "r12" / "sample_ids.json").read_text())
        for ds in DATASETS:
            d = data.load(ds); names = [str(l) for l in d["labels"]]
            rows = d["test"] if args.setting == "zs" else [d["train"][i] for i in ids12["ids"][ds]]
            rows = rows[: args.limit or len(rows)]
            for variant, lt in [("names", names), ("descriptions", load_desc(ds, names))]:
                jobs.append((f"{ds}_{variant}", rows, [zs_body(r["text"], names, lt) for r in rows], names))
    else:
        rq = ReproRequests(); d = data.load("banking77")
        rows = d["test"][: args.limit or len(d["test"])]
        variant = "retrieved24" if args.setting == "b24" else "definitions"
        jobs.append(("banking77", rows, [rq.body(r["text"], variant)[0] for r in rows], rq.raw))
    summary = {}
    for name, rows, bodies, label_order in jobs:
        t0 = time.time(); res = score(J, model, proc, bodies, args.bs)
        P = np.zeros((len(rows), len(label_order)))
        for i, (opt_ids, p, _) in enumerate(res):
            pos = {o: j for j, o in enumerate(opt_ids)}
            P[i] = [p[pos[l]] for l in label_order]                   # back to dataset label order
        gold = np.array([int(r["label"]) for r in rows]); rec = metrics(P, gold)
        rec["prompt_tokens_max"] = int(max(x[2] for x in res)); rec["seconds"] = round(time.time() - t0, 1)
        rec["ms_per_item"] = round(1000 * rec["seconds"] / len(rows), 1)
        fn = (OUTP if args.setting != "train" else OUT) / f"{name}_r14_{tag}{'_limit' + str(args.limit) if args.limit else ''}.jsonl.gz"
        with gzip.open(fn, "wt") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"idx": i, "text": r["text"], "gold": int(gold[i]), "pred": int(P[i].argmax()),
                                    "probs": g7(P[i])}) + "\n")
        if name == "banking77" and not args.limit:
            th, h = repro_hits(), P.argmax(1) == gold
            b, c = int((h & ~th).sum()), int((~h & th).sum())
            rec["mcnemar_vs_reproduction"] = {"b": b, "c": c, "p_two_sided_exact": mcnemar_exact(b, c)}
        summary[name] = rec; print(f"[{tag}/{name}] {json.dumps(rec)}", flush=True)
    meta = {"model": MODELS[args.model][0], "revision": MODELS[args.model][1], "setting": args.setting,
            "limit": args.limit, "batch_size": args.bs, "dtype": "bf16", "custom_code_sha256": code_sha,
            "zs_instruction": ZS_INSTRUCTION, "repro_instruction": ReproRequests.INSTRUCTION, "results": summary,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "device": torch.cuda.get_device_name(0), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / f"summary_{tag}{'_limit' + str(args.limit) if args.limit else ''}.json").write_text(json.dumps(meta, indent=1))
    print("R14_DONE", tag, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["parity", "batchcheck", "run"])
    ap.add_argument("--model", choices=sorted(MODELS)); ap.add_argument("--setting", choices=["zs", "b24", "defs", "train"])
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args(); {"parity": phase_parity, "batchcheck": phase_batchcheck, "run": phase_run}[a.phase](a)
