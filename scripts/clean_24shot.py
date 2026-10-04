#!/usr/bin/env python3
"""Clean Banking77 24-shot selection + ONE test run, exactly as pre-registered in
provenance/banking77_24shot_clean_protocol.md (commit 8455e7e).

    python scripts/clean_24shot.py split            # writes the selection split; loads no model
    python scripts/clean_24shot.py smoke            # pipeline check on 8 pool rows; output discarded
    python scripts/clean_24shot.py select           # selection-split logits (cached) + 48-candidate selection
    python scripts/clean_24shot.py test             # refuses unless selection is written; runs ONLY the winner

Order of operations guarantees: the selection split is written before any model loads; test rows are not
loaded at all until results/clean/selection_scores.json exists; the test phase scores exactly one candidate.
Per-order logits are cached under results/clean/cache/ so an interrupted run resumes without changes.
"""
import argparse, csv, gzip, hashlib, importlib.util, io, json, math, os, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROTOCOL_COMMIT = "8455e7e2290d9ac96b014fc3767ad65303e5261e"
DATASET, DATASET_REV = "legacy-datasets/banking77", "f54121560de48f2852f90be299010d1d6dc612ec"
BASE, BASE_REV = "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"
ADAPTER, ADAPTER_REV = "DataAgent/Quorum-Reader-Qwen3-4B-Adapter", "e0765b768e04ff8dc83a8682963349bd6291ffa3"
KNN_EMBEDDER = "BAAI/bge-large-en-v1.5"
TASK_DESC = "Classify the customer's banking message into the intent it expresses."
SPLIT_SEED, FOLD_SEED, PER_CLASS, N_FOLDS, K = 20261005, 20261006, 12, 5, 77
ORDERS, MAX_ORDERS, BS = (1, 3, 7), 7, 8
MAX_PROMPT_TOKENS = 4096                                                    # addendum A1: assert, never truncate
BOOT_SEED, BOOT_N = 20261007, 10000                                         # addendum A2
REPRO_RESULTS_URL = ("https://raw.githubusercontent.com/simonmesmith/jev-banking77-experiment/"
                     "5cac4ff/results/results.json")                       # reported Jev metrics, read at run time
RULES = ("logprob_mean", "prob_mean", "rank_mean", "weighted_prob_mean")   # listed in tie-break order
TEXTS = ("names", "descriptions")                                          # tie-break: names first
MEMBER_SETS = {"M2": ("reader", "knn"), "M3": ("reader", "knn", "zs")}     # tie-break: fewer members first
REPRO_URL = ("https://raw.githubusercontent.com/simonmesmith/jev-banking77-experiment/"
             "5cac4ff/results/predictions.csv")                            # read at run time, never stored
OUT = ROOT / "results" / "clean"; CACHE = OUT / "cache"


# ---------------- data + split (no model) ----------------
def load_train():
    from datasets import load_dataset
    from quorum.data import _humanize
    ds = load_dataset(DATASET, revision=DATASET_REV, split="train")      # train split only, never test here
    raw = list(ds.features["label"].names)
    train = [{"text": r["text"], "label": int(r["label"])} for r in ds]
    assert len(train) == 10003 and len(raw) == K
    return ds, raw, [_humanize(n) for n in raw], train


def make_split(train):
    n = len(train)
    old = set(int(i) for i in np.random.default_rng(0).permutation(n)[:924])
    labels = np.array([r["label"] for r in train])
    rng = np.random.default_rng(SPLIT_SEED); S = []
    for c in range(K):
        elig = np.array(sorted(int(i) for i in np.flatnonzero(labels == c) if int(i) not in old))
        assert len(elig) >= PER_CLASS
        S += [int(i) for i in rng.permutation(elig)[:PER_CLASS]]
    assert len(S) == K * PER_CLASS and len(set(S)) == len(S) and not (set(S) & old)
    return sorted(S), sorted(old)


def descriptions(raw):
    spec = importlib.util.spec_from_file_location("b77d", ROOT / "examples" / "banking77_descriptions.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return [m.BANKING77_DESCRIPTIONS[r] for r in raw]


def code_fingerprint():
    files = ["scripts/clean_24shot.py", "quorum/fewshot.py", "quorum/prompting.py", "quorum/ensemble.py",
             "quorum/data.py", "examples/banking77_descriptions.py"]
    sha = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--"] + files,
                               capture_output=True, text=True).stdout.strip()
    except Exception:
        head, dirty = None, None
    return {"git_head": head or None, "tracked_files_modified": dirty or "", "sha256": sha}


# ---------------- calibration + combination (as in the study's combiner) ----------------
def softmax(L, T=1.0):
    Z = np.asarray(L, float) / T; Z = Z - Z.max(1, keepdims=True); e = np.exp(Z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(L, y):
    """Golden-section search on NLL over [0.05, 10] (60 iterations)."""
    L = np.asarray(L, float); y = np.asarray(y)
    def nll(T):
        P = softmax(L, T); return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))
    a, b = 0.05, 10.0; gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        if nll(c) < nll(d): b = d
        else: a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return (a + b) / 2


def _nll(P, y):
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))


def fit_weights(Ps, y, iters=60, grid=np.linspace(0, 1, 21)):
    """Coordinate search on the simplex minimising NLL of the weighted probability mean."""
    n = len(Ps); w = np.ones(n) / n; stack = np.stack(Ps)
    best = _nll(np.tensordot(w, stack, axes=1), y)
    for _ in range(iters):
        improved = False
        for i in range(n):
            for v in grid:
                cand = w.copy(); cand[i] = v; s = cand.sum()
                if s <= 0: continue
                cand = cand / s; sc = _nll(np.tensordot(cand, stack, axes=1), y)
                if sc < best - 1e-9: best, w, improved = sc, cand, True
        if not improved: break
    return w


def ranks(P):
    return np.argsort(np.argsort(P, axis=1), axis=1).astype(float)


def fit_params(Ls, y, rule):
    Ts = [fit_temperature(L, y) for L in Ls]
    w = fit_weights([softmax(L, T) for L, T in zip(Ls, Ts)], y) if rule == "weighted_prob_mean" else None
    return {"T": Ts, "w": None if w is None else [float(x) for x in w]}


def combine(Ls, params, rule):
    """Combined scores (N, K). For the three probability rules this is a probability distribution."""
    Ps = [softmax(L, T) for L, T in zip(Ls, params["T"])]
    if rule == "prob_mean":
        return np.mean(Ps, 0)
    if rule == "logprob_mean":
        g = np.exp(np.mean([np.log(p + 1e-12) for p in Ps], 0)); return g / g.sum(1, keepdims=True)
    if rule == "rank_mean":
        return np.mean([ranks(p) for p in Ps], 0)
    if rule == "weighted_prob_mean":
        return np.tensordot(np.asarray(params["w"]), np.stack(Ps), axes=1)
    raise ValueError(rule)


def strat_folds(y):
    """Stratified folds: each class's rows are shuffled (seed FOLD_SEED) and dealt round-robin, starting at fold
    c mod 5 so that fold sizes stay balanced (12 per class does not divide by 5)."""
    rng = np.random.default_rng(FOLD_SEED); f = np.empty(len(y), int)
    for c in range(K):
        for j, i in enumerate(rng.permutation(np.flatnonzero(y == c))):
            f[i] = (j + c) % N_FOLDS
    return f


# ---------------- models ----------------
def build_engine(names_opts, pool_rows):
    from quorum.fewshot import RetrievalEnsemble
    eng = RetrievalEnsemble(names_opts, TASK_DESC, base=BASE, adapter=ADAPTER, embedder=KNN_EMBEDDER,
                            shots=24, perms=1, device="cuda", base_revision=BASE_REV, adapter_revision=ADAPTER_REV,
                            max_length=MAX_PROMPT_TOKENS, truncate=False)    # addendum A1: never truncate
    return eng.index(pool_rows)


def length_stats(eng, rows, opts, text, orders):
    eng.options = list(opts[text])
    L = eng.prompt_lengths(rows, orders).ravel()
    return {"option_text": text, "orders": orders, "n_prompts": int(L.size), "mean": round(float(L.mean()), 1),
            "p50": int(np.percentile(L, 50)), "p95": int(np.percentile(L, 95)), "max": int(L.max()),
            "over_limit": int((L > MAX_PROMPT_TOKENS).sum()), "limit": MAX_PROMPT_TOKENS}


def cached(path, fn):
    path = Path(path)
    if path.exists():
        return np.load(path)
    arr = fn(); tmp = path.with_name(path.stem + ".tmp.npy"); np.save(tmp, arr); os.replace(tmp, path)
    return arr


def member_logits(tag, rows, eng, zs, opts, text, orders, with_zs):
    """{'reader_orders': (orders, N, K), 'knn': (N, K), 'zs': (N, K) log-probs or None} for one option text."""
    eng.options = list(opts[text])
    per = np.stack([cached(CACHE / f"{tag}_reader_{text}_o{pi}.npy",
                           lambda pi=pi: eng.incontext_logits_per_perm(rows, 1, bs=BS, perm_offset=pi)[0])
                    for pi in range(orders)])
    knn = cached(CACHE / f"{tag}_knn.npy", lambda: eng._knn_logits(rows))
    zsl = None
    if with_zs:
        ids = list(range(K))
        zsl = cached(CACHE / f"{tag}_zs_{text}.npy", lambda: np.log(np.stack(
            [zs.predict_proba(r["text"], ids, label_texts=list(opts[text])) for r in rows]) + 1e-12))
    return {"reader_orders": per, "knn": knn, "zs": zsl}


def cand_logits(m, member_set, k):
    Ls = [m["reader_orders"][:k].mean(0), m["knn"]]
    if member_set == "M3":
        Ls.append(m["zs"])
    return Ls


# ---------------- phases ----------------
def phase_split():
    OUT.mkdir(parents=True, exist_ok=True)
    _, raw, names, train = load_train()
    S, old = make_split(train)
    rec = {"protocol_commit": PROTOCOL_COMMIT, "dataset": DATASET, "dataset_revision": DATASET_REV,
           "selection_idx": S, "n": len(S), "per_class": PER_CLASS, "split_seed": SPLIT_SEED,
           "old_split_excluded": len(old), "pool_size": len(train) - len(S),
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "selection_split_idx.json").write_text(json.dumps(rec, indent=1))
    print(f"split: {len(S)} selection rows ({PER_CLASS}/class), pool {rec['pool_size']}", flush=True)
    return S


def load_split():
    rec = json.loads((OUT / "selection_split_idx.json").read_text())
    assert rec["protocol_commit"] == PROTOCOL_COMMIT
    return rec["selection_idx"]


def setup(S):
    _, raw, names, train = load_train()
    assert make_split(train)[0] == S, "selection split differs from the written one"
    Sset = set(S); P = [i for i in range(len(train)) if i not in Sset]
    opts = {"names": names, "descriptions": descriptions(raw)}
    eng = build_engine(names, [train[i] for i in P])
    from quorum import ZeroShotEnsemble
    zs = ZeroShotEnsemble(device="cuda")
    zs.members[0].model = zs.members[0].model.float()       # NLI member in fp32, matching the published dumps
    return raw, train, P, opts, eng, zs


def phase_smoke():
    S = load_split(); raw, train, P, opts, eng, zs = setup(S)
    rows = [train[i] for i in P[:8]]                     # pool rows only, never selection or test rows
    eng.options = list(opts["names"])
    r = eng.incontext_logits_per_perm(rows, 1, bs=BS)[0]; kn = eng._knn_logits(rows)
    p = zs.predict_proba(rows[0]["text"], list(range(K)), label_texts=list(opts["descriptions"]))
    assert r.shape == (8, K) and kn.shape == (8, K) and p.shape == (K,)
    print("SMOKE_OK (outputs discarded)", flush=True)


def phase_select():
    S = load_split(); raw, train, P, opts, eng, zs = setup(S); CACHE.mkdir(parents=True, exist_ok=True)
    rows = [train[i] for i in S]; y = np.array([r["label"] for r in rows]); folds = strat_folds(y)
    plen = [length_stats(eng, rows, opts, t, MAX_ORDERS) for t in TEXTS]      # tokenisation only
    print("prompt lengths (selection):", json.dumps(plen), flush=True)
    mem = {t: member_logits("sel", rows, eng, zs, opts, t, MAX_ORDERS, True) for t in TEXTS}
    table = []
    for ms in MEMBER_SETS:
        for rule in RULES:
            for text in TEXTS:
                for k in ORDERS:
                    Ls = cand_logits(mem[text], ms, k)
                    oof = np.empty(len(y), int)
                    for f in range(N_FOLDS):
                        tr, te = folds != f, folds == f
                        prm = fit_params([L[tr] for L in Ls], y[tr], rule)
                        oof[te] = combine([L[te] for L in Ls], prm, rule).argmax(1)
                    full = fit_params(Ls, y, rule)
                    ins = combine(Ls, full, rule).argmax(1)
                    table.append({"member_set": ms, "members": list(MEMBER_SETS[ms]), "rule": rule,
                                  "option_text": text, "orders": k,
                                  "oof_correct": int((oof == y).sum()), "oof_accuracy": round(float((oof == y).mean()), 4),
                                  "insample_correct": int((ins == y).sum())})
    key = lambda c: (-c["oof_correct"], len(c["members"]), c["orders"], TEXTS.index(c["option_text"]),
                     RULES.index(c["rule"]))
    win = min(table, key=key)
    Ls = cand_logits(mem[win["option_text"]], win["member_set"], win["orders"])
    prm = fit_params(Ls, y, win["rule"])
    comb = combine(Ls, prm, win["rule"])
    tfin = fit_temperature(np.log(comb + 1e-12) if win["rule"] != "rank_mean" else comb, y)
    rec = {"protocol_commit": PROTOCOL_COMMIT, "code": code_fingerprint(), "n_selection": len(y),
           "prompt_lengths_selection": plen, "zero_shot_nli_dtype": "fp32",
           "selection_rule": "max out-of-fold correct; ties -> fewer members, fewer orders, names before "
                             "descriptions, rule order " + " > ".join(RULES),
           "n_candidates": len(table), "candidates": sorted(table, key=key), "selected": win,
           "refit_on_full_selection_split": {"temperatures": dict(zip(MEMBER_SETS[win["member_set"]], prm["T"])),
                                             "weights": prm["w"], "final_temperature": tfin},
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "selection_scores.json").write_text(json.dumps(rec, indent=1))
    print("SELECTED:", json.dumps(win), flush=True)


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def secondary_metrics(Pm, pred, gold):
    """Addendum A2, pre-declared. Definitions follow the reproduction's scorer: macro-F1 = mean over classes of
    2tp/(actual+predicted); log loss clipped at 1e-15; Brier summed over classes; ECE on max-probability
    confidence with 10 equal-width bins [i/10, (i+1)/10), the last bin closed at 1."""
    pred, gold = np.asarray(pred), np.asarray(gold); n = len(gold); hit = (pred == gold).astype(float)
    acc = float(hit.mean()); rng = np.random.default_rng(BOOT_SEED)
    boots = np.array([hit[rng.integers(0, n, n)].mean() for _ in range(BOOT_N)])
    z = 1.959964; den = 1 + z * z / n
    centre = (acc + z * z / (2 * n)) / den; half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / den
    f1 = []
    for c in range(K):
        tp = int(((pred == c) & (gold == c)).sum()); a = int((gold == c).sum()); p_ = int((pred == c).sum())
        f1.append(2 * tp / (a + p_) if a + p_ else 0.0)
    pt = Pm[np.arange(n), gold]; conf = Pm.max(1); ece = 0.0
    for i in range(10):
        m = (conf >= i / 10) & ((conf < (i + 1) / 10) if i < 9 else (conf <= 1.0))
        if m.any():
            ece += m.sum() / n * abs(conf[m].mean() - hit[m].mean())
    return {"accuracy": round(acc, 4),
            "bootstrap_95": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])],
            "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED, "method": "percentile, rows resampled"},
            "wilson_95": [round(centre - half, 4), round(centre + half, 4)],
            "macro_f1": round(float(np.mean(f1)), 4),
            "log_loss_clipped_1e_15": round(float(np.mean(-np.log(np.maximum(1e-15, pt)))), 4),
            "brier_score": round(float(np.mean(((Pm - np.eye(K)[gold]) ** 2).sum(1))), 4),
            "ece_10_bins": round(float(ece), 4)}


def phase_compare_default():
    """Addendum A2, descriptive only: McNemar of the clean run vs the repo default (R2-default dump)."""
    OUTP = ROOT / "results" / "predictions"
    rd = lambda f: [json.loads(l) for l in gzip.open(OUTP / f, "rt")]
    a, b = rd("banking77_24shot_clean.jsonl.gz"), rd("banking77_24shot_default.jsonl.gz")
    assert [r["idx"] for r in a] == [r["idx"] for r in b] and [r["gold"] for r in a] == [r["gold"] for r in b]
    ca = np.array([r["pred"] == r["gold"] for r in a]); cb = np.array([r["pred"] == r["gold"] for r in b])
    bb, cc = int((ca & ~cb).sum()), int((~ca & cb).sum())
    rec = {"descriptive_only": True, "clean_accuracy": round(float(ca.mean()), 4),
           "default_accuracy": round(float(cb.mean()), 4), "b_clean_right_default_wrong": bb,
           "c_clean_wrong_default_right": cc, "p_two_sided_exact": mcnemar_exact(bb, cc)}
    (OUT / "compare_default.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec), flush=True)


def phase_test(args):
    sel_path = OUT / "selection_scores.json"
    if not sel_path.exists():
        sys.exit("refusing: selection_scores.json not written — run `select` first")
    OUTP = ROOT / "results" / "predictions"
    dump_path, meta_path = OUTP / "banking77_24shot_clean.jsonl.gz", OUTP / "banking77_24shot_clean_meta.json"
    if (dump_path.exists() or meta_path.exists()) and not args.rerun_after_crash:
        sys.exit("refusing: the single test run already produced output; pass --rerun-after-crash REASON "
                 "only if that run crashed (the rerun is disclosed in the meta, protocol §8)")
    sel_bytes = sel_path.read_bytes(); sel = json.loads(sel_bytes)
    assert sel["protocol_commit"] == PROTOCOL_COMMIT, "selection was made under a different protocol commit"
    assert sel["code"]["sha256"] == code_fingerprint()["sha256"], "code differs from the code that made the selection"
    win = sel["selected"]; prm_rec = sel["refit_on_full_selection_split"]
    S = load_split(); raw, train, P, opts, eng, zs = setup(S)
    from datasets import load_dataset
    test = [{"text": r["text"], "label": int(r["label"])}
            for r in load_dataset(DATASET, revision=DATASET_REV, split="test")]  # loaded only now
    assert len(test) == 3080
    t0 = time.time(); ms, text, k, rule = win["member_set"], win["option_text"], win["orders"], win["rule"]
    plen_test = length_stats(eng, test, opts, text, k)                          # tokenisation only
    print("prompt lengths (test):", json.dumps(plen_test), flush=True)
    m = member_logits("test", test, eng, zs, opts, text, k, ms == "M3")
    Ls = cand_logits(m, ms, k)
    prm = {"T": [prm_rec["temperatures"][n] for n in MEMBER_SETS[ms]], "w": prm_rec["weights"]}
    comb_raw = combine(Ls, prm, rule)
    comb = comb_raw / comb_raw.sum(1, keepdims=True) if rule == "rank_mean" else comb_raw   # rank: normalised scores
    Pf = softmax(np.log(comb_raw + 1e-12) if rule != "rank_mean" else comb_raw, prm_rec["final_temperature"])
    g7 = lambda v: [float(f"{x:.7g}") for x in v]
    cal = [softmax(L, T) for L, T in zip(Ls, prm["T"])]
    OUTP.mkdir(parents=True, exist_ok=True)
    preds, gold = [], []
    with gzip.open(dump_path, "wt") as f:
        for i, r in enumerate(test):
            probs = g7(comb[i])                                   # rounded for storage only
            pred = int(np.argmax(comb[i])); preds.append(pred); gold.append(r["label"])   # unrounded argmax
            row = {"idx": i, "text": r["text"], "gold": r["label"], "pred": pred, "probs": probs,
                   "reader_probs": g7(cal[0][i]), "knn_probs": g7(cal[1][i]),
                   "retrieved_idx": [P[j] for j in eng.shots_idx(r["text"])],
                   "perm_reader_probs": [g7(softmax(m["reader_orders"][pi][i:i + 1])[0]) for pi in range(k)]}
            if ms == "M3":
                row["zs_probs"] = g7(cal[2][i])
            f.write(json.dumps(row) + "\n")
    acc = float((np.array(preds) == np.array(gold)).mean()); secs = time.time() - t0
    mc = {"source": REPRO_URL}
    try:
        txt = urllib.request.urlopen(REPRO_URL, timeout=60).read().decode()
        rep = {int(row["id"].split("-")[1]): row for row in csv.DictReader(io.StringIO(txt))}
        assert len(rep) == 3080 and all(rep[i]["truth"] == raw[gold[i]] for i in range(3080)), "row order mismatch"
        ours = np.array(preds) == np.array(gold); theirs = np.array([rep[i]["correct"] == "True" for i in range(3080)])
        b, c = int((ours & ~theirs).sum()), int((~ours & theirs).sum())
        mc.update({"join": "id test-NNNNN == official test index; truth labels verified for all 3,080 rows",
                   "reproduction_accuracy": round(float(theirs.mean()), 4),
                   "b_ours_right_theirs_wrong": b, "c_ours_wrong_theirs_right": c, "p_two_sided_exact": mcnemar_exact(b, c)})
    except Exception as e:
        mc["error"] = f"{type(e).__name__}: {e}"
    sec = {"primary_distribution": "final-temperature ensemble distribution (temperature fit on the selection split)",
           "calibrated_final": secondary_metrics(Pf, preds, gold),
           "uncalibrated_combined": secondary_metrics(comb, preds, gold)}
    try:
        rj = json.loads(urllib.request.urlopen(REPRO_RESULTS_URL, timeout=60).read().decode())
        sec["reproduction_reported"] = {k: rj.get(k) for k in ("accuracy", "accuracy_wilson_95", "macro_f1",
                                         "log_loss_clipped_1e_15", "brier_score", "ece_10_bins")}
    except Exception as e:
        sec["reproduction_reported"] = {"error": f"{type(e).__name__}: {e}"}
    import torch, transformers, peft, sentence_transformers
    meta = {"protocol_commit": PROTOCOL_COMMIT, "code": code_fingerprint(),
            "selection_file_sha256": hashlib.sha256(sel_bytes).hexdigest(),
            "rerun": bool(args.rerun_after_crash), "rerun_reason": args.rerun_after_crash or None,
            "pred_rule": "argmax of the unrounded combined scores; stored probabilities are rounded to 7 significant digits",
            "dataset": DATASET,
            "dataset_revision": DATASET_REV, "base": BASE, "base_revision": BASE_REV, "adapter": ADAPTER,
            "adapter_revision": ADAPTER_REV, "knn_embedder": KNN_EMBEDDER, "task_description": TASK_DESC,
            "selected": win, "fitted": prm_rec, "pool_size": len(P), "n_test": len(test),
            "accuracy": round(acc, 4), "correct": int(sum(np.array(preds) == np.array(gold))),
            "mcnemar_vs_reproduction": mc, "secondary": sec, "prompt_lengths_test": plen_test,
            "zero_shot_nli_dtype": "fp32", "max_prompt_tokens": MAX_PROMPT_TOKENS, "truncation": False,
            "fields": {"probs": "combined distribution of the calibrated members (rank_mean: normalised scores)",
                       "reader_probs/knn_probs/zs_probs": "softmax(member logits / fitted member temperature)",
                       "perm_reader_probs": "softmax(per-order reader logits) at T=1",
                       "retrieved_idx": "indices into the pinned train split (pool = train minus selection split)"},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__, "sentence_transformers": sentence_transformers.__version__},
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "wall_seconds": round(secs, 1), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"TEST: accuracy={acc:.4f}  mcnemar={json.dumps(mc)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["split", "smoke", "select", "test", "compare-default"])
    ap.add_argument("--rerun-after-crash", metavar="REASON", default=None,
                    help="test phase only: allow re-running after a crashed test run; the reason goes in the meta")
    a = ap.parse_args()
    if a.phase == "test":
        phase_test(a)
    else:
        {"split": phase_split, "smoke": phase_smoke, "select": phase_select,
         "compare-default": phase_compare_default}[a.phase]()
