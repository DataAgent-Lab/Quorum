"""Decision-service core. Three layers over the `quorum` library, each the validated winner from the study:

  * Layer 1   — define a task (labels only) and predict IMMEDIATELY, zero-shot, no training. Uses the
                calibration-free `quorum.ZeroShotEnsemble` (PrismNLI + bge-large + bge-base, logprob-mean).
  * Layer 1.5 — give it a labeled example POOL (no head training) and it serves the 24-shot retrieval anchor
                (`quorum.fewshot.RetrievalEnsemble`): 24 BM25-retrieved examples read in-context by the gated
                4B reader + a bge-large kNN over the same 24, geometric mean. On the full Banking77 test this
                code path scores 0.932, above the reproduced closed API (Jev) 0.924 (our study's best config
                reached 0.938 and beat that reproduction at paired McNemar p=0.0014; see the repo README).
  * Layer 2   — train a linear head on a frozen embedder + temperature calibration on your own labels.

`predict()` auto-serves the richest layer a task has: trained head → else pool anchor → else zero-shot.
Confidence gate: a raw threshold τ, or a conformal target-precision gate with a finite-sample guarantee.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

REG = Path(os.environ.get("REGISTRY_DIR", str(Path(__file__).resolve().parent / "data" / "registry")))

# ---- backends (lazy singletons) --------------------------------------------------------------------
DEVICE = os.environ.get("DEVICE", "cpu")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")      # Layer-2 frozen embedder
# Layer 1.5 (GPU tier) — the gated 24-shot reader adapter + its base + kNN embedder.
RLCD_BASE = os.environ.get("RLCD_BASE", "Qwen/Qwen3-4B")
RLCD_ADAPTER = os.environ.get("RLCD_ADAPTER", "DataAgent/Quorum-Reader-Qwen3-4B-Adapter")
KNN_EMBED = os.environ.get("KNN_EMBED", "BAAI/bge-large-en-v1.5")

_embedder = None
_zs_ensemble = None


class _Embedder:
    """Frozen sentence-embedding backend for the Layer-2 trained head."""
    def __init__(self, model=EMBED_MODEL, device=DEVICE):
        from sentence_transformers import SentenceTransformer
        self.model_id = model
        self.st = SentenceTransformer(model, trust_remote_code=True, device=device)

    def encode(self, texts):
        v = self.st.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        return np.asarray(v, dtype=np.float64)


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = _Embedder()
    return _embedder


def zeroshot_ensemble():
    """Layer-1 default: the calibration-free zero-shot ensemble. LAYER1_MODE=single drops the embedder members
    (NLI only) for a lighter, lower-accuracy tier."""
    global _zs_ensemble
    if _zs_ensemble is None:
        from quorum import ZeroShotEnsemble
        from quorum.ensemble import DEFAULT_EMBEDDERS
        embedders = () if os.environ.get("LAYER1_MODE") == "single" else DEFAULT_EMBEDDERS
        _zs_ensemble = ZeroShotEnsemble(embedders=embedders, device=DEVICE)
    return _zs_ensemble


# ---- math (self-contained; no external harness) ----------------------------------------------------
def _softmax(Z, T=1.0):
    Z = np.asarray(Z, float) / T
    Z = Z - Z.max(axis=-1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=-1, keepdims=True)


def _train_head(X, y, K, epochs=250, lr=0.05, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed); D = X.shape[1]
    W = np.zeros((D, K)); b = np.zeros(K)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8; t = 0; n = len(y); Y = np.eye(K)[y]; bs = min(128, n)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            bi = idx[s:s+bs]; xb = X[bi]; yb = Y[bi]
            P = _softmax(xb @ W + b); g = (P - yb) / len(bi)
            gW = xb.T @ g + l2 * W; gb = g.sum(0); t += 1
            for par, gp, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= b1; m += (1-b1)*gp; v *= b2; v += (1-b2)*gp*gp
                par -= lr * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
    return W, b


def _fit_T(logits, y):
    def nll(T):
        P = _softmax(logits, T); return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))
    a, b = 0.05, 10.0; gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr*(b-a), a + gr*(b-a)
    for _ in range(50):
        if nll(c) < nll(d): b = d
        else: a = c
        c, d = b - gr*(b-a), a + gr*(b-a)
    return (a + b) / 2


def _ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins+1); e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i+1])
        if m.sum(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def _conformal_from_proba(P, y, targets=(0.90, 0.95, 0.98), delta=0.1, grid=20):
    """Selective-classification thresholds with a finite-sample precision GUARANTEE, computed directly from a
    HELD-OUT calibration set's PREDICTED PROBABILITIES `P` (rows) and true labels `y`. For each target precision
    p: pick the confidence cut τ (over a grid of candidate confidences) with the MOST coverage whose auto-acted
    subset has Wilson-score precision lower bound >= p — so with prob >= 1-delta the auto-acted precision is >= p
    (Wilson LB per candidate, Bonferroni over the grid so testing many τ stays valid). This needs only the served
    confidences, so it is valid whether or not the underlying scores are temperature-calibrated."""
    from statistics import NormalDist
    P = np.asarray(P, float); conf = P.max(1); pred = P.argmax(1); corr = (pred == y).astype(float); n = len(y)
    blank = {f"{t:.2f}": {"tau": None, "coverage": 0.0, "guaranteed": False} for t in targets}
    if n < 10:
        return {"delta": delta, "n_cal": int(n), "targets": blank}
    z = NormalDist().inv_cdf(1 - delta / grid)
    cand = np.unique(np.quantile(conf, np.linspace(0, 1, grid)))

    def wilson_lb(succ, tot):
        if tot == 0: return 0.0
        ph = succ / tot; z2 = z * z; den = 1 + z2 / tot
        return max(0.0, (ph + z2 / (2 * tot) - z * np.sqrt(ph * (1 - ph) / tot + z2 / (4 * tot * tot))) / den)

    out = {}
    for t in targets:
        best_tau, best_cov = None, 0.0
        for tau in cand:
            sel = conf >= tau; tot = int(sel.sum())
            if tot and wilson_lb(int(corr[sel].sum()), tot) >= t:
                cov = tot / n
                if cov > best_cov:
                    best_cov, best_tau = cov, float(tau)
        out[f"{t:.2f}"] = {"tau": round(best_tau, 4) if best_tau is not None else None,
                           "coverage": round(best_cov, 4), "guaranteed": best_tau is not None}
    return {"delta": delta, "n_cal": int(n), "grid": len(cand), "method": "wilson+bonferroni", "targets": out}


def _write_meta(meta):
    REG.mkdir(parents=True, exist_ok=True)
    (REG / f"{meta['name']}.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---- task registry: define / train / define_pool / predict -----------------------------------------
def define_task(name, classes, primitive="choice", descriptions=None):
    """Layer 1: register a task with just its label set — predict works immediately (zero-shot, no training).
    descriptions: OPTIONAL {class: natural-language description} used as the zero-shot label text instead of the
    bare name. Gain scales with how opaque the names are (small when names are already descriptive, large when
    they are short/ambiguous); its effect on guaranteed coverage is model-dependent, so it is offered, not
    promised to raise coverage."""
    if len(classes) < 2:
        raise ValueError("need >= 2 classes")
    descriptions = {k: str(v) for k, v in (descriptions or {}).items() if v}
    unknown = sorted(set(descriptions) - set(classes))
    if unknown:
        raise ValueError(f"descriptions reference unknown classes: {unknown[:5]}")
    return _write_meta({"name": name, "primitive": primitive, "classes": classes, "layer": 1,
                        "descriptions": descriptions or None, "layer1_mode": os.environ.get("LAYER1_MODE", "ensemble"),
                        "defined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def train_task(name, classes, examples, primitive="choice"):
    """Layer 2: train a linear head on a strong frozen embedder + temperature calibration."""
    if len(classes) < 2:
        raise ValueError("need >= 2 classes")
    labidx = {c: i for i, c in enumerate(classes)}
    bad = sorted({e["label"] for e in examples if e["label"] not in labidx})
    if bad:
        raise ValueError(f"examples reference unknown labels: {bad[:5]}")
    per = {c: 0 for c in classes}
    for e in examples:
        per[e["label"]] += 1
    thin = [c for c, n in per.items() if n < 3]

    y = np.array([labidx[e["label"]] for e in examples])
    X = embedder().encode([e["text"] for e in examples])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    K = len(classes); n = len(y)
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    if n >= max(30, 3 * K):
        ncal = max(K, int(0.3 * n)); cal, tr = perm[:ncal], perm[ncal:]
    else:
        cal = tr = perm
    W, b = _train_head(Xn[tr], y[tr], K)
    Lcal = Xn[cal] @ W + b
    T = _fit_T(Lcal, y[cal])
    P = _softmax(Lcal, T); pred = P.argmax(1); conf = P.max(1); corr = (pred == y[cal]).astype(float)
    conformal = _conformal_from_proba(P, y[cal]) if (cal is not tr) else None   # valid only when cal held out

    REG.mkdir(parents=True, exist_ok=True)
    np.savez(REG / f"{name}.npz", W=W, b=b, T=T, mu=mu, sd=sd, classes=np.array(classes, dtype=object))
    return _write_meta({"name": name, "primitive": primitive, "classes": classes, "layer": 2,
                        "n_examples": n, "per_class": per, "thin_classes": thin,
                        "embed_model": embedder().model_id, "temperature": round(float(T), 3),
                        "metrics": {"cal_accuracy": round(float(corr.mean()), 3),
                                    "ece": round(_ece(conf, corr), 3), "held_out": bool(cal is not tr)},
                        "conformal": conformal,
                        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def define_pool_task(name, classes, examples, primitive="choice", descriptions=None):
    """Layer 1.5: register a task WITH a labeled example POOL but no head training. predict() serves the 24-shot
    retrieval anchor — 24 BM25-retrieved examples read in-context by the gated 4B reader + a bge-large kNN over
    the same 24, geometric mean. GPU tier: needs a GPU + access to the reader adapter (RLCD_ADAPTER). Layers 1/2
    stay CPU and never import it."""
    if len(classes) < 2:
        raise ValueError("need >= 2 classes")
    labidx = {c: i for i, c in enumerate(classes)}
    bad = sorted({e["label"] for e in examples if e["label"] not in labidx})
    if bad:
        raise ValueError(f"examples reference unknown labels: {bad[:5]}")
    descriptions = {k: str(v) for k, v in (descriptions or {}).items() if v}
    pool = [{"text": e["text"], "label": labidx[e["label"]]} for e in examples]
    per = {c: 0 for c in classes}
    for e in examples:
        per[e["label"]] += 1
    thin = [c for c, k in per.items() if k < 3]
    n = len(pool); K = len(classes)
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    ncal = max(K, int(0.3 * n)) if n >= max(30, 3 * K) else n
    cal_idx = perm[:ncal].tolist()
    REG.mkdir(parents=True, exist_ok=True)
    (REG / f"{name}.pool.json").write_text(json.dumps({"pool": pool, "cal_idx": cal_idx}))
    return _write_meta({"name": name, "primitive": primitive, "classes": classes, "layer": 1.5,
                        "descriptions": descriptions or None, "n_examples": n, "per_class": per,
                        "thin_classes": thin, "n_cal": len(cal_idx),
                        "base": RLCD_BASE, "adapter": RLCD_ADAPTER, "knn_embed": KNN_EMBED,
                        "defined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


_L15 = {}   # name -> (fitted quorum.fewshot.RetrievalEnsemble indexed on the full pool)


def _get_l15(meta):
    name = meta["name"]
    if name in _L15:
        return _L15[name]
    from quorum.fewshot import RetrievalEnsemble
    pj = json.loads((REG / f"{name}.pool.json").read_text())
    pool, cal_idx = pj["pool"], set(pj["cal_idx"])
    classes = meta["classes"]; desc = meta.get("descriptions") or {}
    options = [desc.get(c) or c.replace("_", " ") for c in classes]
    task_desc = f"Classify the input into one of the {len(classes)} categories."
    eng = RetrievalEnsemble(options, task_desc, base=meta.get("base", RLCD_BASE),
                            adapter=meta.get("adapter", RLCD_ADAPTER), embedder=meta.get("knn_embed", KNN_EMBED),
                            device=os.environ.get("DEVICE", "cuda"))
    # Conformal precision gate from a HELD-OUT cal: index on pool-minus-cal, predict the cal rows, derive taus
    # from those predicted probabilities (the same quantity served at inference). No temperature is fit here, so
    # this is the raw ensemble; the conformal guarantee needs only the served confidences, so it still holds.
    if meta.get("conformal") is None:
        cal_rows = [pool[i] for i in sorted(cal_idx)]
        noncal = [pool[i] for i in range(len(pool)) if i not in cal_idx]
        if cal_rows and noncal:
            eng.index(noncal)
            Pcal = eng.predict_proba(cal_rows, perms=1)
            ycal = np.array([r["label"] for r in cal_rows])
            meta["conformal"] = _conformal_from_proba(Pcal, ycal)
            _write_meta(meta)
    eng.index(pool)                          # serving retrieves from the full pool
    _L15[name] = eng
    return eng


def get_task(name):
    p = REG / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"task '{name}' not found — define or train it first")
    return json.loads(p.read_text())


def predict_task(name, text, tau=0.0, target_precision=None):
    meta = get_task(name)
    classes = meta["classes"]
    npz = REG / f"{name}.npz"
    if npz.exists():                                   # Layer 2: trained head
        m = np.load(npz, allow_pickle=True); T = float(m["T"])
        xn = (embedder().encode([text])[0] - m["mu"]) / m["sd"]
        P = _softmax(xn @ m["W"] + m["b"], T); layer = 2; calibrated = True
    elif meta.get("layer") == 1.5:                     # Layer 1.5: 24-shot retrieval anchor
        P = _get_l15(meta).predict_proba([{"text": text}])[0]; layer = 1.5; calibrated = True
    else:                                              # Layer 1: zero-shot, no training
        desc = meta.get("descriptions") or {}
        label_texts = [desc.get(c) or c.replace("_", " ") for c in classes]
        P = zeroshot_ensemble().predict_proba(text, list(range(len(classes))), label_texts=label_texts)
        layer = 1; calibrated = False
    P = np.asarray(P, float)
    order = np.argsort(-P); top = int(order[0]); conf = float(P[top])

    gate = {"mode": "raw", "tau": tau}
    conf_meta = (meta.get("conformal") or {}).get("targets") if calibrated else None
    if target_precision is not None and conf_meta:
        band = conf_meta.get(f"{float(target_precision):.2f}")
        if band and band.get("guaranteed"):
            tau = band["tau"]; gate = {"mode": "conformal", "target_precision": float(target_precision),
                                       "tau": tau, "guaranteed": True, "cal_coverage": band["coverage"]}
        else:
            gate = {"mode": "conformal", "target_precision": float(target_precision), "tau": None,
                    "guaranteed": False, "note": "cal set too small to guarantee this precision"}
    abstain = (tau is None) or (conf < tau)

    out = {"task": name, "primitive": meta.get("primitive", "choice"), "layer": layer,
           "calibrated": calibrated, "label": classes[top], "confidence": round(conf, 4),
           "abstain": bool(abstain), "gate": gate,
           "distribution": {classes[i]: round(float(P[i]), 4) for i in order}}
    prim = out["primitive"]
    if prim == "score":                # ordered levels: expectation over the level index (classes low->high)
        idx = np.arange(len(classes), dtype=float)
        out["score"] = round(float((P * idx).sum()), 4)
        out["score_normalized"] = round(float((P * idx).sum() / max(len(classes) - 1, 1)), 4)
    elif prim == "noul":               # yes/no: P(true) = last class (classes = [no/false, yes/true])
        out["probability_true"] = round(float(P[-1]), 4)
    else:                              # choice
        out["choice"] = out["distribution"]
    return out


def list_tasks():
    if not REG.exists():
        return []
    return sorted((json.loads((REG / f).read_text()) for f in os.listdir(REG) if f.endswith(".json")),
                  key=lambda t: t.get("trained_at", t.get("defined_at", "")), reverse=True)
