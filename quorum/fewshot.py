"""The 24-shot pipeline that significantly beats the closed API (Jev) on Banking77.

Jev's protocol: index the training set, retrieve 24 examples per query with BM25, condition on them in-context,
never update weights. Quorum runs two mechanisms over the *same* 24 retrieved examples and takes their
geometric mean:
  * in-context reader : a 4B open LLM (`Qwen3-4B` + a small LoRA fine-tuned on public intent data, target
                        benchmark excluded) reads the 24 examples and scores all K labels in one forward pass
                        (answer-token / M1 scoring, option order averaged over a few permutations).
  * kNN               : `bge-large` cosine to the same 24 retrieved examples, best-similarity per class.

The LoRA adapter is released gated on the Hugging Face Hub (see the model card); pass its id/path as `adapter`.
"""
from __future__ import annotations
import math
import random
import re
from collections import Counter
import numpy as np

from . import prompting as P

_TOK = re.compile(r"[a-z0-9]+")
KNN_SCALE = 20.0
KNN_FLOOR = -1.0


class BM25:
    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.docs = [_TOK.findall(d.lower()) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.k1, self.b = k1, b
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def top(self, query: str, k: int) -> list[int]:
        q = _TOK.findall(query.lower())
        scores = np.zeros(self.N)
        for w in set(q):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(w, 0)
                if f:
                    dl = len(self.docs[i])
                    scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return list(np.argsort(-scores)[:k])


def _softmax(z):
    z = np.asarray(z, float); z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _lp_mean(prob_list):
    g = np.exp(np.mean([np.log(p + 1e-12) for p in prob_list], axis=0))
    return g / g.sum(1, keepdims=True)


class RetrievalEnsemble:
    def __init__(self, options, task_desc, base="Qwen/Qwen3-4B", adapter=None,
                 embedder="BAAI/bge-large-en-v1.5", shots=24, perms=3, device="cuda"):
        if adapter is None:
            raise ValueError("pass `adapter` — the gated 24-shot LoRA (a Hugging Face id or local path)")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.options = list(options); self.K = len(self.options); self.task_desc = task_desc
        self.shots, self.perms, self.device = shots, perms, device
        self.tok = AutoTokenizer.from_pretrained(base, padding_side="left")
        self.markers = P.verify_markers(self.tok); self.marker_ids = P.marker_token_ids(self.tok, self.markers)
        if len(self.markers) < self.K:
            raise ValueError(f"K={self.K} > {len(self.markers)} single-token markers")
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to(device)
        self.model = PeftModel.from_pretrained(m, adapter).eval()
        self.embedder = SentenceTransformer(embedder, device=device)

    def index(self, pool):
        """pool = [{'text', 'label': int}] — the retrieval set (Jev conditions on 24 of these per query)."""
        self.pool = list(pool)
        self.bm = BM25([r["text"] for r in self.pool])
        self.pool_vec = np.asarray(self.embedder.encode([r["text"] for r in self.pool], normalize_embeddings=True,
                                                        batch_size=128, show_progress_bar=False), float)
        self.pool_lab = np.array([r["label"] for r in self.pool])
        return self

    def _shots(self, text):
        return self.bm.top(text, self.shots)

    def _incontext_logits(self, rows, perms, bs=8):
        torch = self.torch
        N = len(rows); avg = np.zeros((N, self.K))
        with torch.no_grad():
            for pi in range(perms):
                rng = random.Random(1000 + pi)
                for i in range(0, N, bs):
                    chunk = rows[i:i + bs]; prompts, orders = [], []
                    for r in chunk:
                        shots = [(self.pool[j]["text"], self.pool[j]["label"]) for j in self._shots(r["text"])]
                        ex = P.Example(self.task_desc, self.options, r["text"], None, shots=shots)
                        p, order = P.build_prompt(ex, self.markers, shuffle=True, rng=rng)
                        prompts.append(p); orders.append(order)
                    enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                                   max_length=2048).to(self.device)
                    last = self.model(**enc).logits[:, -1, :].float()
                    for b, order in enumerate(orders):
                        avg[i + b] += P.unpermute(last[b, self.marker_ids[:self.K]].cpu(), order).numpy() / perms
        return avg

    def _knn_logits(self, rows):
        out = np.full((len(rows), self.K), KNN_FLOOR, float)
        qv = np.asarray(self.embedder.encode([r["text"] for r in rows], normalize_embeddings=True,
                                             batch_size=128, show_progress_bar=False), float)
        for i, r in enumerate(rows):
            idx = self._shots(r["text"]); sims = self.pool_vec[idx] @ qv[i]; labs = self.pool_lab[idx]
            for c in np.unique(labs):
                out[i, c] = float(sims[labs == c].max())
        return out * KNN_SCALE

    def predict_proba(self, rows, perms=None):
        ic = self._incontext_logits(rows, perms or self.perms)
        kn = self._knn_logits(rows)
        return _lp_mean([_softmax(ic), _softmax(kn)])           # geometric mean, T=1, no calibration
