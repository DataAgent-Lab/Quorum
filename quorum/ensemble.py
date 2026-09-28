"""A cheap, calibration-free zero-shot intent classifier: an ensemble of small open models.

The idea: no single small open model is a great zero-shot intent classifier, but a *geometric-mean ensemble*
of a few complementary ones — one NLI model and two sentence embedders — reliably beats its own best member,
runs on CPU in milliseconds, and needs no training and no per-task calibration.

Members (all public, all sub-1B):
  * NLI      : `Jaehun/PrismNLI-0.4B`   (~0.4B, trained on synthetic NLI only)
  * embedder : `BAAI/bge-large-en-v1.5` (~0.34B)
  * embedder : `BAAI/bge-base-en-v1.5`  (~0.11B)

Each member scores the K candidate labels, we softmax at temperature 1, then take the geometric mean
(log-probability mean) and renormalise. No labelled examples are used; label text is a short verbalization of
each class ("This message is about {label}.").
"""
from __future__ import annotations
import numpy as np

DEFAULT_NLI = "Jaehun/PrismNLI-0.4B"
DEFAULT_EMBEDDERS = ("BAAI/bge-large-en-v1.5", "BAAI/bge-base-en-v1.5")
TEMPLATE = "This message is about {label}."
COSINE_SCALE = 20.0


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class _NLIMember:
    """Cross-attention NLI entailment scorer: score(premise=text, hypothesis='This message is about {label}.')."""
    def __init__(self, model_name: str = DEFAULT_NLI, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entail = next((i for i, v in id2label.items() if "entail" in v), id2label and max(id2label) or 0)

    def logits(self, text: str, label_texts: list[str], batch: int = 32) -> np.ndarray:
        hyp = [TEMPLATE.format(label=t) for t in label_texts]
        prem = [text] * len(label_texts)
        out = []
        for i in range(0, len(label_texts), batch):
            enc = self.tok(prem[i:i + batch], hyp[i:i + batch], padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out.append(self.model(**enc).logits[:, self.entail].float().cpu().numpy())
        return np.concatenate(out)


class _EmbedMember:
    """Bi-encoder cosine scorer: cosine(embed(text), embed('This message is about {label}.'))."""
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.st = SentenceTransformer(model_name, device=device)
        self._label_cache: dict[tuple, np.ndarray] = {}

    def _label_vecs(self, label_texts: list[str]) -> np.ndarray:
        key = tuple(label_texts)
        if key not in self._label_cache:
            hyp = [TEMPLATE.format(label=t) for t in label_texts]
            self._label_cache[key] = np.asarray(
                self.st.encode(hyp, normalize_embeddings=True, show_progress_bar=False), dtype=float)
        return self._label_cache[key]

    def logits(self, text: str, label_texts: list[str]) -> np.ndarray:
        lv = self._label_vecs(label_texts)
        qv = np.asarray(self.st.encode([text], normalize_embeddings=True, show_progress_bar=False)[0], dtype=float)
        return (lv @ qv) * COSINE_SCALE


class ZeroShotEnsemble:
    """Load once, then `predict_proba(text, labels)` returns a probability distribution over the labels.

    `labels` are class ids/names; pass `label_texts` for a natural-language verbalization of each class
    (recommended — underscores are turned to spaces otherwise). Everything is zero-shot: no training, no cal.
    """
    def __init__(self, nli: str = DEFAULT_NLI, embedders=DEFAULT_EMBEDDERS, device: str = "cpu"):
        self.members = [_NLIMember(nli, device)] + [_EmbedMember(m, device) for m in embedders]
        self.model_names = [nli, *embedders]

    def member_and_ensemble_proba(self, text: str, labels, label_texts=None):
        """Return (list of per-member distributions, ensemble distribution) — handy for ablations."""
        texts = list(label_texts) if label_texts is not None else [str(l).replace("_", " ") for l in labels]
        probs = [_softmax(m.logits(text, texts)) for m in self.members]           # each member -> distribution
        g = np.exp(np.mean([np.log(p + 1e-12) for p in probs], axis=0))           # geometric mean (logprob-mean)
        return probs, g / g.sum()

    def predict_proba(self, text: str, labels, label_texts=None) -> np.ndarray:
        return self.member_and_ensemble_proba(text, labels, label_texts)[1]

    def predict(self, text: str, labels, label_texts=None):
        p = self.predict_proba(text, labels, label_texts)
        i = int(np.argmax(p))
        return labels[i], float(p[i]), p
