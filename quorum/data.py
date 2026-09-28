"""Loaders for the public intent-classification benchmarks used in this study.

Each loader returns a dict: {"labels": [name, ...], "test": [{"text", "label": int}, ...]} plus, where used,
"train"/"pool". Label names are humanized (underscores -> spaces; camelCase split; task-specific prefixes
stripped) so they can be verbalized as "This message is about {name}." — no per-class descriptions authored.
"""
from __future__ import annotations
import re
from collections import defaultdict
import numpy as np

_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _humanize(name: str) -> str:
    name = name[3:] if name.startswith("IN:") else name          # MTOP intents look like IN:GET_MESSAGE
    name = _CAMEL.sub(" ", name).replace("_", " ").lower().strip()
    return "out of scope" if name == "oos" else name


def _index(labels):
    u = sorted(set(labels))
    return u, {l: i for i, l in enumerate(u)}


def load(name: str):
    from datasets import load_dataset
    if name == "banking77":
        ds = load_dataset("legacy-datasets/banking77")
        names = ds["train"].features["label"].names
        conv = lambda sp: [{"text": r["text"], "label": r["label"]} for r in ds[sp]]
        return {"labels": [_humanize(n) for n in names], "raw_labels": names,
                "train": conv("train"), "test": conv("test")}
    if name == "clinc150":
        ds = load_dataset("clinc/clinc_oos", "plus")
        names = ds["train"].features["intent"].names
        conv = lambda sp: [{"text": r["text"], "label": r["intent"]} for r in ds[sp]]
        return {"labels": [_humanize(n) for n in names], "raw_labels": names,
                "train": conv("train"), "test": conv("test")}
    if name == "massive":
        ds = load_dataset("mteb/amazon_massive_intent", "en")
        u, mp = _index(ds["train"]["label"])
        conv = lambda sp: [{"text": r["text"], "label": mp[r["label"]]} for r in ds[sp]]
        return {"labels": [_humanize(x) for x in u], "raw_labels": u, "train": conv("train"), "test": conv("test")}
    if name == "mtop":
        ds = load_dataset("WillHeld/mtop")
        u, mp = _index(ds["train_en"]["intent"])
        conv = lambda sp: [{"text": r["utterance"], "label": mp[r["intent"]]} for r in ds[sp]]
        return {"labels": [_humanize(x) for x in u], "raw_labels": u,
                "train": conv("train_en"), "test": conv("test_en")}
    if name in ("hwu64", "snips"):
        hid = "DeepPavlov/hwu64" if name == "hwu64" else "DeepPavlov/snips"
        ds = load_dataset(hid, "default")
        id2name = {r["id"]: r["name"] for r in load_dataset(hid, "intents")["intents"]}
        names = [id2name[i] for i in range(len(id2name))]
        conv = lambda sp: [{"text": r["utterance"], "label": int(r["label"])} for r in ds[sp]]
        return {"labels": [_humanize(n) for n in names], "raw_labels": names,
                "train": conv("train"), "test": conv("test")}
    if name == "bitext":
        # no official test split -> deterministic stratified 20% test (fixed seed), rest is the retrieval pool
        ds = load_dataset("Bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"]
        u, mp = _index(ds["intent"])
        rows = [{"text": r["instruction"], "label": mp[r["intent"]]} for r in ds]
        by = defaultdict(list)
        for i, r in enumerate(rows):
            by[r["label"]].append(i)
        rng = np.random.default_rng(20250926)
        test_i, pool_i = [], []
        for lab in sorted(by):
            idx = np.array(by[lab]); rng.shuffle(idx); n = max(1, round(0.2 * len(idx)))
            test_i += idx[:n].tolist(); pool_i += idx[n:].tolist()
        return {"labels": [_humanize(x) for x in u], "raw_labels": u,
                "train": [rows[i] for i in pool_i], "test": [rows[i] for i in test_i]}
    raise ValueError(f"unknown dataset: {name}")
