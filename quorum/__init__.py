"""quorum — a jury of tiny open models: a cheap, calibration-free zero-shot intent classifier (an ensemble of
small open models) and the reproduction harness for the accompanying study.

    from quorum import ZeroShotEnsemble
    clf = ZeroShotEnsemble(device="cpu")
    label, confidence, dist = clf.predict("my card was declined", labels, label_texts=verbalized_names)
"""
from .ensemble import ZeroShotEnsemble
from . import data, metrics

__all__ = ["ZeroShotEnsemble", "data", "metrics"]
