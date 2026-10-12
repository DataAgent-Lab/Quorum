#!/usr/bin/env python3
"""R16 Bitext 24-shot, amendment A2: the model-free "non-near-duplicate" test subset and the duplicate audit.

    python paper/scripts/bitext24_subset.py --research-root <checkout of research/phase2.0-paper-evidence>

Uses the protocol's split (`scripts/bitext24_bodies.py` split(): T, S, P = T minus S, test) and text only; no model is
loaded and no prediction is read. A test item is in the subset iff
  (1) its normalised text does not occur among the normalised texts of P, and
  (2) its maximum TF-IDF cosine to P is below 0.8 (word 1-2-grams, sublinear tf, vectoriser fit on P).
Normalisation: lowercase; every {{placeholder}} becomes one token; drop characters outside [a-z0-9 ]; split on space.
Writes docs/phases/2.0/bitext24/subset_a2.json (test indices + audit counts).
"""
import argparse, hashlib, importlib.util, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "phases" / "2.0" / "bitext24" / "subset_a2.json"
THRESHOLD = 0.8


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", re.sub(r"\{\{[^}]*\}\}", "PH", s.lower())).split())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--research-root", required=True)
    rr = Path(ap.parse_args().research_root).resolve(); sys.path.insert(0, str(rr))
    spec = importlib.util.spec_from_file_location("bb", rr / "scripts" / "bitext24_bodies.py")
    bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
    raw, names, T, S, P, test = bb.split()
    pool = [T[i]["text"] for i in P]; qs = [r["text"] for r in test]
    seen = {norm(t) for t in pool}
    exact = np.array([norm(q) in seen for q in qs])
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pool)
    A, B = v.transform(pool), v.transform(qs); best = np.zeros(len(qs))
    for s in range(0, len(qs), 500):
        best[s:s + 500] = (B[s:s + 500] @ A.T).toarray().max(1)
    keep = ~exact & (best < THRESHOLD)
    idx = [int(i) for i in np.flatnonzero(keep)]
    rec = {"amendment": "A2", "threshold": THRESHOLD, "n_test": len(qs), "n_subset": len(idx),
           "exact_duplicates": int(exact.sum()), "cos_ge_threshold": int((best >= THRESHOLD).sum()),
           "manifest_sha256": json.loads((rr / "results/bitext24/manifest_meta.json").read_text())["manifest_sha256"],
           "research_commit_files_sha256": {f: hashlib.sha256((rr / f).read_bytes()).hexdigest()
                                            for f in ("scripts/bitext24_bodies.py", "quorum/data.py")},
           "sklearn": sklearn.__version__, "subset_idx": idx,
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("n_test", "n_subset", "exact_duplicates", "cos_ge_threshold")}))


if __name__ == "__main__":
    main()
