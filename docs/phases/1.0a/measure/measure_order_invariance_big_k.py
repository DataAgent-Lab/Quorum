"""F§15: label-order invariance when K > 32 (NLI runs in padded batches of 32). Real Banking77 label names.
    python docs/phases/1.0a/measure/measure_order_invariance_big_k.py"""
import random
from quorum import ZeroShotEnsemble, data
labels = data.load("banking77")["labels"]          # 77 names, eval order
msg = "I still haven't received my new card, where is it?"
clf = ZeroShotEnsemble(device="cpu")
_, p = clf.member_and_ensemble_proba(msg, list(range(len(labels))), label_texts=labels)
perm = list(range(len(labels))); random.Random(0).shuffle(perm)
_, q = clf.member_and_ensemble_proba(msg, list(range(len(labels))), label_texts=[labels[i] for i in perm])
qd = {labels[perm[j]]: float(q[j]) for j in range(len(perm))}
d = max(abs(float(p[i]) - qd[labels[i]]) for i in range(len(labels)))
print(f"F15 K={len(labels)} max_abs_diff={d:.2e} same_top={labels[int(p.argmax())] == max(qd, key=qd.get)}")
