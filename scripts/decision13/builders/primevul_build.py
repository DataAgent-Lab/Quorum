#!/usr/bin/env python3
"""Decision-model study (provenance/decision13_protocol.md), block E — PrimeVul paired test (C/C++ vulnerable function + its patched version) as two item sets.

Source: PrimeVul **v0** `primevul_test_paired.jsonl` (the release the paper's Table III describes: test paired
564 vulnerable + 564 benign), from the authors' Google Drive folder 19iLaNDS0z99N8kB_jBRTmDLehwZBolMY linked in
github.com/DLVulDet/PrimeVul README @6f54687 (file id 1rNDKb65Yn1rCBURzHeBcH4jVVdQ1oD4r,
sha256 54a16a5ab663d11d1ffa5f48a98b0f996e82ee85f48a2b999334f6c1a7d96e41, 1,128 rows, downloaded 2026-10-08).
Licence: MIT (github.com/DLVulDet/PrimeVul LICENSE @6f54687: "MIT License, Copyright (c) 2024 DLVulDet").
Not used: the HF mirror colin/PrimeVul @4fd7158 (card `license: mit`) is byte-identical to the authors' v0.1
release (sha256 384758c2…, 870 rows = 435 pairs), a different, smaller pairing than the paper's 564.

Rows come as consecutive (target 1 = vulnerable, target 0 = patched) pairs from the same commit (verified: 564/564,
0 identical). Two item sets, every row used:
  pair : choice "Which of these two versions … contains the vulnerability?" A/B order shuffled, Random(20261008)
  func : noul per function "Does this function contain a security vulnerability?" (1,128 items, 564/564)
State = code only (v0 ships no commit message / CVE text; project, commit, CWE are kept as item metadata only).
Leak strip: whole comments containing cve / cwe / vuln / fix / patch / security / exploit / overflow / sanitiz /
bad / good are removed from both versions alike (patch-added comments explain the fix); code is not touched.
Counts before/after (rows by label, pairs where a term is on one side only) are in the manifest.

    python -I primevul_build.py --paired primevul_test_paired.jsonl --out-dir DIR --manifest-dir DIR
"""
import argparse, collections, gzip, hashlib, json, os, random, re

SEED = 20261008
LEX = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.S)
LEAK_COMMENT = re.compile(r"(?i)CVE-\d|\bCWE|vuln|\bfix|\bpatch|\bsecurity\b|\bexploit|overflow|saniti[sz]|\b(bad|good)\b")
TERMS = {"cve": r"CVE-\d{4}-\d+", "cwe": r"(?i)\bCWE[-_ ]?\d+", "vuln": r"(?i)vuln", "fix": r"(?i)\bfix",
         "patch": r"(?i)\bpatch", "security_exploit": r"(?i)\bsecurity\b|\bexploit", "overflow": r"(?i)overflow",
         "bad_good": r"(?i)\b(bad|good)\b", "sanitiz": r"(?i)saniti[sz]"}
PAIR_Q = {
    "type": "choice",
    "instructions": "Which of these two versions of the function contains the security vulnerability?",
    "criteria": {"A": "Version A contains the vulnerability; version B is the corrected version.",
                 "B": "Version B contains the vulnerability; version A is the corrected version."},
}
FUNC_Q = {
    "type": "noul",
    "instructions": "Does this function contain a security vulnerability?",
    "criteria": {"true": "Yes: the function contains a security weakness (for example a memory-safety error, a missing "
                         "or wrong check, injection, an unsafe resource, API or concurrency use) that can be "
                         "triggered when it runs.",
                 "false": "No: the function as written does not contain such a weakness."},
}


def strip_leaky_comments(code):
    n = 0
    def sub(m):
        nonlocal n
        t = m.group(0)
        if t[:2] in ("//", "/*") and LEAK_COMMENT.search(t):
            n += 1
            return "" if t.startswith("//") else " "
        return t
    return LEX.sub(sub, code), n


def count(rows_by_label):
    c = collections.Counter()
    for lab, code in rows_by_label:
        for k, rx in TERMS.items():
            c[lab, k] += bool(re.search(rx, code))
    return {k: {"vulnerable": c[True, k], "patched": c[False, k]} for k in TERMS}


def one_sided(pairs):
    c = collections.Counter()
    for v, p in pairs:
        for k, rx in TERMS.items():
            x, y = bool(re.search(rx, v)), bool(re.search(rx, p))
            if x != y:
                c[k + (":only_vulnerable" if x else ":only_patched")] += 1
    return dict(sorted(c.items()))


def write(path, items):
    with gzip.open(path, "wt") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def manifest(items, question, extra):
    by = sorted(it["bytes"] for it in items)
    return {**extra, "items": len(items), "bytes_p50": by[len(by) // 2], "bytes_max": by[-1], "question": question,
            "items_sha256": hashlib.sha256("".join(it["body_sha256"] for it in items).encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    a = ap.parse_args()
    raw_bytes = open(a.paired, "rb").read()
    rows = [json.loads(l) for l in raw_bytes.decode().splitlines() if l.strip()]
    assert len(rows) % 2 == 0
    pairs = []
    for i in range(0, len(rows), 2):
        v, p = rows[i], rows[i + 1]
        assert str(v["target"]) == "1" and str(p["target"]) == "0" and v["commit_id"] == p["commit_id"], i
        pairs.append((v, p))
    clean, removed = {}, collections.Counter()
    for v, p in pairs:
        for r in (v, p):
            clean[r["idx"]], n = strip_leaky_comments(r["func"])
            removed[str(r["target"]) == "1"] += n
    rng = random.Random(SEED)
    pair_items, func_items = [], []
    for k, (v, p) in enumerate(pairs):
        order = [("vulnerable", v), ("patched", p)]
        rng.shuffle(order)
        body = {"model": "decision", "state": {"version_A": clean[order[0][1]["idx"]], "version_B": clean[order[1][1]["idx"]]},
                "questions": {"vulnerable_version": PAIR_Q}}
        raw = json.dumps(body).encode()
        pair_items.append({"idx": k, "pair": k, "project": v["project"], "commit_id": v["commit_id"],
                           "vulnerable_row_idx": v["idx"], "patched_row_idx": p["idx"],
                           "gold": "A" if order[0][0] == "vulnerable" else "B", "body": body,
                           "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                           "identical_after_strip": clean[v["idx"]] == clean[p["idx"]]})
        for r in (v, p):
            body = {"model": "decision", "state": {"function": clean[r["idx"]]}, "questions": {"vulnerable": FUNC_Q}}
            raw = json.dumps(body).encode()
            func_items.append({"idx": len(func_items), "pair": k, "project": r["project"], "commit_id": r["commit_id"],
                               "row_idx": r["idx"], "gold": str(r["target"]) == "1", "body": body,
                               "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    os.makedirs(a.out_dir, exist_ok=True)
    write(os.path.join(a.out_dir, "pair_test_items.jsonl.gz"), pair_items)
    write(os.path.join(a.out_dir, "func_test_items.jsonl.gz"), func_items)
    leak = {"comments_removed": {"vulnerable": removed[True], "patched": removed[False]},
            "rows_with_term_before": count([(str(r["target"]) == "1", r["func"]) for vp in pairs for r in vp]),
            "rows_with_term_after": count([(str(r["target"]) == "1", clean[r["idx"]]) for vp in pairs for r in vp]),
            "pairs_term_one_side_before": one_sided([(v["func"], p["func"]) for v, p in pairs]),
            "pairs_term_one_side_after": one_sided([(clean[v["idx"]], clean[p["idx"]]) for v, p in pairs])}
    src = {"source": "PrimeVul v0 primevul_test_paired.jsonl (authors' Drive, file id 1rNDKb65Yn1rCBURzHeBcH4jVVdQ1oD4r)",
           "source_sha256": hashlib.sha256(raw_bytes).hexdigest(), "licence": "MIT (DLVulDet/PrimeVul LICENSE @6f54687)",
           "pairs": len(pairs), "leak_check": leak}
    m1 = manifest(pair_items, PAIR_Q, {**src, "seed": SEED, "gold_A": sum(it["gold"] == "A" for it in pair_items),
                                       "pairs_identical_after_strip": sum(it["identical_after_strip"] for it in pair_items)})
    m2 = manifest(func_items, FUNC_Q, {**src, "vulnerable": sum(it["gold"] for it in func_items),
                                       "patched": sum(not it["gold"] for it in func_items)})
    json.dump(m1, open(os.path.join(a.manifest_dir, "primevul_pair_test_manifest.json"), "w"), indent=1)
    json.dump(m2, open(os.path.join(a.manifest_dir, "primevul_func_test_manifest.json"), "w"), indent=1)
    for m in (m1, m2):
        print(json.dumps({k: v for k, v in m.items() if k not in ("question", "leak_check")}, indent=1))
    print(json.dumps(leak, indent=1))


if __name__ == "__main__":
    main()
