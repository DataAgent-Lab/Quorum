#!/usr/bin/env python3
"""Decision-model study (provenance/decision13_protocol.md), block E — SecVulEval (real-world C/C++ functions from NVD-linked commits) as noul decisions.

Source: HF dataset arag0rn/SecVulEval @bde3e7c3225cc61c9ad727133737255428a2ea3d, data/train-00000-of-00001.parquet
(the only split released: 25,440 functions). Licence: card front matter `license: mit`; the card's own "License:"
line is still the template placeholder ("[More Information Needed]", commented out) and the repo has no LICENSE file
→ recorded as "MIT per card metadata only".

Cast: every row (no sampling, no split of our own) is one noul question "Does this function contain a security
vulnerability?" with gold = is_vulnerable. Non-vulnerable rows include the patched versions of vulnerable functions
(fixed_func_idx) and other functions touched by the same commits. State = the function body only; commit message,
CVE / CWE lists, changed lines and the LLM-written `context` stay out of the body (they encode the label) and are
kept only as item metadata where useful. Leak strip as for PrimeVul: whole comments containing cve / cwe / vuln / fix /
patch / security / exploit / overflow / sanitiz / bad / good are removed from every row alike; code is not touched.
Counts before/after by label, and on (vulnerable, fixed) pairs, are in the manifest. No body is truncated.
Exposure: the data is public on HF as a train split → treat as possibly seen in training.

    python -I secvuleval_build.py --parquet train-00000-of-00001.parquet --out items.jsonl.gz --manifest manifest.json
"""
import argparse, collections, gzip, hashlib, json, re

import pandas as pd

REVISION = "bde3e7c3225cc61c9ad727133737255428a2ea3d"
LEX = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.S)
LEAK_COMMENT = re.compile(r"(?i)CVE-\d|\bCWE|vuln|\bfix|\bpatch|\bsecurity\b|\bexploit|overflow|saniti[sz]|\b(bad|good)\b")
TERMS = {"cve": r"CVE-\d{4}-\d+", "cwe": r"(?i)\bCWE[-_ ]?\d+", "vuln": r"(?i)vuln", "fix": r"(?i)\bfix",
         "patch": r"(?i)\bpatch", "security_exploit": r"(?i)\bsecurity\b|\bexploit", "overflow": r"(?i)overflow",
         "bad_good": r"(?i)\b(bad|good)\b", "sanitiz": r"(?i)saniti[sz]"}
QUESTION = {
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


def rows_with(texts):
    c = collections.Counter()
    for lab, code in texts:
        for k, rx in TERMS.items():
            c[lab, k] += bool(re.search(rx, code))
    return {k: {"vulnerable": c[True, k], "not_vulnerable": c[False, k]} for k in TERMS}


def one_sided(pairs):
    c = collections.Counter()
    for v, p in pairs:
        for k, rx in TERMS.items():
            x, y = bool(re.search(rx, v)), bool(re.search(rx, p))
            if x != y:
                c[k + (":only_vulnerable" if x else ":only_fixed")] += 1
    return dict(sorted(c.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.parquet)
    clean, removed, items = {}, collections.Counter(), []
    for _, r in df.iterrows():
        code, n = strip_leaky_comments(r["func_body"])
        clean[int(r["idx"])] = code
        removed[bool(r["is_vulnerable"])] += n
        body = {"model": "decision", "state": {"function": code}, "questions": {"vulnerable": QUESTION}}
        raw = json.dumps(body).encode()
        fx = r["fixed_func_idx"]
        items.append({"idx": len(items), "dataset_idx": int(r["idx"]), "project": r["project"],
                      "commit_id": r["commit_id"], "filepath": r["filepath"], "func_name": r["func_name"],
                      "gold": bool(r["is_vulnerable"]),
                      "fixed_func_idx": None if pd.isna(fx) else int(fx),
                      "body": body, "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    with gzip.open(a.out, "wt") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    orig = {int(r["idx"]): r["func_body"] for _, r in df.iterrows()}
    pairs = [(it["dataset_idx"], it["fixed_func_idx"]) for it in items
             if it["gold"] and it["fixed_func_idx"] is not None and it["fixed_func_idx"] in orig
             and it["fixed_func_idx"] != it["dataset_idx"]]
    by = sorted(it["bytes"] for it in items)
    man = {"source": f"hf:arag0rn/SecVulEval@{REVISION} data/train-00000-of-00001.parquet",
           "parquet_sha256": hashlib.sha256(open(a.parquet, "rb").read()).hexdigest(),
           "licence": "MIT per card metadata only (card License line is an unfilled placeholder; no LICENSE file)",
           "items": len(items), "vulnerable": sum(it["gold"] for it in items),
           "not_vulnerable": sum(not it["gold"] for it in items),
           "vulnerable_without_fixed_pair": sum(1 for it in items if it["gold"] and it["fixed_func_idx"] is None),
           "leak_check": {"comments_removed": {"vulnerable": removed[True], "not_vulnerable": removed[False]},
                          "rows_with_term_before": rows_with([(it["gold"], orig[it["dataset_idx"]]) for it in items]),
                          "rows_with_term_after": rows_with([(it["gold"], clean[it["dataset_idx"]]) for it in items]),
                          "vuln_fixed_pairs": len(pairs),
                          "pairs_term_one_side_before": one_sided([(orig[v], orig[p]) for v, p in pairs]),
                          "pairs_term_one_side_after": one_sided([(clean[v], clean[p]) for v, p in pairs])},
           "bytes_p50": by[len(by) // 2], "bytes_p99": by[int(len(by) * 0.99)], "bytes_max": by[-1],
           "bytes_total": sum(by), "question": QUESTION,
           "items_sha256": hashlib.sha256("".join(it["body_sha256"] for it in items).encode()).hexdigest()}
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "question"}, indent=1))


if __name__ == "__main__":
    main()
