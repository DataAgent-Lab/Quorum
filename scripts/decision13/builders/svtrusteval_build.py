#!/usr/bin/env python3
"""Decision-model study (provenance/decision13_protocol.md), block E — SV-TrustEval-C 1.0 (C vulnerability reasoning, native multiple choice) as choice decisions.

Source: HF dataset LLMs4CodeSecurity/SV-TrustEval-C-1.0 @a9a71254099b64ecacd85e32d655ca9fd5e40e4d, the six parquet
splits under data/. Licence: card front matter `license: mit` with a "License: MIT" badge; the HF repo has no LICENSE
file; the authors' code repo github.com/Jackline97/SV-TrustEval-C @5bb1585 has LICENSE "MIT License, Copyright (c)
2024 Anonymous". Built on Juliet (every case_id is a Juliet test-case name) — overlaps our Juliet set by origin.

Cast: every question of all 6 splits (Base_questions 754, ControlFlow 1,345, DataFlow 2,430, Counterfactual 3,748,
GoalDriven 1,159, Predictive 719 = 10,155; no sampling) is one choice question with the benchmark's own question
text and options (A–C for Base_questions, A–D otherwise). State holds the code fields the split uses: `code`;
Counterfactual `original_code` + `variant_code` (the "Original Code Snippet:" / "Code Snippet Variant:" prefixes
are dropped, they are what the keys say); GoalDriven `masked_code`; Predictive has its code in the options, so state
only says so. 226 DataFlow golds are multi-letter ("A, C"); the authors' scorer counts a prediction correct when it
is any of those letters (Eval_Script/Eval_script.py L105-109 @5bb1585), so `gold_accept` lists every accepted key.
Leak check (manifest): case_id ends in `_bad` for every row whatever the answer (no leak; not in the body anyway);
the code is already anonymised by the authors (function `Sample_Func`, no comments); residual label-ish tokens are
counted per answer and left unchanged because they do not encode the answer.

    python -I svtrusteval_build.py --data-dir <raw>/data --out items.jsonl.gz --manifest manifest.json
"""
import argparse, collections, gzip, hashlib, json, os, re

import pandas as pd

REVISION = "a9a71254099b64ecacd85e32d655ca9fd5e40e4d"
SPLITS = ["Base_questions", "ControlFlow", "DataFlow", "Counterfactual", "GoalDriven", "Predictive"]
PRED_NOTE = "The code variants to compare are given in the answer options."
TERMS = {"bad_good": r"(?i)bad|good", "g2b_b2g": r"(?i)g2b|b2g", "flaw_fix": r"(?i)\bflaw|\bfix\b|POTENTIAL",
         "cwe_in_code": r"(?i)cwe", "comment": r"/\*|//", "omit_includemain": r"OMIT|INCLUDEMAIN"}


def opt(choices):
    out = {}
    for part in choices.split("\n<--->\n"):
        k, v = part.split("<-->", 1)
        out[k.strip()] = v.strip()
    return out


def state(split, r):
    if split == "Counterfactual":
        return {"original_code": re.sub(r"^Original Code Snippet:\n", "", r["org_code"]),
                "variant_code": re.sub(r"^Code Snippet Variant:\n", "", r["new_code"])}
    if split == "GoalDriven":
        return {"masked_code": r["org_code"]}
    if split == "Predictive":
        return {"note": PRED_NOTE}
    return {"code": r["code"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    items, per_split, leak = [], {}, {}
    file_sha = {}
    for split in SPLITS:
        path = os.path.join(a.data_dir, f"{split}-00000-of-00001.parquet")
        file_sha[split] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        df = pd.read_parquet(path)
        suffix, terms = collections.Counter(), collections.Counter()
        for _, r in df.iterrows():
            criteria = opt(r["choices"])
            accept = [x.strip() for x in r["answer"].split(",")]
            assert all(k in criteria for k in accept), (split, r["qid"], r["answer"])
            q = {"type": "choice", "instructions": r["question"].strip(), "criteria": criteria}
            st = state(split, r)
            body = {"model": "decision", "state": st, "questions": {"answer": q}}
            raw = json.dumps(body).encode()
            code_text = " ".join(v for v in st.values() if v != PRED_NOTE)
            suffix[re.search(r"_(bad|good\w*)$", r["case_id"]).group(1) if re.search(r"_(bad|good\w*)$", r["case_id"]) else "other", r["answer"]] += 1
            for k, rx in TERMS.items():
                terms[k, r["answer"]] += bool(re.search(rx, code_text))
            items.append({"split": split, "qid": r["qid"], "case_id": r["case_id"], "cwe": r["CWE_ID"],
                          "distance": r["distance"], "gold": r["answer"], "gold_accept": accept, "body": body,
                          "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        per_split[split] = {"items": len(df), "answers": dict(collections.Counter(df["answer"]).most_common())}
        leak[split] = {"case_id_suffix_x_answer": {f"{s}|{ans}": n for (s, ans), n in sorted(suffix.items())},
                       "code_rows_with_term_by_answer": {k: {ans: terms[k, ans] for ans in sorted(set(df["answer"]))}
                                                         for k in TERMS if any(terms[k, ans] for ans in set(df["answer"]))}}
    with gzip.open(a.out, "wt") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"idx": i, **it}) + "\n")
    by = sorted(it["bytes"] for it in items)
    man = {"source": f"hf:LLMs4CodeSecurity/SV-TrustEval-C-1.0@{REVISION}", "parquet_sha256": file_sha,
           "licence": "MIT (HF card front matter + badge; authors' repo LICENSE @5bb1585)",
           "items": len(items), "per_split": per_split,
           "multi_letter_gold": sum(len(it["gold_accept"]) > 1 for it in items),
           "leak_check": leak, "bytes_p50": by[len(by) // 2], "bytes_max": by[-1],
           "question": "per item: the benchmark's own question text and options (type choice, qid 'answer')",
           "items_sha256": hashlib.sha256("".join(it["body_sha256"] for it in items).encode()).hexdigest()}
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
