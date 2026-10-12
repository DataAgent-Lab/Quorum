#!/usr/bin/env python3
"""Decision-model study (provenance/decision13_protocol.md), block E — POJ-104 same-problem program pairs as noul decisions.

Source: HF `google/code_x_glue_cc_clone_detection_poj104` @ 701869dece36b32174ee48277eada423cdcf699c,
`data/test-00000-of-00001.parquet` (12,000 C/C++ programs, 24 problems `81`..`104` x 500, fields id/code/label).
Licence: card `license: c-uda`, "Computational Use of Data Agreement (C-UDA) License."

Pairing (POJ-104 is a retrieval benchmark; the pairs are ours, frozen here): every test program is an anchor
exactly once, in `id` order (no sampling). Anchor k (0-based) gets a positive partner when k is even (another test
program of the SAME problem) and a negative partner when k is odd (a test program of a DIFFERENT problem). The
partner is drawn uniformly with random.Random(20261008); the A/B order of each pair is then shuffled with the same
RNG (draw, shuffle, next anchor). Code is passed verbatim (CRLF kept, nothing stripped). Leak check: problem ids
do not appear in code more for the own problem than for others; a few programs carry homework/file header comments
(counted per label in the manifest, not stripped).

    python -I poj104_build.py --parquet test-00000-of-00001.parquet --out test_items.jsonl.gz --manifest manifest.json
"""
import argparse, collections, gzip, hashlib, json, random, re

import pyarrow.parquet as pq

REVISION = "701869dece36b32174ee48277eada423cdcf699c"
SEED = 20261008
QUESTION = {
    "type": "noul",
    "instructions": "Do these two programs solve the same programming problem?",
    "criteria": {"true": "Yes: both programs are solutions to the same problem statement, even if written differently.",
                 "false": "No: they solve different problems, even if they look similar or use similar techniques."},
}
HEADER_TAG = re.compile(r"homework|\bhw\b|\bpoj\b|\boj\b|problem|@file", re.I)
NUM = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    rows = pq.read_table(a.parquet).to_pylist()
    rows.sort(key=lambda r: int(r["id"]))
    byid = {int(r["id"]): r for r in rows}
    by_prob = collections.defaultdict(list)
    for r in rows:
        by_prob[r["label"]].append(int(r["id"]))
    others = {p: [i for q, ids in by_prob.items() if q != p for i in ids] for p in by_prob}
    rng, items = random.Random(SEED), []
    for k, r in enumerate(rows):
        anc, p = int(r["id"]), r["label"]
        gold = k % 2 == 0
        partner = rng.choice([i for i in by_prob[p] if i != anc] if gold else others[p])
        pair = [anc, partner]; rng.shuffle(pair)
        ra, rb = byid[pair[0]], byid[pair[1]]
        body = {"model": "decision", "state": {"language": "C/C++", "program_a": ra["code"], "program_b": rb["code"]},
                "questions": {"same_problem": QUESTION}}
        raw = json.dumps(body).encode()
        items.append({"idx": k, "anchor_id": anc, "a_id": pair[0], "b_id": pair[1],
                      "a_problem": ra["label"], "b_problem": rb["label"], "gold": gold,
                      "body": body, "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    assert all(it["gold"] == (it["a_problem"] == it["b_problem"]) for it in items)
    with gzip.open(a.out, "wt") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    # leak checks
    own = sum(int(r["label"]) in {int(x) for x in NUM.findall(r["code"]) if len(x) < 4} for r in rows)
    other = sum(sum(q in {int(x) for x in NUM.findall(r["code"]) if len(x) < 4}
                    for q in map(int, by_prob) if q != int(r["label"])) for r in rows) / (len(by_prob) - 1)
    tag = collections.Counter(it["gold"] for it in items
                              if HEADER_TAG.search(it["body"]["state"]["program_a"] + it["body"]["state"]["program_b"]))
    crlf = collections.Counter(it["gold"] for it in items
                               if ("\r\n" in it["body"]["state"]["program_a"]) != ("\r\n" in it["body"]["state"]["program_b"]))
    upairs = collections.Counter(frozenset((it["a_id"], it["b_id"])) for it in items)
    bs = sorted(it["bytes"] for it in items)
    man = {"source": f"hf:google/code_x_glue_cc_clone_detection_poj104@{REVISION} data/test-00000-of-00001.parquet",
           "parquet_sha256": hashlib.sha256(open(a.parquet, "rb").read()).hexdigest(),
           "licence": "C-UDA (card: 'Computational Use of Data Agreement (C-UDA) License.')",
           "seed": SEED, "programs": len(rows), "problems": len(by_prob), "items": len(items),
           "same_problem": sum(it["gold"] for it in items),
           "pairing": "anchor = every test program in id order; even k -> same-problem partner, odd k -> other-problem "
                      "partner; uniform rng.choice then rng.shuffle(A,B) with random.Random(seed)",
           "duplicate_unordered_pairs": sum(v - 1 for v in upairs.values() if v > 1),
           "leak_own_problem_number_in_code": own, "leak_other_problem_number_in_code_mean": round(other, 1),
           "items_with_header_tag_comment": {"same": tag[True], "different": tag[False]},
           "items_mixed_line_endings": {"same": crlf[True], "different": crlf[False]},
           "bytes_p50": bs[len(bs) // 2], "bytes_max": bs[-1], "question": QUESTION,
           "items_sha256": hashlib.sha256("".join(it["body_sha256"] for it in items).encode()).hexdigest()}
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "question"}, indent=1))


if __name__ == "__main__":
    main()
