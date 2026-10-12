#!/usr/bin/env python3
"""Decision-model study (provenance/decision13_protocol.md), block E — PAWS (Wiki, labeled_final) paraphrase identification as noul decisions (contamination control).

Source: HF `google-research-datasets/paws` @ 161ece9501cf0a11f3e48bd356eaa82de46d6a09,
`labeled_final/test-00000-of-00001.parquet` (8,000 pairs; fields id/sentence1/sentence2/label, 1 = paraphrase).
Every test pair is used in file order (no sampling); sentence1 → A, sentence2 → B as shipped (tokenised text kept).
Licence: card `license: other`; "The dataset may be freely used for any purpose, although acknowledgement of Google
LLC ("Google") as the data source would be appreciated. The dataset is provided "AS IS" without any warranty …".
Exposure: PAWS is in tasksource and FLAN, so this set is a contamination control, reported separately (protocol §3, §9).

    python -I paws_build.py --parquet test-00000-of-00001.parquet --out test_items.jsonl.gz --manifest manifest.json
"""
import argparse, collections, gzip, hashlib, json

import pyarrow.parquet as pq

REVISION = "161ece9501cf0a11f3e48bd356eaa82de46d6a09"
QUESTION = {
    "type": "noul",
    "instructions": "Is sentence B a paraphrase of sentence A (same meaning)?",
    "criteria": {"true": "Yes: both sentences state the same meaning, even if the wording or word order differs.",
                 "false": "No: the meaning differs, even if the two sentences share most of their words."},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    rows = pq.read_table(a.parquet).to_pylist()
    items = []
    for k, r in enumerate(rows):
        body = {"model": "decision", "state": {"sentence_a": r["sentence1"], "sentence_b": r["sentence2"]},
                "questions": {"paraphrase": QUESTION}}
        raw = json.dumps(body).encode()
        items.append({"idx": k, "paws_id": r["id"], "gold": r["label"] == 1, "body": body,
                      "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    assert {r["label"] for r in rows} <= {0, 1}
    with gzip.open(a.out, "wt") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    same_bag = collections.Counter(it["gold"] for it in items
                                   if sorted(it["body"]["state"]["sentence_a"].split()) == sorted(it["body"]["state"]["sentence_b"].split()))
    identical = collections.Counter(it["gold"] for it in items if it["body"]["state"]["sentence_a"] == it["body"]["state"]["sentence_b"])
    bs = sorted(it["bytes"] for it in items)
    man = {"source": f"hf:google-research-datasets/paws@{REVISION} labeled_final/test-00000-of-00001.parquet",
           "parquet_sha256": hashlib.sha256(open(a.parquet, "rb").read()).hexdigest(),
           "licence": "card 'other': free use for any purpose, acknowledgement of Google appreciated, AS IS",
           "exposure": "in tasksource + FLAN -> contamination control only", "seed": None,
           "items": len(items), "paraphrase": sum(it["gold"] for it in items),
           "identical_sentences": {"paraphrase": identical[True], "not": identical[False]},
           "same_bag_of_words": {"paraphrase": same_bag[True], "not": same_bag[False]},
           "distinct_ids": len({it["paws_id"] for it in items}),
           "bytes_p50": bs[len(bs) // 2], "bytes_max": bs[-1], "question": QUESTION,
           "items_sha256": hashlib.sha256("".join(it["body_sha256"] for it in items).encode()).hexdigest()}
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "question"}, indent=1))


if __name__ == "__main__":
    main()
