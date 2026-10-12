#!/usr/bin/env python3
"""Decision-model study (provenance/decision13_protocol.md), block E — EquiBench program-equivalence pairs as noul decisions.

Source: HF `anjiangwei/EquiBench-Datasets` @ 620244852a8bf20614265647ab751219dfc35094, the six configs of the card
(DCE, OJ_A, OJ_V, OJ_VA, STOKE → `preprocess/STOKE/pairs_v2.jsonl` as the card maps it, TVM), each a single `train`
split of 400 pairs (200 equivalent / 200 not). All 2,400 pairs are used as the eval set (no sampling); pair order
and program_1/program_2 order are the dataset's.
Licence: the HF card says `license: mit`; the GitHub repo Anjiang-Wei/EquiBench @ 217332098ab88c5bdf0ee76fb675db7db08e8781
ships an Apache-2.0 LICENSE (README "Apache License 2.0"). Conflict recorded; both permit evaluation use.

Cast (deviates from the planned single generic question — measured on the real data, 2026-10-08):
- Gold labels follow a category-specific notion of equivalence (repo `prompts.yaml` @ 217332…): DCE = same program
  state at every corresponding reachable point (its own few-shot example labels an output-identical pair NO);
  STOKE = same bits in the live-out registers for any bits in the def-in registers; OJ_* = same output for every
  valid input under the problem description; TVM = same mathematical output, ignoring floating-point rounding. A
  generic "same observable behaviour for every input" question would contradict the gold on part of the items, so
  each category gets its own question text carrying the benchmark's definition, and the state carries what the
  definition needs: `problem_description` (the dataset's problem HTML, verbatim) for OJ_*, `input_registers` /
  `output_registers` (def_in / live_out) for STOKE.
- Label leak removed: in STOKE the STOKE-annotated listing format (`#!` header, `OPC=` columns, `.text/.globl/.type/
  .size` directives) occurs in 45/52 programs (program 1/2) of equivalent pairs and 0/0 of non-equivalent pairs.
  Every STOKE program (both labels) is therefore normalised: comments (`#…`), those directives, blank lines and
  indentation removed. Instructions are unchanged.
- File paths (which encode the label: `eq.c`/`neq.c`, `accepted_*`/`wrong_*`) never enter the state.

    python -I equibench_build.py --root raw/preprocess --out test_items.jsonl.gz --manifest manifest.json
"""
import argparse, collections, gzip, hashlib, json, re

REVISION = "620244852a8bf20614265647ab751219dfc35094"
CATS = {"DCE": "DCE/pairs.jsonl", "OJ_A": "OJ_A/pairs.jsonl", "OJ_V": "OJ_V/pairs.jsonl",
        "OJ_VA": "OJ_VA/pairs.jsonl", "STOKE": "STOKE/pairs_v2.jsonl", "TVM": "TVM/pairs.jsonl"}
LANG = {"DCE": "C", "OJ_A": "Python", "OJ_V": "Python", "OJ_VA": "Python",
        "STOKE": "x86-64 assembly (AT&T syntax)", "TVM": "CUDA"}
DEFN = {
    "DCE": "Here equivalent means: when run on the same input, the two programs always have the same program state "
           "at every corresponding point reachable by program execution.",
    "OJ": "Here equivalent means: for every valid input allowed by the problem description, the two programs always "
          "produce the same output.",
    "STOKE": "Here equivalent means: for any input bits in the input registers, the two programs always leave the "
             "same bits in the output registers. Differences in other registers do not matter.",
    "TVM": "Here equivalent means: on the same valid input, the two programs always compute the same mathematical "
           "output (ignoring floating-point rounding differences).",
}
CRITERIA = {"true": "Yes: the two programs are equivalent in the sense defined above.",
            "false": "No: there is at least one (valid) input on which they differ in the sense defined above."}


def question(cat):
    lang = LANG[cat]
    return {"type": "noul",
            "instructions": f"Are these two {lang} programs semantically equivalent? " + DEFN[cat.split("_")[0]],
            "criteria": CRITERIA}


DIRECTIVE = re.compile(r"^\.(text|globl|type|size)\b")
ANNOT = re.compile(r"#!|OPC=")


def norm_asm(code):
    out = []
    for line in code.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not DIRECTIVE.match(line):
            out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    items, per_cat, file_sha = [], {}, {}
    for cat, rel in CATS.items():
        raw_bytes = open(f"{a.root}/{rel}", "rb").read()
        file_sha[rel] = hashlib.sha256(raw_bytes).hexdigest()
        rows = [json.loads(l) for l in raw_bytes.decode().splitlines() if l.strip()]
        q = question(cat)
        annot_before = collections.Counter()
        for k, r in enumerate(rows):
            pa, pb = r["program_1_code"], r["program_2_code"]
            state = {"language": LANG[cat]}
            if cat.startswith("OJ"):
                state["problem_description"] = r["problem_html_content"]
            if cat == "STOKE":
                annot_before[r["truth_label"]] += bool(ANNOT.search(pa)) + bool(ANNOT.search(pb))
                pa, pb = norm_asm(pa), norm_asm(pb)
                state["input_registers"], state["output_registers"] = r["problem_def_in"], r["problem_live_out"]
            state["program_a"], state["program_b"] = pa, pb
            body = {"model": "decision", "state": state, "questions": {"equivalent": q}}
            raw = json.dumps(body).encode()
            items.append({"category": cat, "row": k, "pair_id": r.get("pair_id"), "problem_id": r["problem_id"],
                          "gold": bool(r["truth_label"]), "body": body,
                          "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        cs = [it for it in items if it["category"] == cat]
        per_cat[cat] = {"items": len(cs), "equivalent": sum(it["gold"] for it in cs),
                        "bytes_p50": sorted(it["bytes"] for it in cs)[len(cs) // 2], "bytes_max": max(it["bytes"] for it in cs),
                        "question": q}
        if cat == "STOKE":
            left = collections.Counter()
            for it in cs:
                s = it["body"]["state"]
                left[it["gold"]] += bool(ANNOT.search(s["program_a"]) or ANNOT.search(s["program_b"]) or "#" in s["program_a"] + s["program_b"])
            per_cat[cat]["annotated_programs_before_norm"] = {"equivalent": annot_before[True], "not_equivalent": annot_before[False]}
            per_cat[cat]["items_with_annotation_after_norm"] = {"equivalent": left[True], "not_equivalent": left[False]}
    with gzip.open(a.out, "wt") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"idx": i, **it}) + "\n")
    leak_words = re.compile(r"\b(neq|not[ _]equivalent|inequivalent|wrong[ _]answer|accepted)\b", re.I)
    lw = collections.Counter(it["gold"] for it in items
                             if leak_words.search(it["body"]["state"]["program_a"] + "\n" + it["body"]["state"]["program_b"]))
    bs = sorted(it["bytes"] for it in items)
    man = {"source": f"hf:anjiangwei/EquiBench-Datasets@{REVISION}", "split": "train (only split; used whole as eval set)",
           "licence": "HF card: MIT; GitHub Anjiang-Wei/EquiBench@217332098ab88c5bdf0ee76fb675db7db08e8781 LICENSE: Apache-2.0 (conflict recorded)",
           "file_sha256": file_sha, "seed": None, "items": len(items), "equivalent": sum(it["gold"] for it in items),
           "per_category": per_cat,
           "label_words_in_programs": {"equivalent": lw[True], "not_equivalent": lw[False]},
           "bytes_p50": bs[len(bs) // 2], "bytes_max": bs[-1],
           "items_over_64KiB": sum(b > 65536 for b in bs), "items_over_256KiB": sum(b > 262144 for b in bs),
           "items_sha256": hashlib.sha256("".join(it["body_sha256"] for it in items).encode()).hexdigest()}
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "per_category"}, indent=1))
    print(json.dumps({c: {k: v for k, v in d.items() if k != "question"} for c, d in per_cat.items()}, indent=1))


if __name__ == "__main__":
    main()
