#!/usr/bin/env python3
"""Decision-model study (provenance/decision13_protocol.md), block E — Juliet Test Suite for C/C++ 1.3 as noul decisions (synthetic FLOOR, never a headline).

Source: NIST SARD test suite 112, `2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip`
(sha256 ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb, downloaded 2026-10-08).
Licence (SARD page for suite 112, read 2026-10-08): "not subject to copyright protection in the United States
(17 USC 105) ... To the extent NIST may claim Foreign Rights ... made available to you under the CC0 1.0 Public
Domain License."

Why not the HF mirror LorenzH/juliet_test_suite_c_1_3 @ad16b1c (card: cc0-1.0): its good/bad extraction is broken —
in the 20,177-row test split only 6,882 `good` texts contain any function definition (the rest are a header comment
plus a stray `return 0; }`), and 12,822 `bad` texts keep an orphan `printLine("Calling bad()...")` main fragment
(measured 2026-10-08). An empty sample == "good", i.e. the label is leaked. So we extract ourselves.

Cast: one item per test case per side, every test case in manifest.xml (no sampling):
  bad item  = all files of the case with `#ifndef OMITGOOD` blocks removed  -> gold True  (contains the flaw)
  good item = all files of the case with `#ifndef OMITBAD`  blocks removed  -> gold False
The `#ifdef INCLUDEMAIN` main() is removed from both. Extraction is a nesting-aware text scan of the preprocessor
conditionals (no compiler / preprocessor is run; the code is never compiled or executed).
Leak stripping (counts before/after go in the manifest): every comment is removed (Juliet comments are template
metadata: Filename, BadSource/GoodSink, CWE, POTENTIAL FLAW, FIX); identifiers containing bad/good/G2B/B2G or
starting with CWE<n> (function, class, namespace, global names) are renamed func_N / name_N per item; #include of a
test-case header is mapped to the neutral file name; label words inside string literals are neutralised; the
`good()` dispatcher (body = calls only) is dropped and `static` is dropped from function definitions, because the
bad entry point is never static and has no dispatcher. File names are replaced by file_1.c, file_2.h, ...
Residual structural signals (file count, function count, size by label) are reported, not hidden.
Measured 2026-10-08: 64,123 manifest cases → 24 skipped (51 manifest files absent from the zip), 12 have only one
side (kept, unpaired), 20 cases render to identical bodies on both sides (40 items, flagged `identical_pair`,
kept); 128,186 items = 64,099 vulnerable + 64,087 not. `in_hf_test_split` marks items whose files are in the HF
mirror's 20,177-file test split (35,119 items), so a smaller pre-registered subset can be taken without sampling.

    python -I juliet_build.py --root <unzipped>/C --hf-test jts_c_1_3_test.csv --out items.jsonl.gz --manifest m.json
"""
import argparse, collections, csv, gzip, hashlib, json, os, re

QUESTION = {
    "type": "noul",
    "instructions": "Does this code contain a security vulnerability?",
    "criteria": {"true": "Yes: the code contains a security weakness (for example a memory-safety error, a missing or "
                         "wrong check, injection, an unsafe resource, API or concurrency use) that can be triggered "
                         "when it runs.",
                 "false": "No: the code as written does not contain such a weakness."},
}
DIRECTIVE = re.compile(r"^\s*#\s*(ifndef|ifdef|if|elif|else|endif)\b\s*(\S*)")
SPECIAL = {"OMITBAD", "OMITGOOD", "INCLUDEMAIN"}
LEX = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/|[A-Za-z_]\w*', re.S)
LEAKY_ID = re.compile(r"(?i)bad|good|g2b|b2g|cwe\d+|omit")
STR_WORDS = re.compile(r"(?i)bad|good|\bbenign\b|fixed|\bflaw\b|cwe\d*")   # "fixed" also inside "fixedstringtest" (3,720 good vs 60 bad before)
# Leak measures (rows containing the pattern, counted by label, before and after stripping).
MEASURES = {
    "bad_or_good_token": re.compile(r"(?i)bad|good"),
    "g2b_b2g": re.compile(r"(?i)g2b|b2g"),
    "cwe": re.compile(r"(?i)cwe"),
    "flaw_or_fix_word": re.compile(r"(?i)\bflaw|\bfix\b|\bfix:"),
    "source_sink_label": re.compile(r"(?i)(bad|good)(source|sink)"),
    "omit_or_includemain": re.compile(r"OMITBAD|OMITGOOD|INCLUDEMAIN"),
    "testcase_filename": re.compile(r"CWE\d+_\w+?_\d+[a-z]?(?:_\w+)?\.(?:c|cpp|h)\b"),
    "comment": re.compile(r"/\*|//"),
    "benign_word": re.compile(r"(?i)\bbenign\b"),
}
FUNC_DEF = re.compile(r"^(?:static\s+)?[A-Za-z_][\w\s\*&:<>,~]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\n?\{", re.M)


def select(src, drop):
    """Keep the side whose OMIT macro is NOT defined: drop `#ifndef <drop>` blocks and `#ifdef INCLUDEMAIN`."""
    out, stack = [], []          # frame: [special, skip]
    for line in src.splitlines():
        m = DIRECTIVE.match(line)
        skipping = any(f[1] for f in stack)
        if m:
            kw, arg = m.group(1), m.group(2)
            if kw in ("ifndef", "ifdef", "if"):
                special = arg in SPECIAL and kw in ("ifndef", "ifdef")
                skip = (kw == "ifndef" and arg == drop) or (kw == "ifdef" and arg == "INCLUDEMAIN") if special else False
                stack.append([special, skip])
                if not special and not skipping:
                    out.append(line)
                continue
            if kw in ("else", "elif") and stack:
                if stack[-1][0]:
                    stack[-1][1] = not stack[-1][1]
                elif not any(f[1] for f in stack[:-1]):
                    out.append(line)
                continue
            if kw == "endif" and stack:
                f = stack.pop()
                if not f[0] and not any(g[1] for g in stack):
                    out.append(line)
                continue
        if not skipping:
            out.append(line)
    return "\n".join(out)


def is_empty(code):
    rest = [l for l in code.splitlines()
            if l.strip() and not l.strip().startswith("#include") and not re.match(r"^\s*(namespace\s+\w+\s*)?[{}]?\s*$", l)
            and not re.match(r"^\s*#\s*(pragma|define\s+_CRT)", l)]
    return not rest


def drop_dispatcher(code):
    # `void X_good()\n{\n    goodG2B();\n    goodB2G();\n}` (or C++ `void good()`): calls only.
    return re.sub(r"^void\s+\w*good\s*\(\s*(?:void)?\s*\)\s*\{\s*(?:\w+\s*\(\s*\)\s*;\s*)+\}\s*\n?", "", code, flags=re.M)


def nocomments(code):
    return LEX.sub(lambda m: "" if m.group(0)[:2] in ("//", "/*") else m.group(0), code)


def neutralise(files, file_names, name_map):
    """files: [(orig_basename, code)]; name_map: orig basename -> neutral name. Returns [(neutral_name, code)]."""
    ids, out = {}, []
    def rename(name, nxt):
        if name not in ids:
            kind = "func" if nxt.lstrip().startswith("(") else "name"
            ids[name] = f"{kind}_{sum(1 for v in ids.values() if v.startswith(kind)) + 1}"
        return ids[name]
    for (orig, code), new in zip(files, file_names):
        parts, pos = [], 0
        for m in LEX.finditer(code):
            parts.append(code[pos:m.start()]); tok = m.group(0); pos = m.end()
            if tok.startswith("//") or tok.startswith("/*"):
                continue                                           # every comment removed
            if tok[0] == '"':
                inner = tok[1:-1]
                if os.path.basename(inner) in name_map:
                    tok = '"' + name_map[os.path.basename(inner)] + '"'
                elif re.match(r"CWE\d+_", inner):
                    tok = '"testcase.h"'
                else:
                    tok = '"' + STR_WORDS.sub("", inner).replace("  ", " ") + '"' if STR_WORDS.search(inner) else tok
            elif tok[0] != "'" and LEAKY_ID.search(tok):
                tok = rename(tok, code[pos:pos + 3])
            parts.append(tok)
        parts.append(code[pos:])
        c = "".join(parts)
        c = re.sub(r"^static\s+(?=[A-Za-z_][\w\s\*&:<>,]*?\b\w+\s*\([^;{}]*\)\s*\n?\{)", "", c, flags=re.M)
        c = "\n".join(l.rstrip() for l in c.splitlines())
        c = re.sub(r"\n{3,}", "\n\n", c).strip() + "\n"
        out.append((new, c))
    return out


def measure(text):
    return {k: bool(rx.search(text)) for k, rx in MEASURES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--hf-test", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    man_xml = open(os.path.join(a.root, "manifest.xml"), encoding="utf-8").read()
    cases = [re.findall(r'<file path="([^"]+)"', c) for c in re.findall(r"<testcase>(.*?)</testcase>", man_xml, re.S)]
    where = {}
    for d, _, fs in os.walk(os.path.join(a.root, "testcases")):
        for f in fs:
            where[f] = os.path.join(d, f)
    csv.field_size_limit(10 ** 9)
    hf_test = {os.path.basename(r["filename"]) for r in csv.DictReader(open(a.hf_test, newline=""))}

    items, before, after = [], collections.Counter(), collections.Counter()
    stats = collections.Counter()
    for ci, paths in enumerate(cases):
        if any(os.path.basename(p) not in where for p in paths):     # 51 manifest files are absent from the zip
            stats["case_skipped_file_missing_from_zip"] += 1
            continue
        srcs = [(os.path.basename(p), open(where[os.path.basename(p)], encoding="utf-8", errors="replace").read())
                for p in paths]
        cwe = re.match(r"(CWE\d+)", srcs[0][0]).group(1)
        sides = {}
        for gold, drop in ((True, "OMITGOOD"), (False, "OMITBAD")):
            kept = [(n, select(s, drop)) for n, s in srcs]
            sides[gold] = [(n, c) for n, c in kept if not is_empty(nocomments(c))]
        if not sides[True] or not sides[False]:
            stats["case_missing_a_side"] += 1
            if not sides[True] and not sides[False]:
                continue
        for gold in (True, False):
            kept = sides[gold]
            if not kept:
                continue
            raw_text = "\n".join(c for _, c in kept)
            names = [f"file_{i + 1}{os.path.splitext(n)[1]}" for i, (n, _) in enumerate(kept)]
            nocomm = [(n, drop_dispatcher(nocomments(c)) if not gold else nocomments(c)) for n, c in kept]
            clean = neutralise(nocomm, names, {n: new for (n, _), new in zip(kept, names)})
            state = {"files": [{"name": n, "code": c} for n, c in clean]}
            body = {"model": "decision", "state": state, "questions": {"vulnerable": QUESTION}}
            raw = json.dumps(body).encode()
            text = "\n".join(c for _, c in clean)
            for k, v in measure(raw_text).items():
                before[gold, k] += v
            for k, v in measure(text).items():
                after[gold, k] += v
            items.append({"case": ci, "cwe": cwe, "files": [n for n, _ in kept], "gold": gold,
                          "in_hf_test_split": any(n in hf_test for n, _ in kept),
                          "n_files": len(clean), "n_funcs": len(FUNC_DEF.findall(text)),
                          "body": body, "body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    pairs = collections.defaultdict(dict)
    for it in items:
        pairs[it["case"]][it["gold"]] = it["body_sha256"]
    for it in items:                  # both sides render to the same body: unanswerable, flagged (kept, not dropped)
        p = pairs[it["case"]]
        it["identical_pair"] = len(p) == 2 and p[True] == p[False]
    with gzip.open(a.out, "wt") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"idx": i, **it}) + "\n")

    def bylab(c):
        return {k: {"vulnerable": c[True, k], "not_vulnerable": c[False, k]} for k in MEASURES}
    def dist(key, gold):
        v = sorted(it[key] for it in items if it["gold"] == gold)
        return {"p50": v[len(v) // 2], "mean": round(sum(v) / len(v), 2), "max": v[-1]}
    by = sorted(it["bytes"] for it in items)
    man = {"source": "NIST SARD suite 112 juliet-test-suite-for-c-cplusplus-v1-3.zip",
           "zip_sha256": "ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb",
           "licence": "US public domain (17 USC 105); CC0 1.0 for foreign rights (SARD suite 112 page)",
           "synthetic_floor_not_headline": True,
           "test_cases": len(cases), "items": len(items), "vulnerable": sum(it["gold"] for it in items),
           "not_vulnerable": sum(not it["gold"] for it in items), "stats": dict(stats),
           "items_in_hf_test_split": sum(it["in_hf_test_split"] for it in items),
           "items_identical_pair_flagged": sum(it["identical_pair"] for it in items),
           "leak_rows_before_strip": bylab(before), "leak_rows_after_strip": bylab(after),
           "residual_structure": {k: {"vulnerable": dist(k, True), "not_vulnerable": dist(k, False)}
                                  for k in ("n_files", "n_funcs", "bytes")},
           "bytes_p50": by[len(by) // 2], "bytes_max": by[-1], "bytes_total": sum(by), "question": QUESTION,
           "items_sha256": hashlib.sha256("".join(it["body_sha256"] for it in items).encode()).hexdigest()}
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "question"}, indent=1))


if __name__ == "__main__":
    main()
