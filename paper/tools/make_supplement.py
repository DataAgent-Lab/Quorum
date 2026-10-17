#!/usr/bin/env python3
"""Build the anonymous supplementary archive for review (task T8.9; TMLR limit 100 MB) into paper/supplement/ (gitignored).

    python paper/tools/make_supplement.py trace    # run the generators, record every repo file they read
    python paper/tools/make_supplement.py build    # assemble, anonymise, scan, zip -> paper/supplement/supplement.zip

Contents: the pre-registrations (provenance/), the paper's generators (paper/scripts/), the run and analysis code
(scripts/, the library, label descriptions), the request manifests, and every per-item file a generator reads
(from `trace`). Excluded: anything the reproduction did not licence (its predictions file, instruction and
definitions are never stored), the closed API's spend ledgers, and run logs that hold no evidence.

Anonymisation is a text substitution applied to every text file and to the contents of every gzip file: the
system's public name and package name become "R24", the adapter's Hub identifier becomes an anonymous placeholder,
and git commit identifiers stored under git/commit keys are blanked. Recorded sha256 values of code files therefore
refer to the original files, which the de-anonymised release provides. Every file is then scanned for the local
anonymity terms (paper/tools/anonymity_terms.local), the public name, the account names and links that identify
the authors; any hit stops the build.
"""
from __future__ import annotations
import gzip, json, os, re, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "paper" / "build"; TRACE = ROOT / "paper" / "tools" / "supplement_trace.json"
OUT = ROOT / "paper" / "supplement" / "supplement.zip"                    # gitignored; rebuilt before submission
LIMIT = 100 * 1024 * 1024
PY = str(ROOT / ".venv" / "bin" / "python")
# generator -> extra args; the ones needing unlicensed or external inputs are listed and not traced
GENERATORS = [("paper/scripts/numbers_bitext24.py", []), ("paper/scripts/numbers_decision13.py", []),
              ("paper/scripts/numbers_poolsize.py", []), ("paper/scripts/zero_shot_tables.py", []),
              ("paper/scripts/numbers_zeroshot_compare.py", []), ("paper/scripts/jev_zeroshot_analysis.py", []),
              ("paper/scripts/calibration_confirm.py", []), ("paper/scripts/calibration_confirm.py", ["--tag", "clean_c"])]
NEEDS_REPRO = ["paper/scripts/numbers_24shot.py", "paper/scripts/jev_swap_analysis.py"]   # --repro-csv
STATIC = ["provenance/*.md", "paper/scripts/*.py", "paper/tools/checks.py", "scripts/*.py", "scripts/decision13/**/*",
          "scripts/jev/*.py", "quorum/*.py", "examples/*_descriptions.py", "results/**/manifest*.json*",
          "results/**/request_manifest*", "results/**/summary*.json", "results/**/analysis*.json",
          "results/**/selection*.json", "results/decision13/*.md", "results/decision13/*.json",
          "docs/phases/2.0/jev/train_pinned.json", "docs/phases/2.0/jev/*/PROTOCOL*.md",
          "docs/phases/2.0/bitext24/*.md", "docs/phases/2.0/bitext24/subset_a2.json", "docs/phases/2.0/poolsize/*.md",
          "LICENSE", "requirements*.txt"]
EXCLUDE = re.compile(r"(ledger|smoke_ledger|\.env|node_modules|__pycache__|\.pyc$|latency/|predictions\.csv|"
                     r"repro_|/cache/|commit_hash_map|paper/tools/checks\.py|pyproject\.toml)", re.I)
REPL = [(re.compile(r"DataAgent[\w-]*/[\w.-]+", re.I), "anonymous/reader-adapter"),
        (re.compile(r"Copyright \(c\) (\d{4}) .+"), r"Copyright (c) \1 Anonymous authors"),
        (re.compile(r"github\.com/A-baoYang/Quorum", re.I), "anonymous.repository"),
        (re.compile(r"huggingface\.co/(?:spaces/)?(?:DataAgent|A-baoYang)/[\w.-]+", re.I), "anonymous.hub.page"),
        (re.compile(r"QUORUM"), "R24"), (re.compile(r"Quorum"), "R24"), (re.compile(r"quorum"), "r24"),
        (re.compile(r"\bDataAgent\b", re.I), "anonymous"),
        (re.compile(r'("[\w]*(?:git|commit)[\w]*"\s*:\s*")[0-9a-f]{7,40}(")'), r"\1anonymised\2")]
STRIP = {"member_probs", "reader_probs", "knn_probs", "perm_reader_probs"}   # per-member vectors; no generator reads them
# (file pattern, scan pattern) pairs that are dataset content, not identifying text
ALLOW = [(re.compile(r"results/predictions/massive_"), r"\bDell\b")]   # MASSIVE utterances mention the brand
SCAN_EXTRA = [r"quorum", r"dataagent", r"a-?bao-?yang", r"rubyyang", r"/home/", r"/root/", r"\bsession_01"]


def scan_patterns():
    pats = list(SCAN_EXTRA)
    local = ROOT / "paper" / "tools" / "anonymity_terms.local"
    if local.exists():
        pats += [l.strip() for l in local.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return [re.compile(p, re.I) for p in pats]


def trace():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="supp_trace_"))           # outside paper/build, which the paper build wipes
    hook = tmp / "hook"; hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import sys, os\nlog = open(os.environ['SUPP_TRACE'], 'a')\n"
        "def h(ev, args):\n    if ev == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):\n"
        "        log.write(os.fsdecode(args[0]) + '\\n'); log.flush()\nsys.addaudithook(h)\n")
    raw = tmp / "trace.txt"; raw.write_text("")
    env = {**os.environ, "PYTHONPATH": f"{hook}:{os.environ.get('PYTHONPATH', '')}", "SUPP_TRACE": str(raw),
           "HF_DATASETS_OFFLINE": "1"}
    for g, args in GENERATORS:
        r = subprocess.run([PY, g, *args], cwd=ROOT, env=env, capture_output=True, text=True)
        print(("ok  " if r.returncode == 0 else "FAIL") + f" {g} {' '.join(args)}", flush=True)
        if r.returncode:
            print(r.stderr[-800:]); sys.exit(1)
    files = set()
    for l in raw.read_text().splitlines():
        p = Path(l) if os.path.isabs(l) else (ROOT / l)
        try:
            rel = p.resolve().relative_to(ROOT)
        except ValueError:
            continue
        if p.is_file() and not str(rel).startswith(("paper/generated", "paper/build", ".venv")):
            files.add(str(rel))
    TRACE.write_text(json.dumps(sorted(files), indent=1))
    print(f"traced {len(files)} repo files -> {TRACE.relative_to(ROOT)}")


def anonymise(text: str) -> str:
    for pat, rep in REPL:
        text = pat.sub(rep, text)
    return text


def build():
    files = set(json.loads(TRACE.read_text()))
    for g in STATIC:
        files |= {str(p.relative_to(ROOT)) for p in ROOT.glob(g) if p.is_file()}
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split())
    files = sorted(f for f in files if f in tracked and not EXCLUDE.search(f))
    pats, hits, total, per_dir = scan_patterns(), [], 0, {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    part = OUT.with_suffix(".partial")                           # written to disk as it goes (memory), renamed at the end
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            data = (ROOT / f).read_bytes(); arc = anonymise(f)
            if f.endswith(".gz"):
                txt = gzip.decompress(data).decode("utf-8", "replace")
                if f.endswith(".jsonl.gz") and any(f'"{k}"' in txt for k in STRIP):   # size limit (README)
                    txt = "".join(json.dumps({k: v for k, v in json.loads(l).items() if k not in STRIP}) + "\n"
                                  for l in txt.splitlines() if l.strip())
                txt = anonymise(txt)
                body = gzip.compress(txt.encode(), compresslevel=9, mtime=0); check = txt
            else:
                try:
                    check = anonymise(data.decode("utf-8")); body = check.encode()
                except UnicodeDecodeError:
                    check, body = "", data
            for p in pats:
                if any(fp.search(f) and p.pattern == ap for fp, ap in ALLOW):
                    continue
                if p.search(check) or p.search(arc):
                    hits.append(f"{f}: {p.pattern}")
            z.writestr("supplement/" + arc, body); total += len(body)
            d = "/".join(f.split("/")[:2]); per_dir[d] = per_dir.get(d, 0) + len(body)
        z.writestr("supplement/README.md", README)
    if hits:
        part.unlink(); print("\n".join(hits[:50])); sys.exit(f"anonymity scan: {len(hits)} hit(s); nothing written")
    size = part.stat().st_size
    if size > LIMIT:
        for d, b in sorted(per_dir.items(), key=lambda x: -x[1])[:12]:
            print(f"{b / 2**20:7.1f} MB  {d}")
        part.unlink(); sys.exit(f"archive {size / 2**20:.1f} MB exceeds 100 MB; nothing written")
    part.replace(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(files)} files, {size / 2**20:.1f} MB zipped "
          f"({total / 2**20:.1f} MB uncompressed); anonymity scan clean ({len(pats)} patterns)")


README = """# Supplementary material (anonymised for review)

- `provenance/`: the pre-registrations and amendments of every scored run, in the order described in Appendix B.
- `results/`: request manifests (sha256 of every request body, examples kept, retrieved indices), per-item
  predictions and probabilities, run summaries and the pre-declared analysis outputs.
- `paper/scripts/`: the generators that compute every number in the paper from the per-item files
  (`python paper/scripts/numbers_poolsize.py`, and so on).
- `scripts/`, `r24/`, `examples/`: the run code, the library and the label descriptions.

Not included: the public reproduction's instruction, definitions and predictions file, which carry no licence (the
scripts fetch them at run time; `numbers_24shot.py` and `jev_swap_analysis.py` need the predictions file via
`--repro-csv`), the reader adapter's weights, and item texts of datasets whose licence does not allow
redistribution (see `results/decision13/LICENSES.md`). To stay within the size limit, the per-item files omit the
per-member probability vectors (`member_probs` of the zero-shot jury; `reader_probs`, `knn_probs` and
`perm_reader_probs` of the 24-shot stack); the combined probabilities, predictions and every other field are
unchanged. No generator in `paper/scripts/` reads them except the combiner ablation of Appendix A
(`combiner_ablation.py`), which is reproducible from the full release. Identifiers that would reveal the authors were replaced;
sha256 values recorded for code files refer to the original files, which the de-anonymised release provides.
"""

if __name__ == "__main__":
    {"trace": trace, "build": build}[sys.argv[1]]()
