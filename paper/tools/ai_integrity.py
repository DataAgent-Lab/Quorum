#!/usr/bin/env python3
"""Pre-submission AI-integrity checks: the artifacts that actually get papers desk-rejected under AI-use policies.

    python paper/tools/ai_integrity.py refs       paper/refs.bib           # every entry's title + authors match its record
    python paper/tools/ai_integrity.py cites      paper/refs.bib paper/main.tex paper/sections
    python paper/tools/ai_integrity.py artifacts  paper/main.tex paper/sections [paper/build/main.pdf]
    python paper/tools/ai_integrity.py hidden     paper/build/main.pdf     # invisible / white / tiny / off-page text
    python paper/tools/ai_integrity.py disclosure paper/main.tex paper/sections

Complements paper/tools/checks.py (layout, anonymity, numbers, and `bib`, which only checks that an identifier
RESOLVES). The cases venues have acted on are: references whose identifier exists but whose title or authors are
wrong or invented ("hallucitations"; ACL 2026 desk-rejected accepted papers for them), hidden instructions aimed at
LLM reviewers (forbidden by the ACL publication-ethics policy), leftover chatbot boilerplate, and missing AI-use
disclosure. Nothing here estimates whether prose "sounds AI-written": detector scores are not evidence, and this
tool does not produce one. The `artifacts` report lists a few over-used words as INFO only; it never fails on them.

Each command prints FAIL / WARN / INFO lines and exits non-zero on any FAIL. `hidden` needs PyMuPDF (AGPL-3.0; a
local dev-tool dependency in paper/requirements.txt, not a dependency of the quorum package).
"""
from __future__ import annotations
import difflib, json, re, sys, time, unicodedata, urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks  # noqa: E402  (shared bib parser, HTTP helper, pdftotext wrapper)

# ------------------------------------------------------------------ helpers
_LATEX_ACCENT = re.compile(r"\\[`'^\"~=.uvHtcdbrk]\s*\{?(\w)\}?")


def norm_text(s: str) -> str:
    """Lowercase ASCII words: LaTeX accents/commands/braces dropped, accents stripped, punctuation -> space."""
    s = _LATEX_ACCENT.sub(r"\1", s or "")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s).replace("{", "").replace("}", "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def surnames(author_field: str) -> list[str]:
    """BibTeX author list -> normalised surnames ('Last, First' or 'First Last')."""
    out = []
    for a in re.split(r"\s+and\s+", (author_field or "").strip()):
        a = a.strip()
        if not a or a.lower() == "others":
            continue
        last = a.split(",")[0] if "," in a else a.split()[-1]
        out.append(norm_text(last))
    return [s for s in out if s]


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def _json(url):
    st, txt = checks._get(url)
    return json.loads(txt)


# ------------------------------------------------------------------ refs: identifier -> authoritative title/authors
def record_for(body: str) -> tuple[str, str | None, list[str] | None]:
    """(source, title, surnames) from the entry's own identifier; title None if it cannot be resolved."""
    body = body + "\n"
    doi, eprint, url = checks.FIELD(body, "doi"), checks.FIELD(body, "eprint"), checks.FIELD(body, "url")
    anth = re.search(r"aclanthology\.org/([\w.-]+?)/?$", (url or "").strip())
    if anth:
        st, txt = checks._get(f"https://aclanthology.org/{anth.group(1)}.bib")
        t = txt + "\n"
        return f"ACL Anthology {anth.group(1)}", checks.FIELD(t, "title"), surnames(checks.FIELD(t, "author") or "")
    if doi:
        m = _json("https://api.crossref.org/works/" + urllib.parse.quote(doi.strip()))["message"]
        fam = [norm_text(a.get("family") or a.get("name") or "") for a in m.get("author", [])]
        return f"Crossref {doi.strip()}", (m.get("title") or [None])[0], [f for f in fam if f]
    arx = eprint or (re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url or "") or [None, None])[1]
    if arx:
        st, txt = checks._get(f"https://export.arxiv.org/api/query?id_list={arx.strip()}")
        if "<entry>" not in txt:
            return f"arXiv {arx}", None, None
        e = txt.split("<entry>", 1)[1]
        title = re.search(r"<title>(.*?)</title>", e, re.S)
        names = [n.strip().split()[-1] for n in re.findall(r"<name>(.*?)</name>", e)]
        return f"arXiv {arx}", " ".join(title.group(1).split()) if title else None, [norm_text(n) for n in names]
    return ("web" if url else "none"), None, None


def check_refs(bib: Path, delay: float = 0.4, min_title: float = 0.90) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    for typ, key, body in checks.ENTRY.findall(bib.read_text(errors="ignore")):
        mine_t, mine_a = checks.FIELD(body + "\n", "title") or "", surnames(checks.FIELD(body + "\n", "author") or "")
        try:
            src, title, theirs = record_for(body)
        except Exception as e:  # noqa: BLE001 — any lookup failure is reported, never silently passed
            fails.append(f"{key}: lookup failed ({type(e).__name__}: {e})"); time.sleep(delay); continue
        if title is None:
            (warns if src == "web" else fails).append(
                f"{key}: no scholarly record ({src}) — verify title/authors by hand" if src == "web"
                else f"{key}: identifier does not resolve ({src})")
        else:
            sim = similarity(mine_t, title)
            if sim < min_title:
                fails.append(f"{key}: title mismatch vs {src} (similarity {sim:.2f}): bib={mine_t!r} record={title!r}")
            if theirs:
                if mine_a and mine_a[0] not in theirs:
                    fails.append(f"{key}: first author {mine_a[0]!r} not in {src} authors {theirs[:6]}")
                extra = [a for a in mine_a if a not in theirs]
                if extra and not (mine_a and mine_a[0] in extra):
                    warns.append(f"{key}: authors not in {src}: {extra}")
                if mine_a and len(mine_a) != len(theirs):
                    warns.append(f"{key}: {len(mine_a)} authors in bib vs {len(theirs)} in {src}")
        time.sleep(delay)
    return fails, warns


# ------------------------------------------------------------------ cites: every cited key exists
COMMENT = re.compile(r"(?<!\\)%.*")
CITE = re.compile(r"\\(?:no)?cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")


def tex_files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(p.rglob("*.tex")) if p.is_dir() else [p]
    return out


def check_cites(bib: Path, paths: list[Path]) -> tuple[list[str], list[str]]:
    keys = {k for _, k, _ in checks.ENTRY.findall(bib.read_text(errors="ignore"))}
    used = {}
    for f in tex_files(paths):
        for ln, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            for grp in CITE.findall(COMMENT.sub("", line)):
                for k in (x.strip() for x in grp.split(",")):
                    if k and k != "*":
                        used.setdefault(k, f"{f}:{ln}")
    fails = [f"cited key {k!r} not in {bib.name} ({where})" for k, where in sorted(used.items()) if k not in keys]
    infos = [f"{len(keys - set(used))} bib entries are never cited"] if keys - set(used) else []
    return fails, infos


# ------------------------------------------------------------------ artifacts: chatbot residue + reviewer prompts
HARD = [
    (r"\bas an ai(?: language)? model\b", "chatbot self-reference"),
    (r"\bas a large language model\b", "chatbot self-reference"),
    (r"\bi (?:cannot|can't|am unable to) (?:fulfil+|assist with|provide) (?:this|that)\b", "chatbot refusal"),
    (r"\bcertainly[,!]? here (?:is|are)\b", "chatbot preamble"),
    (r"\bhere (?:is|are) (?:a|an|the|your) (?:revised|rewritten|improved|polished|updated) (?:version|draft|paragraph|text)\b",
     "chatbot preamble"),
    (r"\bi hope this helps\b", "chatbot closing"),
    (r"\b(?:feel free to|let me know if you(?:'d| would)? like)\b", "chatbot closing"),
    (r"\bregenerate response\b", "chat UI residue"),
    (r"\bas of my (?:last )?(?:knowledge (?:cutoff|update)|training data)\b", "chatbot knowledge-cutoff phrase"),
    (r"\[(?:insert|citation needed|add (?:citation|reference)|your [a-z ]+)[^\]]*\]", "unfilled placeholder"),
    (r"\blorem ipsum\b", "placeholder text"),
]
INJECTION = [
    (r"\bignore (?:all |any )?(?:previous|prior|above|earlier) (?:instructions|prompts)\b", "instruction to an LLM"),
    (r"\b(?:give|write|provide) (?:a |only )?(?:positive|favou?rable|glowing) review\b", "reviewer manipulation"),
    (r"\bdo not (?:highlight|mention|point out) (?:any )?(?:negatives|weaknesses|limitations)\b", "reviewer manipulation"),
    (r"\b(?:if you are|as) an? (?:llm|ai|language model)(?: reviewer| assistant)?\b", "addressed to an LLM"),
    (r"\b(?:llm|ai) reviewers?\b", "addressed to LLM reviewers"),
]
SOFT = ["delve", "delves", "tapestry", "intricate", "underscores", "showcasing", "pivotal", "meticulous",
        "commendable", "noteworthy", "in the realm of", "it is worth noting", "seamlessly"]


def _scan(text: str, where: str, patterns, tag: str) -> list[str]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        for rx, why in patterns:
            m = re.search(rx, low)
            if m:
                note = (" (inside a LaTeX comment — still ships in arXiv source)"
                        if COMMENT.search(line[:m.start()]) else "")
                out.append(f"{tag} {why}{note}: {where}:{i}: {line.strip()[:120]!r}")
    return out


def check_artifacts(paths: list[Path]) -> tuple[list[str], list[str]]:
    fails, infos, words, soft = [], [], 0, {}
    for p in paths:
        if p.suffix.lower() == ".pdf":
            sources = [(f"{p}#page{i + 1}", t) for i, t in enumerate(checks.pdf_pages_text(p))]
        else:
            sources = [(str(f), f.read_text(errors="ignore")) for f in tex_files([p])]
        for where, text in sources:
            fails += _scan(text, where, HARD, "artifact") + _scan(text, where, INJECTION, "reviewer-prompt")
            low = text.lower(); words += len(low.split())
            for w in SOFT:
                n = len(re.findall(rf"\b{re.escape(w)}\b", low))
                if n:
                    soft[w] = soft.get(w, 0) + n
    if soft:
        infos.append("over-used-word counts (INFO only, never a verdict): "
                     + ", ".join(f"{w}={n}" for w, n in sorted(soft.items(), key=lambda x: -x[1]))
                     + f" over {words} words")
    return fails, infos


# ------------------------------------------------------------------ hidden: text a reader cannot see
def _near_white(rgb, thr=0.93):
    return rgb is not None and all(c >= thr for c in rgb)


def _rgb(c):
    if c is None:
        return None
    if isinstance(c, int):
        return ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255
    c = tuple(c)
    return (c[0],) * 3 if len(c) == 1 else c[:3] if len(c) >= 3 else None


def check_hidden(pdf: Path, min_size: float = 4.0) -> tuple[list[str], list[str]]:
    import pymupdf
    fails, infos = [], []
    doc = pymupdf.open(str(pdf))
    for pno, page in enumerate(doc, 1):
        box = page.rect
        dark = [d["rect"] for d in page.get_drawings()
                if d.get("fill") is not None and not _near_white(_rgb(d["fill"]))]
        imgs = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
        for t in page.get_texttrace():
            text = "".join(chr(ch[0]) for ch in t["chars"]).strip()
            if not text:
                continue
            r = pymupdf.Rect(t["bbox"]); c = r.tl + (r.br - r.tl) * 0.5
            shown = f"page {pno}: {text[:80]!r}"
            if t.get("type") == 3:
                fails.append(f"invisible render mode (Tr 3) {shown}")
            elif t.get("opacity", 1.0) < 0.1:
                fails.append(f"near-transparent text (opacity {t['opacity']:.2f}) {shown}")
            elif _near_white(_rgb(t.get("color"))) and not any(c in d for d in dark + imgs):
                fails.append(f"white/near-white text not over a dark fill or image {shown}")
            if t.get("size", 99) < min_size:
                fails.append(f"tiny text ({t['size']:.1f} pt < {min_size}) {shown}")
            if not r.intersects(box):
                fails.append(f"text outside the page box {shown}")
        fails += _scan(page.get_text(), f"{pdf}#page{pno}", INJECTION, "reviewer-prompt")
    infos.append(f"{len(doc)} pages scanned")
    return fails, infos


# ------------------------------------------------------------------ disclosure: AI use is declared
ACK = re.compile(r"\\(?:section|subsection|paragraph)\*?\{\s*(acknowledge?ments?|use of ai[^}]*|ai (?:use|assistan)[^}]*)\s*\}",
                 re.I)
AI_WORDS = re.compile(r"\b(?:ai|llms?|language models?|assistants?|chatgpt|claude|gpt-?\d*|gemini|copilot)\b", re.I)


def check_disclosure(paths: list[Path]) -> tuple[list[str], list[str]]:
    text = "\n".join(f.read_text(errors="ignore") for f in tex_files(paths))
    hits = list(ACK.finditer(text))
    if not hits:
        return ["no Acknowledgements / 'Use of AI assistants' section found — ACL policy: disclose AI use there"], []
    for h in hits:
        nxt = re.search(r"\\(?:section|bibliography|end\{document\})", text[h.end():])
        body = text[h.end(): h.end() + (nxt.start() if nxt else 4000)]
        if AI_WORDS.search(body):
            return [], [f"AI-use statement found under '{h.group(1)}'"]
    return [f"section(s) {[h.group(1) for h in hits]} found but none mentions AI / LLM use"], []


# ------------------------------------------------------------------ cli
def main(argv):
    cmd, args = argv[1], [Path(a) for a in argv[2:]]
    fails, notes = {"refs": lambda: check_refs(args[0]), "cites": lambda: check_cites(args[0], args[1:]),
                    "artifacts": lambda: check_artifacts(args), "hidden": lambda: check_hidden(args[0]),
                    "disclosure": lambda: check_disclosure(args)}[cmd]()
    for f in fails:
        print("FAIL", f)
    for n in notes:
        print("WARN" if cmd in ("refs",) else "INFO", n)
    print(f"{cmd}: {'PASS' if not fails else f'{len(fails)} violation(s)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
