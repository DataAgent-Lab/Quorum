#!/usr/bin/env python3
"""Compliance checks for the TACL submission (Phase 2.0, AC-1/2/3/6).

    python paper/tools/checks.py layout-tmlr paper/build/main.pdf   (current venue: TMLR)
    python paper/tools/checks.py layout    paper/build/main.pdf      (TACL, kept for the record)
    python paper/tools/checks.py anonymity paper/build/main.pdf paper/sections paper/main.tex paper/main.bib
    python paper/tools/checks.py numbers   paper/sections
    python paper/tools/checks.py bib       paper/refs.bib

Each command prints its findings and exits non-zero on any violation. Rules are TACL's (transacl.org submission
page, the 2024 appendix announcement, and the formatting PDF vendored in paper/style/).

Anonymity terms: generic patterns live here; personal/identifying terms (names, employer, private domains) are read
from paper/tools/anonymity_terms.local — gitignored, one regex per line — so the list itself never reaches the
public repository.
"""
from __future__ import annotations
import hashlib, json, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "paper" / "style"
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
A4 = (595.28, 841.89)
LETTER = (612.0, 792.0)
MAX_CONTENT_PAGES, MAX_APPX_A, MAX_APPX_B = 10, 5, 3
HEADER = "Confidential TACL submission. DO NOT DISTRIBUTE."
TMLR_HEADER = "Under review as submission to TMLR"
STYLE_TMLR = ROOT / "paper" / "style-tmlr"

GENERIC_ANON = [
    (r"huggingface\.co/(?!datasets/)[\w.-]+", "Hugging Face link (identifies the uploader)"),
    (r"github\.com/[\w.-]+", "GitHub link"),
    (r"\bDataAgent\b", "organisation name"),
    (r"\bQuorum\b", "public system name (use the review-version name)"),
    (r"\b(?:our|my) (?:previous|prior|earlier) (?:work|paper|study)\b", "first-person self-reference"),
    (r"\bwe (?:previously|earlier) (?:showed|proposed|introduced|reported)\b", "first-person self-reference"),
    (r"\bin our (?:previous|prior) (?:work|paper)\b", "first-person self-reference"),
    (r"anonymous\.4open\.science|osf\.io|drive\.google|dropbox", "link to supplementary material (forbidden at TACL)"),
]


def pdf_pages_text(pdf: Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    pages = out.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    norm = []
    for p in pages:
        for k, v in LIGATURES.items():
            p = p.replace(k, v)
        norm.append(p)
    return norm


def _strip_line_numbers(line: str) -> str:
    """TACL submission pages carry 3-digit line numbers in both margins."""
    return re.sub(r"(^\s*\d{3}\s+|\s+\d{3}\s*$)", " ", line).strip()


def _heading_page(pages, pattern):
    rx = re.compile(pattern)
    for i, p in enumerate(pages):
        for line in p.splitlines():
            if rx.fullmatch(_strip_line_numbers(line)):
                return i
    return None


# ------------------------------------------------------------------ layout (AC-1)
def check_layout_tmlr(pdf: Path) -> list[str]:
    """TMLR submission: official style untouched, US Letter, the review header and 'Anonymous authors' on page 1,
    Author metadata empty, fonts embedded. No page limit (length should be justified; appendices optional to read)."""
    errs = []
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    for i, pg in enumerate(r.pages):
        w, h = float(pg.mediabox.width), float(pg.mediabox.height)
        if abs(w - LETTER[0]) > 1.5 or abs(h - LETTER[1]) > 1.5:
            errs.append(f"page {i + 1}: size {w:.1f}x{h:.1f} pt is not US Letter")
    if str((r.metadata or {}).get("/Author", "") or "").strip():
        errs.append("PDF metadata /Author is set — remove it")
    errs += [e for e in _fonts_missing(r)]
    pages = pdf_pages_text(pdf)
    first = re.sub(r"\s+", " ", pages[0]) if pages else ""
    if TMLR_HEADER not in first:
        errs.append(f"page 1 lacks the TMLR review header {TMLR_HEADER!r} (use the style's submission mode)")
    if "Anonymous authors" not in first:
        errs.append("page 1 does not show 'Anonymous authors' (double-blind submission mode)")
    for line in (STYLE_TMLR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name.endswith(".sty") and hashlib.sha256((STYLE_TMLR / name).read_bytes()).hexdigest() != digest:
            errs.append(f"official TMLR style file {name} was modified")
    return errs


def _fonts_missing(r) -> list[str]:
    missing = set()
    for pg in r.pages:
        res = pg.get("/Resources") or {}
        fonts = (res.get("/Font") or {}) if hasattr(res, "get") else {}
        for _, ref in (fonts.items() if hasattr(fonts, "items") else []):
            f = ref.get_object()
            if f.get("/Subtype") == "/Type3":
                continue
            desc = f.get("/FontDescriptor")
            if f.get("/Subtype") == "/Type0":
                desc = f["/DescendantFonts"][0].get_object().get("/FontDescriptor")
            d = desc.get_object() if desc is not None else None
            if d is None or not any(k in d for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                missing.add(str(f.get("/BaseFont")))
    return [f"font not embedded: {m}" for m in sorted(missing)]


def check_layout(pdf: Path) -> list[str]:
    errs = []
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    for i, pg in enumerate(r.pages):
        w, h = float(pg.mediabox.width), float(pg.mediabox.height)
        if abs(w - A4[0]) > 1.5 or abs(h - A4[1]) > 1.5:
            errs.append(f"page {i + 1}: size {w:.1f}x{h:.1f} pt is not A4")
    meta = r.metadata or {}
    for key in ("/Author", "/Creator", "/Producer", "/Title", "/Subject", "/Keywords"):
        val = str(meta.get(key, "") or "")
        if key == "/Author" and val.strip():
            errs.append(f"PDF metadata {key} is set ({val!r}) — remove it (Document Properties)")
    fonts_missing = set()
    for pg in r.pages:
        res = pg.get("/Resources") or {}
        fonts = (res.get("/Font") or {}) if hasattr(res, "get") else {}
        for name, ref in (fonts.items() if hasattr(fonts, "items") else []):
            f = ref.get_object()
            desc = f.get("/FontDescriptor")
            if f.get("/Subtype") == "/Type0":
                desc = f["/DescendantFonts"][0].get_object().get("/FontDescriptor")
            if f.get("/Subtype") == "/Type3":
                continue
            d = desc.get_object() if desc is not None else None
            if d is None or not any(k in d for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                fonts_missing.add(str(f.get("/BaseFont")))
    for fname in sorted(fonts_missing):
        errs.append(f"font not embedded: {fname}")
    pages = pdf_pages_text(pdf)
    if not pages or HEADER not in re.sub(r"\s+", " ", pages[0]):
        errs.append(f"page 1 lacks the submission header {HEADER!r} (use the style's submission mode)")
    if not pages or len(re.findall(r"(?m)^\s*0\d\d\b", pages[0])) < 10:
        errs.append("page 1 has no margin line numbers (submission mode required)")
    ref = _heading_page(pages, r"References")
    if ref is None:
        errs.append("no 'References' heading found")
    else:
        content_pages = ref + (1 if _has_body_before(pages[ref], "References", None) else 0)
        if content_pages > MAX_CONTENT_PAGES:
            errs.append(f"content runs to page {content_pages} (> {MAX_CONTENT_PAGES})")
    a = _heading_page(pages, r"(?:Appendix\s+)?A\s+.*Replication.*|Appendix A\b.*")
    b = _heading_page(pages, r"(?:Appendix\s+)?B\s+.*(?:Complementary|Additional).*|Appendix B\b.*")
    def span(start, nxt, heading_rx):
        """Pages an appendix occupies: start..end inclusive, where end is the page before the next appendix's
        heading when that heading opens its page, otherwise the heading's own page (shared page counts)."""
        if nxt is None:
            return len(pages) - start
        shared = _has_body_before(pages[nxt], None, heading_rx)
        return nxt - start + (1 if shared else 0)
    if a is not None:
        n_a = span(a, b, r"(?:Appendix\s+)?B\b.*")
        if n_a > MAX_APPX_A:
            errs.append(f"Appendix A spans {n_a} pages (> {MAX_APPX_A})")
    if b is not None:
        n_b = span(b, None, None)
        if n_b > MAX_APPX_B:
            errs.append(f"Appendix B spans {n_b} pages (> {MAX_APPX_B})")
    errs += _style_untouched()
    return errs


def _has_body_before(page: str, heading: str | None, heading_rx: str | None) -> bool:
    """True when the page carries body text above the given heading (i.e. the page is shared)."""
    rx = re.compile(heading_rx) if heading_rx else None
    for line in page.splitlines():
        s = _strip_line_numbers(line)
        if (heading is not None and s == heading) or (rx is not None and rx.fullmatch(s)):
            return False
        if s and s != HEADER and not re.fullmatch(r"\d+", s):
            return True
    return False


def _style_untouched() -> list[str]:
    errs, sums = [], STYLE / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name.endswith(".sty") and hashlib.sha256((STYLE / name).read_bytes()).hexdigest() != digest:
            errs.append(f"official style file {name} was modified")
    return errs


# ------------------------------------------------------------------ anonymity (AC-2)
def anonymity_patterns():
    pats = list(GENERIC_ANON)
    local = ROOT / "paper" / "tools" / "anonymity_terms.local"
    if local.exists():
        for line in local.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pats.append((line, "identifying term (local list)"))
    return [(re.compile(p, re.I), why) for p, why in pats]


LINK_KINDS = ("GitHub link", "Hugging Face link (identifies the uploader)")


def check_anonymity(paths: list[Path]) -> list[str]:
    """A .bib among the paths is not scanned: GitHub / Hugging Face links that occur in it are cited third-party
    resources and are allowed in the PDF (the reference list). Personal terms are still flagged everywhere."""
    errs, pats = [], anonymity_patterns()
    cited = set()
    for b in [p for p in paths if p.suffix == ".bib"]:
        cited |= {m.group(0) for rx, why in pats if why in LINK_KINDS for m in rx.finditer(b.read_text(errors="ignore"))}
    paths = [p for p in paths if p.suffix != ".bib"]
    for p in paths:
        files = sorted(p.rglob("*.tex")) if p.is_dir() else [p]
        for f in files:
            if f.suffix == ".pdf":
                texts = [(f"{f.name} p{i + 1}", t) for i, t in enumerate(pdf_pages_text(f))]
            else:
                src = "\n".join(l for l in f.read_text(errors="ignore").splitlines() if not l.lstrip().startswith("%"))
                texts = [(str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f), src)]
            for where, text in texts:
                for rx, why in pats:
                    for m in rx.finditer(text):
                        if why in LINK_KINDS and m.group(0) in cited:
                            continue
                        errs.append(f"{where}: {why}: {m.group(0)!r}")
    return errs


# ------------------------------------------------------------------ numbers (AC-3)
NUM = re.compile(r"(?<![\w\\{])(\d+\.\d+|\d{1,3}(?:,\d{3})+|\d+\s*\\%)(?![\w}])")
ALLOW_CTX = re.compile(r"\\(?:cite\w*|ref|eqref|label|url|href|includegraphics|input|vspace|hspace|setlength)\{[^}]*\}"
                       r"|\[[\d.]+\s*(?:pt|cm|mm|in|em|ex)\]|[\d.]+\s*(?:pt|cm|mm|in|em|ex|\\linewidth|\\textwidth)")


def check_numbers(paths: list[Path]) -> list[str]:
    errs = []
    for p in paths:
        for f in (sorted(p.rglob("*.tex")) if p.is_dir() else [p]):
            for n, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                code = line.split("%", 1)[0] if not re.search(r"\\%", line) else re.sub(r"(?<!\\)%.*", "", line)
                if "numbers-ok" in line or code.lstrip().startswith(("\\newcommand", "\\renewcommand", "\\def")):
                    continue
                for m in NUM.finditer(ALLOW_CTX.sub(" ", code)):
                    errs.append(f"{f.name}:{n}: literal number {m.group(0)!r} — use a generated macro "
                                f"(or mark the line '% numbers-ok' with a citation)")
    return errs


# ------------------------------------------------------------------ bibliography (AC-6)
ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", re.S)
FIELD = lambda body, name: (re.search(rf"\b{name}\s*=\s*[{{\"](.+?)[}}\"]\s*,?\s*\n", body, re.S | re.I) or [None, None])[1]


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "tacl-bib-check/1.0 (mailto:noreply@example.org)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read(400_000).decode("utf-8", "ignore")


def resolve(body: str) -> tuple[bool, str]:
    body = body + "\n"                       # the entry regex consumes the newline after the last field
    doi, eprint, url = FIELD(body, "doi"), FIELD(body, "eprint"), FIELD(body, "url")
    try:
        if doi:
            st, txt = _get("https://doi.org/api/handles/" + urllib.parse.quote(doi.strip()))
            return (json.loads(txt).get("responseCode") == 1, f"doi {doi}")
        arx = eprint or (re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url or "") or [None, None])[1]
        if arx:
            st, txt = _get(f"https://export.arxiv.org/api/query?id_list={arx.strip()}")
            return ("<entry>" in txt and "Error" not in txt.split("<entry>")[1][:400], f"arXiv {arx}")
        if url:
            st, _ = _get(url.strip())
            return (200 <= st < 400, f"url {url}")
        return (False, "no doi / eprint / url")
    except Exception as e:  # noqa: BLE001
        return (False, f"{type(e).__name__}: {e}")


def check_bib(bib: Path, delay: float = 0.4) -> list[str]:
    errs = []
    for typ, key, body in ENTRY.findall(bib.read_text(errors="ignore")):
        ok, how = resolve(body)
        if not ok:
            errs.append(f"{key}: unresolved ({how})")
        time.sleep(delay)
    return errs


def main(argv):
    cmd, args = argv[1], [Path(a) for a in argv[2:]]
    errs = {"layout": lambda: check_layout(args[0]), "layout-tmlr": lambda: check_layout_tmlr(args[0]), "anonymity": lambda: check_anonymity(args),
            "numbers": lambda: check_numbers(args), "bib": lambda: check_bib(args[0])}[cmd]()
    for e in errs:
        print("FAIL", e)
    print(f"{cmd}: {'PASS' if not errs else f'{len(errs)} violation(s)'}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
