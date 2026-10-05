"""Instrument controls for paper/tools/checks.py (T7.0): every check must FLAG a planted violation and must NOT
flag a clean sample — an untested checker that says "PASS" proves nothing.

    .venv/bin/python -m pytest paper/tools/test_checks.py -q
Layout tests compile the vendored TACL template with tectonic (skipped if absent); bib tests need the network
(skipped offline).
"""
import shutil, subprocess, sys, urllib.request
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks  # noqa: E402

STYLE = Path(__file__).resolve().parents[1] / "style"
TECTONIC = shutil.which("tectonic") or str(Path.home() / ".local/bin/tectonic")


def _compile(tmp, tex_text, name):
    for f in ("tacl2021v1.sty", "acl_natbib.bst", "tacl2021.bib"):
        shutil.copy(STYLE / f, tmp / f)
    (tmp / f"{name}.tex").write_text(tex_text)
    subprocess.run([TECTONIC, "-X", "compile", f"{name}.tex"], cwd=tmp, capture_output=True, check=True)
    return tmp / f"{name}.pdf"


needs_tex = pytest.mark.skipif(not Path(TECTONIC).exists(), reason="tectonic not installed")


@needs_tex
def test_layout_clean_submission_passes(tmp_path):
    pdf = _compile(tmp_path, (STYLE / "tacl2021v1-template.tex").read_text(), "good")
    assert checks.check_layout(pdf) == []


@needs_tex
def test_layout_flags_camera_ready_and_author_metadata(tmp_path):
    src = (STYLE / "tacl2021v1-template.tex").read_text().replace(
        r"\usepackage[]{tacl2021v1}",
        r"\usepackage[acceptedWithA]{tacl2021v1}\AtBeginDocument{\hypersetup{pdfauthor={Planted Author}}}")
    errs = " | ".join(checks.check_layout(_compile(tmp_path, src, "bad")))
    assert "Author" in errs and "header" in errs and "line numbers" in errs


@needs_tex
def test_layout_flags_overlong_content_and_appendices(tmp_path, monkeypatch):
    pdf = _compile(tmp_path, (STYLE / "tacl2021v1-template.tex").read_text(), "good")
    head = checks.HEADER + "\n" + "\n".join(f"{i:03d}" for i in range(12)) + "\n"
    body = ["intro text"] * 11 + ["References", "refs", "Appendix A Replication details"] + ["x"] * 6 \
        + ["Appendix B Complementary results"] + ["y"] * 3
    fake = [head + "body\n" if i else head + "Title\n" for i in range(11)]
    fake += ["References\nrefs\n", "refs\n", "Appendix A Replication details\n"] + ["x\n"] * 5 \
        + ["Appendix B Complementary results\n", "y\n", "y\n", "y\n"]
    monkeypatch.setattr(checks, "pdf_pages_text", lambda _p: fake)
    errs = " | ".join(checks.check_layout(pdf))
    assert "content runs to page 11" in errs
    assert "Appendix A spans 6 pages" in errs and "Appendix B spans 4 pages" in errs


@needs_tex
def test_layout_allows_exact_limits(tmp_path, monkeypatch):
    pdf = _compile(tmp_path, (STYLE / "tacl2021v1-template.tex").read_text(), "good")
    head = checks.HEADER + "\n" + "\n".join(f"{i:03d}" for i in range(12)) + "\n"
    fake = [head + "body\n"] * 10 + ["References\nrefs\n", "Appendix A Replication details\n"] + ["x\n"] * 4 \
        + ["Appendix B Complementary results\n", "y\n", "y\n"]
    monkeypatch.setattr(checks, "pdf_pages_text", lambda _p: fake)
    assert checks.check_layout(pdf) == []


def test_anonymity_flags_planted_terms(tmp_path):
    f = tmp_path / "s.tex"
    f.write_text("We release Quorum at github.com/someorg/x; as in our previous work.\n"
                 "% Quorum in a comment is ignored\n")
    errs = " | ".join(checks.check_anonymity([f]))
    assert "public system name" in errs and "GitHub link" in errs and "first-person" in errs
    assert errs.count("Quorum") == 1, "comment lines must be ignored"


def test_anonymity_clean_text_passes(tmp_path):
    f = tmp_path / "s.tex"
    f.write_text("GeoJury pools three open models \\citep{yin2019}. Prior work by Yang et al. is cited.\n")
    assert checks.check_anonymity([f]) == []


def test_numbers_flags_literals_and_allows_macros(tmp_path):
    bad = tmp_path / "bad.tex"
    bad.write_text("Accuracy is 93.57\\% on 3,080 items.\n")
    good = tmp_path / "good.tex"
    good.write_text("Accuracy is \\fsCleanAcc\\% \\citep{milios2023} (Section~\\ref{sec:2}).\n"
                    "\\newcommand{\\x}{93.57}\nLayout \\vspace{0.5em} \\includegraphics[width=0.9\\linewidth]{f}\n"
                    "Reported by them as 92.11 on Banking77. % numbers-ok: cited external value\n")
    errs = checks.check_numbers([bad])
    assert len(errs) == 2 and "93.57" in errs[0] and "3,080" in errs[1]
    assert checks.check_numbers([good]) == []


def _online():
    try:
        urllib.request.urlopen("https://export.arxiv.org", timeout=10)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _online(), reason="offline")
def test_bib_resolves_real_and_flags_fake(tmp_path):
    bib = tmp_path / "r.bib"
    bib.write_text("@inproceedings{real,\n  title = {True Few-Shot Learning},\n  eprint = {2105.11447},\n}\n"
                   "@article{fake,\n  title = {Not a paper},\n  doi = {10.9999/definitely-not-a-real-doi-20261005},\n}\n")
    errs = checks.check_bib(bib, delay=0)
    assert len(errs) == 1 and errs[0].startswith("fake:")
