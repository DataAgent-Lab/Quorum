#!/usr/bin/env bash
# Build the anonymous TMLR submission PDF into paper/build/ (style files copied, never edited), then run the checks.
#   bash paper/build.sh            build + checks
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$(command -v tectonic || echo "$HOME/.local/bin/tectonic")}"
PY="${PY:-../.venv/bin/python}"
rm -rf build && mkdir -p build
"$PY" tools/make_bib.py
cp main.tex main.bib build/
cp -r sections generated build/
cp style-tmlr/{tmlr.sty,tmlr.bst,fancyhdr.sty,math_commands.tex} build/
(cd build && "$TECTONIC" -X compile --keep-logs main.tex >/dev/null)
echo "built build/main.pdf ($("$PY" -c "from pypdf import PdfReader;print(len(PdfReader('build/main.pdf').pages))" 2>/dev/null || echo '?') pages)"
grep -E "Warning.*(undefined|multiply)" build/main.log || true
"$PY" tools/checks.py layout-tmlr build/main.pdf
"$PY" tools/checks.py numbers sections
"$PY" tools/checks.py anonymity build/main.pdf sections main.tex main.bib
