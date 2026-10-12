#!/usr/bin/env bash
# Block C (provenance/decision13_protocol.md amendment 8): run the WorkflowEvals harness (typesafe-ai/WorkflowEvals
# @0ac3b8a, its own code, unedited) in a pinned Python 3.13 container against a local /v1/systemone server.
# Arguments are passed to the harness's `run.py`; the study changed only --base-url and --workers.
#   WE_DIR=/path/to/WorkflowEvals HF_DIR=/path/to/hf-cache run_workflowevals.sh <workflow> --base-url http://HOST:PORT --workers 16 ...
set -euo pipefail
: "${WE_DIR:?clone of typesafe-ai/WorkflowEvals at 0ac3b8a}" "${HF_DIR:?HF cache directory}"
TOOLS=${TOOLS_DIR:-$PWD/.we_tools}; mkdir -p "$TOOLS" "$HF_DIR"
PY=python@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f    # python:3.13-slim, arm64
exec timeout 43200 docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf \
  -e TYPESAFE_API_KEY=unused-local-server -e PYTHONPATH=/tools -v "$WE_DIR":/we -v "$HF_DIR":/hf -v "$TOOLS":/tools -w /we \
  "$PY" bash -c '[ -x /tools/bin/uv ] || pip install -q --target /tools uv==0.12.24; /tools/bin/uv sync --locked -q && /tools/bin/uv run python run.py "$@"' we "$@"
