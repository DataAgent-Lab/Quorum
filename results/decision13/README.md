# Decision-model study (blocks E and C): results

Three decision models (pplx-decider v1.1, Clef 27B, Clef-Flash 9B) on 9 code/firmware decision sets (block E), a
PAWS contamination control, and 4 WorkflowEvals business workflows (block C), on one NVIDIA GB300. Protocol and every
amendment: `provenance/decision13_protocol.md`. Licences: `LICENSES.md`.

| file | content |
|---|---|
| `manifests/<set>_test_manifest.json` | frozen item sets: source pins, counts, leak checks, `items_sha256` (DBRD not published) |
| `predictions/<set>_<model>.jsonl.gz` | per-item outputs for the redistributable sets; pplx EquiBench is the amendment-6 file `_amend6` |
| `summary_<set>_<model>.json` | the runner's own summary per run, engine record de-identified (`device`, `served_model`) |
| `concurrency_checks.json` | amendment-2 concurrency check per model and window |
| `workflowevals/invoice_processing_<model>.jsonl.gz` | block C per-decision rows (ids and flags only) |
| `restricted_aggregates.json` | counts for the sets without a data licence (DBRD; 3 workflows) |
| `analysis_block_e.json`, `analysis_block_c.json` | outputs of `scripts/decision13/analyze_block_{e,c}.py` |

Notes on the summaries:
- `runner_sha256` is the hash of the runner as it ran. `scripts/decision13/run_block_e.py` is that runner with only its
  module docstring and one help string changed, so its hash differs.
- `engine.flags` says `--mem-fraction-static 0.75` for the block-E window; the servers ran 0.55 (amendment 6,
  engine-record correction). Summaries are kept as written.
- `summary_patchnet_pplx_v11.json` is the run as completed; `_lengthrule` is its amendment-4 re-score (idx 3386
  unanswered). `summary_equibench_pplx_v11.json` is the original 624-unanswered run; `_amend6` is the analysed one.

Reproduce: `python scripts/decision13/analyze_block_e.py && python scripts/decision13/analyze_block_c.py`.
