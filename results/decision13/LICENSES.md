# Decision-model study: licences and what is published per set

Protocol: `provenance/decision13_protocol.md`. Published per-item records hold only ids, labels and model outputs
(`idx, gold, group?, pred, probs, unanswered, correct, input_tokens, status?`); no item text is published for any set.
Item bodies are rebuilt locally from the pinned sources with `scripts/decision13/builders/` and checked against the
manifests' `items_sha256`.

| set | source (pinned) | licence as recorded at build time | published here |
|---|---|---|---|
| PatchNet | Zenodo 3965234 `test.pkl` + Linux git history | dataset cc-by-4.0; the item text is Linux kernel commits and diffs (GPL-2.0) | manifest, per-item outputs by `idx` (position in `test.pkl`), summaries; no text |
| TOSEM-DBRD | authors' release (no LICENSE file) | none stated | **counts only**: per-pair b/c, n, accuracies, `items_sha256` (`restricted_aggregates.json`); no manifest, no per-item data, no builder |
| EquiBench | HF `anjiangwei/EquiBench-Datasets` @6202448 | HF card MIT; GitHub repo LICENSE Apache-2.0 (conflict recorded; both permissive) | manifest, per-item outputs, summaries |
| POJ-104 | HF `google/code_x_glue_cc_clone_detection_poj104` @701869d | C-UDA | manifest, per-item outputs, summaries |
| PrimeVul v0 (pair, per function) | authors' release `primevul_test_paired.jsonl` (sha256 `54a16a5a…`) | MIT (DLVulDet/PrimeVul LICENSE @6f54687) | manifests, per-item outputs, summaries |
| SV-TrustEval-C 1.0 | HF `LLMs4CodeSecurity/SV-TrustEval-C-1.0` @a9a7125 | MIT (card + authors' repo LICENSE @5bb1585) | manifest, per-item outputs, summaries |
| SecVulEval | HF `arag0rn/SecVulEval` @bde3e7c | **flag**: MIT in the card metadata only; the card's licence line is an unfilled placeholder and there is no LICENSE file | manifest, per-item outputs, summaries |
| Juliet C/C++ 1.3 | NIST SARD suite 112 zip (sha256 `ada9d7e1…`) | US public domain (17 USC 105); CC0 1.0 for foreign rights | manifest, per-item outputs, summaries |
| PAWS (control) | HF `google-research-datasets/paws` @161ece9 `labeled_final` test | card `other`: free use for any purpose, acknowledgement of Google appreciated | manifest, per-item outputs, summaries |
| WorkflowEvals invoice_processing | HF `typesafe/evalsafe-invoice-processing` @6beeb2d; harness @0ac3b8a (Apache-2.0) | Apache-2.0 per the dataset card (not re-checked at transcription) | per-decision rows: case id, policy id, label set, status, metric flags (`workflowevals/`); no case text |
| WorkflowEvals customer_service, security_incidents, agent_trace_observability | HF `typesafe/evalsafe-*` @ revisions in the protocol, amendment 8 | none stated | **counts only** (`restricted_aggregates.json`), with the harness's dataset hash |

Contamination and exposure flags are in protocol §9. PAWS is a contamination control and never enters the decision.
