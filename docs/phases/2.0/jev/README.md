# Jev (TypeSafe `jev-1.13.0`) runs of Phase 2.0 — index

Every Jev call made for the paper lives here. All runs were **pre-registered** (protocol and analysis script
committed and pushed before any test-set call), use TypeSafe's own API with the model version pinned and asserted on
every response, and never store the API key (it is read from the gitignored `service/.env`). Files from the
third-party reproduction (github.com/simonmesmith/jev-banking77-experiment, unlicensed) are fetched at run time and
never committed; instead, every request is recorded by the sha256 of the exact bytes sent.

| folder | what | protocol | result (claims ledger) |
|---|---|---|---|
| [`24shot/`](24shot/) | Arm A = the reproduction's own 24-shot requests (replication); Arm B = same requests with our clean run's 24 retrieved examples | `provenance/jev_retrieval_swap_protocol.md` + `..._addendum.md` | F15: equal-retrieval accuracy tie; Jev +1.07 pt from our retrieval; exact replication of 92.40%; our calibration better |
| [`zeroshot/`](zeroshot/) | No examples; label text = class names / our descriptions on Banking77, MTOP (+ addendum 1: CLINC150, SNIPS, Bitext) and the reproduction's definitions (Banking77) | [`zeroshot/PROTOCOL.md`](zeroshot/PROTOCOL.md) + [`PROTOCOL_ADDENDUM_clean_datasets.md`](zeroshot/PROTOCOL_ADDENDUM_clean_datasets.md) | J1–J3: Jev far ahead zero-shot on every dataset (e.g. CLINC150 88–91% vs 62–67%); calibration favours Jev; log loss vs Jev is a rounding artifact (J2) |

Files per run: per-item results (`*.jsonl.gz`: choice, confidence, all class probabilities, served model, usage,
attempts), `ledger.jsonl.gz` (every attempt, tokens, cost), `request_manifest.jsonl.gz` (sha256 + size of each
request; for 24shot Arm B also the retrieved train ids), and for the zero-shot run the full raw responses
(`raw_*.jsonl.gz`). `train_pinned.json` = Banking77 train texts/labels at the pinned dataset revision (Arm B input).

Runners: `scripts/jev/run_typesafe.py` (24shot), `scripts/jev/run_zeroshot.py` (zeroshot). Analyses:
`paper/scripts/jev_swap_analysis.py`, `paper/scripts/jev_zeroshot_analysis.py` → `paper/generated/`.
(`scripts/jev/run.mjs` targeted the Vercel route, which never served a call; kept for the record.)
Costs: 24shot US$0.877 (6,160 calls); zeroshot US$3.263 (42,562 calls).
