# Outline v2 — TMLR, evaluation-methodology paper (supersedes `outline-proposal.md`)

> Decisions (author, 2026-10-11): positioning P1 (`../positioning.md`); venue TMLR; the 13-task code/workflow
> comparison joins this paper once verified; writing = **the LLM drafts, the author reviews, edits and takes
> responsibility for every claim** (TMLR: "LLMs may be used as general-purpose assistive tools… authors are fully
> responsible"). Every use is logged in `../ai-use-log.md` and disclosed. Numbers come only from the claims ledger
> via generated macros.
>
> TMLR format: official `tmlr.sty` (US Letter, double-blind, "Under review as submission to TMLR"); no page limit
> (length must be justified; very long main texts slow review); appendix after references, optional for reviewers;
> anonymized supplementary material ≤ 100 MB (PDF/ZIP) allowed; preprints allowed any time.

| § | Section | Rests on | Status |
|---|---|---|---|
| — | Title, abstract (author confirms the claims) | all | after drafting |
| 1 | Introduction: decision models are a new product class; comparing them is easy to get wrong; our protocol and what it reveals | F1, F15, K1–K3, L1, J1–J3, M*, P6 | draft now |
| 2 | Background: calibrated decision APIs (Jev) and open decision models (Clef, Clef-Flash, pplx-decider); intent classification; evaluation practice (related work) | novelty memo, refs | draft now |
| 3 | **The equal-evidence protocol**: pre-registration before every scored run; byte-identical, sha256-pinned request bodies; paired exact tests with Holm; serving-engine parity gate; release-code over-length policy; exact replication of a closed API's published result | F1a, F15, A4–A7, request manifests | draft now |
| 4 | Systems and data: decision models (pins, licences, scoring provenance: release code vs parity-gated engine); open baselines (zero-shot jury, retrieval 24-shot stack, fixed-budget fine-tune); datasets (7 intent sets, Bitext 24-shot, 13 code/workflow tasks); contamination audit per model × dataset | P1–P7, K4, L2, licences | draft now; Bitext/13-task rows later |
| 5 | Results I — Banking77 in depth: replication; 24-shot under both retrieval rules on identical evidence; what in the retrieval matters (cap, pool size); calibration vs accuracy; zero-shot | F1–F15, K1–K3, L1, pool-size curve | draft now |
| 6 | Results II — breadth: zero-shot on 7 datasets (with exposure flags); Bitext 24-shot; 13 code/workflow tasks | J1–J3, L2, Z*, pending runs | after the runs |
| 7 | Methodology findings: metric pitfalls (rounding → log loss), platform-dependent tie-breaking, serving input limits, engine parity, over-length policies flipping rankings, contamination through model lineage, selection on test, smoke tests, unpaired tests | J2, F10/F14, M1–M5, P6, de-identified internal findings | draft now (internal parts after the data arrives) |
| 8 | Cost and latency (de-identified, one hardware type for the open models; Jev API latency not comparable) | R8, internal numbers | after the data arrives |
| 9 | Limitations, broader impact, release statement | all | draft now |
| A | Appendix: the zero-shot jury in full (ensemble vs members, descriptions, single-temperature calibration), protocols, prompts/bodies, per-model details, full tables | Z1–Z9, C* | draft now |
