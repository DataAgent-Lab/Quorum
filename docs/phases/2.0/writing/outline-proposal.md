# Outline proposal — for the author to approve or change (AI-suggested structure; logged as such)

TACL: ≤10 content pages (A4) + references; Appendix A ≤5 pp (replication) and B ≤3 pp (complementary results),
**not reviewed** — so every load-bearing protocol detail stays in the main text. Numbers come only from the claims
ledger via generated macros. **Who writes:** H = the author's own content (dictated, then translated/polished, item
(a)); D = LLM-drafted text about pre-existing ideas with verified citations (item (d), disclosed per section).

| § | Section | Pages | Who | Ledger rows it rests on |
|---|---|---|---|---|
| — | Title, Abstract | — | H (written last) | — |
| 1 | Introduction: problem, what is new, contributions | 1.25 | **H** | F1, F15, Z2, Z4b, Z8, M1–M5 |
| 2 | Related work: zero-shot intent detection, ensembles/pooling, retrieval ICL, calibration, evaluation methodology, Jev replications | 1.0 | **D** | novelty memo, refs |
| 3 | Method: 3.1 the zero-shot jury, 3.2 retrieval-augmented 24-shot reader + kNN, 3.3 one global temperature | 1.5 | component descriptions **D**; design rationale **H** | Z6, Z7 |
| 4 | Evaluation protocol: datasets and splits, metrics and tests (**D**); pre-registration, contamination audit, comparison rules (**H**, our methodological contribution) | 1.25 | D + **H** | P1–P7, F1a, G7 |
| 5 | Results: 5.1 zero-shot jury (incl. clean-NLI robustness); 5.2 calibration; 5.3 against Jev (pre-registered headline, equal-retrieval comparison, retrieval mechanism, fine-tune reference, zero-shot Jev *pending*); 5.4 robustness (seeds, tie-breaking) | 3.0 | **H** interpretation; tables/figures generated | Z1–Z9, F1–F15, B1 |
| 6 | What did not work, and lessons (the self-caught traps, incl. contamination through model lineage) | 1.0 | **H** | M1–M5, P6, F5, C4 items |
| 7 | Limitations and ethics; release statement | 0.75 | **H** | P2–P7, licences |
| — | Acknowledgements incl. AI-use disclosure | — | drafted from `ai-use-log.md`, approved by **H** | — |
| A | Appendix A: hyper-parameters, prompts, label descriptions, adapter training data, compute | ≤5 | D/factual, checked by H | R4, R7 |
| B | Appendix B: per-dataset tables, reliability diagrams | ≤3 | generated | — |

**Decisions for the author:** the title; what the paper leads with (suggestion from the evidence: rigour and
calibration, with the Jev comparison as a case study); whether §6 stays a full section; anything to cut.
