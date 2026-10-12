# R16 Bitext 24-shot — amendment A2 (paper side, 2026-10-12)

Amends the research branch's `provenance/bitext_24shot_protocol.md` (pre-registration 9d6dc44, amendment A1
3d84cc3; code 837018a; manifest ce92f03, sha256 `1420b379…`).

**Timing.** This file is committed and pushed before the Jev arm sends any scored request and before any arm's
result is read by either session. It changes no request body, no item, no pool, no model setting and no primary
analysis. It adds one secondary analysis, one descriptive row and three run and reporting rules.

## Why

A duplicate audit on the pinned split, using text only and no model, shows the full test set is close to a ceiling:
- **Exact duplicates.** 832 of 5,375 test items match a pool row exactly after normalisation (lowercase,
  `{{placeholder}}` → one token, punctuation removed).
- **Near duplicates.** 2,876 items have a TF-IDF cosine of at least 0.8 to some pool row.
- **Trivial baseline.** Measured before this amendment on T (the pool plus S), a TF-IDF 1-nearest-neighbour label
  is already correct on about 97.7 % of the test set.

Full-test comparisons may therefore have too few discordant items to separate the systems.

## Changes

1. **Secondary analysis: the non-near-duplicate subset.**
   - **Membership.** A test item belongs to the subset if both hold:
     - its normalised text does not occur among P's normalised texts;
     - its maximum TF-IDF cosine to P is below 0.8 (word 1–2-grams, sublinear tf, vectoriser fit on P).
   - **Ids.** `paper/scripts/bitext24_subset.py` computes the ids from text only and writes them to
     `docs/phases/2.0/bitext24/subset_a2.json` (2,484 items), committed with this file.
   - **Tests.** The same 10 exact McNemar pairs as §9.1, Holm-corrected within this family, which is separate from
     the primary family.
   - **Calibration.** ECE-10 and Brier, using paired bootstrap against the open stack as in §9.2 (10,000 resamples;
     2,000 for ECE).
   - **Reporting.** This analysis is reported whatever its outcome, labelled secondary. The primary analysis (§9.1,
     full test set) is unchanged.
2. **Descriptive floor row.** The label of the BM25 top-1 example (`retrieved_t_idx[0]`) is reported for both the
   full test set and the subset. It is outside every test family.
3. **Duplicate audit.** The counts above are reported as descriptive.
4. **Jev arm: send order and cap.**
   - Items are sent in the order `np.random.default_rng(20261012).permutation(5375)`.
   - The US$1 cap covers the smoke calls too.
   - If the cap stops the run, unsent items are excluded from every Jev pair, so the analysed set is a random
     subsample. The size of that subsample is reported.
   - An attempted item that fails after three attempts counts as wrong (§8).
5. **Jev's prediction.**
   - Jev's prediction is its returned `choice`, as in the Banking77 analysis (`paper/scripts/jev_swap_analysis.py`).
     Its probabilities are rounded to 0.01 and may tie at the maximum.
   - Calibration uses the returned probabilities over the 27 raw intents.
   - The number of items where `choice` differs from the probability argmax is reported.
6. **Exposure disclosure.**
   - **Descriptions and ensemble defaults.** Both predate R16: `examples/bitext_descriptions.py` and the zero-shot
     ensemble defaults (M3's zero-shot member). They were evaluated zero-shot on this same 20 % test split in
     Phase 1 (commit 3c442d7, 2026-10-04) and have not been revised for R16. They are therefore not blind to this
     test split. If an M3 candidate is selected, the paper says so.
   - **Jev has already seen these test texts.** It received them, without labels, in our zero-shot run
     (`docs/phases/2.0/jev/zeroshot/`).
7. **Smoke calls.**
   - Three pool rows (positions 0, 7000 and 14000 of P), with the row itself excluded from its own retrieval.
   - Used only to check the response format and to project cost. Their outputs are discarded.
