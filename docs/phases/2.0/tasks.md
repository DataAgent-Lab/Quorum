# Phase 2.0 — Tasks (TMLR paper) · v3, status as of 2026-10-17

> Venue: **TMLR** (user decision; supersedes the TACL plan of v1/v2 — TACL-specific items below are marked
> superseded). Spec: [`spec_details.md`](./spec_details.md) · Venues: [`venues.md`](./venues.md)
> Conversation: `docs/archive/conversation-phase2.0-paper-publication.txt` · Claims: [`claims-ledger.md`](./claims-ledger.md)
> Model runs → the research session (git branch, verified here). Nothing outward-facing without the user's OK.
> Order rule: no section is drafted before its ledger rows are VERIFIED.

## Where the paper stands
Full draft, 21 pages, every number from a generator (`paper/generated/`). Build checks: layout-tmlr, numbers,
anonymity, and the AI-integrity checks (cites, artifacts, hidden, disclosure) all PASS; the online reference check
passes (scholarly entries match their records; web entries checked by hand). All pre-registered experiments are in.
Open: three author-only checks, the supplement archive, the review
rounds, and the user's must-ask items (T8).

## T0 — Venue, toolchain, day-1 requests
- [x] T0.1 Venue decision → TMLR (user). T0.1b self-review of spec/tasks v2.
- [x] T0.2 Style vendored (`paper/style-tmlr/`, never edited; TACL style superseded).
- [x] T0.3 TeX engine pinned (tectonic); `bash paper/build.sh` builds and checks.
- [x] T0.4 Scaffold (`paper/main.tex`, `sections/`, `scripts/`, `generated/`, `tools/`, `refs-candidates.bib` → `main.bib`).
- [x] T0.5 Research-session requests R1–R16 delivered and verified; pool size delivered 2026-10-17.
- [x] T0.6 Superseded (TACL submission-system note).

## T1 — Claims ledger (fact-first gate)
- [x] T1.1 `claims-ledger.md`. T1.2 zero-shot generator. T1.3 zero-shot Jev 0.801 DROPPED. T1.5 traps measured.
- [ ] T1.4 Split audit: our splits (CLINC150 plus/OOS, Bitext 20 % stratified, SNIPS 1,400) against each cited
      baseline's split — only needed where the paper compares with a published number.

## T2 — Evidence-gap closure
- [x] T2.1 Zero-shot calibration. T2.2 dump validation. T2.3 McNemar vs the reproduction (`numbers_24shot.py
      --repro-csv`). T2.4 24-shot calibration vs Jev (Table 3, `jev_swap_analysis.py`). T2.5/T2.5b clean
      re-selection protocol. T2.5c selection-on-test disclosure (§7, `sec:find-selection`). T2.6 parity and
      factorial tables. T2.7 Holm, bootstrap, seed variance. T2.11/T2.11b licences and data notice.
      T2.12–T2.15, T2.17 (combiner, baselines, calibration confirmation, contamination).
- [x] T2.8 Exposure table (Table 2) — checked 2026-10-17 against `licences-contamination.md` and the model cards
      (pplx's released data builder includes the PAWS train split; stated in Table 1 and §6.3). Qwen3 probe: not run.
- [ ] T2.9 Description authoring log: confirm from the log that descriptions were written once from label names
      and never revised after seeing test results (the paper states this).
- [x] T2.10 Latency: R17 delivered and verified 2026-10-17 (ledger E2); §8 written (open models + R24 on one GB10;
      one sentence on Jev's measured API latency, marked not comparable — user decision). CPU latency not measured
      and not claimed.
- [~] T2.16 Tie-break: R6c reported (§7). Remaining: library root fix after the evidence freeze.

## T3 — Literature and bibliography
- [x] T3.0 Novelty memo. T3.1 related work (§2).
- [x] T3.2 References: every cited entry verified 2026-10-17 against its record (DOI/arXiv/ACL via
      `ai_integrity.py refs`; cards, blogs and repositories by hand, with quotes). 62 entries in `main.bib`.

## T4–T5 — Outline and drafting
- [x] T4/T5.1–T5.6 All sections and appendices A/B drafted from verified ledger rows.
- [ ] T5.7 Copy-edit pass (English, consistency, terminology) and self-overlap check against the README.
- [ ] Author-only checks left in the text (3): study-purpose wording in §6.3; the AI-use statement's final wording
      (draft corrected 2026-10-17 from the AI-use log: all seven description sets were LLM-written, the Banking77
      set's model not recorded); commit identifiers for the de-anonymised version (Appendix B).

## T6 — Phase 1.2 decision gate
- [ ] T6.1 Passed → section; failed → negative result; not run → future work. Record (AC-11).

## T7 — Verification (one per AC)
- [x] T7.0 Tool fixtures. T7.1–T7.3 layout / anonymity / numbers pass in every build. T7.6 references verified.
- [x] AI-integrity checks merged (research 1715b35) and run in the build; `refs` by hand before submission.
- [ ] T7.4 Statistics checklist. T7.5 REAL spot check. T7.7–T7.9 calibration / parity / contamination sections
      against sources (largely done during verification; record a final pass).
- [ ] T7.10 Review round 1 (two reviewer personas + claims auditor) → `reviews/round1.md` + responses.
- [ ] T7.11 Review round 2 + at least one human reader; no open Critical/Major.
- [ ] T7.12 Phase 1.2 gate recorded. T7.13 arXiv build + licence + endorsement. T7.14 submission packet.
      T7.15 selection outcome reflected in wording. T7.16 release statement + AI disclosure present and anonymous.

## T8 — Must-ask items (never done silently)
- [x] T8.1 Author identity (user, 2026-10-05/12): JiunYi Yang, Taiwan, DataAgent AI Lab, as an individual; the
      email goes only into the submission system.
- [ ] 🔲 T8.2 Adapter access during review (reviewers cannot follow links); ungate at/after acceptance?
- [x] T8.3 Review version (user, 2026-10-17): keep the anonymous names (R24 / GeoJury); no statement about the public
      repository, Space or Hub pages and no link to them. TMLR does not require disclosing prior non-archival
      versions (author guide: only "not linking to another version that includes the authors' names"; editorial
      policy: reviewers do not search for authors), and acceptance rests on evidence and interest only. The
      repository stays public as it is; the de-anonymised version links it.
- [x] T8.4 Employer clearance — resolved 2026-10-12 (user). Pre-rewrite commits still resolve by old SHA on GitHub;
      the user decided on 2026-10-17 not to request a GitHub Support purge (clearance is resolved; the purge is slow).
- [ ] T8.5 README corrections once evidence is in — push is must-ask.
- [ ] T8.6 arXiv endorser — user action; posting is must-ask.
- [ ] T8.7 Submission to TMLR — must-ask, after the checks above pass.
- [x] T8.8 Jev retrieval swap (ledger F15). T8.10 Pool-size curve (ledger P1, 2026-10-17).
- [~] T8.9 Anonymous supplement (≤100 MB): `paper/tools/make_supplement.py` (trace the generators' inputs,
      anonymise, scan, zip into `paper/supplement/`, gitignored); rebuild and re-scan right before submission.
- [x] T8.11 Cost section written from R17 (2026-10-17).
