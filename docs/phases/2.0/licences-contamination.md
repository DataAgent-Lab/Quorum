# Phase 2.0 — Licences and contamination (G6, G10; T2.11, T2.15)

> Compiled 2026-10-06 from model/dataset cards, papers and licence files (every row cites its source in the
> research report summarised in the conversation log, DECISION #20). Rows marked **verified here** were re-checked
> directly in this session.

## Contamination of the zero-shot members (by documented training data)

CLEAN = documented absent · CONTAMINATED = documented present · UNKNOWN = not disclosed · † = clean in every
disclosed supervised/fine-tuning list; web-scale pretraining text not audited.

| Model | Banking77 | CLINC150 | HWU64 | MASSIVE | MTOP | SNIPS | Bitext |
|---|---|---|---|---|---|---|---|
| bge-large / bge-base-en-v1.5 | CLEAN† | CLEAN† | CLEAN† | CLEAN† | CLEAN† | CLEAN† | CLEAN† |
| bge-reranker-v2-m3 (R11 baseline) | CLEAN† | CLEAN† | CLEAN† | CLEAN† | CLEAN† | CLEAN† | CLEAN† |
| **PrismNLI-0.4B** (NLI member) | **CONTAMINATED (train split)** | CLEAN | **indirect** (HWU64↔MASSIVE overlap) | **CONTAMINATED (train split)** | CLEAN | CLEAN | CLEAN |
| Qwen3-4B (24-shot reader base) | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| 24-shot LoRA adapter (R4) | CLEAN (excluded) | CONTAMINATED | CONTAMINATED | indirect, 10.7% test overlap | 0.25% overlap | CONTAMINATED | CLEAN |

- **PrismNLI lineage (verified here):** card — "Instead of starting from scratch, we start from
  deberta-v3-large-zeroshot-v2.0"; that checkpoint's non-"-c" training mix = all datasets with used_in_v1.1=TRUE in
  its pinned list (MoritzLaurer/zeroshot-classifier @ 7f82e4a), which includes banking77 and massive (≤500 train
  rows per class; "No model was trained on test data").
- **HWU64 ↔ MASSIVE text overlap (verified here, normalised exact match, controls 100%/0%):** 470/1,076 HWU64 test
  utterances (43.7%) are in MASSIVE-en train (55.1% in any loaded MASSIVE split); 971/2,974 MASSIVE test (32.6%) are
  in HWU64 train. Shared origin: SLURP.
- bge-v1.5: C-Pack paper + archived BGE README + MTEB model metadata list no intent datasets (MTEB's Banking77 /
  MassiveIntent / MTOPIntent are evaluation tasks; `bge_full_data` belongs to bge-en-icl, not v1.5).
- **Consequence:** the 24-shot Banking77 headline (reader + kNN) does not use PrismNLI. Zero-shot claims are split
  into clean datasets for every member (CLINC150, MTOP, SNIPS, Bitext) and exposed ones (Banking77, MASSIVE, HWU64);
  R13 reruns the zero-shot ensemble with a documented-clean NLI member.

## Licences (cite originals, not mirrors)

| Artifact | Licence | Obligation for this paper/repo |
|---|---|---|
| PrismNLI-0.4B | CC-BY-4.0 (+ Qwen terms for models fine-tuned on its data) | attribution; we do not redistribute derivatives |
| bge-large/base-en-v1.5 | MIT | — |
| bge-reranker-v2-m3 | Apache-2.0 (Hub tag) | — |
| Qwen3-4B | Apache-2.0 | adapter redistribution keeps licence/NOTICE |
| Banking77 | CC-BY-4.0 | attribution |
| CLINC150 | CC-BY-3.0 | attribution |
| HWU64 | CC-BY-4.0 (original); DeepPavlov mirror untagged | attribution to Liu et al. |
| MASSIVE | CC-BY-4.0 (original; mteb mirror tagged apache-2.0) | attribution (Amazon; SLURP CC-BY-4.0) |
| MTOP | **CC-BY-SA-4.0** (official zip) | **share-alike for files containing MTOP utterances** (our prediction dumps) |
| SNIPS (7-intent) | CC0-1.0 (original) | — |
| Bitext customer support | **CDLA-Sharing-1.0** | outputs free; **publishing data/splits with texts carries share-alike** (our dumps include Bitext test texts) |
| Jev reproduction repo | no licence (rights reserved) | cite; never redistribute (we fetch at run time) |
| TypeSafe Terms / Vercel AI Product Terms | outputs owned by the customer; publishing allowed; no distillation/training on Jev outputs | do not train on Jev outputs |

**Action (T2.11b):** add a data notice next to `results/predictions/` stating that utterance texts inside the dumps
keep their source licences (MTOP CC-BY-SA-4.0, Bitext CDLA-Sharing-1.0, others as above), separate from the MIT code
licence. Public-repo change → push with the user's OK.
