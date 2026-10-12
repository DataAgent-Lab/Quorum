# Decision-model study on code and workflow decisions: pre-registration record (blocks E and C)

**Status: record of a completed study.** This protocol was pre-registered in a private repository on 2026-10-08,
before any model call on blocks B–F. Amendments 1–9 were each committed there before the runs they govern, all on
2026-10-09. It is transcribed here, de-identified, after the runs: host names, internal paths, file names of
operational scripts and the names of unrelated services were removed or replaced by a neutral description. **No
rule, number, threshold or timestamp was changed in transcription.** Where the original pointed to internal notes, the
fact it relied on is stated inline instead.

Code: `scripts/decision13/` (runner, builders, analyses). Results: `results/decision13/` (manifests, per-item
predictions for the redistributable sets, runner summaries, analyses). Licences and what is published per set:
`results/decision13/LICENSES.md`.

## 1. Question and the decision this protocol settles

Which of **Clef-Flash, Clef 27B, pplx-decider v1.1** to adopt as the model behind a decision service for code and
workflow decisions. The answer comes only from runs under this protocol, with the same items, request bodies, scoring
and hardware for every model. Vendor model-card numbers are not used: they are measured on different items and
protocols and are not comparable.

## 2. Models and serving

| model | revision | engine for scored runs |
|---|---|---|
| pplx-decider v1.1 27B | `perplexity-ai/pplx-decider-v1.1-27b` @ `3b45dea` | SGLang + noncausal patch (parity gate **PASS**, §2a) |
| Clef 27B | `Cloudflare/clef` @ `2f3de3d` | SGLang + PR #42721 patch **if its parity gate passes**, else the release code |
| Clef-Flash 9B | `Cloudflare/clef-flash` @ `17f0b0a` | same as Clef 27B |

- **Hardware**: one NVIDIA GB300, image `lmsysorg/sglang@sha256:868b0bd8…a8b77f`.
- **Concurrency**: requests go one at a time, the condition under which the parity gates were measured.
- **Failures**: a failed request counts as unanswered (§5).
- **Engine record**: every result names its engine, image, patch sha and gate summary.
- **Jev**: not part of this decision. Adding it would have needed the user's go (paid API; TypeSafe MCA: evaluation
  only). It was not added; no Jev output exists for this study.

### 2a. The serving-engine parity gate (measured 2026-10-08, before this protocol)

Each SGLang engine was compared item by item with that model's release-code dump on the 3,080 Banking77 `b24` bodies
of `provenance/clef_eval_protocol.md`, one request at a time. Pre-set thresholds:
- **G1** argmax agreement with the reference ≥ 99.5 % of items;
- **G2** |Δacc| ≤ 0.5 pt, with exact two-sided McNemar (engine vs reference correctness) p ≥ 0.05;
- **G3** max over items and options |Δp| ≤ 0.10.

Results:
- pplx-decider v1.1 (noncausal patch, sha256 `fa7f6afc…4167`; reference computed on the GB300 with the release
  code): **PASS**. G1 0.99935 (2 items differ), G2 Δacc +0.0003 (b/c 1/0, p = 1), G3 max |Δp| 0.0600 (mean 4e-5,
  p99 0.020), 0 failed requests. This is amendment A7 of `provenance/clef_eval_protocol.md`.
- Clef-Flash (PR #42721 head `d1cea71`, Python-only patch sha256 `b33f25a8…6116`; reference from the release code on
  a GB10): **PASS**. Banking77 b24 agreement 1.000, Δacc 0, max |Δp| 0.039, 0 failed; CLINC150 zero-shot names
  agreement 0.99927, Δacc +0.0002 (b/c 2/1), max |Δp| 0.060, 0 failed.
- Clef 27B (same engine and patch): **FAIL on G3 only**. Agreement 0.99773 (7 items differ), Δacc 0 (b/c 3/3), max
  |Δp| **0.1045** (mean 8.7e-5, item-max p99 0.025), 0 failed. No rerun. Noted for diagnosis only: the Clef
  references were computed on a GB10 while the pplx reference was computed on the GB300 itself, so part of the gap
  may be cross-hardware numerics. See amendment 1.

## 3. Blocks and items (fixed)

**Block E — code and firmware dev/test decisions (primary for the decision).** Items are frozen as built. Bodies,
ids and `items_sha256` are in the manifests (`results/decision13/manifests/`; the DBRD manifest is not published),
and every run checks each body's sha256 before sending it.

| set | items | items_sha256 (prefix) | question / unit | reported breakdown |
|---|---|---|---|---|
| PatchNet (Linux stable patches) | 10,896 | `8e0ecce4` | noul: fix suitable for stable | — |
| TOSEM-DBRD (duplicate bug reports) | 17,126 | `c4b5566e` | noul: same underlying problem | per project; without "Intermittent" pairs |
| EquiBench (6 categories) | 2,400 | `557e570d` | noul: equivalent (per-category definition) | per category |
| POJ-104 pairs | 12,000 | `0ab042fe` | noul: same problem | — |
| PrimeVul v0 paired | 564 | `78a1d2a6` | choice: which version is vulnerable | — |
| PrimeVul v0 per function | 1,128 | `7ca8734a` | noul: contains a vulnerability | — |
| SV-TrustEval-C | 10,155 | `b7b822bb` | choice (own options) | per split |
| SecVulEval | 25,440 | `eb48e4d5` | noul: contains a vulnerability | — |
| Juliet C/C++ 1.3 (synthetic floor) | see §8 | `e65b4d1b` (all 128,186) | noul: contains the flaw | with the structural-cue note |
| PAWS (contamination control) | 8,000 | `24794a9f` | noul: paraphrase | reported apart, never in the decision |

The "structural-cue note" for Juliet (recorded when the items were frozen, 2026-10-08): after all comments are removed
and label names renamed (0/0 remaining bad/good/G2B/B2G/CWE/FLAW/fix substrings), a structural cue remains. Good sides
have more functions (mean 2.83 vs 1.70) and are longer (mean 1,867 vs 1,640 bytes); see `residual_structure` in the
manifest. It is reported with any Juliet score; Juliet is a synthetic floor, not a headline.

**Block C — business workflows (primary).** The 4 Typesafe WorkflowEvals at the pinned revisions of amendment 8, run
with their own harness @`0ac3b8a` against the model's endpoint. Metrics are the harness's own: exact actions /
primary action. Labels are a frontier-model consensus, not human labels.

**Block B — Decision Index 0.3 public suite (secondary).** All 37 benchmarks, rebuilt and hash-verified, run with the
kit @`9eb2dbe` `http` engine. Scoring is the kit's native scorer with chance correction, reported per area. The
board's Full score includes private sets and cannot be reproduced, so only the public index is ours.

**Block D — JevBench public hard (secondary).** 111 items @`bb05a33`, run with its own adapter. Reported as accuracy
plus paired McNemar.

**Block F — JevBench coherence (reliability axis).** Run @`cfd3325` / code @`e187336` with its own metrics. Ranked
separately and never mixed with accuracy.

**Block A — intent (context only).** R14/R15 results under `provenance/clef_eval_protocol.md`. Banking77 b24 is
already measured for all three models. Not re-run here.

*Transcription note:* blocks B, D and F were not run. None of them enters the §7 rule. The study's results are
blocks E and C.

## 4. Request construction

- Blocks E and A use the frozen bodies (`{"model":"decision","state":{…},"questions":{…}}`).
- Blocks B, C, D and F let each harness build its own requests and drive the model's `/v1/systemone`. Harness prompts
  are not edited. The only per-model change is the endpoint URL.
- Dict states (every block E state is a dict) render as in each release: pplx through the decider path, the same
  `json.dumps(value, ensure_ascii=False)` call as the release `describe`; Clef through the release `render` and PR
  #42721, both `json.dumps(..., separators=(",", ":"), sort_keys=True)`.

## 5. Over-length and unanswered items

- **Unanswered = wrong.** An item a model cannot accept counts as wrong, following DI; this covers context limits
  (pplx-decider 8,192 tokens, no truncation; Clef 16,384 by default), a rejected request, and a failed request.
- **No item is truncated, shortened or dropped** to make it fit.
- **Two tables per set.** Every table reports the full set and the subset that all three models accepted. The counts
  of unanswered items are shown per model.

## 6. Metrics

- **noul / choice sets**: accuracy (primary). Also macro-F1, and for binary sets F1 of the positive class.
  Calibration is computed as in R14 (`scripts/r14_clef.py` `metrics`): Brier is summed over all options, and ECE-10
  uses the probability of the predicted option.
- **SV-TrustEval-C multi-letter golds**: any listed letter counts, as in the authors' scorer.
- **Blocks B–D, F**: each harness's native metric, as defined by that harness.
- Probabilities are the engine's full-precision values. No log loss is compared across engines that round.

## 7. Comparisons and decision rule

- **Pairwise accuracy (E, D)**: exact two-sided McNemar for every pair of models on identical items, per set. Holm
  correction within each block, applied across all pairs × sets of that block.
- **Pairwise calibration**: paired bootstrap of the ECE-10 and Brier differences, 10,000 resamples, seed 20261008,
  95 % CI.
- **Block C**: per workflow, paired McNemar on the harness's per-case correctness (exact actions), Holm within the
  block.
- **Block B**: per-area public index difference with a paired bootstrap over benchmarks. Descriptive; not used in the
  decision rule.
- **Decision rule** (fixed now):
  1. **Eligibility**: a model is eligible only if its serving engine passed the per-item parity gate (§2a), or it is
     served by its release code.
  2. **Primary score**: over blocks E (excluding PAWS) + C, count each model's Holm-significant pairwise wins minus
     Holm-significant losses on accuracy.
  3. **Winner**: the highest primary score. If two models are within 1 of each other, the model with better
     calibration wins — more Holm-significant ECE-10 wins over the same sets (bootstrap CI excluding 0).
  4. **Ties**: if they still tie, the tie is reported and the choice goes to the user, with throughput on the GB300
     shown as context.
  5. **Exposure sensitivity**: a recommendation that flips when exposed cells (§9) are removed is reported as "not
     robust", and the user decides.
- **No single cross-block average** is used as the headline.

## 8. Open choices fixed before the affected runs

- **Juliet scope** (user: decide after the engine work settles speed): **all 128,186 items** if the measured rate on
  the GB300 lets all three models finish within one run window of ≤ 12 h each; otherwise the HF mirror's predefined
  test split (35,119 items, the `in_hf_test_split` flag). The choice is recorded by amendment before any Juliet call.
  Juliet is a floor and never decides the winner alone.
- **PatchDB**: excluded unless access is granted before block E starts. *(Transcription note: the dataset is gated and
  access was not obtained before block E; PatchDB is not in the study.)*
- **Clef engine**: see §2.

## 9. Exposure flags (results reported with and without flagged cells)

| model | flagged sets | basis |
|---|---|---|
| pplx-decider v1.1 | PAWS; DI: possibly WinoGrande, BBH, RAGTruth | trained on tasksource data (contains PAWS, MRPC, QQP, STS-B, Devign); v1 used WinoGrande/BBH/RAGTruth rows for checkpoint selection (DI notes; v1.1 not stated) |
| Clef / Clef-Flash | none flagged, **all undisclosed** | training data not disclosed; DI board shows a large public → private drop |
| all | SecVulEval (released only as a train split); PatchNet (Linux history ≤ 2017, very likely in pre-training) | assume exposure |

## 10. Run discipline

- One scored run per model × set.
- Smoke runs use explicit limits and separate file names. They are never results.
- Every run checks body hashes before sending, records engine and hardware, and writes per-item outputs:
  - fields: id, pred, full probabilities, gold, unanswered flag;
  - per-item files > 10 MB are not committed;
  - summaries and manifests are committed.
- Order of runs: block E first, then C, then D and F, then B. Within a block, model order is pplx v1.1 → Clef-Flash →
  Clef 27B. Order does not affect any item or score.
- The GPU was shared with other, unrelated model services. Each run window stopped one co-located service and
  restored it on exit; a second co-located service had to be healthy before a window started. (Transcription:
  service names removed; see amendment 1 for the change to restoring.)

## Amendment 1 (2026-10-09, before any model call on blocks B–F): Clef 27B is served by SGLang

**User decision**, made after seeing the parity-gate result: Clef 27B is scored by the same SGLang + PR #42721 engine
as Clef-Flash, although its parity gate failed. Its gate (Banking77 b24, 3,080 items, vs the GB10 release-code dump)
failed on G3 only:
- 1 item had max |dp| > 0.10 (0.1045; next largest 0.078, 0.072);
- 4 items had max |dp| > 0.05;
- argmax agreement was 0.99773 (7 items differ), and Δacc was 0 (b/c 3/3).

Consequences, fixed now:
- §2 and the eligibility rule in §7.1 treat Clef 27B as eligible on this engine.
- Every Clef 27B result is labelled "engine gate failed on G3 (1 / 3,080 items above 0.10); user-approved".
- If Clef 27B wins under §7, the recommendation states this. It also states that its served outputs may differ from
  the release code by up to ~0.10 in probability on rare items.

Operations: the co-located service that each window stopped may stay offline until this study's runs are complete
(user, 2026-10-09). Windows no longer restore it between runs; it is restored once at the end, or earlier only on the
user's request.

## Amendment 2 (2026-10-09, before any model call on blocks B–F): concurrency, GPU-safety guards

1. **Concurrency rule** (fixed before any result). The parity gates were measured one request at a time. Before a
   model's first call in this study, its engine re-runs the 3,080 Banking77 b24 bodies with 16 requests in flight.
   These outputs are compared item by item with that engine's own one-at-a-time outputs from its parity gate.
   - If argmax agreement is ≥ 99.5 % and max |dp| ≤ 0.10, all of that model's runs use 16 in flight.
   - Otherwise they use 1 in flight.

   The comparison and the choice are recorded per model (`results/decision13/concurrency_checks.json`). The c16
   run's agreement with the release-code dump is reported too, as information only. Rationale: batch composition can
   move probabilities on hybrid models; the check measures that before relying on it.
2. **ECC guard.** Before and after every set, the GPU's volatile ECC counters are read (`nvidia-smi -q -d ECC`;
   volatile DRAM/SRAM correctable and uncorrectable).
   - If any is non-zero, the window stops at once: the server is stopped and nothing more is sent.
   - The set in progress is discarded (never reported), and the user is told.
   - Context: the GB300 had a multi-bit HBM ECC event on 2026-10-09 and was rebooted. After the reboot it showed 6
     remapped rows, no pending remaps and no remap failures.
3. **VRAM guard.** After the server reports ready, total GPU memory use must be ≤ 90 % of the device (user rule,
   2026-10-09). Otherwise the server is stopped before any call. `--mem-fraction-static` starts at 0.75.
4. **Order within the window.** For each model in turn (pplx v1.1 → Clef-Flash → Clef 27B), in this order: the
   concurrency check, 30-item smokes of every block E set (`_limit30` files, never results), then the full block E
   sets. Juliet is not run in this window; its scope is fixed by amendment after the measured rates are known (§8).

## Amendment 3 (2026-10-09, after an operator stop, before any reported result): VRAM guard made continuous

- **What happened.** The first full block-E set for pplx v1.1 (`patchnet`) was **stopped by the operator** after
  about 2.5 min. At start, total GPU memory was 89.7 % (230,352 / 256,703 MiB). It then grew to 96.8 % (248,587 MiB)
  under load, above the 90 % rule. That growth is prefill activations: with chunked prefill disabled, batches reach
  ~16 k tokens.
- **Discarded.** The set's partial output is kept as `DISCARDED_…partial` and never used. The rerun starts that set
  from scratch. No completed set or result exists from that window.
- **Changes**, none of which alter items, bodies or scoring:
  1. `--mem-fraction-static` is lowered from 0.75 to 0.55.
  2. A watchdog checks total GPU memory every 10 s throughout the window. Above 90 % it kills the runner and stops the
     server, and the set in progress is discarded.
- **Concurrency.** Because a server flag changed, the amendment-2 concurrency check is re-run under the new settings
  before any further call. Its outcome replaces the earlier one.

## Amendment 4 (2026-10-09, before any further run): context limits equal to the release code; VRAM-guard race

**What happened.** During pplx v1.1 `equibench`, the VRAM watchdog stopped the run at 90.7 % (232,865 / 256,703 MiB).
The cause: the SGLang server accepted prompts far longer than the release code allows, up to ~156 k tokens in
EquiBench. The release `DecisionModel` refuses any question longer than **8,192 tokens** (no truncation), and Clef's
release code uses `max_length` **16,384**. So the engine was answering items the release would refuse, which breaks
§5. The run is discarded (`DISCARDED_…partial`).

**Measured with pplx v1.1's own tokenizer and prompt builder**, items > 8,192 tokens: patchnet 1 (13,110 tokens),
dbrd 0, equibench 623, poj104 0, primevul_pair 54, primevul_func 52, svtrusteval 3, secvuleval 402, paws 0. Only
items above 8 KB were checked; a smaller item cannot exceed the limit.

**Changes, none of which alter items, bodies or scoring:**
1. Each server enforces its model's release limit, so it refuses exactly what the release code refuses:
   - pplx v1.1: `--context-length 8193`, allowing at most 8,192 prompt tokens;
   - Clef / Clef-Flash: `--context-length 16385`.

   A refused item is unanswered, which §5 counts as wrong. Before the first full set, one over-limit and one at-limit
   item are sent to confirm the boundary.
2. pplx v1.1 `patchnet` already completed without this limit. Its 1 over-limit item is re-scored as unanswered by
   this rule. Nothing is re-run. The re-scored summary keeps the original alongside and names the item. `dbrd` is
   unaffected (0 items over the limit). *(Transcription: the item is idx 3386; files
   `summary_patchnet_pplx_v11.json` (original) and `summary_patchnet_pplx_v11_lengthrule.json` (re-scored); the
   analysis applies the rule to the original per-item file.)*
3. The watchdog now writes the breach marker *before* it stops anything. In the stopped window, the main loop started
   three later sets in the ~10 s gap; each failed at the served-model check before sending any item, so they left no
   output.
4. Completed full sets are not re-run. A window skips any set whose full summary already exists.

*Correction to amendment 4 item 3 (2026-10-09):* of the three sets started in the gap, `poj104` passed the
served-model check before the server stopped. It opened a partial file but wrote **0 records**: the file is 58 bytes,
a gzip header only. `primevul_pair` and `primevul_func` failed at the check. No item result exists from any of them.
The `poj104` file is kept as `DISCARDED_…`.

*Runner fix (2026-10-09 10:10Z; no change to scoring):* the "25 consecutive failures" dead-server abort also counted
HTTP 4xx refusals, i.e. items over the release limit. EquiBench has long runs of such items, so its full run was
aborted at 09:54:51Z and is re-run. The runner now counts only connection errors, timeouts and 5xx toward that abort;
a 4xx refusal is recorded as unanswered (§5) and resets the counter. It was deployed between sets, so `primevul_pair`
(old runner, completed) and later sets differ only in this abort rule. Consistency check: `primevul_pair` has 54
unanswered, exactly the 54 items counted over 8,192 tokens in amendment 4.

## Amendment 5 (2026-10-09, before any Clef call on block E): over-length handling follows each release code

**Read in the release code:**
- **pplx-decider v1.1** refuses a question longer than 8,192 tokens and does not truncate (amendment 4).
- **Clef / Clef-Flash** (`joint_schema_model.py:166-170`) do not refuse. They cut the *state* from its end so that the
  schema fits in 16,384 tokens (`state_ids[: max_length - fixed_length]`), and raise only when the schema alone is too
  long. The SGLang Clef path documents the same behaviour ("truncated so that every question still fits, as in the
  checkpoint's reference code").

**Consequences:**
1. Each engine reproduces its own release behaviour. Over-length items are refused by pplx (unanswered = wrong, §5)
   and truncated by Clef, which then answers. This is the models' own behaviour, not ours, so the §5 rule "we never
   truncate items" still holds.
2. The Clef boundary test expects the over-limit item to be **accepted with exactly 16,384 input tokens**. The earlier
   test, which expected refusal, aborted the first Clef-Flash and Clef 27B attempts before any block E call.
3. The runner now records each item's `input_tokens` (server usage). A Clef item at 16,384 tokens counts as truncated.
4. The like-for-like subset in every table is: items that pplx answered **and** that no Clef model truncated. Full-set
   results are reported too.
5. Not verified: that SGLang's Clef truncation is token-identical to the release on long items. The parity items were
   all short. The boundary test checks the resulting length only.

*Transcription note (measured after the runs):* truncated items per set, identical for Clef-Flash and Clef 27B:
equibench 254, primevul_pair 26, primevul_func 14, secvuleval 86; all other sets 0.

## Amendment 6 (2026-10-09, after block E, before any analysis or further run): SGLang's effective pplx limit was 8,186 tokens

**What was found.** pplx v1.1 `equibench` had **624** unanswered items. Amendment 4 counted 623 items over 8,192
tokens. The extra one is **`equibench` idx 2098**. With pplx v1.1's own tokenizer and prompt builder it is **8,189
tokens**, so the release code accepts it. SGLang refused it with HTTP 400.

**Cause (read in the pinned image).** SGLang applies two limits:

| check | code | refuses when |
|---|---|---|
| tokenizer manager | `tokenizer_manager.py:1299` | n ≥ `context_len` |
| scheduler | `managers/utils.py:261` | n ≥ `max_req_input_len` |

`max_req_input_len` is `min(context_len − 1, kv − 1) − 5` (`tp_worker.py:544-548`). With `--context-length 8193`, the
scheduler therefore refused prompts of **≥ 8,187 tokens**: the served limit was 8,186, not 8,192. The amendment-4
boundary test sent items at 7,937 and 8,820 tokens, so it could not see a 6-token gap.

**Which items are affected.** Only items between 8,187 and 8,192 tokens that were refused.
- `equibench`: idx 2098 only. Its refused set is exactly the 623 predicted items plus 2098 (verified item by item).
- `primevul_pair`, `primevul_func`, `svtrusteval`, `secvuleval`: unanswered counts are 54 / 52 / 3 / 402. Each equals
  that set's count of items over 8,192. Every item over 8,192 is refused under both limits. So no item in
  8,187–8,192 was refused.
- `patchnet`, `dbrd`: ran with no limit (amendment 4), so they are unaffected.
- `paws`, `poj104`: have no item over 8 KB.

**Clef.** The same formula gives a scheduler cap of 16,379 for `--context-length 16385`. However, the amendment-5
boundary test measured an over-limit item accepted at **exactly 16,384** tokens, so the Clef path does not hit this
cap. Clef results are unchanged.

**Change, which alters no item, body or scoring rule:**
1. pplx v1.1 is re-served with `--context-length 8199`. The tokenizer then refuses n ≥ 8,199 and the scheduler refuses
   n ≥ 8,193, so it accepts exactly ≤ 8,192. All other flags are unchanged.
2. Before the item is re-sent, two boundary checks run:
   - idx 2098 must be accepted with `input_tokens` = 8,189, the release count;
   - `equibench` idx 390 (8,201 tokens) must be refused.

   The exact 8,192 / 8,193 edge is not tested, and it decides no item:
   - in `equibench`, the nearest items are 8,189 and 8,201 tokens;
   - no other set is re-sent, and none of them has an item between 8,187 and 8,192 tokens.
3. Only idx 2098 is re-sent, one request alone. Its parity reference was also one at a time, so no concurrency check
   is needed. The result is a separate file, `equibench_pplx_v11_amend6`, and the original 624-unanswered file is
   kept. Analysis uses the `_amend6` file. *(Transcription: measured afterwards, idx 2098 was accepted with
   `input_tokens` 8,189 and idx 390 refused. Only the `_amend6` per-item file is published; the original summary is
   published alongside.)*
4. Guards are the same as the block-E window: the co-located service checks, ECC clean before and after, continuous
   ≤ 90 % VRAM watchdog.

**Engine-record correction.** The block-E window's engine records wrote the hard-coded string
`--mem-fraction-static 0.75` into every summary's `engine.flags`. The servers actually ran **0.55**:
- `mem_fraction_static: 0.55` is in the surviving server logs of the last pplx, Clef-Flash and Clef 27B windows;
- the script default had been 0.55 since amendment 3 (08:25Z). That is before the earliest kept full set (pplx
  `patchnet`, written 08:54Z).

The only 0.75 server log is the discarded amendment-3 run. The window script now writes the real value. Existing
summaries are not rewritten; this note is the correction. *(Transcription: the published summaries keep the recorded
string; this correction applies to them.)*

## Amendment 7 (2026-10-09, before any Juliet call): Juliet scope = all 128,186 items (§8 rule applied)

**Rule (§8).** All items if the measured GB300 rates let each model finish within ≤ 12 h; otherwise the HF test split.

**Measured rates** (block-E window, 16 in flight, from each summary's `n / wall_seconds`):
- Projection set: POJ-104, the closest byte profile to Juliet (p50 1,692 vs 1,382 bytes; Juliet averages 1,753).
- pplx v1.1: 14.5 items/s → **≈ 2.5 h**.
- Clef-Flash: 33.0 items/s → **≈ 1.1 h**.
- Clef 27B: 11.3 items/s → **≈ 3.2 h**.
- Cross-check against the slowest long-item set (SecVulEval, p50 1,809 bytes; Clef 27B 9.8 items/s): ≈ 3.6 h.

Every model is far below 12 h, so **all 128,186 items** are run. Juliet stays a floor. It is reported with the
structural-cue note and never decides the winner alone (§8). It enters the block-E Holm family and primary score like
the other decision sets.

**Lengths** (measured before any call with pplx v1.1's prompt builder):
- 231 items are above 8 KB; the longest is 3,365 tokens.
- No item is over pplx's 8,192 or Clef's 16,384, so no Juliet item is refused or truncated.
- If the run shows any unanswered or truncated item, it is a failure to investigate, not a length effect.

**Serving.** Same window procedure and guards as the block-E window: co-located service checks, ECC before and after
each set, continuous ≤ 90 % VRAM watchdog, boundary test, amendment-2 concurrency check per model, `_limit30` smoke,
then the full set. Model order is pplx v1.1 → Clef-Flash → Clef 27B. pplx uses `--context-length 8199` (amendment 6),
and its under-limit boundary item is now equibench 2098 (8,189 tokens). Breakdown is by CWE. The concurrency checks
of this window are recorded separately, so the block-E window's are kept.

## Amendment 8 (2026-10-09, before any block C call): how block C is run and scored

**Harness.** typesafe-ai/WorkflowEvals @`0ac3b8a`, run with its own code, unedited.
- Container: `python:3.13-slim` @`sha256:70729b46…`, `uv` 0.12.24, `uv sync --locked`.
- Wrapper: `scripts/decision13/run_workflowevals.sh`. The harness defaults are kept:
  - provider `typesafe`, model name `jev-1.13.0` (hard-coded; the local servers do not check the name);
  - `question_mode: all`, one request per decision node with all of its questions;
  - timeout 600 s, `malformed_retries` 1, all policies.
- The only changes are `--base-url` (the local server) and `--workers`.

**Data**, pinned to full revisions. A dry run downloaded and checked all four on the run host:

| workflow | revision |
|---|---|
| invoice_processing | `6beeb2d2acd65c086c835022f5f4d7434114cafc` |
| customer_service | `b1342f5a704587dbc465867c38c2694348ff86e4` |
| security_incidents | `fbe1ea5c69cf494157fd23f2002a0d9d9a418443` |
| agent_trace_observability | `8635540973910a92465fe2bc53e195375aa6e1a8` |

**Concurrency.** `--workers` is the amendment-2 result for the model, 16 for all three. The check re-runs at the start
of the window.

**Unit and scoring** (fixed now; §7 says "per-case correctness", made precise here):
- Unit: one scored decision, that is, a case × policy row in the harness's `scores.json`.
- Primary metric: `exact_actions` against **`labels-consensus`**.
- Rows whose status is not `scored` or `prediction_error` are excluded for every model alike, because they depend only
  on the shared labels.
- `prediction_error` counts as wrong, the harness's own `count_prediction_errors_as_wrong`. This is how §5 applies
  here; it includes pplx refusals of questions over 8,192 tokens.
- Per workflow, an exact McNemar runs for each model pair on identical rows. Holm runs across the 4 workflows × 3
  pairs = 12 tests of block C, and those results enter the §7 primary score.
- `primary_action` and the openai / anthropic label sets are reported as context only.

**Run discipline.**
- A block-C window mode. It keeps the same guards: co-located service checks, ECC, continuous ≤ 90 % VRAM watchdog
  (it also stops the harness container), boundary test, concurrency check.
- Per model: a 3-case smoke of each workflow (`<model>-smoke3`, never a result). Any `prediction_error` in a smoke
  stops that model so it can be investigated. Then the 4 full workflows, run name `<model>`.
- After each workflow the server must still be healthy. Otherwise the run is invalid, because errors would come from a
  dead server, not from the model, and it is re-run after an amendment. No `--retry-errors`.
- Order: pplx v1.1 → Clef-Flash → Clef 27B. Block C starts after the Juliet window finishes.

## Amendment 9 (2026-10-09, after the first block C smoke, before any scored block C result): length refusals in the smoke

**What happened.** The pplx v1.1 `invoice_processing` 3-case smoke had 6 `prediction_error` decisions, so the
amendment-8 smoke rule stopped pplx's block C. Every error is the same HTTP 400: `question 'statement_not_invoice':
the prompt has 8998 tokens, which does not fit the context length`. Each question is over 8,192 tokens, so the
release code refuses it too (`model.py:284-285`). This is the over-length refusal that §5 and amendment 8 already
count as wrong. It is not a fault.

**Change.**
1. A smoke stops a model only on prediction errors whose reason is *not* a context-length refusal. Length refusals are
   scored as wrong, as before.
2. pplx v1.1 block C runs after Clef-Flash and Clef 27B, in a follow-on window. A chain script starts it only after
   the first block-C window has logged that it is done. Its concurrency check is re-run.
3. The smoke cases already saved are reused; the harness resumes from them. No scored result changes.

No item, request or scoring rule changes. The count of pplx length refusals per workflow will be reported with block
C.

*Addendum to amendment 9 (2026-10-09 20:10Z, before any scored block C result).* The Clef-Flash `invoice_processing`
smoke stopped for the same class of reason. All 9 decisions failed with HTTP 400, for example `the questions need
28719 prompt tokens before the state, but at most 16384 fit the context length`. Across the 3 cases the schema needed
16,885, 18,917 and 28,719 tokens.

Clef scores all questions of a node in one prompt. When the schema alone exceeds `max_length`, the release code raises
instead of truncating (`joint_schema_model.py:166-168`). So this is again the release refusal, counted as wrong under
§5. The smoke exemption therefore matches both engines' wording, `fit the context length`.

The follow-on window runs **all three models in protocol order** (pplx v1.1 → Clef-Flash → Clef 27B). It replaces the
pplx-only follow-on. Saved smoke cases are reused, and no scored result exists yet.

*Second addendum to amendment 9 (2026-10-09, after the follow-on smokes, before any full block C result).* The harness
defines `exact_actions` for `invoice_processing`, `customer_service` and `security_incidents`, but only
`primary_action` for `agent_trace_observability` (`evals/agent_trace_observability/eval_adapter.py:25`). Its pplx
smoke showed only `primary_action`. So the block C unit of amendment 8 is that workflow's own metric:
- `exact_actions` for the first three;
- `primary_action` for agent_trace_observability.

The Holm family stays 4 workflows × 3 pairs = 12.

## Published analysis (transcription addendum, not a protocol rule)

`scripts/decision13/analyze_block_e.py` and `analyze_block_c.py` are the analyses that produced the study's result,
ported unchanged in logic. Two consequences of publishing only redistributable per-item data:
- TOSEM-DBRD (block E) and three WorkflowEvals workflows (block C) carry no data licence, so only per-pair b/c counts,
  n, accuracies and hashes are published (`results/decision13/restricted_aggregates.json`). Their accuracy tests are
  recomputed from those counts, so the accuracy Holm families and the §7 primary score are reproduced exactly.
- DBRD's three calibration tests need per-item probabilities. The published ECE-10 Holm family therefore covers the
  redistributable decision sets only (24 tests instead of the study's 27). Checked against the study's own analysis:
  every other raw and Holm-adjusted p-value is identical (DBRD's three bootstrap p were 0, so dropping them leaves the
  Holm multipliers of the rest unchanged); only the ECE wins-minus-losses count differs, because it omits DBRD's
  three tests. The ECE count is the §7.3 tie-breaker only, and it was not reached (no two models were within 1 of
  each other on the primary score).
