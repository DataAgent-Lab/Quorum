# Phase 1.0a — Demo API: permanent prediction cache + lazy, idle-unloaded model worker

> **Status**: in progress (implementation + fast/real tests done; seed import + spot-check await data; deploy awaits
> user go-live) · **Projected-date**: 2026-10-06 · **Predecessor**: Phase 1.0 (live demo, `serve/`)
> **Dependencies**: per-item benchmark predictions from the research session (`results/predictions/*`, via git) for the
> seed import + accuracy verification (AC-9, AC-10). Everything else is independent of that data.
> **Facts**: every claim cites [`verified-facts.md`](./verified-facts.md) (`F§n`).
> **Review**: independent spec review (2026-10-04) returned REVISE (0 Critical, 9 Major, 10 Minor); all Major items
> are addressed below (see conversation DECISION #6).
> **Follow-up**: Phase 1.0b = demo frontend redesign (7 benchmark datasets as real examples, description labels,
> results-panel polish). Out of scope here.

## Goal

The demo API must answer fast and hold almost none of the shared host's resources while idle:
1. A **permanent** (no TTL) prediction cache. Hits never touch the models.
2. Models load **lazily** on the first cache miss and are **released when idle**.
3. The cache is **seeded** with the benchmark-set predictions (all 7 datasets; names + descriptions variants) and
   the current demo presets.
4. The benchmark predictions are **verified**: exact full-set accuracy vs the dump's own meta and the published
   `results/*.json`, plus a live CPU spot-check that the served code reproduces the dumped predictions.

## Design

### D1 — Cache (`serve/cache.py`)
- Store: SQLite file (`QUORUM_CACHE_PATH`, default `serve/data/quorum_cache.sqlite3` — gitignored; in Docker on
  the `quorum-models` volume). Table `predictions(key PK, response, source ∈ {seed, live}, created_at)`. WAL mode;
  **one connection per thread** with a 5 s busy timeout, so readers never wait on writers and an offline seeding run
  and the API can write concurrently. No TTL.
- **Configuration fingerprint** in a `cache_meta` table: key version, member model ids, template, cosine scale,
  combine rule, CPU NLI dtype. Opening the cache with a different fingerprint **empties it** (logged; re-seed needed)
  so an answer from an older configuration can never be served. Residual risk (documented): model *revisions* are
  not pinned by `quorum`; a hub update of the same model id would not change the fingerprint.
- `normalize(message, labels)` = the rules of F§3 (strip; drop empty + duplicate labels, keep first), plus lone
  UTF-16 surrogates replaced by `?` (previously a 500). Shared by API and seeder (SSoT).
- `cache_key` = SHA-256 of ASCII canonical JSON `{"v":1,"m":message,"l":sorted(labels)}` over normalised inputs.
  Sorting is safe: results are label-order invariant to ~1e-9 for K=6 and K=77 (F§11, F§15). Message matched
  exactly after strip (models are not guaranteed case/whitespace invariant).
- `build_response` = the exact construction of F§4 (moved out of `app.py`; API and seeder share it).
  `dump_response` serialises exactly like FastAPI's `JSONResponse`, so a hit is byte-identical to the original miss.
- Bounded growth of **live** rows: at most `QUORUM_CACHE_MAX_LIVE` (default 20 000; oldest live evicted first) and a
  live body over 16 KB is not stored → worst case ≈ 320 MB. **Seed rows are never evicted or size-capped.** Seed
  imports commit every 2 000 rows.

### D2 — Model worker (`serve/worker.py`)
- Inference runs in **one child process** (`ProcessPoolExecutor(max_workers=1)`, `spawn`) whose initializer builds
  `ZeroShotEnsemble(device=QUORUM_DEVICE)`. The API process never builds a member → never imports torch → stays
  ≈ 52–67 MB (F§6, F§9, F§14). In-process unloading was measured insufficient (F§12).
- Started on the first cache miss. **Cold miss ≈ 37–82 s** depending on host load and page cache (spawn + imports +
  load + first inference, F§14). Right after spawning, a warm-up job runs under its own
  `QUORUM_LOAD_TIMEOUT_SECONDS` budget (default 300), so a slow cold start is never killed by the per-job timeout
  (which would leave the worker cold forever); a load over budget → 504 + 30 s backoff.
- **Idle unload**: a reaper stops the child when no request is running or waiting and the last use is older than
  `QUORUM_IDLE_UNLOAD_SECONDS` (default 900; `<= 0` disables). The child is joined, so memory really returns to
  the OS. (A miss arriving while an old child is still exiting may briefly overlap two children; accepted.)
- **Load control** (public endpoint, small shared host):
  - one job runs at a time; at most `QUORUM_MAX_PENDING` (default 4) requests are admitted (running + waiting);
    beyond that → `WorkerBusy` → **503 immediately**, so misses can never occupy the request thread pool and stall
    cache hits;
  - a waiting request gives up after `QUORUM_QUEUE_TIMEOUT_SECONDS` (default 60) → 503, **without** disturbing the
    running job;
  - `QUORUM_JOB_TIMEOUT_SECONDS` (default 120) bounds the **running** prediction only (the cold start has its own
    budget, above) → 504 + that child is killed and recycled;
  - a crash or failed model load → 503, and new jobs fail fast for 30 s (no re-load storm).
- The child bounds each embedder's per-label-set vector memo (cleared above 256 sets).
- Killing a hung job uses `ProcessPoolExecutor._processes` (private, stable across CPython 3.8–3.13; documented).

### D3 — API changes (`serve/app.py`) — backward compatible for every previously accepted request
- `POST /predict` body and success response shape **unchanged** (F§7). New header `X-Quorum-Cache: hit|miss`
  (exposed via CORS `expose_headers`).
- Label limits (user decision): request schema accepts up to `MAX_LABELS_REQUEST = 200` labels (CLINC150 = 151).
  **More than `MAX_LABELS = 50` labels — counted after normalisation — is served only on a cache hit**; a miss with
  > 50 labels → `422 {"detail": "…at most 50 labels can be computed live…"}` before any model work. Previously every
  > 50 request was a 422 (F§2), so no accepted request changes behaviour.
- Unchanged: empty message / < 2 distinct labels → 422 string detail (F§3).
- Miss path: **single-flight** (concurrent identical misses share one computation) → worker → `build_response` →
  stored as `live` (before waiters are released) → returned. A failed cache write is logged and the answer is still
  returned. Busy/crash → 503, running-job timeout → 504; failures are never cached.
- `QUORUM_CACHE_LIVE`: `persist` (store live results — their message + labels are then kept in the cache file
  indefinitely, within the cap) or `off` (only seeded rows are cached). **Default for the deployed demo = user
  decision (BLOCKED #2).**
- `GET /health` keeps `ok`, `service`, `members` (same model ids) and **adds** `model_loaded` and
  `cache: {seed, live}`. It never loads the models.

### D4 — Seeding + verification tools
- **Dump contract** (agreed with the research session): `results/predictions/{dataset}_meta.json` with `raw_labels`,
  `labels_names`, `labels_descriptions` (exact strings fed), `members`, `template`, `cosine_scale`, `combine`,
  `nli_dtype` (= fp32), `quorum_git_commit`, versions, `accuracy_{variant}`; and
  `{dataset}_{variant}.jsonl.gz`, one item per line `{idx, text, gold, pred, probs[K] (6 dp, label order),
  member_picks[3] (indices)}`, variant ∈ {names, descriptions}. Probabilities computed with fp32 NLI on GPU
  (expected GPU-vs-CPU noise ~1e-5).
- `python -m serve.seed_cache --predictions results/predictions [--dry-run]`: **refuses** any dump whose meta
  configuration differs from the server's (members, template, cosine scale, `nli_dtype == fp32`); refuses label
  strings a client could not send (not normalised); validates every row (K probs, `pred` = argmax, picks/gold in
  range, non-empty text) — bad rows are reported and skipped, exit code 1. Each item is stored under the key of
  `{message: text.strip(), labels: meta labels}` with `build_response(labels, probs, member_picks)`. Report per
  dataset/variant: items, unique keys (duplicate test sentences collapse), exact accuracy, Δ vs meta accuracy, Δ vs
  published number.
  - Published mapping: `names` → `results/{dataset}.json:ensemble_accuracy`; `descriptions` →
    `results/{dataset}_descriptions.json:ensemble_accuracy_descriptions` (or `ensemble_accuracy`). No published file
    → reported as n/a (no gate on it).
  - Known nuance: 3 Banking77 test texts start with "\n"; the dump scored the raw text while the key uses the stripped
    text, so those seed answers may differ in the 4th–5th decimal from a live computation. Accepted.
- `python -m serve.seed_cache --demo`: computes the current demo presets (constant `DEMO_EXAMPLES` in `serve/`,
  drift-guarded against `space/index.html`) and stores them as seed.
- `python -m serve.spot_check --dataset banking77 --variant names --n 30 --seed 0`: re-runs N random dumped items
  live with the served code path; reports argmax agreement, max |Δp|, live accuracy on N with a 95 % Wilson
  interval, and the exact full-set accuracy from the dump vs the published number.
- 1.0b will build its sample sentences + label lists from the **same** meta/dumps (so the strings sent are exactly
  the seeded ones); the shuffled-label HTTP test below proves a client sending meta labels in any order hits.

## Acceptance criteria

| # | Criterion | Measurement |
|---|-----------|-------------|
| AC-1 | **Hit path never loads models**: cached request → stored body, `X-Quorum-Cache: hit`, no worker call | unit + real E2E |
| AC-2 | **Miss path**: computed once, stored `live`, returned `miss`; repeat → byte-identical body, `hit` | unit + real E2E |
| AC-3 | **Order-insensitive key**: permuted/whitespace-padded/duplicated labels hit the same entry; a different message or label set misses | unit |
| AC-4 | **Label limits**: > 50 (after normalisation) + hit → 200; > 50 + miss → 422 string detail, no model work; > 200 → 422; exactly 50 → computed; legacy 422s unchanged | unit |
| AC-5 | **Backward compatibility**: success body equals the Phase 1.0 construction for the same probabilities; served answer equals the ensemble's own answer | unit (vs legacy builder) + real E2E |
| AC-6 | **Permanence**: entries survive reopen; no expiry; live cap evicts oldest live only; seed never evicted; live never overwrites seed | unit |
| AC-7 | **Idle unload frees RAM**: after the idle window with nothing running/waiting the child exits; API-process RSS < 150 MB with torch never imported; the next miss starts a new worker | unit (stub worker, real processes) + real E2E |
| AC-8 | **Failure isolation**: crash / failed load → 503 then recovery after backoff (fast-fail meanwhile, no re-spawn); running-job timeout → 504 + recycle; a cold start slower than the job timeout still succeeds; a cold start over the load budget → 504 + backoff; failures not cached | unit (stub worker) + API unit |
| AC-9 | **Seed import**: every provided dump imports with 0 validation errors; config mismatch / non-sendable labels refused; exact accuracy per dataset/variant equals meta accuracy ±0.0005 and published ±0.001 where published; a client sending meta labels (any order) hits | unit (synthetic) + REAL on the dumps |
| AC-10 | **Spot-check (user ask)**: 30 random Banking77 items (names) re-run live on this CPU: argmax agreement ≥ 29/30 and max |Δp| < 1e-3; the published full-set accuracy lies inside the 30-item 95 % Wilson interval (reported as a sanity check — n=30 is ±~0.15); plus 5 items per other dataset/variant for consistency | REAL CPU run |
| AC-11 | **Health**: Phase 1.0 fields kept; adds `model_loaded` + cache counts; never loads models | unit |
| AC-12 | **Demo presets cached**: after `--demo`, each `space/index.html` preset is a hit; list drift-guarded | unit + real E2E |
| AC-13 | **Deployed demo** (after user go-live OK): container runs the new code, cache on the persistent volume; idle container RSS ≤ 200 MB; a seeded request answers < 100 ms server time | real, measured on host |
| AC-14 | **Hits stay fast under load**: while a miss is computing, a hit returns < 0.5 s; admission cap → immediate 503; a waiting request's timeout → 503 without killing the running job | unit |
| AC-15 | **No stale answers**: a cache opened with a different configuration fingerprint is emptied; dumps from another configuration are refused | unit |
| AC-16 | **Bounded storage**: live rows capped by count and per-row size (16 KB); `QUORUM_CACHE_LIVE=off` stores nothing live; a failed cache write still returns the answer; concurrent seeding + live writes do not error | unit |
| AC-17 | **Single-flight**: N concurrent identical misses → 1 computation, N identical 200s | unit |

Performance targets: cache hit < 20 ms server time (measured 2.4 ms in real E2E); idle API-process RSS < 150 MB
(measured 67 MB).

## Out of scope
- Any frontend change (Phase 1.0b), including how the frontend handles a ~37 s cold miss.
- Authoring descriptions for HWU64 / MASSIVE / MTOP / Bitext (the research session, approved by user — BLOCKED #1 → A).
- Changes to `scripts/eval_zeroshot.py` or `results/` (research outputs: produced by the research session, reviewed + merged here).
- Pinning model revisions in `quorum` (residual risk noted in D1).
