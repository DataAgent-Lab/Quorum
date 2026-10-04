# Phase 1.0a — Tasks (demo API cache + lazy worker)

> Spec: [`spec_details.md`](./spec_details.md) · Facts: [`verified-facts.md`](./verified-facts.md)
> Conversation: `docs/archive/conversation-phase1.0a-demo-api-cache.txt`

## T0 — Fact-first gate
- [x] T0.1 Trace serve/app.py, space/index.html, quorum/ensemble.py → F§1–F§8.
- [x] T0.2 Measure idle/loaded/unloaded RSS, cold/re-load time, label-order invariance → F§9–F§13.
- [x] T0.3 Confirm per-item benchmark predictions were never persisted; request regeneration (all 7 datasets,
      names + descriptions where available) from the research session via git.

## T1 — Implementation (serve/ only)
- [x] T1.1 `serve/cache.py`: `normalize`, `cache_key`, `build_response`, `MEMBER_DISPLAY_NAMES`,
      `PredictionCache` (SQLite/WAL, get/put/put_many/counts, live cap eviction).
- [x] T1.2 `serve/worker.py`: `ModelWorker` (spawn single-process pool, lazy start, in-flight counter, idle reaper,
      crash → 503 / timeout → 504 + recycle, `close()`); child entrypoints `init_model` / `predict`.
- [x] T1.3 `serve/app.py`: cache-first `/predict`, > 50-labels-only-on-hit rule, `X-Quorum-Cache` header + CORS
      expose, `/health` additions, startup/shutdown wiring, env config.
- [x] T1.4 `serve/seed_cache.py`: `--predictions DIR` importer with validation + accuracy report; `--demo` presets.
- [x] T1.5 `serve/spot_check.py`: live re-run of N dumped items + agreement / Wilson CI / full-set accuracy report.
- [x] T1.6 `serve/requirements-dev.txt` (pytest, httpx) + `serve/README.md` cache/worker/env/seeding docs.

## T2 — Tests (one per AC)
- [x] T2.1 AC-1 hit path: fake worker, assert 0 calls, header `hit`.
- [x] T2.2 AC-2 miss path: computed once, stored `live`, repeat is byte-identical `hit`.
- [x] T2.3 AC-3 permuted labels hit; different message/labels miss.
- [x] T2.4 AC-4 label limits (>50 hit 200, >50 miss 422 no worker call, >200 422, legacy 422s).
- [x] T2.5 AC-5 response equals the Phase 1.0 builder for the same probs (unit).
- [x] T2.6 AC-6 persistence across reopen, no expiry, live cap eviction keeps seed rows.
- [x] T2.7 AC-7 stub worker: idle unload exits the child; next miss restarts.
- [x] T2.8 AC-8 stub worker crash → 503 then recovery; reaper skips in-flight worker.
- [x] T2.9 AC-11 `/health` fields; does not start the worker.
- [x] T2.10 AC-12 demo preset list matches `space/index.html` (drift guard).
- [x] T2.11 REAL E2E (env-guarded `QUORUM_REAL_MODEL_TEST=1`, non-mock): real worker miss vs direct ensemble
      (AC-5), repeat hit identical (AC-2), idle unload → child gone + API RSS < 150 MB (AC-7), demo seed hits (AC-12).
- [x] T2.12 AC-9 REAL: research branch 40fa9e2 (7 datasets × names/descriptions, 47,582 rows) — importer dry run:
      0 validation errors; config check passed; exact accuracy == meta (Δ 0.0000) and == published within ±0.0002
      (fp16-vs-fp32 on bitext/mtop names). Merged to main; production seeded; shuffled 77/113/151-label benchmark
      requests via the public URL are hits in 76–162 ms with no model loaded.
- [x] T2.13 AC-10 REAL: Banking77 descriptions, 30 random items live on this CPU: argmax 30/30, max|Δp| 1.46e-6,
      live accuracy 19/30 = 0.633 (95% CI 0.455–0.781, contains the full-set 0.7737); 3 items per other dataset:
      18/18 agree, max|Δp| ≤ 2.7e-6.

## T2b — Review fixes (independent spec review: REVISE, 9 Major — all addressed)
- [x] T2b.1 Load control: admission cap → immediate 503; one job at a time; queue timeout → 503 without killing the
      running job; job timeout covers the running prediction only; separate cold-start (load) budget; crash backoff.
- [x] T2b.2 Single-flight for concurrent identical misses (AC-17).
- [x] T2b.3 Configuration fingerprint → cache emptied on mismatch; seeder refuses dumps from another config (AC-15).
- [x] T2b.4 Bounded storage: live count cap 20k + 16 KB per-row cap; `QUORUM_CACHE_LIVE=off`; non-fatal cache write;
      per-thread SQLite connections + busy timeout; chunked seed commits (AC-16).
- [x] T2b.5 Lone-surrogate sanitising; ASCII key; per-label-set vector memo bounded in the child; `serve/data/`
      gitignored.
- [x] T2b.6 Measured order invariance at K=77 (F§15) and the spawned cold miss (F§14); fact sheet corrected (F§5, F§13).
- [x] T2b.7 Tests for every fix: AC-14 (hits fast while busy, admission cap, queue timeout), AC-15, AC-16, AC-17,
      load-budget tests. Fast suite 46 passed; real E2E passed (hit 13 ms, API RSS 67 MB, torch never imported).

## T3 — Deploy (must-ask: user go-live OK in this session)
- [x] T3.1 Rebuilt `quorum-api` image (1.0a); running with `QUORUM_CACHE_PATH=/models/quorum_cache.sqlite3`,
      `QUORUM_CACHE_LIVE=persist`, idle unload 900 s, models pinned offline (`HF_HUB_OFFLINE=1`) — user go-live OK.
- [x] T3.2 Demo presets seeded in the container volume (3 rows).
- [x] T3.2b Production cache seeded: 47,221 seed rows (47,218 benchmark + 3 demo presets), 18 s, API kept serving.
- [x] T3.3 AC-13 measured on the host (2026-10-04, load ~5–6): idle API process RSS 59 MB (cgroup anon 41 MB);
      seeded preset via public HTTPS 0.12–0.31 s total (`hit`); cold miss 28.7 s; repeat 0.07 s (`hit`);
      warm miss 1.47 s; worker loaded = 2.27 GB anon; uncached 60 labels → 422 in 0.10 s; CORS + exposed header OK.

## T4 — Docs + handoff
- [x] T4.1 Spec/tasks statuses updated; `[STATUS SUMMARY]` written to the conversation file.
- [x] T4.2 Pushed the 1.0a commits to main (049c620..1c995be) — user: main is managed by this session.
