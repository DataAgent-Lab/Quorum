# Jev latency at long inputs (2026-10-14)

User request, for comparison with pplx on one GB300. Script: `scripts/jev/latency_probe.py`. State: newline-joined
public Bitext training utterances; one 4-option choice question. Client: GCP VM in asia-east1 (Taiwan), public
internet, a new TLS connection per call (GET to the API host: 0.21–0.25 s total). Records hold latency, status,
input tokens, served model (jev-1.13.0 throughout) and error text; never answer text.

- `calls.jsonl.gz`: first run. Bodies at each size were identical across repeats and concurrent calls.
- `calls_distinct.jsonl.gz`: rerun with every body distinct (a fresh seeded shuffle per call; `seed` per record).
  Three passes in one file:
  1. `char_scale` absent, seeds 1–37: all sizes. Shuffled text has about 7% more tokens per character, so every
     top-size body (about 33.6k tokens) was refused (`max_tokens_exceeded`); sizes 1k–24k and the over-limit calls are valid.
  2. `char_scale` 0.93, seeds 1–26: top size only. **Superseded**: seeds restarted at 1, so these bodies shared
     prefixes (up to 24k tokens) with pass-1 bodies.
  3. `char_scale` 0.93, seeds 1001–1026: top size only; seeds used by no other call. **Valid top-size set.**

Distinct-body medians (s): 1k 0.296 · 4.3k 0.301 · 8.8k 0.295 · 17.8k 0.352 · 26.0k 0.424 · 31.2k 0.470;
5 in flight at 31.2k 0.536; 16 in flight 0.714 (all 200). Over-limit (~33k, ~40k): HTTP 400 `max_tokens_exceeded`.
No sign of response caching: distinct bodies are as fast as repeated ones.
