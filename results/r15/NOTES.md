# R15 notes (pplx-decider v1.1)

- `b24` ran on the release code (A6). All other cells (`zs`, `defs`, `train`, `b24B`) were scored by the patched
  SGLang engine under A7. Their files carry `_sgl`, and every summary records the engine and its parity gate.
- **`defs` McNemar field — do not use.** `summary_pplx_decider_defs_sgl.json` has a `mcnemar_vs_reproduction` field
  (b/c 21/236). It compares against the reproduction's **24-shot (retrieved24)** per-item Jev hits, not against a
  Jev run on the `definitions` bodies, so it is not a valid comparison. The same applies to R14 `defs` once it runs.
  The equal-request Jev comparison for `defs` is the zero-shot `repro_definitions` arm
  (`docs/phases/2.0/jev/zeroshot/banking77_repro_definitions.jsonl.gz`).
- `host` in the engine-mode summaries was redacted. The client ran in the GPU host's network namespace, so the
  field held the machine name.
- Exposure (A3): MASSIVE, MTOP and SNIPS appear in the training manifest. HWU64 is indirectly exposed (overlap with
  MASSIVE). Banking77, CLINC150 and Bitext are "not documented".
