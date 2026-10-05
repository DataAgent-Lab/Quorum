# Commit hash map (history rewrite of 2026-10-06)

On 2026-10-06 the history of this repository was rewritten once, with `git filter-repo --mailmap`, to give every
commit a single author/committer identity (`A-baoYang <rubyyang0615@gmail.com>`). Only the identity fields
changed: every tree (file content), commit message, author date and committer date is unchanged, and abbreviated
hashes inside commit messages were rewritten by filter-repo to the new hashes. Because a commit hash covers the
identity fields, every commit received a new hash.

Files committed before the rewrite (result metas, provenance documents, the phase-2.0 ledger, script docstrings)
cite the OLD hashes; they are kept verbatim as historical records. Resolve any old hash with this table. The
pre-registration ordering they document (protocol and selection commits before the single test run) is preserved:
the rewritten commits keep their original order, parents and dates.

82 commits, sorted by committer date.

| committer date | old | new | subject |
|---|---|---|---|
| 2026-09-29T01:09:39+08:00 | `73984ca69bf42e898165120141af441753b165d9` | `4d7d98c847a2d70162195bd6cfa8ad56b3314d9b` | Quorum: a jury of tiny open models for cheap, calibrated zero-shot intent classification |
| 2026-09-29T01:16:24+08:00 | `ca87326e0c5914f38da8e67df69627a0113dcbee` | `8fa9c5b53bfbbd47d4b6b13d2c96373ab5eb7871` | docs: add mechanism diagrams (zero-shot ensemble + 24-shot pipeline) |
| 2026-09-29T11:35:33+08:00 | `2bd9154804a76a7c4d80adc664fdc9b1bedfb726` | `33a2b859aed32f2aa18e7d88f4db297eda67d910` | Add the 24-shot pipeline (in-context reader + kNN) and its gated adapter |
| 2026-09-29T11:37:19+08:00 | `fefb72a6f077bb71fdb0dde3638e9b276f3d5560` | `e098ec3e1a5c5de85674a5911a8404b50203ae5b` | docs: link the Hugging Face Quorum collection and gated adapter |
| 2026-09-29T11:57:08+08:00 | `7642b9ad87fed78b78f117bd19c9885a600c0339` | `c7cd9d1379452551c47b9d50a9e14e67f27dc897` | feat(phase-1.0): Quorum live demo Gradio Space (+ SDD phase docs) |
| 2026-09-29T12:25:56+08:00 | `223d5d847c527d290dcb8f920678b27182b072a5` | `5c15a5633359eebc681cb2f0aa5b48b9164c9675` | feat(phase-1.0): pivot to free static Space + self-hosted FastAPI service |
| 2026-09-29T12:28:17+08:00 | `74fc26e0c4811cda840e2fcf952be6c15ee18a31` | `a37fadb4ea4d638942a5852d4b55af9f72f52b5e` | docs(phase-1.0): link live demo Space; mark deploy tasks done |
| 2026-09-29T12:30:09+08:00 | `bc92e1397c5764440ca67ae6d885bd81d03ca5cc` | `76fd07a55e890f52c81a965b2df8c2e4eec8072e` | docs(phase-1.0): status summary — static demo Space live; API self-host is the remainin... |
| 2026-09-29T13:32:49+08:00 | `f7053bda0d1a47d8debb0f866b4791d4b708d8d3` | `389a24ac6f05feef7df9f1e612ef9370ad8159a5` | docs: correct Jev provenance — official eval is a private workflow suite, not public ac... |
| 2026-09-29T13:51:13+08:00 | `0b1d1922922e650d2512b0ef6608478800257797` | `e86cd7b4bda333f6a4c7d3f8aeedab0998feb9e2` | feat(phase-1.1): descriptions as first-class label text (+0.091 on SNIPS, above dataless) |
| 2026-09-29T14:42:21+08:00 | `c315b9c240df30f8c866f86f6c23d5d12769cf96` | `9c49cbdece3c24eaeb6163106af5378696a3c9f2` | feat(service): self-hostable decision service (3-layer) + point demo Space at stable API |
| 2026-09-29T15:04:34+08:00 | `80c972e9dccee3afccf9c7a6b4c6d8e1cc5cb6a2` | `f07f0349ad9c599bcc5e9567f7e58faacc0897d2` | docs(phase-1.1): T2.3 CLINC descriptions (+3.0, significant) + P2 measured-negative (Di... |
| 2026-09-29T16:38:13+08:00 | `e6df2e3f285d3668e844ec9d2710d7020c135186` | `f6d95f78cb1cfa3756880ef02d8f8bdaeee242af` | feat(demo): product-grade demo page + plan Phase 1.2 (train our own intent encoder) |
| 2026-09-29T16:55:30+08:00 | `bb5f0c2bd2e9a64156b9400b0d79f2cf64192b88` | `35d438743905ba38cd63f6edea9d7708b46e9c1d` | feat(demo): remove Advanced/API-URL field (demo just works); correct latency copy (per-... |
| 2026-09-29T16:57:08+08:00 | `144e76a0fb16f5ede17697482f47411d9e4ad8e1` | `b526b2c4b37192e0c8d60ae26791e90b19bc9420` | fix(service): Banking77 zero-shot = this repo's verified 0.756 (not the study's 0.780);... |
| 2026-09-29T17:21:06+08:00 | `23908331aa68125c50265f601cc6dfc449f6c3fd` | `849eae6152573fd197373f981f13bf7c17956c3a` | docs: report zero-shot with generic descriptions (0.774) as primary Banking77 number |
| 2026-09-29T19:45:22+08:00 | `04e670974d9724ac5e7e12cbc59b97658817d741` | `03a520b40cefdb0ba038341de55f31be4051a139` | perf(ensemble): load the NLI in fp32 on CPU (~4-5x faster), fp16 on GPU |
| 2026-09-30T02:25:48+08:00 | `8a23bd4382157bb110c283c6c552281450a7b4d0` | `21bbb59ad576946326ba6095904b760dab257050` | docs(serve): record measured CPU latency (fp32 NLI ~1.5-2s/4 labels, ~4x fp16) + fix st... |
| 2026-10-04T11:22:54+00:00 | `049c620a64e905461bca9245389b0fbc159cb42e` | `e885312336bdfa42e501d860b72cfaedb316d8aa` | docs(references): paper-venue guide for Quorum (journals/conferences/workshops) |
| 2026-10-04T11:44:55+00:00 | `0bd4bc56dd25dc69766083cb153cc9a5a4fbc27c` | `973a037abcc767d2793c7728d32ccc7dd3ba6db1` | feat(serve): seed the cache from benchmark dumps + live spot-check |
| 2026-10-04T11:44:55+00:00 | `a92043e30424869c426d9b1d28cdfd88e2f1a670` | `ad3fd6a0f9d5da9fd484b89d9907f5db0870d351` | test(serve): cache, worker lifecycle, API contract, seeding, real E2E |
| 2026-10-04T11:44:55+00:00 | `e9751a560ca783f8405ce8677dd112cb1d88adc9` | `7a4f0fe4e6a7835d74513b0af23b47c9d9698f8c` | feat(serve): permanent prediction cache + idle-unloaded model worker |
| 2026-10-04T11:48:09+00:00 | `1c995bec4c081819198e8c46696224da692c19c7` | `aee550001e30857b194105407b9676c2cce1e05b` | docs(phase-1.0a): spec, fact sheet, tasks and decision log for the demo API cache |
| 2026-10-04T15:45:38+00:00 | `b382c337b91334c0bd0b1044863796e5b793d7da` | `34d992681519c6d66f6a685c1ee7a0c6aa1c0645` | docs(results): keep the Jev zero-shot reference in banking77_descriptions.json |
| 2026-10-04T16:11:14+00:00 | `43fa5b3bef5de00e1c365dcde6dece9ecdf34d06` | `31fb400d42e3d1a144e4349873acc63633095ace` | docs(phase-1.0a): close the phase — benchmark seed import + spot-check verified, produc... |
| 2026-10-04T17:56:43+00:00 | `07c607ae320e572132076bfcea4669183f5fb23b` | `bdef8ebeeb8c39c3188377e24a3ab714d5ae1ac1` | feat(demo): ask the jury — real benchmark examples, the verdict reveal, cold-start hand... |
| 2026-10-04T17:56:43+00:00 | `3a2b46aef02134854c842fa069bface7a1408ef0` | `8343259951d5649f6a81dbe8db8d6e0b79158776` | feat(serve): export the demo page's real benchmark examples (space/samples.json) |
| 2026-10-04T17:56:43+00:00 | `80d9cc0ac4242fb9f4161cdbd2c5ad387e51e00a` | `90ac47e2dca76a5613e248f2e1fba622a5fdf857` | docs(phase-1.0b): spec, prototype, browser E2E (19/19) and screenshots for the demo red... |
| 2026-10-04T18:32:46+00:00 | `c013b4ef6f18de2190db56216402becf6ccf2e16` | `e4e6672b0f3fb2fa14b13b9f5314475e1201ecbf` | fix(demo): honour ?api= only on a locally served page |
| 2026-10-04T18:47:34+00:00 | `f21a4a06193d6c0f77b92afde7da1fd6dedd5046` | `8f5ed6b6ac8fde72e3925dd4c5a22a3e0160ef7d` | docs(phase-1.0b): live on the Space — published e4e6672, verified 9/9 in a real browser |
| 2026-10-04T19:17:54+00:00 | `1202009b0fd805d0aa10605a7ba1135383be767b` | `3bab84022588c382613e6090f9ca54abfdec3cfd` | docs(references): venue guide — TACL submission rules (no supplements/links, appendices... |
| 2026-10-04T19:17:55+00:00 | `62cd83fc96d5ab21324dbac6a0177ea032f8e2c0` | `69bf2edc701dbdf751c1cdd6fbf0afe8fe3a52ed` | docs(phase-2.0): plan the TACL paper — spec/tasks v2 after independent review, venue ta... |
| 2026-10-04T19:18:49+00:00 | `d454460ae816964e0565ded2a7e589e0d23066a1` | `cb4b0cc637879d2807d8d9aa0edb29c3c6caac90` | docs(phase-2.0): log research-session acknowledgement and R7 preliminary provenance |
| 2026-10-04T19:58:21+00:00 | `698812b91a646bf2dfe7cba6465f7d8c98cfc3d5` | `ec26c7e74b005ab69e570642bc3d3ed3aa872d66` | docs(phase-2.0): G11 outcome — 0.938 provenance verified; clean pre-registered re-selec... |
| 2026-10-04T20:05:37+00:00 | `1b2972347bbbc4fc287255e2ec232455005db9bf` | `aa38079bfff283f852921b2525241bcd4ec7b78a` | docs(phase-2.0): review of the pre-registered clean 24-shot protocol — go with three am... |
| 2026-10-04T20:13:34+00:00 | `bdfe8bbf4895682b14ea6c2f6f15c993d4e11e63` | `2e55eebcf15770bedc0628b49d65f660f50e5b9c` | feat(serve): CORS_ORIGIN_REGEX for per-deploy preview origins |
| 2026-10-04T20:13:43+00:00 | `17dd50839f1ac28a7290d7ca01a006d0fd851d3d` | `bb268626d3da38340a60fa251db26c3201b3211b` | docs(phase-2.0): run-script review for the clean 24-shot protocol — go after three code... |
| 2026-10-04T23:13:17+08:00 | `40fa9e249a50e119e5d356a2c9e9aa1de4505e5b` | `3c442d77a463786a8eda70e5bf8a88ea543b60f5` | data(predictions): per-item benchmark predictions (7 datasets, names + descriptions) fo... |
| 2026-10-05T03:27:42+08:00 | `8bea0c0a8c1e03d0988deb5816ceb1032e1c635a` | `30732fd8bc370e5aca535d4ef4fa8abedf7de0d9` | provenance(R1): how the Banking77 24-shot 0.938 configuration was selected |
| 2026-10-05T04:01:41+08:00 | `8455e7e2290d9ac96b014fc3767ad65303e5261e` | `0c8602fd8d2e9f563263a932e3699c0abd50ddbb` | provenance: pre-register the clean Banking77 24-shot selection protocol |
| 2026-10-05T04:11:42+08:00 | `9363066ad7cc5193a5f0556a3d0ab7f68f204ebe` | `a301132fb8796068d57d068b047ce8dc290c3960` | feat(clean-24shot): run script for the pre-registered protocol + library hooks |
| 2026-10-05T04:11:42+08:00 | `f8f0476199514acd0905de8a33439bcf9b3b597a` | `22a9b68468679bc889239a9b079d1eb0a1380b77` | provenance: protocol addendum A1-A3 (no truncation, pre-declared secondary analyses, de... |
| 2026-10-05T04:14:14+08:00 | `24230069772cc47cd3f4aff007c001303f0558b6` | `a4c803bed2d12bf373b7bf83c9349e996df7a008` | fix(clean-24shot): F1 pred from unrounded scores, F2 single-run guard, F3 selection/cod... |
| 2026-10-05T04:16:09+08:00 | `8df80ae0b31556980c3b0cfdb2adef0900ea2aea` | `fc096810da27cc4e504b0cbc036e8c8aa0a4568a` | chore(clean-24shot): record loaded model/dataset revisions (protocol §1) and git HEAD w... |
| 2026-10-05T05:26:38+00:00 | `39e59328c41ec0742c484eedf93cab6f90bdeff0` | `f2f479e41f31e757f369da4ffc263122eefe8832` | feat(paper): vendor the official TACL style; zero-shot tables/macros generated from the... |
| 2026-10-05T05:26:38+00:00 | `a14954c907f1b79e8308ff3d87df04b7dbc63d9e` | `1f02bbeb07bac1eda7aab91e73a0d6027c63aa1d` | feat(metrics): unrounded exact McNemar p (mcnemar_exact_p) + calibration metrics with t... |
| 2026-10-05T05:27:49+00:00 | `b004a3dedcd6db81d82ca70c847b661a58b7a957` | `e2d7d76e89d4e5295630b912be6216f676fd379e` | docs(phase-2.0): claims ledger — zero-shot numbers + calibration verified, 24-shot pre-... |
| 2026-10-05T05:30:47+00:00 | `783f4fbc52d6746a619f0a205185a4342475880a` | `b17474efd48e9fb124241da98fcb0e1a0bd05782` | docs(phase-2.0): novelty memo + candidate bibliography; reviewer-risk tasks (combiner a... |
| 2026-10-05T06:15:42+00:00 | `4eb0b4132a97615660dd509e80f84fdcc9cae682` | `50b86f2ccfbbbfec0a1b5fa99df9c14d9fa86783` | docs(phase-2.0): log R12 pre-registration check and R10/R11 approvals |
| 2026-10-05T06:25:33+00:00 | `919eede567377ac8bd1f3b8cbbdb7f38cf941fda` | `1779c0f7dbc3c2fa4b235610c3bab0ca2eacaf45` | feat(paper): compliance checks (layout, anonymity, numbers, bibliography) with planted-... |
| 2026-10-05T06:25:49+00:00 | `722cb2f27e9dd8e2737e5da29bdf447737e526f1` | `c3ac19c3e23225dbf319b6364ff7b7fa7e1488d8` | docs(phase-2.0): log compliance tooling decisions; tick T7.0/T7.6/T0.4 |
| 2026-10-05T06:48:07+08:00 | `d513aa9727500f0ddba93af3300dacfd96bb5416` | `fcfc351e149eec5d48189b20def181351fc3a524` | results(clean-24shot): selection split + selection scores, committed before the single ... |
| 2026-10-05T07:03:11+00:00 | `4f3270247157299ce1b2da290d979b867ea01ac9` | `d2e98068861a30bce01be5803ae831398273adb3` | merge research/phase2.0-paper-evidence: verified paper evidence (R1 provenance, clean p... |
| 2026-10-05T07:04:30+00:00 | `5e7eed93f97d410bb8893158ad84cddc642c6401` | `4440c5f712158a7df3ea4b076b6b336add5a4843` | feat(paper): combiner ablation + member calibration from R3 member probabilities (claim... |
| 2026-10-05T07:22:12+08:00 | `09217ba1ce9c20ca91ad60e1bc57d81228008b99` | `66f5768c81e648c4423c2c18f69dde469d9d41b9` | results(clean-24shot): the single pre-registered test run — 0.9357 vs the Jev reproduct... |
| 2026-10-05T07:56:30+00:00 | `08344124345f16844d88c5692ffe03308dacd668` | `73f105016841777ee57e591d12644c667bedea9b` | merge research/phase2.0-paper-evidence: R12 train-side zero-shot outputs (pre-registere... |
| 2026-10-05T07:57:01+00:00 | `b93b37d21e1e1b55af2d78c92f7ec46d6e3f89e8` | `dc3c66d4b72ba69f0467bb127f9a6e5a79c3aa3f` | prereg(paper): confirmatory calibration analysis declared before running — one global t... |
| 2026-10-05T08:01:33+00:00 | `663b684569bb6100776756a5b2bf867afc8bb415` | `4353031fb970da46c51eda75e0006d54d1a0f737` | results(paper): confirmatory calibration — one train-only global temperature (T*=0.508)... |
| 2026-10-05T08:19:10+00:00 | `995efe6748497676a80341d9eb4e3b82af123e6c` | `e616dfce3713090bd063e93295e5b89dc9fff90c` | merge research/phase2.0-paper-evidence: R5 24-shot ablations |
| 2026-10-05T08:19:48+00:00 | `1454d48ea3e67ba0e80d81cc910634b8cb22cafb` | `b5809d5de3a960675e9b2cd1b94a0746c908aab6` | docs(phase-2.0): R5 ablations verified — kNN alone matches Jev; reader adds +0.55 pt (p... |
| 2026-10-05T10:15:49+08:00 | `d796121041cb9058d0fe3e29861878b96c3c8962` | `155b724cb27e67d25d6330c326c6a69a8e0b5697` | feat(R2): per-item dump of the repo-default 24-shot configuration (pinned revisions, tr... |
| 2026-10-05T11:37:25+08:00 | `5e0f1b3d43ee7180d58bd439d7058ad2bcf46e6b` | `cb66ad29dc3ca22252a3a5eef26e745a47f1c83c` | results(R2-default): per-item dump of the repo-default 24-shot config + descriptive com... |
| 2026-10-05T12:14:46+08:00 | `7c6212af92abfc509a75c1724517e6d92fd1f6c2` | `0d6e2293ca313600db65896c29bd8085a4cd350f` | provenance(R4, R7): adapter training record + description authoring log |
| 2026-10-05T12:45:29+00:00 | `fdf3535f2eee50b22ccacec5ba3e8abc41f8c6d3` | `43d4bb5f481b31f6a469e3b24776c34fc7caafec` | merge research/phase2.0-paper-evidence: R6 reproduction-parity retrieval runs |
| 2026-10-05T12:48:01+08:00 | `cce33f4424b05bdd073d7c2a294fb66b9eaecc6d` | `691d2042cc227758ad19e54bac4e7187f49fe6f2` | feat(R5,R6,R8,R9): paper evidence runs — ablations, reproduction-parity retrieval, orde... |
| 2026-10-05T12:49:35+00:00 | `c44793439377b0d9b67d70c0f2da8e96940a1b33` | `c7dd24030c08fdc587e8e92c7b8d101cbddeca3f` | prereg(paper): paired calibration comparison with the Jev reproduction's per-item proba... |
| 2026-10-05T12:51:53+00:00 | `1c3fa025b927250f010406c52ef59265facaf905` | `9f21bc20e250ae9e81a238d96b2d4637ef18926c` | results(paper): R6 verified — accuracy gain is retrieval-driven (parity ties Jev); pair... |
| 2026-10-05T13:07:19+00:00 | `2908ca5ede8381415009752fd330da29b034142b` | `b683d2679adbec3bf08b549b901f89ff0082ff4c` | docs(phase-2.0): retrieval tie-breaking is platform-dependent (427 set / ~1,760 order d... |
| 2026-10-05T13:13:51+00:00 | `1df260236a09e71097d327bc46573e9874a0f320` | `c0195c70b8a20ed06ed9996d9c13fb757ae13a84` | docs(phase-2.0): F10 tie-break caveat independently confirmed; R6c queued with a cross-... |
| 2026-10-05T14:02:10+00:00 | `e5b27c69a5614eada9809bd8d8b70bde297973b7` | `86790443a285b8943de56dfe38a3b0834bcc696a` | docs(readme): ~120 ms/item is GPU latency, not CPU |
| 2026-10-05T14:13:26+08:00 | `7c0bed17d5fba964ea6141ed98705512998f54c0` | `9a4bdd9710f63bd391b1b20bfcce14c9e0df998c` | feat(R12): train-side zero-shot outputs for a global temperature (ids phase writes the ... |
| 2026-10-05T14:13:40+08:00 | `3ee8fee85d0a3189e026d58d95bc998637e236f2` | `ffc298e34192ae5a8ce30667c3a9d5276cadde08` | prereg(R12): commit the 7x1,000 train-side sample ids + seed 20261008 before any R12 run |
| 2026-10-05T14:19:00+08:00 | `44c8413ac7ac15b8b5bbaeeb337fe7f04112de58` | `b457b51fb61932af99869ee73119687e80b2f9f1` | feat(R10,R11): LoRA upper reference (marker scoring, S for early stopping only) and rer... |
| 2026-10-05T14:59:46+08:00 | `20a6911b01a5cc8f44ea52abe93913c0d1207c2b` | `0ba107a2fe62566fa874c7cf0779233698f47a8b` | data(R3): per-member zero-shot distributions (member_probs) for all 7 datasets x {names... |
| 2026-10-05T15:00:43+08:00 | `e9e5509691ecef9827bfac31e34a9f376a6fe9e3` | `9c3da0bcb2c550520425284d7d693dd4532971a0` | chore(R3): correct row count in 0ba107a message — 47,582 rows per side (2 x 23,791), no... |
| 2026-10-05T15:55:31+08:00 | `a416f0fe01c84e7309e33791f0bfda3340c0819b` | `39d49cc82c061c67f24cc0fa70f7fdda14aa3060` | data(R12): train-side zero-shot outputs on the pre-registered 7x1,000 sample (ids commi... |
| 2026-10-05T16:18:20+08:00 | `d02cd680a671f6719e4cbbc0ae45db4647dd5168` | `fc6c07a892ce0a0b1c66e2b548fac5b7427ab7b6` | data(R5): ablations of the clean 24-shot run (reader/kNN alone, BM25 label votes, base ... |
| 2026-10-05T20:44:20+08:00 | `c9b7fbfb0b329a1f51109386e57adda11436c85d` | `fc9c65aa16202d12acb425e8662915f0e70043b4` | data(R6): parity runs with the reproduction's retrieval rule (word+bigram BM25, <=4 per... |
| 2026-10-05T20:58:55+08:00 | `b9a2f857334879278e2238eb313cffc96b6fbd83` | `cebfa25dc2e26b1f21cb08f581fbe4bf2f93046c` | feat(R6b): post hoc retrieval-mechanism runs separating per-class cap, tokenisation and... |
| 2026-10-05T21:13:12+08:00 | `0069ab99f68711de2e5ecf9b7684de1944a7387f` | `4e13f671cd6bb78687e80244d02b5b6381d09096` | feat(R6c): post hoc sensitivity of the clean run to BM25 tie-breaking |
| 2026-10-06T00:19:45+08:00 | `9b60b9e0138b9088b89191e1db39609d623cc04c` | `a2d1a1b120110bb3e19a1b56c2a256af1c1c225d` | data(R9,R8): option-order seed robustness and latency |
| 2026-10-06T00:52:40+08:00 | `e2c39d120631f2465af6a3ac2551ae75268b8cdb` | `b7615125abfc6ae313cba72b6ba106e25960d2f6` | data(R11): bge-reranker-v2-m3 as a zero-shot classifier and as a 4th ensemble member (7... |
