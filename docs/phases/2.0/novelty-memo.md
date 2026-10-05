# GeoJury: novelty, positioning and related-work map (T3.0 / T3.1)

Compiled 2026-10-05 for the TACL submission. Bracketed keys such as `[yin-etal-2019-benchmarking]` point to `refs-candidates.bib`. Every key was resolved against its ACL Anthology id, arXiv id or DOI, and its title was checked. Web sources were checked with an HTTP 200 URL fetch.

**How closely each paper was read:**
- **Read**: I read the text, its tables, or a summary of the body.
- **Abstract**: I read the abstract only.
- **Known**: standard background; I resolved the identifier but did not re-read the paper.

Do not cite a number from an **Abstract** or **Known** item until it has been checked in the paper.

---

## 0. Bottom line (read this first)

**The zero-shot ensemble is built from known pieces.**
- NLI zero-shot classification [yin-etal-2019-benchmarking].
- Embedding and label-description zero-shot intent detection [hu-etal-2024-exploring; sung-etal-2023-pre].
- Log-linear pooling / product of experts [genest-zidek-1986-combining; kittler-etal-1998-combining; hinton-2002-poe].

**The closest single paper** is Hu et al. (NLP4ConvAI 2024) [hu-etal-2024-exploring]. It already does training-free zero-shot intent classification with **bge-large-en-v1.5 plus per-intent descriptions** on SNIPS, CLINC, MASSIVE and ATIS.

**What I found no prior work doing**, all four together:
- a training-free pool that mixes architectures: an NLI cross-encoder plus bi-encoder embedders, with no per-task weights or temperature;
- evaluated on 7 intent datasets using full official test splits;
- with paired exact McNemar tests against the best single member;
- with an analysis of when label descriptions help (opaque label names).

**BTZSC (ICLR 2026)** [aarab-2026-btzsc] benchmarks the same model families on official test splits: NLI cross-encoders, embedders including bge-large-en-v1.5, rerankers and LLMs. Its tasks include Banking77 and MASSIVE intent. It reports **no ensembles and no significance tests**. That is our opening, and also the paper reviewers will compare us against.

**The retrieval-augmented ICL design is also built from known pieces.**
- Retrieval-augmented ICL for many-label intent classification [milios-etal-2023-context; bertsch-etal-2024-long-context-icl].
- kNN over LM outputs [xu-etal-2023-knn-prompting; shi-etal-2022-nearest].
- Meta-training on other tasks before ICL [min-etal-2022-metaicl].
- Scoring all options in one forward pass [robinson-wingate-2023-mcp].

**What is genuinely new and now backed by numbers** is a pre-registered, matched-protocol comparison of an open-weights system against a closed commercial "calibrated decision" system, paired item by item on Banking77:
- **Accuracy:** 0.9357 vs 0.9240; exact McNemar b=119, c=83, **p=0.0136**.
- **Calibration:** ECE 0.0105 vs 0.034.
- As of 2026-10-05, no Jev clone reports a 77-way Banking77 number at or above 92.40%.

**Novelty verdict:** this is an empirical and methodological contribution, not a new method. It counts as "substantial, original" for TACL only if framed as three things:
1. a controlled, pre-registered study of when training-free pooling of heterogeneous sub-1B models beats its members, and how close retrieval-augmented small models get to a closed system on both accuracy and calibration;
2. measured negative results;
3. evaluation-methodology lessons. We have a measured case where the conclusion flips: the repo default is not significant against Jev (p=0.066) but the clean pre-registered run is (p=0.0136).

---

## 0b. New in-house facts to fold into the positioning (from coordinator, 2026-10-05)

| Fact | Where it sits | Positioning consequence |
|---|---|---|
| Clean pre-registered 24-shot run: **0.9357** on Banking77 (3,080 test) vs Jev reproduction **0.9240**. Exact McNemar **b=119, c=83, p=0.0136** | C4 headline | This is a **paired** test on identical items, possible because the reproduction publishes `results/predictions.csv`. It beats the community's only comparison point. It is also within 0.1 pt of the published full-data fine-tuned BERT (93.66 [casanueva-etal-2020-efficient]), with 4B open weights and no Banking77 fine-tuning. Report the discordant counts, not only p. |
| Repo-default configuration: **0.9321**, **not significant** vs Jev (**p=0.066**) | C6 methodology | This is the concrete example the methodology section needs. The significance verdict depends on configuration choice, which is why the headline must come from the pre-registered clean protocol [van-miltenburg-etal-2021-preregistering; perez-etal-2021-true-few-shot]. Report both numbers openly. Hiding 0.9321 would look like selection. Reporting it is the strongest argument that the 0.9357 was not cherry-picked, **provided the pre-registration's timestamp precedes the test run**. |
| Zero-shot ensemble is **systematically under-confident**. Banking77: ECE **0.22**, mean confidence **0.54**, accuracy **0.76** | C1 / calibration (f) | **Confirms risk R3.** The members use T=1 softmax over cosine scores, which yields flat distributions, and an equal-weight geometric mean (weights summing to 1) keeps the pool flat. Rescaling summed log-probs by one global constant does not change the argmax, so a calibration fix costs **zero accuracy**. Options: (i) a true product of experts (weights of 1 each, i.e. unnormalised sum of log-probs), which sharpens; (ii) one global temperature fitted once on a *different* dataset and frozen. Either keeps "no per-task calibration" honest. State the under-confidence plainly; it is a finding, not a bug to hide. |
| Clean 24-shot run is **well calibrated**: ECE **0.0105** vs Jev **0.034** | C4 | We beat Jev on **its own marketing axis** ("calibrated probabilities" [typesafe-2026-jev]), not only on accuracy. This removes risk R8's "accuracy-only" objection. Still needed: a bootstrap CI on the ECE difference (paired resampling of items), the binning scheme (15 equal-width bins? adaptive?), Brier/NLL as proper scoring rules, and the exact Jev probability field used. |

---

## 1. Closest prior work per contribution

### (a) NLI / entailment-based zero-shot classification

| Work | What they did | How we differ (blunt) |
|---|---|---|
| Yin, Hay & Roth 2019 [yin-etal-2019-benchmarking] (**Known**) | Framed zero-shot text classification as entailment, with hypotheses like "this text is about ⟨label⟩". Also used label definitions (WordNet) as hypotheses. | Our NLI member *is* this method, with a newer model (PrismNLI-0.4B), and the template is theirs. The scorer is not novel. |
| Ma et al. 2021 [ma-etal-2021-issues] (**Known**) | Entailment-based zero-shot classifiers rely on spurious lexical overlap and are unstable. | Motivates pairing NLI with members that have different errors. |
| Zhang et al. 2020, DNNC [zhang-etal-2020-discriminative] (**Abstract**) | Few-shot intent detection (incl. CLINC150) that transfers NLI into a pairwise utterance-to-example nearest-neighbour classifier. | The NLI-for-intent precedent. DNNC trains a pairwise model and is few-shot; ours is training-free label matching plus pooling. |
| Xia et al. 2021 [xia-etal-2021-incremental]; Comi et al. 2023, Z-BERT-A [comi-etal-2023-zero] | Entailment formulations for incremental and unknown-intent detection. | Same scoring family. |
| TARS [halder-etal-2020-task]; PET [schick-schutze-2021-exploiting]; Sainz et al. 2021 [sainz-etal-2021-label] | Task-aware cross-encoders; cloze verbalizers; verbalization plus entailment. | Background for verbalized label text. |
| Laurer et al. 2023 [laurer-etal-2023-less; laurer-etal-2023-universal]; tasksource [sileo-2024-tasksource]; Gera et al. 2022 [gera-etal-2022-zero]; Clarke et al. 2023 [clarke-etal-2023-label-agnostic] | "Universal" NLI zero-shot classifiers; self-training; label-agnostic pre-training. | The NLI baselines reviewers expect. **Check each one's training mix for intent data.** |
| Jung et al. 2025 [jung-etal-2025-prismatic] (**Known**; HF card [jung-2025-prismnli-card]) | Source of PrismNLI-0.4B: DeBERTa-v3-large [he-etal-2021-debertav3] trained on gradient-diversified synthetic NLI. | Used off the shelf. "Synthetic NLI only, no intent data" is a useful contamination argument, but it is not our contribution. |
| **Aarab 2026, BTZSC** [aarab-2026-btzsc] (**Read**) | 22 zero-shot datasets on official HF test splits, including **banking77 and massive_intent**. 38 checkpoints across NLI cross-encoders, embedders, rerankers and LLMs. Intent macro-F1: Qwen3-Reranker-8B ≈ 0.70, gte-large ≈ 0.59, bge-large-en-v1.5 ≈ 0.58, best NLI ≈ 0.45. NLI cross-encoders plateau as they scale. **No ensembles, no significance tests.** | **The main benchmark threat and the main opportunity.** We should (i) report on BTZSC's intent subsets; (ii) add a sub-1B reranker (Qwen3-Reranker-0.6B) as baseline or member; (iii) show the pool beats BTZSC's best sub-1B single model. BTZSC's NLI-plateau finding supports our story: the NLI model is weak alone but complementary. |

### (b) Embedding / similarity zero-shot intent detection and intent-aware pre-training

| Work | What they did | How we differ |
|---|---|---|
| Sung et al. 2023, **PIE** [sung-etal-2023-pre] (**Abstract**) | Contrastive pre-training of an intent-aware encoder on pseudo-labelled intent phrases, aligning utterances with intent names. N-way zero- and one-shot on 4 intent datasets; up to +5.4 over the best sentence encoder. | PIE trains an encoder; we train nothing. **Check whether its protocol uses N-way episodes or the full label space** before putting it in the same table. Use it as a baseline or member if it is released. |
| **Hu, Khosmood & Edalat 2024** [hu-etal-2024-exploring] (**Read**) | Dataless intent classification with MTEB embedders, best **bge-large-en-v1.5**. **Hand-written** declarative descriptions under label-preservation and format templates ("User wants to …"); an LLM was used only for utterance paraphrases. Datasets: ATIS, SNIPS, CLINC, MASSIVE en-US. **The entire dataset serves as the test set**; the metric is the mean of accuracy and macro-F1. Descriptions add +7.86 over tokenized labels, averaged across models; paraphrase and masking add +3.16. Robustness checked over 200 description combinations. No significance tests, no NLI member, no ensemble across models. | **Closest prior for the embedder and description half.** We differ on: (1) LLM-written descriptions from label names only; (2) pooling with NLI and a second embedder; (3) official test splits; (4) Banking77, HWU64, MTOP and Bitext; (5) paired McNemar tests. **Our SNIPS +9 from descriptions repeats their SNIPS direction. Present it as confirmation.** |
| Casanueva et al. 2020 [casanueva-etal-2020-efficient] (**Read**, Table 3); ConveRT [henderson-etal-2020-convert]; CPFT [zhang-etal-2021-shot]; Zhang et al. 2022 [zhang-etal-2022-fine]; Mehri & Eric 2021 [mehri-eric-2021-example]; ProtAugment [dopierre-etal-2021-protaugment]; Ma et al. 2022 [ma-etal-2022-effectiveness]; Du et al. 2023 [du-etal-2023-all-labels] | Intent encoders and few-shot intent detection. Full-data Banking77: BERT-tuned **93.66**, USE+ConveRT 93.36. Full-data CLINC150 97.16 and HWU64 92.62 (USE+ConveRT / ConveRT). | Background, and the full-data supervised references for the Banking77 retrieval comparison. |
| Xia et al. 2018 [xia-etal-2018-zero]; Siddique et al. 2021 [siddique-etal-2021-generalized]; Lamanov et al. 2022 [lamanov-etal-2022-template]; Maqbool et al. 2024 [maqbool-etal-2024-model-agnostic]; Zhang et al. 2022 [zhang-etal-2022-learn] | Zero-shot and generalized zero-shot intent detection. | All of these train on seen intents. Our zero-shot setting trains on no intent data. |
| SBERT [reimers-gurevych-2019-sentence]; BGE / C-Pack [xiao-etal-2024-cpack]; E5 [wang-etal-2022-e5]; INSTRUCTOR [su-etal-2023-one]; MTEB [muennighoff-etal-2023-mteb]; Schopf et al. 2022 [schopf-etal-2022-evaluating]; QZero [abdullahi-etal-2024-qzero] | Embedding backbones; similarity-based zero-shot classification. | **Contamination risk:** MTEB classification includes Banking77 and MASSIVE/MTOP intent. Disclose bge-v1.5's training data or argue the point empirically. This also bears on our negative result that a bigger MTEB-top embedder hurts as a kNN member (leaderboard overfitting). |
| Zhang et al. 2024 [zhang-etal-2024-negation-implicature]; TOD-BERT [wu-etal-2020-tod] | Intent embedders fail on negation and implicature; dialogue-specialised encoders. | Supports our negative result that a dialogue-specialised member hurts. |

### (c) Description-based ("dataless") classification and label descriptions

| Work | What they did | How we differ |
|---|---|---|
| Song & Roth 2014 [song-roth-2014-dataless]; Chai et al. 2020 [chai-etal-2020-description] | Dataless classification from label semantics; description-based classification. | Historical framing. |
| LSAP [mueller-etal-2022-label]; Zhong et al. 2021 [zhong-etal-2021-adapting-language] | Label-semantics-aware pre-training; meta-tuning on descriptions. | Both train; we do not. |
| Hong et al. 2024 [hong-etal-2024-exploring] (**Abstract**) | Intent descriptions for an LLM (FLAN-T5 3B) on unseen-domain zero-shot intent; studied description quality, quantity and length. | Generative LLM scorer trained with descriptions. Evidence from the LLM side. |
| Basile et al. 2022 [basile-etal-2022-ranking] (**Abstract**) | Unsupervised ranking and **aggregation of multiple noisy label descriptions** for Siamese zero-shot classifiers. | Pools across descriptions, not across models. A natural extension for us: several LLM-written descriptions per class. |
| Park et al. 2024 [park-etal-2024-dynamic-label] (**Abstract**) | An LLM refines intent label names per query inside retrieval-based ICL. | Per-query rewriting. Ours is one static description per class. |
| Menon & Vondrick 2022 [menon-vondrick-2022-visual]; CuPL [pratt-etal-2022-cupl]; CLIP prompt ensembling [radford-etal-2021-clip]; ZPE [allingham-etal-2023-zpe] (**Known**) | **LLM-generated class descriptions from class names only** improve CLIP zero-shot; prompt ensembling and how to weight it. | **Vision did our "LLM-authored descriptions from names" recipe first.** Cite it. Our addition is the text/intent analogue plus the opacity analysis. |
| Parikh et al. 2023 [parikh-etal-2023-exploring] (**Abstract**) | Zero-shot intent by prompting LLMs **with intent descriptions**, compared with T-Few PEFT (best at 1 shot), domain adaptation and augmentation. | LLM side; human-written descriptions. |

### (d) Ensembles, log-linear pooling, product of experts, "juries"

| Work | What they did | How we differ |
|---|---|---|
| Genest & Zidek 1986 [genest-zidek-1986-combining]; Kittler et al. 1998 [kittler-etal-1998-combining]; Hinton 2002 [hinton-2002-poe]; Dietterich 2000 [dietterich-2000-ensemble]; Wolpert 1992 [wolpert-1992-stacked] | Log vs linear opinion pools; the **product vs sum rule**; product of experts; ensemble theory; stacking. | **Our combiner is an equal-weight log-linear pool (geometric mean), so it is not novel.** Cite this openly. Note that the geometric mean (weights sum to 1) differs from a true PoE (weights 1 each). The difference governs sharpness, which explains the **under-confidence we measured** (ECE 0.22 on Banking77). Ablate product vs geometric vs arithmetic mean vs vote. |
| Smith, Cohn & Osborne 2005 [smith-etal-2005-logarithmic]; Garmash & Monz 2016 [garmash-monz-2016-ensemble]; Kurniawan et al. 2022 [kurniawan-etal-2022-unsupervised] | Log-opinion pools for CRFs; NMT ensembling; log opinion pooling for cross-lingual transfer. | NLP precedent for log-linear pooling. |
| Clark et al. 2019 [clark-etal-2019-dont]; Sanh et al. 2021 [sanh-etal-2021-others-mistakes]; DExperts [liu-etal-2021-dexperts] | Product of experts for debiasing and controlled decoding. | Same algebra, different goal. |
| Jiang et al. 2020, TACL [jiang-etal-2020-know] | Prompt ensembles for querying LMs. | TACL precedent for ensembling. |
| Deep ensembles [lakshminarayanan-etal-2017-deep-ensembles] | Accuracy and uncertainty from seed ensembles. | Our members are different pretrained architectures, not seeds. |
| **Verga et al. 2024, PoLL** [verga-etal-2024-juries] (**Known**) | "Replacing Judges with Juries": a panel of **smaller, diverse LLMs** beats a single large judge, at lower cost. | **Naming collision and conceptual precedent for "GeoJury".** Cite it and distinguish: classification, sub-1B non-generative members, log-linear rather than vote pooling. |
| LLM-Blender [jiang-etal-2023-llm]; self-consistency [wang-etal-2023-self-consistency]; MoA [wang-etal-2024-moa]; More Agents [li-etal-2024-more-agents]; survey [chen-etal-2025-llm-ensemble-survey] | Ensembles of generative LLMs. | Billion-scale generative members. |
| Kamen & Kamen 2025 [kamen-kamen-2025-majority]; Elgabry & Hamdi 2025 [elgabry-hamdi-2025-confidence]; RGPT [zhang-etal-2024-rgpt] (**Abstract**) | LLM ensemble for zero-shot IAB categorization (up to +65% F1 over the best single model); **confidence-weighted ensembles of small fine-tuned transformers beat large LLMs** on emotion; boosted LLM ensembles. | "Small ensembles beat big models" is already a talking point. Ours is zero-shot, training-free, cross-architecture, on intent, with paired tests. |

**Not found:** any paper that pools an NLI cross-encoder with bi-encoder embedders, training-free, for zero-shot intent classification, with paired tests of each member against the pool. The nearest neighbours are BTZSC (same model families, no pooling), Hu 2024 (same embedder and descriptions, no pooling), Basile 2022 (pools across descriptions) and CLIP prompt ensembling (pools across prompts). This is the novelty claim we can defend. It is empirical, not methodological.

### (e) Retrieval-augmented in-context classification with many labels

| Work | What they did | How we differ |
|---|---|---|
| **Milios, Reddy & Bahdanau 2023** [milios-etal-2023-context] (**Read**) | Retrieval-augmented ICL on **Banking77, HWU64, CLINC150** (plus GoEmotions). SBERT retriever (all-mpnet-base-v2), about 110 demonstrations, free generation mapped to the nearest label. OPT/LLaMA up to 70B. DialoGLUE 5- and 10-shot splits. Banking77 10-shot: LLaMA-2-70B-4K **92.11**, vs SetFit 84.51 and DeBERTa 88.41. | **Closest prior for contribution 2.** Ours uses: the full training set as the retrieval bank; a 4B reader rather than 70B; all 77 options scored in one pass rather than generation then mapping; a LoRA meta-trained on other intent data; and a geometric mean with kNN. Each change is incremental; together they form a configuration study. |
| Bertsch et al. 2024 [bertsch-etal-2024-long-context-icl] (**Abstract**) | Retrieval ICL vs many-shot long-context ICL vs fine-tuning on large label spaces, including **Banking77** and Clinic150. Fine-tuning can overtake ICL given enough data. | Reviewers will ask for Qwen3-4B fine-tuned directly on Banking77 as an upper reference. |
| Many-shot ICL [agarwal-etal-2024-many-shot]; LongICLBench [li-etal-2024-longiclbench] | Many-shot ICL; long-context LLMs struggle with many labels. | Supports using a short retrieved context. |
| Liu et al. 2022 [liu-etal-2022-makes]; Rubin et al. 2022 [rubin-etal-2022-learning]; LM-BFF [gao-etal-2021-making] | Similarity-selected and learned demonstration retrieval. Rubin includes BM25 baselines. | We use BM25 because the community protocol does. Say so, and add a dense-retriever ablation. |
| **kNN Prompting** [xu-etal-2023-knn-prompting] (**Abstract**); kNN-Prompt [shi-etal-2022-nearest]; kNN-LM [khandelwal-etal-2020-knnlm] | kNN over LM output distributions needs no calibration and scales "beyond context" to the full training set. kNN-Prompt interpolates the LM with kNN. | **Our reader × kNN geometric mean is a variant of these.** Cite them prominently. Our version takes the product of an option-scoring reader with a lexical-retrieval kNN. Note that our clean 24-shot ECE is 0.0105, which echoes kNN Prompting's "calibration-free" claim. |
| CARP [sun-etal-2023-text] (**Known**) | kNN demonstrations plus clue-and-reasoning prompting. | Same family. |
| MetaICL [min-etal-2022-metaicl]; in-context tuning [chen-etal-2022-meta]; Min et al. 2022 [min-etal-2022-rethinking] | **Meta-train an LM to do ICL across many tasks, then apply it to unseen tasks.** | **Our "LoRA on other public intent data" is domain-restricted MetaICL.** Name it that way. This also frames the negative result that more fine-tuning data saturates. |
| Robinson & Wingate 2023 [robinson-wingate-2023-mcp] | Score multiple-choice option symbols in **one forward pass**. | Our one-pass 77-option scoring is MCP at a large option count. That is a detail. The clone SemIf-OpenJev [lee-2026-semif] does the same with Qwen3.5-4B. |
| Loukas et al. 2023a, b [loukas-etal-2023-every-penny; loukas-etal-2023-breaking-bank] (**Abstract**) | **Banking77**: few-shot GPT-3.5/4, Cohere and Anthropic models vs SetFit; a low-cost **RAG-based LLM query method**. | Precedent for retrieval + LLM on Banking77, with cost reported. |
| Arora et al. 2024 [arora-etal-2024-intent] (**Abstract**) | Adaptive ICL with 7 LLMs vs SetFit; **hybrid uncertainty-based routing**, within 2% of LLM accuracy at 50% less latency. | Closest production-style comparison. |
| REIC [zhang-etal-2025-reic]; LARA [liu-etal-2024-lara] (**Abstract**) | RAG intent classification in industry; retrieval plus a fine-tuned small model for multi-turn intent. | Industry systems. |
| BM25 [robertson-zaragoza-2009-bm25]; LoRA [hu-etal-2021-lora]; Qwen3 [yang-etal-2025-qwen3] | Components we use. | — |

### (f) Calibration of zero-shot and ICL classifiers

**Prior work:**
- Guo et al. 2017 [guo-etal-2017-calibration] and Desai & Durrett 2020 [desai-durrett-2020-calibration]: calibration of neural classifiers and of pretrained transformers.
- Jiang et al. 2021, TACL [jiang-etal-2021-know-when]: calibration of LMs for QA.
- Contextual calibration [zhao-etal-2021-calibrate].
- Surface-form competition / PMI scoring [holtzman-etal-2021-surface].
- Noisy-channel prompting [min-etal-2022-noisy].
- Prototypical calibration [han-etal-2023-prototypical].
- Domain-context calibration [fei-etal-2023-mitigating].
- Batch calibration [zhou-etal-2024-batch-calibration].
- kNN Prompting's "calibration-free" claim [xu-etal-2023-knn-prompting].

**Our two calibration findings sit at opposite ends.**
- **Zero-shot pool: badly under-confident** (Banking77 ECE 0.22; confidence 0.54 vs accuracy 0.76).
  - Cause: T=1 softmax over cosine scores gives flat member distributions, and the geometric mean keeps the pool flat.
  - Fixes that keep accuracy unchanged: (i) a true PoE (unnormalised sum of log-probs); (ii) one global temperature fitted on a held-out *other* dataset; (iii) batch or contextual calibration, which use no labels. Accuracy cannot change under (i) and (ii) because the argmax is invariant.
  - Report ECE before and after. Under-confidence is the safer failure for routing and abstention (fewer confident errors). Say so, but do not oversell it.
- **Retrieval 24-shot run: well calibrated**, ECE 0.0105 vs Jev 0.034.
  - Jev's main claim is calibration [typesafe-2026-jev], so this result is central to C4.
  - Add: Brier and NLL; reliability diagrams; a paired bootstrap CI on ΔECE; a fixed binning scheme; and the exact Jev field used.

### (g) Significance testing and evaluation methodology

**Prior work:**
- **Paired tests for two classifiers:** McNemar 1947 [mcnemar-1947] and Dietterich 1998 [dietterich-1998-approximate], who recommend McNemar for comparing two classifiers on one test set.
- **Significance testing in NLP:** Berg-Kirkpatrick et al. 2012 [berg-kirkpatrick-etal-2012-empirical]; Søgaard et al. 2014 [sogaard-etal-2014-whats]; **Dror et al. 2018** [dror-etal-2018-hitchhikers].
- **Dror et al. 2017, TACL** [dror-etal-2017-replicability]: replicability analysis across **multiple datasets**. This is exactly the tool for "beats best member on most of 7 datasets". Use partial-conjunction counting or Holm correction, and report the number of significant wins.
- **Reporting and protocol:**
  - Expected validation performance [dodge-etal-2019-show].
  - Standard splits can mislead [gorman-bedrick-2019-need].
  - Statistical power [card-etal-2020-little].
  - Pre-registration in NLP [van-miltenburg-etal-2021-preregistering].
  - True few-shot selection [perez-etal-2021-true-few-shot].
- **Contamination:** [magar-schwartz-2022-data; sainz-etal-2023-nlp].
- **Label noise:** **Ying & Thomas 2022** [ying-thomas-2022-label] (**Read**) found about 14% (1,400+) of Banking77 training items possibly mislabeled.

**Our measured lesson:**
- Same reader family, same retrieval protocol, same test set, two configurations: the repo default (0.9321, p=0.066 vs Jev) and the clean pre-registered run (0.9357, p=0.0136). A 0.36-pt configuration difference changes the significance verdict.
- So C6 now rests on a measurement we made, not on principle alone. Use it as the worked example for pre-registration plus paired exact tests.
- Also note that the reproduction's own Wilson CI (91.41–93.29) contains 0.9321, so an unpaired comparison could not separate the two systems either. Pairing is what gives the test its power. Check that the 0.9357 point is not merely "outside the Wilson CI": the paired test is the right analysis, not that interval.

### (h) Dataset papers and the exact splits to state

| Dataset | Cite | Split note |
|---|---|---|
| Banking77 | [casanueva-etal-2020-efficient] | Official 10,003 train / 3,080 test. 25 test texts also occur in training (per [smith-2026-jev-banking77]). Report retrieval and kNN results with and without them. |
| CLINC150 | [larson-etal-2019-evaluation] | State which variant (full, small or imbalanced) and how out-of-scope is handled. |
| HWU64 | [liu-etal-2021-hwu64] (arXiv 1903.05566) | **No single canonical test split** in the original release. Name the split you used (e.g., DialoGLUE [mehri-etal-2020-dialoglue] or the PolyAI release). Avoid the word "official". |
| MASSIVE | [fitzgerald-etal-2023-massive] | en-US test, 60 intents. |
| MTOP | [li-etal-2021-mtop] | English test, intent labels only. |
| SNIPS | [coucke-etal-2018-snips]; split from [goo-etal-2018-slot] | Ganesh et al. 2026 [ganesh-etal-2026-selecting-open-weight] call SNIPS **saturated**, so reviewers will discount SNIPS gains. |
| Bitext | [bitext-2023-customer-support] (HF card, no paper) | **No official test split.** Release your split ids. |
| Surveys / few-shot splits | [larson-leach-2022-survey; mehri-etal-2020-dialoglue] | — |

### (i) Recent small-model, SetFit and LLM zero-shot intent numbers

| Work | Setting | Split | Use |
|---|---|---|---|
| SetFit [tunstall-etal-2022-setfit] | Few-shot sentence-transformer fine-tuning | — | Standard small baseline. Banking77 10-shot = 84.51 (as reported by Milios). |
| BTZSC [aarab-2026-btzsc] | True zero-shot, 38 checkpoints | Official HF test | Intent macro-F1: bge-large 0.58, best NLI 0.45, Qwen3-Reranker-8B 0.70. |
| Ganesh et al. 2026 [ganesh-etal-2026-selecting-open-weight] (**Read**) | Zero-shot, 41 open LLMs from 135M to 9B, **label names only** | **First 500 test items** | McNemar on MASSIVE: top models tied. SNIPS saturated. Best MASSIVE ≈ 0.72 (Qwen2-7B-Instruct, n=500). We evaluate on full tests with sub-1B models. |
| Hu et al. 2024 [hu-etal-2024-exploring] | Dataless, bge-large plus descriptions | Whole datasets | Mean of acc and F1: SNIPS ≈ 92.6, CLINC ≈ 81.5, MASSIVE ≈ 65.7, ATIS ≈ 61.0. **Not comparable** (different split and metric). |
| PIE [sung-etal-2023-pre] | Zero/one-shot, N-way | Check | Encoder baseline. |
| Parikh 2023 [parikh-etal-2023-exploring]; He & Garner 2023 [he-garner-2023-chatgpt-intent]; Edwards & Camacho-Collados 2024 [edwards-camacho-collados-2024-icl-enough]; Yu et al. 2023 [yu-etal-2023-open-closed-small]; Bucher & Martini 2024 [bucher-martini-2024-fine-tuned-small]; GenCo [zhang-etal-2023-genco] | LLM zero/few-shot vs fine-tuned small models | Various | Fine-tuned small models still beat zero-shot LLMs. Our training-free pool sits between the two. |
| Milios 2023; Loukas 2023; Arora 2024 | Retrieval/few-shot LLMs on Banking77 | DialoGLUE few-shot (Milios) | Few-shot retrieval-ICL references. |
| Community: jevbench [mehra-2026-jevbench]; AbdelStark [abdelstark-2026-jev-benchmarks] | Banking77 with descriptions only | n=500 sample; 72-label subset n=100 | Claude Sonnet 5 77.4, Jev 76.4, GPT-5-mini 73.6, BART-MNLI 42.8, fine-tuned DistilBERT 88.0. **Not peer reviewed.** This puts our zero-shot pool (Banking77 accuracy ≈ 0.76) **on par with Jev and frontier LLMs zero-shot**. That is a strong framing point, but the numbers come from a different sample, so re-run on the full test or caveat it. |

---

## 2. The Jev replication landscape (verified 2026-10-05)

### The closed system [typesafe-2026-jev]

- **Source:** TypeSafe blog post "Introducing System One Models and Jev", dated **2026-09-15** (updated 09-28).
- **What it is:** a hosted "System One Model" that takes typed Choice, Score and yes/no questions and returns probabilities. Model version `jev-1.13.0`.
- **Claims:** "Reinforcement Learning for Calibrated Decisions (RLCD)"; "All answers are accompanied with calibrated probabilities and confidence scores"; "193.6x Faster, 244.6x Cheaper"; 70–500 ms latency.
- **Undisclosed:** model size and training data ("We make all the data ourselves").
- **Documentation:** **no paper or tech report.** Vendor evals are internal workflows only [typesafe-2026-evals]. **TypeSafe publishes no Banking77 or intent number and no ECE.**

### The community reproduction [smith-2026-jev-banking77]

I read its README and file tree.

- **Protocol:**
  - Full official test, **3,080** items, with all **10,003** training items as the example bank.
  - Pre-registered (PROTOCOL.md) under a $5 budget.
  - Hand-written definitions for all 77 classes.
  - **BM25 retrieval (unigrams and bigrams) of 24 examples, at most 4 per class.**
  - All 77 options are given in one Jev Choice call.
  - Configuration selected on 770 held-out training items, deduplicated (154 screening + 616 confirmation).
- **Result:** **2,846/3,080 = 92.40%**, Wilson 95% CI 91.41–93.29.
- **Comparison:** only against the published BERT-tuned 93.66.
- **Licence and data:** no LICENSE file. It ships **`results/predictions.csv`**, which is what made our paired test possible.
- **Our position:** we matched its retrieval protocol (BM25, 24, ≤4 per class), so the comparison isolates reader plus pooling. **Result: 0.9357 vs 0.9240, b=119 / c=83, exact p=0.0136; ECE 0.0105 vs 0.034.**
- **Caveats:**
  - The run is a community run, not vendor-endorsed.
  - It pins `jev-1.13.0` at a date; Jev may change.
  - The repo has no licence, so ask the author before redistributing their predictions. Citing them and computing statistics from them is fine.
  - Confirm that our deduplication and per-class cap match theirs exactly.

### Other community evaluations

These are not peer reviewed.
- jevbench [mehra-2026-jevbench]: Jev 76.4% with descriptions only, n=500.
- AbdelStark [abdelstark-2026-jev-benchmarks]: 72-label subset, n=100.

### September 2026 open clones

| Repo | What it is | Banking77? | Small-model ensemble? | Claims to beat Jev? |
|---|---|---|---|---|
| Zefan-Cai/Open-Jev [cai-2026-open-jev] | Qwen-based 2B/9B/27B with LoRA, scalar decision head and temperature calibration | Trains on BANKING77 train. Its only test is a reported failure: 2B gets 64.8% (166/256) on 8 BM25 candidates, below BM25 top-1 alone at 82.4% (211/256). | No | No ("Jev remains ahead" on JevBench public tasks) |
| SiliconLabAI/OpenJev [siliconlab-2026-openjev] | Playground with per-option LLM scoring, one-shot JSON, or a proxy | None | No | No |
| TheoLeeCJ/SemIf-OpenJev [lee-2026-semif] | **One-pass option probabilities** from Qwen3.5-4B | None. 0.845 agreement vs Jev's 0.883 on a 102-row TypeSafe subset. | No | No |
| Heman10x-NGU/openJev-verdict-2.0 [heman-2026-openjev-verdict] | Single 149.6M ModernBERT-base with a calibration head | Banking77-derived **5-option** data only; no 77-way result | No | On another benchmark: 77.10 vs a "Jev 72.70" copied from elsewhere |
| featherless-ai/simple-jev [featherless-2026-simple-jev] | Logit-reading server | Not run; its audit says the API caps choices at 50 | No | No |

**Conclusion:**
- No clone reports a 77-way Banking77 accuracy at or above 92.40%. No clone uses a sub-1B ensemble.
- **Ours appears to be the first public result to beat the Jev reproduction on Banking77 with a paired significance test, and the first to compare calibration.**
- Open-Jev's 2B retrieval reader scored below BM25 top-1, which shows the reader + kNN design is not trivial.
- This search covered GitHub and the web. X posts and blogs may have been missed. **Re-check before submission.**

---

## 3. Novelty verdict

| Contribution | Status | Reason |
|---|---|---|
| C1. Training-free log-linear pool of an NLI cross-encoder and 2 bi-encoders (all sub-1B) for zero-shot intent | **Incremental method; new as a systematic empirical result** | Kittler's product rule / PoE with off-the-shelf members. No prior pools NLI with embedders for intent with paired tests; BTZSC covers the same families without pooling. The under-confidence finding (ECE 0.22) and its accuracy-neutral fix add analysis depth. |
| C2. LLM-authored descriptions from names only; largest gains for opaque names | **Incremental / partly known** | Hu 2024, Basile 2022, Hong 2024, Parikh 2023, and CuPL / Menon & Vondrick in vision. New: the opacity-conditioned analysis across 7 sets with paired tests. Make opacity measurable to strengthen it. |
| C3. Retrieval-augmented one-pass option scoring with a meta-trained 4B reader × kNN on Banking77 | **Incremental configuration** | Milios 2023, kNN Prompting, kNN-Prompt, MetaICL, MCP, Loukas 2023. New: full-bank BM25 at 4B scale reaching **0.9357 (≈ BERT-tuned 93.66) with ECE 0.0105**. |
| C4. Pre-registered, matched-protocol, *paired* comparison against a closed "calibrated decision" system on accuracy **and** calibration | **Genuinely new (timely)** | No peer-reviewed evaluation of Jev exists, and no clone reports 77-way Banking77. Ours is significant (p=0.0136) and better calibrated (0.0105 vs 0.034). Weakness: one dataset and one community run. |
| C5. Measured negative results (data saturation, MTEB-top embedder hurts as kNN member, top-k reranking fails, dialogue encoder hurts) | **Novel as evidence** | Plausible from prior work but rarely reported. Each needs a paired test and an explanation. |
| C6. Evaluation-methodology lessons | **Known principles, now with a measured flip** | Repo default p=0.066 vs clean pre-registered p=0.0136. Add: subset vs full-test evaluation (Ganesh 2026 uses the first 500 items); unpaired Wilson-CI comparisons vs paired McNemar; multi-dataset claims (Dror 2017). |

### Framing that maximises originality for TACL

Lead with a question, not a system:

> *"How far can small, open, training-free or lightly-adapted models go on many-label intent classification, on accuracy and calibration, and what does it take to show it rigorously?"*

Organise the answer in three parts:
1. **Zero-shot regime.** Pooling a cross-encoder with bi-encoders beats its members. Make complementarity the scientific core: error overlap, oracle-of-members bound, effective weight per member, and product vs sum vs vote. Label text matters most when names are opaque. The pool is under-confident by construction, and the fix costs no accuracy.
2. **Retrieval regime.** A 4B open reader with a kNN product matches full-data fine-tuned BERT and significantly beats a closed calibrated-decision system, while being better calibrated.
3. **Methodology.** Pre-registration and paired exact tests changed the verdict in our own data.

Present Jev as a case study of evaluating closed systems, not as the headline. TACL reviewers distrust "we beat a 3-week-old product". The calibration result is what turns it from benchmark-chasing into substance.

---

## 4. Risks a TACL reviewer will raise (ranked)

- **R1. "The method is a textbook product-of-experts ensemble."**
  - Concede it in the introduction. Make the controlled study and the complementarity analysis the contribution. Ablate the combiner (true PoE / geometric / arithmetic / vote / max).
- **R2. Overlap with Hu 2024 and BTZSC.**
  - Baseline: bge-large with Hu-style descriptions.
  - Report on BTZSC's intent subsets.
  - Add a sub-1B reranker (e.g., Qwen3-Reranker-0.6B). BTZSC finds rerankers best, so leaving one out will be noticed.
- **R3. T=1 softmax over cosine scores and the geometric mean make the pool flat.**
  - **This is now confirmed by our under-confidence** (Banking77 ECE 0.22, confidence 0.54 vs accuracy 0.76).
  - Reviewers will also ask whether the embedders contribute or merely break ties for NLI.
  - Report per-member entropy and effective weight, sensitivity to T and to weights chosen without test labels, and the accuracy-neutral recalibration.
- **R4. Selection on test** for the 7-dataset zero-shot claim (members, template, T=1).
  - State that the configuration was frozen before evaluation, and cite Perez 2021.
  - The Banking77 headline is covered by the pre-registration, but **show the pre-registration timestamp precedes the test run**, and disclose the repo-default 0.9321 result.
- **R5. Multiple comparisons.** "Most of 7" needs Holm or Dror-2017 counting. Report effect sizes with CIs.
- **R6. Contamination.**
  - bge-v1.5 is MTEB-tuned, and MTEB includes Banking77 and MASSIVE/MTOP intent.
  - Qwen3 pretraining may include the test sets.
  - The LoRA's "other public intent data" may overlap Banking77 through paraphrases or derived sets.
  - Mitigations: an n-gram overlap audit, disclosure of training data, and a contamination probe [sainz-etal-2023-nlp].
- **R7. Split hygiene.** HWU64 and Bitext have no official test. State the CLINC150 variant. 25 Banking77 test duplicates matter for kNN, so report with and without them.
- **R8. The Jev comparison is fragile.**
  - One community run, a moving closed model, no licence, and Banking77 label noise (about 14% of training).
  - Mitigations already in hand: the paired test and ECE.
  - Still needed: a ΔECE CI with Brier/NLL; pinning the Jev version and date; a Qwen3-4B fine-tuned on Banking77 as an upper reference; citing Ying & Thomas; and a statement that the 1.2-pt gain is 25 net items in a noisy-label regime.
- **R9. Missing baselines.**
  - C1: the best ≤9B zero-shot LLM on the same full tests.
  - C3: BM25 top-1 alone, kNN alone, reader alone, a dense retriever, and many-shot or random-demonstration ICL [bertsch-etal-2024-long-context-icl].
- **R10. "Sub-1B" wording.** The zero-shot pool is about 0.85B in total. The 4B reader is not sub-1B. Report parameter count and latency per query.
- **R11. Novelty half-life.** The Jev ecosystem changes weekly. Freeze a dated snapshot and re-check before camera-ready.
- **R12. Naming.** "Jury" evokes PoLL [verga-etal-2024-juries]; "Geo" may read as geographic.

---

## 5. Suggested titles

1. *Small, Open, and Calibrated: Pooled Encoders and Retrieval-Augmented Readers for Many-Label Intent Classification*
2. *Pooling Heterogeneous Sub-1B Models for Zero-Shot Intent Classification: A Controlled Study with Paired Significance Tests*
3. *When Do Small Zero-Shot Scorers Make a Good Jury? Log-Linear Pooling of NLI and Embedding Models for Intent Classification*
4. *GeoJury: Training-Free Product-of-Experts Pooling of Small Encoders, and an Open Retrieval-Augmented Reader that Beats a Closed Calibrated-Decision Model*
5. *Complementarity, Label Text, and Retrieval Memory: What Makes Small Open Models Competitive for Intent Classification*

I recommend 1 or 5. They are neutral, cover both regimes and the calibration result, and avoid the "jury" collision.

---

## 6. Action items before writing related work

1. Already done: the paired McNemar test against `predictions.csv`. Add the ΔECE bootstrap CI and Brier/NLL, and reliability diagrams for the zero-shot pool and the 24-shot run.
2. Run the zero-shot recalibration ablation (true PoE, or one global temperature fitted on another dataset) and report ECE before and after with accuracy unchanged.
3. Add baselines: BTZSC intent subsets, a sub-1B reranker, Hu-style bge-large with descriptions, the best ≤9B zero-shot LLM on full tests, and Qwen3-4B fine-tuned on Banking77.
4. Run a contamination audit of the LoRA data against the Banking77 test.
5. Fix the split table: HWU64 source, CLINC150 variant, Bitext ids, and handling of Banking77 duplicates.
6. Verify any number taken from an **Abstract** or **Known** item in the paper before citing it (PIE protocol, Parikh datasets, Bertsch Banking77 numbers).
