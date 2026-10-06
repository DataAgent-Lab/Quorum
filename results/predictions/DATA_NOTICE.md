# Data notice for the files in this directory

The repository's code is MIT-licensed (see `LICENSE`). The prediction dumps here contain **utterance texts and labels
from public datasets**; those texts keep the licences of their sources, independently of the code licence:

| Dataset (files) | Licence of the source | Source |
|---|---|---|
| Banking77 (`banking77_*`) | CC BY 4.0 | PolyAI — https://github.com/PolyAI-LDN/task-specific-datasets |
| CLINC150 (`clinc150_*`) | CC BY 3.0 | https://github.com/clinc/oos-eval |
| HWU64 (`hwu64_*`) | CC BY 4.0 | https://github.com/xliuhw/NLU-Evaluation-Data |
| MASSIVE en-US (`massive_*`) | CC BY 4.0 | https://github.com/alexa/massive |
| MTOP en (`mtop_*`) | **CC BY-SA 4.0** — files containing MTOP utterances are shared under the same licence | https://dl.fbaipublicfiles.com/mtop/mtop.zip |
| SNIPS, 7-intent benchmark (`snips_*`) | CC0 1.0 | https://github.com/sonos/nlu-benchmark |
| Bitext customer support (`bitext_*`) | **CDLA-Sharing-1.0** — texts and splits are shared under the same licence | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset |

Model outputs (predictions, probabilities) are ours. No files from the third-party Jev reproduction
(github.com/simonmesmith/jev-banking77-experiment, unlicensed) are redistributed here; analyses that need them fetch
them at run time.
