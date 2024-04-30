<h1 align="center">MambaRALM: Analyzing RALMs with Selective State Space and Transformer Based Architectures for Long Sequence Modeling</h1>

| ![](https://github.com/abarton51/MambaRALM/blob/main/mamba_vs_transformer_ralms_fig.png?raw=true) |
|:--:|
| *Left: RAG Pipeline. Right: RALM Analysis Pipeline.* |

> [Sebastian Jaskowski](https://github.com/Sebiancoder), [Austin T. Barton](github.com/abarton51), [Nolan Bridges](https://github.com/NMBridges)
>
> Abstract: This study examines the efficacy of Retrieval Augmented Language Models (RALMs), a recent paradigm incorporating retrievers to enhance standalone language models during inference. While most RALMs rely on transformer architecture, which suffers from scalability issues limiting context windows, this project explores the potential of the Mamba architecture, known for its proficiency with long sequences and Long Range Dependencies (LRDs), in improving RALMs' performance. The study constructs a RALM based on the Mamba architecture and evaluates it alongside a transformer-based RALM on a subset of the TriviaQA dataset. Results show comparable performance for small to medium context chunks (k ≤ 7), but the Mamba-based RALM demonstrates better resilience to larger context sizes (k > 7), indicating its potential for handling irrelevant information more effectively.

The MambaRALM project aims to construct an evaluate a  Retrieval Augmented Generation (RAG) QA language model based on an instruction-tuned language model based on the [Mamba architecture](https://arxiv.org/abs/2312.00752). In our case, we our using the 2.8B parameter instruction-tuned [Mamba-Chat model](https://huggingface.co/havenhq/mamba-chat). 

We will compare the performance of MambaRALM to a Transformer-based RALM: [Dolly-v2-3B](https://huggingface.co/databricks/dolly-v2-3b) by Databricks. The models will be evaluated over the TriviaQA QA dataset.
