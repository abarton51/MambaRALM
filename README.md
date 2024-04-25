<h1 align="center">MambaRALM: Analyzing RALMs with Selective State Space and Transformer Based Architectures for Long Sequence Modeling</h1>

![](https://github.com/abarton51/MambaRALM/blob/main/mamba_ralm_fig.png?raw=true)

The MambaRALM project aims to construct an evaluate a  Retrieval Augmented Generation (RAG) QA language model based on an instruction-tuned language model based on the [Mamba architecture](https://arxiv.org/abs/2312.00752). In our case, we our using the 2.8B parameter instruction-tuned [Mamba-Chat model](https://huggingface.co/havenhq/mamba-chat). 

We will compare the performance of MambaRALM to a Transformer-based RALM: [Dolly-v2-3B](https://huggingface.co/databricks/dolly-v2-3b) by Databricks. The models will be evaluated over the TriviaQA QA dataset.
