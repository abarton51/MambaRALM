# MambaRALM

The MambaRALM project aims to construct an evaluate a  Retrieval Augmented Generation (RAG) QA language model based on an instruction-tuned language model based on the [Mamba architecture](https://arxiv.org/abs/2312.00752). In our case, we our using the 2.8B parameter instruction-tuned [Mamba-Chat model](https://huggingface.co/havenhq/mamba-chat). 

We will compare the performance of MambaRALM to two similarly-sized Transformer-based RALMs: [Phi-2](https://huggingface.co/microsoft/phi-2) by Microsoft and [Dolly-v2-3B](https://huggingface.co/databricks/dolly-v2-3b) by Databricks. The models will be evaluated over the TriviaQA QA dataset.
