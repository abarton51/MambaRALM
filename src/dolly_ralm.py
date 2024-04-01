from src.ralm import RALM
from typing import Any
from src.config import device
from transformers import pipeline
from langchain_core.vectorstores import VectorStore
import torch

class DollyRALM(RALM):
    '''RALM class dedicated to wrapping a Transformer architecture (Dolly) for RAG tasks'''
    def __init__(self, pretrained_prefix : str = "databricks/dolly-v2-3b", vector_db : VectorStore = None):
        super().__init__(vector_db)
        # TODO: Instantiate Dolly RALM model

        self.lm = pipeline(model=pretrained_prefix, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device)

    def format_output(self, output : str) -> str:

        return output[0]["generated_text"]

    def predict(self, question : str) -> str:

        prompt = self.generate_prompt(question)
        response = self.lm(prompt)

        try:

            return self.format_output(response)

        except:
            
            print("Error formatting output. Raw output string returned.")
            return response
            