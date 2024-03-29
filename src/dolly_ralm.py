from ralm import RALM
from typing import Any
from langchain_core.vectorstores import VectorStore

class DollyRALM(RALM):
    '''RALM class dedicated to wrapping a Transformer architecture (Dolly) for RAG tasks'''
    def __init__(self, pretrained_prefix : str = "databricks/dolly-v2-3b", vector_db : VectorStore = None):
        super().__init__(vector_db)
        # TODO: Instantiate Dolly RALM model
        self.lm = None

    def format_output(self, output : str) -> str:
        pass
        # TODO: write this

    def predict(self, question : str) -> str:
        prompt = self.generate_prompt(question)
        response = self.lm(prompt) # TODO: This may be called differently
        return self.format_output(response)