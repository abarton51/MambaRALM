from ralm import RALM
from typing import Any

class DollyRALM(RALM):
    '''RALM class dedicated to wrapping a Transformer architecture (Dolly) for RAG tasks'''
    def __init__(self, pretrained_prefix : str, vector_db : Any):
        super().__init__(RALM, vector_db)
        # TODO: Instantiate mamba model
        self.lm = None

    def format_output(self, output : str) -> str:
        pass
        # TODO: write this

    def predict(self, question : str) -> str:
        prompt = self.generate_prompt(question)
        response = self.lm(prompt) # TODO: This may be called differently
        return self.format_output(response)