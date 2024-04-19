from src.ralm import RALM
from typing import Any
from src.config import device
from transformers import pipeline
from langchain_core.vectorstores import VectorStore
from langchain.prompts import PromptTemplate
import torch

class DollyRALM(RALM):
    '''RALM class dedicated to wrapping a Transformer architecture (Dolly) for RAG tasks'''
    def __init__(self, pretrained_prefix : str = "databricks/dolly-v2-3b", vector_db : VectorStore = None):
        super().__init__(vector_db)

        self.lm = pipeline(model=pretrained_prefix, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device)

    def format_output(self, output : str) -> str:
        return output[0]["generated_text"]

    #override
    def generate_prompt(self, question : str, k : int = 4) -> str:
        '''Generates an LLM prompt (context, question, specifications), provided the question, utilizing k context chunks'''
        context = " , ".join(self.retrieve_context(question, k))

        messages = PromptTemplate(
            input_variables=["context", "question"],
            template="""

            Document Context:
            
            {context}

            Original Query:

            {question}

            "Please respond to the original query. If the selected document context is relevant and informative, provide a detailed answer based on its content. However, if the selected document context does not offer useful information or is not applicable, simply state 'No answer found'.

            """
        )

        return messages.format(context=context, question=question)

    def predict(self, question : str, k : int = 4) -> str:
        prompt = self.generate_prompt(question, k=k)
        response = self.lm(prompt)

        try:
            return self.format_output(response)
        except:
            print("Error formatting output. Raw output string returned.")
            return response
            