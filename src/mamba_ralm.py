from src.ralm import RALM
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from langchain_core.vectorstores import VectorStore
from src.config import device
from typing import Any
import torch

class MambaRALM(RALM):
    '''RALM class dedicated to wrapping a Mamba architecture for RAG tasks'''
    def __init__(self, pretrained_prefix : str = "havenhq/mamba-chat", vector_db : VectorStore = None):
        super().__init__(vector_db)

        self.lm = MambaLMHeadModel.from_pretrained(pretrained_prefix, device=device, dtype=torch.float16)
        # self.lm = AutoModelForCausalLM.from_pretrained(pretrained_prefix, trust_remote_code=True).to(device) # Non-instruction tuned
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_prefix)

        self.ANSWER_START = "<|assistant|>\n"
        self.ANSWER_END = "<|endoftext|>"

        self.tokenizer.eos_token = ANSWER_END
        self.tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer.chat_template = AutoTokenizer.from_pretrained(
            "BAAI/bge-small-en-v1.5"
        ).chat_template

    def format_output(self, output : list[str]) -> str:

        return output[0].split("<|assistant|>\n")[1].split("<|endoftext|>\n")[0]

    def predict(self, question : str, k : int = 4, max_response_length : int = 150) -> str:

        prompt = self.generate_prompt(question, k=k)
        input_ids = self.tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True).to(device)
        response = self.lm.generate(
            input_ids=input_ids, 
            max_length=(input_ids.shape[1] + max_response_length), 
            temperature=0,
            eos_token_id=self.tokenizer.eos_token_id)
        output = self.tokenizer.decode(response[0])

        try:

            return self.format_output(output)

        except:

            print("Error formatting output. Raw output string returned.")
            
            return output