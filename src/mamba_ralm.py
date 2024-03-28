from ralm import RALM
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from config import device

class MambaRALM(RALM):
    '''RALM class dedicated to wrapping a Mamba architecture for RAG tasks'''
    def __init__(self, pretrained_prefix : str):
        super(RALM, self).__init__()
        self.lm = MambaLMHeadModel.from_pretrained(pretrained_prefix, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_prefix)

    def format_output(self, output : list[str]) -> str:
        return output[0].split("<|assistant|>\n")[1].split("<|endoftext|>\n")[0]

    def predict(self, question : str, max_response_length : int = 150) -> str:
        prompt = self.generate_prompt(question)
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
        response = self.lm.generate(input_ids, max_length=(input_ids.shape[1] + max_response_length))
        output = self.tokenizer.batch_decode(response)
        return self.format_output(output)