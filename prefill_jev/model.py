import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="cuda",
)

model.eval()

def get_token_id(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1: 
      raise ValueError( f"'{text}' must tokenize to exactly one token. Got {ids}" ) 
    return ids[0]