# inference.py - Script for running inference on the fine-tuned model.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "fine-tuned-qwen"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "The movie was fantastic because"
print("Generated Text:", generate_text(prompt))
