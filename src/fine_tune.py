# fine_tune.py - Script for fine-tuning the Qwen-0.5B model using LoRA.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset
dataset = load_dataset("imdb")

# Load model and tokenizer
model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

print("Fine-tuning setup is complete!")
