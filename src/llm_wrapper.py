import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def load_llm(model_name:str, load_in_4bit=True, lora=True, r=16, alpha=16, dropout=0.05, target_modules=None):
    if load_in_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        quant = None
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
    if lora:
        peft_cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules or ["q_proj","v_proj","k_proj","o_proj"])
        model = get_peft_model(model, peft_cfg)
    return tok, model
