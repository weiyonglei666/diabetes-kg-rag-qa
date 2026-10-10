# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
LORA_PATH = "../output/qwen3_8b_ner_lora/best_lora"
MERGED_PATH = "../output/qwen3_8b_ner_merged"

def main():
    os.makedirs(MERGED_PATH, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()

    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)

    print(f"merged model saved to: {MERGED_PATH}")

if __name__ == "__main__":
    main()
