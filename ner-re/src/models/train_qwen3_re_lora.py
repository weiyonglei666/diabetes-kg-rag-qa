#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-8B 医学关系抽取：无 bitsandbytes 的纯 LoRA 训练脚本
参数全部写在代码里，不走命令行

适用：
- 单卡 RTX 4090 24GB
- 不使用 bitsandbytes / QLoRA
- 使用 transformers + peft + trl
- 输入数据为 messages 格式 jsonl

输入样本示例：
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"relation\":\"Drug_Disease\"}"}
  ],
  "meta": {...}
}
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# =========================================================
# 配置区：直接改这里
# =========================================================

# ===== 数据路径 =====
TRAIN_FILE = "../re_qwen_data/train.jsonl"
DEV_FILE = "../re_qwen_data/dev.jsonl"
TEST_FILE = "../re_qwen_data/test.jsonl"   # 这里只留着，不在本脚本里使用
OUTPUT_DIR = "../qwen3_re_lora_no_bnb_output"

# ===== 模型路径 =====
MODEL_NAME_OR_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

# ===== 随机种子 =====
SEED = 42

# ===== 精度 =====
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = not USE_BF16

# ===== 序列长度 =====
# 4090 24GB 建议先从 384 起
MAX_SEQ_LENGTH = 1024

# ===== Batch =====
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32

# ===== 训练轮数与学习率 =====
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 3
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# ===== LoRA 配置 =====
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 先只打注意力层，最稳、省显存
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# 如果显存够、训练稳定，可以再换成：
# LORA_TARGET_MODULES = [
#     "q_proj", "k_proj", "v_proj", "o_proj",
#     "up_proj", "down_proj", "gate_proj"
# ]

# ===== 训练行为 =====
GRADIENT_CHECKPOINTING = True
PACKING = False

# ===== 日志 / 保存 / 评估 =====
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
SAVE_STRATEGY = "steps"
EVAL_STRATEGY = "steps"

# ===== 是否只训练 assistant 回复部分 =====
TRAIN_ON_ASSISTANT_ONLY = True

# ===== tokenizer =====
SET_PAD_TOKEN_AS_EOS = True

# ===== 其他 =====
REPORT_TO = "none"   # 如需 wandb 可改
MAX_GRAD_NORM = 1.0
LR_SCHEDULER_TYPE = "cosine"


# =========================================================
# 工具函数
# =========================================================

def print_config():
    print("=" * 80)
    print("Qwen3-8B RE LoRA Training Config (No bitsandbytes)")
    print("=" * 80)
    config_dict = {
        "TRAIN_FILE": TRAIN_FILE,
        "DEV_FILE": DEV_FILE,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MODEL_NAME_OR_PATH": MODEL_NAME_OR_PATH,
        "SEED": SEED,
        "USE_BF16": USE_BF16,
        "USE_FP16": USE_FP16,
        "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH,
        "PER_DEVICE_TRAIN_BATCH_SIZE": PER_DEVICE_TRAIN_BATCH_SIZE,
        "PER_DEVICE_EVAL_BATCH_SIZE": PER_DEVICE_EVAL_BATCH_SIZE,
        "GRADIENT_ACCUMULATION_STEPS": GRADIENT_ACCUMULATION_STEPS,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_TRAIN_EPOCHS": NUM_TRAIN_EPOCHS,
        "WARMUP_STEPS": WARMUP_STEPS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LORA_R": LORA_R,
        "LORA_ALPHA": LORA_ALPHA,
        "LORA_DROPOUT": LORA_DROPOUT,
        "LORA_TARGET_MODULES": LORA_TARGET_MODULES,
        "GRADIENT_CHECKPOINTING": GRADIENT_CHECKPOINTING,
        "PACKING": PACKING,
        "LOGGING_STEPS": LOGGING_STEPS,
        "EVAL_STEPS": EVAL_STEPS,
        "SAVE_STEPS": SAVE_STEPS,
        "SAVE_TOTAL_LIMIT": SAVE_TOTAL_LIMIT,
        "SAVE_STRATEGY": SAVE_STRATEGY,
        "EVAL_STRATEGY": EVAL_STRATEGY,
        "TRAIN_ON_ASSISTANT_ONLY": TRAIN_ON_ASSISTANT_ONLY,
        "REPORT_TO": REPORT_TO,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR_SCHEDULER_TYPE": LR_SCHEDULER_TYPE,
    }
    for k, v in config_dict.items():
        print(f"{k}: {v}")
    print("=" * 80)


def load_jsonl_preview(path: str, n: int = 2):
    print(f"\n预览数据: {path}")
    if not os.path.exists(path):
        print("文件不存在")
        return
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            print(json.dumps(obj, ensure_ascii=False, indent=2)[:1000])
            print("-" * 60)


def strip_think_blocks(text: str) -> str:
    """
    清理 assistant 内容里的 <think>...</think>
    避免模型学到思维块
    """
    if not isinstance(text, str):
        return text
    while True:
        start = text.find("<think>")
        end = text.find("</think>")
        if start != -1 and end != -1 and end > start:
            text = text[:start] + text[end + len("</think>"):]
        else:
            break
    return text.strip()


def sanitize_messages(messages):
    """
    只清理 assistant content，其他角色保持不变
    """
    new_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant":
            content = strip_think_blocks(content)
        new_messages.append({"role": role, "content": content})
    return new_messages


def apply_chat_template(example, tokenizer):
    """
    将 messages 转成单条训练文本
    """
    messages = sanitize_messages(example["messages"])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def find_response_template_ids(tokenizer):
    """
    给 assistant-only loss 用
    兼容不同 chat template 风格
    """
    candidate_texts = [
        "<|im_start|>assistant\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant\n",
    ]

    for text in candidate_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 0:
            return ids

    raise ValueError("无法自动找到 assistant response template token ids")


def main():
    print_config()
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== 数据预览 ==========
    load_jsonl_preview(TRAIN_FILE, n=1)
    load_jsonl_preview(DEV_FILE, n=1)

    # ========== 数据集 ==========
    data_files = {
        "train": TRAIN_FILE,
        "validation": DEV_FILE,
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    # ========== tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        use_fast=False,
        trust_remote_code=True,
    )

    if SET_PAD_TOKEN_AS_EOS and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # ========== 模型 ==========
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    # ========== LoRA ==========
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    # ========== 数据转文本 ==========
    train_dataset = raw_datasets["train"].map(
        lambda x: apply_chat_template(x, tokenizer),
        remove_columns=raw_datasets["train"].column_names,
        desc="Formatting train dataset",
    )
    eval_dataset = raw_datasets["validation"].map(
        lambda x: apply_chat_template(x, tokenizer),
        remove_columns=raw_datasets["validation"].column_names,
        desc="Formatting dev dataset",
    )

    print("\n示例训练文本：")
    print(train_dataset[0]["text"][:1500])
    print("-" * 80)

    # ========== 训练配置 ==========
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy=SAVE_STRATEGY,
        eval_strategy=EVAL_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=USE_BF16,
        fp16=USE_FP16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        packing=PACKING,
        report_to=REPORT_TO,
        seed=SEED,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
    )

    # ========== Trainer ==========
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
    )

    # ========== assistant-only loss ==========
    if TRAIN_ON_ASSISTANT_ONLY:
        try:
            from trl import DataCollatorForCompletionOnlyLM

            response_template_ids = find_response_template_ids(tokenizer)
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template_ids,
                tokenizer=tokenizer,
            )
            trainer.data_collator = data_collator
            print("\n已启用 assistant-only loss")
            print(f"response_template_ids: {response_template_ids}")
        except Exception as e:
            print("\n警告：assistant-only loss 启用失败，将退回普通 SFT。")
            print(f"原因: {e}")

    # ========== 打印可训练参数 ==========
    if hasattr(trainer.model, "print_trainable_parameters"):
        print("\n可训练参数：")
        trainer.model.print_trainable_parameters()

    # ========== 开始训练 ==========
    print("\n开始训练...")
    train_result = trainer.train()

    # ========== 保存 ==========
    print("\n保存 LoRA adapter 与 tokenizer ...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 保存训练指标
    metrics = train_result.metrics
    with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ========== 验证集评估 ==========
    print("\n开始验证集评估...")
    eval_metrics = trainer.evaluate()
    with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    print("\n训练完成")
    print("训练指标:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n验证指标:")
    print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
