#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GLM-4-9B-Chat 医学关系抽取 QLoRA 训练脚本
可直接替换版本

特点：
- 单卡 RTX 4090 24GB
- bitsandbytes 4bit QLoRA
- transformers + peft + trl
- 兼容当前环境的 SFTTrainer / SFTConfig 接口
- 不依赖 tokenizer.apply_chat_template
- 自动清理 <think>...</think>
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# =========================================================
# 配置区：直接改这里
# =========================================================

# ===== 数据路径 =====
TRAIN_FILE = "../glm4_re_data/train.jsonl"
DEV_FILE = "../glm4_re_data/dev.jsonl"
TEST_FILE = "../glm4_re_data/test.jsonl"   # 当前脚本不使用
OUTPUT_DIR = "../glm4_re_qlora_output"

# ===== 模型路径 =====
MODEL_NAME_OR_PATH = "/root/.cache/modelscope/hub/models/ZhipuAI/glm-4-9b-chat"

# ===== 随机种子 =====
SEED = 42

# ===== 精度 =====
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = not USE_BF16

# ===== 4bit QLoRA =====
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# ===== 序列长度 =====
MAX_SEQ_LENGTH = 1024

# ===== Batch =====
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

# ===== 学习率 / epoch =====
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1.5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# ===== LoRA =====
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 最稳的写法：自动对所有线性层注入 LoRA
LORA_TARGET_MODULES = "all-linear"

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
REPORT_TO = "none"
MAX_GRAD_NORM = 1.0
LR_SCHEDULER_TYPE = "cosine"


# =========================================================
# 工具函数
# =========================================================

def print_config():
    print("=" * 80)
    print("GLM-4-9B-Chat RE QLoRA Training Config")
    print("=" * 80)
    config_dict = {
        "TRAIN_FILE": TRAIN_FILE,
        "DEV_FILE": DEV_FILE,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MODEL_NAME_OR_PATH": MODEL_NAME_OR_PATH,
        "SEED": SEED,
        "USE_BF16": USE_BF16,
        "USE_FP16": USE_FP16,
        "USE_4BIT": USE_4BIT,
        "BNB_4BIT_QUANT_TYPE": BNB_4BIT_QUANT_TYPE,
        "BNB_4BIT_USE_DOUBLE_QUANT": BNB_4BIT_USE_DOUBLE_QUANT,
        "BNB_4BIT_COMPUTE_DTYPE": str(BNB_4BIT_COMPUTE_DTYPE),
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


def load_jsonl_preview(path: str, n: int = 1):
    print(f"\n预览数据: {path}")
    if not os.path.exists(path):
        print("文件不存在")
        return
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            print(json.dumps(obj, ensure_ascii=False, indent=2)[:1200])
            print("-" * 60)


def strip_think_blocks(text: str) -> str:
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
    new_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant":
            content = strip_think_blocks(content)
        new_messages.append({"role": role, "content": content})
    return new_messages


def format_glm_dialog(messages):
    """
    手工拼接训练文本，避免不同 tokenizer/chat_template 差异
    """
    system = ""
    user = ""
    assistant = ""

    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            user = m["content"]
        elif m["role"] == "assistant":
            assistant = m["content"]

    text = (
        f"系统：{system}\n"
        f"用户：{user}\n"
        f"助手：{assistant}"
    )
    return text


def apply_chat_template(example, tokenizer):
    messages = sanitize_messages(example["messages"])
    text = format_glm_dialog(messages)
    return {"text": text}


def find_response_template_ids(tokenizer):
    """
    给 assistant-only loss 用
    """
    candidate_texts = [
        "助手：",
        "\n助手：",
    ]
    for text in candidate_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 0:
            return ids
    raise ValueError("无法自动找到 assistant response template token ids")


def build_bnb_config():
    if not USE_4BIT:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
    )


def print_linear_module_preview(model, limit=80):
    """
    调试用：打印部分线性层名，确认模型结构
    """
    print("\n===== 模型中的线性层模块名预览 =====")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name)
            count += 1
            if count >= limit:
                break
    print("===== 预览结束 =====\n")


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
        trust_remote_code=True,
        use_fast=False,
    )

    if SET_PAD_TOKEN_AS_EOS and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # ========== bitsandbytes 4bit ==========
    bnb_config = build_bnb_config()

    # ========== 模型 ==========
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    # 可选：看一下模块名
    print_linear_module_preview(model, limit=50)

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
