# -*- coding: utf-8 -*-
"""
GLM NER QLoRA 训练脚本（完整修正版）

修复内容：
1. 兼容 GLM-4-9B remote code 访问 config.max_length，但配置里只有 seq_length 的问题
2. 使用 dtype 替代 torch_dtype
3. 兼容 messages 格式的 JSONL 数据
4. 仅在 assistant 回复部分计算 loss
5. 使用 4bit + LoRA / QLoRA，适合单卡 24GB

数据格式：
每行一个 JSON，例如：
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"entities\":[...]}"}
  ]
}
"""

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ================= 路径配置（按你现在的目录写） =================
MODEL_NAME_OR_PATH = "/root/.cache/modelscope/hub/models/ZhipuAI/glm-4-9b-chat"
TRAIN_FILE = "../sft_data/train.jsonl"
VAL_FILE = "../sft_data/val.jsonl"
TEST_FILE = "../sft_data/test.jsonl"
OUTPUT_DIR = "../output/glm4_9b_ner_lora"

# ================= 训练参数 =================
MAX_SEQ_LENGTH = 1024
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 200
EVAL_STEPS = 200
SAVE_TOTAL_LIMIT = 2
SEED = 42

# ================= LoRA 参数 =================
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ================= 精度参数 =================
USE_BF16 = True
USE_FP16 = False

# ================= QLoRA 参数 =================
LOAD_IN_4BIT = True
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16" if USE_BF16 else "float16"

# ================= 其他参数 =================
GRADIENT_CHECKPOINTING = True
SAVE_SAFETENSORS = True
REPORT_TO = "none"
IGNORE_INDEX = -100


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_rank0(*args, **kwargs):
    print(*args, **kwargs)


def load_jsonl_preview(path: str, n: int = 2):
    print_rank0(f"\n[Preview] {path}")
    if not os.path.exists(path):
        print_rank0(f"[WARN] 文件不存在: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            preview = json.dumps(obj, ensure_ascii=False, indent=2)
            print_rank0(preview[:1200])


def get_bnb_config():
    if not LOAD_IN_4BIT:
        return None

    compute_dtype = torch.bfloat16 if BNB_4BIT_COMPUTE_DTYPE == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def find_all_linear_names(model) -> List[str]:
    linear_cls = torch.nn.Linear
    lora_module_names = set()

    try:
        import bitsandbytes as bnb
        linear4bit_cls = bnb.nn.Linear4bit
        linear_classes = (linear_cls, linear4bit_cls)
    except Exception:
        linear_classes = (linear_cls,)

    for name, module in model.named_modules():
        if isinstance(module, linear_classes):
            parts = name.split(".")
            lora_module_names.add(parts[-1])

    for bad_name in ["lm_head", "output_layer", "output"]:
        if bad_name in lora_module_names:
            lora_module_names.remove(bad_name)

    return sorted(list(lora_module_names))


def load_model_and_tokenizer():
    print_rank0("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_rank0("[2/6] Loading config...")
    config = AutoConfig.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
    )

    # ===== 兼容 GLM-4 remote code 读取 config.max_length =====
    if not hasattr(config, "max_length"):
        if hasattr(config, "seq_length"):
            config.max_length = config.seq_length
        else:
            config.max_length = MAX_SEQ_LENGTH

    print_rank0(f"[Info] config.max_length = {getattr(config, 'max_length', None)}")
    print_rank0(f"[Info] config.seq_length = {getattr(config, 'seq_length', None)}")

    print_rank0("[3/6] Loading model...")
    quant_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        config=config,
        dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        quantization_config=quant_config,
        device_map="auto",
    )

    if LOAD_IN_4BIT:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
        )

    target_modules = find_all_linear_names(model)
    print_rank0(f"[Info] LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
        model.config.use_cache = False

    model.print_trainable_parameters()
    return model, tokenizer


def safe_apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass

    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|{role}|>\n{content}\n"
    if add_generation_prompt:
        text += "<|assistant|>\n"
    return text


def build_prompt_and_answer(messages: List[Dict[str, str]], tokenizer):
    if len(messages) < 2:
        raise ValueError("messages 至少需要包含 user / assistant")

    if messages[-1]["role"] != "assistant":
        raise ValueError("最后一条消息必须是 assistant")

    prompt_messages = messages[:-1]
    answer = messages[-1]["content"]

    prompt_text = safe_apply_chat_template(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )

    full_text = safe_apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )

    return prompt_text, answer, full_text


def preprocess_example(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    messages = example["messages"]
    prompt_text, answer_text, full_text = build_prompt_and_answer(messages, tokenizer)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )["input_ids"]

    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "full_text": full_text,
    }


@dataclass
class SupervisedDataCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        if self.pad_to_multiple_of is not None:
            seq_len = batch_input_ids.shape[1]
            padded_len = int(math.ceil(seq_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)
            if padded_len > seq_len:
                pad_len = padded_len - seq_len
                batch_input_ids = torch.nn.functional.pad(
                    batch_input_ids, (0, pad_len), value=self.tokenizer.pad_token_id
                )
                batch_attention_mask = torch.nn.functional.pad(
                    batch_attention_mask, (0, pad_len), value=0
                )
                batch_labels = torch.nn.functional.pad(
                    batch_labels, (0, pad_len), value=IGNORE_INDEX
                )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }


def make_datasets(tokenizer):
    print_rank0("[4/6] Loading dataset...")
    data_files = {
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    print_rank0(raw_datasets)

    print_rank0("[5/6] Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_example(x, tokenizer),
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing",
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print_rank0(f"train size = {len(train_dataset)}")
    print_rank0(f"eval size  = {len(eval_dataset)}")

    if len(train_dataset) > 0:
        one = train_dataset[0]
        valid_label_count = sum(1 for x in one["labels"] if x != IGNORE_INDEX)
        print_rank0(f"[Check] first train sample input len = {len(one['input_ids'])}")
        print_rank0(f"[Check] first train sample supervised tokens = {valid_label_count}")

    return train_dataset, eval_dataset


def build_training_args():
    eval_strategy_name = "steps"
    try:
        return TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            bf16=USE_BF16,
            fp16=USE_FP16,
            eval_strategy=eval_strategy_name,
            save_strategy="steps",
            logging_strategy="steps",
            report_to=REPORT_TO,
            lr_scheduler_type="cosine",
            seed=SEED,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            optim="paged_adamw_32bit" if LOAD_IN_4BIT else "adamw_torch",
            ddp_find_unused_parameters=False,
            save_safetensors=SAVE_SAFETENSORS,
            load_best_model_at_end=False,
        )
    except TypeError:
        return TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            bf16=USE_BF16,
            fp16=USE_FP16,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            report_to=REPORT_TO,
            lr_scheduler_type="cosine",
            seed=SEED,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            optim="paged_adamw_32bit" if LOAD_IN_4BIT else "adamw_torch",
            ddp_find_unused_parameters=False,
            save_safetensors=SAVE_SAFETENSORS,
            load_best_model_at_end=False,
        )


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_rank0("=" * 80)
    print_rank0("GLM NER QLoRA Training")
    print_rank0("=" * 80)
    print_rank0(f"MODEL_NAME_OR_PATH = {MODEL_NAME_OR_PATH}")
    print_rank0(f"TRAIN_FILE         = {TRAIN_FILE}")
    print_rank0(f"VAL_FILE           = {VAL_FILE}")
    print_rank0(f"TEST_FILE          = {TEST_FILE}")
    print_rank0(f"OUTPUT_DIR         = {OUTPUT_DIR}")
    print_rank0("-" * 80)
    print_rank0(f"MAX_SEQ_LENGTH     = {MAX_SEQ_LENGTH}")
    print_rank0(f"NUM_TRAIN_EPOCHS   = {NUM_TRAIN_EPOCHS}")
    print_rank0(f"TRAIN_BATCH_SIZE   = {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print_rank0(f"EVAL_BATCH_SIZE    = {PER_DEVICE_EVAL_BATCH_SIZE}")
    print_rank0(f"GRAD_ACC_STEPS     = {GRADIENT_ACCUMULATION_STEPS}")
    print_rank0(f"LEARNING_RATE      = {LEARNING_RATE}")
    print_rank0(f"LORA_R             = {LORA_R}")
    print_rank0(f"LORA_ALPHA         = {LORA_ALPHA}")
    print_rank0(f"LORA_DROPOUT       = {LORA_DROPOUT}")
    print_rank0(f"LOAD_IN_4BIT       = {LOAD_IN_4BIT}")
    print_rank0(f"USE_BF16           = {USE_BF16}")
    print_rank0("=" * 80)

    load_jsonl_preview(TRAIN_FILE, n=1)
    load_jsonl_preview(VAL_FILE, n=1)

    model, tokenizer = load_model_and_tokenizer()
    train_dataset, eval_dataset = make_datasets(tokenizer)
    data_collator = SupervisedDataCollator(tokenizer=tokenizer)
    training_args = build_training_args()

    print_rank0("[6/6] Building trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print_rank0("Start training...")
    train_result = trainer.train()

    print_rank0("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print_rank0("\nTraining finished.")
    print_rank0(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
