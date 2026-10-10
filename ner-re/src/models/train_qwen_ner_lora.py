# -*- coding: utf-8 -*-
"""
Qwen3-8B LoRA 训练脚本（非量化版本，不使用 bitsandbytes）

assistant 输出格式：
{"entities":[{"entity":"实体文本","entity_type":"实体类型"}]}

依赖:
pip install -U torch transformers datasets peft accelerate

运行:
python train_qwen3_ner_lora.py
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


# ================= 路径配置 =================
MODEL_NAME_OR_PATH = os.getenv(
    "QWEN3_8B_MODEL_NAME",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
)

TRAIN_FILE = "../sft_data/train.jsonl"
VAL_FILE = "../sft_data/val.jsonl"
TEST_FILE = "../sft_data/test.jsonl"
OUTPUT_DIR = "../output/qwen3_8b_ner_lora"

# ================= 训练参数（4090 24GB） =================
MAX_LENGTH = 384
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LOGGING_STEPS = 20
SAVE_STEPS = 100
EVAL_STEPS = 100
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50

# ================= LoRA 参数 =================
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ================= 精度设置（自动选择） =================
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = torch.cuda.is_available() and (not BF16)

# ================= 训练 assistant-only loss =================
IGNORE_INDEX = -100

# ================= 验证时生成评估样本数 =================
EVAL_GEN_EXAMPLES = 100
MAX_NEW_TOKENS = 96


def print_rank0(msg: str):
    print(msg, flush=True)


def load_model_and_tokenizer():
    print_rank0(f"Loading tokenizer from: {MODEL_NAME_OR_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        use_fast=True          # Qwen3 推荐使用 fast tokenizer
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32),
    }

    print_rank0(f"Loading model from: {MODEL_NAME_OR_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        **model_kwargs
    )

    # 关闭 cache 以支持 gradient checkpointing
    model.config.use_cache = False

    # LoRA 配置（针对 Qwen3 的标准模块）
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# 全局 tokenizer，供 preprocess 使用
tokenizer = None


def build_prompt_and_response(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """
    从 chat messages 中拆出 prompt 和 response
    默认 messages 格式:
      system (可选)
      user
      assistant
    """
    global tokenizer

    if len(messages) < 2:
        raise ValueError("messages 至少需要两轮")

    prompt_messages = messages[:-1]
    assistant_message = messages[-1]

    if assistant_message["role"] != "assistant":
        raise ValueError("最后一条消息必须是 assistant")

    response_text = assistant_message["content"]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # 备用模板（Qwen 风格）
        chunks = []
        for m in prompt_messages:
            role = m["role"]
            content = m["content"]
            chunks.append(f"<|{role}|>\n{content}")
        chunks.append("<|assistant|>\n")
        prompt_text = "\n".join(chunks)

    return {
        "prompt_text": prompt_text,
        "response_text": response_text
    }


def preprocess_function(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    只对 assistant 输出计算 loss
    """
    global tokenizer

    converted = build_prompt_and_response(example["messages"])
    prompt_text = converted["prompt_text"]
    response_text = converted["response_text"]

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False
    )["input_ids"]

    response_ids = tokenizer(
        response_text,
        add_special_tokens=False
    )["input_ids"]

    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]

    # 截断
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class NERDataCollator:
    tokenizer: Any
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def normalize_entities_from_obj(obj: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    规范化为:
    (entity, entity_type)
    """
    if not isinstance(obj, dict):
        return []

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return []

    normalized = set()

    for item in entities:
        if not isinstance(item, dict):
            continue

        entity = str(item.get("entity", "")).strip()
        ent_type = str(item.get("entity_type", "")).strip()

        if not entity or not ent_type:
            continue

        normalized.add((entity, ent_type))

    return sorted(normalized)


def compute_ner_metrics_from_texts(pred_texts: List[str], gold_texts: List[str]) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    strict_json_ok = 0

    for pred_text, gold_text in zip(pred_texts, gold_texts):
        pred_stripped = pred_text.strip()
        gold_stripped = gold_text.strip()

        pred_obj = try_parse_json_object(pred_stripped)
        gold_obj = try_parse_json_object(gold_stripped)

        if pred_obj is not None:
            try:
                reparsed = json.loads(pred_stripped)
                if isinstance(reparsed, dict):
                    strict_json_ok += 1
            except Exception:
                pass

        pred_set = set(normalize_entities_from_obj(pred_obj))
        gold_set = set(normalize_entities_from_obj(gold_obj))

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    strict_json_rate = strict_json_ok / len(pred_texts) if pred_texts else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "strict_json_rate": strict_json_rate,
    }


class GenerativeNERTrainer(Trainer):
    def __init__(self, *args, eval_gen_examples: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_gen_examples = eval_gen_examples

    @property
    def tok(self):
        return getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return metrics

        model = self.model
        model.eval()

        tok = self.tok
        if tok is None:
            return metrics

        sample_n = min(len(dataset), self.eval_gen_examples)
        pred_texts = []
        gold_texts = []

        for i in range(sample_n):
            item = dataset[i]

            input_ids = torch.tensor([item["input_ids"]], device=model.device)
            attention_mask = torch.tensor([item["attention_mask"]], device=model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )

            gen_ids = outputs[0][len(item["input_ids"]):]
            pred_text = tok.decode(gen_ids, skip_special_tokens=True)

            gold_ids = [x for x in item["labels"] if x != IGNORE_INDEX]
            gold_text = tok.decode(gold_ids, skip_special_tokens=True)

            pred_texts.append(pred_text)
            gold_texts.append(gold_text)

        ner_metrics = compute_ner_metrics_from_texts(pred_texts, gold_texts)
        metrics[f"{metric_key_prefix}_precision"] = ner_metrics["precision"]
        metrics[f"{metric_key_prefix}_recall"] = ner_metrics["recall"]
        metrics[f"{metric_key_prefix}_f1"] = ner_metrics["f1"]
        metrics[f"{metric_key_prefix}_strict_json_rate"] = ner_metrics["strict_json_rate"]

        self.log(metrics)
        return metrics


def main():
    global tokenizer

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()

    print_rank0("Loading dataset...")
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
            "test": TEST_FILE,
        }
    )

    print_rank0("Preprocessing dataset...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing"
    )

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    print_rank0(f"train size: {len(train_dataset)}")
    print_rank0(f"val size: {len(val_dataset)}")
    print_rank0(f"test size: {len(test_dataset)}")

    data_collator = NERDataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        bf16=BF16,
        fp16=FP16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
    )

    trainer = GenerativeNERTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        eval_gen_examples=EVAL_GEN_EXAMPLES,
    )

    print_rank0("Start training...")
    trainer.train()

    save_dir = os.path.join(OUTPUT_DIR, "best_lora")
    print_rank0(f"Saving model to {save_dir} ...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    print_rank0("Final evaluation on validation...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    print_rank0(json.dumps(val_metrics, ensure_ascii=False, indent=2))

    print_rank0("Final evaluation on test...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print_rank0(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    print_rank0("Done.")


if __name__ == "__main__":
    main()