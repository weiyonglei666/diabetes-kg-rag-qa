#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GLM-4-9B-Chat + QLoRA 医学关系抽取 推理与评估脚本
可直接替换版本

支持：
1. 加载 4bit 底座
2. 加载 LoRA adapter
3. 单条预测
4. 批量评估 dev/test

本版本额外修复：
- 规避 GLM-4 remote code 在 generate 时的 past_key_values/cache 兼容问题
- 显式关闭 use_cache
- 清理 generation_config 中可能遗留的 sampling 参数
"""

import os
import re
import json
import traceback
import torch
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# =========================================================
# 配置区：直接改这里
# =========================================================

# ===== 模型路径 =====
BASE_MODEL_PATH = "/root/.cache/modelscope/hub/models/ZhipuAI/glm-4-9b-chat"
LORA_ADAPTER_PATH = "../glm4_re_qlora_output"

# ===== 数据路径 =====
EVAL_FILE = "../glm4_re_data/dev.jsonl"   # 可改成 ../glm4_re_data/test.jsonl
OUTPUT_DIR = "../glm4_re_qlora_infer_eval_output"
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.jsonl")
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.json")

# ===== 推理模式 =====
RUN_SINGLE_EXAMPLE_FIRST = True
RUN_BATCH_EVAL = True

# ===== 单条测试样例 =====
SINGLE_EXAMPLE = {
    "sentence": "中国成人2型糖尿病胰岛素促泌剂应用的专家共识",
    "head": {"text": "胰岛素促泌剂", "type": "Drug", "start": 9, "end": 15},
    "tail": {"text": "2型糖尿病", "type": "Disease", "start": 4, "end": 9},
    "candidate_relations": ["No_Relation", "Drug_Disease"],
    "gold_relation": "Drug_Disease",
}

# ===== 精度 =====
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = not USE_BF16

# ===== 4bit 量化 =====
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# ===== 推理参数 =====
MAX_NEW_TOKENS = 64
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0

# ===== 约束 =====
USE_CANDIDATE_RELATION_CONSTRAINT = True
FALLBACK_TO_NO_RELATION = True

# ===== 标签集合 =====
RELATION_TYPES = [
    "No_Relation",
    "Anatomy_Disease",
    "Drug_Disease",
    "Test_items_Disease",
    "Class_Disease",
    "ADE_Drug",
    "Treatment_Disease",
    "Test_Disease",
    "Symptom_Disease",
    "Amount_Drug",
    "Method_Drug",
    "Reason_Disease",
    "Pathogenesis_Disease",
    "Frequency_Drug",
    "Duration_Drug",
    "Operation_Disease",
    "ADE_Disease",
]

# ===== 与训练时一致的类型约束 =====
RELATION_CONSTRAINTS = {
    "Anatomy_Disease": ("Anatomy", "Disease"),
    "Drug_Disease": ("Drug", "Disease"),
    "Test_items_Disease": ("Test_items", "Disease"),
    "Class_Disease": ("Class", "Disease"),
    "ADE_Drug": ("ADE", "Drug"),
    "Treatment_Disease": ("Treatment", "Disease"),
    "Test_Disease": ("Test", "Disease"),
    "Symptom_Disease": ("Symptom", "Disease"),
    "Amount_Drug": ("Amount", "Drug"),
    "Method_Drug": ("Method", "Drug"),
    "Reason_Disease": ("Reason", "Disease"),
    "Pathogenesis_Disease": ("Pathogenesis", "Disease"),
    "Frequency_Drug": ("Frequency", "Drug"),
    "Duration_Drug": ("Duration", "Drug"),
    "Operation_Disease": ("Operation", "Disease"),
    "ADE_Disease": ("ADE", "Disease"),
}

SYSTEM_PROMPT = (
    "你是一个医学关系抽取模型。"
    "请根据给定句子和两个实体，判断它们之间的关系。"
    "只输出JSON，不要输出解释，不要输出额外文本。"
)


# =========================================================
# 工具函数
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_config():
    print("=" * 80)
    print("GLM-4-9B-Chat QLoRA Inference + Evaluation Config")
    print("=" * 80)
    config = {
        "BASE_MODEL_PATH": BASE_MODEL_PATH,
        "LORA_ADAPTER_PATH": LORA_ADAPTER_PATH,
        "EVAL_FILE": EVAL_FILE,
        "OUTPUT_DIR": OUTPUT_DIR,
        "USE_BF16": USE_BF16,
        "USE_FP16": USE_FP16,
        "USE_4BIT": USE_4BIT,
        "BNB_4BIT_QUANT_TYPE": BNB_4BIT_QUANT_TYPE,
        "BNB_4BIT_USE_DOUBLE_QUANT": BNB_4BIT_USE_DOUBLE_QUANT,
        "BNB_4BIT_COMPUTE_DTYPE": str(BNB_4BIT_COMPUTE_DTYPE),
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "DO_SAMPLE": DO_SAMPLE,
        "TEMPERATURE": TEMPERATURE,
        "TOP_P": TOP_P,
        "USE_CANDIDATE_RELATION_CONSTRAINT": USE_CANDIDATE_RELATION_CONSTRAINT,
        "FALLBACK_TO_NO_RELATION": FALLBACK_TO_NO_RELATION,
        "RUN_SINGLE_EXAMPLE_FIRST": RUN_SINGLE_EXAMPLE_FIRST,
        "RUN_BATCH_EVAL": RUN_BATCH_EVAL,
    }
    for k, v in config.items():
        print(f"{k}: {v}")
    print("=" * 80)


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


def build_bnb_config():
    if not USE_4BIT:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
    )


def cleanup_generation_config(model):
    """
    清理 generation_config，避免遗留的 sampling 参数触发无效警告，
    并显式关闭 use_cache。
    """
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.do_sample = DO_SAMPLE
            model.generation_config.use_cache = False

            if not DO_SAMPLE:
                # 某些 transformers 版本允许设为 None，能避免 warning
                try:
                    model.generation_config.temperature = None
                except Exception:
                    pass
                try:
                    model.generation_config.top_p = None
                except Exception:
                    pass
    except Exception:
        pass


def load_model_and_tokenizer():
    print("\n加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_ADAPTER_PATH if os.path.exists(LORA_ADAPTER_PATH) else BASE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None and tokenizer.eos_token is not None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    tokenizer.padding_side = "left"

    print("加载 4bit base model ...")
    bnb_config = build_bnb_config()

    # 对 trust_remote_code 模型，torch_dtype 往往兼容性更稳
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )

    if not os.path.exists(LORA_ADAPTER_PATH):
        raise FileNotFoundError(f"LoRA adapter 路径不存在: {LORA_ADAPTER_PATH}")

    print("加载 LoRA adapter ...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    # 关键：关闭 cache，规避 ChatGLM/GLM remote code 与当前 transformers 的缓存兼容问题
    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
            model.base_model.config.use_cache = False
    except Exception:
        pass

    cleanup_generation_config(model)

    model.eval()
    return model, tokenizer


def legal_relations_for_entity_types(head_type: str, tail_type: str):
    rels = []
    for rel, (h_type, t_type) in RELATION_CONSTRAINTS.items():
        if head_type == h_type and tail_type == t_type:
            rels.append(rel)
    return rels


def make_user_prompt(sentence, head, tail, candidate_relations, include_offsets=True):
    if include_offsets:
        entity1 = json.dumps(
            {
                "text": head["text"],
                "type": head["type"],
                "start": head.get("start", -1),
                "end": head.get("end", -1),
            },
            ensure_ascii=False,
        )
        entity2 = json.dumps(
            {
                "text": tail["text"],
                "type": tail["type"],
                "start": tail.get("start", -1),
                "end": tail.get("end", -1),
            },
            ensure_ascii=False,
        )
    else:
        entity1 = json.dumps({"text": head["text"], "type": head["type"]}, ensure_ascii=False)
        entity2 = json.dumps({"text": tail["text"], "type": tail["type"]}, ensure_ascii=False)

    prompt = (
        "任务：判断给定两个实体之间的关系。\n"
        "只输出 JSON。\n\n"
        f"句子：{sentence}\n"
        f"实体1：{entity1}\n"
        f"实体2：{entity2}\n"
        f"可选关系：{json.dumps(candidate_relations, ensure_ascii=False)}"
    )
    return prompt


def build_glm_infer_text(system_prompt, user_prompt):
    return (
        f"系统：{system_prompt}\n"
        f"用户：{user_prompt}\n"
        f"助手："
    )


def parse_relation_from_json_text(text: str):
    if text is None:
        return None, "none"

    text = strip_think_blocks(text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "relation" in obj:
            return obj["relation"], "direct_json"
    except Exception:
        pass

    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "relation" in obj:
                return obj["relation"], "regex_json"
        except Exception:
            pass

    m = re.search(r'"relation"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1), "regex_field"

    simple = text.strip().replace("`", "").replace("\n", " ")
    if simple in RELATION_TYPES:
        return simple, "plain_label"

    return None, "parse_failed"


def constrained_relation(pred_relation, head_type, tail_type, candidate_relations=None):
    if pred_relation is None:
        return "No_Relation", "none_to_no_relation"

    if pred_relation not in RELATION_TYPES:
        return "No_Relation", "unknown_label"

    if pred_relation == "No_Relation":
        return pred_relation, "ok"

    if USE_CANDIDATE_RELATION_CONSTRAINT and candidate_relations is not None:
        if pred_relation not in candidate_relations:
            return "No_Relation", "not_in_candidates"

    if pred_relation in RELATION_CONSTRAINTS:
        h, t = RELATION_CONSTRAINTS[pred_relation]
        if head_type != h or tail_type != t:
            return "No_Relation", "type_mismatch"

    return pred_relation, "ok"


def prepare_inputs(tokenizer, input_text, device):
    """
    对 remote code 模型更保守地准备输入，尽量只传常用字段。
    """
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )

    allowed_keys = {"input_ids", "attention_mask", "position_ids"}
    filtered = {}
    for k, v in inputs.items():
        if k in allowed_keys and v is not None:
            filtered[k] = v.to(device)
    return filtered


def decode_new_tokens(tokenizer, outputs, input_len):
    gen_ids = outputs[0][input_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return strip_think_blocks(gen_text)


def generate_prediction(model, tokenizer, input_text):
    first_param_device = next(model.parameters()).device
    inputs = prepare_inputs(tokenizer, input_text, first_param_device)

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": False,   # 关键：规避 past_key_values 问题
        "return_dict_in_generate": False,
    }

    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P

    input_len = inputs["input_ids"].shape[1]

    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        return decode_new_tokens(tokenizer, outputs, input_len)
    except Exception as e:
        print("\n[Warn] 首次 generate 失败，尝试更保守兜底模式 ...")
        print(f"[Warn] Error type: {type(e).__name__}")
        print(f"[Warn] Error msg : {e}")

        # 再次清理 generation_config
        cleanup_generation_config(model)

        # 更保守的兜底参数
        fallback_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": False,
            "return_dict_in_generate": False,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **fallback_kwargs)
        return decode_new_tokens(tokenizer, outputs, input_len)


def predict_single(model, tokenizer, sentence, head, tail, candidate_relations):
    user_prompt = make_user_prompt(sentence, head, tail, candidate_relations, include_offsets=True)
    input_text = build_glm_infer_text(SYSTEM_PROMPT, user_prompt)

    raw_output = generate_prediction(model, tokenizer, input_text)
    parsed_relation, parse_method = parse_relation_from_json_text(raw_output)
    final_relation, constraint_status = constrained_relation(
        parsed_relation,
        head["type"],
        tail["type"],
        candidate_relations=candidate_relations,
    )

    return {
        "raw_output": raw_output,
        "parsed_relation": parsed_relation,
        "parse_method": parse_method,
        "final_relation": final_relation,
        "constraint_status": constraint_status,
    }


def extract_user_from_messages(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def extract_assistant_gold_from_messages(messages):
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def get_gold_relation_from_record(record):
    meta = record.get("meta", {})
    if "label" in meta:
        return meta["label"]

    assistant_content = extract_assistant_gold_from_messages(record.get("messages", []))
    relation, _ = parse_relation_from_json_text(assistant_content)
    return relation


def get_candidate_relations_from_user_text(user_text):
    m = re.search(r"可选关系：(\[.*\])", user_text, flags=re.S)
    if not m:
        return None
    arr_text = m.group(1).strip()
    try:
        arr = json.loads(arr_text)
        if isinstance(arr, list):
            return arr
    except Exception:
        return None
    return None


def get_entity_info_from_meta(record):
    meta = record.get("meta", {})
    head = {
        "text": meta.get("head_text", ""),
        "type": meta.get("head_type", ""),
        "start": -1,
        "end": -1,
    }
    tail = {
        "text": meta.get("tail_text", ""),
        "type": meta.get("tail_type", ""),
        "start": -1,
        "end": -1,
    }
    return head, tail


def evaluate_classification(golds, preds, labels):
    label_to_idx = {label: i for i, label in enumerate(labels)}
    confusion = [[0 for _ in labels] for _ in labels]

    for g, p in zip(golds, preds):
        if g not in label_to_idx or p not in label_to_idx:
            continue
        confusion[label_to_idx[g]][label_to_idx[p]] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(len(labels)))
    accuracy = correct / total if total > 0 else 0.0

    per_label = {}
    macro_p = macro_r = macro_f1 = 0.0
    weighted_p = weighted_r = weighted_f1 = 0.0
    total_support = 0

    tp_sum = fp_sum = fn_sum = 0

    for i, label in enumerate(labels):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(len(labels)) if r != i)
        fn = sum(confusion[i][c] for c in range(len(labels)) if c != i)
        support = sum(confusion[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support
        total_support += support

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    n_labels = len(labels)
    macro_p /= n_labels
    macro_r /= n_labels
    macro_f1 /= n_labels

    weighted_p = weighted_p / total_support if total_support > 0 else 0.0
    weighted_r = weighted_r / total_support if total_support > 0 else 0.0
    weighted_f1 = weighted_f1 / total_support if total_support > 0 else 0.0

    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0 else 0.0
    )

    pos_labels = [x for x in labels if x != "No_Relation"]
    pos_indices = [label_to_idx[x] for x in pos_labels]

    pos_tp_sum = pos_fp_sum = pos_fn_sum = 0
    for i in pos_indices:
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(len(labels)) if r != i)
        fn = sum(confusion[i][c] for c in range(len(labels)) if c != i)
        pos_tp_sum += tp
        pos_fp_sum += fp
        pos_fn_sum += fn

    pos_micro_p = pos_tp_sum / (pos_tp_sum + pos_fp_sum) if (pos_tp_sum + pos_fp_sum) > 0 else 0.0
    pos_micro_r = pos_tp_sum / (pos_tp_sum + pos_fn_sum) if (pos_tp_sum + pos_fn_sum) > 0 else 0.0
    pos_micro_f1 = (
        2 * pos_micro_p * pos_micro_r / (pos_micro_p + pos_micro_r)
        if (pos_micro_p + pos_micro_r) > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "positive_micro_precision_excluding_no_relation": pos_micro_p,
        "positive_micro_recall_excluding_no_relation": pos_micro_r,
        "positive_micro_f1_excluding_no_relation": pos_micro_f1,
        "per_label": per_label,
        "labels": labels,
        "confusion_matrix": confusion,
    }


def get_top_confusions(golds, preds, topk=10):
    counter = Counter()
    for g, p in zip(golds, preds):
        if g != p:
            counter[(g, p)] += 1
    return [
        {"gold": g, "pred": p, "count": c}
        for (g, p), c in counter.most_common(topk)
    ]


def batch_evaluate(model, tokenizer, eval_file):
    records = load_jsonl(eval_file)
    print(f"\n加载评估数据: {eval_file}")
    print(f"样本数: {len(records)}")

    prediction_records = []
    golds = []
    preds = []

    parse_method_counter = Counter()
    constraint_status_counter = Counter()

    for idx, record in enumerate(records):
        messages = record.get("messages", [])
        meta = record.get("meta", {})

        gold = get_gold_relation_from_record(record)
        user_text = extract_user_from_messages(messages)
        candidate_relations = get_candidate_relations_from_user_text(user_text)

        head, tail = get_entity_info_from_meta(record)
        if candidate_relations is None:
            candidate_relations = ["No_Relation"] + legal_relations_for_entity_types(
                head["type"], tail["type"]
            )

        input_text = build_glm_infer_text(SYSTEM_PROMPT, user_text)

        try:
            raw_output = generate_prediction(model, tokenizer, input_text)
            parsed_relation, parse_method = parse_relation_from_json_text(raw_output)
            final_relation, constraint_status = constrained_relation(
                parsed_relation,
                head["type"],
                tail["type"],
                candidate_relations=candidate_relations,
            )
        except Exception as e:
            raw_output = f"[INFERENCE_ERROR] {type(e).__name__}: {e}"
            parsed_relation = None
            parse_method = "inference_error"
            final_relation = "No_Relation" if FALLBACK_TO_NO_RELATION else None
            constraint_status = "generation_failed"

        if final_relation is None and FALLBACK_TO_NO_RELATION:
            final_relation = "No_Relation"

        parse_method_counter[parse_method] += 1
        constraint_status_counter[constraint_status] += 1

        golds.append(gold)
        preds.append(final_relation)

        prediction_records.append({
            "idx": idx,
            "gold": gold,
            "pred": final_relation,
            "correct": gold == final_relation,
            "raw_output": raw_output,
            "parsed_relation": parsed_relation,
            "parse_method": parse_method,
            "constraint_status": constraint_status,
            "meta": meta,
        })

        if (idx + 1) % 20 == 0:
            print(f"已评估 {idx + 1}/{len(records)}")

    metrics = evaluate_classification(golds, preds, RELATION_TYPES)
    metrics["num_examples"] = len(records)
    metrics["parse_method_distribution"] = dict(parse_method_counter)
    metrics["constraint_status_distribution"] = dict(constraint_status_counter)
    metrics["top_confusions"] = get_top_confusions(golds, preds, topk=15)

    return metrics, prediction_records


def pretty_print_metrics(metrics):
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"num_examples: {metrics['num_examples']}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"macro_f1: {metrics['macro_f1']:.6f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.6f}")
    print(f"micro_f1: {metrics['micro_f1']:.6f}")
    print(
        "positive_micro_f1_excluding_no_relation: "
        f"{metrics['positive_micro_f1_excluding_no_relation']:.6f}"
    )

    print("\n解析方式分布:")
    for k, v in metrics["parse_method_distribution"].items():
        print(f"  {k}: {v}")

    print("\n约束状态分布:")
    for k, v in metrics["constraint_status_distribution"].items():
        print(f"  {k}: {v}")

    print("\n每类指标:")
    for label, m in metrics["per_label"].items():
        print(
            f"{label:25s} | "
            f"P={m['precision']:.4f} "
            f"R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} "
            f"support={m['support']}"
        )

    print("\n最常见混淆:")
    for item in metrics["top_confusions"]:
        print(f"gold={item['gold']}, pred={item['pred']}, count={item['count']}")
    print("=" * 80)


def run_single_demo(model, tokenizer):
    print("\n开始单条样例预测...")
    result = predict_single(
        model=model,
        tokenizer=tokenizer,
        sentence=SINGLE_EXAMPLE["sentence"],
        head=SINGLE_EXAMPLE["head"],
        tail=SINGLE_EXAMPLE["tail"],
        candidate_relations=SINGLE_EXAMPLE["candidate_relations"],
    )

    print("-" * 80)
    print("单条样例")
    print(json.dumps(SINGLE_EXAMPLE, ensure_ascii=False, indent=2))
    print("\n模型输出:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("-" * 80)


def main():
    ensure_dir(OUTPUT_DIR)
    print_config()

    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print("\n模型加载失败：")
        print(f"{type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise

    if RUN_SINGLE_EXAMPLE_FIRST:
        try:
            run_single_demo(model, tokenizer)
        except Exception as e:
            print("\n单条样例预测失败：")
            print(f"{type(e).__name__}: {e}")
            print(traceback.format_exc())
            raise

    if RUN_BATCH_EVAL:
        metrics, prediction_records = batch_evaluate(model, tokenizer, EVAL_FILE)
        pretty_print_metrics(metrics)

        save_json(METRICS_FILE, metrics)
        save_jsonl(PREDICTIONS_FILE, prediction_records)

        print(f"\n指标已保存到: {METRICS_FILE}")
        print(f"预测明细已保存到: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
