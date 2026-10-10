
# -*- coding: utf-8 -*-
"""
GLM NER 评估脚本（完整修正版，路径与参数写死版）

功能：
1. 加载 GLM 基座 + LoRA 权重
2. 读取 test.jsonl（messages 格式）
3. 逐条生成预测 JSON
4. 解析预测结果中的 entities
5. 按 (entity, entity_type) 做严格集合匹配
6. 计算 precision / recall / f1
7. 保存逐条预测结果到 jsonl

输入数据格式：
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"entities\":[...]}"}
  ]
}
"""

import os
import re
import json
from typing import Dict, List, Tuple, Any, Set

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

# ================= 路径配置（直接写死） =================
BASE_MODEL_NAME_OR_PATH = "/root/.cache/modelscope/hub/models/ZhipuAI/glm-4-9b-chat"
LORA_MODEL_PATH = "../output/glm4_9b_ner_lora"
TEST_FILE = "../sft_data/test.jsonl"
OUTPUT_FILE = "../output/glm4_9b_ner_lora/test_predictions.jsonl"

# ================= 推理参数（直接写死） =================
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0

# ================= 其他参数 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def print0(*args, **kwargs):
    print(*args, **kwargs)


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


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


def build_infer_prompt(messages: List[Dict[str, str]], tokenizer) -> str:
    if len(messages) < 2:
        raise ValueError("messages 至少需要包含 system/user")
    prompt_messages = messages[:-1] if messages[-1]["role"] == "assistant" else messages
    return safe_apply_chat_template(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )


def extract_first_json_object(text: str) -> str:
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        return fence_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()

    return ""


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_entities_from_obj(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return []

    results = []
    seen = set()

    for item in entities:
        if not isinstance(item, dict):
            continue

        entity = normalize_text(item.get("entity", ""))
        entity_type = normalize_text(item.get("entity_type", ""))

        if not entity or not entity_type:
            continue

        key = (entity, entity_type)
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "entity": entity,
            "entity_type": entity_type
        })

    results.sort(key=lambda x: (x["entity_type"], x["entity"]))
    return results


def parse_prediction_text(pred_text: str) -> Tuple[List[Dict[str, str]], str]:
    json_text = extract_first_json_object(pred_text)
    if not json_text:
        return [], "no_json_found"

    try:
        obj = json.loads(json_text)
    except Exception:
        return [], "json_parse_error"

    ents = normalize_entities_from_obj(obj)
    return ents, "ok"


def get_gold_entities(example: Dict[str, Any]) -> List[Dict[str, str]]:
    assistant_text = example["messages"][-1]["content"]
    try:
        obj = json.loads(assistant_text)
    except Exception:
        return []
    return normalize_entities_from_obj(obj)


def entity_list_to_set(entities: List[Dict[str, str]]) -> Set[Tuple[str, str]]:
    return {(x["entity"], x["entity_type"]) for x in entities}


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_model():
    print0("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print0("[2/4] Loading config...")
    config = AutoConfig.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        trust_remote_code=True,
    )

    if hasattr(config, "seq_length"):
        config.max_length = config.seq_length

    print0(f"[Info] config.max_length = {getattr(config, 'max_length', None)}")
    print0(f"[Info] config.seq_length = {getattr(config, 'seq_length', None)}")

    print0("[3/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        config=config,
        dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    print0("[4/4] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()

    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        model.base_model.config.use_cache = False
    except Exception:
        pass

    if hasattr(model, "generation_config") and model.generation_config is not None:
        try:
            model.generation_config.use_cache = False
            model.generation_config.do_sample = DO_SAMPLE
            if not DO_SAMPLE:
                model.generation_config.temperature = None
                model.generation_config.top_p = None
        except Exception:
            pass

    if DEVICE != "cuda":
        model = model.to(DEVICE)

    print0("[Info] Model loaded.")
    return model, tokenizer


@torch.no_grad()
def generate_one(model, tokenizer, prompt_text: str) -> str:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    if DEVICE == "cuda":
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P

    outputs = model.generate(**inputs, **gen_kwargs)

    if hasattr(outputs, "sequences"):
        full_output_ids = outputs.sequences[0]
    else:
        full_output_ids = outputs[0]

    prompt_len = inputs["input_ids"].shape[1]
    output_ids = full_output_ids[prompt_len:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text.strip()


def evaluate(model, tokenizer, dataset: List[Dict[str, Any]]):
    print0("[Info] Running evaluation...")

    ensure_dir(OUTPUT_FILE)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_examples = 0
    json_ok = 0
    json_fail = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
            messages = example["messages"]
            prompt_text = build_infer_prompt(messages, tokenizer)

            try:
                pred_text = generate_one(model, tokenizer, prompt_text)
                pred_entities, parse_status = parse_prediction_text(pred_text)
            except Exception as e:
                pred_text = ""
                pred_entities = []
                parse_status = f"generate_error: {type(e).__name__}: {str(e)}"

            gold_entities = get_gold_entities(example)

            pred_set = entity_list_to_set(pred_entities)
            gold_set = entity_list_to_set(gold_entities)

            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_examples += 1

            if parse_status == "ok":
                json_ok += 1
            else:
                json_fail += 1

            record = {
                "id": idx,
                "parse_status": parse_status,
                "prompt": prompt_text,
                "raw_prediction": pred_text,
                "pred_entities": pred_entities,
                "gold_entities": gold_entities,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = compute_prf(total_tp, total_fp, total_fn)
    metrics.update({
        "total_examples": total_examples,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "json_ok": json_ok,
        "json_fail": json_fail,
        "json_ok_rate": json_ok / total_examples if total_examples > 0 else 0.0,
        "output_file": OUTPUT_FILE,
    })
    return metrics


def main():
    print0("=" * 80)
    print0("GLM NER Evaluation")
    print0("=" * 80)
    print0(f"BASE_MODEL_NAME_OR_PATH = {BASE_MODEL_NAME_OR_PATH}")
    print0(f"LORA_MODEL_PATH         = {LORA_MODEL_PATH}")
    print0(f"TEST_FILE               = {TEST_FILE}")
    print0(f"OUTPUT_FILE             = {OUTPUT_FILE}")
    print0(f"DEVICE                  = {DEVICE}")
    print0("=" * 80)

    dataset = load_jsonl(TEST_FILE)
    print0(f"Loaded test samples: {len(dataset)}")

    model, tokenizer = load_model()
    metrics = evaluate(model, tokenizer, dataset)

    print0("\n" + "=" * 80)
    print0("Final Metrics")
    print0("=" * 80)
    print0(json.dumps(metrics, ensure_ascii=False, indent=2))

    metric_file = OUTPUT_FILE + ".metrics.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print0(f"\nSaved predictions to: {OUTPUT_FILE}")
    print0(f"Saved metrics to: {metric_file}")


if __name__ == "__main__":
    main()
