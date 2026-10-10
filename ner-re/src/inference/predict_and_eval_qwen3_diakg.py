# -*- coding: utf-8 -*-
"""
Qwen3-8B + LoRA
直接在全量 diakg 原始数据上：
1. 做实体识别预测
2. 直接计算实体级 Precision / Recall / F1
3. 保存逐句预测结果

依赖：
pip install -U torch transformers peft tqdm

运行：
python predict_and_eval_qwen3_diakg.py
"""

import os
import re
import json
from typing import Dict, List, Any, Tuple, Set

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ================= 路径配置 =================
DATA_DIR = "../diakg_data"
BASE_MODEL_NAME_OR_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
LORA_MODEL_PATH = "../output/qwen3_8b_ner_lora/checkpoint-300"


OUTPUT_DIR = "../output/qwen3_8b_ner_predict_eval"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "diakg_predictions.jsonl")
METRIC_FILE = os.path.join(OUTPUT_DIR, "diakg_metrics.json")

# ================= 推理参数 =================
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 96
DO_SAMPLE = False

# ================= 设备 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = torch.cuda.is_available() and (not BF16)
DTYPE = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

# ================= 标签定义 =================
ENTITY_TYPES = [
    'Disease', 'Drug', 'Test_items', 'Anatomy', 'Test_Value',
    'Class', 'ADE', 'Treatment', 'Test', 'Symptom',
    'Method', 'Amount', 'Level', 'Pathogenesis', 'Reason',
    'Frequency', 'Operation', 'Duration'
]
ENTITY_TYPE_SET = set(ENTITY_TYPES)

SYSTEM_PROMPT = (
    "你是中文医学实体识别专家。"
    "请从输入文本中抽取医学命名实体，并严格输出 JSON。"
    "输出格式必须为："
    "{\"entities\":[{\"entity\":\"实体文本\",\"entity_type\":\"实体类型\"}]}。"
    "实体类型只能从以下集合中选择："
    f"{ENTITY_TYPES}。"
    "要求："
    "1. 只输出 JSON，不要解释，不要补充说明；"
    "2. entity 必须是原文中的连续片段；"
    "3. 不重复输出；"
    "4. 如果没有明确实体，输出 {\"entities\":[]}；"
    "5. 不确定时不要输出；"
    "6. 不要把研究名、标题词、统计词、泛化描述词当成实体。"
)


def print0(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def safe_get_sentence_text(sent: Dict[str, Any]) -> str:
    for key in ["sentence", "text", "sent", "content"]:
        value = sent.get(key, "")
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return ""


def safe_apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        except Exception:
            pass

    chunks = []
    for m in messages:
        chunks.append(f"<|{m['role']}|>\n{m['content']}")
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "\n".join(chunks)


def build_messages(sentence_text: str) -> List[Dict[str, str]]:
    user_prompt = (
        "请抽取下面医学文本中的所有明确医学命名实体，并严格输出 JSON。\n\n"
        f"文本：{sentence_text}\n\n"
        f"实体类型集合：{ENTITY_TYPES}\n\n"
        "输出格式必须为："
        "{\"entities\":[{\"entity\":\"实体文本\",\"entity_type\":\"实体类型\"}]}。\n"
        "不确定时不要输出，不要解释。"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_prompt(sentence_text: str, tokenizer) -> str:
    messages = build_messages(sentence_text)
    return safe_apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True
    )


def extract_first_json_object(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

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


def normalize_entities_from_obj(obj: Dict[str, Any], sentence_text: str) -> List[Tuple[str, str]]:
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
        if entity_type not in ENTITY_TYPE_SET:
            continue
        if entity not in sentence_text:
            continue

        key = (entity, entity_type)
        if key in seen:
            continue
        seen.add(key)
        results.append(key)

    results.sort()
    return results


def parse_prediction_text(pred_text: str, sentence_text: str) -> Tuple[Set[Tuple[str, str]], str]:
    json_text = extract_first_json_object(pred_text)
    if not json_text:
        return set(), "no_json_found"

    try:
        obj = json.loads(json_text)
    except Exception:
        return set(), "json_parse_error"

    ents = set(normalize_entities_from_obj(obj, sentence_text))
    return ents, "ok"


def normalize_gold_entities(entities: List[Dict[str, Any]], sentence_text: str) -> Set[Tuple[str, str]]:
    results = []
    seen = set()

    for ent in entities:
        entity = normalize_text((ent.get("entity") or ent.get("entity_name") or "").strip())
        ent_type = normalize_text((ent.get("entity_type") or ent.get("type") or "").strip())

        if not entity or not ent_type:
            continue
        if ent_type not in ENTITY_TYPE_SET:
            continue
        if entity not in sentence_text:
            continue

        key = (entity, ent_type)
        if key in seen:
            continue
        seen.add(key)
        results.append(key)

    return set(results)


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def iter_all_samples(data_dir: str):
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    files.sort()

    for filename in files:
        file_path = os.path.join(data_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            yield {
                "type": "file_error",
                "filename": filename,
                "error": str(e),
            }
            continue

        docs = data if isinstance(data, list) else data.get("data", [])
        if not docs and isinstance(data, dict):
            docs = [data]

        for doc_idx, doc in enumerate(docs):
            paragraphs = doc.get("paragraphs", [])
            for para_idx, para in enumerate(paragraphs):
                sentences = para.get("sentences", [])
                for sent_idx, sent in enumerate(sentences):
                    sentence_text = safe_get_sentence_text(sent)
                    if not sentence_text:
                        continue

                    gold_entities = normalize_gold_entities(
                        sent.get("entities", []),
                        sentence_text
                    )

                    yield {
                        "type": "sample",
                        "filename": filename,
                        "doc_idx": doc_idx,
                        "para_idx": para_idx,
                        "sent_idx": sent_idx,
                        "sentence_text": sentence_text,
                        "gold_entities": gold_entities,
                    }


def load_model_and_tokenizer():
    print0(f"Loading tokenizer from: {BASE_MODEL_NAME_OR_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print0(f"Loading base model from: {BASE_MODEL_NAME_OR_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    print0(f"Loading LoRA from: {LORA_MODEL_PATH}")
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    model.eval()

    try:
        model.config.use_cache = True
    except Exception:
        pass

    if DEVICE != "cuda":
        model = model.to(DEVICE)

    return model, tokenizer


@torch.no_grad()
def generate_one(model, tokenizer, sentence_text: str) -> str:
    prompt_text = build_prompt(sentence_text, tokenizer)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    if DEVICE == "cuda":
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return pred_text.strip()


def main():
    print0("=" * 80)
    print0("Qwen3-8B LoRA DIAKG Prediction + Evaluation")
    print0("=" * 80)
    print0(f"DATA_DIR                = {DATA_DIR}")
    print0(f"BASE_MODEL_NAME_OR_PATH = {BASE_MODEL_NAME_OR_PATH}")
    print0(f"LORA_MODEL_PATH         = {LORA_MODEL_PATH}")
    print0(f"OUTPUT_FILE             = {OUTPUT_FILE}")
    print0(f"METRIC_FILE             = {METRIC_FILE}")
    print0(f"DEVICE                  = {DEVICE}")
    print0("=" * 80)

    ensure_dir(OUTPUT_DIR)

    model, tokenizer = load_model_and_tokenizer()

    samples = list(iter_all_samples(DATA_DIR))
    print0(f"Loaded items: {len(samples)}")

    total_sentences = 0
    total_errors = 0
    json_ok = 0
    json_fail = 0
    tp = 0
    fp = 0
    fn = 0
    file_errors = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(tqdm(samples, desc="Predicting+Evaluating")):
            if item["type"] == "file_error":
                file_errors.append({
                    "filename": item["filename"],
                    "error": item["error"],
                })
                total_errors += 1
                continue

            sentence_text = item["sentence_text"]
            gold_set = item["gold_entities"]

            try:
                raw_prediction = generate_one(model, tokenizer, sentence_text)
                pred_set, parse_status = parse_prediction_text(raw_prediction, sentence_text)
            except Exception as e:
                raw_prediction = ""
                pred_set = set()
                parse_status = f"generate_error: {type(e).__name__}: {str(e)}"
                total_errors += 1

            if parse_status == "ok":
                json_ok += 1
            else:
                json_fail += 1

            cur_tp = len(pred_set & gold_set)
            cur_fp = len(pred_set - gold_set)
            cur_fn = len(gold_set - pred_set)

            tp += cur_tp
            fp += cur_fp
            fn += cur_fn
            total_sentences += 1

            record = {
                "filename": item["filename"],
                "doc_idx": item["doc_idx"],
                "para_idx": item["para_idx"],
                "sent_idx": item["sent_idx"],
                "sentence_text": sentence_text,
                "parse_status": parse_status,
                "raw_prediction": raw_prediction,
                "pred_entities": [
                    {"entity": x[0], "entity_type": x[1]}
                    for x in sorted(pred_set)
                ],
                "gold_entities": [
                    {"entity": x[0], "entity_type": x[1]}
                    for x in sorted(gold_set)
                ],
                "tp": cur_tp,
                "fp": cur_fp,
                "fn": cur_fn,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if total_sentences % 50 == 0:
                approx = compute_prf(tp, fp, fn)
                print0(
                    f"[Progress] sentences={total_sentences}, "
                    f"json_ok_rate={json_ok / total_sentences:.4f}, "
                    f"f1={approx['f1']:.4f}"
                )

    metrics = compute_prf(tp, fp, fn)
    metrics.update({
        "data_dir": DATA_DIR,
        "base_model_name_or_path": BASE_MODEL_NAME_OR_PATH,
        "lora_model_path": LORA_MODEL_PATH,
        "output_file": OUTPUT_FILE,
        "metric_file": METRIC_FILE,
        "device": DEVICE,
        "total_sentences": total_sentences,
        "total_errors": total_errors,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "json_ok": json_ok,
        "json_fail": json_fail,
        "json_ok_rate": json_ok / total_sentences if total_sentences > 0 else 0.0,
        "file_errors": file_errors,
    })

    with open(METRIC_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print0("\n" + "=" * 80)
    print0("Final Metrics")
    print0("=" * 80)
    print0(json.dumps(metrics, ensure_ascii=False, indent=2))
    print0(f"\nSaved predictions to: {OUTPUT_FILE}")
    print0(f"Saved metrics to: {METRIC_FILE}")


if __name__ == "__main__":
    main()
