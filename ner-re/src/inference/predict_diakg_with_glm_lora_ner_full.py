# -*- coding: utf-8 -*-
"""
一体化脚本：
GLM-4-9B + LoRA 做 NER
→ 直接在 diakg 全量数据上预测
→ 同时计算 F1
→ 保存逐句预测结果与最终指标

运行:
python predict_and_eval_glm_ner.py
"""

import os
import json
import re
from typing import Set

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel


# ================= 配置 =================
DATA_DIR = "../diakg_data"

BASE_MODEL = "/root/.cache/modelscope/hub/models/ZhipuAI/glm-4-9b-chat"
LORA_PATH = "../output/glm4_9b_ner_lora"

OUTPUT_DIR = "../output/glm4_9b_ner_predict_eval"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "diakg_predictions.jsonl")
METRIC_FILE = os.path.join(OUTPUT_DIR, "diakg_metrics.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MAX_LEN = 1024
MAX_NEW_TOKENS = 256

ENTITY_TYPES = [
    'Disease', 'Drug', 'Test_items', 'Anatomy', 'Test_Value',
    'Class', 'ADE', 'Treatment', 'Test', 'Symptom',
    'Method', 'Amount', 'Level', 'Pathogenesis', 'Reason',
    'Frequency', 'Operation', 'Duration'
]
ENTITY_TYPE_SET = set(ENTITY_TYPES)


# ================= 基础工具 =================
def norm(x):
    return " ".join(str(x).strip().split())


def safe_get_sentence(sent):
    for k in ["sentence", "text", "sent", "content"]:
        if k in sent and sent[k]:
            return norm(sent[k])
    return ""


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ================= prompt =================
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

def build_prompt(sentence, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "请抽取下面医学文本中的所有明确医学命名实体，并严格输出 JSON。\n\n"
                f"文本：{sentence}\n\n"
                f"实体类型集合：{ENTITY_TYPES}\n\n"
                "输出格式必须为："
                "{\"entities\":[{\"entity\":\"实体文本\",\"entity_type\":\"实体类型\"}]}。\n"
                "不确定时不要输出，不要解释。"
            )
        }
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    chunks = []
    for m in messages:
        chunks.append(f"<|{m['role']}|>\n{m['content']}")
    chunks.append("<|assistant|>\n")
    return "\n".join(chunks)


# ================= JSON解析 =================
def extract_json(text):
    text = text.strip()
    if not text:
        return ""

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1)

    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def parse_pred(text, sentence):
    js = extract_json(text)
    if not js:
        return set(), "no_json_found"

    try:
        obj = json.loads(js)
    except Exception:
        return set(), "json_parse_error"

    ents = set()
    for e in obj.get("entities", []):
        if not isinstance(e, dict):
            continue

        ent = norm(e.get("entity", ""))
        typ = norm(e.get("entity_type", ""))

        if ent and typ and typ in ENTITY_TYPE_SET and ent in sentence:
            ents.add((ent, typ))

    return ents, "ok"


def parse_gold(entities, sentence):
    res = set()
    for e in entities:
        ent = norm(e.get("entity", "") or e.get("entity_name", ""))
        typ = norm(e.get("entity_type", "") or e.get("type", ""))

        if ent and typ and typ in ENTITY_TYPE_SET and ent in sentence:
            res.add((ent, typ))
    return res


# ================= 模型 =================
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if hasattr(config, "seq_length"):
        config.max_length = config.seq_length

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        config=config,
        dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None
    )

    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        model.base_model.config.use_cache = False
    except Exception:
        pass

    if DEVICE != "cuda":
        model = model.to(DEVICE)

    return model, tokenizer


@torch.no_grad()
def infer(model, tokenizer, sentence):
    prompt = build_prompt(sentence, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    res = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(res, skip_special_tokens=True).strip()


# ================= 指标 =================
def calc_all(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


# ================= 主逻辑 =================
def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading model...", flush=True)
    model, tokenizer = load_model()

    tp = fp = fn = 0
    json_ok = 0
    json_fail = 0
    total = 0

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for file in files:
            path = os.path.join(DATA_DIR, file)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            docs = data if isinstance(data, list) else data.get("data", [])
            if not docs and isinstance(data, dict):
                docs = [data]

            for doc_idx, doc in enumerate(docs):
                for para_idx, para in enumerate(doc.get("paragraphs", [])):
                    for sent_idx, sent in enumerate(tqdm(para.get("sentences", []), desc=file, leave=False)):
                        sentence = safe_get_sentence(sent)
                        if not sentence:
                            continue

                        gold = parse_gold(sent.get("entities", []), sentence)

                        try:
                            pred_text = infer(model, tokenizer, sentence)
                            pred, parse_status = parse_pred(pred_text, sentence)
                        except Exception as e:
                            pred_text = ""
                            pred = set()
                            parse_status = f"generate_error: {type(e).__name__}: {str(e)}"

                        if parse_status == "ok":
                            json_ok += 1
                        else:
                            json_fail += 1

                        cur_tp = len(pred & gold)
                        cur_fp = len(pred - gold)
                        cur_fn = len(gold - pred)

                        tp += cur_tp
                        fp += cur_fp
                        fn += cur_fn
                        total += 1

                        record = {
                            "filename": file,
                            "doc_idx": doc_idx,
                            "para_idx": para_idx,
                            "sent_idx": sent_idx,
                            "sentence_text": sentence,
                            "parse_status": parse_status,
                            "raw_prediction": pred_text,
                            "pred_entities": [
                                {"entity": x[0], "entity_type": x[1]}
                                for x in sorted(pred)
                            ],
                            "gold_entities": [
                                {"entity": x[0], "entity_type": x[1]}
                                for x in sorted(gold)
                            ],
                            "tp": cur_tp,
                            "fp": cur_fp,
                            "fn": cur_fn,
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                        if total % 50 == 0:
                            p, r, f1 = calc_all(tp, fp, fn)
                            print(
                                f"Progress: {total} | JSON: {json_ok/total:.4f} | "
                                f"P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}",
                                flush=True
                            )

    precision, recall, f1 = calc_all(tp, fp, fn)

    metrics = {
        "total_sentences": total,
        "json_ok": json_ok,
        "json_fail": json_fail,
        "json_ok_rate": json_ok / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "output_file": OUTPUT_FILE,
    }

    with open(METRIC_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n====== FINAL RESULT ======", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"\nSaved predictions to: {OUTPUT_FILE}", flush=True)
    print(f"Saved metrics to: {METRIC_FILE}", flush=True)


if __name__ == "__main__":
    main()
