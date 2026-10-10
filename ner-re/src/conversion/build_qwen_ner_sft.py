# -*- coding: utf-8 -*-
"""
把 diakg 原始 JSON 转成 Qwen SFT 数据（chat messages 格式）
输出:
  ../sft_data/train.jsonl
  ../sft_data/val.jsonl
  ../sft_data/test.jsonl

目标:
1. 以提升 NER precision / F1 为优先
2. assistant 只输出最核心结构:
   {"entities":[{"entity":"实体文本","entity_type":"实体类型"}]}
3. 降低任务复杂度，减少模型多输出实体
4. 支持长尾类别适度过采样
"""

import os
import json
import random
from collections import Counter
from typing import List, Dict, Any, Tuple

DATA_DIR = "../diakg_data"
OUT_DIR = "../sft_data"

ENTITY_TYPES = [
    "Disease", "Drug", "Test_items", "Anatomy", "Test_Value",
    "Class", "ADE", "Treatment", "Test", "Symptom",
    "Method", "Amount", "Level", "Pathogenesis", "Reason",
    "Frequency", "Operation", "Duration"
]
ENTITY_TYPE_SET = set(ENTITY_TYPES)

TAIL_ENTITY_TYPES = {
    "Duration", "Operation", "Frequency", "Reason", "Pathogenesis"
}

# 长尾类别适度增强，别太大，避免带歪主分布
TAIL_REPEAT = 2

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

RANDOM_SEED = 42

# 过滤过长句，先提升可学性与边界稳定性
MAX_SENTENCE_LEN = 180

# 保留一定空样本，抑制模型过度乱报
KEEP_EMPTY_SAMPLE_PROB = 1

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


def safe_get_sentence_text(sent: Dict[str, Any]) -> str:
    for key in ["sentence", "text", "sent", "content"]:
        value = sent.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_entities(entities: List[Dict[str, Any]], sentence_text: str) -> List[Dict[str, str]]:
    """
    输出最小任务标签:
    {"entity":"...", "entity_type":"..."}
    """
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

        results.append({
            "entity": entity,
            "entity_type": ent_type
        })

    # 固定排序，降低学习难度
    results.sort(key=lambda x: (x["entity_type"], x["entity"]))
    return results


def should_keep_sample(sentence_text: str, entities: List[Dict[str, str]]) -> bool:
    if not sentence_text:
        return False

    if len(sentence_text) > MAX_SENTENCE_LEN:
        return False

    if len(entities) == 0:
        return random.random() < KEEP_EMPTY_SAMPLE_PROB

    return True


def build_messages(sentence_text: str, entities: List[Dict[str, str]]) -> Dict[str, Any]:
    user_prompt = (
        "请抽取下面医学文本中的所有明确医学命名实体，并严格输出 JSON。\n\n"
        f"文本：{sentence_text}\n\n"
        f"实体类型集合：{ENTITY_TYPES}\n\n"
        "输出格式必须为："
        "{\"entities\":[{\"entity\":\"实体文本\",\"entity_type\":\"实体类型\"}]}。\n"
        "不确定时不要输出，不要解释。"
    )

    assistant_obj = {
        "entities": entities
    }

    assistant_output = json.dumps(
        assistant_obj,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ]
    }


def iter_samples_from_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data if isinstance(data, list) else data.get("data", [])
    if not docs and isinstance(data, dict):
        docs = [data]

    samples = []
    seen_sample_keys = set()

    for doc in docs:
        paragraphs = doc.get("paragraphs", [])
        for para in paragraphs:
            sentences = para.get("sentences", [])
            for sent in sentences:
                sentence_text = normalize_text(safe_get_sentence_text(sent))
                if not sentence_text:
                    continue

                entities = normalize_entities(sent.get("entities", []), sentence_text)

                if not should_keep_sample(sentence_text, entities):
                    continue

                # 同一句 + 同一组实体 只保留一次
                sample_key = (
                    sentence_text,
                    json.dumps(entities, ensure_ascii=False, sort_keys=True)
                )
                if sample_key in seen_sample_keys:
                    continue
                seen_sample_keys.add(sample_key)

                sample = build_messages(sentence_text, entities)
                sample["_meta"] = {
                    "source_file": os.path.basename(file_path),
                    "num_entities": len(entities),
                    "entity_types": sorted(list({e["entity_type"] for e in entities})),
                    "is_empty": len(entities) == 0,
                    "sentence_len": len(sentence_text)
                }
                samples.append(sample)

    return samples


def load_all_samples(data_dir: str) -> List[Dict[str, Any]]:
    all_samples = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    files.sort()

    print(f"检测到 {len(files)} 个 JSON 文件，开始转换...")

    for fn in files:
        fp = os.path.join(data_dir, fn)
        try:
            samples = iter_samples_from_file(fp)
            all_samples.extend(samples)
            print(f"[OK] {fn}: {len(samples)} 条样本")
        except Exception as e:
            print(f"[WARN] {fn} 处理失败: {e}")

    return all_samples


def oversample_tail_samples(samples: List[Dict[str, Any]], repeat: int = 2) -> List[Dict[str, Any]]:
    boosted = []
    for sample in samples:
        boosted.append(sample)

        meta = sample.get("_meta", {})
        entity_types = set(meta.get("entity_types", []))
        if entity_types & TAIL_ENTITY_TYPES:
            for _ in range(repeat - 1):
                boosted.append(sample)

    return boosted


def split_dataset(samples: List[Dict[str, Any]]) -> Tuple[List, List, List]:
    random.shuffle(samples)
    n = len(samples)

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]

    return train_data, val_data, test_data


def strip_meta(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for x in samples:
        y = dict(x)
        y.pop("_meta", None)
        cleaned.append(y)
    return cleaned


def save_jsonl(samples: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_stats(name: str, samples: List[Dict[str, Any]]) -> None:
    counter = Counter()
    empty_count = 0
    sentence_lens = []

    for s in samples:
        assistant = s["messages"][-1]["content"]
        obj = json.loads(assistant)
        ents = obj.get("entities", [])

        if not ents:
            empty_count += 1

        for e in ents:
            counter[e["entity_type"]] += 1

        user_text = s["messages"][1]["content"]
        sentence_lens.append(len(user_text))

    print(f"\n{name} 统计:")
    print(f"  样本数: {len(samples)}")
    print(f"  空实体样本: {empty_count}")
    if sentence_lens:
        print(f"  平均样本长度: {sum(sentence_lens) / len(sentence_lens):.2f}")
    print(f"  实体分布 Top 20: {counter.most_common(20)}")


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_samples = load_all_samples(DATA_DIR)
    print(f"\n原始样本总数: {len(all_samples)}")

    train_data, val_data, test_data = split_dataset(all_samples)

    print("\n原始划分完成:")
    print(f"  train={len(train_data)}")
    print(f"  val={len(val_data)}")
    print(f"  test={len(test_data)}")

    train_data = oversample_tail_samples(train_data, repeat=TAIL_REPEAT)

    print("\n训练集长尾过采样后:")
    print(f"  train={len(train_data)}")

    print_stats("TRAIN", train_data)
    print_stats("VAL", val_data)
    print_stats("TEST", test_data)

    train_out = os.path.join(OUT_DIR, "train.jsonl")
    val_out = os.path.join(OUT_DIR, "val.jsonl")
    test_out = os.path.join(OUT_DIR, "test.jsonl")

    save_jsonl(strip_meta(train_data), train_out)
    save_jsonl(strip_meta(val_data), val_out)
    save_jsonl(strip_meta(test_data), test_out)

    print("\n保存完成:")
    print(f"  {train_out}")
    print(f"  {val_out}")
    print(f"  {test_out}")


if __name__ == "__main__":
    main()
