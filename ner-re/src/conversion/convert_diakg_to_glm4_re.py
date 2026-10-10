#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 DIaKG 多个 JSON 文件转换为 GLM-4-9B-Chat 关系抽取训练 jsonl

任务形式：
给定句子 + 两个已知实体 -> 预测关系

输出格式：
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"relation\":\"Drug_Disease\"}"}
  ],
  "meta": {...}
}
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any


# =========================================================
# 配置区：直接改这里
# =========================================================

INPUT_DIR = Path("../diakg_data")
OUTPUT_DIR = Path("../glm4_re_data")

SEED = 42

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

NEGATIVE_RATIO = 1.0

OUTPUT_FORMAT = "messages"   # "messages" / "instruction"
USE_DYNAMIC_CANDIDATES = True
INCLUDE_OFFSETS = True
KEEP_ONLY_SENTENCES_WITH_POSITIVE = True
ALLOW_SAME_TEXT_DIFFERENT_SPAN = True

SYSTEM_PROMPT = (
    "你是一个医学关系抽取模型。"
    "请根据给定句子和两个实体，判断它们之间的关系。"
    "只输出JSON，不要输出解释，不要输出额外文本。"
)


# =========================================================
# 标签与约束
# =========================================================

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


def set_seed(seed: int) -> None:
    random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_doc_id(doc: Dict[str, Any], file_path: Path) -> str:
    return str(doc.get("doc_id", file_path.stem))


def build_relation_map(relations: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    rel_map = {}
    for rel in relations:
        head_id = rel["head_entity_id"]
        tail_id = rel["tail_entity_id"]
        rel_type = rel["relation_type"]
        if (head_id, tail_id) not in rel_map:
            rel_map[(head_id, tail_id)] = rel_type
    return rel_map


def normalize_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entity_id": entity["entity_id"],
        "text": entity["entity"],
        "type": entity["entity_type"],
        "start": entity.get("start_idx", -1),
        "end": entity.get("end_idx", -1),
    }


def legal_relations_for_entity_types(head_type: str, tail_type: str) -> List[str]:
    rels = []
    for rel, (h_type, t_type) in RELATION_CONSTRAINTS.items():
        if head_type == h_type and tail_type == t_type:
            rels.append(rel)
    return rels


def make_user_prompt(
    sentence: str,
    head: Dict[str, Any],
    tail: Dict[str, Any],
    candidate_relations: List[str],
    include_offsets: bool = True,
) -> str:
    if include_offsets:
        entity1 = json.dumps(
            {
                "text": head["text"],
                "type": head["type"],
                "start": head["start"],
                "end": head["end"],
            },
            ensure_ascii=False,
        )
        entity2 = json.dumps(
            {
                "text": tail["text"],
                "type": tail["type"],
                "start": tail["start"],
                "end": tail["end"],
            },
            ensure_ascii=False,
        )
    else:
        entity1 = json.dumps({"text": head["text"], "type": head["type"]}, ensure_ascii=False)
        entity2 = json.dumps({"text": tail["text"], "type": tail["type"]}, ensure_ascii=False)

    return (
        "任务：判断给定两个实体之间的关系。\n"
        "只输出 JSON。\n\n"
        f"句子：{sentence}\n"
        f"实体1：{entity1}\n"
        f"实体2：{entity2}\n"
        f"可选关系：{json.dumps(candidate_relations, ensure_ascii=False)}"
    )


def make_sample(
    sentence: str,
    head: Dict[str, Any],
    tail: Dict[str, Any],
    relation: str,
    candidate_relations: List[str],
    doc_id: str,
    paragraph_id: str,
    sentence_id: str,
    source_file: str,
    output_format: str,
    include_offsets: bool,
) -> Dict[str, Any]:
    user_prompt = make_user_prompt(
        sentence=sentence,
        head=head,
        tail=tail,
        candidate_relations=candidate_relations,
        include_offsets=include_offsets,
    )
    assistant_content = json.dumps({"relation": relation}, ensure_ascii=False)

    meta = {
        "doc_id": doc_id,
        "paragraph_id": paragraph_id,
        "sentence_id": sentence_id,
        "source_file": source_file,
        "head_entity_id": head["entity_id"],
        "tail_entity_id": tail["entity_id"],
        "head_text": head["text"],
        "tail_text": tail["text"],
        "head_type": head["type"],
        "tail_type": tail["type"],
        "label": relation,
    }

    if output_format == "messages":
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_content},
            ],
            "meta": meta,
        }

    if output_format == "instruction":
        return {
            "instruction": SYSTEM_PROMPT,
            "input": user_prompt,
            "output": assistant_content,
            "meta": meta,
        }

    raise ValueError(f"未知 OUTPUT_FORMAT: {output_format}")


def generate_samples_from_sentence(
    sentence_obj: Dict[str, Any],
    doc_id: str,
    paragraph_id: str,
    source_file: str,
) -> List[Dict[str, Any]]:
    sentence = sentence_obj.get("sentence", "").strip()
    sentence_id = str(sentence_obj.get("sentence_id", ""))

    entities = sentence_obj.get("entities", [])
    relations = sentence_obj.get("relations", [])

    if not sentence or not entities:
        return []

    rel_map = build_relation_map(relations)
    normalized_entities = [normalize_entity(e) for e in entities]

    positive_samples = []
    negative_candidates = []

    for head in normalized_entities:
        for tail in normalized_entities:
            if head["entity_id"] == tail["entity_id"]:
                continue

            if not ALLOW_SAME_TEXT_DIFFERENT_SPAN:
                if (
                    head["text"] == tail["text"]
                    and head["start"] == tail["start"]
                    and head["end"] == tail["end"]
                ):
                    continue

            legal_rels = legal_relations_for_entity_types(head["type"], tail["type"])
            if not legal_rels:
                continue

            candidate_relations = (
                ["No_Relation"] + legal_rels
                if USE_DYNAMIC_CANDIDATES
                else RELATION_TYPES
            )

            pair_key = (head["entity_id"], tail["entity_id"])
            if pair_key in rel_map:
                rel_label = rel_map[pair_key]
                if rel_label not in candidate_relations:
                    candidate_relations = ["No_Relation"] + sorted(
                        list(set(candidate_relations + [rel_label]))
                    )

                positive_samples.append(
                    make_sample(
                        sentence=sentence,
                        head=head,
                        tail=tail,
                        relation=rel_label,
                        candidate_relations=candidate_relations,
                        doc_id=doc_id,
                        paragraph_id=paragraph_id,
                        sentence_id=sentence_id,
                        source_file=source_file,
                        output_format=OUTPUT_FORMAT,
                        include_offsets=INCLUDE_OFFSETS,
                    )
                )
            else:
                negative_candidates.append(
                    make_sample(
                        sentence=sentence,
                        head=head,
                        tail=tail,
                        relation="No_Relation",
                        candidate_relations=candidate_relations,
                        doc_id=doc_id,
                        paragraph_id=paragraph_id,
                        sentence_id=sentence_id,
                        source_file=source_file,
                        output_format=OUTPUT_FORMAT,
                        include_offsets=INCLUDE_OFFSETS,
                    )
                )

    if positive_samples:
        target_neg = int(len(positive_samples) * NEGATIVE_RATIO)
        negative_samples = random.sample(
            negative_candidates,
            min(target_neg, len(negative_candidates)),
        )
    else:
        negative_samples = [] if KEEP_ONLY_SENTENCES_WITH_POSITIVE else negative_candidates

    return positive_samples + negative_samples


def generate_samples_from_doc(doc: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
    doc_id = get_doc_id(doc, Path(source_file))
    all_samples = []

    for para in doc.get("paragraphs", []):
        paragraph_id = str(para.get("paragraph_id", ""))
        for sent in para.get("sentences", []):
            samples = generate_samples_from_sentence(
                sentence_obj=sent,
                doc_id=doc_id,
                paragraph_id=paragraph_id,
                source_file=source_file,
            )
            all_samples.extend(samples)

    return all_samples


def split_docs(file_paths: List[Path]):
    assert abs(TRAIN_RATIO + DEV_RATIO + TEST_RATIO - 1.0) < 1e-8

    shuffled = file_paths[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_dev = int(n * DEV_RATIO)

    train_files = shuffled[:n_train]
    dev_files = shuffled[n_train:n_train + n_dev]
    test_files = shuffled[n_train + n_dev:]

    return train_files, dev_files, test_files


def collect_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counter = Counter()
    pair_type_counter = Counter()

    for rec in records:
        meta = rec["meta"]
        label_counter[meta["label"]] += 1
        pair_type_counter[f'{meta["head_type"]}->{meta["tail_type"]}'] += 1

    return {
        "num_samples": len(records),
        "label_distribution": dict(label_counter),
        "pair_type_distribution": dict(pair_type_counter),
        "output_format": OUTPUT_FORMAT,
    }


def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    file_paths = sorted(INPUT_DIR.glob("*.json"))
    if not file_paths:
        raise FileNotFoundError(f"在目录 {INPUT_DIR} 下未找到任何 .json 文件")

    train_files, dev_files, test_files = split_docs(file_paths)

    def build_split_records(split_files: List[Path]) -> List[Dict[str, Any]]:
        records = []
        for fp in split_files:
            doc = load_json(fp)
            samples = generate_samples_from_doc(doc, str(fp))
            records.extend(samples)
        random.shuffle(records)
        return records

    train_records = build_split_records(train_files)
    dev_records = build_split_records(dev_files)
    test_records = build_split_records(test_files)

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_records)
    write_jsonl(OUTPUT_DIR / "dev.jsonl", dev_records)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test_records)

    stats = {
        "num_docs": len(file_paths),
        "train_docs": [p.name for p in train_files],
        "dev_docs": [p.name for p in dev_files],
        "test_docs": [p.name for p in test_files],
        "train_stats": collect_stats(train_records),
        "dev_stats": collect_stats(dev_records),
        "test_stats": collect_stats(test_records),
        "config": {
            "INPUT_DIR": str(INPUT_DIR),
            "OUTPUT_DIR": str(OUTPUT_DIR),
            "SEED": SEED,
            "TRAIN_RATIO": TRAIN_RATIO,
            "DEV_RATIO": DEV_RATIO,
            "TEST_RATIO": TEST_RATIO,
            "NEGATIVE_RATIO": NEGATIVE_RATIO,
            "OUTPUT_FORMAT": OUTPUT_FORMAT,
            "USE_DYNAMIC_CANDIDATES": USE_DYNAMIC_CANDIDATES,
            "INCLUDE_OFFSETS": INCLUDE_OFFSETS,
            "KEEP_ONLY_SENTENCES_WITH_POSITIVE": KEEP_ONLY_SENTENCES_WITH_POSITIVE,
            "ALLOW_SAME_TEXT_DIFFERENT_SPAN": ALLOW_SAME_TEXT_DIFFERENT_SPAN,
        },
    }

    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("转换完成")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"文档总数: {len(file_paths)}")
    print(f"train 文档数: {len(train_files)} | 样本数: {len(train_records)}")
    print(f"dev   文档数: {len(dev_files)} | 样本数: {len(dev_records)}")
    print(f"test  文档数: {len(test_files)} | 样本数: {len(test_records)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
