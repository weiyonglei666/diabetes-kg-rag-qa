import os
import json
import re
import time
from collections import Counter
from typing import List, Dict, Optional

import jieba
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase


# =========================
# 配置
# =========================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password")

EVAL_FILE = "trainlast_cleaned.jsonl"
OUTPUT_FILE = "kg_qa_eval_results_purekg.csv"

TOPK = 10


# =========================
# 工具函数
# =========================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_zh(text: str) -> List[str]:
    return [w.strip() for w in jieba.lcut(normalize_text(text)) if w.strip()]


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize_zh(pred)
    gold_tokens = tokenize_zh(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    res = []
    for x in items:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


# =========================
# 问题分类
# =========================
class QuestionRouter:
    def classify(self, question: str) -> str:
        q = normalize_text(question)

        if any(x in q for x in ["症状", "表现"]):
            return "symptom"
        if any(x in q for x in ["治疗"]):
            return "treatment"
        if any(x in q for x in ["原因", "病因"]):
            return "reason"
        if any(x in q for x in ["检查", "诊断"]):
            return "test"
        if any(x in q for x in ["并发症", "相关疾病"]):
            return "related_disease"
        if any(x in q for x in ["科室"]):
            return "department"

        return "other"


# =========================
# 纯 KG 检索器
# =========================
class PureKGRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.router = QuestionRouter()

    def close(self):
        self.driver.close()

    def retrieve(self, question: str, topk: int = 10):
        intent = self.router.classify(question)
        entity = self.extract_entity(question)

        if not entity:
            return []

        with self.driver.session() as session:
            return session.execute_read(self._query, entity, intent, topk)

    def extract_entity(self, question: str) -> Optional[str]:
        q = normalize_text(question)

        patterns = [
            r"^(.*?)的症状",
            r"^(.*?)的治疗",
            r"^(.*?)的原因",
            r"^(.*?)如何诊断",
            r"^(.*?)需要做哪些检查",
            r"^(.*?)有哪些并发症",
        ]

        for p in patterns:
            m = re.match(p, q)
            if m:
                return normalize_text(m.group(1))

        words = tokenize_zh(q)
        words = sorted(words, key=lambda x: len(x), reverse=True)
        return words[0] if words else None

    @staticmethod
    def _query(tx, entity: str, intent: str, topk: int):
        query_map = {
            "symptom": """
                MATCH (d:Entity)-[:Symptom_Disease]->(s:Entity)
                WHERE d.name CONTAINS $entity
                RETURN s.name AS value
                LIMIT $topk
            """,
            "treatment": """
                MATCH (d:Entity)-[:Treatment_Disease]->(t:Entity)
                WHERE d.name CONTAINS $entity
                RETURN t.name AS value
                LIMIT $topk
            """,
            "reason": """
                MATCH (x:Entity)-[:Reason_Disease]->(d:Entity)
                WHERE d.name CONTAINS $entity
                RETURN x.name AS value
                LIMIT $topk
            """,
            "test": """
                MATCH (x:Entity)-[:Test_Disease]->(d:Entity)
                WHERE d.name CONTAINS $entity
                RETURN x.name AS value
                LIMIT $topk
            """,
            "related_disease": """
                MATCH (d:Entity)-[:Related_Disease]->(x:Entity)
                WHERE d.name CONTAINS $entity
                RETURN x.name AS value
                LIMIT $topk
            """,
            "department": """
                MATCH (d:Entity)-[:Department_Disease]->(x:Entity)
                WHERE d.name CONTAINS $entity
                RETURN x.name AS value
                LIMIT $topk
            """
        }

        cypher = query_map.get(intent)
        if not cypher:
            return []

        result = tx.run(cypher, entity=entity, topk=topk)
        return [r.data() for r in result]


# =========================
# 生成器
# =========================
class PureKGGenerator:
    def __init__(self):
        self.router = QuestionRouter()

    def generate(self, question: str, retrieved):
        if not retrieved:
            return "未在知识图谱中找到相关结构化信息。"

        intent = self.router.classify(question)

        values = [x["value"] for x in retrieved if x.get("value")]
        values = uniq_keep_order(values)

        if not values:
            return "未在知识图谱中找到相关结构化信息。"

        if intent == "symptom":
            return f"症状包括：{'、'.join(values)}。"
        if intent == "treatment":
            return f"治疗方法包括：{'、'.join(values)}。"
        if intent == "reason":
            return f"可能病因包括：{'、'.join(values)}。"
        if intent == "test":
            return f"相关检查包括：{'、'.join(values)}。"
        if intent == "related_disease":
            return f"相关疾病包括：{'、'.join(values)}。"
        if intent == "department":
            return f"就诊科室包括：{'、'.join(values)}。"

        return "该问题不属于知识图谱结构化查询范围。"


# =========================
# 数据读取
# =========================
def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# =========================
# 评测
# =========================
def evaluate():
    data = load_jsonl(EVAL_FILE)

    retriever = PureKGRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    generator = PureKGGenerator()
    router = QuestionRouter()

    rows = []

    try:
        for item in tqdm(data):
            q = normalize_text(item["question"])
            gold = normalize_text(item["context"])

            t1 = time.perf_counter()
            retrieved = retriever.retrieve(q, TOPK)
            t2 = time.perf_counter()

            pred = generator.generate(q, retrieved)
            t3 = time.perf_counter()

            rows.append({
                "question": q,
                "intent": router.classify(q),
                "gold": gold,
                "pred": pred,
                "f1": compute_f1(pred, gold),
                "time": t3 - t1,
                "retrieved": len(retrieved)
            })

    finally:
        retriever.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("平均 F1:", df["f1"].mean())
    print(df.groupby("intent")["f1"].mean())


if __name__ == "__main__":
    evaluate()
