import os

import json
import re
import time
from collections import Counter
from typing import List, Dict, Optional, Tuple

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

EVAL_FILE = "./trainlast_cleaned.jsonl"
OUTPUT_FILE = "./kg_qa_eval_results_enhanced.csv"

TOPK = 10


# =========================
# 工具函数
# =========================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_zh(text: str) -> List[str]:
    text = normalize_text(text)
    return [w.strip() for w in jieba.lcut(text) if w.strip()]


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize_zh(pred)
    gold_tokens = tokenize_zh(gold)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# 问题路由器
# =========================
class QuestionRouter:
    def classify(self, question: str) -> str:
        q = normalize_text(question)

        if any(x in q for x in ["英文", "英文名", "英文名称", "英文叫什么"]):
            return "english_name"

        if any(x in q for x in ["定义", "什么是", "是一种什么疾病", "是什么病", "属于什么"]):
            return "definition"

        if any(x in q for x in ["症状", "表现", "会出现哪些", "会有什么感觉", "主要症状", "具体症状"]):
            return "symptom"

        if any(x in q for x in ["治疗", "怎么治", "如何治疗", "治疗方法", "主要治疗方法"]):
            return "treatment"

        if any(x in q for x in ["原因", "病因", "为什么会", "发生原因"]):
            return "reason"

        if any(x in q for x in ["发病机制", "机制"]):
            return "pathogenesis"

        if any(x in q for x in ["检查", "诊断", "如何诊断", "需要做哪些检查", "最重要的检查"]):
            return "test"

        if any(x in q for x in ["科室", "挂什么科", "就诊", "去哪个科"]):
            return "department"

        if any(x in q for x in ["并发症", "相关疾病", "有关疾病", "关联疾病"]):
            return "related_disease"

        if any(x in q for x in ["预防", "注意事项", "需要注意", "如何预防"]):
            return "precaution"

        if any(x in q for x in ["病程", "发展", "会如何变化", "如何进展"]):
            return "course"

        return "general"


# =========================
# Neo4j 检索器
# =========================
class KGQARetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.router = QuestionRouter()

    def close(self):
        self.driver.close()

    def retrieve(self, question: str, topk: int = 10) -> List[Dict]:
        """
        先做：
        1. 问题分类
        2. 实体识别
        3. 按意图路由专用查询
        4. 失败后回退到泛化查询
        """
        intent = self.router.classify(question)
        entity = self.extract_entity(question)

        with self.driver.session() as session:
            # 一级：实体 + 专用路由
            if entity:
                results = session.execute_read(self._route_query, entity, intent, topk)
                if results:
                    return results

                # 二级：实体 + 一跳邻居
                results = session.execute_read(self._query_neighbors_by_entity, entity, topk)
                if results:
                    return results

            # 三级：关键词回退
            keywords = self.extract_keywords(question)
            if keywords:
                results = session.execute_read(self._query_neighbors_by_keywords, keywords, topk)
                return results

        return []

    def extract_entity(self, question: str) -> Optional[str]:
        """
        规则式实体提取：
        优先去掉常见问句模板后，取前面的主实体
        """
        q = normalize_text(question)

        patterns = [
            r"^(.*?)的?(主要)?症状是什么",
            r"^(.*?)会出现哪些症状",
            r"^(.*?)的治疗方法有哪些",
            r"^(.*?)如何治疗",
            r"^(.*?)的原因是什么",
            r"^(.*?)的病因是什么",
            r"^(.*?)的英文名称是什么",
            r"^(.*?)的英文名是什么",
            r"^(.*?)如何诊断",
            r"^(.*?)需要做哪些检查",
            r"^(.*?)最重要的检查是什么",
            r"^(.*?)的定义是什么",
            r"^什么是(.*?)$",
            r"^(.*?)是什么",
            r"^(.*?)有哪些并发症",
            r"^(.*?)相关的疾病有哪些",
            r"^(.*?)应多频繁接受眼科检查",
        ]

        for p in patterns:
            m = re.match(p, q)
            if m:
                ent = normalize_text(m.group(1))
                if ent:
                    return ent

        # 回退：取较长中文连续片段
        words = tokenize_zh(q)
        stopwords = {
            "什么", "哪些", "多少", "怎么", "如何", "为什么",
            "是否", "属于", "有关", "相关", "主要", "具体",
            "一种", "这个", "那个", "会", "是", "的", "了", "吗",
            "英文", "名称", "定义", "症状", "治疗", "原因", "病因",
            "检查", "诊断", "科室", "并发症", "注意事项", "预防", "病程"
        }
        cands = [w for w in words if len(w) >= 2 and w not in stopwords]
        cands = uniq_keep_order(cands)
        if cands:
            # 优先最长
            cands = sorted(cands, key=lambda x: len(x), reverse=True)
            return cands[0]
        return None

    @staticmethod
    def extract_keywords(question: str) -> List[str]:
        stopwords = {
            "什么", "哪些", "多少", "怎么", "如何", "为什么",
            "是否", "属于", "有关", "相关", "主要", "具体",
            "一种", "这个", "那个", "会", "是", "的", "了", "吗",
            "请问", "一下", "英文", "名称", "定义"
        }

        words = tokenize_zh(question)
        keywords = []
        for w in words:
            w = w.strip()
            if not w or w in stopwords:
                continue
            if len(w) >= 2:
                keywords.append(w)

        return uniq_keep_order(keywords)[:8]

    @staticmethod
    def _route_query(tx, entity: str, intent: str, topk: int):
        """
        不同问题类型走不同查询模板
        """
        route_map = {
            "symptom": "症状",
            "treatment": "治疗方法",
            "reason": "病因",
            "pathogenesis": "发病机制",
            "test": "检查",
            "department": "就诊科室",
            "english_name": "疾病英文名",
            "definition": "定义",
            "related_disease": "相关疾病",
            "precaution": "注意事项",
            "course": "病程",
        }

        prefix = route_map.get(intent)

        # 这类数据的 evidence 常含“某实体的定义：...” 这种前缀
        if prefix:
            cypher = """
            MATCH (n:Entity)
            WHERE n.name = $entity OR n.name CONTAINS $entity
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE r.evidence CONTAINS ($entity + '的' + $prefix)
               OR r.evidence CONTAINS ($entity + '的' + replace($prefix, '就诊科室', '诊疗科室'))
               OR r.evidence CONTAINS ($entity + '的' + replace($prefix, '病因', '原因'))
            RETURN n.name AS center_name,
                   n.etype AS center_etype,
                   n.etype_zh AS center_etype_zh,
                   type(r) AS rel_type,
                   r.rel_type_zh AS rel_type_zh,
                   m.name AS neighbor_name,
                   m.etype AS neighbor_etype,
                   m.etype_zh AS neighbor_etype_zh,
                   r.evidence AS evidence
            LIMIT $topk
            """
            result = tx.run(cypher, entity=entity, prefix=prefix, topk=topk)
            data = [record.data() for record in result]
            if data:
                return data

        # 如果是症状类，优先查症状关系
        if intent == "symptom":
            cypher = """
            MATCH (d:Entity)-[r:Symptom_Disease]-(s:Entity)
            WHERE d.name = $entity OR d.name CONTAINS $entity
            RETURN d.name AS center_name,
                   d.etype AS center_etype,
                   d.etype_zh AS center_etype_zh,
                   type(r) AS rel_type,
                   r.rel_type_zh AS rel_type_zh,
                   s.name AS neighbor_name,
                   s.etype AS neighbor_etype,
                   s.etype_zh AS neighbor_etype_zh,
                   r.evidence AS evidence
            LIMIT $topk
            """
            return [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]

        # 治疗类
        if intent == "treatment":
            cypher = """
            MATCH (d:Entity)-[r:Treatment_Disease]-(t:Entity)
            WHERE d.name = $entity OR d.name CONTAINS $entity
            RETURN d.name AS center_name,
                   d.etype AS center_etype,
                   d.etype_zh AS center_etype_zh,
                   type(r) AS rel_type,
                   r.rel_type_zh AS rel_type_zh,
                   t.name AS neighbor_name,
                   t.etype AS neighbor_etype,
                   t.etype_zh AS neighbor_etype_zh,
                   r.evidence AS evidence
            LIMIT $topk
            """
            return [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]

        # 病因类
        if intent == "reason":
            cypher = """
            MATCH (x:Entity)-[r:Reason_Disease]-(d:Entity)
            WHERE d.name = $entity OR d.name CONTAINS $entity
            RETURN d.name AS center_name,
                   d.etype AS center_etype,
                   d.etype_zh AS center_etype_zh,
                   type(r) AS rel_type,
                   r.rel_type_zh AS rel_type_zh,
                   x.name AS neighbor_name,
                   x.etype AS neighbor_etype,
                   x.etype_zh AS neighbor_etype_zh,
                   r.evidence AS evidence
            LIMIT $topk
            """
            return [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]

        # 发病机制类
        if intent == "pathogenesis":
            cypher = """
            MATCH (x:Entity)-[r:Pathogenesis_Disease]-(d:Entity)
            WHERE d.name = $entity OR d.name CONTAINS $entity
            RETURN d.name AS center_name,
                   d.etype AS center_etype,
                   d.etype_zh AS center_etype_zh,
                   type(r) AS rel_type,
                   r.rel_type_zh AS rel_type_zh,
                   x.name AS neighbor_name,
                   x.etype AS neighbor_etype,
                   x.etype_zh AS neighbor_etype_zh,
                   r.evidence AS evidence
            LIMIT $topk
            """
            return [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]

        # 检查/诊断类
        if intent == "test":
            cypher = """
            MATCH (x:Entity)-[r:Test_Disease]-(d:Entity)
            WHERE d.name = $entity OR d.name CONTAINS $entity
            RETURN d.name AS center_name,
                   d.etype AS center_etype,
                   d.etype_zh AS center_etype_zh,
                   type(r) AS rel_type,
                   r.rel_type_zh AS rel_type_zh,
                   x.name AS neighbor_name,
                   x.etype AS neighbor_etype,
                   x.etype_zh AS neighbor_etype_zh,
                   r.evidence AS evidence
            LIMIT $topk
            """
            return [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]

        return []

    @staticmethod
    def _query_neighbors_by_entity(tx, entity: str, topk: int):
        cypher = """
        MATCH (n:Entity)
        WHERE n.name = $entity OR n.name CONTAINS $entity
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n.name AS center_name,
               n.etype AS center_etype,
               n.etype_zh AS center_etype_zh,
               type(r) AS rel_type,
               r.rel_type_zh AS rel_type_zh,
               m.name AS neighbor_name,
               m.etype AS neighbor_etype,
               m.etype_zh AS neighbor_etype_zh,
               r.evidence AS evidence
        LIMIT $topk
        """
        result = tx.run(cypher, entity=entity, topk=topk)
        return [record.data() for record in result]

    @staticmethod
    def _query_neighbors_by_keywords(tx, keywords: List[str], topk: int):
        if not keywords:
            return []

        cypher = """
        MATCH (n:Entity)
        WITH n, [k IN $keywords WHERE n.name CONTAINS k] AS matched_keywords
        WHERE size(matched_keywords) > 0
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n.name AS center_name,
               n.etype AS center_etype,
               n.etype_zh AS center_etype_zh,
               type(r) AS rel_type,
               r.rel_type_zh AS rel_type_zh,
               m.name AS neighbor_name,
               m.etype AS neighbor_etype,
               m.etype_zh AS neighbor_etype_zh,
               r.evidence AS evidence,
               size(matched_keywords) AS match_score
        ORDER BY match_score DESC
        LIMIT $topk
        """
        result = tx.run(cypher, keywords=keywords, topk=topk)
        return [record.data() for record in result]


# =========================
# 生成器
# =========================
class AdvancedGenerator:
    def __init__(self):
        self.router = QuestionRouter()

    def generate(self, question: str, retrieved: List[Dict]) -> str:
        if not retrieved:
            return "未从知识图谱中检索到相关信息。"

        intent = self.router.classify(question)

        # 优先提取证据
        evidences = []
        facts = []
        seen = set()

        for item in retrieved:
            center = normalize_text(item.get("center_name", ""))
            center_type = normalize_text(item.get("center_etype_zh", "")) or normalize_text(item.get("center_etype", ""))
            rel = normalize_text(item.get("rel_type_zh", "")) or normalize_text(item.get("rel_type", ""))
            neighbor = normalize_text(item.get("neighbor_name", ""))
            neighbor_type = normalize_text(item.get("neighbor_etype_zh", "")) or normalize_text(item.get("neighbor_etype", ""))
            evidence = normalize_text(item.get("evidence", ""))

            key = (center, rel, neighbor, evidence)
            if key in seen:
                continue
            seen.add(key)

            if evidence:
                evidences.append(evidence)

            if center and rel and neighbor:
                facts.append({
                    "center": center,
                    "center_type": center_type,
                    "rel": rel,
                    "neighbor": neighbor,
                    "neighbor_type": neighbor_type,
                    "evidence": evidence,
                })

        # 不同意图，采用不同生成策略
        if intent in {"definition", "reason", "course", "precaution", "english_name", "department"}:
            # 优先直接返回最相关证据
            if evidences:
                return evidences[0]

        if intent in {"symptom", "treatment", "pathogenesis", "test", "related_disease"}:
            # 优先汇总邻居实体
            values = []
            for f in facts:
                if f["neighbor"]:
                    values.append(f["neighbor"])
            values = uniq_keep_order(values)

            if values:
                if intent == "symptom":
                    return f"相关症状包括：{'、'.join(values[:10])}。"
                elif intent == "treatment":
                    return f"相关治疗方法包括：{'、'.join(values[:10])}。"
                elif intent == "pathogenesis":
                    return f"相关发病机制包括：{'、'.join(values[:10])}。"
                elif intent == "test":
                    return f"相关检查包括：{'、'.join(values[:10])}。"
                elif intent == "related_disease":
                    return f"相关疾病包括：{'、'.join(values[:10])}。"

            if evidences:
                return evidences[0]

        # 默认：事实 + 证据混合
        lines = []
        for f in facts[:5]:
            line = f"{f['center_type']}“{f['center']}”与{f['neighbor_type']}“{f['neighbor']}”存在“{f['rel']}”关系。"
            if f["evidence"]:
                line += f" 证据：{f['evidence']}"
            lines.append(line)

        if lines:
            return " ".join(lines)

        if evidences:
            return evidences[0]

        return "已检索到实体，但未找到足够的关系信息。"


# =========================
# 数据读取
# =========================
def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# =========================
# 评测主流程
# =========================
def evaluate():
    samples = load_jsonl(EVAL_FILE)

    retriever = KGQARetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    generator = AdvancedGenerator()
    router = QuestionRouter()

    rows = []

    try:
        for sample in tqdm(samples, desc="Evaluating"):
            question = normalize_text(sample.get("question", ""))
            gold = normalize_text(sample.get("context", ""))

            # 检索计时
            t1 = time.perf_counter()
            retrieved = retriever.retrieve(question, topk=TOPK)
            t2 = time.perf_counter()

            # 生成计时
            pred = generator.generate(question, retrieved)
            t3 = time.perf_counter()

            retrieval_time = t2 - t1
            generation_time = t3 - t2
            total_time = t3 - t1

            f1 = compute_f1(pred, gold)

            rows.append({
                "question": question,
                "intent": router.classify(question),
                "gold_answer": gold,
                "pred_answer": pred,
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": generation_time,
                "total_time_sec": total_time,
                "f1": f1,
                "retrieved_count": len(retrieved)
            })

    finally:
        retriever.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n========= 评测结果 =========")
    print(f"样本数: {len(df)}")
    print(f"平均 F1: {df['f1'].mean():.4f}")
    print(f"平均检索耗时(秒): {df['retrieval_time_sec'].mean():.4f}")
    print(f"平均生成耗时(秒): {df['generation_time_sec'].mean():.4f}")
    print(f"平均总耗时(秒): {df['total_time_sec'].mean():.4f}")
    print(f"结果文件: {OUTPUT_FILE}")

    print("\n========= 各问题类型平均 F1 =========")
    print(df.groupby("intent")["f1"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    evaluate()
