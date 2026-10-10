# -*- coding: utf-8 -*-
import os
import re
import time
from typing import List, Dict, Optional, Any

import jieba
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neo4j import GraphDatabase


# =========================
# 配置
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password")
TOPK = int(os.getenv("TOPK", "10"))


# =========================
# 工具函数
# =========================
def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_zh(text: str) -> List[str]:
    text = normalize_text(text)
    return [w.strip() for w in jieba.lcut(text) if w.strip()]


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# 返回数据模型
# =========================
class EvidenceItem(BaseModel):
    center_name: str = ""
    center_etype: str = ""
    center_etype_zh: str = ""
    rel_type: str = ""
    rel_type_zh: str = ""
    neighbor_name: str = ""
    neighbor_etype: str = ""
    neighbor_etype_zh: str = ""
    evidence: str = ""


class QARequest(BaseModel):
    question: str = Field(..., description="用户输入的问题")


class QAResponse(BaseModel):
    question: str
    entity: Optional[str]
    intent: str
    answer: str
    cypher: Optional[str]
    retrieved_count: int
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    evidence_chain: List[EvidenceItem]


# =========================
# 问题路由
# =========================
class QuestionRouter:
    def classify(self, question: str) -> str:
        q = normalize_text(question)

        # 证据优先类
        if any(x in q for x in ["英文", "英文名", "英文名称", "英文叫什么"]):
            return "english_name"

        if any(x in q for x in ["同义词", "别名", "另一名称", "别称"]):
            return "alias"

        if any(x in q for x in ["定义", "什么是", "是一种什么疾病", "是什么病", "属于什么"]):
            return "definition"

        if any(x in q for x in ["科室", "挂什么科", "就诊", "去哪个科", "诊疗科室", "就诊科室"]):
            return "department"

        if any(x in q for x in ["预防", "注意事项", "需要注意", "如何预防"]):
            return "precaution"

        if any(x in q for x in ["病程", "发展", "会如何变化", "如何进展"]):
            return "course"

        # 结构化优先类
        if any(x in q for x in ["不良反应", "副作用"]):
            return "ade"

        if any(x in q for x in ["药物", "用药", "什么药"]):
            return "drug"

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

        if any(x in q for x in ["检验指标", "指标", "化验", "HbA1c", "血糖", "空腹血糖", "PPG"]):
            return "test_items"

        if any(x in q for x in ["并发症", "相关疾病", "有关疾病", "关联疾病"]):
            return "related_disease"

        if any(x in q for x in ["分类", "分型", "属于哪一类"]):
            return "class"

        if any(x in q for x in ["解剖部位", "部位", "发生在哪", "涉及哪些部位"]):
            return "anatomy"

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

    def ping(self) -> bool:
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS ok")
                record = result.single()
                return bool(record and record["ok"] == 1)
        except Exception:
            return False

    @staticmethod
    def _normalize_item(item: Dict[str, Any]) -> Dict[str, str]:
        return {
            "center_name": normalize_text(item.get("center_name", "")),
            "center_etype": normalize_text(item.get("center_etype", "")),
            "center_etype_zh": normalize_text(item.get("center_etype_zh", "")),
            "rel_type": normalize_text(item.get("rel_type", "")),
            "rel_type_zh": normalize_text(item.get("rel_type_zh", "")),
            "neighbor_name": normalize_text(item.get("neighbor_name", "")),
            "neighbor_etype": normalize_text(item.get("neighbor_etype", "")),
            "neighbor_etype_zh": normalize_text(item.get("neighbor_etype_zh", "")),
            "evidence": normalize_text(item.get("evidence", "")),
        }

    def retrieve(self, question: str, topk: int = 10) -> Dict[str, Any]:
        intent = self.router.classify(question)
        entity = self.extract_entity(question)

        with self.driver.session() as session:
            if entity:
                result = session.execute_read(self._route_query, entity, intent, topk)
                if result["items"]:
                    return {
                        "entity": entity,
                        "intent": intent,
                        "items": [self._normalize_item(x) for x in result["items"]],
                        "cypher": result["cypher"],
                    }

                result = session.execute_read(self._query_neighbors_by_entity, entity, topk)
                if result["items"]:
                    return {
                        "entity": entity,
                        "intent": intent,
                        "items": [self._normalize_item(x) for x in result["items"]],
                        "cypher": result["cypher"],
                    }

            keywords = self.extract_keywords(question)
            if keywords:
                result = session.execute_read(self._query_neighbors_by_keywords, keywords, topk)
                return {
                    "entity": entity,
                    "intent": intent,
                    "items": [self._normalize_item(x) for x in result["items"]],
                    "cypher": result["cypher"],
                }

        return {
            "entity": entity,
            "intent": intent,
            "items": [],
            "cypher": None,
        }

    def extract_entity(self, question: str) -> Optional[str]:
        q = normalize_text(question)

        patterns = [
            r"^(.*?)的?(主要)?症状是什么",
            r"^(.*?)会出现哪些症状",
            r"^(.*?)有哪些不良反应",
            r"^(.*?)的副作用是什么",
            r"^(.*?)相关药物有哪些",
            r"^(.*?)的治疗方法有哪些",
            r"^(.*?)如何治疗",
            r"^(.*?)的原因是什么",
            r"^(.*?)的病因是什么",
            r"^(.*?)的英文名称是什么",
            r"^(.*?)的英文名是什么",
            r"^(.*?)的别名是什么",
            r"^(.*?)的同义词是什么",
            r"^(.*?)如何诊断",
            r"^(.*?)需要做哪些检查",
            r"^(.*?)最重要的检查是什么",
            r"^(.*?)的定义是什么",
            r"^什么是(.*?)$",
            r"^(.*?)是什么",
            r"^(.*?)有哪些并发症",
            r"^(.*?)相关的疾病有哪些",
            r"^(.*?)的注意事项是什么",
            r"^(.*?)如何预防",
            r"^(.*?)的病程是如何发展的",
            r"^(.*?)相关检验指标有哪些",
            r"^(.*?)相关发病机制有哪些",
            r"^(.*?)涉及哪些解剖部位",
        ]

        for p in patterns:
            m = re.match(p, q)
            if m:
                ent = normalize_text(m.group(1))
                if ent:
                    return ent

        words = tokenize_zh(q)
        stopwords = {
            "什么", "哪些", "多少", "怎么", "如何", "为什么",
            "是否", "属于", "有关", "相关", "主要", "具体",
            "一种", "这个", "那个", "会", "是", "的", "了", "吗",
            "英文", "名称", "定义", "症状", "治疗", "原因", "病因",
            "检查", "诊断", "科室", "并发症", "注意事项", "预防", "病程",
            "药物", "副作用", "不良反应", "检验指标", "发病机制"
        }
        cands = [w for w in words if len(w) >= 2 and w not in stopwords]
        cands = uniq_keep_order(cands)
        if cands:
            cands = sorted(cands, key=lambda x: len(x), reverse=True)
            return cands[0]
        return None

    @staticmethod
    def extract_keywords(question: str) -> List[str]:
        stopwords = {
            "什么", "哪些", "多少", "怎么", "如何", "为什么",
            "是否", "属于", "有关", "相关", "主要", "具体",
            "一种", "这个", "那个", "会", "是", "的", "了", "吗",
            "请问", "一下", "英文", "名称", "定义", "主要症状",
            "不良反应", "副作用", "治疗方法", "病因", "检查", "诊断"
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
    def _return_clause() -> str:
        return """
        RETURN n.name AS center_name,
               n.etype AS center_etype,
               n.etype_zh AS center_etype_zh,
               type(r) AS rel_type,
               r.rel_type_zh AS rel_type_zh,
               m.name AS neighbor_name,
               m.etype AS neighbor_etype,
               m.etype_zh AS neighbor_etype_zh,
               r.evidence AS evidence
        """

    @classmethod
    def _evidence_query(cls, tx, entity: str, prefixes: List[str], topk: int):
        cypher = f"""
        MATCH (n:Entity)
        WHERE n.name = $entity OR n.name CONTAINS $entity
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE ANY(p IN $prefixes WHERE r.evidence CONTAINS ($entity + p))
        {cls._return_clause()}
        LIMIT $topk
        """
        items = [record.data() for record in tx.run(cypher, entity=entity, prefixes=prefixes, topk=topk)]
        return {"items": items, "cypher": cypher}

    @classmethod
    def _relation_query(cls, tx, entity: str, rel_type: str, topk: int):
        cypher = f"""
        MATCH (n:Entity)-[r:{rel_type}]-(m:Entity)
        WHERE n.name = $entity OR n.name CONTAINS $entity
        {cls._return_clause()}
        LIMIT $topk
        """
        items = [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]
        return {"items": items, "cypher": cypher}

    @classmethod
    def _route_query(cls, tx, entity: str, intent: str, topk: int):
        evidence_prefix_map = {
            "definition": ["的定义", "是什么"],
            "english_name": ["的疾病英文名", "的英文名称", "的英文名", "英文名称", "英文名"],
            "alias": ["的同义词", "的别名", "别名", "同义词"],
            "department": ["的诊疗科室", "的就诊科室"],
            "precaution": ["的注意事项", "如何预防"],
            "course": ["的病程"],
        }

        if intent in evidence_prefix_map:
            return cls._evidence_query(tx, entity, evidence_prefix_map[intent], topk)

        relation_map = {
            "symptom": "Symptom_Disease",
            "treatment": "Treatment_Disease",
            "reason": "Reason_Disease",
            "pathogenesis": "Pathogenesis_Disease",
            "test": "Test_Disease",
            "drug": "Drug_Disease",
            "ade": "ADE_Drug",
            "test_items": "Test_items_Disease",
            "class": "Class_Disease",
            "anatomy": "Anatomy_Disease",
            "related_disease": "ADE_Disease",
        }

        if intent in relation_map:
            result = cls._relation_query(tx, entity, relation_map[intent], topk)
            if result["items"]:
                return result

        if intent == "related_disease":
            result = cls._evidence_query(tx, entity, ["的相关疾病"], topk)
            if result["items"]:
                return result

        structured_fallback_prefix = {
            "symptom": ["的症状", "的主要症状", "的具体症状"],
            "treatment": ["的治疗", "的治疗方法"],
            "reason": ["的病因", "的原因"],
            "pathogenesis": ["的发病机制"],
            "test": ["的诊断", "的检查"],
            "drug": ["的相关药物"],
            "ade": ["的不良反应", "的副作用"],
            "test_items": ["的检验指标"],
            "class": ["的分类"],
            "anatomy": ["的解剖部位"],
        }

        if intent in structured_fallback_prefix:
            result = cls._evidence_query(tx, entity, structured_fallback_prefix[intent], topk)
            if result["items"]:
                return result

        return {"items": [], "cypher": None}

    @classmethod
    def _query_neighbors_by_entity(cls, tx, entity: str, topk: int):
        cypher = f"""
        MATCH (n:Entity)
        WHERE n.name = $entity OR n.name CONTAINS $entity
        OPTIONAL MATCH (n)-[r]-(m)
        {cls._return_clause()}
        LIMIT $topk
        """
        items = [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]
        return {"items": items, "cypher": cypher}

    @classmethod
    def _query_neighbors_by_keywords(cls, tx, keywords: List[str], topk: int):
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
        items = [record.data() for record in tx.run(cypher, keywords=keywords, topk=topk)]
        return {"items": items, "cypher": cypher}


# =========================
# 答案生成器
# =========================
class AdvancedGenerator:
    def __init__(self):
        self.router = QuestionRouter()

    def _extract_english_candidates(self, text: str) -> List[str]:
        text = normalize_text(text)
        if not text:
            return []

        matches = re.findall(r"[A-Za-z][A-Za-z\s\-\(\)/]+[A-Za-z]", text)

        candidates = []
        for m in matches:
            m = normalize_text(m)
            if len(m) < 4:
                continue
            if len(m.split()) == 1 and len(m) < 6:
                continue
            candidates.append(m)

        return uniq_keep_order(candidates)

    def _extract_alias_candidates(self, text: str) -> List[str]:
        text = normalize_text(text)
        if not text:
            return []

        patterns = [
            r"同义词[:：]\s*([^\n。；;]+)",
            r"别名[:：]\s*([^\n。；;]+)",
            r"又称[:：]?\s*([^\n。；;]+)",
        ]

        results = []
        for p in patterns:
            matches = re.findall(p, text)
            for m in matches:
                m = normalize_text(m)
                if m:
                    parts = re.split(r"[、,，/；; ]+", m)
                    for part in parts:
                        part = normalize_text(part)
                        if len(part) >= 2:
                            results.append(part)

        return uniq_keep_order(results)

    def _pick_best_evidence(self, evidences: List[str], center_name: str = "") -> Optional[str]:
        evidences = [normalize_text(x) for x in evidences if normalize_text(x)]
        if not evidences:
            return None

        if center_name:
            with_center = [x for x in evidences if center_name in x]
            if with_center:
                with_center = sorted(with_center, key=len, reverse=True)
                return with_center[0]

        evidences = sorted(evidences, key=len, reverse=True)
        return evidences[0]

    def generate(self, question: str, retrieved: List[Dict[str, str]]) -> str:
        if not retrieved:
            return "未从知识图谱中检索到相关信息。"

        intent = self.router.classify(question)

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

        center_name = facts[0]["center"] if facts else ""

        if intent == "english_name":
            english_candidates = []
            for ev in evidences:
                english_candidates.extend(self._extract_english_candidates(ev))
            english_candidates = uniq_keep_order(english_candidates)

            if english_candidates:
                english_candidates = sorted(english_candidates, key=len, reverse=True)
                best = english_candidates[0]
                if center_name:
                    return f"{center_name}的英文名称是 {best}。"
                return f"英文名称是 {best}。"

            best_ev = self._pick_best_evidence(evidences, center_name)
            if best_ev:
                return best_ev

        if intent == "alias":
            alias_candidates = []
            for ev in evidences:
                alias_candidates.extend(self._extract_alias_candidates(ev))
            alias_candidates = uniq_keep_order(alias_candidates)

            if alias_candidates:
                if center_name:
                    return f"{center_name}的别名或同义词包括：{'、'.join(alias_candidates[:10])}。"
                return f"别名或同义词包括：{'、'.join(alias_candidates[:10])}。"

            best_ev = self._pick_best_evidence(evidences, center_name)
            if best_ev:
                return best_ev

        if intent in {"definition", "department", "precaution", "course"}:
            best_ev = self._pick_best_evidence(evidences, center_name)
            if best_ev:
                return best_ev

        if intent in {
            "symptom", "treatment", "reason", "pathogenesis",
            "test", "drug", "ade", "test_items",
            "class", "anatomy", "related_disease"
        }:
            values = [f["neighbor"] for f in facts if f["neighbor"]]
            values = uniq_keep_order(values)

            if values:
                prefix_map = {
                    "symptom": "相关症状包括",
                    "treatment": "相关治疗方法包括",
                    "reason": "相关病因包括",
                    "pathogenesis": "相关发病机制包括",
                    "test": "相关检查包括",
                    "drug": "相关药物包括",
                    "ade": "相关不良反应包括",
                    "test_items": "相关检验指标包括",
                    "class": "相关分类包括",
                    "anatomy": "相关解剖部位包括",
                    "related_disease": "相关疾病包括",
                }
                prefix = prefix_map.get(intent, "相关内容包括")
                return f"{prefix}：{'、'.join(values[:10])}。"

            best_ev = self._pick_best_evidence(evidences, center_name)
            if best_ev:
                return best_ev

        lines = []
        fact_seen = set()

        for f in facts[:5]:
            key = (f["center"], f["rel"], f["neighbor"])
            if key in fact_seen:
                continue
            fact_seen.add(key)
            line = f"{f['center_type']}“{f['center']}”与{f['neighbor_type']}“{f['neighbor']}”存在“{f['rel']}”关系。"
            lines.append(line)

        if lines:
            best_ev = self._pick_best_evidence(evidences, center_name)
            if best_ev:
                return " ".join(lines) + f" 证据：{best_ev}"
            return " ".join(lines)

        best_ev = self._pick_best_evidence(evidences, center_name)
        if best_ev:
            return best_ev

        return "已检索到实体，但未找到足够的关系信息。"


# =========================
# FastAPI 应用
# =========================
app = FastAPI(
    title="糖尿病知识图谱问答系统",
    description="基于 Neo4j 的问答系统后端（结构化优先 + 证据优先混合策略）",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = KGQARetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
generator = AdvancedGenerator()


@app.get("/")
def root():
    return {
        "message": "糖尿病知识图谱问答系统后端已启动",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    ok = retriever.ping()
    return {
        "status": "ok" if ok else "error",
        "neo4j_connected": ok
    }


@app.get("/schema")
def schema():
    return {
        "supported_intents": [
            "definition",
            "english_name",
            "alias",
            "department",
            "precaution",
            "course",
            "symptom",
            "treatment",
            "reason",
            "pathogenesis",
            "test",
            "drug",
            "ade",
            "test_items",
            "class",
            "anatomy",
            "related_disease",
            "general"
        ]
    }


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    question = normalize_text(req.question)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    t1 = time.perf_counter()
    retrieved_result = retriever.retrieve(question, topk=TOPK)
    t2 = time.perf_counter()

    answer = generator.generate(question, retrieved_result["items"])
    t3 = time.perf_counter()

    evidence_chain = [EvidenceItem(**item) for item in retrieved_result["items"][:8]]

    return QAResponse(
        question=question,
        entity=retrieved_result["entity"],
        intent=retrieved_result["intent"],
        answer=answer,
        cypher=retrieved_result["cypher"],
        retrieved_count=len(retrieved_result["items"]),
        retrieval_time_sec=round(t2 - t1, 6),
        generation_time_sec=round(t3 - t2, 6),
        total_time_sec=round(t3 - t1, 6),
        evidence_chain=evidence_chain,
    )


@app.on_event("shutdown")
def shutdown_event():
    retriever.close()
