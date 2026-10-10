# -*- coding: utf-8 -*-
import os
import re
import time
from typing import List, Dict, Optional, Any, Tuple

import jieba
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from openai import AsyncOpenAI

# =========================
# 基础配置
# =========================
# ============================================================================
# 重要说明：API 密钥和数据库密码配置
# ============================================================================
# 本项目使用 Neo4j 图数据库和多个大语言模型 API，所有敏感信息必须通过环境变量配置。
# 请勿将真实的 API 密钥、数据库密码写入代码或提交到版本控制系统！
#
# 配置方法：
# 1. 在项目根目录创建 .env 文件（已提供 .env 示例文件）
# 2. 在 .env 文件中配置以下环境变量：
#    NEO4J_URI=bolt://localhost:7687
#    NEO4J_USER=neo4j
#    NEO4J_PASSWORD=your_neo4j_password
#    DEEPSEEK_API_KEY=your_deepseek_api_key
#    MOONSHOT_API_KEY=your_moonshot_api_key
#    GLM_API_KEY=your_glm_api_key
#    DASHSCOPE_API_KEY=your_dashscope_api_key
# 3. 或者在系统环境变量中配置上述变量
#
# 获取 API Key：
# - DeepSeek: https://platform.deepseek.com/
# - Moonshot (Kimi): https://platform.moonshot.cn/
# - 智谱 GLM: https://open.bigmodel.cn/
# - 阿里云百炼 (Qwen): https://dashscope.aliyun.com/
# ============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # 必须从环境变量读取，禁止硬编码
TOPK = int(os.getenv("TOPK", "10"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
KG_CONTEXT_MAX_ITEMS = int(os.getenv("KG_CONTEXT_MAX_ITEMS", "8"))

DEFAULT_THINKING_HINT = "start deep thinking, please always use the thinking mode"
DEFAULT_WEB_HINT = "When needed, search authoritative web sources before answering and explicitly integrate up-to-date online information."

SYSTEM_PROMPT = """你是中文医学糖尿病问答专家。
请严格遵守：
1. 只回答与糖尿病、糖尿病并发症、糖尿病药物、糖尿病检查、糖尿病治疗相关的问题。
2. 优先依据给定的知识图谱证据回答，禁止编造证据中没有明确支持的内容。
3. 如果证据足够，先直接给出结论，再用自然语言补充。
4. 如果证据只能支持部分答案，请先回答可确定的部分，再补一句“图谱中未见更直接证据，建议结合医生意见”。
5. 如果完全没有相关证据，可以结合通用医学常识谨慎回答，但必须明确说明“知识图谱未检索到充分证据，以下为通用医学建议”。
6. 回答简洁、专业、自然，不要输出思维过程。
7. 尽量控制在 2~5 句话。
"""

BUILTIN_MODELS: Dict[str, Dict[str, str]] = {
    "deepseek-v3": {
        "provider_name": "DeepSeek",
        "display_name": "deepseek-v3（内置免费）",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "deepseek-chat",
        "param_style": "standard",
    },
    "deepseek-r1": {
        "provider_name": "DeepSeek",
        "display_name": "deepseek-r1（内置免费）",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "deepseek-reasoner",
        "param_style": "no_temperature",
    },
    "kimi-2.5": {
        "provider_name": "Moonshot",
        "display_name": "kimi-2.5（内置免费）",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": os.getenv("MOONSHOT_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "kimi-k2.5",
        "param_style": "kimi_k25",
    },
    "glm-5.1": {
        "provider_name": "智谱 GLM",
        "display_name": "glm-5.1（内置免费）",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.getenv("GLM_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "glm-5.1",
        "param_style": "standard",
    },
    "qwen-3.6": {
        "provider_name": "阿里云百炼",
        "display_name": "qwen-3.6（内置免费）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "qwen-plus",
        "param_style": "standard",
    },
}

LEGACY_MODEL_ALIASES: Dict[str, str] = {
    "deepseek-chat": "deepseek-v3",
    "deepseek-reasoner": "deepseek-r1",
    "kimi-k2.5": "kimi-2.5",
    "glm-5.1": "glm-5.1",
    "qwen-plus": "qwen-3.6",
}

SOFT_STOPWORDS = {
    "什么", "哪些", "多少", "怎么", "如何", "为什么", "是否", "属于", "有关", "相关", "主要", "具体",
    "一种", "这个", "那个", "会", "是", "的", "了", "吗", "请问", "一下", "英文", "名称", "定义",
    "症状", "治疗", "原因", "病因", "检查", "诊断", "科室", "并发症", "注意事项", "预防", "病程",
    "方法", "包括", "出现", "会有", "什么样", "如何进行", "进行", "哪些方面", "多久", "频繁"
}


# =========================
# 工具函数
# =========================
def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().replace("\u3000", " ")
    return re.sub(r"\s+", " ", text)


def mask_secret(secret: str) -> str:
    secret = normalize_text(secret)
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * max(4, len(secret) - 8)}{secret[-4:]}"


def tokenize_zh(text: str) -> List[str]:
    return [w.strip() for w in jieba.lcut(normalize_text(text)) if w.strip()]


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def jaccard_char_similarity(a: str, b: str) -> float:
    a_set = set(normalize_text(a))
    b_set = set(normalize_text(b))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def overlap_score(query_terms: List[str], text: str) -> float:
    text_norm = normalize_text(text)
    if not query_terms or not text_norm:
        return 0.0
    hit = sum(1 for t in query_terms if t and t in text_norm)
    return hit / len(query_terms)


# =========================
# 数据模型
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
    score: Optional[float] = None


class RetrievalDebug(BaseModel):
    raw_retrieved_count: int = 0
    filtered_count: int = 0
    retrieval_stage: str = "none"
    relevant_score: float = 0.0
    keywords: List[str] = Field(default_factory=list)
    kg_context: str = ""


class CustomModelConfig(BaseModel):
    config_id: Optional[str] = ""
    display_name: Optional[str] = ""
    provider_name: Optional[str] = "custom"
    base_url: str
    api_key: str
    model: str
    note: Optional[str] = ""


class QARequest(BaseModel):
    question: str
    model_name: str = Field(default="deepseek-v3")
    enable_thinking: bool = Field(default=True)
    enable_web_search: bool = Field(default=False)
    custom_model_config: Optional[CustomModelConfig] = None


class QAResponse(BaseModel):
    question: str
    entity: Optional[str]
    intent: str
    answer: str
    model_name: str
    provider_name: Optional[str] = None
    display_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key_masked: Optional[str] = None
    actual_model: Optional[str] = None
    custom_config_id: Optional[str] = None
    cypher: Optional[str] = None
    retrieved_count: int
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    evidence_chain: List[EvidenceItem]
    retrieval_debug: RetrievalDebug


# =========================
# 路由与知识图谱检索
# =========================
class QuestionRouter:
    def classify(self, question: str) -> str:
        q = normalize_text(question)

        if any(x in q for x in ["英文", "英文名", "英文名称", "英文叫什么"]):
            return "english_name"
        if any(x in q for x in ["同义词", "别名", "另一名称", "别称"]):
            return "alias"
        if any(x in q for x in ["定义", "什么是", "是一种什么疾病", "是什么病"]):
            return "definition"
        if any(x in q for x in ["属于什么", "属于哪类", "属于哪一种", "类别", "分型", "类型", "分类"]):
            return "category"
        if any(x in q for x in ["症状", "表现", "会出现哪些", "会有什么感觉", "主要症状", "具体症状"]):
            return "symptom"
        if any(x in q for x in ["治疗", "怎么治", "如何治疗", "治疗方法", "主要治疗方法"]):
            return "treatment"
        if any(x in q for x in ["饮食疗法", "饮食", "怎么吃", "吃什么"]):
            return "diet"
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
        if any(x in q for x in ["病程", "发展", "会如何变化", "如何进展", "预后"]):
            return "course"
        if any(x in q for x in ["多久", "多频繁", "频率", "间隔", "多长时间"]):
            return "frequency"
        return "general"


class KGQARetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.router = QuestionRouter()

    def close(self):
        self.driver.close()

    def ping(self) -> bool:
        try:
            with self.driver.session() as session:
                record = session.run("RETURN 1 AS ok").single()
                return bool(record and record["ok"] == 1)
        except Exception:
            return False

    def retrieve(self, question: str, topk: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        intent = self.router.classify(question)
        entity = self.extract_entity(question)
        keywords = self.extract_keywords(question)
        debug = {
            "intent": intent,
            "entity": entity or "",
            "keywords": keywords,
            "raw_retrieved_count": 0,
            "filtered_count": 0,
            "retrieval_stage": "none",
            "relevant_score": 0.0,
        }

        candidates: List[Dict[str, Any]] = []
        stage = "none"
        with self.driver.session() as session:
            if entity:
                exact_results = session.execute_read(self._route_query_exact, entity, intent, topk * 3)
                if exact_results:
                    candidates.extend(exact_results)
                    stage = "route_exact"

                exact_neighbor = session.execute_read(self._query_neighbors_by_entity_exact, entity, topk * 3)
                if exact_neighbor:
                    candidates.extend(exact_neighbor)
                    if stage == "none":
                        stage = "entity_exact"

                contains_results = session.execute_read(self._route_query_contains, entity, intent, topk * 3)
                if contains_results:
                    candidates.extend(contains_results)
                    if stage == "none":
                        stage = "route_contains"

                contains_neighbor = session.execute_read(self._query_neighbors_by_entity_contains, entity, topk * 3)
                if contains_neighbor:
                    candidates.extend(contains_neighbor)
                    if stage == "none":
                        stage = "entity_contains"

            if keywords:
                keyword_results = session.execute_read(self._query_neighbors_by_keywords, keywords, topk * 4)
                if keyword_results:
                    candidates.extend(keyword_results)
                    if stage == "none":
                        stage = "keyword"

        debug["raw_retrieved_count"] = len(candidates)
        filtered = self.filter_and_rank_results(question, entity, intent, keywords, candidates, topk)
        debug["filtered_count"] = len(filtered)
        debug["retrieval_stage"] = stage
        debug["relevant_score"] = safe_ratio(len(filtered), max(len(candidates), 1))
        return filtered, debug

    def extract_entity(self, question: str) -> Optional[str]:
        q = normalize_text(question)
        patterns = [
            r"^(.*?)的?(主要)?症状是什么",
            r"^(.*?)会出现哪些症状",
            r"^(.*?)的治疗方法有哪些",
            r"^(.*?)如何治疗",
            r"^(.*?)的饮食疗法.*",
            r"^(.*?)的原因是什么",
            r"^(.*?)的病因是什么",
            r"^(.*?)的英文名称是什么",
            r"^(.*?)的英文名是什么",
            r"^(.*?)如何诊断",
            r"^(.*?)需要做哪些检查",
            r"^(.*?)最重要的检查是什么",
            r"^(.*?)的定义是什么",
            r"^(.*?)属于什么.*",
            r"^(.*?)属于哪.*",
            r"^什么是(.*?)$",
            r"^(.*?)是什么",
            r"^(.*?)有哪些并发症",
            r"^(.*?)相关的疾病有哪些",
            r"^(.*?)应多频繁接受.*检查",
            r"^(.*?)多久.*检查",
        ]

        for p in patterns:
            m = re.match(p, q)
            if m:
                ent = normalize_text(m.group(1))
                ent = re.sub(r"(是否|应该|需要|可以)$", "", ent)
                if ent and len(ent) >= 2:
                    return ent

        tokens = tokenize_zh(q)
        cands = [w for w in tokens if len(w) >= 2 and w not in SOFT_STOPWORDS]
        cands = uniq_keep_order(cands)
        cands = [c for c in cands if not re.fullmatch(r"[a-zA-Z0-9\-\.]+", c)]
        if cands:
            cands = sorted(cands, key=lambda x: (len(x), q.find(x) if x in q else 9999), reverse=True)
            return cands[0]
        return None

    @staticmethod
    def extract_keywords(question: str) -> List[str]:
        tokens = tokenize_zh(question)
        keywords = []
        for w in tokens:
            w = w.strip()
            if not w or w in SOFT_STOPWORDS or len(w) < 2:
                continue
            keywords.append(w)
        return uniq_keep_order(keywords)[:8]

    def filter_and_rank_results(
        self,
        question: str,
        entity: Optional[str],
        intent: str,
        keywords: List[str],
        candidates: List[Dict[str, Any]],
        topk: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        route_map = {
            "symptom": ["症状"],
            "treatment": ["治疗"],
            "reason": ["病因", "原因"],
            "pathogenesis": ["发病机制", "机制"],
            "test": ["检查", "诊断"],
            "department": ["科室", "就诊"],
            "english_name": ["英文"],
            "definition": ["定义", "是什么"],
            "category": ["属于", "类别", "类型"],
            "related_disease": ["并发症", "相关疾病"],
            "precaution": ["注意", "预防"],
            "course": ["病程", "进展", "预后"],
            "diet": ["饮食", "饮食疗法"],
            "frequency": ["频率", "多久", "多频繁"],
        }
        intent_clues = route_map.get(intent, [])

        scored = []
        seen = set()
        for item in candidates:
            center = normalize_text(item.get("center_name", ""))
            rel = normalize_text(item.get("rel_type_zh", "")) or normalize_text(item.get("rel_type", ""))
            neighbor = normalize_text(item.get("neighbor_name", ""))
            evidence = normalize_text(item.get("evidence", ""))
            combined_text = " ".join([center, rel, neighbor, evidence])
            key = (center, rel, neighbor, evidence)
            if key in seen:
                continue
            seen.add(key)

            score = 0.0
            strong_entity_match = False
            if entity:
                if center == entity or neighbor == entity:
                    score += 4.0
                    strong_entity_match = True
                if entity in center:
                    score += 2.2
                if entity in evidence:
                    score += 2.0
                if entity in neighbor:
                    score += 1.5
                score += 1.5 * jaccard_char_similarity(entity, center)
                score += 0.8 * jaccard_char_similarity(entity, neighbor)

            score += 2.0 * overlap_score(keywords, combined_text)
            score += 1.0 * overlap_score(intent_clues, combined_text)

            if rel and any(clue in rel for clue in intent_clues):
                score += 1.2
            if evidence and len(evidence) > 10:
                score += 0.3
            if center and len(center) >= 3:
                score += 0.2

            kw_overlap = overlap_score(keywords, combined_text)
            if entity:
                if not strong_entity_match and entity not in center and entity not in evidence and jaccard_char_similarity(entity, center) < 0.35 and kw_overlap < 0.34:
                    continue
            elif kw_overlap < 0.34:
                continue

            item2 = dict(item)
            item2["score"] = round(score, 4)
            scored.append(item2)

        scored.sort(key=lambda x: (x.get("score", 0.0), len(normalize_text(x.get("evidence", "")))), reverse=True)
        if not scored:
            return []

        best_score = scored[0].get("score", 0.0)
        threshold = max(1.4, best_score * 0.35)
        pruned = [x for x in scored if x.get("score", 0.0) >= threshold]
        return pruned[:topk]

    @staticmethod
    def _base_return_clause() -> str:
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
        LIMIT $topk
        """

    @staticmethod
    def _route_query_exact(tx, entity: str, intent: str, topk: int):
        route_map = {
            "symptom": "症状",
            "treatment": "治疗方法",
            "reason": "病因",
            "pathogenesis": "发病机制",
            "test": "检查",
            "department": "就诊科室",
            "english_name": "疾病英文名",
            "definition": "定义",
            "category": "属于",
            "related_disease": "相关疾病",
            "precaution": "注意事项",
            "course": "病程",
            "diet": "饮食",
            "frequency": "频率",
        }
        prefix = route_map.get(intent)
        if not prefix:
            return []
        cypher = """
        MATCH (n:Entity)
        WHERE n.name = $entity
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE r.evidence CONTAINS ($entity + '的' + $prefix)
           OR r.evidence CONTAINS ($entity + '的' + replace($prefix, '就诊科室', '诊疗科室'))
           OR r.evidence CONTAINS ($entity + '的' + replace($prefix, '病因', '原因'))
           OR r.rel_type_zh CONTAINS $prefix
        """ + KGQARetriever._base_return_clause()
        return [record.data() for record in tx.run(cypher, entity=entity, prefix=prefix, topk=topk)]

    @staticmethod
    def _route_query_contains(tx, entity: str, intent: str, topk: int):
        route_queries = {
            "symptom": "MATCH (n:Entity)-[r:Symptom_Disease]-(m:Entity) WHERE n.name CONTAINS $entity ",
            "treatment": "MATCH (n:Entity)-[r:Treatment_Disease]-(m:Entity) WHERE n.name CONTAINS $entity ",
            "reason": "MATCH (m:Entity)-[r:Reason_Disease]-(n:Entity) WHERE n.name CONTAINS $entity ",
            "pathogenesis": "MATCH (m:Entity)-[r:Pathogenesis_Disease]-(n:Entity) WHERE n.name CONTAINS $entity ",
            "test": "MATCH (m:Entity)-[r:Test_Disease]-(n:Entity) WHERE n.name CONTAINS $entity ",
        }
        cypher = route_queries.get(intent)
        if not cypher:
            return []
        cypher += KGQARetriever._base_return_clause()
        return [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]

    @staticmethod
    def _query_neighbors_by_entity_exact(tx, entity: str, topk: int):
        cypher = """
        MATCH (n:Entity)
        WHERE n.name = $entity
        OPTIONAL MATCH (n)-[r]-(m)
        """ + KGQARetriever._base_return_clause()
        return [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]

    @staticmethod
    def _query_neighbors_by_entity_contains(tx, entity: str, topk: int):
        cypher = """
        MATCH (n:Entity)
        WHERE n.name CONTAINS $entity
        OPTIONAL MATCH (n)-[r]-(m)
        """ + KGQARetriever._base_return_clause()
        return [record.data() for record in tx.run(cypher, entity=entity, topk=topk)]

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
        return [record.data() for record in tx.run(cypher, keywords=keywords, topk=topk)]


# =========================
# KG 上下文序列化
# =========================
def serialize_kg_context(question: str, entity: str, intent: str, retrieved: List[Dict[str, Any]], max_items: int = 10) -> str:
    if not retrieved:
        return "未检索到直接相关的知识图谱证据。"

    lines = [f"问题：{normalize_text(question)}"]
    if entity:
        lines.append(f"识别到的核心实体：{entity}")
    if intent:
        lines.append(f"问题类型：{intent}")
    lines.append("知识图谱证据：")

    for idx, item in enumerate(retrieved[:max_items], 1):
        center = normalize_text(item.get("center_name", ""))
        center_type = normalize_text(item.get("center_etype_zh", "")) or normalize_text(item.get("center_etype", ""))
        rel = normalize_text(item.get("rel_type_zh", "")) or normalize_text(item.get("rel_type", ""))
        neighbor = normalize_text(item.get("neighbor_name", ""))
        neighbor_type = normalize_text(item.get("neighbor_etype_zh", "")) or normalize_text(item.get("neighbor_etype", ""))
        evidence = normalize_text(item.get("evidence", ""))
        score = item.get("score", "")

        parts = [f"{idx}. 中心实体：{center}"]
        if center_type:
            parts.append(f"中心类型：{center_type}")
        if rel:
            parts.append(f"关系：{rel}")
        if neighbor:
            parts.append(f"相关实体：{neighbor}")
        if neighbor_type:
            parts.append(f"相关类型：{neighbor_type}")
        if evidence:
            parts.append(f"证据：{evidence}")
        if score != "":
            parts.append(f"相关性得分：{score}")
        lines.append("；".join(parts))

    lines.append("回答要求：优先根据高相关证据直接作答；只有在没有直接相关信息时，才说明证据不足。")
    return "\n".join(lines)


# =========================
# 融合式 LLM 生成器
# =========================
class HybridLLMGenerator:
    def resolve_model(self, model_name: str, custom_model_config: Optional[CustomModelConfig]) -> Tuple[str, Dict[str, str]]:
        normalized = normalize_text(model_name)
        normalized = LEGACY_MODEL_ALIASES.get(normalized, normalized)
        if normalized.startswith("custom:"):
            normalized = "custom"
        if normalized == "custom":
            if not custom_model_config:
                raise HTTPException(status_code=400, detail="当前选择了自定义模型，但未提供 custom_model_config")
            return "custom", {
                "provider_name": normalize_text(custom_model_config.provider_name) or "custom",
                "display_name": normalize_text(custom_model_config.display_name) or "自定义模型",
                "base_url": normalize_text(custom_model_config.base_url),
                "api_key": normalize_text(custom_model_config.api_key),
                "actual_model": normalize_text(custom_model_config.model),
                "param_style": "standard",
                "config_id": normalize_text(custom_model_config.config_id),
            }
        cfg = BUILTIN_MODELS.get(normalized)
        if not cfg:
            raise HTTPException(status_code=400, detail=f"不支持的模型：{model_name}")
        return normalized, cfg.copy()

    async def generate(
        self,
        question: str,
        entity: Optional[str],
        intent: str,
        model_name: str,
        enable_thinking: bool,
        enable_web_search: bool,
        custom_model_config: Optional[CustomModelConfig],
        kg_context: str,
        evidence_count: int,
    ) -> Tuple[str, Dict[str, str]]:
        resolved_name, cfg = self.resolve_model(model_name, custom_model_config)
        if not cfg.get("api_key"):
            return f"模型 {resolved_name} 未配置 API Key，无法生成回答。", cfg

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if enable_thinking:
            messages.append({"role": "system", "content": DEFAULT_THINKING_HINT})
        if enable_web_search:
            messages.append({"role": "system", "content": DEFAULT_WEB_HINT})

        user_prompt = (
            f"请回答下面这个糖尿病相关问题。\n\n"
            f"问题类型：{intent}\n"
            f"识别实体：{entity or '未识别'}\n"
            f"知识图谱命中证据数：{evidence_count}\n"
            f"联网模式：{'开启' if enable_web_search else '关闭'}\n"
            f"深度思考：{'开启' if enable_thinking else '关闭'}\n\n"
            f"【知识图谱证据】\n{kg_context}\n\n"
            f"【用户问题】\n{question}\n\n"
            "输出要求：\n"
            "- 直接回答，中文输出\n"
            "- 优先整合知识图谱证据作答\n"
            "- 不要逐条复述证据，也不要输出分析过程\n"
            "- 若证据不足但可以给出通用建议，必须明确提示知识图谱证据不足\n"
            "- 尽量控制在 2~5 句话\n"
        )
        messages.append({"role": "user", "content": user_prompt})

        client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=LLM_TIMEOUT)
        request_kwargs: Dict[str, Any] = {
            "model": cfg["actual_model"],
            "messages": messages,
        }

        param_style = cfg.get("param_style", "standard")
        if param_style == "standard":
            request_kwargs["temperature"] = 0.2
        elif param_style == "no_temperature":
            pass
        elif param_style == "kimi_k25":
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if enable_thinking else "disabled"}
            }
        else:
            request_kwargs["temperature"] = 0.2

        try:
            resp = await client.chat.completions.create(**request_kwargs)
            answer = ""
            if resp and resp.choices and resp.choices[0].message:
                answer = resp.choices[0].message.content or ""
            return normalize_text(answer) or "模型未返回有效内容。", cfg
        except Exception as e:
            return f"模型调用失败：{e}", cfg


# =========================
# FastAPI 应用
# =========================
app = FastAPI(
    title="糖尿病大模型 + 知识图谱融合问答系统",
    description="基于 Neo4j 检索增强与多模型生成的融合式糖尿病问答系统",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = QuestionRouter()
retriever = KGQARetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
generator = HybridLLMGenerator()


@app.get("/")
def root():
    return {
        "message": "糖尿病大模型 + 知识图谱融合问答系统后端已启动",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    neo4j_ok = retriever.ping()
    return {
        "status": "ok" if neo4j_ok else "degraded",
        "neo4j_connected": neo4j_ok,
        "supported_models": list(BUILTIN_MODELS.keys()) + ["custom"],
        "builtin_models": [
            {
                "model_name": k,
                "display_name": v["display_name"],
                "provider_name": v["provider_name"],
                "actual_model": v["actual_model"],
                "base_url": v["base_url"],
            }
            for k, v in BUILTIN_MODELS.items()
        ],
        "default_thinking_hint": DEFAULT_THINKING_HINT,
        "default_web_hint": DEFAULT_WEB_HINT,
    }


@app.get("/schema")
def schema():
    return {
        "supported_models": list(BUILTIN_MODELS.keys()) + ["custom"],
        "supported_intents": [
            "definition", "english_name", "alias", "department", "precaution", "course", "category",
            "symptom", "treatment", "reason", "pathogenesis", "test", "diet", "related_disease",
            "frequency", "general"
        ],
    }


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    question = normalize_text(req.question)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    t1 = time.perf_counter()
    retrieved_items, debug = retriever.retrieve(question, topk=TOPK)
    entity = debug.get("entity") or router.classify(question)
    intent = debug.get("intent") or router.classify(question)
    kg_context = serialize_kg_context(question, debug.get("entity", ""), intent, retrieved_items, max_items=KG_CONTEXT_MAX_ITEMS)
    t2 = time.perf_counter()

    answer, cfg = await generator.generate(
        question=question,
        entity=debug.get("entity", ""),
        intent=intent,
        model_name=req.model_name,
        enable_thinking=req.enable_thinking,
        enable_web_search=req.enable_web_search,
        custom_model_config=req.custom_model_config,
        kg_context=kg_context,
        evidence_count=len(retrieved_items),
    )
    t3 = time.perf_counter()

    model_name = LEGACY_MODEL_ALIASES.get(normalize_text(req.model_name), normalize_text(req.model_name))
    if model_name.startswith("custom:"):
        model_name = "custom"

    return QAResponse(
        question=question,
        entity=debug.get("entity") or None,
        intent=intent,
        answer=answer,
        model_name=model_name,
        provider_name=cfg.get("provider_name"),
        display_name=cfg.get("display_name"),
        base_url=cfg.get("base_url"),
        api_key_masked=mask_secret(cfg.get("api_key", "")),
        actual_model=cfg.get("actual_model"),
        custom_config_id=cfg.get("config_id", ""),
        cypher=None,
        retrieved_count=len(retrieved_items),
        retrieval_time_sec=round(t2 - t1, 6),
        generation_time_sec=round(t3 - t2, 6),
        total_time_sec=round(t3 - t1, 6),
        evidence_chain=[EvidenceItem(**item) for item in retrieved_items[:TOPK]],
        retrieval_debug=RetrievalDebug(
            raw_retrieved_count=debug.get("raw_retrieved_count", 0),
            filtered_count=debug.get("filtered_count", 0),
            retrieval_stage=debug.get("retrieval_stage", "none"),
            relevant_score=round(float(debug.get("relevant_score", 0.0)), 4),
            keywords=debug.get("keywords", []),
            kg_context=kg_context,
        ),
    )


@app.on_event("shutdown")
def shutdown_event():
    retriever.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_kg_llm:app", host="0.0.0.0", port=8003, reload=True)
