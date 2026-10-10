# -*- coding: utf-8 -*-
"""
糖尿病 大模型 + RAG + 知识图谱 融合问答系统后端

启动：
  python app_rag_llm_kg.py
或：
  uvicorn app_llm_rag_kg:app --host 0.0.0.0 --port 8005 --reload

依赖：
  pip install fastapi uvicorn neo4j jieba numpy openai faiss-cpu pydantic

说明：
  - RAG 检索复用 FAISS 向量库目录 ./diakg_faiss_index
  - KG 检索连接 Neo4j，默认 bolt://localhost:7687
  - 两路证据通过 Reciprocal Rank Fusion 融合后交给 LLM 生成答案
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jieba
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

# ============================================================
# 内置配置区
# ============================================================
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
#    DASHSCOPE_API_KEY=your_dashscope_api_key
#    DEEPSEEK_API_KEY=your_deepseek_api_key
#    MOONSHOT_API_KEY=your_moonshot_api_key
#    GLM_API_KEY=your_glm_api_key
# 3. 或者在系统环境变量中配置上述变量
#
# 获取 API Key：
# - DeepSeek: https://platform.deepseek.com/
# - Moonshot (Kimi): https://platform.moonshot.cn/
# - 智谱 GLM: https://open.bigmodel.cn/
# - 阿里云百炼 (Qwen): https://dashscope.aliyun.com/
# ============================================================================

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8005
LLM_TIMEOUT = 120.0

RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "./diakg_faiss_index")
RAG_TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "5"))
KG_TOP_K_DEFAULT = int(os.getenv("KG_TOP_K", "10"))
FUSION_TOP_K_DEFAULT = int(os.getenv("FUSION_TOP_K", "8"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.0"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # 必须从环境变量读取，禁止硬编码

DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 必须从环境变量读取，禁止硬编码
DEFAULT_EMBEDDING_BASE_URL = DASHSCOPE_BASE_URL
DEFAULT_EMBEDDING_API_KEY = DASHSCOPE_API_KEY
DEFAULT_EMBEDDING_MODEL = ""  # 留空则读取 meta.json
DEFAULT_EMBEDDING_DIM = 0

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 必须从环境变量读取，禁止硬编码
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")  # 必须从环境变量读取，禁止硬编码
GLM_API_KEY = os.getenv("GLM_API_KEY")  # 必须从环境变量读取，禁止硬编码

DEFAULT_THINKING_HINT = "start deep thinking, please always use the thinking mode"
DEFAULT_WEB_HINT = "When needed, search authoritative web sources before answering and explicitly integrate up-to-date online information."

SYSTEM_PROMPT = """你是中文医学糖尿病问答专家。
请严格遵守：
1. 只回答与糖尿病、糖尿病并发症、糖尿病药物、糖尿病检查、糖尿病治疗相关的问题。
2. 必须优先依据系统提供的 KG 结构化证据和 RAG 文本证据回答。
3. KG 证据优先用于实体、关系、类型、药物、不良反应、症状、检查等明确事实；RAG 证据优先用于定义、解释、机制和长文本补充。
4. 如果 KG 与 RAG 证据冲突，请说明“现有证据存在不一致，无法确定”，不要编造。
5. 如果证据不足，请回答“根据现有资料无法确定”，并建议结合医生意见。
6. 回答风格简洁、专业、自然，不输出思维过程，通常 1~3 句话。
"""

BUILTIN_MODELS: Dict[str, Dict[str, str]] = {
    "deepseek-v3": {
        "provider_name": "DeepSeek",
        "display_name": "deepseek-v3（内置免费）",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": DEEPSEEK_API_KEY,
        "actual_model": "deepseek-chat",
        "param_style": "standard",
    },
    "deepseek-r1": {
        "provider_name": "DeepSeek",
        "display_name": "deepseek-r1（内置免费）",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": DEEPSEEK_API_KEY,
        "actual_model": "deepseek-reasoner",
        "param_style": "no_temperature",
    },
    "kimi-2.5": {
        "provider_name": "Moonshot",
        "display_name": "kimi-2.5（内置免费）",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": MOONSHOT_API_KEY,
        "actual_model": "kimi-k2.5",
        "param_style": "kimi_temperature_1",
    },
    "glm-5.1": {
        "provider_name": "智谱 GLM",
        "display_name": "glm-5.1（内置免费）",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": GLM_API_KEY,
        "actual_model": "glm-5.1",
        "param_style": "standard",
    },
    "qwen-3.6": {
        "provider_name": "阿里云百炼",
        "display_name": "qwen-3.6（内置免费）",
        "base_url": DASHSCOPE_BASE_URL,
        "api_key": DASHSCOPE_API_KEY,
        "actual_model": "qwen-plus",
        "param_style": "standard",
    },
}

LEGACY_MODEL_ALIASES = {
    "deepseek-chat": "deepseek-v3",
    "deepseek-reasoner": "deepseek-r1",
    "kimi-k2.5": "kimi-2.5",
    "glm-5.1": "glm-5.1",
    "qwen-plus": "qwen-3.6",
}

# ============================================================
# 通用工具
# ============================================================
def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().replace("\u3000", " "))


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
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

# ============================================================
# 数据模型
# ============================================================
class EvidenceItem(BaseModel):
    source_type: str = ""  # rag / kg
    rank: int = 0
    score: float = 0.0
    fusion_score: float = 0.0
    title: str = ""
    text: str = ""
    sources: str = ""
    # RAG 字段
    chunk_id: str = ""
    source_file: str = ""
    doc_id: str = ""
    paragraph_id: str = ""
    # KG 字段
    center_name: str = ""
    center_etype: str = ""
    center_etype_zh: str = ""
    rel_type: str = ""
    rel_type_zh: str = ""
    neighbor_name: str = ""
    neighbor_etype: str = ""
    neighbor_etype_zh: str = ""
    evidence: str = ""
    intent: str = ""


class CustomModelConfig(BaseModel):
    config_id: Optional[str] = ""
    display_name: Optional[str] = ""
    provider_name: Optional[str] = "custom"
    base_url: str
    api_key: str
    model: str
    note: Optional[str] = ""


class EmbeddingConfig(BaseModel):
    base_url: Optional[str] = DEFAULT_EMBEDDING_BASE_URL
    api_key: Optional[str] = DEFAULT_EMBEDDING_API_KEY
    model: Optional[str] = DEFAULT_EMBEDDING_MODEL
    dimensions: Optional[int] = Field(default=DEFAULT_EMBEDDING_DIM, ge=0)


class QARequest(BaseModel):
    question: str
    model_name: str = Field(default="deepseek-v3")
    enable_thinking: bool = Field(default=True)
    enable_web_search: bool = Field(default=False)
    rag_top_k: int = Field(default=RAG_TOP_K_DEFAULT, ge=1, le=20)
    kg_top_k: int = Field(default=KG_TOP_K_DEFAULT, ge=1, le=30)
    fusion_top_k: int = Field(default=FUSION_TOP_K_DEFAULT, ge=1, le=30)
    custom_model_config: Optional[CustomModelConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None


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
    rag_count: int
    kg_count: int
    fusion_count: int
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    evidence_chain: List[EvidenceItem]
    rag_evidence: List[EvidenceItem] = []
    kg_evidence: List[EvidenceItem] = []
    retrieved_context: str = ""
    rag_index_dir: str = RAG_INDEX_DIR
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_dim: Optional[int] = None

# ============================================================
# 问题路由
# ============================================================
class QuestionRouter:
    def classify(self, question: str) -> str:
        q = normalize_text(question)
        if any(x in q for x in ["英文", "英文名", "英文名称", "英文叫什么"]): return "english_name"
        if any(x in q for x in ["同义词", "别名", "另一名称", "别称"]): return "alias"
        if any(x in q for x in ["定义", "什么是", "是一种什么疾病", "是什么病", "属于什么"]): return "definition"
        if any(x in q for x in ["科室", "挂什么科", "就诊", "去哪个科", "诊疗科室", "就诊科室"]): return "department"
        if any(x in q for x in ["预防", "注意事项", "需要注意", "如何预防"]): return "precaution"
        if any(x in q for x in ["病程", "发展", "会如何变化", "如何进展"]): return "course"
        if any(x in q for x in ["不良反应", "副作用"]): return "ade"
        if any(x in q for x in ["药物", "用药", "什么药"]): return "drug"
        if any(x in q for x in ["症状", "表现", "会出现哪些", "会有什么感觉", "主要症状", "具体症状"]): return "symptom"
        if any(x in q for x in ["治疗", "怎么治", "如何治疗", "治疗方法", "主要治疗方法"]): return "treatment"
        if any(x in q for x in ["原因", "病因", "为什么会", "发生原因"]): return "reason"
        if any(x in q for x in ["发病机制", "机制"]): return "pathogenesis"
        if any(x in q for x in ["检查", "诊断", "如何诊断", "需要做哪些检查", "最重要的检查"]): return "test"
        if any(x in q for x in ["检验指标", "指标", "化验", "HbA1c", "血糖", "空腹血糖", "PPG"]): return "test_items"
        if any(x in q for x in ["并发症", "相关疾病", "有关疾病", "关联疾病"]): return "related_disease"
        if any(x in q for x in ["分类", "分型", "属于哪一类"]): return "class"
        if any(x in q for x in ["解剖部位", "部位", "发生在哪", "涉及哪些部位"]): return "anatomy"
        return "general"

    def extract_entity(self, question: str) -> Optional[str]:
        q = normalize_text(question)
        patterns = [
            r"^(.*?)的?(主要)?症状是什么", r"^(.*?)会出现哪些症状", r"^(.*?)有哪些不良反应",
            r"^(.*?)的副作用是什么", r"^(.*?)相关药物有哪些", r"^(.*?)的治疗方法有哪些",
            r"^(.*?)如何治疗", r"^(.*?)的原因是什么", r"^(.*?)的病因是什么",
            r"^(.*?)的英文名称是什么", r"^(.*?)的英文名是什么", r"^(.*?)的别名是什么",
            r"^(.*?)的同义词是什么", r"^(.*?)如何诊断", r"^(.*?)需要做哪些检查",
            r"^(.*?)最重要的检查是什么", r"^(.*?)的定义是什么", r"^什么是(.*?)$",
            r"^(.*?)是什么", r"^(.*?)有哪些并发症", r"^(.*?)相关的疾病有哪些",
            r"^(.*?)的注意事项是什么", r"^(.*?)如何预防", r"^(.*?)的病程是如何发展的",
            r"^(.*?)相关检验指标有哪些", r"^(.*?)相关发病机制有哪些", r"^(.*?)涉及哪些解剖部位",
        ]
        for p in patterns:
            m = re.match(p, q)
            if m:
                ent = normalize_text(m.group(1))
                if ent:
                    return ent
        stopwords = {"什么", "哪些", "多少", "怎么", "如何", "为什么", "是否", "属于", "有关", "相关", "主要", "具体", "一种", "这个", "那个", "会", "是", "的", "了", "吗", "英文", "名称", "定义", "症状", "治疗", "原因", "病因", "检查", "诊断", "科室", "并发症", "注意事项", "预防", "病程", "药物", "副作用", "不良反应", "检验指标", "发病机制"}
        cands = [w for w in tokenize_zh(q) if len(w) >= 2 and w not in stopwords]
        cands = uniq_keep_order(cands)
        return sorted(cands, key=len, reverse=True)[0] if cands else None

    @staticmethod
    def extract_keywords(question: str) -> List[str]:
        stopwords = {"什么", "哪些", "多少", "怎么", "如何", "为什么", "是否", "属于", "有关", "相关", "主要", "具体", "一种", "这个", "那个", "会", "是", "的", "了", "吗", "请问", "一下", "英文", "名称", "定义", "主要症状", "不良反应", "副作用", "治疗方法", "病因", "检查", "诊断"}
        return uniq_keep_order([w for w in tokenize_zh(question) if len(w) >= 2 and w not in stopwords])[:8]

# ============================================================
# RAG 检索
# ============================================================
class FaissRetriever:
    def __init__(self, index_dir: str, embedding_base_url: str, embedding_api_key: str, embedding_model: str, embedding_dim: int):
        if faiss is None:
            raise RuntimeError("未安装 faiss，请执行：pip install faiss-cpu")
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise RuntimeError(f"向量库目录不存在：{self.index_dir.resolve()}")
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.chunks = load_jsonl(self.index_dir / "chunks.jsonl")
        self.embedding_model = normalize_text(embedding_model) or normalize_text(self.meta.get("embedding_model", ""))
        self.embedding_dim = int(embedding_dim or 0) or int(self.meta.get("embedding_dim", 0))
        self.embedding_base_url = normalize_text(embedding_base_url) or DEFAULT_EMBEDDING_BASE_URL
        self.embedding_api_key = normalize_text(embedding_api_key) or DEFAULT_EMBEDDING_API_KEY
        if not self.embedding_model:
            raise RuntimeError("Embedding 模型为空，请在 meta.json 或 embedding_config.model 中配置。")
        if not self.embedding_dim:
            raise RuntimeError("Embedding 维度为空，请在 meta.json 或 embedding_config.dimensions 中配置。")
        if not self.embedding_api_key:
            raise RuntimeError("请配置 Embedding API Key。")
        self.client = OpenAI(api_key=self.embedding_api_key, base_url=self.embedding_base_url)

    def embed_query(self, question: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=[question],
            dimensions=self.embedding_dim,
            extra_body={"text_type": "query"},
        )
        arr = np.asarray([resp.data[0].embedding], dtype="float32")
        faiss.normalize_L2(arr)
        return arr

    def search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        q = self.embed_query(question)
        scores, ids = self.index.search(q, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx < 0:
                continue
            item = dict(self.chunks[int(idx)])
            item.update({
                "source_type": "rag",
                "rank": rank,
                "score": float(score),
                "title": f"{item.get('source_file', '')}#{item.get('paragraph_id', '')}",
                "text": normalize_text(item.get("text", "")),
            })
            if item["score"] >= RAG_MIN_SCORE:
                results.append(item)
        return results

# ============================================================
# KG 检索
# ============================================================
class KGQARetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.router = QuestionRouter()

    def close(self):
        self.driver.close()

    def ping(self) -> bool:
        try:
            with self.driver.session() as session:
                return session.run("RETURN 1 AS ok").single()["ok"] == 1
        except Exception:
            return False

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
        WHERE r.evidence IS NOT NULL AND ANY(p IN $prefixes WHERE r.evidence CONTAINS ($entity + p) OR r.evidence CONTAINS p)
        {cls._return_clause()}
        LIMIT $topk
        """
        return {"items": [record.data() for record in tx.run(cypher, entity=entity, prefixes=prefixes, topk=topk)], "cypher": cypher}

    @classmethod
    def _relation_query(cls, tx, entity: str, rel_type: str, topk: int):
        cypher = f"""
        MATCH (n:Entity)-[r:{rel_type}]-(m:Entity)
        WHERE n.name = $entity OR n.name CONTAINS $entity
        {cls._return_clause()}
        LIMIT $topk
        """
        return {"items": [record.data() for record in tx.run(cypher, entity=entity, topk=topk)], "cypher": cypher}

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
            "related_disease": ["的相关疾病"],
        }
        if intent in structured_fallback_prefix:
            return cls._evidence_query(tx, entity, structured_fallback_prefix[intent], topk)
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
        return {"items": [record.data() for record in tx.run(cypher, entity=entity, topk=topk)], "cypher": cypher}

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
        return {"items": [record.data() for record in tx.run(cypher, keywords=keywords, topk=topk)], "cypher": cypher}

    @staticmethod
    def _normalize_item(item: Dict[str, Any], rank: int, intent: str) -> Dict[str, Any]:
        center = normalize_text(item.get("center_name", ""))
        rel = normalize_text(item.get("rel_type_zh", "")) or normalize_text(item.get("rel_type", ""))
        neighbor = normalize_text(item.get("neighbor_name", ""))
        evidence = normalize_text(item.get("evidence", ""))
        fact = f"{center} -[{rel}]- {neighbor}" if center and rel and neighbor else center
        return {
            "source_type": "kg",
            "rank": rank,
            "score": 1.0 / max(rank, 1),
            "title": fact,
            "text": evidence or fact,
            "center_name": center,
            "center_etype": normalize_text(item.get("center_etype", "")),
            "center_etype_zh": normalize_text(item.get("center_etype_zh", "")),
            "rel_type": normalize_text(item.get("rel_type", "")),
            "rel_type_zh": normalize_text(item.get("rel_type_zh", "")),
            "neighbor_name": neighbor,
            "neighbor_etype": normalize_text(item.get("neighbor_etype", "")),
            "neighbor_etype_zh": normalize_text(item.get("neighbor_etype_zh", "")),
            "evidence": evidence,
            "intent": intent,
        }

    def retrieve(self, question: str, topk: int) -> Dict[str, Any]:
        intent = self.router.classify(question)
        entity = self.router.extract_entity(question)
        with self.driver.session() as session:
            if entity:
                result = session.execute_read(self._route_query, entity, intent, topk)
                if not result["items"]:
                    result = session.execute_read(self._query_neighbors_by_entity, entity, topk)
                if result["items"]:
                    return {
                        "entity": entity,
                        "intent": intent,
                        "items": [self._normalize_item(x, i + 1, intent) for i, x in enumerate(result["items"])],
                        "cypher": result["cypher"],
                    }
            keywords = self.router.extract_keywords(question)
            if keywords:
                result = session.execute_read(self._query_neighbors_by_keywords, keywords, topk)
                return {
                    "entity": entity,
                    "intent": intent,
                    "items": [self._normalize_item(x, i + 1, intent) for i, x in enumerate(result["items"])],
                    "cypher": result["cypher"],
                }
        return {"entity": entity, "intent": intent, "items": [], "cypher": None}

# ============================================================
# 融合与生成
# ============================================================
def reciprocal_rank_fusion(rag_docs: List[Dict[str, Any]], kg_docs: List[Dict[str, Any]], top_k: int, k: int = 60) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for docs, weight in [(rag_docs, 1.0), (kg_docs, 1.2)]:
        for d in docs:
            text = normalize_text(d.get("text", ""))
            if not text:
                continue
            key = f"{d.get('source_type')}::{text[:220]}"
            if key not in merged:
                merged[key] = dict(d)
                merged[key]["fusion_score"] = 0.0
                merged[key]["sources"] = set()
            merged[key]["fusion_score"] += weight / (k + int(d.get("rank", 999)))
            merged[key]["sources"].add(d.get("source_type", ""))
    out = sorted(merged.values(), key=lambda x: x.get("fusion_score", 0), reverse=True)
    for idx, d in enumerate(out, start=1):
        d["rank"] = idx
        d["sources"] = ",".join(sorted(x for x in d.get("sources", set()) if x))
    return out[:top_k]


class HybridGenerator:
    def __init__(self):
        self._retrievers: Dict[Tuple[str, str, int, str], FaissRetriever] = {}
        self._retriever_errors: Dict[Tuple[str, str, int, str], str] = {}

    def embedding_key(self, embedding_config: Optional[EmbeddingConfig]) -> Tuple[str, str, int, str]:
        cfg = embedding_config or EmbeddingConfig()
        return (
            normalize_text(cfg.base_url) or DEFAULT_EMBEDDING_BASE_URL,
            normalize_text(cfg.model) or DEFAULT_EMBEDDING_MODEL,
            int(cfg.dimensions or 0),
            normalize_text(cfg.api_key) or DEFAULT_EMBEDDING_API_KEY,
        )

    def retriever(self, embedding_config: Optional[EmbeddingConfig] = None) -> FaissRetriever:
        key = self.embedding_key(embedding_config)
        if key in self._retrievers:
            return self._retrievers[key]
        if key in self._retriever_errors:
            raise RuntimeError(self._retriever_errors[key])
        try:
            base_url, model, dim, api_key = key
            self._retrievers[key] = FaissRetriever(RAG_INDEX_DIR, base_url, api_key, model, dim)
            return self._retrievers[key]
        except Exception as e:
            self._retriever_errors[key] = str(e)
            raise

    def resolve_model(self, model_name: str, custom_model_config: Optional[CustomModelConfig]) -> Tuple[str, Dict[str, str]]:
        normalized = LEGACY_MODEL_ALIASES.get(normalize_text(model_name), normalize_text(model_name))
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

    def build_messages(self, question: str, intent: str, entity: Optional[str], docs: List[Dict[str, Any]], enable_thinking: bool, enable_web_search: bool) -> List[Dict[str, str]]:
        context_text = "\n\n".join(
            f"[{i + 1}] source={d.get('source_type','')} title={d.get('title','')} fusion_score={float(d.get('fusion_score', d.get('score', 0))):.4f}\n{d.get('text','')}"
            for i, d in enumerate(docs)
        )
        user = f"""请基于 KG + RAG 融合证据回答下面这个糖尿病相关问题。

问题类型：{intent}
识别实体：{entity or '未识别'}
用户问题：{question}
联网模式：{'开启' if enable_web_search else '关闭'}
深度思考：{'开启' if enable_thinking else '关闭'}

融合证据：
{context_text or '无'}

要求：
- 必须优先依据融合证据回答
- KG 证据用于事实约束，RAG 证据用于解释补充
- 如果证据不足，回答“根据现有资料无法确定”
- 中文输出，1~3 句话
- 不要输出分析过程
"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if enable_thinking:
            messages.append({"role": "system", "content": DEFAULT_THINKING_HINT})
        if enable_web_search:
            messages.append({"role": "system", "content": DEFAULT_WEB_HINT})
        messages.append({"role": "user", "content": user})
        return messages

    async def generate(self, question: str, intent: str, entity: Optional[str], docs: List[Dict[str, Any]], model_name: str, enable_thinking: bool, enable_web_search: bool, custom_model_config: Optional[CustomModelConfig]) -> Tuple[str, Dict[str, str], str]:
        resolved_name, cfg = self.resolve_model(model_name, custom_model_config)
        if not cfg.get("api_key"):
            return f"模型 {resolved_name} 未配置 API Key，无法生成回答。", cfg, resolved_name
        client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=LLM_TIMEOUT)
        request_kwargs: Dict[str, Any] = {
            "model": cfg["actual_model"],
            "messages": self.build_messages(question, intent, entity, docs, enable_thinking, enable_web_search),
        }
        param_style = cfg.get("param_style", "standard")
        if param_style == "standard":
            request_kwargs["temperature"] = 0.2
        elif param_style == "kimi_temperature_1":
            request_kwargs["temperature"] = 1
        try:
            resp = await client.chat.completions.create(**request_kwargs)
            answer = resp.choices[0].message.content if resp and resp.choices else ""
            return normalize_text(answer) or "模型未返回有效内容。", cfg, resolved_name
        except Exception as e:
            return f"模型调用失败：{e}", cfg, resolved_name


def to_evidence_item(d: Dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        source_type=normalize_text(d.get("source_type", "")),
        rank=int(d.get("rank", 0) or 0),
        score=float(d.get("score", 0.0) or 0.0),
        fusion_score=float(d.get("fusion_score", 0.0) or 0.0),
        title=normalize_text(d.get("title", "")),
        text=normalize_text(d.get("text", "")),
        sources=normalize_text(d.get("sources", "")),
        chunk_id=normalize_text(d.get("chunk_id", "")),
        source_file=normalize_text(d.get("source_file", "")),
        doc_id=normalize_text(d.get("doc_id", "")),
        paragraph_id=normalize_text(d.get("paragraph_id", "")),
        center_name=normalize_text(d.get("center_name", "")),
        center_etype=normalize_text(d.get("center_etype", "")),
        center_etype_zh=normalize_text(d.get("center_etype_zh", "")),
        rel_type=normalize_text(d.get("rel_type", "")),
        rel_type_zh=normalize_text(d.get("rel_type_zh", "")),
        neighbor_name=normalize_text(d.get("neighbor_name", "")),
        neighbor_etype=normalize_text(d.get("neighbor_etype", "")),
        neighbor_etype_zh=normalize_text(d.get("neighbor_etype_zh", "")),
        evidence=normalize_text(d.get("evidence", "")),
        intent=normalize_text(d.get("intent", "")),
    )

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="糖尿病 LLM + RAG + KG 融合问答系统", description="FAISS 检索增强 + Neo4j 知识图谱 + 多模型生成", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
router = QuestionRouter()
hybrid_generator = HybridGenerator()
kg_retriever = KGQARetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


@app.get("/")
def root():
    return {"message": "糖尿病 LLM + RAG + KG 融合问答系统后端已启动", "docs": "/docs", "health": "/health", "port": SERVER_PORT}


@app.get("/health")
def health():
    rag_status = "not_loaded"
    rag_error = None
    embedding_model = None
    try:
        r = hybrid_generator.retriever(None)
        rag_status = "ok"
        embedding_model = r.embedding_model
    except Exception as e:
        rag_status = "error"
        rag_error = str(e)
    kg_ok = kg_retriever.ping()
    return {
        "status": "ok" if rag_status == "ok" and kg_ok else "degraded",
        "rag_status": rag_status,
        "rag_error": rag_error,
        "kg_status": "ok" if kg_ok else "error",
        "neo4j_uri": NEO4J_URI,
        "rag_index_dir": RAG_INDEX_DIR,
        "default_rag_top_k": RAG_TOP_K_DEFAULT,
        "default_kg_top_k": KG_TOP_K_DEFAULT,
        "default_fusion_top_k": FUSION_TOP_K_DEFAULT,
        "embedding_model": embedding_model,
        "supported_models": list(BUILTIN_MODELS.keys()) + ["custom"],
        "builtin_models": [{"model_name": k, "display_name": v["display_name"], "provider_name": v["provider_name"], "actual_model": v["actual_model"], "base_url": v["base_url"]} for k, v in BUILTIN_MODELS.items()],
    }


@app.get("/schema")
def schema():
    return {
        "supported_models": list(BUILTIN_MODELS.keys()) + ["custom"],
        "supported_intents": ["definition", "english_name", "alias", "department", "precaution", "course", "symptom", "treatment", "reason", "pathogenesis", "test", "drug", "ade", "test_items", "class", "anatomy", "related_disease", "general"],
        "fields": ["rag_top_k", "kg_top_k", "fusion_top_k", "embedding_config", "custom_model_config"],
    }


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    question = normalize_text(req.question)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    t1 = time.perf_counter()
    intent = router.classify(question)
    entity = router.extract_entity(question)

    rag_docs: List[Dict[str, Any]] = []
    kg_docs: List[Dict[str, Any]] = []
    cypher = None
    retriever = None
    rag_error = None
    kg_error = None

    try:
        retriever = hybrid_generator.retriever(req.embedding_config)
        rag_docs = retriever.search(question=question, top_k=req.rag_top_k)
    except Exception as e:
        rag_error = str(e)

    try:
        kg_result = kg_retriever.retrieve(question, topk=req.kg_top_k)
        kg_docs = kg_result["items"]
        cypher = kg_result["cypher"]
        entity = kg_result.get("entity") or entity
        intent = kg_result.get("intent") or intent
    except Exception as e:
        kg_error = str(e)

    if not rag_docs and not kg_docs:
        detail = "；".join(x for x in [f"RAG 检索失败：{rag_error}" if rag_error else "", f"KG 检索失败：{kg_error}" if kg_error else ""] if x)
        raise HTTPException(status_code=500, detail=detail or "RAG 与 KG 均未检索到有效证据")

    fusion_docs = reciprocal_rank_fusion(rag_docs, kg_docs, top_k=req.fusion_top_k)
    t2 = time.perf_counter()

    answer, cfg, resolved_name = await hybrid_generator.generate(
        question=question,
        intent=intent,
        entity=entity,
        docs=fusion_docs,
        model_name=req.model_name,
        enable_thinking=req.enable_thinking,
        enable_web_search=req.enable_web_search,
        custom_model_config=req.custom_model_config,
    )
    t3 = time.perf_counter()

    model_name = LEGACY_MODEL_ALIASES.get(normalize_text(req.model_name), normalize_text(req.model_name))
    if model_name.startswith("custom:") or resolved_name == "custom":
        model_name = "custom"

    evidence_chain = [to_evidence_item(d) for d in fusion_docs]
    rag_evidence = [to_evidence_item(d) for d in rag_docs]
    kg_evidence = [to_evidence_item(d) for d in kg_docs]

    return QAResponse(
        question=question,
        entity=entity,
        intent=intent,
        answer=answer,
        model_name=model_name,
        provider_name=cfg.get("provider_name"),
        display_name=cfg.get("display_name"),
        base_url=cfg.get("base_url"),
        api_key_masked=mask_secret(cfg.get("api_key", "")),
        actual_model=cfg.get("actual_model"),
        custom_config_id=cfg.get("config_id", ""),
        cypher=cypher,
        retrieved_count=len(evidence_chain),
        rag_count=len(rag_docs),
        kg_count=len(kg_docs),
        fusion_count=len(evidence_chain),
        retrieval_time_sec=round(t2 - t1, 6),
        generation_time_sec=round(t3 - t2, 6),
        total_time_sec=round(t3 - t1, 6),
        evidence_chain=evidence_chain,
        rag_evidence=rag_evidence,
        kg_evidence=kg_evidence,
        retrieved_context="\n\n".join(x.text for x in evidence_chain),
        rag_index_dir=RAG_INDEX_DIR,
        embedding_model=getattr(retriever, "embedding_model", None),
        embedding_base_url=getattr(retriever, "embedding_base_url", None),
        embedding_dim=getattr(retriever, "embedding_dim", None),
    )


@app.on_event("shutdown")
def shutdown_event():
    kg_retriever.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_llm_rag_kg:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
