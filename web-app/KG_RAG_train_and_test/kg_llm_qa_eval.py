import json
import re
import time
import asyncio
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any

import jieba
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from openai import AsyncOpenAI


# =========================
# 配置
# =========================
# ============================================================================
# 重要说明：API 密钥和数据库密码配置
# ============================================================================
# 本脚本使用 Neo4j 图数据库和多个大语言模型 API，所有敏感信息必须通过环境变量配置。
# 请勿将真实的 API 密钥、数据库密码写入代码或提交到版本控制系统！
#
# 配置方法：
# 1. 在项目根目录创建 .env 文件
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

import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # 必须从环境变量读取，禁止硬编码

EVAL_FILE = "trainlast_cleaned.jsonl"
OUTPUT_FILE = "kg_llm_eval_results.csv"
TOPK = 10

MAX_CONCURRENCY = 10
SLEEP_SEC = 0.05

# 所有 API Key 必须从环境变量读取，禁止硬编码
MODELS = {
    "deepseek-v3": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取
        "model": "deepseek-chat"
    },
    "deepseek-r1": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取
        "model": "deepseek-reasoner"
    },
    "kimi-2.5": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": os.getenv("MOONSHOT_API_KEY"),  # 必须从环境变量读取
        "model": "moonshot-v1-128k"
    },
    "glm-5.1": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.getenv("GLM_API_KEY"),  # 必须从环境变量读取
        "model": "glm-4.5-air"
    },
    "qwen-3.6": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),  # 必须从环境变量读取
        "model": "qwen-plus"
    }
}

REFUSAL_PHRASES = [
    "根据当前知识图谱未检索到充分证据",
    "未检索到充分证据",
    "证据不足",
    "无法根据当前知识图谱",
    "未找到足够信息",
]

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
    text = re.sub(r"\s+", " ", text)
    return text


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


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


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


def contains_refusal(pred: str) -> bool:
    pred_norm = normalize_text(pred)
    return any(p in pred_norm for p in REFUSAL_PHRASES)


# =========================
# 指标
# =========================
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
    return 2 * precision * recall / (precision + recall)


_bleu_smoother = SmoothingFunction().method1
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def compute_bleu(pred: str, gold: str) -> float:
    pred_tokens = tokenize_zh(pred)
    gold_tokens = tokenize_zh(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    return sentence_bleu([gold_tokens], pred_tokens, smoothing_function=_bleu_smoother)


def compute_rouge(pred: str, gold: str) -> Tuple[float, float, float]:
    s = _rouge.score(gold, pred)
    return s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeL"].fmeasure


def compute_em(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def compute_groundedness(pred: str, kg_context: str) -> float:
    pred_tokens = [t for t in tokenize_zh(pred) if len(t) >= 2]
    context_tokens = set(t for t in tokenize_zh(kg_context) if len(t) >= 2)
    if not pred_tokens:
        return 0.0
    hit = sum(1 for t in pred_tokens if t in context_tokens)
    return hit / len(pred_tokens)


# =========================
# 问题路由器
# =========================
class QuestionRouter:
    def classify(self, question: str) -> str:
        q = normalize_text(question)

        if any(x in q for x in ["英文", "英文名", "英文名称", "英文叫什么"]):
            return "english_name"
        if any(x in q for x in ["定义", "什么是", "是一种什么疾病", "是什么病"]):
            return "definition"
        if any(x in q for x in ["属于什么", "属于哪类", "属于哪一种", "类别", "分型", "类型"]):
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


# =========================
# Neo4j 检索器
# =========================
class KGQARetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.router = QuestionRouter()

    def close(self):
        self.driver.close()

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
                    stage = stage or "entity_exact"

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
                if entity and entity in center:
                    score += 2.2
                if entity and entity in evidence:
                    score += 2.0
                if entity and entity in neighbor:
                    score += 1.5
                score += 1.5 * jaccard_char_similarity(entity, center)
                score += 0.8 * jaccard_char_similarity(entity, neighbor)

            score += 2.0 * overlap_score(keywords, combined_text)
            score += 1.0 * overlap_score(intent_clues, combined_text)

            if rel:
                if any(clue in rel for clue in intent_clues):
                    score += 1.2
            if evidence and len(evidence) > 10:
                score += 0.3
            if center and len(center) >= 3:
                score += 0.2

            # 过滤明显不相关结果：既没有实体强匹配，也几乎没有关键词重合
            kw_overlap = overlap_score(keywords, combined_text)
            if entity:
                if not strong_entity_match and entity not in center and entity not in evidence and jaccard_char_similarity(entity, center) < 0.35 and kw_overlap < 0.34:
                    continue
            else:
                if kw_overlap < 0.34:
                    continue

            if intent != "general" and not intent_clues:
                pass

            item2 = dict(item)
            item2["score"] = round(score, 4)
            scored.append(item2)

        scored.sort(key=lambda x: (x.get("score", 0.0), len(normalize_text(x.get("evidence", "")))), reverse=True)
        if not scored:
            return []

        # 如果高分证据很少，减少噪声，不硬凑满 topk
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
# 多模型 LLM 服务
# =========================
class MultiLLMService:
    def __init__(self, model_keys: Optional[List[str]] = None, max_concurrency: int = 10, sleep_sec: float = 0.05):
        self.model_keys = model_keys or list(MODELS.keys())
        unsupported = [k for k in self.model_keys if k not in MODELS]
        if unsupported:
            raise ValueError(f"Unsupported model keys: {unsupported}. Available: {list(MODELS.keys())}")

        self.clients = {
            model_key: AsyncOpenAI(api_key=MODELS[model_key]["api_key"], base_url=MODELS[model_key]["base_url"])
            for model_key in self.model_keys
        }
        self.semaphores = {model_key: asyncio.Semaphore(max_concurrency) for model_key in self.model_keys}
        self.sleep_sec = sleep_sec
        self.cache: Dict[Tuple[str, str, str], str] = {}

    async def answer_with_kg(self, model_key: str, question: str, kg_context: str) -> str:
        cache_key = (model_key, question, kg_context)
        if cache_key in self.cache:
            return self.cache[cache_key]

        cfg = MODELS[model_key]
        client = self.clients[model_key]
        sem = self.semaphores[model_key]

        system_prompt = (
            "你是中文医学问答助手。"
            "请以给定的知识图谱证据为主要依据作答，不要编造证据中没有明确支持的事实。"
            "如果证据与问题部分相关，先回答能确定的部分，再简短说明证据的局限。"
            "只有在完全没有直接相关信息时，才回答：根据当前知识图谱未检索到充分证据。"
            "回答尽量简洁、像正常问答，不要机械复述规则。"
        )
        user_prompt = (
            f"{kg_context}\n\n"
            f"请回答这个问题：{question}\n"
            "输出要求：\n"
            "1. 优先给出明确结论。\n"
            "2. 不要照抄整段证据。\n"
            "3. 若只能部分回答，先回答已确定部分，再补一句“图谱中未见更直接证据”。\n"
            "4. 只有完全无关时，才输出固定句：根据当前知识图谱未检索到充分证据。"
        )

        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=cfg["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                answer = normalize_text(resp.choices[0].message.content)
            except Exception as e:
                answer = f"ERROR: {e}"

            self.cache[cache_key] = answer
            if self.sleep_sec > 0:
                await asyncio.sleep(self.sleep_sec)
            return answer

    async def answer_all_models(self, question: str, kg_context: str) -> Dict[str, str]:
        tasks = [self.answer_with_kg(model_key, question, kg_context) for model_key in self.model_keys]
        answers = await asyncio.gather(*tasks)
        return {model_key: answer for model_key, answer in zip(self.model_keys, answers)}


# =========================
# 失败类型
# =========================
def classify_failure_type(pred: str, retrieved: List[Dict[str, Any]], debug: Dict[str, Any]) -> str:
    pred_norm = normalize_text(pred)
    if pred_norm.startswith("ERROR:"):
        return "llm_error"
    if not retrieved:
        return "no_retrieval"
    if contains_refusal(pred_norm):
        if debug.get("raw_retrieved_count", 0) > 0:
            return "wrong_or_weak_retrieval"
        return "no_retrieval"
    if len(pred_norm) <= 8:
        return "too_short_answer"
    return "answered"


# =========================
# 单样本评测：一次检索，五模型生成
# =========================
async def evaluate_sample(
    sample: Dict[str, Any],
    retriever: KGQARetriever,
    llm_service: MultiLLMService,
    topk: int,
) -> List[Dict[str, Any]]:
    question = normalize_text(sample.get("question", ""))
    gold = normalize_text(sample.get("context", ""))

    t1 = time.perf_counter()
    retrieved, debug = retriever.retrieve(question, topk=topk)
    t2 = time.perf_counter()

    entity = debug.get("entity", "")
    intent = debug.get("intent", retriever.router.classify(question))
    kg_context = serialize_kg_context(question, entity, intent, retrieved, max_items=topk)
    model_answers = await llm_service.answer_all_models(question, kg_context)
    t3 = time.perf_counter()

    rows: List[Dict[str, Any]] = []
    generation_time_total = t3 - t2
    total_time = t3 - t1
    per_model_generation_time = generation_time_total / max(len(model_answers), 1)

    for model_key, pred in model_answers.items():
        rouge1, rouge2, rougeL = compute_rouge(pred, gold)
        failure_type = classify_failure_type(pred, retrieved, debug)
        rows.append({
            "model": model_key,
            "question": question,
            "intent": intent,
            "entity": entity,
            "keywords": " | ".join(debug.get("keywords", [])),
            "gold_answer": gold,
            "pred_answer": pred,
            "kg_context": kg_context,
            "retrieved_count": len(retrieved),
            "raw_retrieved_count": debug.get("raw_retrieved_count", 0),
            "retrieval_stage": debug.get("retrieval_stage", "none"),
            "retrieval_relevant_score": debug.get("relevant_score", 0.0),
            "failure_type": failure_type,
            "is_refusal": float(contains_refusal(pred)),
            "retrieval_time_sec": t2 - t1,
            "generation_time_sec": per_model_generation_time,
            "generation_time_total_sec": generation_time_total,
            "total_time_sec": total_time,
            "f1": compute_f1(pred, gold),
            "bleu": compute_bleu(pred, gold),
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "em": compute_em(pred, gold),
            "groundedness": compute_groundedness(pred, kg_context),
        })
    return rows


# =========================
# 主流程：默认全量 + 五模型同时跑
# =========================
async def evaluate() -> None:
    samples = load_jsonl(EVAL_FILE)

    retriever = KGQARetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    llm_service = MultiLLMService(model_keys=list(MODELS.keys()), max_concurrency=MAX_CONCURRENCY, sleep_sec=SLEEP_SEC)

    results: List[Dict[str, Any]] = []

    try:
        tasks = [evaluate_sample(sample, retriever, llm_service, TOPK) for sample in samples]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="KG-LLM 5 Models Evaluating"):
            rows = await future
            results.extend(rows)
    finally:
        retriever.close()

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n===== KG-LLM 5 Models Summary (Reduced Empty Answers) =====")
    print(f"样本数: {len(samples)}")
    print(f"模型数: {len(MODELS)}")
    print(f"结果文件: {OUTPUT_FILE}")

    print("\n===== 各模型平均指标 =====")
    print(df.groupby("model")[[
        "f1", "bleu", "rouge1", "rouge2", "rougeL", "em", "groundedness",
        "is_refusal", "retrieval_relevant_score",
        "retrieval_time_sec", "generation_time_sec", "generation_time_total_sec", "total_time_sec"
    ]].mean().sort_values(by="f1", ascending=False))

    print("\n===== 各模型 x 失败类型计数 =====")
    print(df.groupby(["model", "failure_type"]).size().sort_values(ascending=False))

    print("\n===== 各模型 x 问题类型平均 F1 =====")
    print(df.groupby(["model", "intent"])["f1"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    asyncio.run(evaluate())
