#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG + KG + LLM Fusion Evaluation

Usage:
  1) Put this file in the same directory as trainlast_cleaned.jsonl.
  2) Build your FAISS index first.
  3) Start Neo4j if KG retrieval is enabled.
  4) API keys and Neo4j password are configured in plaintext inside CONFIG and MODELS.
  5) Run:
       python rag_llm_kg_fusion_eval.py

This script evaluates:
  - KG-only rule answer
  - LLM-only
  - LLM + KG
  - RAG + LLM
  - Hybrid KG + RAG + LLM

This version adds:
  - Explainability score:
      Explainability = N_e / N_k

    N_e: number of answer key facts explicitly traceable to KG paths or RAG evidence
    N_k: number of all answer key facts

  - Detailed QA result saving:
      final_qa_eval_results.csv
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import jieba
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from openai import AsyncOpenAI, OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm


CONFIG = {
    "EVAL_FILE": "trainlast_cleaned.jsonl",
    "INDEX_DIR": "../diakg_faiss_index",

    # 平均结果 Excel
    "OUTPUT_FILE": "final_average_eval_results.xlsx",

    # 逐条问答结果 CSV
    "QA_OUTPUT_CSV": "final_qa_eval_results.csv",

    "RAG_TOP_K": 5,
    "KG_TOP_K": 10,
    "FUSION_TOP_K": 8,
    "LIMIT": 0,

    "MAX_CONCURRENCY": 100,
    "SLEEP_SEC": 0.05,

    "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),  # 必须从环境变量读取，禁止硬编码

    "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY"),  # 必须从环境变量读取，禁止硬编码
    "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",

    "SELECTED_MODELS": [
        "deepseek-v3",
        "deepseek-r1",
        "kimi-2.5",
        "glm-5.1",
        "qwen-3.6",
    ],
}

# 所有 API Key 必须从环境变量读取，禁止硬编码
MODELS = {
    "deepseek-v3": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取
        "model": "deepseek-chat",
    },
    "deepseek-r1": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取
        "model": "deepseek-reasoner",
    },
    "kimi-2.5": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": os.getenv("MOONSHOT_API_KEY"),  # 必须从环境变量读取
        "model": "moonshot-v1-128k",
    },
    "glm-5.1": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.getenv("GLM_API_KEY"),  # 必须从环境变量读取
        "model": "glm-4.5-air",
    },
    "qwen-3.6": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": CONFIG["DASHSCOPE_API_KEY"],  # 从环境变量读取
        "model": "qwen-plus",
    },
}


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def tokenize_zh(text):
    return [w for w in jieba.lcut(normalize_text(text)) if w.strip()]


rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def compute_rouge_l(pred, gold):
    s = rouge.score(gold, pred)
    return s["rougeL"].fmeasure


def split_key_facts(answer: str) -> List[str]:
    """
    Split the generated answer into key fact units.

    For Chinese medical QA, answers are often short sentences or enumerations.
    This rule-based splitter keeps the metric reproducible and independent of
    another LLM evaluator.
    """
    text = normalize_text(answer)

    if not text or text.startswith("ERROR:"):
        return []

    text = re.sub(r"^(答案是|答：|答案：|根据证据|由证据可知|可知)[:：，, ]*", "", text)

    # Split by sentence-level separators and common answer-introducing structures.
    parts = re.split(r"[。；;！!？?\n]+|(?:包括|有|为|是)[:：]", text)

    facts = []
    for part in parts:
        part = normalize_text(part)
        if not part:
            continue

        # Split enumerated facts.
        sub_parts = re.split(r"[、，,]+", part)
        for sp in sub_parts:
            sp = normalize_text(sp)
            if len(sp) >= 2:
                facts.append(sp)

    return uniq_keep_order(facts)


def fact_supported_by_evidence(fact: str, evidence_texts: List[str]) -> bool:
    """
    Judge whether a key fact is traceable to KG paths or RAG evidence.

    Matching priority:
      1) exact containment;
      2) approximate Chinese token overlap.
    """
    fact = normalize_text(fact)
    if not fact:
        return False

    for ev in evidence_texts:
        ev = normalize_text(ev)
        if not ev:
            continue

        if fact in ev or ev in fact:
            return True

        fact_tokens = set(tokenize_zh(fact))
        ev_tokens = set(tokenize_zh(ev))

        fact_tokens = {t for t in fact_tokens if len(t) >= 2}
        ev_tokens = {t for t in ev_tokens if len(t) >= 2}

        if not fact_tokens:
            continue

        overlap = len(fact_tokens & ev_tokens) / len(fact_tokens)

        # A 0.6 overlap threshold is a practical default for short Chinese facts.
        if overlap >= 0.6:
            return True

    return False


def compute_explainability(pred: str, docs: List[Dict]) -> Tuple[float, int, int]:
    """
    Explainability = N_e / N_k

    N_e: number of answer key facts explicitly traceable to KG paths or RAG evidence.
    N_k: number of all key facts in the answer.
    """
    key_facts = split_key_facts(pred)
    n_k = len(key_facts)

    if n_k == 0:
        return 0.0, 0, 0

    evidence_texts = []

    for d in docs:
        # RAG original text or KG evidence text.
        if d.get("text"):
            evidence_texts.append(d.get("text", ""))

        # KG path title, e.g. disease -[symptom]- polydipsia.
        if d.get("title"):
            evidence_texts.append(d.get("title", ""))

        # KG structured path fields.
        kg_path = " ".join([
            normalize_text(d.get("center_name", "")),
            normalize_text(d.get("rel", "")),
            normalize_text(d.get("neighbor_name", "")),
        ])
        if kg_path.strip():
            evidence_texts.append(kg_path)

    n_e = sum(1 for fact in key_facts if fact_supported_by_evidence(fact, evidence_texts))

    return n_e / n_k if n_k else 0.0, n_e, n_k


def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def uniq_keep_order(items):
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


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


class FaissRetriever:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.chunks = load_jsonl(self.index_dir / "chunks.jsonl")
        self.embedding_model = self.meta["embedding_model"]
        self.embedding_dim = int(self.meta["embedding_dim"])
        if not CONFIG["DASHSCOPE_API_KEY"]:
            raise RuntimeError("DASHSCOPE_API_KEY is empty.")
        self.client = OpenAI(
            api_key=CONFIG["DASHSCOPE_API_KEY"],
            base_url=CONFIG["DASHSCOPE_BASE_URL"],
        )

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

    def search(self, question: str, top_k: int) -> List[Dict]:
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
            results.append(item)
        return results


class KGQARetriever:
    def __init__(self):
        if not CONFIG["NEO4J_PASSWORD"]:
            raise RuntimeError("NEO4J_PASSWORD is empty.")
        self.driver = GraphDatabase.driver(
            CONFIG["NEO4J_URI"],
            auth=(CONFIG["NEO4J_USER"], CONFIG["NEO4J_PASSWORD"]),
        )
        self.router = QuestionRouter()

    def close(self):
        self.driver.close()

    def extract_entity(self, question: str) -> Optional[str]:
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
        ]
        for p in patterns:
            m = re.match(p, q)
            if m:
                ent = normalize_text(m.group(1))
                if ent:
                    return ent

        stopwords = {
            "什么", "哪些", "多少", "怎么", "如何", "为什么", "是否",
            "属于", "有关", "相关", "主要", "具体", "一种", "这个",
            "那个", "会", "是", "的", "了", "吗", "英文", "名称",
            "定义", "症状", "治疗", "原因", "病因", "检查", "诊断",
            "科室", "并发症", "注意事项", "预防", "病程"
        }
        cands = [w for w in tokenize_zh(q) if len(w) >= 2 and w not in stopwords]
        cands = uniq_keep_order(cands)
        return sorted(cands, key=len, reverse=True)[0] if cands else None

    @staticmethod
    def extract_keywords(question: str) -> List[str]:
        stopwords = {
            "什么", "哪些", "多少", "怎么", "如何", "为什么", "是否",
            "属于", "有关", "相关", "主要", "具体", "一种", "这个",
            "那个", "会", "是", "的", "了", "吗", "请问", "一下",
            "英文", "名称", "定义"
        }
        return uniq_keep_order([
            w for w in tokenize_zh(question)
            if len(w) >= 2 and w not in stopwords
        ])[:8]

    def retrieve(self, question: str, top_k: int) -> List[Dict]:
        intent = self.router.classify(question)
        entity = self.extract_entity(question)
        keywords = self.extract_keywords(question)

        with self.driver.session() as session:
            rows = []
            if entity:
                rows = session.execute_read(
                    self._query_by_entity_and_intent,
                    entity,
                    intent,
                    top_k,
                )
            if not rows and keywords:
                rows = session.execute_read(
                    self._query_by_keywords,
                    keywords,
                    top_k,
                )

        docs = []
        for rank, r in enumerate(rows, start=1):
            evidence = normalize_text(r.get("evidence", ""))
            center = normalize_text(r.get("center_name", ""))
            rel = normalize_text(r.get("rel_type_zh", "") or r.get("rel_type", ""))
            neighbor = normalize_text(r.get("neighbor_name", ""))
            fact = f"{center} -[{rel}]- {neighbor}" if center and rel and neighbor else center
            text = evidence or fact
            if not text:
                continue
            docs.append({
                "source_type": "kg",
                "rank": rank,
                "score": 1.0 / rank,
                "title": fact,
                "text": text,
                "center_name": center,
                "rel": rel,
                "neighbor_name": neighbor,
                "intent": intent,
            })
        return docs

    @staticmethod
    def _query_by_entity_and_intent(tx, entity: str, intent: str, top_k: int):
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
        prefix = route_map.get(intent, "")
        cypher = """
        MATCH (n:Entity)
        WHERE n.name = $entity OR n.name CONTAINS $entity
        OPTIONAL MATCH (n)-[r]-(m)
        WITH n, r, m,
             CASE
               WHEN $prefix <> '' AND r.evidence CONTAINS ($entity + '的' + $prefix) THEN 3
               WHEN r.evidence CONTAINS $entity THEN 2
               WHEN m.name IS NOT NULL THEN 1
               ELSE 0
             END AS kg_score
        RETURN n.name AS center_name,
               n.etype_zh AS center_etype_zh,
               type(r) AS rel_type,
               r.rel_type_zh AS rel_type_zh,
               m.name AS neighbor_name,
               m.etype_zh AS neighbor_etype_zh,
               r.evidence AS evidence,
               kg_score
        ORDER BY kg_score DESC
        LIMIT $top_k
        """
        return [
            record.data()
            for record in tx.run(
                cypher,
                entity=entity,
                prefix=prefix,
                top_k=top_k,
            )
        ]

    @staticmethod
    def _query_by_keywords(tx, keywords: List[str], top_k: int):
        cypher = """
        MATCH (n:Entity)
        WITH n, [k IN $keywords WHERE n.name CONTAINS k] AS matched_keywords
        WHERE size(matched_keywords) > 0
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n.name AS center_name,
               n.etype_zh AS center_etype_zh,
               type(r) AS rel_type,
               r.rel_type_zh AS rel_type_zh,
               m.name AS neighbor_name,
               m.etype_zh AS neighbor_etype_zh,
               r.evidence AS evidence,
               size(matched_keywords) AS kg_score
        ORDER BY kg_score DESC
        LIMIT $top_k
        """
        return [
            record.data()
            for record in tx.run(
                cypher,
                keywords=keywords,
                top_k=top_k,
            )
        ]


def reciprocal_rank_fusion(
    rag_docs: List[Dict],
    kg_docs: List[Dict],
    top_k: int,
    k: int = 60,
) -> List[Dict]:
    """
    RRF avoids putting FAISS cosine scores and KG rule scores on the same absolute scale.
    """
    merged = {}
    for docs, weight in [(rag_docs, 1.0), (kg_docs, 1.2)]:
        for d in docs:
            key = normalize_text(d["text"])[:200]
            if key not in merged:
                merged[key] = dict(d)
                merged[key]["fusion_score"] = 0.0
                merged[key]["sources"] = set()
            merged[key]["fusion_score"] += weight / (k + int(d.get("rank", 999)))
            merged[key]["sources"].add(d["source_type"])

    out = sorted(merged.values(), key=lambda x: x["fusion_score"], reverse=True)
    for d in out:
        d["sources"] = ",".join(sorted(d["sources"]))
    return out[:top_k]


def build_prompt(question: str, docs: List[Dict]) -> List[Dict]:
    context_text = "\n\n".join(
        f"[{i + 1}] source={d['source_type']} "
        f"title={d.get('title', '')} "
        f"score={d.get('fusion_score', d.get('score', 0)):.4f}\n"
        f"{d['text']}"
        for i, d in enumerate(docs)
    )

    if docs:
        system = (
            "你是中文医学糖尿病问答专家。"
            "只能依据给定的 RAG 文本证据和 KG 结构化证据回答。"
            "KG 证据优先用于实体、关系、类型和明确事实；"
            "RAG 证据优先用于解释、定义和长文本补充。"
            "若两类证据冲突，请指出无法确定；不要编造。"
            "答案要简洁、准确，通常用一句话。"
        )
        user = f"问题：{question}\n\n证据：\n{context_text}\n\n请给出答案："
    else:
        system = (
            "你是中文医学糖尿病问答专家。"
            "请根据自身医学知识回答问题。"
            "答案要简洁、准确，通常用一句话。"
        )
        user = f"问题：{question}\n\n请给出答案："

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def kg_rule_answer(question: str, kg_docs: List[Dict]) -> str:
    if not kg_docs:
        return "未从知识图谱中检索到相关信息。"

    intent = QuestionRouter().classify(question)

    if intent in {
        "definition",
        "reason",
        "course",
        "precaution",
        "english_name",
        "department",
    }:
        return kg_docs[0]["text"]

    values = uniq_keep_order([
        d.get("neighbor_name", "")
        for d in kg_docs
        if d.get("neighbor_name")
    ])

    if values:
        label = {
            "symptom": "相关症状包括",
            "treatment": "相关治疗方法包括",
            "pathogenesis": "相关发病机制包括",
            "test": "相关检查包括",
            "related_disease": "相关疾病包括",
        }.get(intent, "相关信息包括")
        return f"{label}：{'、'.join(values[:10])}。"

    return kg_docs[0]["text"]


def make_clients() -> Dict[str, AsyncOpenAI]:
    clients = {}
    for name in CONFIG["SELECTED_MODELS"]:
        cfg = MODELS[name]
        if cfg["api_key"]:
            clients[name] = AsyncOpenAI(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
            )

    if not clients:
        raise RuntimeError("No LLM clients. Please check plaintext API keys in MODELS.")

    return clients


def serialize_docs(
    docs: List[Dict],
    max_docs: int = 8,
    max_text_len: int = 300,
) -> str:
    """
    将 RAG / KG 证据序列化为便于 CSV 查看的一段文本。
    """
    items = []

    for i, d in enumerate(docs[:max_docs], start=1):
        source_type = normalize_text(d.get("source_type", ""))
        title = normalize_text(d.get("title", ""))
        text = normalize_text(d.get("text", ""))
        score = d.get("fusion_score", d.get("score", ""))

        if isinstance(score, float):
            score = f"{score:.6f}"

        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."

        items.append(
            f"[{i}] source={source_type}; score={score}; title={title}; text={text}"
        )

    return "\n".join(items)


cache: Dict[Tuple[str, str, str], str] = {}
sem = asyncio.Semaphore(CONFIG["MAX_CONCURRENCY"])


async def llm_answer(
    question: str,
    docs: List[Dict],
    model_key: str,
    clients: Dict[str, AsyncOpenAI],
) -> str:
    context_key = "|".join(normalize_text(d["text"])[:80] for d in docs)
    key = (question, model_key, context_key)

    if key in cache:
        return cache[key]

    async with sem:
        try:
            cfg = MODELS[model_key]
            resp = await clients[model_key].chat.completions.create(
                model=cfg["model"],
                messages=build_prompt(question, docs),
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"ERROR: {e}"

        cache[key] = answer
        await asyncio.sleep(CONFIG["SLEEP_SEC"])
        return answer


async def evaluate_one(sample, rag_ret, kg_ret, clients):
    question = normalize_text(sample.get("question", ""))
    gold = normalize_text(sample.get("context", "") or sample.get("answer", ""))

    t0 = time.perf_counter()
    rag_docs = rag_ret.search(question, CONFIG["RAG_TOP_K"])
    t1 = time.perf_counter()

    kg_docs = kg_ret.retrieve(question, CONFIG["KG_TOP_K"])
    t2 = time.perf_counter()

    fusion_docs = reciprocal_rank_fusion(
        rag_docs,
        kg_docs,
        CONFIG["FUSION_TOP_K"],
    )

    rows = []

    # 方案一：纯知识图谱问答
    kg_pred = kg_rule_answer(question, kg_docs)
    rows.append(
        make_row(
            model="kg_rule",
            method="kg-only",
            question=question,
            gold=gold,
            pred=kg_pred,
            docs=kg_docs,
            rag_time=0.0,
            kg_time=t2 - t1,
            gen_time=0.0,
        )
    )

    for model_key in clients:
        # 方案二：纯大模型
        gen_start = time.perf_counter()
        llm_only_pred = await llm_answer(question, [], model_key, clients)
        rows.append(
            make_row(
                model=model_key,
                method="llm-only",
                question=question,
                gold=gold,
                pred=llm_only_pred,
                docs=[],
                rag_time=0.0,
                kg_time=0.0,
                gen_time=time.perf_counter() - gen_start,
            )
        )

        # 方案三：大模型 + 知识图谱
        gen_start = time.perf_counter()
        llm_kg_pred = await llm_answer(question, kg_docs, model_key, clients)
        rows.append(
            make_row(
                model=model_key,
                method="llm-kg",
                question=question,
                gold=gold,
                pred=llm_kg_pred,
                docs=kg_docs,
                rag_time=0.0,
                kg_time=t2 - t1,
                gen_time=time.perf_counter() - gen_start,
            )
        )

        # 方案四：大模型 + RAG
        gen_start = time.perf_counter()
        llm_rag_pred = await llm_answer(question, rag_docs, model_key, clients)
        rows.append(
            make_row(
                model=model_key,
                method="llm-rag",
                question=question,
                gold=gold,
                pred=llm_rag_pred,
                docs=rag_docs,
                rag_time=t1 - t0,
                kg_time=0.0,
                gen_time=time.perf_counter() - gen_start,
            )
        )

        # 方案五：大模型 + RAG + 知识图谱
        gen_start = time.perf_counter()
        llm_rag_kg_pred = await llm_answer(question, fusion_docs, model_key, clients)
        rows.append(
            make_row(
                model=model_key,
                method="llm-rag-kg",
                question=question,
                gold=gold,
                pred=llm_rag_kg_pred,
                docs=fusion_docs,
                rag_time=t1 - t0,
                kg_time=t2 - t1,
                gen_time=time.perf_counter() - gen_start,
            )
        )

    return rows


def make_row(
    model,
    method,
    question,
    gold,
    pred,
    docs,
    rag_time,
    kg_time,
    gen_time,
):
    rouge_l = compute_rouge_l(pred, gold)
    explainability, explainable_fact_count, total_fact_count = compute_explainability(
        pred,
        docs,
    )
    response_time = rag_time + kg_time + gen_time

    return {
        "method": method,
        "model": model,

        # 逐条问答结果
        "question": question,
        "gold_answer": gold,
        "pred_answer": pred,
        "evidence": serialize_docs(docs),

        # 评价指标
        "rougeL": rouge_l,
        "explainability": explainability,
        "explainable_fact_count": explainable_fact_count,
        "total_fact_count": total_fact_count,
        "response_time_sec": response_time,
        "rag_time_sec": rag_time,
        "kg_time_sec": kg_time,
        "generation_time_sec": gen_time,
        "sample_count": 1,
    }


async def main():
    samples = load_jsonl(CONFIG["EVAL_FILE"])

    if CONFIG["LIMIT"] and CONFIG["LIMIT"] > 0:
        samples = samples[:CONFIG["LIMIT"]]

    rag_ret = FaissRetriever(CONFIG["INDEX_DIR"])
    kg_ret = KGQARetriever()
    clients = make_clients()

    all_rows = []

    try:
        tasks = [
            evaluate_one(s, rag_ret, kg_ret, clients)
            for s in samples
        ]

        for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Evaluating",
        ):
            all_rows.extend(await fut)

    finally:
        kg_ret.close()

    df = pd.DataFrame(all_rows)

    qa_columns = [
        "method",
        "model",
        "question",
        "gold_answer",
        "pred_answer",
        "evidence",
        "rougeL",
        "explainability",
        "explainable_fact_count",
        "total_fact_count",
        "response_time_sec",
        "rag_time_sec",
        "kg_time_sec",
        "generation_time_sec",
    ]

    # 保存逐条问答结果 CSV
    df[qa_columns].to_csv(
        CONFIG["QA_OUTPUT_CSV"],
        index=False,
        encoding="utf-8-sig",
    )

    avg_by_method_model = (
        df.groupby(["method", "model"], as_index=False)
        .agg(
            rougeL=("rougeL", "mean"),
            explainability=("explainability", "mean"),
            explainable_fact_count=("explainable_fact_count", "sum"),
            total_fact_count=("total_fact_count", "sum"),
            response_time_sec=("response_time_sec", "mean"),
            rag_time_sec=("rag_time_sec", "mean"),
            kg_time_sec=("kg_time_sec", "mean"),
            generation_time_sec=("generation_time_sec", "mean"),
            sample_count=("sample_count", "sum"),
        )
        .sort_values(["method", "model"])
    )

    avg_by_method = (
        df.groupby(["method"], as_index=False)
        .agg(
            rougeL=("rougeL", "mean"),
            explainability=("explainability", "mean"),
            explainable_fact_count=("explainable_fact_count", "sum"),
            total_fact_count=("total_fact_count", "sum"),
            response_time_sec=("response_time_sec", "mean"),
            rag_time_sec=("rag_time_sec", "mean"),
            kg_time_sec=("kg_time_sec", "mean"),
            generation_time_sec=("generation_time_sec", "mean"),
            sample_count=("sample_count", "sum"),
        )
        .sort_values("method")
    )

    readme = pd.DataFrame([
        {
            "item": "说明",
            "value": "本文件保存最终平均评估结果，并额外保存逐条问答结果。逐条问答结果也会单独导出为 CSV。",
        },
        {
            "item": "逐条问答结果文件",
            "value": CONFIG["QA_OUTPUT_CSV"],
        },
        {
            "item": "已删除指标",
            "value": "F1、BLEU、ROUGE-1、ROUGE-2、推理路径Jaccard、幻觉率。",
        },
        {
            "item": "保留指标",
            "value": "ROUGE-L、平均响应时间、可解释性得分。BERTScore建议使用单独程序补算后合并。",
        },
        {
            "item": "可解释性得分",
            "value": "Explainability = N_e / N_k，其中 N_e 表示能够明确对应知识图谱路径或原文证据的关键事实数量，N_k 表示答案中的全部关键事实数量。",
        },
        {
            "item": "可解释性计算方式",
            "value": "系统将生成答案切分为关键事实单元，并判断每个事实是否能由 KG 路径、KG 结构化字段或 RAG 原文片段支撑。",
        },
        {
            "item": "对比方案",
            "value": "kg-only、llm-only、llm-kg、llm-rag、llm-rag-kg。",
        },
        {
            "item": "QA_Details字段说明",
            "value": "question为问题，gold_answer为标准答案或原始context，pred_answer为模型生成答案，evidence为该方法使用的RAG/KG证据。",
        },
    ])

    with pd.ExcelWriter(CONFIG["OUTPUT_FILE"], engine="openpyxl") as writer:
        avg_by_method_model.to_excel(
            writer,
            sheet_name="Average_By_Method_Model",
            index=False,
        )
        avg_by_method.to_excel(
            writer,
            sheet_name="Average_By_Method",
            index=False,
        )
        df[qa_columns].to_excel(
            writer,
            sheet_name="QA_Details",
            index=False,
        )
        readme.to_excel(
            writer,
            sheet_name="Readme",
            index=False,
        )

    print()
    print("===== Average By Method + Model =====")
    print(avg_by_method_model)
    print()
    print("===== Average By Method =====")
    print(avg_by_method)
    print()
    print(f"Saved final average results to: {CONFIG['OUTPUT_FILE']}")
    print(f"Saved detailed QA results to: {CONFIG['QA_OUTPUT_CSV']}")


if __name__ == "__main__":
    asyncio.run(main())
