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
  - RAG+LLM
  - Hybrid KG+RAG+LLM
"""

import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import jieba
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import AsyncOpenAI, OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm


CONFIG = {
    "EVAL_FILE": "trainlast_cleaned.jsonl",
    "INDEX_DIR": "../diakg_faiss_index",
    "OUTPUT_FILE": "rag_llm_kg_qa_eval.csv",

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


def compute_f1(pred, gold):
    p, g = tokenize_zh(pred), tokenize_zh(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num = sum(common.values())
    if num == 0:
        return 0.0
    precision = num / len(p)
    recall = num / len(g)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(pred, gold):
    smoothie = SmoothingFunction().method1
    return sentence_bleu([tokenize_zh(gold)], tokenize_zh(pred), smoothing_function=smoothie)


rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def compute_rouge(pred, gold):
    s = rouge.score(gold, pred)
    return s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeL"].fmeasure


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
        self.client = OpenAI(api_key=CONFIG["DASHSCOPE_API_KEY"], base_url=CONFIG["DASHSCOPE_BASE_URL"])

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

        stopwords = {"什么", "哪些", "多少", "怎么", "如何", "为什么", "是否", "属于", "有关", "相关", "主要", "具体", "一种", "这个", "那个", "会", "是", "的", "了", "吗", "英文", "名称", "定义", "症状", "治疗", "原因", "病因", "检查", "诊断", "科室", "并发症", "注意事项", "预防", "病程"}
        cands = [w for w in tokenize_zh(q) if len(w) >= 2 and w not in stopwords]
        cands = uniq_keep_order(cands)
        return sorted(cands, key=len, reverse=True)[0] if cands else None

    @staticmethod
    def extract_keywords(question: str) -> List[str]:
        stopwords = {"什么", "哪些", "多少", "怎么", "如何", "为什么", "是否", "属于", "有关", "相关", "主要", "具体", "一种", "这个", "那个", "会", "是", "的", "了", "吗", "请问", "一下", "英文", "名称", "定义"}
        return uniq_keep_order([w for w in tokenize_zh(question) if len(w) >= 2 and w not in stopwords])[:8]

    def retrieve(self, question: str, top_k: int) -> List[Dict]:
        intent = self.router.classify(question)
        entity = self.extract_entity(question)
        keywords = self.extract_keywords(question)

        with self.driver.session() as session:
            rows = []
            if entity:
                rows = session.execute_read(self._query_by_entity_and_intent, entity, intent, top_k)
            if not rows and keywords:
                rows = session.execute_read(self._query_by_keywords, keywords, top_k)

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
        return [record.data() for record in tx.run(cypher, entity=entity, prefix=prefix, top_k=top_k)]

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
        return [record.data() for record in tx.run(cypher, keywords=keywords, top_k=top_k)]


def reciprocal_rank_fusion(rag_docs: List[Dict], kg_docs: List[Dict], top_k: int, k: int = 60) -> List[Dict]:
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
        f"[{i+1}] source={d['source_type']} title={d.get('title','')} score={d.get('fusion_score', d.get('score', 0)):.4f}\n{d['text']}"
        for i, d in enumerate(docs)
    )
    system = (
        "你是中文医学糖尿病问答专家。"
        "只能依据给定的 RAG 文本证据和 KG 结构化证据回答。"
        "KG 证据优先用于实体、关系、类型和明确事实；RAG 证据优先用于解释、定义和长文本补充。"
        "若两类证据冲突，请指出无法确定；不要编造。"
        "答案要简洁、准确，通常用一句话。"
    )
    user = f"问题：{question}\n\n证据：\n{context_text}\n\n请给出答案："
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def kg_rule_answer(question: str, kg_docs: List[Dict]) -> str:
    if not kg_docs:
        return "未从知识图谱中检索到相关信息。"
    intent = QuestionRouter().classify(question)
    if intent in {"definition", "reason", "course", "precaution", "english_name", "department"}:
        return kg_docs[0]["text"]
    values = uniq_keep_order([d.get("neighbor_name", "") for d in kg_docs if d.get("neighbor_name")])
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
            clients[name] = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    if not clients:
        raise RuntimeError("No LLM clients. Please check plaintext API keys in MODELS.")
    return clients


cache: Dict[Tuple[str, str, str], str] = {}
sem = asyncio.Semaphore(CONFIG["MAX_CONCURRENCY"])


async def llm_answer(question: str, docs: List[Dict], model_key: str, clients: Dict[str, AsyncOpenAI]) -> str:
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


def retrieval_proxy_metrics(docs: List[Dict], gold: str) -> Dict[str, float]:
    joined = normalize_text(" ".join(d["text"] for d in docs))
    top1 = normalize_text(docs[0]["text"]) if docs else ""
    return {
        "ret_f1_top1": compute_f1(top1, gold),
        "ret_f1_topk": compute_f1(joined, gold),
        "ret_rougeL_topk": compute_rouge(joined, gold)[2],
    }


async def evaluate_one(sample, rag_ret, kg_ret, clients):
    question = normalize_text(sample.get("question", ""))
    gold = normalize_text(sample.get("context", "") or sample.get("answer", ""))

    t0 = time.perf_counter()
    rag_docs = rag_ret.search(question, CONFIG["RAG_TOP_K"])
    t1 = time.perf_counter()
    kg_docs = kg_ret.retrieve(question, CONFIG["KG_TOP_K"])
    t2 = time.perf_counter()

    fusion_docs = reciprocal_rank_fusion(rag_docs, kg_docs, CONFIG["FUSION_TOP_K"])

    rows = []

    kg_pred = kg_rule_answer(question, kg_docs)
    rows.append(make_row("kg_rule", "kg-only", question, gold, kg_pred, kg_docs, t1 - t0, t2 - t1, 0.0))

    for model_key in clients:
        rag_pred = await llm_answer(question, rag_docs, model_key, clients)
        rows.append(make_row(model_key, "rag-llm", question, gold, rag_pred, rag_docs, t1 - t0, 0.0, 0.0))

        fusion_pred = await llm_answer(question, fusion_docs, model_key, clients)
        rows.append(make_row(model_key, "kg-rag-llm-fusion", question, gold, fusion_pred, fusion_docs, t1 - t0, t2 - t1, 0.0))

    return rows


def make_row(model, method, question, gold, pred, docs, rag_time, kg_time, gen_time):
    r1, r2, rl = compute_rouge(pred, gold)
    row = {
        "method": method,
        "model": model,
        "question": question,
        "pred": pred,
        "gold": gold,
        "f1": compute_f1(pred, gold),
        "bleu": compute_bleu(pred, gold),
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rl,
        "rag_time_sec": rag_time,
        "kg_time_sec": kg_time,
        "generation_time_sec": gen_time,
        "evidence_count": len(docs),
        "evidence_sources": ",".join(sorted(set(d.get("source_type", "") for d in docs))),
        "evidence_text": "\n\n".join(d["text"] for d in docs),
        "evidence_meta": json.dumps([
            {
                "source_type": d.get("source_type"),
                "title": d.get("title"),
                "score": d.get("score"),
                "fusion_score": d.get("fusion_score"),
                "rank": d.get("rank"),
            }
            for d in docs
        ], ensure_ascii=False),
    }
    row.update(retrieval_proxy_metrics(docs, gold))
    return row


async def main():
    samples = load_jsonl(CONFIG["EVAL_FILE"])
    if CONFIG["LIMIT"] and CONFIG["LIMIT"] > 0:
        samples = samples[:CONFIG["LIMIT"]]

    rag_ret = FaissRetriever(CONFIG["INDEX_DIR"])
    kg_ret = KGQARetriever()
    clients = make_clients()

    all_rows = []
    try:
        tasks = [evaluate_one(s, rag_ret, kg_ret, clients) for s in samples]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating fusion"):
            all_rows.extend(await fut)
    finally:
        kg_ret.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(CONFIG["OUTPUT_FILE"], index=False, encoding="utf-8-sig")

    print("\n===== Answer Quality Summary =====")
    print(df.groupby(["method", "model"])[["f1", "bleu", "rougeL"]].mean().sort_values("f1", ascending=False))

    print("\n===== Retrieval Proxy Summary =====")
    print(df.groupby(["method"])[["ret_f1_top1", "ret_f1_topk", "ret_rougeL_topk"]].mean())

    print(f"\nSaved to: {CONFIG['OUTPUT_FILE']}")


if __name__ == "__main__":
    asyncio.run(main())
