import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import jieba
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import AsyncOpenAI, OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm


# ============================================================
# 配置区：所有评估参数都写这里
# ============================================================
# ============================================================================
# 重要说明：API 密钥配置
# ============================================================================
# 本脚本使用多个大语言模型 API 和 Embedding 服务，所有 API 密钥必须通过环境变量配置。
# 请勿将真实的 API 密钥写入代码或提交到版本控制系统！
#
# 配置方法：
# 1. 在项目根目录创建 .env 文件
# 2. 在 .env 文件中配置以下环境变量：
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

import os

CONFIG = {
    # 评估集：你的 trainlast_cleaned.jsonl
    "EVAL_FILE": "trainlast_cleaned.jsonl",

    # build_faiss_index_inline.py 生成的向量库目录
    "INDEX_DIR": "../diakg_faiss_index",

    # 输出结果
    "OUTPUT_FILE": "rag_eval_async.csv",

    # 检索 top-k
    "TOP_K": 5,

    # 限制评估条数；0 表示全量。建议先设 20 试跑。
    "LIMIT": 0,

    # 并发数：RAG 会调用检索 + LLM，建议先小一点
    "MAX_CONCURRENCY": 1,
    "SLEEP_SEC": 0.05,

    # 使用哪些模型评估；可选：
    "SELECTED_MODELS": [
        "deepseek-v3",
        "deepseek-r1",
        "kimi-2.5",
        "glm-5.1",
        "qwen-3.6",
    ],

    # Qwen Embedding 配置（必须从环境变量读取）
    "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY"),  # 必须从环境变量读取，禁止硬编码
    "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


# ============================================================
# LLM 配置区：API Key 必须从环境变量读取
# ============================================================
MODELS = {
    "deepseek-v3": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "model": "deepseek-chat"
    },
    "deepseek-r1": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "model": "deepseek-reasoner"
    },
    "kimi-2.5": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": os.getenv("MOONSHOT_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "model": "moonshot-v1-128k"
    },
    "glm-5.1": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.getenv("GLM_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "model": "glm-4.5-air"
    },
    "qwen-3.6": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": CONFIG["DASHSCOPE_API_KEY"],  # 从环境变量读取
        "model": "qwen-plus"
    }
}


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def tokenize_zh(text):
    return [w for w in jieba.lcut(normalize_text(text)) if w.strip()]


def compute_f1(pred, gold):
    p, g = tokenize_zh(pred), tokenize_zh(gold)
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
    return sentence_bleu(
        [tokenize_zh(gold)],
        tokenize_zh(pred),
        smoothing_function=smoothie,
    )


rouge = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
)


def compute_rouge(pred, gold):
    s = rouge.score(gold, pred)
    return (
        s["rouge1"].fmeasure,
        s["rouge2"].fmeasure,
        s["rougeL"].fmeasure,
    )


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class FaissRetriever:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.meta = json.loads(
            (self.index_dir / "meta.json").read_text(encoding="utf-8")
        )
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.chunks = load_jsonl(self.index_dir / "chunks.jsonl")

        self.embedding_model = self.meta["embedding_model"]
        self.embedding_dim = int(self.meta["embedding_dim"])

        api_key = CONFIG["DASHSCOPE_API_KEY"]
        if not api_key or api_key == "请填你的DASHSCOPE_API_KEY":
            raise RuntimeError("请先在 CONFIG['DASHSCOPE_API_KEY'] 填入真实 API Key。")

        self.client = OpenAI(
            api_key=api_key,
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
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue

            item = dict(self.chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)

        return results


def build_prompt(question: str, contexts: List[Dict]) -> List[Dict]:
    context_text = "\n\n".join(
        f"[{i + 1}] 来源={c['source_file']} doc={c['doc_id']} "
        f"paragraph={c['paragraph_id']} score={c['score']:.4f}\n{c['text']}"
        for i, c in enumerate(contexts)
    )

    system = (
        "你是中文医学糖尿病问答专家。"
        "你必须只依据给定检索资料回答。"
        "如果检索资料不足以回答，请回答：根据现有资料无法确定。"
        "回答要简洁、准确，尽量用一句话，不要编造。"
    )

    user = (
        f"问题：{question}\n\n"
        f"检索资料：\n{context_text}\n\n"
        "请给出答案："
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def make_clients() -> Dict[str, AsyncOpenAI]:
    clients = {}

    for name in CONFIG["SELECTED_MODELS"]:
        cfg = MODELS[name]
        api_key = cfg["api_key"]

        if not api_key or api_key.startswith("请填你的"):
            print(f"Warning: skip {name}, API Key 未填写。")
            continue

        clients[name] = AsyncOpenAI(
            api_key=api_key,
            base_url=cfg["base_url"],
        )

    if not clients:
        raise RuntimeError("没有可用模型。请在 MODELS 中填写至少一个模型的 API Key。")

    return clients


cache: Dict[Tuple[str, str, str], str] = {}
sem = asyncio.Semaphore(CONFIG["MAX_CONCURRENCY"])


async def llm_call(
    question: str,
    contexts: List[Dict],
    model_key: str,
    clients: Dict[str, AsyncOpenAI],
) -> str:
    context_key = "|".join(c["chunk_id"] for c in contexts)
    key = (question, model_key, context_key)

    if key in cache:
        return cache[key]

    async with sem:
        try:
            cfg = MODELS[model_key]
            resp = await clients[model_key].chat.completions.create(
                model=cfg["model"],
                messages=build_prompt(question, contexts),
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()

        except Exception as e:
            answer = f"ERROR: {e}"

        cache[key] = answer
        await asyncio.sleep(CONFIG["SLEEP_SEC"])
        return answer


def retrieval_metrics(contexts: List[Dict], gold: str) -> Dict[str, float]:
    """
    没有人工标注的“标准证据段落”时，可以用 gold answer/context
    与 retrieved_context 的重合度作为 proxy retrieval metric。
    """
    gold = normalize_text(gold)
    ctx_joined = normalize_text(" ".join(c["text"] for c in contexts))
    top1 = normalize_text(contexts[0]["text"]) if contexts else ""

    return {
        "retrieval_f1_top1": compute_f1(top1, gold),
        "retrieval_f1_topk": compute_f1(ctx_joined, gold),
        "retrieval_rougeL_topk": compute_rouge(ctx_joined, gold)[2],
    }


async def process_sample(sample, retriever, clients):
    question = normalize_text(sample.get("question", ""))

    # 兼容你的原脚本：gold 默认取 context
    gold = normalize_text(sample.get("context", "") or sample.get("answer", ""))

    contexts = retriever.search(
        question=question,
        top_k=CONFIG["TOP_K"],
    )

    ret = retrieval_metrics(contexts, gold)

    tasks = [
        llm_call(question, contexts, model_key, clients)
        for model_key in clients
    ]
    preds = await asyncio.gather(*tasks)

    retrieved_context = "\n\n".join(c["text"] for c in contexts)
    retrieved_sources = json.dumps(
        [
            {
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
                "doc_id": c["doc_id"],
                "paragraph_id": c["paragraph_id"],
                "score": c["score"],
            }
            for c in contexts
        ],
        ensure_ascii=False,
    )

    rows = []
    for model_key, pred in zip(clients.keys(), preds):
        f1 = compute_f1(pred, gold)
        bleu = compute_bleu(pred, gold)
        r1, r2, rl = compute_rouge(pred, gold)

        rows.append({
            "model": model_key,
            "question": question,
            "pred": pred,
            "gold": gold,
            "retrieved_context": retrieved_context,
            "retrieved_sources": retrieved_sources,
            "f1": f1,
            "bleu": bleu,
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rl,
            **ret,
        })

    return rows


async def evaluate():
    retriever = FaissRetriever(CONFIG["INDEX_DIR"])
    samples = load_jsonl(CONFIG["EVAL_FILE"])

    if CONFIG["LIMIT"] and CONFIG["LIMIT"] > 0:
        samples = samples[:CONFIG["LIMIT"]]

    clients = make_clients()

    results = []
    tasks = [process_sample(s, retriever, clients) for s in samples]

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        rows = await future
        results.extend(rows)

    df = pd.DataFrame(results)
    df.to_csv(
        CONFIG["OUTPUT_FILE"],
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== Generation Summary =====")
    print(df.groupby("model")[["f1", "bleu", "rougeL"]].mean())

    print("\n===== Retrieval Summary =====")
    print(df[[
        "retrieval_f1_top1",
        "retrieval_f1_topk",
        "retrieval_rougeL_topk",
    ]].mean())

    print(f"\nSaved to: {CONFIG['OUTPUT_FILE']}")


if __name__ == "__main__":
    asyncio.run(evaluate())
