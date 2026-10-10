import json
import re
import time
import asyncio
from collections import Counter
from typing import List, Dict, Tuple

import jieba
import pandas as pd
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from openai import AsyncOpenAI


# =========================
# 配置
# =========================
# ============================================================================
# 重要说明：API 密钥配置
# ============================================================================
# 本脚本使用多个大语言模型 API，所有 API 密钥必须通过环境变量配置。
# 请勿将真实的 API 密钥写入代码或提交到版本控制系统！
#
# 配置方法：
# 1. 在项目根目录创建 .env 文件
# 2. 在 .env 文件中配置以下环境变量：
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

EVAL_FILE = "trainlast_cleaned.jsonl"
OUTPUT_FILE = "llm_eval_async.csv"

MAX_CONCURRENCY = 10   # 并发数（建议 5~20）
SLEEP_SEC = 0.05       # 微限速

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
        "model": "glm-5.1"
    },
    "qwen-3.6": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),  # 必须从环境变量读取
        "model": "qwen-plus"
    }
}


# =========================
# 初始化 Async Client
# =========================
clients = {
    k: AsyncOpenAI(api_key=v["api_key"], base_url=v["base_url"])
    for k, v in MODELS.items()
}


# =========================
# 工具函数
# =========================
def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip())


def tokenize_zh(text):
    return [w for w in jieba.lcut(normalize_text(text)) if w.strip()]


# =========================
# 指标
# =========================
def compute_f1(pred, gold):
    p, g = tokenize_zh(pred), tokenize_zh(gold)
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
        smoothing_function=smoothie
    )


rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def compute_rouge(pred, gold):
    s = rouge.score(gold, pred)
    return s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeL"].fmeasure


# =========================
# 缓存
# =========================
cache: Dict[Tuple[str, str], str] = {}


# =========================
# 并发调用
# =========================
sem = asyncio.Semaphore(MAX_CONCURRENCY)


async def llm_call(question, model_key):
    key = (question, model_key)
    if key in cache:
        return cache[key]

    async with sem:
        try:
            client = clients[model_key]
            cfg = MODELS[model_key]

            resp = await client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": "你是中文医学糖尿病问答专家，用一句话回答，不要解释"},
                    {"role": "user", "content": question}
                ],
                temperature=0.2
            )

            answer = resp.choices[0].message.content.strip()

        except Exception as e:
            answer = f"ERROR: {e}"

        cache[key] = answer
        await asyncio.sleep(SLEEP_SEC)
        return answer


# =========================
# 单样本任务
# =========================
async def process_sample(sample):
    question = normalize_text(sample.get("question", ""))
    gold = normalize_text(sample.get("context", ""))

    tasks = []
    for model_key in MODELS.keys():
        tasks.append(llm_call(question, model_key))

    preds = await asyncio.gather(*tasks)

    rows = []

    for model_key, pred in zip(MODELS.keys(), preds):
        f1 = compute_f1(pred, gold)
        bleu = compute_bleu(pred, gold)
        r1, r2, rl = compute_rouge(pred, gold)

        rows.append({
            "model": model_key,
            "question": question,
            "pred": pred,
            "gold": gold,
            "f1": f1,
            "bleu": bleu,
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rl,
        })

    return rows


# =========================
# 主函数
# =========================
async def evaluate():
    with open(EVAL_FILE, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    all_tasks = [process_sample(s) for s in samples]

    results = []
    for future in tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks)):
        res = await future
        results.extend(res)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n===== Summary =====")
    print(df.groupby("model")[["f1", "bleu", "rougeL"]].mean())


# =========================
# 入口
# =========================
if __name__ == "__main__":
    asyncio.run(evaluate())
