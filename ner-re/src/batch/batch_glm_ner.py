# -*- coding: utf-8 -*-
"""
批量NER处理脚本 - 使用智谱AI API

【API密钥配置说明】
请在 src/config/.env 文件中配置 ZHIPU_API_KEY
获取密钥地址: https://open.bigmodel.cn/
"""

import os
import json
import time
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

DATA_DIR = "../diakg_data/"
OUTPUT_FILE = "glm_ner_results.jsonl"

ENTITY_TYPES = [
    "Disease", "Drug", "Test_items", "Anatomy", "Test_Value",
    "Class", "ADE", "Treatment", "Test", "Symptom",
    "Method", "Amount", "Level", "Pathogenesis", "Reason",
    "Frequency", "Operation", "Duration"
]

# 初始化客户端
# 注意：API密钥从环境变量 ZHIPU_API_KEY 读取
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


def looks_like_title(text):
    text = text.strip()
    if len(text) <= 25:
        return True
    bad_keywords = ["专家共识", "指南", "研究", "综述", "分会", "中华医学会"]
    return any(k in text for k in bad_keywords)


def load_sentences(data_dir, max_samples=5, max_len=120):
    samples = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    for filename in files:
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = data if isinstance(data, list) else data.get("data", [])
        if not docs and isinstance(data, dict):
            docs = [data]

        for doc in docs:
            for para in doc.get("paragraphs", []):
                for sent in para.get("sentences", []):
                    text = sent.get("sentence", "").strip()
                    if not text:
                        continue

                    # 先过滤标题句和超长句
                    if looks_like_title(text):
                        continue
                    if len(text) > max_len:
                        continue

                    gold = sent.get("entities", [])
                    gold_simple = [
                        {
                            "text": e.get("entity", ""),
                            "type": e.get("entity_type", "")
                        }
                        for e in gold
                    ]

                    samples.append({
                        "text": text,
                        "gold": gold_simple
                    })

                    if len(samples) >= max_samples:
                        return samples

    return samples


def build_messages(text):
    type_text = ",".join(ENTITY_TYPES)
    return [
        {
            "role": "system",
            "content": "你是医疗NER助手，只返回JSON数组，不要解释，不要markdown代码块。"
        },
        {
            "role": "user",
            "content": (
                f"从句子中抽取实体，类型只能是[{type_text}]。"
                f"返回格式：[{{\"text\":\"实体\",\"type\":\"类型\"}}]。"
                f"没有实体时返回[]。"
                f"句子：{text}"
            )
        }
    ]


def parse_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data, "ok"
        return None, "non_list_json"
    except Exception:
        l = text.find("[")
        r = text.rfind("]")
        if l != -1 and r != -1 and l < r:
            try:
                data = json.loads(text[l:r+1])
                if isinstance(data, list):
                    return data, "ok"
            except Exception:
                pass
        return None, "invalid_json"


def infer_one(text, max_retries=6):
    delay = 15
    messages = build_messages(text)

    for i in range(max_retries):
        try:
            print(f"开始请求，第 {i+1}/{max_retries} 次")
            resp = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=messages,
                temperature=0,
                max_tokens=256,
            )
            content = resp.choices[0].message.content
            parsed, parse_status = parse_json(content)

            if parsed is None:
                return {
                    "status": "parse_error",
                    "pred": None,
                    "raw_output": content,
                    "error": parse_status
                }

            return {
                "status": "ok",
                "pred": parsed,
                "raw_output": content,
                "error": None
            }

        except RateLimitError:
            print(f"限流，第 {i+1} 次失败，等待 {delay} 秒")
            if i == max_retries - 1:
                return {
                    "status": "rate_limited",
                    "pred": None,
                    "raw_output": None,
                    "error": "RATE_LIMIT_ERROR"
                }
            time.sleep(delay)
            delay = min(delay * 2, 120)

        except Exception as e:
            return {
                "status": "error",
                "pred": None,
                "raw_output": None,
                "error": str(e)
            }


def main():
    samples = load_sentences(DATA_DIR, max_samples=5, max_len=120)
    print(f"加载到 {len(samples)} 条句子")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples, 1):
            text = sample["text"]
            gold = sample["gold"]

            print(f"\n[{idx}/{len(samples)}] 处理中...")
            print("句子：", text)

            result = infer_one(text)

            item = {
                "id": idx,
                "text": text,
                "gold": gold,
                "status": result["status"],
                "pred": result["pred"],
                "raw_output": result["raw_output"],
                "error": result["error"]
            }

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

            print("状态：", result["status"])
            if result["status"] == "ok":
                print("预测：", result["pred"])
            else:
                print("错误：", result["error"])

            print("本条结束，等待 60 秒再处理下一条...")
            time.sleep(60)

    print(f"\n完成，结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
