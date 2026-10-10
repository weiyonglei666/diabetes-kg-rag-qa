# -*- coding: utf-8 -*-
"""
测试脚本 - 智谱AI API连接测试

【API密钥配置说明】
请在 src/config/.env 文件中配置 ZHIPU_API_KEY
获取密钥地址: https://open.bigmodel.cn/
"""

import os
import time
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化客户端
# 注意：API密钥从环境变量 ZHIPU_API_KEY 读取
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def infer_one(text, max_retries=5):
    delay = 5

    for i in range(max_retries):
        try:
            print(f"开始请求，第 {i+1} 次尝试...")
            resp = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=[
                    {"role": "system", "content": "你是医疗NER助手，只返回JSON数组。"},
                    {"role": "user", "content": f"抽取句子中的Disease/Drug/Symptom实体，返回JSON数组。句子：{text}"}
                ],
                temperature=0,
            )
            return resp.choices[0].message.content

        except RateLimitError:
            print(f"触发限流，等待 {delay} 秒后重试...")
            if i == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)

def main():
    text = "患者出现发热，给予阿莫西林治疗。"
    result = infer_one(text)
    print("模型输出：")
    print(result)

if __name__ == "__main__":
    main()
