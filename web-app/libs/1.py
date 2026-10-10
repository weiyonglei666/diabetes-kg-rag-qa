
# ============================================================================
# 重要说明：API 密钥配置
# ============================================================================
# 本脚本使用 Moonshot (Kimi) API，API 密钥必须通过环境变量配置。
# 请勿将真实的 API 密钥写入代码或提交到版本控制系统！
#
# 配置方法：
# 1. 在项目根目录创建 .env 文件
# 2. 在 .env 文件中配置：MOONSHOT_API_KEY=your_moonshot_api_key
# 3. 或者在系统环境变量中配置 MOONSHOT_API_KEY
#
# 获取 API Key：https://platform.moonshot.cn/
# ============================================================================

import os
from openai import OpenAI

# 从环境变量读取 API Key，禁止硬编码
client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),  # 必须从环境变量读取
    base_url="https://api.moonshot.cn/v1",
)

resp = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "user", "content": "你好"}
    ],
)

print(resp.choices[0].message.content)


