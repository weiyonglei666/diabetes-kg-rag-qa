# -*- coding: utf-8 -*-
"""
[配置] 模型路径与参数

【API密钥配置说明】
================================================================================
本项目的API密钥统一在 .env 文件中配置，请按以下步骤操作：

1. 打开 src/config/.env 文件
2. 将对应的 YOUR_XXX_API_KEY_HERE 替换为您的实际API密钥
3. 获取API密钥的地址：
   - 智谱AI: https://open.bigmodel.cn/
   - DeepSeek: https://platform.deepseek.com/
   - Moonshot (Kimi): https://platform.moonshot.cn/
   - 百川智能: https://platform.baichuan-ai.com/

【安全提示】
- 请勿将真实API密钥提交到代码仓库
- .env 文件已添加到 .gitignore，不会被git追踪
- 本地模型无需API密钥，使用占位符即可
================================================================================
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 路径配置 =================
DATA_DIR = '../diakg_data/'
OUTPUT_DIR = '../output/'

# ================= 本地模型选择 =================
NER_MODEL_NAME = "hfl/chinese-macbert-large"
RE_MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"

# ================= LLM 配置 =================
# 远程 + 本地统一管理
LLM_PROVIDERS = [
    ("zhipu", "智谱AI"),
    ("deepseek_v32", "DeepSeek V3.2"),
    ("deepseek_r1", "DeepSeek R1"),
    ("kimi_k25", "Kimi K2.5"),
    ("qwen3_8b_local", "Qwen3-8B Local"),
    ("glm4_9b_local", "GLM4-9B Local"),
    ("llama3_8b_local", "Llama-3-8B Local"),
]

# 选择索引
# 0=zhipu
# 1=deepseek_v32
# 2=deepseek_r1
# 3=kimi_k25
# 4=qwen3_8b_local
# 5=glm4_9b_local
# 6=llama3_8b_local
SELECTED_INDEX = 4
LLM_PROVIDER = LLM_PROVIDERS[SELECTED_INDEX][0]

print("可用模型：")
for i, (key, desc) in enumerate(LLM_PROVIDERS):
    print(f"  {i}: {desc} ({key})")

LLM_CONFIGS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model_name": "glm-4.5-air",
        "api_key_env": "ZHIPU_API_KEY",
        "supports_thinking": True,
        "supports_json_mode": True,
        "supports_system": True,
        "temperature": 0.1,
        "fixed_temperature": False,
        "disable_thinking_by_default": False,
        "max_tokens": 1024,
        "is_local": False,
    },

    "deepseek_v32": {
        "base_url": "https://api.deepseek.com/chat/completions",
        "model_name": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "supports_thinking": False,
        "supports_json_mode": False,
        "supports_system": True,
        "temperature": 0.1,
        "fixed_temperature": False,
        "disable_thinking_by_default": False,
        "max_tokens": 1024,
        "is_local": False,
    },

    "deepseek_r1": {
        "base_url": "https://api.deepseek.com/chat/completions",
        "model_name": "deepseek-reasoner",
        "api_key_env": "DEEPSEEK_API_KEY",
        "supports_thinking": False,
        "supports_json_mode": False,
        "supports_system": True,
        "temperature": 0.1,
        "fixed_temperature": False,
        "disable_thinking_by_default": False,
        "max_tokens": 2048,
        "is_local": False,
    },

    "kimi_k25": {
        "base_url": "https://api.moonshot.cn/v1/chat/completions",
        "model_name": "kimi-k2.5",
        "api_key_env": "MOONSHOT_API_KEY",
        "supports_thinking": True,
        "supports_json_mode": False,
        "supports_system": True,
        "temperature": 0.6,
        "fixed_temperature": True,
        "disable_thinking_by_default": True,
        "max_tokens": 1024,
        "is_local": False,
    },

    # ===== 本地模型：默认按 OpenAI 兼容接口接入 =====
    "qwen3_8b_local": {
        "base_url": os.getenv("QWEN3_8B_BASE_URL", "http://127.0.0.1:8000/v1/chat/completions"),  # 注意是 8000 端口
        "model_name": os.getenv("QWEN3_8B_MODEL_NAME", "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"),  # 使用完整路径
        "api_key_env": "LOCAL_LLM_API_KEY",
        "supports_thinking": False,
        "supports_json_mode": True,
        "supports_system": True,
        "temperature": 0.1,
        "fixed_temperature": False,
        "disable_thinking_by_default": False,
        "max_tokens": 1024,
        "is_local": True,
    },

    "glm4_9b_local": {
        "base_url": os.getenv("GLM4_9B_BASE_URL", "http://127.0.0.1:8002/v1/chat/completions"),
        "model_name": os.getenv("GLM4_9B_MODEL_NAME", "GLM-4-9B"),
        "api_key_env": "LOCAL_LLM_API_KEY",
        "supports_thinking": False,
        "supports_json_mode": False,
        "supports_system": True,
        "temperature": 0.1,
        "fixed_temperature": False,
        "disable_thinking_by_default": False,
        "max_tokens": 1024,
        "is_local": True,
    },

    "llama3_8b_local": {
        "base_url": os.getenv("LLAMA3_8B_BASE_URL", "http://127.0.0.1:8003/v1/chat/completions"),
        "model_name": os.getenv("LLAMA3_8B_MODEL_NAME", "Llama-3-8B"),
        "api_key_env": "LOCAL_LLM_API_KEY",
        "supports_thinking": False,
        "supports_json_mode": False,
        "supports_system": True,
        "temperature": 0.1,
        "fixed_temperature": False,
        "disable_thinking_by_default": False,
        "max_tokens": 1024,
        "is_local": True,
    },
}

CURRENT_LLM_CONFIG = LLM_CONFIGS[LLM_PROVIDER]

GLM_BASE_URL = CURRENT_LLM_CONFIG["base_url"]
GLM_MODEL_NAME = CURRENT_LLM_CONFIG["model_name"]
GLM_TEMPERATURE = CURRENT_LLM_CONFIG["temperature"]
GLM_MAX_TOKENS = CURRENT_LLM_CONFIG["max_tokens"]
GLM_THINKING_ENABLED = CURRENT_LLM_CONFIG["supports_thinking"]

# ================= 训练参数 =================
NER_MAX_LEN = 128
RE_MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 5

# ================= 标签定义 =================
# 18 类实体
ENTITY_TYPES = [
    'Disease', 'Drug', 'Test_items', 'Anatomy', 'Test_Value',
    'Class', 'ADE', 'Treatment', 'Test', 'Symptom',
    'Method', 'Amount', 'Level', 'Pathogenesis', 'Reason',
    'Frequency', 'Operation', 'Duration'
]

# 16 类关系 + No_Relation
RELATION_TYPES = [
    'No_Relation', # 0号标签预留给负样本
    'Anatomy_Disease', 'Drug_Disease', 'Test_items_Disease', 'Class_Disease',
    'ADE_Drug', 'Treatment_Disease', 'Test_Disease', 'Symptom_Disease',
    'Amount_Drug', 'Method_Drug', 'Reason_Disease', 'Pathogenesis_Disease',
    'Frequency_Drug', 'Duration_Drug', 'Operation_Disease', 'ADE_Disease'
]

# 关系类型约束 (用于推理阶段过滤)
RELATION_CONSTRAINTS = {
    'Drug_Disease': ('Drug', 'Disease'),
    'Class_Disease': ('Class', 'Disease'),
    'Anatomy_Disease': ('Anatomy', 'Disease'),
    'Test_items_Disease': ('Test_items', 'Disease'),
    'ADE_Drug': ('ADE', 'Drug'),
    'Treatment_Disease': ('Treatment', 'Disease'),
    'Test_Disease': ('Test', 'Disease'),
    'Symptom_Disease': ('Symptom', 'Disease'),
}
