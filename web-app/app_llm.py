# -*- coding: utf-8 -*-
import os
import re
import time
from typing import List, Dict, Optional, Any, Tuple

import jieba
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
DEFAULT_THINKING_HINT = "start deep thinking, please always use the thinking mode"
DEFAULT_WEB_HINT = "When needed, search authoritative web sources before answering and explicitly integrate up-to-date online information."

SYSTEM_PROMPT = """你是中文医学糖尿病问答专家。
请严格遵守：
1. 只回答与糖尿病、糖尿病并发症、糖尿病药物、糖尿病检查、糖尿病治疗相关的问题。
2. 回答风格简洁、专业、自然。
3. 优先给出直接答案，不要输出思维过程。
4. 如果问题信息不足，给出谨慎回答，并说明“建议结合医生意见”。
5. 如果你不确定，不要编造，明确说“暂时无法确定”。
6. 尽量用 1~3 句话回答。
"""

# ============================================================================
# 重要说明：API 密钥配置
# ============================================================================
# 本项目使用多个大语言模型 API，所有 API 密钥必须通过环境变量配置。
# 请勿将真实的 API 密钥写入代码或提交到版本控制系统！
#
# 配置方法：
# 1. 在项目根目录创建 .env 文件（已提供 .env 示例文件）
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

# 按用户要求：界面内置展示名固定为这 5 个；真实调用模型单独映射。
BUILTIN_MODELS: Dict[str, Dict[str, str]] = {
    "deepseek-v3": {
        "provider_name": "DeepSeek",
        "display_name": "deepseek-v3（内置免费）",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "deepseek-chat",
        "param_style": "standard",
    },
    "deepseek-r1": {
        "provider_name": "DeepSeek",
        "display_name": "deepseek-r1（内置免费）",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "deepseek-reasoner",
        "param_style": "no_temperature",
    },
    "kimi-2.5": {
        "provider_name": "Moonshot",
        "display_name": "kimi-2.5（内置免费）",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": os.getenv("MOONSHOT_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "kimi-k2.5",
        "param_style": "kimi_k25",
    },
    "glm-5.1": {
        "provider_name": "智谱 GLM",
        "display_name": "glm-5.1（内置免费）",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.getenv("GLM_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "glm-5.1",
        "param_style": "standard",
    },
    "qwen-3.6": {
        "provider_name": "阿里云百炼",
        "display_name": "qwen-3.6（内置免费）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),  # 必须从环境变量读取，禁止硬编码
        "actual_model": "qwen-3.6",
        "param_style": "standard",
    },
}

LEGACY_MODEL_ALIASES: Dict[str, str] = {
    "deepseek-chat": "deepseek-v3",
    "deepseek-reasoner": "deepseek-r1",
    "kimi-k2.5": "kimi-2.5",
    "glm-5.1": "glm-5.1",
    "qwen-plus": "qwen-3.6",
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().replace("\u3000", " ")
    return re.sub(r"\s+", " ", text)


def mask_secret(secret: str) -> str:
    secret = normalize_text(secret)
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * max(4, len(secret) - 8)}{secret[-4:]}"


def tokenize_zh(text: str) -> List[str]:
    text = normalize_text(text)
    return [w.strip() for w in jieba.lcut(text) if w.strip()]


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class EvidenceItem(BaseModel):
    center_name: str = ""
    center_etype: str = ""
    center_etype_zh: str = ""
    rel_type: str = ""
    rel_type_zh: str = ""
    neighbor_name: str = ""
    neighbor_etype: str = ""
    neighbor_etype_zh: str = ""
    evidence: str = ""


class CustomModelConfig(BaseModel):
    config_id: Optional[str] = ""
    display_name: Optional[str] = ""
    provider_name: Optional[str] = "custom"
    base_url: str
    api_key: str
    model: str
    note: Optional[str] = ""


class QARequest(BaseModel):
    question: str
    model_name: str = Field(default="deepseek-v3")
    enable_thinking: bool = Field(default=True)
    enable_web_search: bool = Field(default=False)
    custom_model_config: Optional[CustomModelConfig] = None


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
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    evidence_chain: List[EvidenceItem]


class QuestionRouter:
    def classify(self, question: str) -> str:
        q = normalize_text(question)
        if any(x in q for x in ["英文", "英文名", "英文名称", "英文叫什么"]):
            return "english_name"
        if any(x in q for x in ["同义词", "别名", "另一名称", "别称"]):
            return "alias"
        if any(x in q for x in ["定义", "什么是", "是一种什么疾病", "是什么病", "属于什么"]):
            return "definition"
        if any(x in q for x in ["科室", "挂什么科", "就诊", "去哪个科", "诊疗科室", "就诊科室"]):
            return "department"
        if any(x in q for x in ["预防", "注意事项", "需要注意", "如何预防"]):
            return "precaution"
        if any(x in q for x in ["病程", "发展", "会如何变化", "如何进展"]):
            return "course"
        if any(x in q for x in ["不良反应", "副作用"]):
            return "ade"
        if any(x in q for x in ["药物", "用药", "什么药"]):
            return "drug"
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
        if any(x in q for x in ["检验指标", "指标", "化验", "HbA1c", "血糖", "空腹血糖", "PPG"]):
            return "test_items"
        if any(x in q for x in ["并发症", "相关疾病", "有关疾病", "关联疾病"]):
            return "related_disease"
        if any(x in q for x in ["分类", "分型", "属于哪一类"]):
            return "class"
        if any(x in q for x in ["解剖部位", "部位", "发生在哪", "涉及哪些部位"]):
            return "anatomy"
        return "general"

    def extract_entity(self, question: str) -> Optional[str]:
        q = normalize_text(question)
        patterns = [
            r"^(.*?)的?(主要)?症状是什么",
            r"^(.*?)会出现哪些症状",
            r"^(.*?)有哪些不良反应",
            r"^(.*?)的副作用是什么",
            r"^(.*?)相关药物有哪些",
            r"^(.*?)的治疗方法有哪些",
            r"^(.*?)如何治疗",
            r"^(.*?)的原因是什么",
            r"^(.*?)的病因是什么",
            r"^(.*?)的英文名称是什么",
            r"^(.*?)的英文名是什么",
            r"^(.*?)的别名是什么",
            r"^(.*?)的同义词是什么",
            r"^(.*?)如何诊断",
            r"^(.*?)需要做哪些检查",
            r"^(.*?)最重要的检查是什么",
            r"^(.*?)的定义是什么",
            r"^什么是(.*?)$",
            r"^(.*?)是什么",
            r"^(.*?)有哪些并发症",
            r"^(.*?)相关的疾病有哪些",
            r"^(.*?)的注意事项是什么",
            r"^(.*?)如何预防",
            r"^(.*?)的病程是如何发展的",
            r"^(.*?)相关检验指标有哪些",
            r"^(.*?)相关发病机制有哪些",
            r"^(.*?)涉及哪些解剖部位",
        ]
        for p in patterns:
            m = re.match(p, q)
            if m:
                ent = normalize_text(m.group(1))
                if ent:
                    return ent
        stopwords = {
            "什么", "哪些", "多少", "怎么", "如何", "为什么", "是否", "属于", "有关", "相关", "主要", "具体",
            "一种", "这个", "那个", "会", "是", "的", "了", "吗", "英文", "名称", "定义", "症状", "治疗", "原因",
            "病因", "检查", "诊断", "科室", "并发症", "注意事项", "预防", "病程", "药物", "副作用", "不良反应",
            "检验指标", "发病机制"
        }
        cands = [w for w in tokenize_zh(q) if len(w) >= 2 and w not in stopwords]
        cands = uniq_keep_order(cands)
        if cands:
            return sorted(cands, key=lambda x: len(x), reverse=True)[0]
        return None


class PureLLMGenerator:
    def resolve_model(self, model_name: str, custom_model_config: Optional[CustomModelConfig]) -> Tuple[str, Dict[str, str]]:
        normalized = normalize_text(model_name)
        normalized = LEGACY_MODEL_ALIASES.get(normalized, normalized)
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

    async def generate(
        self,
        question: str,
        intent: str,
        entity: Optional[str],
        model_name: str,
        enable_thinking: bool,
        enable_web_search: bool,
        custom_model_config: Optional[CustomModelConfig],
    ) -> Tuple[str, Dict[str, str]]:
        resolved_name, cfg = self.resolve_model(model_name, custom_model_config)
        if not cfg.get("api_key"):
            return f"模型 {resolved_name} 未配置 API Key，无法生成回答。", cfg

        prompt = f"""请回答下面这个糖尿病相关问题。

问题类型：{intent}
识别实体：{entity or '未识别'}
用户问题：{question}
联网模式：{'开启' if enable_web_search else '关闭'}
深度思考：{'开启' if enable_thinking else '关闭'}

要求：
- 直接回答
- 中文输出
- 尽量控制在 1~3 句话
- 不要输出分析过程
- 若医学上存在不确定性，请用审慎措辞
"""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if enable_thinking:
            messages.append({"role": "system", "content": DEFAULT_THINKING_HINT})
        if enable_web_search:
            messages.append({"role": "system", "content": DEFAULT_WEB_HINT})
        messages.append({"role": "user", "content": prompt})

        client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=LLM_TIMEOUT)
        request_kwargs: Dict[str, Any] = {
            "model": cfg["actual_model"],
            "messages": messages,
        }

        param_style = cfg.get("param_style", "standard")
        if param_style == "standard":
            request_kwargs["temperature"] = 0.2
        elif param_style == "no_temperature":
            pass
        elif param_style == "kimi_k25":
            # Kimi K2.5 对 temperature/top_p/n 等参数有固定限制，直接不传 temperature。
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if enable_thinking else "disabled"}
            }
        else:
            request_kwargs["temperature"] = 0.2

        try:
            resp = await client.chat.completions.create(**request_kwargs)
            answer = ""
            if resp and resp.choices and resp.choices[0].message:
                answer = resp.choices[0].message.content or ""
            return normalize_text(answer) or "模型未返回有效内容。", cfg
        except Exception as e:
            return f"模型调用失败：{e}", cfg


app = FastAPI(title="糖尿病纯大模型问答系统", description="支持内置免费模型与多自定义 API 配置", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = QuestionRouter()
generator = PureLLMGenerator()


@app.get("/")
def root():
    return {"message": "糖尿病纯大模型问答系统后端已启动", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supported_models": list(BUILTIN_MODELS.keys()) + ["custom"],
        "builtin_models": [
            {
                "model_name": k,
                "display_name": v["display_name"],
                "provider_name": v["provider_name"],
                "actual_model": v["actual_model"],
                "base_url": v["base_url"],
            }
            for k, v in BUILTIN_MODELS.items()
        ],
        "default_thinking_hint": DEFAULT_THINKING_HINT,
        "default_web_hint": DEFAULT_WEB_HINT,
    }


@app.get("/schema")
def schema():
    return {
        "supported_models": list(BUILTIN_MODELS.keys()) + ["custom"],
        "supported_intents": [
            "definition", "english_name", "alias", "department", "precaution", "course", "symptom", "treatment",
            "reason", "pathogenesis", "test", "drug", "ade", "test_items", "class", "anatomy", "related_disease", "general"
        ]
    }


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    question = normalize_text(req.question)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    t1 = time.perf_counter()
    intent = router.classify(question)
    entity = router.extract_entity(question)
    t2 = time.perf_counter()

    answer, cfg = await generator.generate(
        question=question,
        intent=intent,
        entity=entity,
        model_name=req.model_name,
        enable_thinking=req.enable_thinking,
        enable_web_search=req.enable_web_search,
        custom_model_config=req.custom_model_config,
    )
    t3 = time.perf_counter()

    model_name = LEGACY_MODEL_ALIASES.get(normalize_text(req.model_name), normalize_text(req.model_name))
    if model_name.startswith("custom:"):
        model_name = "custom"

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
        cypher=None,
        retrieved_count=0,
        retrieval_time_sec=round(t2 - t1, 6),
        generation_time_sec=round(t3 - t2, 6),
        total_time_sec=round(t3 - t1, 6),
        evidence_chain=[],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_llm_custom:app", host="0.0.0.0", port=8002, reload=True)
