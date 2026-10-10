import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jieba
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

# ============================================================
# 配置区
# ============================================================
# ============================================================================
# 重要说明：API 密钥配置
# ============================================================================
# 本项目使用多个大语言模型 API 和 Embedding 服务，所有 API 密钥必须通过环境变量配置。
# 请勿将真实的 API 密钥写入代码或提交到版本控制系统！
#
# 配置方法：
# 1. 在项目根目录创建 .env 文件（已提供 .env 示例文件）
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

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8004

LLM_TIMEOUT = 120.0

# build_faiss_index_inline.py / rag_qa_eval_inline.py 使用的向量库目录
RAG_INDEX_DIR = "./diakg_faiss_index"

# 默认检索参数
RAG_TOP_K_DEFAULT = 5
RAG_MIN_SCORE = 0.0

# Qwen Embedding 配置（必须从环境变量读取）
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 必须从环境变量读取，禁止硬编码
DEFAULT_EMBEDDING_BASE_URL = DASHSCOPE_BASE_URL
DEFAULT_EMBEDDING_API_KEY = DASHSCOPE_API_KEY
DEFAULT_EMBEDDING_MODEL = ""  # 留空表示读取向量库 meta.json 中的 embedding_model
DEFAULT_EMBEDDING_DIM = 0  # 0 表示读取向量库 meta.json 中的 embedding_dim

# 内置聊天模型 API Key（必须从环境变量读取）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 必须从环境变量读取，禁止硬编码
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")  # 必须从环境变量读取，禁止硬编码
GLM_API_KEY = os.getenv("GLM_API_KEY")  # 必须从环境变量读取，禁止硬编码

# 内置聊天模型 ID
DEEPSEEK_V3_MODEL = "deepseek-chat"
DEEPSEEK_R1_MODEL = "deepseek-reasoner"
MOONSHOT_MODEL = "kimi-k2.5"
GLM_MODEL = "glm-5.1"
QWEN_CHAT_MODEL = "qwen-plus"

DEFAULT_THINKING_HINT = "start deep thinking, please always use the thinking mode"
DEFAULT_WEB_HINT = "When needed, search authoritative web sources before answering and explicitly integrate up-to-date online information."

SYSTEM_PROMPT = """你是中文医学糖尿病问答专家。
请严格遵守：
1. 只回答与糖尿病、糖尿病并发症、糖尿病药物、糖尿病检查、糖尿病治疗相关的问题。
2. 必须优先依据系统提供的 RAG 检索资料回答。
3. 如果检索资料不足以回答，请回答“根据现有资料无法确定”，并建议结合医生意见。
4. 回答风格简洁、专业、自然，不输出思维过程。
5. 尽量用 1~3 句话回答。
"""

BUILTIN_MODELS: Dict[str, Dict[str, str]] = {
    "deepseek-v3": {
        "provider_name": "DeepSeek",
        "display_name": "deepseek-v3（内置免费）",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": DEEPSEEK_API_KEY,
        "actual_model": DEEPSEEK_V3_MODEL,
        "param_style": "standard",
    },
    "deepseek-r1": {
        "provider_name": "DeepSeek",
        "display_name": "deepseek-r1（内置免费）",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": DEEPSEEK_API_KEY,
        "actual_model": DEEPSEEK_R1_MODEL,
        "param_style": "no_temperature",
    },
    "kimi-2.5": {
        "provider_name": "Moonshot",
        "display_name": "kimi-2.5（内置免费）",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": MOONSHOT_API_KEY,
        "actual_model": MOONSHOT_MODEL,
        "param_style": "kimi_temperature_1",
    },
    "glm-5.1": {
        "provider_name": "智谱 GLM",
        "display_name": "glm-5.1（内置免费）",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": GLM_API_KEY,
        "actual_model": GLM_MODEL,
        "param_style": "standard",
    },
    "qwen-3.6": {
        "provider_name": "阿里云百炼",
        "display_name": "qwen-3.6（内置免费）",
        "base_url": DASHSCOPE_BASE_URL,
        "api_key": DASHSCOPE_API_KEY,
        "actual_model": QWEN_CHAT_MODEL,
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
    return [w.strip() for w in jieba.lcut(normalize_text(text)) if w.strip()]


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class EvidenceItem(BaseModel):
    chunk_id: str = ""
    source_file: str = ""
    doc_id: str = ""
    paragraph_id: str = ""
    score: float = 0.0
    text: str = ""


class CustomModelConfig(BaseModel):
    config_id: Optional[str] = ""
    display_name: Optional[str] = ""
    provider_name: Optional[str] = "custom"
    base_url: str
    api_key: str
    model: str
    note: Optional[str] = ""


class EmbeddingConfig(BaseModel):
    base_url: Optional[str] = DEFAULT_EMBEDDING_BASE_URL
    api_key: Optional[str] = DEFAULT_EMBEDDING_API_KEY
    model: Optional[str] = DEFAULT_EMBEDDING_MODEL
    dimensions: Optional[int] = Field(default=DEFAULT_EMBEDDING_DIM, ge=0)


class QARequest(BaseModel):
    question: str
    model_name: str = Field(default="deepseek-v3")
    enable_thinking: bool = Field(default=True)
    enable_web_search: bool = Field(default=False)
    top_k: int = Field(default=RAG_TOP_K_DEFAULT, ge=1, le=20)
    custom_model_config: Optional[CustomModelConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None


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
    retrieved_context: str = ""
    rag_index_dir: str = RAG_INDEX_DIR
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_dim: Optional[int] = None


class QuestionRouter:
    def classify(self, question: str) -> str:
        q = normalize_text(question)
        if any(x in q for x in ["英文", "英文名", "英文名称", "英文叫什么"]): return "english_name"
        if any(x in q for x in ["同义词", "别名", "另一名称", "别称"]): return "alias"
        if any(x in q for x in ["定义", "什么是", "是一种什么疾病", "是什么病", "属于什么"]): return "definition"
        if any(x in q for x in ["科室", "挂什么科", "就诊", "去哪个科", "诊疗科室", "就诊科室"]): return "department"
        if any(x in q for x in ["预防", "注意事项", "需要注意", "如何预防"]): return "precaution"
        if any(x in q for x in ["病程", "发展", "会如何变化", "如何进展"]): return "course"
        if any(x in q for x in ["不良反应", "副作用"]): return "ade"
        if any(x in q for x in ["药物", "用药", "什么药"]): return "drug"
        if any(x in q for x in ["症状", "表现", "会出现哪些", "会有什么感觉", "主要症状", "具体症状"]): return "symptom"
        if any(x in q for x in ["治疗", "怎么治", "如何治疗", "治疗方法", "主要治疗方法"]): return "treatment"
        if any(x in q for x in ["原因", "病因", "为什么会", "发生原因"]): return "reason"
        if any(x in q for x in ["发病机制", "机制"]): return "pathogenesis"
        if any(x in q for x in ["检查", "诊断", "如何诊断", "需要做哪些检查", "最重要的检查"]): return "test"
        if any(x in q for x in ["检验指标", "指标", "化验", "HbA1c", "血糖", "空腹血糖", "PPG"]): return "test_items"
        if any(x in q for x in ["并发症", "相关疾病", "有关疾病", "关联疾病"]): return "related_disease"
        if any(x in q for x in ["分类", "分型", "属于哪一类"]): return "class"
        if any(x in q for x in ["解剖部位", "部位", "发生在哪", "涉及哪些部位"]): return "anatomy"
        return "general"

    def extract_entity(self, question: str) -> Optional[str]:
        q = normalize_text(question)
        patterns = [
            r"^(.*?)的?(主要)?症状是什么", r"^(.*?)会出现哪些症状", r"^(.*?)有哪些不良反应",
            r"^(.*?)的副作用是什么", r"^(.*?)相关药物有哪些", r"^(.*?)的治疗方法有哪些",
            r"^(.*?)如何治疗", r"^(.*?)的原因是什么", r"^(.*?)的病因是什么",
            r"^(.*?)的英文名称是什么", r"^(.*?)的英文名是什么", r"^(.*?)的别名是什么",
            r"^(.*?)的同义词是什么", r"^(.*?)如何诊断", r"^(.*?)需要做哪些检查",
            r"^(.*?)最重要的检查是什么", r"^(.*?)的定义是什么", r"^什么是(.*?)$",
            r"^(.*?)是什么", r"^(.*?)有哪些并发症", r"^(.*?)相关的疾病有哪些",
            r"^(.*?)的注意事项是什么", r"^(.*?)如何预防", r"^(.*?)的病程是如何发展的",
            r"^(.*?)相关检验指标有哪些", r"^(.*?)相关发病机制有哪些", r"^(.*?)涉及哪些解剖部位",
        ]
        for p in patterns:
            m = re.match(p, q)
            if m:
                ent = normalize_text(m.group(1))
                if ent:
                    return ent
        stopwords = {"什么", "哪些", "多少", "怎么", "如何", "为什么", "是否", "属于", "有关", "相关", "主要", "具体", "一种", "这个", "那个", "会", "是", "的", "了", "吗", "英文", "名称", "定义", "症状", "治疗", "原因", "病因", "检查", "诊断", "科室", "并发症", "注意事项", "预防", "病程", "药物", "副作用", "不良反应", "检验指标", "发病机制"}
        cands = [w for w in tokenize_zh(q) if len(w) >= 2 and w not in stopwords]
        cands = uniq_keep_order(cands)
        return sorted(cands, key=lambda x: len(x), reverse=True)[0] if cands else None


class FaissRetriever:
    def __init__(
        self,
        index_dir: str,
        embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
        embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        if faiss is None:
            raise RuntimeError("未安装 faiss，请执行：pip install faiss-cpu")
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise RuntimeError(f"向量库目录不存在：{self.index_dir.resolve()}")
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.chunks = load_jsonl(self.index_dir / "chunks.jsonl")

        meta_model = normalize_text(self.meta.get("embedding_model", ""))
        meta_dim = int(self.meta.get("embedding_dim", 0))
        self.embedding_model = normalize_text(embedding_model) or meta_model
        self.embedding_dim = int(embedding_dim or 0) or meta_dim
        self.embedding_base_url = normalize_text(embedding_base_url) or DEFAULT_EMBEDDING_BASE_URL
        self.embedding_api_key = normalize_text(embedding_api_key) or DEFAULT_EMBEDDING_API_KEY

        if not self.embedding_model:
            raise RuntimeError("Embedding 模型为空，请在 meta.json 或 embedding_config.model 中配置。")
        if not self.embedding_dim:
            raise RuntimeError("Embedding 维度为空，请在 meta.json 或 embedding_config.dimensions 中配置。")
        if not self.embedding_api_key or self.embedding_api_key.startswith("请填"):
            raise RuntimeError("请配置 Embedding API Key，用于向量检索。")

        self.client = OpenAI(api_key=self.embedding_api_key, base_url=self.embedding_base_url)

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

    def search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        q = self.embed_query(question)
        scores, ids = self.index.search(q, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = dict(self.chunks[int(idx)])
            item["score"] = float(score)
            if item["score"] >= RAG_MIN_SCORE:
                results.append(item)
        return results


class RAGGenerator:
    def __init__(self):
        self._retrievers: Dict[Tuple[str, str, int, str], FaissRetriever] = {}
        self._retriever_errors: Dict[Tuple[str, str, int, str], str] = {}

    def resolve_model(self, model_name: str, custom_model_config: Optional[CustomModelConfig]) -> Tuple[str, Dict[str, str]]:
        normalized = LEGACY_MODEL_ALIASES.get(normalize_text(model_name), normalize_text(model_name))
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

    def embedding_key(self, embedding_config: Optional[EmbeddingConfig]) -> Tuple[str, str, int, str]:
        cfg = embedding_config or EmbeddingConfig()
        base_url = normalize_text(cfg.base_url) or DEFAULT_EMBEDDING_BASE_URL
        model = normalize_text(cfg.model) or DEFAULT_EMBEDDING_MODEL
        dim = int(cfg.dimensions or 0)
        api_key = normalize_text(cfg.api_key) or DEFAULT_EMBEDDING_API_KEY
        return base_url, model, dim, api_key

    def retriever(self, embedding_config: Optional[EmbeddingConfig] = None) -> FaissRetriever:
        key = self.embedding_key(embedding_config)
        if key in self._retrievers:
            return self._retrievers[key]
        if key in self._retriever_errors:
            raise RuntimeError(self._retriever_errors[key])
        try:
            base_url, model, dim, api_key = key
            self._retrievers[key] = FaissRetriever(
                RAG_INDEX_DIR,
                embedding_base_url=base_url,
                embedding_api_key=api_key,
                embedding_model=model,
                embedding_dim=dim,
            )
            return self._retrievers[key]
        except Exception as e:
            self._retriever_errors[key] = str(e)
            raise

    def build_messages(self, question: str, intent: str, entity: Optional[str], contexts: List[Dict[str, Any]], enable_thinking: bool, enable_web_search: bool) -> List[Dict[str, str]]:
        context_text = "\n\n".join(
            f"[{i + 1}] 来源={c.get('source_file','')} doc={c.get('doc_id','')} paragraph={c.get('paragraph_id','')} score={float(c.get('score', 0)):.4f}\n{c.get('text','')}"
            for i, c in enumerate(contexts)
        )
        user = f"""请基于 RAG 检索资料回答下面这个糖尿病相关问题。

问题类型：{intent}
识别实体：{entity or '未识别'}
用户问题：{question}
联网模式：{'开启' if enable_web_search else '关闭'}
深度思考：{'开启' if enable_thinking else '关闭'}

检索资料：
{context_text or '无'}

要求：
- 必须优先依据检索资料回答
- 如果资料不足，回答“根据现有资料无法确定”
- 中文输出，1~3 句话
- 不要输出分析过程
"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if enable_thinking:
            messages.append({"role": "system", "content": DEFAULT_THINKING_HINT})
        if enable_web_search:
            messages.append({"role": "system", "content": DEFAULT_WEB_HINT})
        messages.append({"role": "user", "content": user})
        return messages

    async def generate(self, question: str, intent: str, entity: Optional[str], contexts: List[Dict[str, Any]], model_name: str, enable_thinking: bool, enable_web_search: bool, custom_model_config: Optional[CustomModelConfig]) -> Tuple[str, Dict[str, str], str]:
        resolved_name, cfg = self.resolve_model(model_name, custom_model_config)
        if not cfg.get("api_key"):
            return f"模型 {resolved_name} 未配置 API Key，无法生成回答。", cfg, resolved_name
        client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=LLM_TIMEOUT)
        request_kwargs: Dict[str, Any] = {
            "model": cfg["actual_model"],
            "messages": self.build_messages(question, intent, entity, contexts, enable_thinking, enable_web_search),
        }
        param_style = cfg.get("param_style", "standard")
        if param_style == "standard":
            request_kwargs["temperature"] = 0.2
        elif param_style == "kimi_temperature_1":
            request_kwargs["temperature"] = 1
        elif param_style == "kimi_k25":
            request_kwargs["temperature"] = 1
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}
        try:
            resp = await client.chat.completions.create(**request_kwargs)
            answer = resp.choices[0].message.content if resp and resp.choices else ""
            return normalize_text(answer) or "模型未返回有效内容。", cfg, resolved_name
        except Exception as e:
            return f"模型调用失败：{e}", cfg, resolved_name


app = FastAPI(title="糖尿病 LLM + RAG 问答系统", description="FAISS 检索增强 + 多模型问答", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
router = QuestionRouter()
generator = RAGGenerator()


@app.get("/")
def root():
    return {"message": "糖尿病 LLM + RAG 问答系统后端已启动", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    retriever_status = "not_loaded"
    retriever_error = None
    embedding_model = None
    try:
        r = generator.retriever(None)
        retriever_status = "ok"
        embedding_model = r.embedding_model
    except Exception as e:
        retriever_status = "error"
        retriever_error = str(e)
    return {
        "status": "ok" if retriever_status == "ok" else "degraded",
        "retriever_status": retriever_status,
        "retriever_error": retriever_error,
        "rag_index_dir": RAG_INDEX_DIR,
        "default_top_k": RAG_TOP_K_DEFAULT,
        "embedding_model": embedding_model,
        "default_embedding_base_url": DEFAULT_EMBEDDING_BASE_URL,
        "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
        "default_embedding_dim": DEFAULT_EMBEDDING_DIM,
        "supported_models": list(BUILTIN_MODELS.keys()) + ["custom"],
        "builtin_models": [{"model_name": k, "display_name": v["display_name"], "provider_name": v["provider_name"], "actual_model": v["actual_model"], "base_url": v["base_url"]} for k, v in BUILTIN_MODELS.items()],
        "default_thinking_hint": DEFAULT_THINKING_HINT,
        "default_web_hint": DEFAULT_WEB_HINT,
    }


@app.get("/schema")
def schema():
    return {
        "supported_models": list(BUILTIN_MODELS.keys()) + ["custom"],
        "supported_intents": ["definition", "english_name", "alias", "department", "precaution", "course", "symptom", "treatment", "reason", "pathogenesis", "test", "drug", "ade", "test_items", "class", "anatomy", "related_disease", "general"],
        "rag_fields": ["top_k", "embedding_config", "evidence_chain", "retrieved_context", "rag_index_dir", "embedding_model", "embedding_base_url", "embedding_dim"],
    }


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    question = normalize_text(req.question)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    t1 = time.perf_counter()
    intent = router.classify(question)
    entity = router.extract_entity(question)
    try:
        retriever = generator.retriever(req.embedding_config)
        contexts = retriever.search(question=question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 检索失败：{e}")
    t2 = time.perf_counter()

    answer, cfg, resolved_name = await generator.generate(
        question=question,
        intent=intent,
        entity=entity,
        contexts=contexts,
        model_name=req.model_name,
        enable_thinking=req.enable_thinking,
        enable_web_search=req.enable_web_search,
        custom_model_config=req.custom_model_config,
    )
    t3 = time.perf_counter()

    model_name = LEGACY_MODEL_ALIASES.get(normalize_text(req.model_name), normalize_text(req.model_name))
    if model_name.startswith("custom:"):
        model_name = "custom"
    if resolved_name == "custom":
        model_name = "custom"

    evidence = [
        EvidenceItem(
            chunk_id=str(c.get("chunk_id", "")),
            source_file=str(c.get("source_file", "")),
            doc_id=str(c.get("doc_id", "")),
            paragraph_id=str(c.get("paragraph_id", "")),
            score=float(c.get("score", 0.0)),
            text=normalize_text(c.get("text", "")),
        )
        for c in contexts
    ]

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
        retrieved_count=len(evidence),
        retrieval_time_sec=round(t2 - t1, 6),
        generation_time_sec=round(t3 - t2, 6),
        total_time_sec=round(t3 - t1, 6),
        evidence_chain=evidence,
        retrieved_context="\n\n".join(x.text for x in evidence),
        rag_index_dir=RAG_INDEX_DIR,
        embedding_model=getattr(retriever, "embedding_model", None),
        embedding_base_url=getattr(retriever, "embedding_base_url", None),
        embedding_dim=getattr(retriever, "embedding_dim", None),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_rag_llm_8004_embedding_custom:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
