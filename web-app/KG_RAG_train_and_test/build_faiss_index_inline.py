import json
import time
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from openai import OpenAI
from tqdm import tqdm


# ============================================================
# 配置区：所有参数都写这里
# ============================================================
# ============================================================================
# 重要说明：API 密钥配置
# ============================================================================
# 本脚本使用阿里云百炼 (DashScope) 的 Embedding 服务，API 密钥必须通过环境变量配置。
# 请勿将真实的 API 密钥写入代码或提交到版本控制系统！
#
# 配置方法：
# 1. 在项目根目录创建 .env 文件
# 2. 在 .env 文件中配置：DASHSCOPE_API_KEY=your_dashscope_api_key
# 3. 或者在系统环境变量中配置 DASHSCOPE_API_KEY
#
# 获取 API Key：https://dashscope.aliyun.com/
# ============================================================================

import os

CONFIG = {
    # 阿里云百炼 / DashScope API Key（必须从环境变量读取）
    # 请勿在此处硬编码真实密钥！
    "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY"),  # 必须从环境变量读取

    # diakg_data 解压后的目录
    "DIAKG_DIR": "../diakg_data",

    # 向量库输出目录
    "INDEX_DIR": "../diakg_faiss_index",

    # Qwen3 Embedding 系列，按你的要求使用 text-embedding-v4
    "EMBEDDING_MODEL": "text-embedding-v4",

    # text-embedding-v4 支持可变维度；1024 是常用默认选择
    "EMBEDDING_DIM": 1024,

    # 中文医学文本按字符滑窗切块
    "MAX_CHARS": 800,
    "OVERLAP_CHARS": 120,

    # DashScope OpenAI 兼容模式地址
    "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",

    # embedding 批量大小；text-embedding-v4 批量建议不要超过 10
    "EMBED_BATCH_SIZE": 10,
}


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def load_diakg_chunks(diakg_dir: str, max_chars: int, overlap_chars: int) -> List[Dict]:
    """
    把 DIAKG JSON 转成 RAG chunk。

    diakg_data 的结构通常是：
    - doc_id
    - paragraphs
      - paragraph_id
      - paragraph
      - sentences
        - entities
        - relations

    我们优先用 paragraph 字段作为检索文本；
    如果 paragraph 为空，就拼接 sentences 中的 sentence。
    """
    chunks = []
    paths = sorted(
        Path(diakg_dir).glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    if not paths:
        raise RuntimeError(f"没有在 {diakg_dir} 找到 .json 文件，请检查 diakg_data.zip 是否已解压。")

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        doc_id = str(doc.get("doc_id", path.stem))
        paragraphs = doc.get("paragraphs", [])

        for p in paragraphs:
            paragraph_id = str(p.get("paragraph_id", ""))
            text = normalize_text(p.get("paragraph", ""))

            if not text:
                sentence_texts = [
                    normalize_text(s.get("sentence", ""))
                    for s in p.get("sentences", [])
                ]
                text = normalize_text("。".join([s for s in sentence_texts if s]))

            if not text:
                continue

            entities = []
            relations = []
            for s in p.get("sentences", []):
                entities.extend(s.get("entities", []) or [])
                relations.extend(s.get("relations", []) or [])

            entity_names = sorted({
                e.get("entity", "")
                for e in entities
                if e.get("entity")
            })
            entity_types = sorted({
                e.get("entity_type", "")
                for e in entities
                if e.get("entity_type")
            })
            relation_types = sorted({
                r.get("relation_type", "")
                for r in relations
                if r.get("relation_type")
            })

            start = 0
            chunk_no = 0
            while start < len(text):
                end = min(len(text), start + max_chars)
                chunk_text = text[start:end]

                if len(chunk_text) >= 20:
                    chunks.append({
                        "chunk_id": f"{doc_id}-{paragraph_id}-{chunk_no}",
                        "doc_id": doc_id,
                        "paragraph_id": paragraph_id,
                        "chunk_no": chunk_no,
                        "text": chunk_text,
                        "entities": entity_names[:80],
                        "entity_types": entity_types,
                        "relation_types": relation_types,
                        "source_file": path.name,
                    })

                if end >= len(text):
                    break

                start = max(0, end - overlap_chars)
                chunk_no += 1

    return chunks


def get_client() -> OpenAI:
    api_key = CONFIG["DASHSCOPE_API_KEY"]
    if not api_key or api_key == "请填你的DASHSCOPE_API_KEY":
        raise RuntimeError("请先在 CONFIG['DASHSCOPE_API_KEY'] 填入真实 API Key。")

    return OpenAI(
        api_key=api_key,
        base_url=CONFIG["DASHSCOPE_BASE_URL"],
    )


def embed_texts(client: OpenAI, texts: List[str], text_type: str) -> np.ndarray:
    """
    text_type:
    - document：构建知识库向量时使用
    - query：检索问题向量时使用
    """
    vectors = []
    batch_size = CONFIG["EMBED_BATCH_SIZE"]

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {text_type}"):
        batch = texts[i:i + batch_size]

        for retry in range(4):
            try:
                resp = client.embeddings.create(
                    model=CONFIG["EMBEDDING_MODEL"],
                    input=batch,
                    dimensions=CONFIG["EMBEDDING_DIM"],
                    extra_body={"text_type": text_type},
                )
                vectors.extend([item.embedding for item in resp.data])
                break
            except Exception:
                if retry == 3:
                    raise
                time.sleep(2 ** retry)

    arr = np.asarray(vectors, dtype="float32")

    # 归一化后使用内积 IndexFlatIP，相当于 cosine similarity
    faiss.normalize_L2(arr)
    return arr


def main():
    index_dir = Path(CONFIG["INDEX_DIR"])
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_diakg_chunks(
        diakg_dir=CONFIG["DIAKG_DIR"],
        max_chars=CONFIG["MAX_CHARS"],
        overlap_chars=CONFIG["OVERLAP_CHARS"],
    )

    print(f"Loaded chunks: {len(chunks)}")

    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    client = get_client()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(client, texts, text_type="document")

    index = faiss.IndexFlatIP(CONFIG["EMBEDDING_DIM"])
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    meta = {
        "embedding_model": CONFIG["EMBEDDING_MODEL"],
        "embedding_dim": CONFIG["EMBEDDING_DIM"],
        "chunk_count": len(chunks),
        "metric": "inner_product_on_l2_normalized_vectors_cosine",
        "max_chars": CONFIG["MAX_CHARS"],
        "overlap_chars": CONFIG["OVERLAP_CHARS"],
    }

    (index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Build Index Done =====")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
