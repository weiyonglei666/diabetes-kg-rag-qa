import os
import json
import time
import random
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from utils.glm_client import GLMClient
from utils.prompts import NER_SYSTEM_PROMPT
from data.postprocess import validate_ner_result
from config.config import DATA_DIR, OUTPUT_DIR, LLM_PROVIDER, CURRENT_LLM_CONFIG


# ================= 并发/重试配置 =================
MAX_WORKERS = 200              # 线程池大小
MAX_INFLIGHT_REQUESTS = 120    # 实际同时发出的 API 请求数（信号量控制）
MAX_RETRIES = 2              # 单次请求最大重试次数
BASE_BACKOFF = 1.5           # 指数退避起始秒数
MAX_BACKOFF = 20             # 单次等待上限
REQUEST_TIMEOUT_HINT = 180   # 仅注释提示，真正 timeout 在 glm_client.py 中


REQUEST_SEMAPHORE = threading.BoundedSemaphore(MAX_INFLIGHT_REQUESTS)


def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def build_output_paths():
    provider = LLM_PROVIDER
    model_name = CURRENT_LLM_CONFIG["model_name"].replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    timestamped_path = os.path.join(
        OUTPUT_DIR,
        f"glm_ner_results_{provider}_{model_name}_{timestamp}.jsonl"
    )

    latest_model_path = os.path.join(
        OUTPUT_DIR,
        f"glm_ner_results_latest_{provider}_{model_name}.jsonl"
    )

    legacy_path = os.path.join(
        OUTPUT_DIR,
        "glm_ner_results.jsonl"
    )

    return timestamped_path, latest_model_path, legacy_path


def init_output_files(paths):
    for path in paths:
        with open(path, "w", encoding="utf-8") as fw:
            pass


def append_jsonl(paths, item):
    line = json.dumps(item, ensure_ascii=False) + "\n"
    for path in paths:
        with open(path, "a", encoding="utf-8") as fw:
            fw.write(line)


def split_text_with_offsets(text, max_len=60):
    """
    仅用于推理时切分长句，不改变原始句子。
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_len:
        return [{"sub_text": text, "offset": 0}]

    pieces = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_len, n)
        window = text[start:end]

        split_pos = -1

        for p in ["。", "；", ";", "！", "!", "？", "?"]:
            pos = window.rfind(p)
            if pos > split_pos:
                split_pos = pos

        if split_pos == -1:
            for p in ["，", ","]:
                pos = window.rfind(p)
                if pos > split_pos:
                    split_pos = pos

        if split_pos != -1 and split_pos != len(window) - 1:
            cut_end = start + split_pos + 1
        else:
            cut_end = end

        raw_piece = text[start:cut_end]
        sub_text = raw_piece.strip()

        if sub_text:
            left_trim = len(raw_piece) - len(raw_piece.lstrip())
            real_offset = start + left_trim
            pieces.append({
                "sub_text": sub_text,
                "offset": real_offset
            })

        if cut_end <= start:
            cut_end = min(start + max_len, n)
            if cut_end <= start:
                break

        start = cut_end

    return pieces


def load_all_sentences(data_dir, max_samples=None):
    samples = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到数据目录: {data_dir}")

    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    for filename in files:
        file_path = os.path.join(data_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except Exception:
                continue

        docs = raw_data if isinstance(raw_data, list) else [raw_data]

        for doc_idx, doc in enumerate(docs):
            for para_idx, para in enumerate(doc.get("paragraphs", [])):
                for sent_idx, sent in enumerate(para.get("sentences", [])):
                    text = sent.get("sentence", "").strip()
                    if not text:
                        continue

                    samples.append({
                        "file_name": filename,
                        "doc_idx": doc_idx,
                        "para_idx": para_idx,
                        "sent_idx": sent_idx,
                        "text": text,
                        "gold_entities": sent.get("entities", []),
                        "gold_relations": sent.get("relations", [])
                    })

                    if max_samples is not None and len(samples) >= max_samples:
                        return samples

    return samples


def normalize_ner_result(text, ner_result):
    if isinstance(ner_result, dict):
        entities = ner_result.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        return {
            "text": ner_result.get("text", text),
            "entities": entities
        }

    if isinstance(ner_result, list):
        return {
            "text": text,
            "entities": ner_result
        }

    return {
        "text": text,
        "entities": []
    }


def is_retryable_error(exc: Exception) -> bool:
    """
    判断是否属于可重试错误
    """
    msg = repr(exc).lower()

    retry_keywords = [
        "429",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "read timed out",
        "connecttimeout",
        "readtimeout",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "502",
        "503",
        "504",
        "remote disconnected",
        "connection broken",
    ]

    return any(k in msg for k in retry_keywords)


def compute_backoff(attempt: int) -> float:
    """
    指数退避 + 随机抖动
    """
    base = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt))
    jitter = random.uniform(0, 0.8)
    return min(MAX_BACKOFF, base + jitter)


def safe_chat_json(client, messages, temperature, max_tokens, use_thinking):
    """
    包一层：
    - 并发信号量
    - 自动重试
    - 429/超时退避
    """
    last_exc = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            with REQUEST_SEMAPHORE:
                return client.chat_json(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=0,   # 这里关掉 client 内部重试，统一由外层控制
                    use_thinking=use_thinking,
                )

        except Exception as e:
            last_exc = e

            if not is_retryable_error(e):
                raise

            if attempt >= MAX_RETRIES:
                raise

            sleep_s = compute_backoff(attempt)
            time.sleep(sleep_s)

    raise last_exc


def extract_entities_from_sub_text(client, sub_text):
    messages = [
        {"role": "system", "content": NER_SYSTEM_PROMPT},
        {"role": "user", "content": sub_text}
    ]

    ner_result, raw_result = safe_chat_json(
        client=client,
        messages=messages,
        temperature=0.0,
        max_tokens=9192,
        use_thinking=False
    )

    ner_result = normalize_ner_result(sub_text, ner_result)
    ner_result = validate_ner_result(sub_text, ner_result)

    return ner_result, raw_result


def remap_entities_to_original(entities, offset):
    mapped = []

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        start_idx = ent.get("start_idx")
        end_idx = ent.get("end_idx")

        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            continue

        mapped.append({
            "entity": ent.get("entity", ""),
            "entity_type": ent.get("entity_type", ""),
            "start_idx": start_idx + offset,
            "end_idx": end_idx + offset
        })

    return mapped


def deduplicate_entities(entities):
    seen = set()
    results = []

    for ent in entities:
        key = (
            ent.get("entity", ""),
            ent.get("entity_type", ""),
            ent.get("start_idx", -1),
            ent.get("end_idx", -1)
        )
        if key in seen:
            continue
        seen.add(key)
        results.append(ent)

    results.sort(key=lambda x: (
        x.get("start_idx", -1),
        x.get("end_idx", -1),
        x.get("entity_type", "")
    ))
    return results


def merge_usage(raw_results):
    merged = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    has_any = False

    for item in raw_results:
        raw_result = item.get("raw_result", {})
        if not isinstance(raw_result, dict):
            continue

        usage = raw_result.get("usage", {})
        if not isinstance(usage, dict):
            continue

        has_any = True
        merged["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        merged["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        merged["total_tokens"] += usage.get("total_tokens", 0) or 0

        for k, v in usage.items():
            if k in merged:
                continue
            if isinstance(v, (int, float)):
                merged[k] = merged.get(k, 0) + v

    return merged if has_any else {}


def extract_entities_for_full_sentence(client, text, max_len=60):
    chunks = split_text_with_offsets(text, max_len=max_len)

    all_entities = []
    raw_results = []

    for chunk in chunks:
        sub_text = chunk["sub_text"]
        offset = chunk["offset"]

        ner_result, raw_result = extract_entities_from_sub_text(client, sub_text)

        mapped_entities = remap_entities_to_original(
            ner_result.get("entities", []),
            offset
        )

        all_entities.extend(mapped_entities)
        raw_results.append({
            "sub_text": sub_text,
            "offset": offset,
            "raw_result": raw_result
        })

    final_result = {
        "text": text,
        "entities": deduplicate_entities(all_entities)
    }

    return final_result, raw_results


def process_one_sample(sample):
    """
    单句处理函数，供线程池调用
    """
    client = GLMClient()
    text = sample["text"]

    try:
        ner_result, raw_results = extract_entities_for_full_sentence(
            client,
            text,
            max_len=60
        )

        result_item = {
            "provider": LLM_PROVIDER,
            "model_name": CURRENT_LLM_CONFIG["model_name"],
            "file_name": sample["file_name"],
            "doc_idx": sample["doc_idx"],
            "para_idx": sample["para_idx"],
            "sent_idx": sample["sent_idx"],
            "text": text,
            "pred_entities": ner_result.get("entities", []),
            "gold_entities": sample.get("gold_entities", []),
            "usage": merge_usage(raw_results)
        }
        return result_item, None

    except Exception as e:
        result_item = {
            "provider": LLM_PROVIDER,
            "model_name": CURRENT_LLM_CONFIG["model_name"],
            "file_name": sample["file_name"],
            "doc_idx": sample["doc_idx"],
            "para_idx": sample["para_idx"],
            "sent_idx": sample["sent_idx"],
            "text": text,
            "pred_entities": [],
            "gold_entities": sample.get("gold_entities", []),
            "error": repr(e)
        }
        return result_item, repr(e)


def main():
    ensure_output_dir(OUTPUT_DIR)

    provider = LLM_PROVIDER
    model_name = CURRENT_LLM_CONFIG["model_name"]

    timestamped_path, latest_model_path, legacy_path = build_output_paths()
    init_output_files([timestamped_path, latest_model_path, legacy_path])

    samples = load_all_sentences(
        DATA_DIR,
        max_samples=None
    )

    print(f"当前 provider: {provider}")
    print(f"当前 model_name: {model_name}")
    print(f"线程池并发数: {MAX_WORKERS}")
    print(f"实际请求并发上限: {MAX_INFLIGHT_REQUESTS}")
    print(f"单请求最大重试次数: {MAX_RETRIES}")
    print(f"共加载 {len(samples)} 个原始句子，开始进行 GLM NER 批量抽取...")

    error_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_sample, sample) for sample in samples]

        for future in tqdm(as_completed(futures), total=len(futures), desc="NER extracting"):
            result_item, error = future.result()

            if error is not None:
                error_count += 1
                print(f"\n[ERROR] text={result_item['text']}")
                print(f"[ERROR] {error}")

            append_jsonl([timestamped_path, latest_model_path, legacy_path], result_item)

    print("\n抽取完成。")
    print(f"provider: {provider}")
    print(f"model_name: {model_name}")
    print(f"报错数: {error_count}")
    print(f"时间戳结果文件: {timestamped_path}")
    print(f"当前模型 latest: {latest_model_path}")
    print(f"兼容旧脚本文件: {legacy_path}")


if __name__ == "__main__":
    main()
