import json
from collections import Counter
from config.config import OUTPUT_DIR
import os


def main():
    path = os.path.join(OUTPUT_DIR, "glm_ner_results.jsonl")
    if not os.path.exists(path):
        print(f"找不到结果文件: {path}")
        return

    entity_counter = Counter()
    error_count = 0
    total = 0
    error_examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            item = json.loads(line)

            if "error" in item:
                error_count += 1
                if len(error_examples) < 5:
                    error_examples.append({
                        "text": item.get("text", ""),
                        "error": item.get("error", "")
                    })

            for ent in item.get("pred_entities", []):
                entity_counter[ent["entity_type"]] += 1

    print("=" * 50)
    print(f"总句子数: {total}")
    print(f"报错句子数: {error_count}")
    print("预测实体分布 Top 20:")
    print(entity_counter.most_common(20))
    print("=" * 50)

    if error_examples:
        print("\n前 5 个报错样例：")
        for i, ex in enumerate(error_examples, 1):
            print(f"\n[{i}] text: {ex['text']}")
            print(f"error: {ex['error']}")


if __name__ == "__main__":
    main()
