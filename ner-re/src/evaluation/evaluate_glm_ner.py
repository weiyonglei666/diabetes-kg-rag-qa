import os
import json
from collections import defaultdict
from datetime import datetime

from config.config import OUTPUT_DIR, LLM_PROVIDER, CURRENT_LLM_CONFIG


def normalize_entity(ent):
    """
    将实体统一为可比较的 tuple
    """
    return (
        ent.get("entity", "").strip(),
        ent.get("entity_type", "").strip(),
        int(ent.get("start_idx", -1)),
        int(ent.get("end_idx", -1)),
    )


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) != 0 else 0.0


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def evaluate_ner(results):
    total_tp = 0
    total_pred = 0
    total_gold = 0
    total_errors = 0

    # 按类别统计
    type_stats = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})

    error_cases = []

    for item in results:
        text = item.get("text", "")
        pred_entities = item.get("pred_entities", [])
        gold_entities = item.get("gold_entities", [])

        if "error" in item:
            total_errors += 1

        pred_set = set(normalize_entity(e) for e in pred_entities)
        gold_set = set(normalize_entity(e) for e in gold_entities)

        tp_set = pred_set & gold_set
        pred_only = pred_set - gold_set
        gold_only = gold_set - pred_set

        total_tp += len(tp_set)
        total_pred += len(pred_set)
        total_gold += len(gold_set)

        # 分类统计
        for ent in pred_set:
            ent_type = ent[1]
            type_stats[ent_type]["pred"] += 1

        for ent in gold_set:
            ent_type = ent[1]
            type_stats[ent_type]["gold"] += 1

        for ent in tp_set:
            ent_type = ent[1]
            type_stats[ent_type]["tp"] += 1

        # 错误样例
        if pred_only or gold_only:
            error_cases.append({
                "text": text,
                "pred_only": [list(x) for x in sorted(pred_only, key=lambda x: (x[2], x[3], x[1], x[0]))],
                "gold_only": [list(x) for x in sorted(gold_only, key=lambda x: (x[2], x[3], x[1], x[0]))],
            })

    overall_precision = safe_div(total_tp, total_pred)
    overall_recall = safe_div(total_tp, total_gold)
    overall_f1 = f1_score(overall_precision, overall_recall)

    per_type_metrics = {}
    for ent_type, stats in sorted(type_stats.items()):
        p = safe_div(stats["tp"], stats["pred"])
        r = safe_div(stats["tp"], stats["gold"])
        f1 = f1_score(p, r)
        per_type_metrics[ent_type] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": stats["tp"],
            "pred": stats["pred"],
            "gold": stats["gold"],
        }

    summary = {
        "provider": LLM_PROVIDER,
        "model_name": CURRENT_LLM_CONFIG["model_name"],
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_sentences": len(results),
        "num_errors": total_errors,
        "overall": {
            "tp": total_tp,
            "pred": total_pred,
            "gold": total_gold,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        },
        "per_type": per_type_metrics,
        "error_cases": error_cases,
    }

    return summary


def print_summary(summary, max_error_cases=10):
    overall = summary["overall"]
    per_type = summary["per_type"]
    error_cases = summary["error_cases"]

    print("=" * 60)
    print(f"【模型信息】")
    print(f"provider:   {summary['provider']}")
    print(f"model_name: {summary['model_name']}")
    print(f"evaluated_at: {summary['evaluated_at']}")
    print("=" * 60)

    print("【整体指标】")
    print(f"句子数: {summary['num_sentences']}")
    print(f"报错数: {summary['num_errors']}")
    print(f"TP: {overall['tp']}")
    print(f"Pred: {overall['pred']}")
    print(f"Gold: {overall['gold']}")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1:        {overall['f1']:.4f}")
    print("=" * 60)

    print("【分类型指标】")
    for ent_type, m in per_type.items():
        print(
            f"{ent_type:<15} "
            f"P={m['precision']:.4f} "
            f"R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} "
            f"(TP={m['tp']}, Pred={m['pred']}, Gold={m['gold']})"
        )

    print("=" * 60)
    print(f"【前{min(max_error_cases, len(error_cases))}个错误样例】")
    print()

    for i, case in enumerate(error_cases[:max_error_cases], 1):
        print(f"[{i}] text: {case['text']}")
        print(f"pred_only: {case['pred_only']}")
        print(f"gold_only: {case['gold_only']}")
        print()


def save_summary(summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    provider = summary["provider"]
    model_name = summary["model_name"].replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 带时间戳的独立评测结果
    summary_path = os.path.join(
        OUTPUT_DIR,
        f"ner_eval_{provider}_{model_name}_{timestamp}.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 2. 当前模型的 latest 结果
    latest_path = os.path.join(
        OUTPUT_DIR,
        f"ner_eval_latest_{provider}_{model_name}.json"
    )

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 3. 汇总到一个 leaderboard jsonl，方便后续多模型对比
    leaderboard_item = {
        "provider": summary["provider"],
        "model_name": summary["model_name"],
        "evaluated_at": summary["evaluated_at"],
        "num_sentences": summary["num_sentences"],
        "num_errors": summary["num_errors"],
        "tp": summary["overall"]["tp"],
        "pred": summary["overall"]["pred"],
        "gold": summary["overall"]["gold"],
        "precision": summary["overall"]["precision"],
        "recall": summary["overall"]["recall"],
        "f1": summary["overall"]["f1"],
        "summary_file": os.path.basename(summary_path),
    }

    leaderboard_path = os.path.join(OUTPUT_DIR, "ner_eval_leaderboard.jsonl")
    with open(leaderboard_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(leaderboard_item, ensure_ascii=False) + "\n")

    return summary_path, latest_path, leaderboard_path


def main():
    # results_path = os.path.join(OUTPUT_DIR, "glm_ner_results.jsonl")
    results_path = os.path.join(
        OUTPUT_DIR,
        f"glm_ner_results_latest_{LLM_PROVIDER}_{CURRENT_LLM_CONFIG['model_name'].replace('/', '_')}.jsonl"
    )

    if not os.path.exists(results_path):
        print(f"找不到结果文件: {results_path}")
        return

    results = load_jsonl(results_path)
    if not results:
        print(f"结果文件为空: {results_path}")
        return

    summary = evaluate_ner(results)
    print_summary(summary)

    summary_path, latest_path, leaderboard_path = save_summary(summary)

    print("=" * 60)
    print("【结果已保存】")
    print(f"详细评测结果: {summary_path}")
    print(f"当前模型 latest: {latest_path}")
    print(f"汇总榜单文件: {leaderboard_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
