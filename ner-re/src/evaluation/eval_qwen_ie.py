# -*- coding: utf-8 -*-
"""
评估 Qwen / LLM 信息抽取结果
支持:
1. NER strict: entity + entity_type + start_idx + end_idx 全匹配
2. NER soft: entity + entity_type 匹配
3. RE strict: head/tail entity span/type + relation_type 全匹配

适配 DiaKG 风格:
{
  "paragraphs": [
    {
      "sentences": [
        {
          "sentence": "...",
          "entities": [
            {
              "entity_id": "1",
              "entity": "阿司匹林",
              "entity_type": "Drug",
              "start_idx": 4,
              "end_idx": 8
            }
          ],
          "relations": [
            {
              "relation_id": "1",
              "relation_type": "Drug_Disease",
              "head_entity_id": "1",
              "tail_entity_id": "2"
            }
          ]
        }
      ]
    }
  ]
}

用法:
python eval_qwen_ie.py \
    --gold_dir ../diakg_data \
    --pred_dir ../output_qwen3_pipeline/rebuilt

可选:
python eval_qwen_ie.py \
    --gold_dir ../diakg_data \
    --pred_dir ../output_qwen3_pipeline/rebuilt \
    --save_json ../output_qwen3_pipeline/eval_result.json
"""

import os
import json
import argparse
from collections import defaultdict


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_sentences(raw_data):
    """
    兼容:
    1) list[doc]
    2) {"data": [doc, ...]}
    3) 单个 doc(dict)
    """
    if isinstance(raw_data, list):
        docs = raw_data
    elif isinstance(raw_data, dict) and "data" in raw_data:
        docs = raw_data["data"]
    else:
        docs = [raw_data]

    for doc in docs:
        for para in doc.get("paragraphs", []):
            for sent in para.get("sentences", []):
                yield sent


def safe_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return default


def normalize_text(x):
    return str(x).strip()


def entity_key_strict(ent):
    return (
        normalize_text(ent.get("entity", "")),
        normalize_text(ent.get("entity_type", "")),
        safe_int(ent.get("start_idx", -1)),
        safe_int(ent.get("end_idx", -1)),
    )


def entity_key_soft(ent):
    return (
        normalize_text(ent.get("entity", "")),
        normalize_text(ent.get("entity_type", "")),
    )


def build_entity_id_map(sentence):
    """
    返回:
    {
      entity_id: {
        "entity": ...,
        "entity_type": ...,
        "start_idx": ...,
        "end_idx": ...
      }
    }
    """
    out = {}
    for ent in sentence.get("entities", []):
        ent_id = normalize_text(ent.get("entity_id", ""))
        if not ent_id:
            continue
        out[ent_id] = {
            "entity": normalize_text(ent.get("entity", "")),
            "entity_type": normalize_text(ent.get("entity_type", "")),
            "start_idx": safe_int(ent.get("start_idx", -1)),
            "end_idx": safe_int(ent.get("end_idx", -1)),
        }
    return out


def relation_key_strict(sentence):
    """
    用关系引用到的 head_entity_id / tail_entity_id
    还原成完整 strict key:
    (
      head_text, head_type, head_start, head_end,
      tail_text, tail_type, tail_start, tail_end,
      relation_type
    )
    """
    ent_map = build_entity_id_map(sentence)
    keys = set()

    for rel in sentence.get("relations", []):
        h_id = normalize_text(rel.get("head_entity_id", ""))
        t_id = normalize_text(rel.get("tail_entity_id", ""))
        rel_type = normalize_text(rel.get("relation_type", ""))

        if h_id not in ent_map or t_id not in ent_map:
            continue

        h = ent_map[h_id]
        t = ent_map[t_id]

        keys.add((
            h["entity"], h["entity_type"], h["start_idx"], h["end_idx"],
            t["entity"], t["entity_type"], t["start_idx"], t["end_idx"],
            rel_type
        ))

    return keys


def calc_prf(tp, pred_n, gold_n):
    p = tp / pred_n if pred_n > 0 else 0.0
    r = tp / gold_n if gold_n > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def pct(x):
    return round(x * 100, 2)


def evaluate(gold_dir, pred_dir):
    gold_files = sorted([f for f in os.listdir(gold_dir) if f.endswith(".json")])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".json")])

    pred_set = set(pred_files)
    common_files = [f for f in gold_files if f in pred_set]

    if not common_files:
        raise ValueError("gold_dir 和 pred_dir 没有同名 json 文件")

    ner_strict_tp = 0
    ner_strict_pred = 0
    ner_strict_gold = 0

    ner_soft_tp = 0
    ner_soft_pred = 0
    ner_soft_gold = 0

    re_strict_tp = 0
    re_strict_pred = 0
    re_strict_gold = 0

    ner_type_stats = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})
    re_type_stats = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})

    file_stats = []
    mismatched_files = []

    total_sent_pairs = 0

    for fname in common_files:
        gold_data = read_json(os.path.join(gold_dir, fname))
        pred_data = read_json(os.path.join(pred_dir, fname))

        gold_sents = list(iter_sentences(gold_data))
        pred_sents = list(iter_sentences(pred_data))

        if len(gold_sents) != len(pred_sents):
            mismatched_files.append({
                "file_name": fname,
                "gold_sent_count": len(gold_sents),
                "pred_sent_count": len(pred_sents)
            })

        pair_num = min(len(gold_sents), len(pred_sents))
        total_sent_pairs += pair_num

        f_ner_strict_tp = 0
        f_ner_strict_pred = 0
        f_ner_strict_gold = 0

        f_ner_soft_tp = 0
        f_ner_soft_pred = 0
        f_ner_soft_gold = 0

        f_re_tp = 0
        f_re_pred = 0
        f_re_gold = 0

        for i in range(pair_num):
            gold_sent = gold_sents[i]
            pred_sent = pred_sents[i]

            gold_ents = gold_sent.get("entities", [])
            pred_ents = pred_sent.get("entities", [])

            gold_ner_strict = set(entity_key_strict(e) for e in gold_ents)
            pred_ner_strict = set(entity_key_strict(e) for e in pred_ents)

            gold_ner_soft = set(entity_key_soft(e) for e in gold_ents)
            pred_ner_soft = set(entity_key_soft(e) for e in pred_ents)

            strict_tp = len(gold_ner_strict & pred_ner_strict)
            soft_tp = len(gold_ner_soft & pred_ner_soft)

            ner_strict_tp += strict_tp
            ner_strict_pred += len(pred_ner_strict)
            ner_strict_gold += len(gold_ner_strict)

            ner_soft_tp += soft_tp
            ner_soft_pred += len(pred_ner_soft)
            ner_soft_gold += len(gold_ner_soft)

            f_ner_strict_tp += strict_tp
            f_ner_strict_pred += len(pred_ner_strict)
            f_ner_strict_gold += len(gold_ner_strict)

            f_ner_soft_tp += soft_tp
            f_ner_soft_pred += len(pred_ner_soft)
            f_ner_soft_gold += len(gold_ner_soft)

            # 实体分类别统计（strict）
            gold_ent_by_type = defaultdict(set)
            pred_ent_by_type = defaultdict(set)

            for e in gold_ents:
                gold_ent_by_type[normalize_text(e.get("entity_type", ""))].add(entity_key_strict(e))
            for e in pred_ents:
                pred_ent_by_type[normalize_text(e.get("entity_type", ""))].add(entity_key_strict(e))

            all_ent_types = set(gold_ent_by_type.keys()) | set(pred_ent_by_type.keys())
            for t in all_ent_types:
                g = gold_ent_by_type[t]
                p = pred_ent_by_type[t]
                ner_type_stats[t]["tp"] += len(g & p)
                ner_type_stats[t]["pred"] += len(p)
                ner_type_stats[t]["gold"] += len(g)

            # RE strict
            gold_re = relation_key_strict(gold_sent)
            pred_re = relation_key_strict(pred_sent)

            tp_re = len(gold_re & pred_re)

            re_strict_tp += tp_re
            re_strict_pred += len(pred_re)
            re_strict_gold += len(gold_re)

            f_re_tp += tp_re
            f_re_pred += len(pred_re)
            f_re_gold += len(gold_re)

            # 关系分类别统计
            gold_re_by_type = defaultdict(set)
            pred_re_by_type = defaultdict(set)

            for r in gold_re:
                gold_re_by_type[r[-1]].add(r)
            for r in pred_re:
                pred_re_by_type[r[-1]].add(r)

            all_rel_types = set(gold_re_by_type.keys()) | set(pred_re_by_type.keys())
            for t in all_rel_types:
                g = gold_re_by_type[t]
                p = pred_re_by_type[t]
                re_type_stats[t]["tp"] += len(g & p)
                re_type_stats[t]["pred"] += len(p)
                re_type_stats[t]["gold"] += len(g)

        f_ns_p, f_ns_r, f_ns_f1 = calc_prf(f_ner_strict_tp, f_ner_strict_pred, f_ner_strict_gold)
        f_nf_p, f_nf_r, f_nf_f1 = calc_prf(f_ner_soft_tp, f_ner_soft_pred, f_ner_soft_gold)
        f_re_p, f_re_r, f_re_f1 = calc_prf(f_re_tp, f_re_pred, f_re_gold)

        file_stats.append({
            "file_name": fname,
            "ner_strict": {
                "tp": f_ner_strict_tp,
                "pred": f_ner_strict_pred,
                "gold": f_ner_strict_gold,
                "precision": f_ns_p,
                "recall": f_ns_r,
                "f1": f_ns_f1,
            },
            "ner_soft": {
                "tp": f_ner_soft_tp,
                "pred": f_ner_soft_pred,
                "gold": f_ner_soft_gold,
                "precision": f_nf_p,
                "recall": f_nf_r,
                "f1": f_nf_f1,
            },
            "re_strict": {
                "tp": f_re_tp,
                "pred": f_re_pred,
                "gold": f_re_gold,
                "precision": f_re_p,
                "recall": f_re_r,
                "f1": f_re_f1,
            }
        })

    ner_strict_p, ner_strict_r, ner_strict_f1 = calc_prf(ner_strict_tp, ner_strict_pred, ner_strict_gold)
    ner_soft_p, ner_soft_r, ner_soft_f1 = calc_prf(ner_soft_tp, ner_soft_pred, ner_soft_gold)
    re_strict_p, re_strict_r, re_strict_f1 = calc_prf(re_strict_tp, re_strict_pred, re_strict_gold)

    result = {
        "summary": {
            "file_count": len(common_files),
            "sentence_pair_count": total_sent_pairs,
            "mismatched_file_count": len(mismatched_files),
            "ner_strict": {
                "tp": ner_strict_tp,
                "pred": ner_strict_pred,
                "gold": ner_strict_gold,
                "precision": ner_strict_p,
                "recall": ner_strict_r,
                "f1": ner_strict_f1
            },
            "ner_soft": {
                "tp": ner_soft_tp,
                "pred": ner_soft_pred,
                "gold": ner_soft_gold,
                "precision": ner_soft_p,
                "recall": ner_soft_r,
                "f1": ner_soft_f1
            },
            "re_strict": {
                "tp": re_strict_tp,
                "pred": re_strict_pred,
                "gold": re_strict_gold,
                "precision": re_strict_p,
                "recall": re_strict_r,
                "f1": re_strict_f1
            }
        },
        "entity_type_stats": {},
        "relation_type_stats": {},
        "file_stats": file_stats,
        "mismatched_files": mismatched_files
    }

    for t in sorted(ner_type_stats.keys()):
        s = ner_type_stats[t]
        p, r, f1 = calc_prf(s["tp"], s["pred"], s["gold"])
        result["entity_type_stats"][t] = {
            "tp": s["tp"],
            "pred": s["pred"],
            "gold": s["gold"],
            "precision": p,
            "recall": r,
            "f1": f1
        }

    for t in sorted(re_type_stats.keys()):
        s = re_type_stats[t]
        p, r, f1 = calc_prf(s["tp"], s["pred"], s["gold"])
        result["relation_type_stats"][t] = {
            "tp": s["tp"],
            "pred": s["pred"],
            "gold": s["gold"],
            "precision": p,
            "recall": r,
            "f1": f1
        }

    return result


def print_result(result):
    s = result["summary"]

    print("=" * 90)
    print("总体评测结果")
    print("=" * 90)
    print(f"文件数: {s['file_count']}")
    print(f"句子对数: {s['sentence_pair_count']}")
    print(f"句子数量不一致文件数: {s['mismatched_file_count']}")
    print("-" * 90)

    print("[NER-Strict]")
    print(
        f"TP={s['ner_strict']['tp']}, Pred={s['ner_strict']['pred']}, Gold={s['ner_strict']['gold']} | "
        f"P={pct(s['ner_strict']['precision'])} R={pct(s['ner_strict']['recall'])} F1={pct(s['ner_strict']['f1'])}"
    )
    print("-" * 90)

    print("[NER-Soft]")
    print(
        f"TP={s['ner_soft']['tp']}, Pred={s['ner_soft']['pred']}, Gold={s['ner_soft']['gold']} | "
        f"P={pct(s['ner_soft']['precision'])} R={pct(s['ner_soft']['recall'])} F1={pct(s['ner_soft']['f1'])}"
    )
    print("-" * 90)

    print("[RE-Strict]")
    print(
        f"TP={s['re_strict']['tp']}, Pred={s['re_strict']['pred']}, Gold={s['re_strict']['gold']} | "
        f"P={pct(s['re_strict']['precision'])} R={pct(s['re_strict']['recall'])} F1={pct(s['re_strict']['f1'])}"
    )
    print("=" * 90)

    print("\n实体分类别结果（Strict）")
    print("=" * 90)
    for t in sorted(result["entity_type_stats"].keys()):
        x = result["entity_type_stats"][t]
        print(
            f"{t:20s} | "
            f"P={pct(x['precision']):6.2f} "
            f"R={pct(x['recall']):6.2f} "
            f"F1={pct(x['f1']):6.2f} | "
            f"TP={x['tp']}, Pred={x['pred']}, Gold={x['gold']}"
        )

    print("\n关系分类别结果（Strict）")
    print("=" * 90)
    for t in sorted(result["relation_type_stats"].keys()):
        x = result["relation_type_stats"][t]
        print(
            f"{t:20s} | "
            f"P={pct(x['precision']):6.2f} "
            f"R={pct(x['recall']):6.2f} "
            f"F1={pct(x['f1']):6.2f} | "
            f"TP={x['tp']}, Pred={x['pred']}, Gold={x['gold']}"
        )

    if result["mismatched_files"]:
        print("\n句子数量不一致文件")
        print("=" * 90)
        for item in result["mismatched_files"]:
            print(
                f"{item['file_name']}: "
                f"gold={item['gold_sent_count']}, pred={item['pred_sent_count']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_dir", type=str, required=True, help="标注真值目录")
    parser.add_argument("--pred_dir", type=str, required=True, help="预测结果目录")
    parser.add_argument("--save_json", type=str, default="", help="可选，保存评测结果 JSON")
    args = parser.parse_args()

    result = evaluate(args.gold_dir, args.pred_dir)
    print_result(result)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n评测结果已保存到: {args.save_json}")


if __name__ == "__main__":
    main()
