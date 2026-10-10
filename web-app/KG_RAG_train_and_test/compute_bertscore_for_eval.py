#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate BERTScore based on saved QA results.

Input:
  final_qa_eval_results.csv

Required columns:
  method
  model
  question
  gold_answer
  pred_answer

Output:
  final_qa_eval_results_with_bertscore.csv
  final_bertscore_average_results.xlsx

Install:
  pip install bert-score pandas openpyxl tqdm torch transformers

Run:
  python eval_bertscore_from_qa_csv.py
"""

import os
import re
import math
from typing import List

import pandas as pd
from tqdm import tqdm
from bert_score import score as bert_score


CONFIG = {
    # 输入：前一个主程序生成的逐条问答结果
    "INPUT_QA_CSV": "final_qa_eval_results.csv",

    # 输出：带逐条 BERTScore 的 CSV
    "OUTPUT_QA_CSV": "final_qa_eval_results_with_bertscore.csv",

    # 输出：按 method/model 汇总的 Excel
    "OUTPUT_AVERAGE_EXCEL": "final_bertscore_average_results.xlsx",

    # 中文建议使用 bert-base-chinese，也可以改为 hfl/chinese-roberta-wwm-ext
    "BERTSCORE_MODEL": "bert-base-chinese",

    # 是否使用 IDF 加权
    # 数据量较大时可以设为 True，但会更慢
    "USE_IDF": False,

    # 批大小，显存小就调低
    "BATCH_SIZE": 32,

    # 语言
    "LANG": "zh",

    # 是否重缩放到官方 baseline
    # 对中文 bert-base-chinese 通常可以 False
    "RESCALE_WITH_BASELINE": False,

    # 是否保存每条样本的 P/R/F1
    "SAVE_DETAIL": True,
}


def normalize_text(text) -> str:
    """
    清洗文本，避免 NaN、None 或多余空白影响评估。
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip())


def is_valid_pair(pred: str, gold: str) -> bool:
    """
    判断一条样本是否可以计算 BERTScore。
    """
    pred = normalize_text(pred)
    gold = normalize_text(gold)

    if not pred or not gold:
        return False

    if pred.startswith("ERROR:"):
        return False

    return True


def compute_bertscore_batch(
    preds: List[str],
    refs: List[str],
):
    """
    批量计算 BERTScore。
    """
    P, R, F1 = bert_score(
        cands=preds,
        refs=refs,
        model_type=CONFIG["BERTSCORE_MODEL"],
        lang=CONFIG["LANG"],
        idf=CONFIG["USE_IDF"],
        batch_size=CONFIG["BATCH_SIZE"],
        rescale_with_baseline=CONFIG["RESCALE_WITH_BASELINE"],
        verbose=True,
    )

    return (
        P.detach().cpu().numpy().tolist(),
        R.detach().cpu().numpy().tolist(),
        F1.detach().cpu().numpy().tolist(),
    )


def main():
    if not os.path.exists(CONFIG["INPUT_QA_CSV"]):
        raise FileNotFoundError(
            f"Input file not found: {CONFIG['INPUT_QA_CSV']}"
        )

    df = pd.read_csv(CONFIG["INPUT_QA_CSV"], encoding="utf-8-sig")

    required_cols = [
        "method",
        "model",
        "question",
        "gold_answer",
        "pred_answer",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in input CSV: {missing_cols}"
        )

    df["gold_answer"] = df["gold_answer"].apply(normalize_text)
    df["pred_answer"] = df["pred_answer"].apply(normalize_text)

    df["bertscore_precision"] = 0.0
    df["bertscore_recall"] = 0.0
    df["bertscore_f1"] = 0.0
    df["bertscore_valid"] = False

    valid_indices = []
    preds = []
    refs = []

    for idx, row in df.iterrows():
        pred = normalize_text(row["pred_answer"])
        gold = normalize_text(row["gold_answer"])

        if is_valid_pair(pred, gold):
            valid_indices.append(idx)
            preds.append(pred)
            refs.append(gold)

    print(f"Total rows: {len(df)}")
    print(f"Valid rows for BERTScore: {len(valid_indices)}")
    print(f"Invalid rows skipped: {len(df) - len(valid_indices)}")
    print(f"BERTScore model: {CONFIG['BERTSCORE_MODEL']}")

    if valid_indices:
        p_list, r_list, f1_list = compute_bertscore_batch(preds, refs)

        for idx, p, r, f1 in tqdm(
            zip(valid_indices, p_list, r_list, f1_list),
            total=len(valid_indices),
            desc="Writing BERTScore",
        ):
            df.at[idx, "bertscore_precision"] = float(p)
            df.at[idx, "bertscore_recall"] = float(r)
            df.at[idx, "bertscore_f1"] = float(f1)
            df.at[idx, "bertscore_valid"] = True

    # 保存逐条结果
    if CONFIG["SAVE_DETAIL"]:
        df.to_csv(
            CONFIG["OUTPUT_QA_CSV"],
            index=False,
            encoding="utf-8-sig",
        )

    # 只对有效样本做平均
    valid_df = df[df["bertscore_valid"] == True].copy()

    if valid_df.empty:
        raise RuntimeError("No valid rows for BERTScore evaluation.")

    avg_by_method_model = (
        valid_df.groupby(["method", "model"], as_index=False)
        .agg(
            bertscore_precision=("bertscore_precision", "mean"),
            bertscore_recall=("bertscore_recall", "mean"),
            bertscore_f1=("bertscore_f1", "mean"),
            valid_sample_count=("bertscore_valid", "sum"),
        )
        .sort_values(["method", "model"])
    )

    avg_by_method = (
        valid_df.groupby(["method"], as_index=False)
        .agg(
            bertscore_precision=("bertscore_precision", "mean"),
            bertscore_recall=("bertscore_recall", "mean"),
            bertscore_f1=("bertscore_f1", "mean"),
            valid_sample_count=("bertscore_valid", "sum"),
        )
        .sort_values("method")
    )

    avg_by_model = (
        valid_df.groupby(["model"], as_index=False)
        .agg(
            bertscore_precision=("bertscore_precision", "mean"),
            bertscore_recall=("bertscore_recall", "mean"),
            bertscore_f1=("bertscore_f1", "mean"),
            valid_sample_count=("bertscore_valid", "sum"),
        )
        .sort_values("model")
    )

    readme = pd.DataFrame([
        {
            "item": "说明",
            "value": "本程序基于 final_qa_eval_results.csv 中保存的逐条问答结果计算 BERTScore。",
        },
        {
            "item": "输入文件",
            "value": CONFIG["INPUT_QA_CSV"],
        },
        {
            "item": "逐条输出文件",
            "value": CONFIG["OUTPUT_QA_CSV"],
        },
        {
            "item": "平均结果文件",
            "value": CONFIG["OUTPUT_AVERAGE_EXCEL"],
        },
        {
            "item": "使用模型",
            "value": CONFIG["BERTSCORE_MODEL"],
        },
        {
            "item": "指标说明",
            "value": "bertscore_precision、bertscore_recall、bertscore_f1 分别表示语义精确率、语义召回率和语义 F1。",
        },
        {
            "item": "有效样本",
            "value": "空答案、空标准答案、ERROR 开头的模型回答不会参与 BERTScore 平均计算。",
        },
        {
            "item": "建议",
            "value": "论文或实验表格中通常报告 bertscore_f1。",
        },
    ])

    with pd.ExcelWriter(CONFIG["OUTPUT_AVERAGE_EXCEL"], engine="openpyxl") as writer:
        avg_by_method_model.to_excel(
            writer,
            sheet_name="Average_By_Method_Model",
            index=False,
        )
        avg_by_method.to_excel(
            writer,
            sheet_name="Average_By_Method",
            index=False,
        )
        avg_by_model.to_excel(
            writer,
            sheet_name="Average_By_Model",
            index=False,
        )
        readme.to_excel(
            writer,
            sheet_name="Readme",
            index=False,
        )

    print()
    print("===== Average BERTScore By Method + Model =====")
    print(avg_by_method_model)
    print()
    print("===== Average BERTScore By Method =====")
    print(avg_by_method)
    print()
    print("===== Average BERTScore By Model =====")
    print(avg_by_model)
    print()
    print(f"Saved detailed QA+BERTScore results to: {CONFIG['OUTPUT_QA_CSV']}")
    print(f"Saved average BERTScore results to: {CONFIG['OUTPUT_AVERAGE_EXCEL']}")


if __name__ == "__main__":
    main()
