import os
import pandas as pd
from tqdm import tqdm
from bert_score import score


# =========================
# 配置
# =========================
INPUT_CSV = "llm_eval_async.csv"                   # 你的原始评测结果
OUTPUT_DETAIL_CSV = "bertscore_detail.csv"         # 明细结果
OUTPUT_SUMMARY_CSV = "bertscore_summary.csv"       # 按模型汇总结果

BATCH_SIZE = 32
LANG = "zh"                                        # 中文任务建议 zh
MODEL_TYPE = None                                  # 可留空，bert-score 会自动选
USE_IDF = False
DEVICE = "cuda"                                    # 没有 GPU 就改成 "cpu"


# =========================
# 自动识别列名
# =========================
PRED_CANDIDATES = ["pred", "pred_answer", "prediction", "answer"]
GOLD_CANDIDATES = ["gold", "gold_answer", "reference", "label"]
MODEL_CANDIDATES = ["model", "model_name"]


def find_first_existing(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"找不到列，候选列为: {candidates}")
    return None


# =========================
# 主逻辑
# =========================
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    pred_col = find_first_existing(df, PRED_CANDIDATES, required=True)
    gold_col = find_first_existing(df, GOLD_CANDIDATES, required=True)
    model_col = find_first_existing(df, MODEL_CANDIDATES, required=False)

    # 清洗
    df[pred_col] = df[pred_col].fillna("").astype(str)
    df[gold_col] = df[gold_col].fillna("").astype(str)

    preds = df[pred_col].tolist()
    golds = df[gold_col].tolist()

    print("开始计算 BERTScore...")
    print(f"样本数: {len(df)}")
    print(f"预测列: {pred_col}")
    print(f"参考列: {gold_col}")
    print(f"模型列: {model_col if model_col else '无'}")
    print(f"device: {DEVICE}")

    # 计算 BERTScore
    # rescale_with_baseline=False 更直观；如需更稳定可改 True
    P, R, F1 = score(
        cands=preds,
        refs=golds,
        lang=LANG,
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        verbose=True,
        device=DEVICE,
        idf=USE_IDF,
        rescale_with_baseline=False,
    )

    df["bertscore_precision"] = [float(x) for x in P]
    df["bertscore_recall"] = [float(x) for x in R]
    df["bertscore_f1"] = [float(x) for x in F1]

    # 保存明细
    df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")

    print("\n===== Overall BERTScore =====")
    print(f"Precision: {df['bertscore_precision'].mean():.6f}")
    print(f"Recall:    {df['bertscore_recall'].mean():.6f}")
    print(f"F1:        {df['bertscore_f1'].mean():.6f}")

    # 按模型汇总
    if model_col:
        summary = (
            df.groupby(model_col)[
                ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
            ]
            .mean()
            .sort_values(by="bertscore_f1", ascending=False)
            .reset_index()
        )

        summary.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

        print("\n===== BERTScore Summary by Model =====")
        print(summary.to_string(index=False))
        print(f"\n已保存汇总文件: {OUTPUT_SUMMARY_CSV}")

    print(f"已保存明细文件: {OUTPUT_DETAIL_CSV}")


if __name__ == "__main__":
    main()
