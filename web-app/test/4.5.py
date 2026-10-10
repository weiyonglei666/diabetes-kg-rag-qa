import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 中文显示设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 使 SVG 中的文字保持为可编辑文本
plt.rcParams["svg.fonttype"] = "none"

# 输出目录
output_dir = Path("./figures")
output_dir.mkdir(exist_ok=True)


def save_grouped_bar_svg(filename, labels, metrics, title, ylabel="指标值"):
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))

    metric_names = list(metrics.keys())
    offsets = np.linspace(
        -width * (len(metric_names) - 1) / 2,
        width * (len(metric_names) - 1) / 2,
        len(metric_names)
    )

    for offset, name in zip(offsets, metric_names):
        values = metrics[name]
        bars = ax.bar(x + offset, values, width, label=name)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.70, 1.00)
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_dir / filename, format="svg")
    plt.close(fig)


# 图4-5 不同模型在命名实体识别任务上的性能对比图
ner_labels = ["MacBERT-CRF", "Chinese-RoBERTa-wwm-ext", "PubMedBERT"]
ner_metrics = {
    "精确率": [0.925, 0.870, 0.850],
    "召回率": [0.899, 0.909, 0.822],
    "F1值": [0.912, 0.889, 0.836],
}

save_grouped_bar_svg(
    filename="图4-5_命名实体识别模型性能对比.svg",
    labels=ner_labels,
    metrics=ner_metrics,
    title="不同模型在命名实体识别任务上的性能对比"
)


# 图4-6 不同模型在关系抽取任务上的性能对比图
re_labels = ["DREEAM", "Chinese-RoBERTa-wwm-ext", "BioLinkBERT"]
re_metrics = {
    "精确率": [0.952, 0.958, 0.830],
    "召回率": [0.936, 0.976, 0.854],
    "F1值": [0.944, 0.967, 0.842],
}

save_grouped_bar_svg(
    filename="图4-6_关系抽取模型性能对比.svg",
    labels=re_labels,
    metrics=re_metrics,
    title="不同模型在关系抽取任务上的性能对比"
)


# 图4-7 大语言模型微调前后 F1-score 对比图
llm_labels = [
    "GLM-4-9B\n微调前",
    "GLM-4-9B\n微调后",
    "Qwen3-8B\n微调前",
    "Qwen3-8B\n微调后"
]

llm_ner_f1 = [0.761, 0.842, 0.724, 0.816]
llm_re_f1 = [0.739, 0.904, 0.743, 0.871]

save_grouped_bar_svg(
    filename="图4-7_大语言模型微调前后F1值对比.svg",
    labels=llm_labels,
    metrics={
        "实体识别F1值": llm_ner_f1,
        "关系抽取F1值": llm_re_f1,
    },
    title="大语言模型微调前后F1值对比"
)


# 图4-8 传统模型与大语言模型微调后最优 F1-score 对比图
best_labels = ["实体识别", "关系抽取"]

save_grouped_bar_svg(
    filename="图4-8_传统模型与大语言模型最优F1值对比.svg",
    labels=best_labels,
    metrics={
        "传统模型": [0.912, 0.967],
        "微调后大语言模型": [0.842, 0.904],
    },
    title="传统模型与微调后大语言模型最优F1值对比"
)

print(f"SVG图已保存至：{output_dir.resolve()}")
