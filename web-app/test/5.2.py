from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 中文字体设置
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 数据重建 (根据原图片可视化信息手动输入)
methods = ["KG-only", "LLM-only", "LLM+KG", "LLM+RAG", "KG+RAG+LLM"]
bert_f1 = [0.60, 0.44, 0.41, 0.42, 0.43]
explainability = [0.91, 0.0, 0.16, 0.21, 0.35]
rouge_l = [0.02, 0.04, 0.02, 0.02, 0.02]
response_time = [0.0, 20.1, 16.1, 14.0, 17.2]

# 热力图数据
models = ["deepseek-r1", "deepseek-v3", "glm-5.1", "kimi-2.5", "qwen-3.6"]
heatmap_data = np.array([
    [0.706, 0.705, 0.055, 0.050, 0.685],
    [0.676, 0.676, 0.029, 0.002, 0.684],
    [0.679, 0.680, 0.046, 0.023, 0.684],
    [0.694, 0.700, 0.027, 0.025, 0.691]
])

# 1. 平均分条形图
fig1, ax1 = plt.subplots(figsize=(8,4))
width = 0.2
x = np.arange(len(methods))
ax1.bar(x - width, bert_f1, width, label='BERTScore F1')
ax1.bar(x, explainability, width, label='可解释性')
ax1.bar(x + width, rouge_l, width, label='ROUGE-L')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylabel('分数')
ax1.set_title('不同方法的平均分')
ax1.legend()
fig1.tight_layout()
fig1.savefig('avg_scores_cn.svg', format='svg')

# 2. 响应时间条形图
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.bar(methods, response_time, color='steelblue')
ax2.set_ylabel('秒')
ax2.set_title('不同方法的平均响应时间')
fig2.tight_layout()
fig2.savefig('response_time_cn.svg', format='svg')

# 3. BERTScore F1 热力图
fig3, ax3 = plt.subplots(figsize=(8,4))
cax = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=0.8)
ax3.set_yticks(np.arange(len(methods[1:])))  # LLM-only -> KG+RAG+LLM
ax3.set_yticklabels(methods[1:])
ax3.set_xticks(np.arange(len(models)))
ax3.set_xticklabels(models, rotation=30)
ax3.set_title('不同方法与模型的 BERTScore F1')
# 显示热力值
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        ax3.text(j, i, f"{heatmap_data[i,j]:.3f}", ha='center', va='center', color='white' if heatmap_data[i,j]<0.4 else 'black')
fig3.colorbar(cax, ax=ax3, label='BERTScore F1')
fig3.tight_layout()
fig3.savefig('bert_f1_heatmap_cn.svg', format='svg')

# 输出路径
[
    'avg_scores_cn.svg',
    'response_time_cn.svg',
    'bert_f1_heatmap_cn.svg'
]
