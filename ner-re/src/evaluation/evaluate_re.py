import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score

import config.config as config
from data.data_loader import DiaKG_RE_Dataset

# 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def evaluate():
    model_path = os.path.join(config.OUTPUT_DIR, "re_model")
    data_dir = config.DATA_DIR
    
    print(f"\n{'='*20} 正在评估 RE 模型 {'='*20}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print(f"Loading RE model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请检查是否已运行 python train_re.py 完成训练。")
        return

    # 2. 准备数据
    label2id = model.config.label2id
    # 确保 id2label 的 key 是 int 类型 (加载 config 时可能是 string)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    
    print("Loading test data (Positive + Merged + Negative samples)...")
    full_dataset = DiaKG_RE_Dataset(data_dir, tokenizer, label2id, config.RE_MAX_LEN)
    train_size = int(0.9 * len(full_dataset))
    test_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"测试集样本数: {len(test_dataset)}")

    # 3. 推理
    y_true = []
    y_pred = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # 4. 生成报告
    # [关键修复] 显式指定 labels 列表，确保包含所有类别
    all_labels_ids = sorted(list(id2label.keys()))
    target_names = [id2label[i] for i in all_labels_ids]
    
    print("\n" + "="*20 + " RE 评估结果 " + "="*20)
    # 传入 labels 参数，强制评估所有类别
    print(classification_report(
        y_true, 
        y_pred, 
        labels=all_labels_ids, 
        target_names=target_names, 
        digits=4, 
        zero_division=0
    ))
    print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("="*55)

if __name__ == "__main__":
    evaluate()
