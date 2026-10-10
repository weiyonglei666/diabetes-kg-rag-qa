import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score

import config.config as config
from data.data_loader import DiaKG_NER_Dataset

# 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def evaluate():
    model_path = os.path.join(config.OUTPUT_DIR, "ner_model")
    data_dir = config.DATA_DIR
    
    print(f"\n{'='*20} 正在评估 NER 模型 {'='*20}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    print(f"Loading NER model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请检查是否已运行 python train_ner.py 完成训练。")
        return

    # 2. 准备数据 (取最后 10% 作为测试集)
    label2id = model.config.label2id
    id2label = model.config.id2label
    
    print("Loading test data...")
    full_dataset = DiaKG_NER_Dataset(data_dir, tokenizer, label2id, config.NER_MAX_LEN)
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
            predictions = torch.argmax(outputs.logits, dim=-1)

            # 解码标签 (忽略 -100)
            for i in range(len(labels)):
                true_sent = []
                pred_sent = []
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        true_sent.append(id2label[labels[i][j].item()])
                        pred_sent.append(id2label[predictions[i][j].item()])
                
                y_true.append(true_sent)
                y_pred.append(pred_sent)

    # 4. 生成报告
    print("\n" + "="*20 + " NER 评估结果 " + "="*20)
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Entities F1 Score: {f1_score(y_true, y_pred):.4f}")
    print("="*55)

if __name__ == "__main__":
    evaluate()
