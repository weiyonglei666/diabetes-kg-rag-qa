# # [推理] 最终预测脚本

import os
import torch
import config.config as config
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, pipeline
from utils.utils import merge_compound_entities

# 模型路径
NER_PATH = os.path.join(config.OUTPUT_DIR, "ner_model")
RE_PATH = os.path.join(config.OUTPUT_DIR, "re_model")

def load_models():
    device = 0 if torch.cuda.is_available() else -1
    print("Loading models...")
    
    # 1. Load NER
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_PATH)
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_PATH)
    ner_pipe = pipeline("token-classification", model=ner_model, tokenizer=ner_tokenizer, 
                        device=device, aggregation_strategy="simple")
    
    # 2. Load RE
    re_model = AutoModelForSequenceClassification.from_pretrained(RE_PATH)
    re_tokenizer = AutoTokenizer.from_pretrained(RE_PATH)
    re_model.to("cuda" if device == 0 else "cpu").eval()
    
    return ner_pipe, re_model, re_tokenizer

def extract_triples(text, ner_pipe, re_model, re_tokenizer):
    device = next(re_model.parameters()).device
    
    # 1. NER
    ner_results = ner_pipe(text)
    
    # 2. Entity Merge (关键优化)
    merged_entities = merge_compound_entities(ner_results, text)
    
    # <--- 修改点 1: 如果实体太少，直接返回空列表和当前识别到的实体 --->
    if len(merged_entities) < 2: 
        return [], merged_entities
    
    triples = []
    MAX_DISTANCE = 40 # 宽松的距离约束
    
    for head in merged_entities:
        for tail in merged_entities:
            if head == tail: continue
            
            # 距离约束
            if abs(head['start_idx'] - tail['start_idx']) > MAX_DISTANCE: continue
            
            # 类型约束
            matched_rel = None
            for rel_name, (h_type, t_type) in config.RELATION_CONSTRAINTS.items():
                if head['entity_type'] == h_type and tail['entity_type'] == t_type:
                    matched_rel = rel_name
                    break
            if not matched_rel: continue
            
            # 模型预测
            text_pair = f"{head['entity']}[SEP]{tail['entity']}"
            inputs = re_tokenizer(text, text_pair, return_tensors="pt", truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                logits = re_model(**inputs).logits
                probs = F.softmax(logits, dim=1)[0]
                pred_id = torch.argmax(probs).item()
                score = probs[pred_id].item()
                label_name = re_model.config.id2label[pred_id]
            
            if label_name != 'No_Relation' and score > 0.6:
                triples.append({
                    "head": head['entity'], "relation": label_name, "tail": tail['entity'], "score": score
                })
    
    # <--- 修改点 2: 同时返回三元组和实体列表 --->
    return triples, merged_entities

if __name__ == "__main__":
    ner_pipe, re_model, re_tokenizer = load_models()
    
    test_text = "患者确诊为2型糖尿病，伴有高血压，建议服用二甲双胍进行治疗。"
    print(f"\nText: {test_text}")
    
    # <--- 修改点 3: 接收两个返回值 --->
    triples, entities = extract_triples(test_text, ner_pipe, re_model, re_tokenizer)
    
    # <--- 修改点 4: 打印实体 --->
    print("\n" + "="*20 + " [1] 识别到的实体 " + "="*20)
    for e in entities:
        print(f"[{e['entity_type']}] {e['entity']}")
        
    print("\n" + "="*20 + " [2] 抽取到的关系 " + "="*20)
    if not triples:
        print("未发现关系。")
    else:
        for t in triples:
            print(f"({t['head']}) --[{t['relation']}]--> ({t['tail']}) {t['score']:.4f}")

