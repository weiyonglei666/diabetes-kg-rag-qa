# # 分析数据

# import json
# import os
# from collections import Counter
# from pprint import pprint
# from tqdm import tqdm  # 建议安装 tqdm 用于显示进度条: pip install tqdm

# # 配置文件目录路径 (注意这里指向文件夹)
# DATA_DIR = '../diakg_data/'

# def analyze_all_files(data_dir):
#     if not os.path.exists(data_dir):
#         raise FileNotFoundError(f"找不到目录: {data_dir}")

#     entity_counter = Counter()
#     relation_counter = Counter()
    
#     total_docs = 0
#     total_sentences = 0
    
#     # 获取目录下所有 json 文件
#     files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
#     print(f"检测到 {len(files)} 个 JSON 文件，开始统计...")

#     # 遍历文件
#     for filename in tqdm(files, desc="Processing files"):
#         file_path = os.path.join(data_dir, filename)
        
#         with open(file_path, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#             except json.JSONDecodeError:
#                 print(f"警告: 文件 {filename} 解析失败，跳过。")
#                 continue

#         # 兼容处理：有些文件可能直接是 dict，有些是 list
#         docs = data if isinstance(data, list) else data.get('data', [])
#         # 兜底：如果 data 是 dict 但没有 'data' 字段，可能它本身就是一个 doc
#         if not docs and isinstance(data, dict):
#             docs = [data]

#         total_docs += len(docs)

#         for doc in docs:
#             paragraphs = doc.get('paragraphs', [])
#             for para in paragraphs:
#                 sentences = para.get('sentences', [])
#                 total_sentences += len(sentences)
                
#                 for sent in sentences:
#                     # 统计实体
#                     entities = sent.get('entities', [])
#                     for ent in entities:
#                         entity_counter[ent['entity_type']] += 1
                    
#                     # 统计关系
#                     relations = sent.get('relations', [])
#                     for rel in relations:
#                         relation_counter[rel['relation_type']] += 1

#     # --- 新增统计逻辑 ---
#     total_entities = sum(entity_counter.values())
#     total_relations = sum(relation_counter.values())

#     print("\n" + "="*30)
#     print(f"【全局数据概览】")
#     print(f"文件数量: {len(files)}")
#     print(f"文档总数: {total_docs}")
#     print(f"句子总数: {total_sentences}")
#     print(f"实体总数: {total_entities}")  # <--- 新增
#     print(f"关系总数: {total_relations}") # <--- 新增
#     print("="*30)
#     print(f"【实体分布 (Top 20)】")
#     pprint(entity_counter.most_common(20))
#     print("-" * 30)
#     print(f"【关系分布 (Top 20)】")
#     pprint(relation_counter.most_common(20))
#     print("="*30)
    
#     return entity_counter, relation_counter

# if __name__ == "__main__":
#     analyze_all_files(DATA_DIR)
import json
import os
import csv
from collections import Counter
from pprint import pprint
from tqdm import tqdm  # pip install tqdm

# 配置文件目录路径
DATA_DIR = '../diakg_data/'
# 输出目录
OUTPUT_DIR = '../output/'


def analyze_all_files(data_dir, output_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到目录: {data_dir}")

    os.makedirs(output_dir, exist_ok=True)

    entity_type_counter = Counter()
    entity_name_counter = Counter()
    relation_counter = Counter()

    total_docs = 0
    total_sentences = 0

    # 获取目录下所有 json 文件
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"检测到 {len(files)} 个 JSON 文件，开始统计...")

    # 遍历文件
    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(data_dir, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"警告: 文件 {filename} 解析失败，跳过。")
                continue

        # 兼容处理：有些文件可能直接是 dict，有些是 list
        docs = data if isinstance(data, list) else data.get('data', [])
        # 兜底：如果 data 是 dict 但没有 'data' 字段，可能它本身就是一个 doc
        if not docs and isinstance(data, dict):
            docs = [data]

        total_docs += len(docs)

        for doc in docs:
            paragraphs = doc.get('paragraphs', [])
            for para in paragraphs:
                sentences = para.get('sentences', [])
                total_sentences += len(sentences)

                for sent in sentences:
                    # 统计实体
                    entities = sent.get('entities', [])
                    for ent in entities:
                        entity_type = ent.get('entity_type', '').strip()
                        entity_name = ent.get('entity', '').strip()

                        if entity_type:
                            entity_type_counter[entity_type] += 1
                        if entity_name:
                            entity_name_counter[entity_name] += 1

                    # 统计关系
                    relations = sent.get('relations', [])
                    for rel in relations:
                        relation_type = rel.get('relation_type', '').strip()
                        if relation_type:
                            relation_counter[relation_type] += 1

    # 汇总统计
    total_entities = sum(entity_type_counter.values())
    total_relations = sum(relation_counter.values())

    print("\n" + "=" * 30)
    print("【全局数据概览】")
    print(f"文件数量: {len(files)}")
    print(f"文档总数: {total_docs}")
    print(f"句子总数: {total_sentences}")
    print(f"实体总数: {total_entities}")
    print(f"关系总数: {total_relations}")
    print("=" * 30)

    print("【实体类型分布 (Top 20)】")
    pprint(entity_type_counter.most_common(20))
    print("-" * 30)

    print("【实体名称分布 (Top 20)】")
    pprint(entity_name_counter.most_common(20))
    print("-" * 30)

    print("【关系分布 (Top 20)】")
    pprint(relation_counter.most_common(20))
    print("=" * 30)

    # 保存 CSV
    entity_name_csv = os.path.join(output_dir, 'entity_name_counts.csv')
    entity_type_csv = os.path.join(output_dir, 'entity_type_counts.csv')
    relation_csv = os.path.join(output_dir, 'relation_type_counts.csv')

    # 1. 保存实体名称统计
    with open(entity_name_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['entity_name', 'count'])
        for entity_name, count in entity_name_counter.most_common():
            writer.writerow([entity_name, count])

    # 2. 保存实体类型统计
    with open(entity_type_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['entity_type', 'count'])
        for entity_type, count in entity_type_counter.most_common():
            writer.writerow([entity_type, count])

    # 3. 保存关系类型统计
    with open(relation_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['relation_type', 'count'])
        for relation_type, count in relation_counter.most_common():
            writer.writerow([relation_type, count])

    print(f"\n实体名称统计已保存到: {entity_name_csv}")
    print(f"实体类型统计已保存到: {entity_type_csv}")
    print(f"关系类型统计已保存到: {relation_csv}")

    return entity_type_counter, entity_name_counter, relation_counter


if __name__ == "__main__":
    analyze_all_files(DATA_DIR, OUTPUT_DIR)
