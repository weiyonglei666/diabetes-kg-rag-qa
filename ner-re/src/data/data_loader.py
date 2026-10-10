# [数据] 数据处理与加载 (含双重样本注入)
import json
import os
import random
import torch
from torch.utils.data import Dataset
from utils.utils import merge_compound_entities


class DiaKGBaseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _load_all_json_files(self):
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        for filename in files:
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    yield data
                except:
                    continue


class DiaKG_NER_Dataset(DiaKGBaseDataset):
    def __init__(self, data_dir, tokenizer, label2id, max_len=128):
        super().__init__(data_dir)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self._process_data()

    def _process_data(self):
        for raw_data in self._load_all_json_files():
            docs = raw_data if isinstance(raw_data, list) else [raw_data]

            for doc in docs:
                for para in doc.get('paragraphs', []):
                    for sent in para.get('sentences', []):
                        text = sent['sentence']
                        entities = sent.get('entities', [])

                        # 初始化全 'O' 标签
                        char_labels = ['O'] * len(text)

                        for ent in entities:
                            start, end = ent['start_idx'], ent['end_idx']
                            e_type = ent['entity_type']
                            if start < len(text) and end <= len(text):
                                char_labels[start] = f"B-{e_type}"
                                for i in range(start + 1, end):
                                    char_labels[i] = f"I-{e_type}"

                        self.samples.append({
                            'text': text,
                            'labels': char_labels
                        })

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            list(item['text']),
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id.get(item['labels'][word_id], 0))

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


class DiaKG_RE_Dataset(DiaKGBaseDataset):
    def __init__(self, data_dir, tokenizer, rel2id, max_len=128, negative_ratio=3):
        super().__init__(data_dir)
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.max_len = max_len
        self.negative_ratio = negative_ratio
        self._process_data()

    def _process_data(self):
        for raw_data in self._load_all_json_files():
            docs = raw_data if isinstance(raw_data, list) else [raw_data]

            for doc in docs:
                for para in doc.get('paragraphs', []):
                    for sent in para.get('sentences', []):
                        text = sent['sentence']
                        original_entities = sent.get('entities', [])
                        relations = sent.get('relations', [])

                        if not original_entities:
                            continue

                        # ------------------------------------------------
                        # 1. 注入【原始】实体关系 (Sample Type A)
                        # ------------------------------------------------
                        orig_map = {e['entity_id']: e for e in original_entities}
                        pos_pairs = set()

                        for rel in relations:
                            h = orig_map.get(rel['head_entity_id'])
                            t = orig_map.get(rel['tail_entity_id'])
                            if h and t:
                                self.samples.append({
                                    'text': text,
                                    'h': h['entity'],
                                    't': t['entity'],
                                    'r': rel['relation_type']
                                })
                                pos_pairs.add((rel['head_entity_id'], rel['tail_entity_id']))

                        # ------------------------------------------------
                        # 2. 注入【合并后】实体关系 (Sample Type B)
                        # ------------------------------------------------
                        merged_ents = merge_compound_entities(original_entities, text)

                        for rel in relations:
                            # 在合并后的列表中查找包含原ID的实体
                            # 注意：这里的split逻辑要和 utils.py 中的合并逻辑一致 (用 '--' 连接)
                            h_merged = None
                            t_merged = None

                            for m in merged_ents:
                                if str(rel['head_entity_id']) in str(m['entity_id']).split('--'):
                                    h_merged = m
                                    break

                            for m in merged_ents:
                                if str(rel['tail_entity_id']) in str(m['entity_id']).split('--'):
                                    t_merged = m
                                    break

                            if h_merged and t_merged and h_merged['entity_id'] != t_merged['entity_id']:
                                # 检查是否重复：构建临时key
                                current_pair_key = (h_merged['entity'], t_merged['entity'])
                                # 简单检查最近加入的样本，避免完全重复
                                # (更严谨的做法是把所有样本存set，但这里为了内存效率做简单处理)
                                duplicate = False
                                if len(self.samples) > 0:
                                    last_sample = self.samples[-1]
                                    if last_sample['h'] == h_merged['entity'] and last_sample['t'] == t_merged[
                                        'entity']:
                                        duplicate = True

                                if not duplicate:
                                    self.samples.append({
                                        'text': text,
                                        'h': h_merged['entity'],
                                        't': t_merged['entity'],
                                        'r': rel['relation_type']
                                    })
                                    pos_pairs.add((h_merged['entity_id'], t_merged['entity_id']))

                        # ------------------------------------------------
                        # 3. 负采样 (基于合并后的实体)
                        # ------------------------------------------------
                        merged_map = {e['entity_id']: e for e in merged_ents}
                        all_ids = list(merged_map.keys())
                        neg_candidates = []

                        for i in all_ids:
                            for j in all_ids:
                                # 不在正样本对中，且不是同一个实体
                                if i != j and (i, j) not in pos_pairs:
                                    neg_candidates.append((i, j))

                        if neg_candidates:
                            # 动态计算采样数量
                            sample_num = min(len(neg_candidates), max(1, len(pos_pairs)) * self.negative_ratio)

                            for h_id, t_id in random.sample(neg_candidates, sample_num):
                                self.samples.append({
                                    'text': text,
                                    'h': merged_map[h_id]['entity'],
                                    't': merged_map[t_id]['entity'],
                                    'r': 'No_Relation'
                                })

    def __getitem__(self, idx):
        item = self.samples[idx]
        text_pair = f"{item['h']}[SEP]{item['t']}"

        encoding = self.tokenizer(
            item['text'],
            text_pair,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.rel2id.get(item['r'], 0), dtype=torch.long)
        }

