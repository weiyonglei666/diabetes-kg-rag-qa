 # [工具] 实体合并逻辑
def merge_compound_entities(entities, text):
    """
    通用实体合并逻辑，支持 A+B 和 A+B+C 模式
    """
    if not entities: return []
    
    # 1. 格式标准化 (兼容 dataset 和 pipeline 的不同输入key)
    fmt_entities = []
    for e in entities:
        new_e = e.copy()
        if 'start' in e: new_e['start_idx'] = e['start']
        if 'end' in e: new_e['end_idx'] = e['end']
        if 'entity_group' in e: new_e['entity_type'] = e['entity_group']
        
        # [关键修复] 保证 'entity' 键存在
        # Pipeline 输出只有 'word'，Dataset 输出有 'entity'
        if 'entity' not in new_e and 'word' in new_e:
            new_e['entity'] = new_e['word']
            
        fmt_entities.append(new_e)

    # 2. 排序
    fmt_entities.sort(key=lambda x: x['start_idx'])
    
    merged = []
    i = 0
    while i < len(fmt_entities):
        current = fmt_entities[i]
        
        # 规则 3: Disease + Anatomy + Disease (如: 糖尿病 + 视网膜 + 病变)
        if i < len(fmt_entities) - 2:
            next1 = fmt_entities[i+1]
            next2 = fmt_entities[i+2]
            gap1 = next1['start_idx'] - current['end_idx']
            gap2 = next2['start_idx'] - next1['end_idx']
            
            if (current['entity_type'] == 'Disease' and 
                next1['entity_type'] == 'Anatomy' and 
                next2['entity_type'] == 'Disease' and
                gap1 <= 1 and gap2 <= 1):
                
                new_word = text[current['start_idx']:next2['end_idx']]
                merged_ent = _create_merged_ent(current, next2, new_word)
                # 级联ID
                merged_ent['entity_id'] = f"{current.get('entity_id','')}--{next2.get('entity_id','')}"
                merged.append(merged_ent)
                i += 3
                continue

        # 规则 1 & 2: Class/Anatomy + Disease
        if i < len(fmt_entities) - 1:
            next1 = fmt_entities[i+1]
            gap = next1['start_idx'] - current['end_idx']
            
            if gap <= 1:
                is_rule1 = (current['entity_type'] == 'Class' and next1['entity_type'] == 'Disease')
                is_rule2 = (current['entity_type'] == 'Anatomy' and next1['entity_type'] == 'Disease')
                
                if is_rule1 or is_rule2:
                    new_word = text[current['start_idx']:next1['end_idx']]
                    merged_ent = _create_merged_ent(current, next1, new_word)
                    merged_ent['entity_id'] = f"{current.get('entity_id','')}--{next1.get('entity_id','')}"
                    merged.append(merged_ent)
                    i += 2
                    continue
        
        merged.append(current)
        i += 1
    return merged

def _create_merged_ent(start_ent, end_ent, new_word):
    """辅助函数"""
    new_ent = start_ent.copy()
    new_ent.update({
        'entity_type': 'Disease',
        'entity_group': 'Disease',
        'word': new_word,
        'entity': new_word, # 确保合并后的实体也有 entity 键
        'start_idx': start_ent['start_idx'],
        'end_idx': end_ent['end_idx'],
        'start': start_ent['start_idx'],
        'end': end_ent['end_idx']
    })
    return new_ent
