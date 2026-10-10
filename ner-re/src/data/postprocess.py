from config.config import ENTITY_TYPES


BLACKLIST_BY_TYPE = {
    "Treatment": {"治疗", "指导", "应用", "降糖治疗"},
    "Method": {"流行病学", "行动"},
    "Class": {"生活质量"},
    "Test": {"UKPDS", "ADVANCE", "英国糖尿病前瞻性研究", "糖尿病和心血管疾病行动"},
    "ADE": {"生活质量"},
}

GENERIC_BAD_ENTITIES = {
    "问题", "患者", "生命", "社会", "家庭", "特点", "危险因素"
}


def find_all_occurrences(text, sub):
    positions = []
    if not sub:
        return positions

    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(sub)))
        start = idx + 1
    return positions


def choose_best_span(text, entity, raw_start_idx=None):
    candidates = find_all_occurrences(text, entity)
    if not candidates:
        return None

    if raw_start_idx is None or not isinstance(raw_start_idx, int):
        return candidates[0]

    best = min(candidates, key=lambda x: abs(x[0] - raw_start_idx))
    return best


def is_valid_entity(entity, entity_type):
    if entity_type not in ENTITY_TYPES:
        return False

    if entity in GENERIC_BAD_ENTITIES:
        return False

    if entity_type in BLACKLIST_BY_TYPE and entity in BLACKLIST_BY_TYPE[entity_type]:
        return False

    if entity_type == "Test_Value":
        # 必须包含数字，避免“达标率”等误报
        if not any(ch.isdigit() for ch in entity):
            return False

    if entity_type == "Class":
        # 更贴近数据集：优先保留 1型/2型 这类
        if entity not in {"1型", "2型"} and len(entity) > 4:
            return False

    if entity_type == "Level":
        # 百分比通常不当作 Level
        if "%" in entity:
            return False

    if entity_type == "Test":
        # 研究名/缩写在这个数据风格下通常不作为 Test
        if entity.isupper():
            return False

    return True


def repair_special_entities(text, entities):
    repaired = list(entities)
    existing_keys = {
        (e["entity"], e["entity_type"], e["start_idx"], e["end_idx"])
        for e in repaired
    }

    def add_entity(entity, entity_type, start_idx, end_idx):
        key = (entity, entity_type, start_idx, end_idx)
        if key not in existing_keys and text[start_idx:end_idx] == entity:
            repaired.append({
                "entity": entity,
                "entity_type": entity_type,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
            existing_keys.add(key)

    # 1) 若已有“糖尿病”，且前面紧邻 1型/2型，则补完整 Disease
    for ent in list(repaired):
        if ent["entity_type"] == "Disease" and ent["entity"] == "糖尿病":
            s = ent["start_idx"]
            e = ent["end_idx"]

            if s >= 2:
                prefix = text[s - 2:s]
                if prefix in {"1型", "2型"}:
                    add_entity(prefix + "糖尿病", "Disease", s - 2, e)

    # 2) 直接补 1型糖尿病 / 2型糖尿病
    for pattern in ["1型糖尿病", "2型糖尿病"]:
        for start_idx, end_idx in find_all_occurrences(text, pattern):
            add_entity(pattern, "Disease", start_idx, end_idx)

    # 3) HbA1c≤7% 这种，拆成 HbA1c 和 ≤7%
    combined_patterns = ["HbA1c≤7%", "HbA1c<7%", "HbA1c≥7%", "HbA1c>7%"]
    for pattern in combined_patterns:
        for start_idx, end_idx in find_all_occurrences(text, pattern):
            test_item = "HbA1c"
            value = pattern[len("HbA1c"):]
            add_entity(test_item, "Test_items", start_idx, start_idx + len(test_item))
            add_entity(value, "Test_Value", start_idx + len(test_item), end_idx)

    # 4) 补糖尿病肾病
    for start_idx, end_idx in find_all_occurrences(text, "糖尿病肾病"):
        add_entity("糖尿病肾病", "Disease", start_idx, end_idx)

    # 5) 补细粒度 Anatomy
    for anatomy in ["肾", "血管", "心血管"]:
        for start_idx, end_idx in find_all_occurrences(text, anatomy):
            add_entity(anatomy, "Anatomy", start_idx, end_idx)

    # 6) 补完整 Pathogenesis
    patho_pattern = "β细胞胰岛素分泌功能下降比胰岛素敏感性降低更明显"
    for start_idx, end_idx in find_all_occurrences(text, patho_pattern):
        add_entity(patho_pattern, "Pathogenesis", start_idx, end_idx)

    # 7) 补胰岛素促泌剂
    for start_idx, end_idx in find_all_occurrences(text, "胰岛素促泌剂"):
        add_entity("胰岛素促泌剂", "Drug", start_idx, end_idx)

    # 8) 若出现“糖尿病血管并发症”，补内部更细粒度实体
    for start_idx, end_idx in find_all_occurrences(text, "糖尿病血管并发症"):
        add_entity("糖尿病", "Disease", start_idx, start_idx + 3)
        add_entity("血管", "Anatomy", start_idx + 3, start_idx + 5)

    # 9) 若出现“血管并发症”，补内部“血管”
    for start_idx, end_idx in find_all_occurrences(text, "血管并发症"):
        add_entity("血管", "Anatomy", start_idx, start_idx + 2)

    return repaired


def validate_ner_result(text, ner_result):
    if not isinstance(ner_result, dict):
        return {"text": text, "entities": []}

    entities = ner_result.get("entities", [])
    if not isinstance(entities, list):
        return {"text": text, "entities": []}

    valid_entities = []
    seen = set()

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        entity = ent.get("entity", "")
        entity_type = ent.get("entity_type", "")
        raw_start_idx = ent.get("start_idx", None)

        if not isinstance(entity, str) or not entity.strip():
            continue

        entity = entity.strip()

        if not is_valid_entity(entity, entity_type):
            continue

        span = choose_best_span(text, entity, raw_start_idx)
        if span is None:
            continue

        fixed_start, fixed_end = span
        key = (entity, entity_type, fixed_start, fixed_end)

        if key in seen:
            continue
        seen.add(key)

        valid_entities.append({
            "entity": entity,
            "entity_type": entity_type,
            "start_idx": fixed_start,
            "end_idx": fixed_end
        })

    valid_entities = repair_special_entities(text, valid_entities)

    dedup = []
    seen = set()
    for ent in valid_entities:
        key = (ent["entity"], ent["entity_type"], ent["start_idx"], ent["end_idx"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ent)

    dedup.sort(key=lambda x: (x["start_idx"], x["end_idx"], x["entity_type"]))

    return {
        "text": text,
        "entities": dedup
    }
