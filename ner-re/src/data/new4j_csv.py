import os
import json
import csv
from pathlib import Path


INPUT_DIR = '../diakg_data/'
OUTPUT_DIR = '../kg_project/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

nodes = {}
rels = []

def make_gid(entity_type, entity_name):
    return f"{entity_type}::{entity_name.strip()}"

json_files = list(Path(INPUT_DIR).glob("*.json"))

for fp in json_files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = str(data.get("doc_id", fp.stem))

    for para in data.get("paragraphs", []):
        paragraph_id = str(para.get("paragraph_id", ""))
        paragraph_text = para.get("paragraph", "")

        for sent in para.get("sentences", []):
            sentence_id = str(sent.get("sentence_id", ""))
            sentence_text = sent.get("sentence", "") or paragraph_text

            entity_map = {}

            for ent in sent.get("entities", []):
                entity_id = ent.get("entity_id", "")
                entity_name = str(ent.get("entity", "")).strip()
                entity_type = str(ent.get("entity_type", "")).strip()

                if not entity_name or not entity_type:
                    continue

                gid = make_gid(entity_type, entity_name)
                entity_map[entity_id] = gid

                if gid not in nodes:
                    nodes[gid] = {
                        "gid": gid,
                        "name": entity_name,
                        "etype": entity_type,
                        "label": entity_type,
                        "first_doc_id": doc_id,
                        "first_paragraph_id": paragraph_id,
                        "first_sentence_id": sentence_id,
                        "source_text": sentence_text
                    }

            for rel in sent.get("relations", []):
                relation_id = rel.get("relation_id", "")
                relation_type = rel.get("relation_type", "RELATED_TO")
                head_entity_id = rel.get("head_entity_id", "")
                tail_entity_id = rel.get("tail_entity_id", "")

                if head_entity_id not in entity_map or tail_entity_id not in entity_map:
                    continue

                rels.append({
                    "rid": f"{doc_id}_{paragraph_id}_{sentence_id}_{relation_id}",
                    "start_gid": entity_map[head_entity_id],
                    "end_gid": entity_map[tail_entity_id],
                    "rel_type": relation_type,
                    "doc_id": doc_id,
                    "paragraph_id": paragraph_id,
                    "sentence_id": sentence_id,
                    "evidence": sentence_text
                })

nodes_csv = os.path.join(OUTPUT_DIR, "nodes.csv")
rels_csv = os.path.join(OUTPUT_DIR, "rels.csv")

with open(nodes_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "gid", "name", "etype", "label",
            "first_doc_id", "first_paragraph_id",
            "first_sentence_id", "source_text"
        ]
    )
    writer.writeheader()
    for row in nodes.values():
        writer.writerow(row)

with open(rels_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rid", "start_gid", "end_gid", "rel_type",
            "doc_id", "paragraph_id", "sentence_id", "evidence"
        ]
    )
    writer.writeheader()
    for row in rels:
        writer.writerow(row)

print("转换完成")
print("nodes.csv:", nodes_csv)
print("rels.csv :", rels_csv)
print("节点数:", len(nodes))
print("关系数:", len(rels))
