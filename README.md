# Diabetes KG-RAG Question Answering

面向糖尿病健康知识问答的毕业设计代码，包含医学实体/关系抽取、Neo4j 知识图谱查询、Faiss 文本检索与大语言模型问答服务。

## Repository layout

- `ner-re/`: MacBERT/CRF、Chinese-RoBERTa 和 QLoRA 实验代码。
- `web-app/`: FastAPI 接口、知识图谱/问答前端、KG/RAG 评测脚本。
- `data/`: 数据、索引和模型文件的放置说明。

## Quick start

1. Create a Python 3.10+ environment.
2. Install the relevant dependencies:

   ```bash
   pip install -r web-app/requirements.txt
   # For NER/RE experiments:
   pip install -r ner-re/requirements.txt
   ```

3. Copy `web-app/.env.example` to `web-app/.env`, then fill in your own Neo4j and model-provider credentials. Never commit `.env` or real API keys.
4. Put local DiaKG/QA data, model checkpoints and FAISS indexes under a local data directory, then update the paths in the scripts.
5. Start the API, for example:

   ```bash
   uvicorn web-app.diabetes_graph_api:app --reload
   ```

## Scope and reproducibility

This repository contains source code and configuration templates. Generated predictions, evaluation spreadsheets, downloaded models, virtual environments, private credentials and local database files are excluded. Reproduce the reported experiments by supplying the original datasets, model checkpoints and service credentials locally.

## Safety

The system is for research and health education. It is not a substitute for medical diagnosis or treatment.
