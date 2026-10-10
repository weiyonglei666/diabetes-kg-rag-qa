# -*- coding: utf-8 -*-
import os
from contextlib import suppress
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
from pydantic import BaseModel, Field


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# 更克制的默认值，避免前端一次性渲染过多节点导致拥挤
GRAPH_DEFAULT_LIMIT = int(os.getenv("GRAPH_DEFAULT_LIMIT", "80"))
GRAPH_NEIGHBOR_LIMIT = int(os.getenv("GRAPH_NEIGHBOR_LIMIT", "40"))
GRAPH_SEARCH_LIMIT = int(os.getenv("GRAPH_SEARCH_LIMIT", "60"))
GRAPH_TOPIC_LIMIT = int(os.getenv("GRAPH_TOPIC_LIMIT", "70"))
HOT_TERMS_LIMIT = int(os.getenv("HOT_TERMS_LIMIT", "10"))


REL_TYPE_ZH_FALLBACK = {
    "HAS_SYMPTOM": "表现为",
    "TREATED_BY": "治疗方式",
    "USE_DRUG": "常用药物",
    "CAUSES": "导致",
    "RELATED_TO": "相关",
    "HAS_COMPLICATION": "并发症",
    "CHECK_BY": "检查方式",
    "RISK_FACTOR": "风险因素",
    "MANAGED_BY": "管理方式",
    "CONTRAINDICATED_WITH": "禁忌",
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return " ".join(text.replace("\u3000", " ").split())


class GraphNode(BaseModel):
    id: str
    label: str
    group: str = "实体"
    group_zh: str = "实体"
    description: str = ""
    topic: str = "全部知识点"
    value: int = 1
    degree: int = 0


class GraphEdge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    id: str
    label: str = "关系"
    rel_type: str = ""
    evidence: str = ""

    model_config = {"populate_by_name": True}


class GraphStats(BaseModel):
    node_count: int
    edge_count: int
    category_count: int
    relation_count: int


class GraphMeta(BaseModel):
    applied_limit: int
    returned_rows: int
    truncated: bool = False
    mode: str = "overview"
    center: str = ""


class HotTerm(BaseModel):
    name: str
    degree: int
    group: str = "实体"


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: GraphStats
    topics: List[str]
    keyword: str = ""
    topic: str = "全部知识点"
    meta: GraphMeta


class HotTermsResponse(BaseModel):
    hot_terms: List[HotTerm]


class KGGraphRetriever:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        with suppress(Exception):
            self.driver.verify_connectivity()

    def close(self):
        with suppress(Exception):
            self.driver.close()

    def session(self):
        return self.driver.session(database=self.database)

    def ping(self) -> bool:
        try:
            with self.session() as session:
                record = session.run("RETURN 1 AS ok").single()
                return bool(record and record["ok"] == 1)
        except Exception:
            return False

    def get_topics(self) -> List[str]:
        with self.session() as session:
            return session.execute_read(self._get_topics_tx)

    def get_global_stats(self) -> Dict[str, int]:
        with self.session() as session:
            return session.execute_read(self._get_global_stats_tx)

    def get_hot_terms(self, limit: int = HOT_TERMS_LIMIT) -> Dict[str, Any]:
        with self.session() as session:
            return {"hot_terms": session.execute_read(self._get_hot_terms_tx, limit)}

    def get_overview(self, limit: int = GRAPH_DEFAULT_LIMIT) -> Dict[str, Any]:
        with self.session() as session:
            data = session.execute_read(self._get_overview_tx, limit)
            data["topics"] = session.execute_read(self._get_topics_tx)
            return data

    def search_subgraph(self, keyword: str, limit: int = GRAPH_SEARCH_LIMIT) -> Dict[str, Any]:
        with self.session() as session:
            data = session.execute_read(self._search_subgraph_tx, keyword, limit)
            data["topics"] = session.execute_read(self._get_topics_tx)
            return data

    def get_topic_subgraph(self, topic: str, limit: int = GRAPH_TOPIC_LIMIT) -> Dict[str, Any]:
        with self.session() as session:
            data = session.execute_read(self._get_topic_subgraph_tx, topic, limit)
            data["topics"] = session.execute_read(self._get_topics_tx)
            return data

    def get_neighbors(self, node_name: str, limit: int = GRAPH_NEIGHBOR_LIMIT) -> Dict[str, Any]:
        with self.session() as session:
            data = session.execute_read(self._get_neighbors_tx, node_name, limit)
            data["topics"] = session.execute_read(self._get_topics_tx)
            return data

    @staticmethod
    def _safe_node_dict(node: Any) -> Dict[str, Any]:
        if node is None:
            return {}
        if isinstance(node, dict):
            data = dict(node)
        else:
            try:
                data = dict(node)
            except Exception:
                data = {}
        try:
            if "id" not in data:
                data["id"] = str(getattr(node, "element_id", "") or getattr(node, "id", ""))
        except Exception:
            pass
        return data

    @staticmethod
    def _node_topic(node: Dict[str, Any]) -> str:
        group_zh = normalize_text(node.get("etype_zh"))
        group_en = normalize_text(node.get("etype"))
        return group_zh or group_en or "实体"

    @staticmethod
    def _node_description(node: Dict[str, Any]) -> str:
        for key in ["summary", "desc", "description", "definition", "note"]:
            value = normalize_text(node.get(key))
            if value:
                return value
        return ""

    @staticmethod
    def _rel_label_from_values(rel_type: str, rel_type_zh: str) -> str:
        rel_type = normalize_text(rel_type)
        rel_type_zh = normalize_text(rel_type_zh)
        return rel_type_zh or REL_TYPE_ZH_FALLBACK.get(rel_type, rel_type or "关系")

    def _record_to_node(self, node_raw: Any, degree: int = 0) -> GraphNode:
        node = self._safe_node_dict(node_raw)
        node_id = normalize_text(node.get("id")) or normalize_text(node.get("name"))
        group_en = normalize_text(node.get("etype")) or "实体"
        group_zh = normalize_text(node.get("etype_zh")) or group_en
        label = normalize_text(node.get("name")) or node_id
        degree_value = int(node.get("degree") or degree or 0)
        node_value = max(1, min(12, degree_value if degree_value else 2))
        return GraphNode(
            id=node_id,
            label=label,
            group=group_en,
            group_zh=group_zh,
            description=self._node_description(node),
            topic=self._node_topic(node),
            value=node_value,
            degree=degree_value,
        )

    def _record_to_edge(self, row: Dict[str, Any], source_id: str, target_id: str, seq: int) -> GraphEdge:
        rel_type = normalize_text(row.get("rel_type"))
        rel_type_zh = normalize_text(row.get("rel_type_zh"))
        evidence = normalize_text(row.get("rel_evidence"))
        label = self._rel_label_from_values(rel_type=rel_type, rel_type_zh=rel_type_zh)
        return GraphEdge(
            **{
                "from": source_id,
                "to": target_id,
                "id": f"e_{seq}_{source_id}_{target_id}_{rel_type or 'REL'}",
                "label": label,
                "rel_type": rel_type,
                "evidence": evidence,
            }
        )

    def _package_graph(
        self,
        rows: List[Dict[str, Any]],
        topic: str,
        keyword: str,
        applied_limit: int,
        mode: str,
        center: str = "",
        total_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        node_map: Dict[str, GraphNode] = {}
        edge_map: Dict[str, GraphEdge] = {}
        group_set = set()
        rel_set = set()
        degree_counter: Dict[str, int] = {}

        for row in rows:
            for key in ("n", "m"):
                raw = self._safe_node_dict(row.get(key))
                node_id = normalize_text(raw.get("id")) or normalize_text(raw.get("name"))
                if node_id:
                    degree_counter[node_id] = degree_counter.get(node_id, 0) + 1

        for idx, row in enumerate(rows):
            source = self._record_to_node(row.get("n"), degree_counter.get(self._node_id(row.get("n")), 0))
            target = self._record_to_node(row.get("m"), degree_counter.get(self._node_id(row.get("m")), 0))
            if not source.id or not target.id:
                continue
            edge = self._record_to_edge(row, source.id, target.id, idx)
            node_map[source.id] = source
            node_map[target.id] = target
            group_set.add(source.group_zh)
            group_set.add(target.group_zh)
            edge_key = f"{edge.from_}|{edge.to}|{edge.rel_type}|{edge.label}"
            if edge_key not in edge_map:
                edge_map[edge_key] = edge
                rel_set.add(edge.label)

        nodes = sorted(list(node_map.values()), key=lambda item: (-item.degree, item.label))
        edges = list(edge_map.values())
        total_rows_value = len(rows) if total_rows is None else int(total_rows)
        truncated = total_rows_value > len(rows)

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": GraphStats(
                node_count=len(nodes),
                edge_count=len(edges),
                category_count=len(group_set),
                relation_count=len(rel_set),
            ),
            "topic": topic,
            "keyword": keyword,
            "meta": GraphMeta(
                applied_limit=applied_limit,
                returned_rows=len(rows),
                truncated=truncated,
                mode=mode,
                center=center,
            ),
        }

    @staticmethod
    def _node_id(node_raw: Any) -> str:
        if node_raw is None:
            return ""
        node = dict(node_raw) if isinstance(node_raw, dict) else KGGraphRetriever._safe_node_dict(node_raw)
        return normalize_text(node.get("id")) or normalize_text(node.get("name"))

    def _get_topics_tx(self, tx) -> List[str]:
        cypher = """
        MATCH (n:Entity)
        WITH coalesce(n.etype_zh, n.etype, '实体') AS topic, count(*) AS cnt
        RETURN topic
        ORDER BY cnt DESC, topic ASC
        """
        rows = tx.run(cypher)
        topics = [normalize_text(r["topic"]) for r in rows if normalize_text(r["topic"])]
        return ["全部知识点"] + topics

    def _get_global_stats_tx(self, tx) -> Dict[str, int]:
        node_count = tx.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        edge_count = tx.run("MATCH (:Entity)-[r]-(:Entity) RETURN count(r) AS c").single()["c"]
        category_count = tx.run(
            "MATCH (n:Entity) RETURN count(DISTINCT coalesce(n.etype_zh, n.etype, '实体')) AS c"
        ).single()["c"]
        relation_count = tx.run(
            "MATCH (:Entity)-[r]-(:Entity) RETURN count(DISTINCT coalesce(r.rel_type_zh, type(r), '关系')) AS c"
        ).single()["c"]
        return {
            "node_count": int(node_count or 0),
            "edge_count": int(edge_count or 0),
            "category_count": int(category_count or 0),
            "relation_count": int(relation_count or 0),
        }

    def _get_hot_terms_tx(self, tx, limit: int) -> List[HotTerm]:
        cypher = """
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]-(:Entity)
        WITH n, count(r) AS degree
        ORDER BY degree DESC, coalesce(n.name, '') ASC
        LIMIT $limit
        RETURN coalesce(n.name, '') AS name,
               degree,
               coalesce(n.etype_zh, n.etype, '实体') AS group
        """
        rows = tx.run(cypher, limit=limit)
        return [
            HotTerm(
                name=normalize_text(row["name"]),
                degree=int(row["degree"] or 0),
                group=normalize_text(row["group"]) or "实体",
            )
            for row in rows
            if normalize_text(row["name"])
        ]

    def _get_overview_tx(self, tx, limit: int) -> Dict[str, Any]:
        count_cypher = "MATCH (:Entity)-[r]-(:Entity) RETURN count(r) AS c"
        total_rows = tx.run(count_cypher).single()["c"]
        cypher = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WITH n, r, m,
             COUNT { (n)--() } + COUNT { (m)--() } AS weight
        ORDER BY weight DESC, coalesce(n.name, ''), coalesce(m.name, '')
        LIMIT $limit
        RETURN n,
               r,
               m,
               type(r) AS rel_type,
               coalesce(r.rel_type_zh, '') AS rel_type_zh,
               coalesce(r.evidence, '') AS rel_evidence
        """
        rows = [record.data() for record in tx.run(cypher, limit=limit)]
        return self._package_graph(
            rows=rows,
            topic="全部知识点",
            keyword="",
            applied_limit=limit,
            mode="overview",
            total_rows=int(total_rows or 0),
        )

    def _search_subgraph_tx(self, tx, keyword: str, limit: int) -> Dict[str, Any]:
        count_cypher = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE n.name CONTAINS $keyword
           OR m.name CONTAINS $keyword
           OR coalesce(n.etype_zh, '') CONTAINS $keyword
           OR coalesce(m.etype_zh, '') CONTAINS $keyword
           OR coalesce(r.evidence, '') CONTAINS $keyword
           OR coalesce(r.rel_type_zh, '') CONTAINS $keyword
           OR type(r) CONTAINS $keyword
        RETURN count(r) AS c
        """
        total_rows = tx.run(count_cypher, keyword=keyword).single()["c"]
        cypher = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE n.name CONTAINS $keyword
           OR m.name CONTAINS $keyword
           OR coalesce(n.etype_zh, '') CONTAINS $keyword
           OR coalesce(m.etype_zh, '') CONTAINS $keyword
           OR coalesce(r.evidence, '') CONTAINS $keyword
           OR coalesce(r.rel_type_zh, '') CONTAINS $keyword
           OR type(r) CONTAINS $keyword
        WITH n, r, m,
             CASE
               WHEN coalesce(n.name, '') = $keyword OR coalesce(m.name, '') = $keyword THEN 4
               WHEN coalesce(n.name, '') CONTAINS $keyword OR coalesce(m.name, '') CONTAINS $keyword THEN 3
               WHEN coalesce(r.rel_type_zh, '') CONTAINS $keyword OR type(r) CONTAINS $keyword THEN 2
               ELSE 1
             END AS score
        WITH n, r, m, score,
             COUNT { (n)--() } + COUNT { (m)--() } AS weight
        ORDER BY score DESC, weight DESC, coalesce(n.name, ''), coalesce(m.name, '')
        LIMIT $limit
        RETURN n,
               r,
               m,
               type(r) AS rel_type,
               coalesce(r.rel_type_zh, '') AS rel_type_zh,
               coalesce(r.evidence, '') AS rel_evidence
        """
        rows = [record.data() for record in tx.run(cypher, keyword=keyword, limit=limit)]
        return self._package_graph(
            rows=rows,
            topic="搜索结果",
            keyword=keyword,
            applied_limit=limit,
            mode="search",
            center=keyword,
            total_rows=int(total_rows or 0),
        )

    def _get_topic_subgraph_tx(self, tx, topic: str, limit: int) -> Dict[str, Any]:
        count_cypher = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE coalesce(n.etype_zh, n.etype, '实体') = $topic
           OR coalesce(m.etype_zh, m.etype, '实体') = $topic
        RETURN count(r) AS c
        """
        total_rows = tx.run(count_cypher, topic=topic).single()["c"]
        cypher = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE coalesce(n.etype_zh, n.etype, '实体') = $topic
           OR coalesce(m.etype_zh, m.etype, '实体') = $topic
        WITH n, r, m,
             CASE
               WHEN coalesce(n.etype_zh, n.etype, '实体') = $topic
                AND coalesce(m.etype_zh, m.etype, '实体') = $topic THEN 2
               ELSE 1
             END AS score
        WITH n, r, m, score,
             COUNT { (n)--() } + COUNT { (m)--() } AS weight
        ORDER BY score DESC, weight DESC, coalesce(n.name, ''), coalesce(m.name, '')
        LIMIT $limit
        RETURN n,
               r,
               m,
               type(r) AS rel_type,
               coalesce(r.rel_type_zh, '') AS rel_type_zh,
               coalesce(r.evidence, '') AS rel_evidence
        """
        rows = [record.data() for record in tx.run(cypher, topic=topic, limit=limit)]
        return self._package_graph(
            rows=rows,
            topic=topic,
            keyword="",
            applied_limit=limit,
            mode="topic",
            center=topic,
            total_rows=int(total_rows or 0),
        )

    def _find_best_node_name(self, tx, node_name: str) -> Tuple[str, int]:
        cypher = """
        MATCH (n:Entity)
        WHERE coalesce(n.name, '') = $node_name OR coalesce(n.name, '') CONTAINS $node_name
        OPTIONAL MATCH (n)-[r]-(:Entity)
        WITH n, count(r) AS degree,
             CASE WHEN coalesce(n.name, '') = $node_name THEN 2 ELSE 1 END AS score
        ORDER BY score DESC, degree DESC, coalesce(n.name, '') ASC
        LIMIT 1
        RETURN coalesce(n.name, '') AS name, degree
        """
        row = tx.run(cypher, node_name=node_name).single()
        if not row:
            return node_name, 0
        return normalize_text(row["name"]) or node_name, int(row["degree"] or 0)

    def _get_neighbors_tx(self, tx, node_name: str, limit: int) -> Dict[str, Any]:
        center_name, _ = self._find_best_node_name(tx, node_name)
        count_cypher = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE coalesce(n.name, '') = $center_name
        RETURN count(r) AS c
        """
        total_rows = tx.run(count_cypher, center_name=center_name).single()["c"]
        cypher = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE coalesce(n.name, '') = $center_name
        WITH n, r, m, COUNT { (m)--() } AS neighbor_degree
        ORDER BY neighbor_degree DESC, coalesce(m.name, '') ASC
        LIMIT $limit
        RETURN n,
               r,
               m,
               type(r) AS rel_type,
               coalesce(r.rel_type_zh, '') AS rel_type_zh,
               coalesce(r.evidence, '') AS rel_evidence
        """
        rows = [record.data() for record in tx.run(cypher, center_name=center_name, limit=limit)]
        return self._package_graph(
            rows=rows,
            topic="局部关系",
            keyword=center_name,
            applied_limit=limit,
            mode="neighbors",
            center=center_name,
            total_rows=int(total_rows or 0),
        )


app = FastAPI(
    title="糖尿病知识图谱图谱模式后端",
    description="为图谱模式页面提供概览、搜索、主题筛选、局部关系网络接口（Neo4j 2026 显式关系类型版）",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = KGGraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)


@app.get("/")
def root():
    return {
        "message": "糖尿病知识图谱图谱模式后端已启动",
        "health": "/health",
        "overview": "/graph/overview",
        "topics": "/graph/topics",
        "global_stats": "/graph/global-stats",
        "hot_terms": "/graph/hot-terms",
        "database": NEO4J_DATABASE,
    }


@app.get("/health")
def health():
    ok = retriever.ping()
    return {
        "status": "ok" if ok else "error",
        "neo4j_connected": ok,
        "database": NEO4J_DATABASE,
    }


@app.get("/graph/global-stats")
def graph_global_stats():
    return retriever.get_global_stats()


@app.get("/graph/topics")
def graph_topics():
    return {"topics": retriever.get_topics()}


@app.get("/graph/hot-terms", response_model=HotTermsResponse)
def graph_hot_terms(limit: int = Query(HOT_TERMS_LIMIT, ge=1, le=50)):
    return HotTermsResponse(**retriever.get_hot_terms(limit=limit))


@app.get("/graph/overview", response_model=GraphResponse)
def graph_overview(limit: int = Query(GRAPH_DEFAULT_LIMIT, ge=1, le=300)):
    data = retriever.get_overview(limit=limit)
    return GraphResponse(**data)


@app.get("/graph/search", response_model=GraphResponse)
def graph_search(
    keyword: str = Query(..., min_length=1),
    limit: int = Query(GRAPH_SEARCH_LIMIT, ge=1, le=300),
):
    keyword = normalize_text(keyword)
    if not keyword:
        raise HTTPException(status_code=400, detail="keyword 不能为空")
    data = retriever.search_subgraph(keyword=keyword, limit=limit)
    return GraphResponse(**data)


@app.get("/graph/topic/{topic}", response_model=GraphResponse)
def graph_topic(topic: str, limit: int = Query(GRAPH_TOPIC_LIMIT, ge=1, le=300)):
    topic = normalize_text(topic)
    if topic == "全部知识点":
        data = retriever.get_overview(limit=limit)
    else:
        data = retriever.get_topic_subgraph(topic=topic, limit=limit)
    return GraphResponse(**data)


@app.get("/graph/node/{node_name}", response_model=GraphResponse)
def graph_node_neighbors(node_name: str, limit: int = Query(GRAPH_NEIGHBOR_LIMIT, ge=1, le=300)):
    node_name = normalize_text(node_name)
    if not node_name:
        raise HTTPException(status_code=400, detail="node_name 不能为空")
    data = retriever.get_neighbors(node_name=node_name, limit=limit)
    return GraphResponse(**data)


@app.on_event("shutdown")
def shutdown_event():
    retriever.close()


# 依赖建议：
#   pip install -U fastapi uvicorn neo4j pydantic
# 运行方式：
#   uvicorn diabetes_graph_api_neo4j2026_explicit_reltype:app --host 0.0.0.0 --port 8001 --reload
# 常用环境变量：
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_USER=neo4j
#   NEO4J_PASSWORD=your_password
#   NEO4J_DATABASE=neo4j
