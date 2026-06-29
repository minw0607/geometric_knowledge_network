from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx

from .graph_config import GraphRetrievalConfig
from .schema import NodeType
from .vector_store import RetrievalResult, SimpleVectorStore


@dataclass
class HybridRetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    vector_score: float
    graph_bonus: float
    final_score: float
    source: str
    seed_chunk_id: str | None = None


class HybridRetriever:
    def __init__(self, vector_store: SimpleVectorStore, graph: nx.Graph, config: GraphRetrievalConfig | None = None) -> None:
        self.vector_store = vector_store
        self.graph = graph
        self.config = config or GraphRetrievalConfig()
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in vector_store.chunks}

    def search(self, query: str, top_k: int = 3, graph_hops: int = 2) -> List[HybridRetrievalResult]:
        baseline_results = self.vector_store.search(query, top_k=top_k)
        candidate_scores: Dict[str, HybridRetrievalResult] = {}
        query_terms = self._query_terms(query)

        for result in baseline_results:
            graph_bonus = self._graph_bonus(result.chunk_id, query_terms=query_terms, graph_hops=graph_hops)
            candidate_scores[result.chunk_id] = HybridRetrievalResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                text=result.text,
                vector_score=result.score,
                graph_bonus=graph_bonus,
                final_score=result.score + graph_bonus,
                source="vector",
                seed_chunk_id=result.chunk_id,
            )

            for neighbor_chunk_id, neighbor_bonus in self._expand_neighbor_chunks(result.chunk_id, query_terms=query_terms, graph_hops=graph_hops).items():
                if neighbor_chunk_id == result.chunk_id:
                    continue
                chunk = self.chunk_lookup.get(neighbor_chunk_id)
                if chunk is None:
                    continue

                existing = candidate_scores.get(neighbor_chunk_id)
                if existing is None or neighbor_bonus > existing.graph_bonus:
                    candidate_scores[neighbor_chunk_id] = HybridRetrievalResult(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        vector_score=0.0,
                        graph_bonus=neighbor_bonus,
                        final_score=neighbor_bonus,
                        source="graph_expansion",
                        seed_chunk_id=result.chunk_id,
                    )

        ranked = sorted(candidate_scores.values(), key=lambda item: item.final_score, reverse=True)
        return ranked[:top_k]

    def _expand_neighbor_chunks(self, chunk_id: str, query_terms: Set[str], graph_hops: int = 2) -> Dict[str, float]:
        if chunk_id not in self.graph:
            return {}

        lengths = nx.single_source_shortest_path_length(self.graph, chunk_id, cutoff=graph_hops)
        expanded_chunks: Dict[str, float] = {}

        for node_id, dist in lengths.items():
            if node_id == chunk_id or dist == 0:
                continue
            node_type = self.graph.nodes[node_id].get("node_type")
            if node_type == NodeType.CHUNK.value:
                edge_weight = 1.0
                query_bonus = 1.0
                try:
                    path = nx.shortest_path(self.graph, source=chunk_id, target=node_id)
                    edge_weights = []
                    path_labels = []
                    path_edge_types = []
                    for left, right in zip(path[:-1], path[1:]):
                        edge_data = self.graph.edges[left, right]
                        edge_weights.append(edge_data.get("weight", 1.0))
                        path_edge_types.append(edge_data.get("edge_type", "Unknown"))
                    for node in path:
                        path_labels.append(str(self.graph.nodes[node].get("label") or self.graph.nodes[node].get("title") or "").lower())
                    edge_weight = sum(edge_weights) / max(1, len(edge_weights))
                    if any(term in " ".join(path_labels) for term in query_terms):
                        query_bonus += self.config.query_overlap_bonus
                    if "SUPPORTS" in path_edge_types:
                        query_bonus += 0.05
                    if "SHARES_ENTITY" in path_edge_types:
                        query_bonus -= 0.03
                except nx.NetworkXNoPath:
                    edge_weight = 0.5
                expanded_chunks[node_id] = max(expanded_chunks.get(node_id, 0.0), ((self.config.path_edge_weight_scale / dist) * edge_weight) * query_bonus)

        return expanded_chunks

    def _graph_bonus(self, chunk_id: str, query_terms: Set[str], graph_hops: int = 2) -> float:
        if chunk_id not in self.graph:
            return 0.0

        lengths = nx.single_source_shortest_path_length(self.graph, chunk_id, cutoff=graph_hops)
        structured_neighbor_score = 0.0
        related_count = 0

        for node_id, dist in lengths.items():
            if node_id == chunk_id or dist == 0:
                continue
            node_type = self.graph.nodes[node_id].get("node_type")
            if node_type in {
                NodeType.REQUIREMENT.value,
                NodeType.CONTROL.value,
                NodeType.EVIDENCE.value,
                NodeType.INCIDENT.value,
                NodeType.CONCEPT.value,
                "TitleEntity",
                "NamedEntity",
                "SupportingFact",
            }:
                node_label = str(self.graph.nodes[node_id].get("label") or self.graph.nodes[node_id].get("title") or "").lower()
                node_weight = self.graph.nodes[node_id].get("confidence", 0.8)
                if any(term in node_label for term in query_terms):
                    node_weight += self.config.query_overlap_bonus
                structured_neighbor_score += node_weight / dist
                related_count += max(0, self.graph.degree(node_id) - 1)

        return (self.config.graph_bonus_scale * structured_neighbor_score) + (self.config.graph_degree_scale * related_count)

    def _query_terms(self, query: str) -> Set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9\-]+", query)
            if token.lower() not in self.config.question_stopwords and len(token) >= 4
        }
