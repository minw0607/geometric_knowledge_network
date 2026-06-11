from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

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


class HybridRetriever:
    def __init__(self, vector_store: SimpleVectorStore, graph: nx.Graph) -> None:
        self.vector_store = vector_store
        self.graph = graph
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in vector_store.chunks}

    def search(self, query: str, top_k: int = 3, graph_hops: int = 2) -> List[HybridRetrievalResult]:
        baseline_results = self.vector_store.search(query, top_k=top_k)
        candidate_scores: Dict[str, HybridRetrievalResult] = {}

        for result in baseline_results:
            graph_bonus = self._graph_bonus(result.chunk_id, graph_hops=graph_hops)
            candidate_scores[result.chunk_id] = HybridRetrievalResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                text=result.text,
                vector_score=result.score,
                graph_bonus=graph_bonus,
                final_score=result.score + graph_bonus,
                source="vector",
            )

            for neighbor_chunk_id, neighbor_bonus in self._expand_neighbor_chunks(result.chunk_id, graph_hops=graph_hops).items():
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
                    )

        ranked = sorted(candidate_scores.values(), key=lambda item: item.final_score, reverse=True)
        return ranked[:top_k]

    def _expand_neighbor_chunks(self, chunk_id: str, graph_hops: int = 2) -> Dict[str, float]:
        if chunk_id not in self.graph:
            return {}

        lengths = nx.single_source_shortest_path_length(self.graph, chunk_id, cutoff=graph_hops)
        expanded_chunks: Dict[str, float] = {}

        for node_id, dist in lengths.items():
            if node_id == chunk_id or dist == 0:
                continue
            node_type = self.graph.nodes[node_id].get("node_type")
            if node_type == NodeType.CHUNK.value:
                expanded_chunks[node_id] = max(expanded_chunks.get(node_id, 0.0), 0.12 / dist)

        return expanded_chunks

    def _graph_bonus(self, chunk_id: str, graph_hops: int = 2) -> float:
        if chunk_id not in self.graph:
            return 0.0

        lengths = nx.single_source_shortest_path_length(self.graph, chunk_id, cutoff=graph_hops)
        structured_neighbor_count = 0
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
            }:
                structured_neighbor_count += 1
                related_count += max(0, self.graph.degree(node_id) - 1)

        return (0.02 * structured_neighbor_count) + (0.003 * related_count)
