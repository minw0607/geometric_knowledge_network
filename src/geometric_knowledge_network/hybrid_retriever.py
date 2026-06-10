from __future__ import annotations

from dataclasses import dataclass
from typing import List

import networkx as nx

from .vector_store import RetrievalResult, SimpleVectorStore


@dataclass
class HybridRetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    vector_score: float
    graph_bonus: float
    final_score: float


class HybridRetriever:
    def __init__(self, vector_store: SimpleVectorStore, graph: nx.Graph) -> None:
        self.vector_store = vector_store
        self.graph = graph

    def search(self, query: str, top_k: int = 3, graph_hops: int = 1) -> List[HybridRetrievalResult]:
        baseline_results = self.vector_store.search(query, top_k=top_k)
        boosted_results: List[HybridRetrievalResult] = []

        for result in baseline_results:
            graph_bonus = self._graph_bonus(result.chunk_id, graph_hops=graph_hops)
            final_score = result.score + graph_bonus
            boosted_results.append(
                HybridRetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    text=result.text,
                    vector_score=result.score,
                    graph_bonus=graph_bonus,
                    final_score=final_score,
                )
            )

        boosted_results.sort(key=lambda item: item.final_score, reverse=True)
        return boosted_results

    def _graph_bonus(self, chunk_id: str, graph_hops: int = 1) -> float:
        if chunk_id not in self.graph:
            return 0.0

        lengths = nx.single_source_shortest_path_length(self.graph, chunk_id, cutoff=graph_hops)
        concept_count = 0
        related_count = 0

        for node_id, dist in lengths.items():
            if node_id == chunk_id or dist == 0:
                continue
            node_data = self.graph.nodes[node_id]
            if node_data.get("node_type") == "Concept":
                concept_count += 1
                related_count += max(0, self.graph.degree(node_id) - 1)

        return (0.02 * concept_count) + (0.005 * related_count)
