"""Agentic multi-hop retriever with a Personalized PageRank scorer.

Motivation
----------
Vector-only retrieval is strong at the *first* hop (chunks semantically close to
the query) but weak at the *second* "bridge" hop — a supporting document that is
reachable only through a shared entity and is often semantically distant from the
query. This retriever demonstrates where a knowledge network helps:

1. Hop 1 (vector): take the dense top-k. These are kept in their original order
   so the first relevant hit is never demoted (preserves MRR).
2. Bridge discovery (graph): seed a localized **Personalized PageRank** on the
   hop-1 chunks plus the query-matching entities adjacent to them, then propagate
   relevance across the graph. The highest-scoring chunks from *new* documents are
   added to fill a small bridge budget.

This is "add-not-demote": graph structure can only *add* bridge documents into the
result set; it never reshuffles a strong vector hit downward. PPR replaces the
earlier ad-hoc, query-agnostic graph bonus with a principled diffusion process
(random walk with restart) seeded on query-relevant nodes.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

import networkx as nx

from .graph_config import GraphRetrievalConfig
from .hybrid_retriever import HybridRetrievalResult, VectorStoreLike
from .schema import NodeType

_BRIDGE_ENTITY_TYPES = {"NamedEntity", "TitleEntity", NodeType.CONCEPT.value}


class MultiHopRetriever:
    def __init__(
        self,
        vector_store: VectorStoreLike,
        graph: nx.Graph,
        config: GraphRetrievalConfig | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.graph = graph
        self.config = config or GraphRetrievalConfig()
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in vector_store.chunks}

    def search(
        self,
        query: str,
        top_k: int = 10,
        bridge_budget: int | None = None,
        ppr_alpha: float = 0.85,
        ego_radius: int = 2,
        max_subgraph_nodes: int = 6000,
    ) -> List[HybridRetrievalResult]:
        """Return up to ``top_k`` results: vector hits first, then PPR bridges.

        ``bridge_budget`` slots at the tail are reserved for graph-discovered
        bridge chunks from documents not already covered by the vector hits.
        """
        if bridge_budget is None:
            bridge_budget = max(1, top_k // 3)
        vector_keep = max(1, top_k - bridge_budget)

        query_terms = self._query_terms(query)
        vector_results = self.vector_store.search(query, top_k=top_k)

        # Hop 1: keep vector hits in order (never demoted).
        core: List[HybridRetrievalResult] = []
        covered_docs: Set[str] = set()
        for result in vector_results[:vector_keep]:
            core.append(
                HybridRetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    text=result.text,
                    vector_score=result.score,
                    graph_bonus=0.0,
                    final_score=result.score,
                    source="vector",
                    seed_chunk_id=result.chunk_id,
                )
            )
            covered_docs.add(result.doc_id)
        core_ids = {item.chunk_id for item in core}

        # Bridge discovery via Personalized PageRank seeded on the hop-1 region.
        ppr_scores = self._personalized_pagerank(
            vector_results, query_terms, ppr_alpha=ppr_alpha, ego_radius=ego_radius, max_subgraph_nodes=max_subgraph_nodes
        )
        bridges = self._select_bridges(ppr_scores, core_ids, covered_docs, bridge_budget)

        # If PPR surfaced fewer new-document bridges than the budget, backfill from
        # the remaining vector hits so we still return up to top_k results.
        results = core + bridges
        if len(results) < top_k:
            present = {item.chunk_id for item in results}
            for result in vector_results:
                if len(results) >= top_k:
                    break
                if result.chunk_id in present:
                    continue
                results.append(
                    HybridRetrievalResult(
                        chunk_id=result.chunk_id,
                        doc_id=result.doc_id,
                        text=result.text,
                        vector_score=result.score,
                        graph_bonus=0.0,
                        final_score=result.score,
                        source="vector",
                        seed_chunk_id=result.chunk_id,
                    )
                )
                present.add(result.chunk_id)

        return results[:top_k]

    # ------------------------------------------------------------------ PPR ---
    def _personalized_pagerank(
        self,
        vector_results,
        query_terms: Set[str],
        ppr_alpha: float,
        ego_radius: int,
        max_subgraph_nodes: int,
    ) -> Dict[str, float]:
        """Localized random-walk-with-restart, seeded on hop-1 chunks + query entities."""
        seeds: Dict[str, float] = {}
        for result in vector_results:
            if result.chunk_id in self.graph:
                # Seed weight follows the vector score so stronger hits restart more often.
                seeds[result.chunk_id] = seeds.get(result.chunk_id, 0.0) + max(result.score, 1e-3)
                # Query-matching entities adjacent to this chunk are strong bridge anchors.
                for neighbor in self.graph.neighbors(result.chunk_id):
                    node = self.graph.nodes[neighbor]
                    if node.get("node_type") in _BRIDGE_ENTITY_TYPES:
                        label = str(node.get("label") or "").lower()
                        if query_terms and any(term in label for term in query_terms):
                            seeds[neighbor] = seeds.get(neighbor, 0.0) + self.config.query_overlap_bonus

        if not seeds:
            return {}

        # Build a localized subgraph so PPR is fast and focused on the reachable region.
        nodes: Set[str] = set()
        for seed_node in seeds:
            nodes.update(nx.single_source_shortest_path_length(self.graph, seed_node, cutoff=ego_radius).keys())
            if len(nodes) > max_subgraph_nodes:
                break
        subgraph = self.graph.subgraph(nodes)
        if subgraph.number_of_nodes() == 0:
            return {}

        personalization = {node: seeds.get(node, 0.0) for node in subgraph.nodes}
        total = sum(personalization.values())
        if total <= 0:
            return {}
        personalization = {node: weight / total for node, weight in personalization.items()}

        try:
            ranks = nx.pagerank(
                subgraph,
                alpha=ppr_alpha,
                personalization=personalization,
                weight="weight",
                max_iter=100,
                tol=1.0e-6,
            )
        except nx.PowerIterationFailedConvergence:
            return {}

        return {
            node: score
            for node, score in ranks.items()
            if self.graph.nodes[node].get("node_type") == NodeType.CHUNK.value
        }

    def _select_bridges(
        self,
        ppr_scores: Dict[str, float],
        core_ids: Set[str],
        covered_docs: Set[str],
        bridge_budget: int,
    ) -> List[HybridRetrievalResult]:
        """Pick the highest-PPR chunks from documents not already covered."""
        candidates = [
            (chunk_id, score)
            for chunk_id, score in ppr_scores.items()
            if chunk_id not in core_ids and chunk_id in self.chunk_lookup
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)

        bridges: List[HybridRetrievalResult] = []
        used_docs = set(covered_docs)
        for chunk_id, score in candidates:
            if len(bridges) >= bridge_budget:
                break
            chunk = self.chunk_lookup[chunk_id]
            # Prefer bringing in *new* documents (the bridge hop).
            if chunk.doc_id in used_docs:
                continue
            bridges.append(
                HybridRetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    vector_score=0.0,
                    graph_bonus=float(score),
                    final_score=float(score),
                    source="graph_ppr",
                    seed_chunk_id=None,
                )
            )
            used_docs.add(chunk.doc_id)
        return bridges

    def _query_terms(self, query: str) -> Set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9\-]+", query)
            if token.lower() not in self.config.question_stopwords and len(token) >= 4
        }
