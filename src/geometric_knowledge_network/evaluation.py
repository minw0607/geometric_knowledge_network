from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Set

from .hybrid_retriever import HybridRetrievalResult
from .vector_store import RetrievalResult


@dataclass
class QueryTestCase:
    query: str
    relevant_chunk_ids: Set[str]


def top_k_hit_rate(results: Iterable[RetrievalResult | HybridRetrievalResult], relevant_chunk_ids: Set[str]) -> float:
    result_ids = {result.chunk_id for result in results}
    return float(bool(result_ids & relevant_chunk_ids))


def reciprocal_rank(results: List[RetrievalResult | HybridRetrievalResult], relevant_chunk_ids: Set[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0
