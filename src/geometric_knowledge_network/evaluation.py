from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set

from .hybrid_retriever import HybridRetrievalResult
from .vector_store import RetrievalResult


@dataclass
class QueryTestCase:
    query: str
    relevant_chunk_ids: Set[str]
    category: str = "general"
    description: str = ""


def load_query_test_cases(filepath: Path) -> List[QueryTestCase]:
    payload = json.loads(filepath.read_text(encoding="utf-8"))
    return [
        QueryTestCase(
            query=item["query"],
            relevant_chunk_ids=set(item["relevant_chunk_ids"]),
            category=item.get("category", "general"),
            description=item.get("description", ""),
        )
        for item in payload
    ]


def top_k_hit_rate(results: Iterable[RetrievalResult | HybridRetrievalResult], relevant_chunk_ids: Set[str]) -> float:
    result_ids = {result.chunk_id for result in results}
    return float(bool(result_ids & relevant_chunk_ids))


def recall_at_k(results: Iterable[RetrievalResult | HybridRetrievalResult], relevant_chunk_ids: Set[str]) -> float:
    if not relevant_chunk_ids:
        return 0.0
    result_ids = {result.chunk_id for result in results}
    return len(result_ids & relevant_chunk_ids) / len(relevant_chunk_ids)


def precision_at_k(results: List[RetrievalResult | HybridRetrievalResult], relevant_chunk_ids: Set[str]) -> float:
    if not results:
        return 0.0
    hits = sum(1 for result in results if result.chunk_id in relevant_chunk_ids)
    return hits / len(results)


def reciprocal_rank(results: List[RetrievalResult | HybridRetrievalResult], relevant_chunk_ids: Set[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(results: List[RetrievalResult | HybridRetrievalResult], relevant_chunk_ids: Set[str]) -> dict:
    return {
        "hit_rate": top_k_hit_rate(results, relevant_chunk_ids),
        "recall_at_k": recall_at_k(results, relevant_chunk_ids),
        "precision_at_k": precision_at_k(results, relevant_chunk_ids),
        "mrr": reciprocal_rank(results, relevant_chunk_ids),
    }
