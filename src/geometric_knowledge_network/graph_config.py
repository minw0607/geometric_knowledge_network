from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class GraphRetrievalConfig:
    min_named_entity_tokens: int = 2
    min_named_entity_confidence: float = 0.85
    max_entity_document_frequency_ratio: float = 0.02
    max_entity_chunk_frequency_absolute: int = 40
    shares_entity_edge_weight: float = 0.35
    title_link_weight: float = 0.95
    mentions_weight: float = 0.9
    supports_weight: float = 1.0
    contains_weight: float = 1.0
    max_graph_hops: int = 4
    graph_bonus_scale: float = 0.03
    graph_degree_scale: float = 0.002
    query_overlap_bonus: float = 0.08
    path_edge_weight_scale: float = 0.12
    require_query_overlap_for_expansion: bool = True
    require_supporting_fact_hint: bool = False
    question_stopwords: Set[str] = field(
        default_factory=lambda: {
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "of",
            "in",
            "to",
            "for",
            "and",
            "or",
            "did",
            "do",
            "does",
        }
    )
