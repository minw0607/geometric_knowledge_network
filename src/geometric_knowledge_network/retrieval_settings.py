from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphRetrievalSettings:
    expansion_base_score: float = 0.12
    graph_bonus_weight: float = 0.03
    graph_degree_weight: float = 0.002
    max_explanation_hops: int = 4
    allowed_structured_node_types: set[str] = field(
        default_factory=lambda: {
            "Requirement",
            "Control",
            "Evidence",
            "Incident",
            "Concept",
            "TitleEntity",
            "NamedEntity",
            "SupportingFact",
        }
    )
