from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class GKNConfig:
    sample_docs_dir: Path = Path("data/sample_docs")
    artifacts_dir: Path = Path("artifacts")
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 3
    graph_hops: int = 1
    concept_keywords: List[str] = field(
        default_factory=lambda: [
            "requirement",
            "control",
            "evidence",
            "validation",
            "monitoring",
            "approval",
            "incident",
            "review",
            "drift",
            "risk",
            "testing",
            "policy",
        ]
    )
