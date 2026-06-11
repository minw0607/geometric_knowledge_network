from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    return current.parents[2]


@dataclass
class GKNConfig:
    repo_root: Path = field(default_factory=_resolve_repo_root)
    sample_docs_dir: Path = field(default_factory=lambda: _resolve_repo_root() / "data/sample_docs")
    eval_queries_path: Path = field(default_factory=lambda: _resolve_repo_root() / "data/eval_queries.json")
    artifacts_dir: Path = field(default_factory=lambda: _resolve_repo_root() / "artifacts")
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 3
    graph_hops: int = 2
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
