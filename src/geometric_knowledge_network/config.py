import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    return current.parents[2]


load_dotenv(_resolve_repo_root() / ".env")


@dataclass
class GKNConfig:
    repo_root: Path = field(default_factory=_resolve_repo_root)
    sample_docs_dir: Path = field(default_factory=lambda: _resolve_repo_root() / "data/sample_docs")
    eval_queries_path: Path = field(default_factory=lambda: _resolve_repo_root() / "data/eval_queries.json")
    artifacts_dir: Path = field(default_factory=lambda: _resolve_repo_root() / "artifacts")
    checkpoints_dir: Path = field(default_factory=lambda: _resolve_repo_root() / "checkpoints")
    hotpotqa_data_path: Path = field(default_factory=lambda: Path(os.getenv("HOTPOTQA_DATA_PATH", _resolve_repo_root() / "data/hotpot_train_v1.1.json")))
    hotpotqa_sample_size: int = field(default_factory=lambda: int(os.getenv("HOTPOTQA_SAMPLE_SIZE", "100")))
    hotpotqa_random_seed: int = field(default_factory=lambda: int(os.getenv("HOTPOTQA_RANDOM_SEED", "42")))
    force_rebuild_vector_store: bool = field(default_factory=lambda: os.getenv("FORCE_REBUILD_VECTOR_STORE", "false").lower() == "true")
    embedding_choice: str = field(default_factory=lambda: os.getenv("EMBEDDING_CHOICE", "small").lower())
    local_embedding_model: str = field(default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "all-mpnet-base-v2"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_version: str = field(default_factory=lambda: os.getenv("OPENAI_API_VERSION", ""))
    openai_apim_header_name: str = field(default_factory=lambda: os.getenv("OPENAI_APIM_HEADER_NAME", ""))
    openai_apim_subscription_key: str = field(default_factory=lambda: os.getenv("OPENAI_APIM_SUBSCRIPTION_KEY", ""))
    openai_generation_model: str = field(default_factory=lambda: os.getenv("OPENAI_GENERATION_MODEL", ""))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
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
