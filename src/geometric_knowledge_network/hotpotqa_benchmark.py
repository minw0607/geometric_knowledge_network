from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import pandas as pd

from .evaluation import evaluate_retrieval
from .hotpotqa_loader import HotpotQASample
from .hotpotqa_relevance import HotpotRelevanceMapper
from .hybrid_retriever import HybridRetriever
from .path_explainer import GraphPathExplainer
from .vector_store import EmbeddingVectorStore
from .ingest import Chunk


@dataclass
class HotpotBenchmarkResult:
    evaluation_df: pd.DataFrame
    aggregate_df: pd.DataFrame
    paths_df: pd.DataFrame


class HotpotBenchmarkRunner:
    def __init__(
        self,
        vector_store: EmbeddingVectorStore,
        hybrid_retriever: HybridRetriever,
        chunks: List[Chunk],
        path_explainer: GraphPathExplainer,
    ) -> None:
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.chunks = chunks
        self.path_explainer = path_explainer
        self.relevance_mapper = HotpotRelevanceMapper()

    def run(self, samples: List[HotpotQASample], top_k: int, graph_hops: int) -> HotpotBenchmarkResult:
        evaluation_rows = []
        path_rows = []

        for sample in samples:
            relevant_chunk_ids = self.relevance_mapper.relevant_chunk_ids(sample, self.chunks)
            baseline_results = self.vector_store.search(sample.question, top_k=top_k)
            hybrid_results = self.hybrid_retriever.search(sample.question, top_k=top_k, graph_hops=graph_hops)

            baseline_metrics = evaluate_retrieval(baseline_results, relevant_chunk_ids)
            hybrid_metrics = evaluate_retrieval(hybrid_results, relevant_chunk_ids)

            evaluation_rows.append(
                {
                    "question_id": sample.question_id,
                    "question": sample.question,
                    "category": "hotpotqa",
                    "retriever": "baseline",
                    **baseline_metrics,
                }
            )
            evaluation_rows.append(
                {
                    "question_id": sample.question_id,
                    "question": sample.question,
                    "category": "hotpotqa",
                    "retriever": "hybrid",
                    **hybrid_metrics,
                }
            )

            for result in hybrid_results:
                if result.source == "graph_expansion" and result.seed_chunk_id:
                    explanation = self.path_explainer.explain(result.seed_chunk_id, result.chunk_id, max_hops=4)
                    path_rows.append(
                        {
                            "question_id": sample.question_id,
                            "question": sample.question,
                            "seed_chunk_id": result.seed_chunk_id,
                            "target_chunk_id": result.chunk_id,
                            "path_exists": explanation.get("path_exists", False),
                            "hop_count": explanation.get("hop_count", 0),
                            "edge_types": " | ".join(explanation.get("edge_types", [])),
                            "path_labels": " | ".join(node["label"] for node in explanation.get("path", [])),
                        }
                    )

        evaluation_df = pd.DataFrame(evaluation_rows)
        aggregate_df = evaluation_df.groupby("retriever")[["hit_rate", "recall_at_k", "precision_at_k", "mrr"]].mean().reset_index()
        paths_df = pd.DataFrame(path_rows) if path_rows else pd.DataFrame(
            columns=["question_id", "question", "seed_chunk_id", "target_chunk_id", "path_exists", "hop_count", "edge_types", "path_labels"]
        )
        return HotpotBenchmarkResult(
            evaluation_df=evaluation_df,
            aggregate_df=aggregate_df,
            paths_df=paths_df,
        )


def hotpot_sample_signature(samples: List[HotpotQASample]) -> str:
    joined = "|".join(sorted(sample.question_id for sample in samples))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
