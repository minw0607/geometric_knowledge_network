"""Benchmark comparing baseline vs hybrid vs multi-hop retrieval on HotpotQA.

Reports both chunk-level metrics (hit_rate / recall@k / precision@k / MRR) and the
document-level multi-hop metrics (doc_recall, all_docs_hit) that actually capture
whether the second "bridge" supporting document was recovered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .evaluation import evaluate_retrieval
from .hotpotqa_loader import HotpotQASample
from .hotpotqa_relevance import HotpotRelevanceMapper
from .hybrid_retriever import HybridRetriever
from .ingest import Chunk
from .multihop_retriever import MultiHopRetriever
from .vector_store import EmbeddingVectorStore, SimpleVectorStore


@dataclass
class MultiHopBenchmarkResult:
    evaluation_df: pd.DataFrame
    aggregate_df: pd.DataFrame


class MultiHopBenchmarkRunner:
    def __init__(
        self,
        vector_store: EmbeddingVectorStore | SimpleVectorStore,
        hybrid_retriever: HybridRetriever,
        multihop_retriever: MultiHopRetriever,
        chunks: List[Chunk],
    ) -> None:
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.multihop_retriever = multihop_retriever
        self.chunks = chunks
        self.relevance_mapper = HotpotRelevanceMapper()

    def run(self, samples: List[HotpotQASample], top_k: int = 10, graph_hops: int = 2) -> MultiHopBenchmarkResult:
        rows = []
        for sample in samples:
            relevant_chunk_ids = self.relevance_mapper.relevant_chunk_ids(sample, self.chunks)
            relevant_doc_ids = self.relevance_mapper.relevant_doc_ids(sample)

            retriever_outputs = {
                "baseline": self.vector_store.search(sample.question, top_k=top_k),
                "hybrid": self.hybrid_retriever.search(sample.question, top_k=top_k, graph_hops=graph_hops),
                "multihop": self.multihop_retriever.search(sample.question, top_k=top_k),
            }

            for retriever, results in retriever_outputs.items():
                metrics = evaluate_retrieval(results, relevant_chunk_ids, relevant_doc_ids)
                rows.append(
                    {
                        "question_id": sample.question_id,
                        "question": sample.question,
                        "retriever": retriever,
                        **metrics,
                    }
                )

        evaluation_df = pd.DataFrame(rows)
        metric_cols = ["hit_rate", "recall_at_k", "precision_at_k", "mrr", "doc_recall", "all_docs_hit"]
        aggregate_df = (
            evaluation_df.groupby("retriever")[metric_cols]
            .mean()
            .reindex(["baseline", "hybrid", "multihop"])
            .reset_index()
        )
        return MultiHopBenchmarkResult(evaluation_df=evaluation_df, aggregate_df=aggregate_df)
