"""Offline tests for the multi-hop / PPR retriever and document-level metrics."""
from __future__ import annotations

from geometric_knowledge_network.evaluation import (
    all_supporting_docs_hit,
    document_recall,
    evaluate_retrieval,
)
from geometric_knowledge_network.hotpotqa_graph import HotpotEntityExtractor, HotpotGraphBuilder
from geometric_knowledge_network.hotpotqa_relevance import HotpotRelevanceMapper
from geometric_knowledge_network.hybrid_retriever import HybridRetriever
from geometric_knowledge_network.multihop_benchmark import MultiHopBenchmarkRunner
from geometric_knowledge_network.multihop_retriever import MultiHopRetriever
from geometric_knowledge_network.vector_store import SimpleVectorStore

from tests.test_hotpotqa import _build_corpus


def test_document_level_metrics():
    class R:
        def __init__(self, doc_id):
            self.doc_id = doc_id
            self.chunk_id = doc_id + "_c0"

    results = [R("a"), R("b")]
    assert document_recall(results, {"a", "b"}) == 1.0
    assert document_recall(results, {"a", "c"}) == 0.5
    assert all_supporting_docs_hit(results, {"a", "b"}) == 1.0
    assert all_supporting_docs_hit(results, {"a", "c"}) == 0.0


def test_multihop_search_does_not_demote_first_vector_hit():
    documents, chunks, _samples = _build_corpus()
    graph = HotpotGraphBuilder(HotpotEntityExtractor()).build(documents, chunks)
    vector_store = SimpleVectorStore()
    vector_store.build(chunks)
    multihop = MultiHopRetriever(vector_store, graph)

    query = "Which film directed by Scott Derrickson is a horror film?"
    baseline = vector_store.search(query, top_k=5)
    results = multihop.search(query, top_k=5, bridge_budget=2)

    assert len(results) <= 5
    # add-not-demote: the top vector hit keeps rank 1
    assert results[0].chunk_id == baseline[0].chunk_id


def test_multihop_benchmark_reports_doc_metrics():
    documents, chunks, samples = _build_corpus()
    graph = HotpotGraphBuilder(HotpotEntityExtractor()).build(documents, chunks)
    vector_store = SimpleVectorStore()
    vector_store.build(chunks)
    hybrid = HybridRetriever(vector_store, graph)
    multihop = MultiHopRetriever(vector_store, graph)

    result = MultiHopBenchmarkRunner(vector_store, hybrid, multihop, chunks).run(samples, top_k=5)
    assert list(result.aggregate_df["retriever"]) == ["baseline", "hybrid", "multihop"]
    for column in ["hit_rate", "recall_at_k", "precision_at_k", "mrr", "doc_recall", "all_docs_hit"]:
        assert column in result.aggregate_df.columns


def test_evaluate_retrieval_doc_metrics_optional():
    documents, chunks, samples = _build_corpus()
    mapper = HotpotRelevanceMapper()
    rel_chunks = mapper.relevant_chunk_ids(samples[0], chunks)
    rel_docs = mapper.relevant_doc_ids(samples[0])

    vector_store = SimpleVectorStore()
    vector_store.build(chunks)
    results = vector_store.search(samples[0].question, top_k=5)

    without_docs = evaluate_retrieval(results, rel_chunks)
    with_docs = evaluate_retrieval(results, rel_chunks, rel_docs)
    assert "doc_recall" not in without_docs
    assert "doc_recall" in with_docs and "all_docs_hit" in with_docs
