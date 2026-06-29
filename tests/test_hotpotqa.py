"""Offline tests for the HotpotQA retrieval path.

These tests build a tiny in-memory corpus so they run without downloading the
HotpotQA dataset and without faiss/openai (they use the TF-IDF SimpleVectorStore).
The key assertion is that gold supporting facts never leak into the retrieval
graph.
"""
from __future__ import annotations

from hashlib import sha256

from geometric_knowledge_network.hotpotqa_benchmark import HotpotBenchmarkRunner
from geometric_knowledge_network.hotpotqa_graph import HotpotEntityExtractor, HotpotGraphBuilder
from geometric_knowledge_network.hotpotqa_loader import HotpotQASample
from geometric_knowledge_network.hotpotqa_relevance import HotpotRelevanceMapper
from geometric_knowledge_network.hybrid_retriever import HybridRetriever
from geometric_knowledge_network.ingest import Chunk, Document, DocumentIngestor
from geometric_knowledge_network.path_explainer import GraphPathExplainer
from geometric_knowledge_network.vector_store import SimpleVectorStore


def _build_corpus():
    raw = {
        "Scott Derrickson": "Scott Derrickson is an American director and producer. "
        "Scott Derrickson directed the film Sinister.",
        "Ed Wood": "Ed Wood was an American director and producer. "
        "Ed Wood is known for low budget films.",
        "Sinister (film)": "Sinister is a 2012 horror film. "
        "Sinister was directed by Scott Derrickson.",
    }
    documents = []
    for title, text in raw.items():
        # Mirror HotpotQALoader._normalize_title (only space and slash replaced).
        doc_id = title.lower().replace(" ", "_").replace("/", "_")
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                text=title + "\n\n" + text,
                source_path="memory",
                text_hash=sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    chunks = DocumentIngestor().chunk_documents(documents, chunk_size=500, chunk_overlap=80)
    samples = [
        HotpotQASample(
            question_id="q1",
            question="Which film directed by Scott Derrickson is a horror film?",
            answer="Sinister",
            supporting_titles=["Scott Derrickson", "Sinister (film)"],
            supporting_sent_ids=[("Scott Derrickson", 1), ("Sinister (film)", 1)],
        )
    ]
    return documents, chunks, samples


def test_graph_has_no_gold_label_leakage():
    documents, chunks, _samples = _build_corpus()
    graph = HotpotGraphBuilder(HotpotEntityExtractor()).build(documents, chunks)

    node_types = {data.get("node_type") for _, data in graph.nodes(data=True)}
    edge_types = {data.get("edge_type") for _, _, data in graph.edges(data=True)}

    # Gold supporting facts must never be encoded in the retrieval graph.
    assert "SupportingFact" not in node_types
    assert "SUPPORTS" not in edge_types
    # The corpus-derived structure should still be present.
    assert "Chunk" in node_types
    assert "TitleEntity" in node_types
    assert graph.number_of_edges() > 0


def test_hotpot_benchmark_runs_offline():
    documents, chunks, samples = _build_corpus()
    graph = HotpotGraphBuilder(HotpotEntityExtractor()).build(documents, chunks)

    vector_store = SimpleVectorStore()
    vector_store.build(chunks)
    hybrid = HybridRetriever(vector_store, graph)
    benchmark = HotpotBenchmarkRunner(vector_store, hybrid, chunks, GraphPathExplainer(graph))

    result = benchmark.run(samples, top_k=3, graph_hops=2)
    assert not result.evaluation_df.empty
    assert set(result.aggregate_df["retriever"]) == {"baseline", "hybrid"}
    for column in ["hit_rate", "recall_at_k", "precision_at_k", "mrr"]:
        assert column in result.aggregate_df.columns


def test_relevance_mapper_uses_supporting_titles():
    documents, chunks, samples = _build_corpus()
    relevant = HotpotRelevanceMapper().relevant_chunk_ids(samples[0], chunks)

    relevant_doc_ids = {chunk.doc_id for chunk in chunks if chunk.chunk_id in relevant}
    assert "scott_derrickson" in relevant_doc_ids
    assert "sinister_(film)" in relevant_doc_ids
    assert "ed_wood" not in relevant_doc_ids
