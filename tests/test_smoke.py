from pathlib import Path

from geometric_knowledge_network.config import GKNConfig
from geometric_knowledge_network.extraction import ConceptExtractor
from geometric_knowledge_network.graph_builder import KnowledgeNetworkBuilder
from geometric_knowledge_network.ingest import DocumentIngestor
from geometric_knowledge_network.vector_store import SimpleVectorStore


def test_smoke_pipeline():
    config = GKNConfig()
    ingestor = DocumentIngestor()
    docs = ingestor.load_text_documents(Path("data/sample_docs"))
    assert docs

    chunks = ingestor.chunk_documents(docs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    assert chunks

    vector_store = SimpleVectorStore()
    vector_store.build(chunks)
    results = vector_store.search("What evidence is required for validation approval?", top_k=2)
    assert results

    extractor = ConceptExtractor(config.concept_keywords)
    graph = KnowledgeNetworkBuilder(extractor).build(docs, chunks)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
