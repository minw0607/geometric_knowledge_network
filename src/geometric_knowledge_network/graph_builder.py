from __future__ import annotations


from itertools import combinations

import networkx as nx

from .extraction import ConceptExtractor
from .ingest import Chunk, Document
from .schema import EdgeType, NodeType


class KnowledgeNetworkBuilder:
    def __init__(self, extractor: ConceptExtractor) -> None:
        self.extractor = extractor

    def build(self, documents: list[Document], chunks: list[Chunk]) -> nx.Graph:
        graph = nx.Graph()

        for doc in documents:
            graph.add_node(
                doc.doc_id,
                node_type=NodeType.DOCUMENT.value,
                title=doc.title,
                source_path=doc.source_path,
            )

        for chunk in chunks:
            graph.add_node(
                chunk.chunk_id,
                node_type=NodeType.CHUNK.value,
                doc_id=chunk.doc_id,
                text=chunk.text,
            )
            graph.add_edge(
                chunk.doc_id,
                chunk.chunk_id,
                edge_type=EdgeType.CONTAINS.value,
            )

            concepts = self.extractor.extract_concepts(chunk)
            for concept in concepts:
                concept_id = f"concept::{concept}"
                if not graph.has_node(concept_id):
                    graph.add_node(
                        concept_id,
                        node_type=NodeType.CONCEPT.value,
                        label=concept,
                    )
                graph.add_edge(
                    chunk.chunk_id,
                    concept_id,
                    edge_type=EdgeType.MENTIONS.value,
                )

            for left, right in combinations(concepts, 2):
                left_id = f"concept::{left}"
                right_id = f"concept::{right}"
                if graph.has_edge(left_id, right_id):
                    continue
                graph.add_edge(
                    left_id,
                    right_id,
                    edge_type=EdgeType.RELATED_TO.value,
                )

        return graph
