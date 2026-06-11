from __future__ import annotations

from itertools import combinations

import networkx as nx

from .extraction import ConceptExtractor, ExtractedEntity
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

            entities = self.extractor.extract_entities(chunk)
            for entity in entities:
                if not graph.has_node(entity.entity_id):
                    graph.add_node(
                        entity.entity_id,
                        node_type=entity.node_type,
                        label=entity.label,
                        confidence=entity.confidence,
                    )
                graph.add_edge(
                    chunk.chunk_id,
                    entity.entity_id,
                    edge_type=EdgeType.MENTIONS.value,
                )

            for left, right in combinations(entities, 2):
                if graph.has_edge(left.entity_id, right.entity_id):
                    continue
                graph.add_edge(
                    left.entity_id,
                    right.entity_id,
                    edge_type=self._infer_edge_type(left, right),
                )

        return graph

    def _infer_edge_type(self, left: ExtractedEntity, right: ExtractedEntity) -> str:
        pair = {left.node_type, right.node_type}

        if NodeType.REQUIREMENT.value in pair and NodeType.CONTROL.value in pair:
            return EdgeType.REQUIRES.value
        if NodeType.EVIDENCE.value in pair and NodeType.REQUIREMENT.value in pair:
            return EdgeType.SUPPORTS.value
        if NodeType.INCIDENT.value in pair and NodeType.CONTROL.value in pair:
            return EdgeType.TRIGGERS.value
        return EdgeType.RELATED_TO.value
