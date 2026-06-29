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
                weight=1.0,
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
                    weight=self._mention_weight(entity),
                )

            for left, right in combinations(entities, 2):
                edge_type = self._infer_edge_type(left, right)
                if edge_type is None:
                    continue
                if graph.has_edge(left.entity_id, right.entity_id):
                    continue
                graph.add_edge(
                    left.entity_id,
                    right.entity_id,
                    edge_type=edge_type,
                    weight=self._edge_weight(edge_type),
                )

        return graph

    def _infer_edge_type(self, left: ExtractedEntity, right: ExtractedEntity) -> str | None:
        pair = {left.node_type, right.node_type}

        if NodeType.REQUIREMENT.value in pair and NodeType.CONTROL.value in pair:
            return EdgeType.REQUIRES.value
        if NodeType.EVIDENCE.value in pair and NodeType.REQUIREMENT.value in pair:
            return EdgeType.SUPPORTS.value
        if NodeType.INCIDENT.value in pair and NodeType.CONTROL.value in pair:
            return EdgeType.TRIGGERS.value
        if pair == {NodeType.CONTROL.value, NodeType.EVIDENCE.value}:
            return EdgeType.SUPPORTS.value
        if pair == {NodeType.REQUIREMENT.value, NodeType.EVIDENCE.value}:
            return EdgeType.SUPPORTS.value
        if pair == {NodeType.CONCEPT.value, NodeType.REQUIREMENT.value} and self._shared_signal(left, right):
            return EdgeType.RELATED_TO.value
        if pair == {NodeType.CONCEPT.value, NodeType.CONTROL.value} and self._shared_signal(left, right):
            return EdgeType.RELATED_TO.value
        return None

    def _shared_signal(self, left: ExtractedEntity, right: ExtractedEntity) -> bool:
        left_tokens = set(left.label.lower().split())
        right_tokens = set(right.label.lower().split())
        return bool(left_tokens & right_tokens) or left.confidence >= 0.85 or right.confidence >= 0.85

    def _mention_weight(self, entity: ExtractedEntity) -> float:
        base = {
            NodeType.REQUIREMENT.value: 1.0,
            NodeType.CONTROL.value: 1.0,
            NodeType.EVIDENCE.value: 0.95,
            NodeType.INCIDENT.value: 0.95,
            NodeType.CONCEPT.value: 0.65,
        }
        return base.get(entity.node_type, 0.75)

    def _edge_weight(self, edge_type: str) -> float:
        weights = {
            EdgeType.REQUIRES.value: 1.0,
            EdgeType.SUPPORTS.value: 0.95,
            EdgeType.TRIGGERS.value: 0.9,
            EdgeType.RELATED_TO.value: 0.45,
        }
        return weights.get(edge_type, 0.5)
