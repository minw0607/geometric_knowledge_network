from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx

from .graph_config import GraphRetrievalConfig
from .ingest import Chunk, Document


@dataclass(frozen=True)
class HotpotEntity:
    entity_id: str
    label: str
    node_type: str
    confidence: float


class HotpotEntityExtractor:
    def __init__(self, config: GraphRetrievalConfig | None = None) -> None:
        self.config = config or GraphRetrievalConfig()

    def extract_entities(self, chunk: Chunk, document_title: str) -> List[HotpotEntity]:
        text = chunk.text
        entities: Dict[str, HotpotEntity] = {}

        self._store_entity(entities, document_title, "TitleEntity", 1.0)

        for entity in self._extract_named_entities(text):
            token_count = len(entity.split())
            if token_count >= self.config.min_named_entity_tokens:
                self._store_entity(entities, entity, "NamedEntity", 0.9)

        for concept in self._extract_concepts(text):
            self._store_entity(entities, concept, "Concept", 0.6)

        return sorted(entities.values(), key=lambda item: (item.node_type, item.label))

    def _extract_named_entities(self, text: str) -> List[str]:
        matches = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        unique = []
        seen = set()
        for match in matches:
            normalized = match.strip()
            if normalized not in seen and not self._is_generic_entity(normalized):
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _extract_concepts(self, text: str) -> List[str]:
        concept_terms = [
            "actor",
            "director",
            "film",
            "book",
            "city",
            "country",
            "president",
            "governor",
            "river",
            "mountain",
            "university",
            "team",
        ]
        text_lower = text.lower()
        return [term for term in concept_terms if term in text_lower]

    def _store_entity(self, entities: Dict[str, HotpotEntity], label: str, node_type: str, confidence: float) -> None:
        normalized = label.strip().lower().replace(" ", "_")
        entity_id = f"{node_type.lower()}::{normalized}"
        entities[entity_id] = HotpotEntity(
            entity_id=entity_id,
            label=label.strip(),
            node_type=node_type,
            confidence=confidence,
        )

    def _is_generic_entity(self, label: str) -> bool:
        generic_terms = {
            "United States",
            "New York",
            "World War",
            "South Korea",
            "North America",
        }
        return label in generic_terms


class HotpotGraphBuilder:
    def __init__(self, extractor: HotpotEntityExtractor, config: GraphRetrievalConfig | None = None) -> None:
        self.extractor = extractor
        self.config = config or GraphRetrievalConfig()

    def build(self, documents: List[Document], chunks: List[Chunk]) -> nx.Graph:
        """Build the retrieval graph from observable corpus text only.

        IMPORTANT (benchmark fairness): the graph is constructed *exclusively*
        from documents, chunks, and entities extracted from chunk text. Gold
        HotpotQA supporting facts are deliberately NOT encoded here, because the
        evaluation defines relevance from those same supporting facts. Baking
        them into the retrieval graph would leak the answer key into the
        retriever and inflate the hybrid results. Supporting facts are used only
        downstream, for scoring (see HotpotRelevanceMapper).
        """
        graph = nx.Graph()
        doc_map = {doc.doc_id: doc for doc in documents}
        entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)

        for doc in documents:
            graph.add_node(
                doc.doc_id,
                node_type="Document",
                title=doc.title,
                source_path=doc.source_path,
            )
            title_entity_id = self._title_entity_id(doc.title)
            graph.add_node(
                title_entity_id,
                node_type="TitleEntity",
                label=doc.title,
                confidence=1.0,
            )
            graph.add_edge(doc.doc_id, title_entity_id, edge_type="HAS_TITLE", weight=self.config.contains_weight)

        for chunk in chunks:
            document = doc_map[chunk.doc_id]
            graph.add_node(
                chunk.chunk_id,
                node_type="Chunk",
                doc_id=chunk.doc_id,
                text=chunk.text,
            )
            graph.add_edge(chunk.doc_id, chunk.chunk_id, edge_type="CONTAINS", weight=self.config.contains_weight)

            title_entity_id = self._title_entity_id(document.title)
            graph.add_edge(chunk.chunk_id, title_entity_id, edge_type="LINKS_TO_TITLE", weight=self.config.title_link_weight)

            entities = self.extractor.extract_entities(chunk, document.title)
            for entity in entities:
                if not graph.has_node(entity.entity_id):
                    graph.add_node(
                        entity.entity_id,
                        node_type=entity.node_type,
                        label=entity.label,
                        confidence=entity.confidence,
                    )
                graph.add_edge(chunk.chunk_id, entity.entity_id, edge_type="MENTIONS", weight=self.config.mentions_weight)
                if entity.node_type == "NamedEntity":
                    entity_to_chunks[entity.entity_id].add(chunk.chunk_id)

        max_chunk_frequency = min(
            self.config.max_entity_chunk_frequency_absolute,
            max(2, int(len(chunks) * self.config.max_entity_document_frequency_ratio)),
        )

        for entity_id, chunk_ids in entity_to_chunks.items():
            if len(chunk_ids) > max_chunk_frequency:
                continue

            entity_label = graph.nodes[entity_id].get("label", "")
            entity_confidence = graph.nodes[entity_id].get("confidence", 0.0)
            if entity_confidence < self.config.min_named_entity_confidence:
                continue
            if len(entity_label.split()) < self.config.min_named_entity_tokens:
                continue

            # Corpus-only bridge edges: two chunks that share a distinctive named
            # entity are linked. This is query-independent on purpose, so the
            # same graph serves every question without being tuned to the eval set.
            chunk_list = sorted(chunk_ids)
            for i in range(len(chunk_list)):
                for j in range(i + 1, len(chunk_list)):
                    left, right = chunk_list[i], chunk_list[j]
                    if graph.has_edge(left, right):
                        continue
                    graph.add_edge(
                        left,
                        right,
                        edge_type="SHARES_ENTITY",
                        weight=self.config.shares_entity_edge_weight,
                        via_entity=entity_id,
                    )

        return graph

    def _title_entity_id(self, title: str) -> str:
        return f"titleentity::{self._normalize(title)}"

    def _normalize(self, text: str) -> str:
        return text.lower().replace(" ", "_").replace("/", "_")
