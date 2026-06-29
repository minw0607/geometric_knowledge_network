from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .ingest import Chunk
from .schema import NodeType


@dataclass(frozen=True)
class ExtractedEntity:
    entity_id: str
    label: str
    node_type: str
    confidence: float


class ConceptExtractor:
    def __init__(self, keywords: Iterable[str]):
        self.keywords = {kw.lower() for kw in keywords}

    def extract_entities(self, chunk: Chunk) -> List[ExtractedEntity]:
        text = chunk.text
        text_lower = text.lower()
        entities: dict[str, ExtractedEntity] = {}

        self._add_keyword_entities(text_lower, entities)
        self._add_requirement_entities(text, text_lower, entities)
        self._add_control_entities(text, entities)
        self._add_evidence_entities(text_lower, entities)
        self._add_incident_entities(text_lower, entities)

        filtered_entities = [
            entity for entity in entities.values() if self._should_keep_entity(entity, text_lower)
        ]
        return sorted(filtered_entities, key=lambda item: (item.node_type, item.label))

    def extract_concepts(self, chunk: Chunk) -> List[str]:
        return [entity.label for entity in self.extract_entities(chunk)]

    def _add_keyword_entities(self, text_lower: str, entities: dict[str, ExtractedEntity]) -> None:
        keyword_type_map = {
            "requirement": NodeType.REQUIREMENT.value,
            "control": NodeType.CONTROL.value,
            "evidence": NodeType.EVIDENCE.value,
            "incident": NodeType.INCIDENT.value,
            "drift": NodeType.INCIDENT.value,
            "validation": NodeType.CONCEPT.value,
            "monitoring": NodeType.CONCEPT.value,
            "approval": NodeType.CONCEPT.value,
            "review": NodeType.CONCEPT.value,
            "risk": NodeType.CONCEPT.value,
            "testing": NodeType.CONCEPT.value,
            "policy": NodeType.CONCEPT.value,
        }

        for keyword in self.keywords:
            if keyword in text_lower:
                node_type = keyword_type_map.get(keyword, NodeType.CONCEPT.value)
                confidence = 0.65 if node_type != NodeType.CONCEPT.value else 0.55
                self._store_entity(
                    entities,
                    label=keyword,
                    node_type=node_type,
                    confidence=confidence,
                )

    def _add_requirement_entities(self, text: str, text_lower: str, entities: dict[str, ExtractedEntity]) -> None:
        if any(token in text_lower for token in ["must", "shall", "required"]):
            statements = [segment.strip() for segment in re.split(r"[\.\n]", text) if segment.strip()]
            for statement in statements:
                statement_lower = statement.lower()
                if any(token in statement_lower for token in ["must", "shall", "required"]):
                    label = self._normalize_phrase(statement, max_words=12)
                    self._store_entity(
                        entities,
                        label=label,
                        node_type=NodeType.REQUIREMENT.value,
                        confidence=0.9,
                    )

    def _add_control_entities(self, text: str, entities: dict[str, ExtractedEntity]) -> None:
        for match in re.findall(r"Control\s+([A-Z]-\d+)", text, flags=re.IGNORECASE):
            self._store_entity(
                entities,
                label=f"Control {match.upper()}",
                node_type=NodeType.CONTROL.value,
                confidence=0.95,
            )

    def _add_evidence_entities(self, text_lower: str, entities: dict[str, ExtractedEntity]) -> None:
        evidence_patterns = [
            r"validation evidence",
            r"testing evidence",
            r"approval evidence",
            r"documented evidence",
            r"evidence collection",
            r"monitoring readiness evidence",
        ]
        for pattern in evidence_patterns:
            for match in re.findall(pattern, text_lower):
                self._store_entity(
                    entities,
                    label=match,
                    node_type=NodeType.EVIDENCE.value,
                    confidence=0.85,
                )

    def _add_incident_entities(self, text_lower: str, entities: dict[str, ExtractedEntity]) -> None:
        incident_patterns = [
            r"model drift",
            r"drift",
            r"harmful behavior",
            r"harmful outputs",
            r"material incidents?",
            r"incident reporting",
            r"incident escalation",
            r"issue escalated",
        ]
        for pattern in incident_patterns:
            for match in re.findall(pattern, text_lower):
                self._store_entity(
                    entities,
                    label=match,
                    node_type=NodeType.INCIDENT.value,
                    confidence=0.85,
                )

    def _should_keep_entity(self, entity: ExtractedEntity, text_lower: str) -> bool:
        if entity.node_type != NodeType.CONCEPT.value:
            return True

        keepable_concepts = {"validation", "monitoring", "approval", "review", "risk", "testing"}
        return entity.label.lower() in keepable_concepts and len(text_lower.split()) >= 10

    def _store_entity(
        self,
        entities: dict[str, ExtractedEntity],
        label: str,
        node_type: str,
        confidence: float,
    ) -> None:
        normalized_label = label.strip().lower()
        entity_id = f"{node_type.lower()}::{normalized_label.replace(' ', '_')}"
        existing = entities.get(entity_id)
        if existing is None or confidence > existing.confidence:
            entities[entity_id] = ExtractedEntity(
                entity_id=entity_id,
                label=label.strip(),
                node_type=node_type,
                confidence=confidence,
            )

    def _normalize_phrase(self, phrase: str, max_words: int = 10) -> str:
        words = re.findall(r"[A-Za-z0-9\-]+", phrase)
        return " ".join(words[:max_words])
