from enum import Enum


class NodeType(str, Enum):
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    REQUIREMENT = "Requirement"
    CONTROL = "Control"
    EVIDENCE = "Evidence"
    INCIDENT = "Incident"
    CONCEPT = "Concept"


class EdgeType(str, Enum):
    CONTAINS = "CONTAINS"
    MENTIONS = "MENTIONS"
    REQUIRES = "REQUIRES"
    SUPPORTS = "SUPPORTS"
    TRIGGERS = "TRIGGERS"
    RELATED_TO = "RELATED_TO"
